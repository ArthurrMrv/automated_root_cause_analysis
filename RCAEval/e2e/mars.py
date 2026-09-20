"""PRISM + MARS: Multi-Aspect Robust Anomaly Scoring (report.md, Add-On 1).

Standard PRISM scores a property with a single deviation measure -- a z-score or
an IQR ratio (Definition 3.1, `prism.py`). A z-score is the right instrument for
a sudden Gaussian spike and the wrong one for a step-function shift on a
heavy-tailed baseline: a handful of pre-fault outliers inflate sigma until a real
shift scores near zero. MARS ensembles the two regimes at the *property* level,
before any pooling:

    S(x) = max( |x - mu| / sigma , |x - median| / (1.4826 * MAD) )

Both halves are monotone and injective in |x - center|, so their max is too, and
PRISM's internal boundedness proofs (M(S^I, S^E) <= f(S^I)) carry over unchanged
-- which is why the report calls this a zero-downside add-on.

This module is also the base layer of the rest of the stack: `tgfi`, `lstr`,
`dterwr` and `pave` all window, score and rank through the helpers here rather
than re-deriving PRISM. Everything PRISM already settles (the Component-Property
Model, the additive score of Equation 3, the witness-property convention) is
imported from `prism.py`, which stays untouched.

Values the report leaves unspecified are marked `# gap N:`.
"""

import warnings
from typing import Any, Callable, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from RCAEval.e2e.prism import (
    COMBINERS,
    POOLINGS,
    _POOL_FUNCS,
    _combine,
    _group_by_component,
)
from RCAEval.io.time_series import preprocess

# consistency constant making 1.4826 * MAD an unbiased estimator of sigma for
# Gaussian data, so the two halves of the max are on one scale
MAD_SCALE = 1.4826

# "Outputs Deterministic Shortlist of Top-5 Candidate Microservices" -- the
# shortlist width every downstream add-on re-ranks within
TOP_K = 5

# {component: (S^I, S^E, witness property column)}
ComponentTable = dict[str, tuple[float, float, str]]


def _windows(
    data: pd.DataFrame,
    inject_time: Optional[int],
    anomalies: Optional[list],
    dataset: Optional[str],
    dk_select_useful: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Split into the pre- and post-fault windows, preprocessed and aligned.

    Lifted out of `prism.prism` so the whole stack splits telemetry identically;
    returns (normal, anomal, shared columns).
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"expected a metric DataFrame, got {type(data).__name__}")

    if anomalies is None:
        if inject_time is None:
            raise ValueError("needs either inject_time or anomalies to split the windows")
        if "time" not in data.columns:
            raise ValueError("needs a 'time' column to split on inject_time")
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        normal_df = data.head(anomalies[0])
        anomal_df = data.tail(len(data) - anomalies[0])

    normal_df = preprocess(data=normal_df, dataset=dataset, dk_select_useful=dk_select_useful)
    anomal_df = preprocess(data=anomal_df, dataset=dataset, dk_select_useful=dk_select_useful)

    intersects = [c for c in normal_df.columns if c in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]

    if normal_df.empty or anomal_df.empty or not intersects:
        raise ValueError(
            f"needs non-empty pre- and post-fault windows over shared columns "
            f"(got {len(normal_df)} and {len(anomal_df)} rows over {len(intersects)} columns)"
        )
    return normal_df, anomal_df, intersects


def _mars_scores(
    normal: pd.DataFrame,
    anomal: pd.DataFrame,
    time_agg: Callable[..., np.ndarray],
) -> pd.Series:
    """Add-On 1: per-property MARS scores, S(x) = max(z-score, MAD ratio).

    Both centers and both scales are estimated on the pre-fault reference window,
    as PRISM's Definition 3.1 does. The max is taken *per observation*, before
    `time_agg` collapses the post-fault window and before the pooling of Section
    3.2 -- the report specifies the ensemble "at the individual property level
    prior to score pooling", and taking it later would let one regime's
    time-aggregate mask the other's peak.
    """
    reference = normal.to_numpy(dtype=float)
    observed = anomal.to_numpy(dtype=float)

    mean = reference.mean(axis=0)
    std = reference.std(axis=0)
    median = np.median(reference, axis=0)
    mad = np.median(np.abs(reference - median), axis=0) * MAD_SCALE

    # a property constant over the reference window has zero scale; fall back to
    # 1.0, the same rule (and reason) as `prism._deviation_scores` gap 5, applied
    # to both halves so neither can divide by zero
    std = np.where(std > 0, std, 1.0)
    mad = np.where(mad > 0, mad, 1.0)

    gaussian = np.abs(observed - mean) / std
    robust = np.abs(observed - median) / mad
    return pd.Series(time_agg(np.maximum(gaussian, robust), axis=0), index=normal.columns)


def _component_table(
    scores: pd.Series, pool: Callable[..., float]
) -> tuple[ComponentTable, list[str]]:
    """Pool property scores into per-component (S^I, S^E, witness) -- Section 3.2.

    Returns the table plus the columns PRISM's Component-Property Model cannot
    classify, which stay out of the scoring and trail the ranking.
    """
    grouped, unclassified = _group_by_component(scores)

    table: ComponentTable = {}
    for component, properties in grouped.items():
        everything = properties["internal"] + properties["external"]
        # a component missing a whole class scores 0 there rather than dropping
        # out of the ranking (`prism` gap 4); TG-FI is what repairs that case
        internal = pool([s for _, s in properties["internal"]]) if properties["internal"] else 0.0
        external = pool([s for _, s in properties["external"]]) if properties["external"] else 0.0
        witness = max(everything, key=lambda x: x[1])[0]
        table[component] = (float(internal), float(external), witness)
    return table, unclassified


def _order(table: ComponentTable, combine: str = "additive") -> list[str]:
    """Rank components by the root cause score M(S^I, S^E) -- Section 3.3."""
    scored = [(c, _combine(i, e, combine)) for c, (i, e, _) in table.items()]
    return [component for component, _ in sorted(scored, key=lambda x: x[1], reverse=True)]


def _emit(
    order: list[str], table: ComponentTable, unclassified: list[str], scores: pd.Series
) -> list[str]:
    """Render a component ranking as RCAEval ranks, "<component>_<witness>".

    Unclassified properties are no evidence either way but stay in the list after
    the scored components, so the candidate set the evaluator sees is covered.
    """
    return [table[c][2] for c in order] + sorted(unclassified, key=lambda c: -scores[c])


def _validate(pooling: str, combine: str, time_agg: str) -> None:
    if pooling not in POOLINGS:
        raise ValueError(f"{pooling=} must be one of {POOLINGS}")
    if combine not in COMBINERS:
        raise ValueError(f"{combine=} must be one of {COMBINERS}")
    if time_agg not in POOLINGS:
        raise ValueError(f"{time_agg=} must be one of {POOLINGS}")


def mars(
    data: pd.DataFrame,
    inject_time: Optional[int] = None,
    dataset: Optional[str] = None,
    num_loop: Optional[int] = None,
    sli: Optional[str] = None,
    anomalies: Optional[list] = None,
    pooling: str = "max",
    combine: str = "additive",
    **kwargs: Any,
) -> dict[str, list]:
    """Rank root cause candidates with PRISM + MARS (report.md, Add-On 1).

    Args:
        data: metric DataFrame with a "time" column spanning both windows.
        inject_time: fault injection timestamp; ignored when `anomalies` is given.
        dataset: dataset name, forwarded to `preprocess`.
        anomalies: optional `[n]`, splitting after the first `n` rows instead.
        pooling: pooling function phi over a component's property scores.
        combine: root cause score M; "additive" is PRISM's Equation (3).

    Returns:
        {"node_names": [...], "ranks": ["<component>_<witness property>", ...]},
        most likely root cause first.
    """
    time_agg = kwargs.get("time_agg", "max")
    _validate(pooling, combine, time_agg)

    normal_df, anomal_df, intersects = _windows(
        data, inject_time, anomalies, dataset, kwargs.get("dk_select_useful", False)
    )
    scores = _mars_scores(normal_df, anomal_df, _POOL_FUNCS[time_agg])
    table, unclassified = _component_table(scores, _POOL_FUNCS[pooling])
    ranks = _emit(_order(table, combine), table, unclassified, scores)

    if kwargs.get("verbose") is True:
        for component in _order(table, combine)[:20]:
            internal, external, witness = table[component]
            print(f"{witness}: S^I={internal:.2f} S^E={external:.2f}")

    return {"node_names": intersects, "ranks": ranks}


# ==========================================================================
# Runnable check -- no datasets
# ==========================================================================


def _heavy_tailed_case() -> pd.DataFrame:
    """The case a single z-score cannot see.

    `checkout` is the root cause: its cpu baseline carries a few large pre-fault
    outliers, so sigma is inflated and the post-fault step shift barely registers
    as a z-score -- while the median/MAD pair, untouched by the outliers, scores
    it enormous. `ads` is the downstream victim, with a clean baseline and a large
    latency spike, so a z-score-only PRISM ranks the victim first.
    """
    rng = np.random.default_rng(1)
    n = 120
    frame = pd.DataFrame({"time": np.arange(n)})
    for column in ("checkout_cpu", "checkout_latency", "ads_cpu", "ads_latency"):
        frame[column] = rng.normal(10, 0.5, n)
    frame.loc[[5, 17, 31, 44], "checkout_cpu"] = 400.0  # contaminates sigma only

    post = frame["time"] >= n // 2
    frame.loc[post, "checkout_cpu"] = 25.0 + rng.normal(0, 0.5, int(post.sum()))
    frame.loc[post, "ads_latency"] += 8.0  # the victim's inherited spike
    return frame


def _demo() -> None:
    """The add-on's claims, as a runnable check."""
    from RCAEval.e2e.prism import _deviation_scores, _propagation_case, prism

    # == MARS never scores below the z-score it ensembles (monotone max) ==
    heavy = _heavy_tailed_case()
    normal_df, anomal_df, _ = _windows(heavy, 60, None, "demo")
    zscore = _deviation_scores(normal_df, anomal_df, "zscore", np.max)
    ensemble = _mars_scores(normal_df, anomal_df, np.max)
    assert (ensemble >= zscore - 1e-9).all(), (ensemble - zscore).min()

    # the contaminated baseline is exactly where the halves diverge
    assert ensemble["checkout_cpu"] > 10 * zscore["checkout_cpu"], (
        ensemble["checkout_cpu"], zscore["checkout_cpu"]
    )

    # == and that difference decides the ranking ==
    assert prism(heavy, inject_time=60, dataset="demo")["ranks"][0].startswith("ads")
    assert mars(heavy, inject_time=60, dataset="demo")["ranks"][0].startswith("checkout")

    # == no regression on the Gaussian case PRISM already handles ==
    bounded = _propagation_case(external_amplification=1.7)
    assert mars(bounded, inject_time=60, dataset="demo")["ranks"][0].startswith("cartservice")

    # == every documented configuration produces a full, valid ranking ==
    for pooling in POOLINGS:
        for combine in COMBINERS:
            out = mars(bounded, inject_time=60, dataset="demo", pooling=pooling, combine=combine)
            assert len(out["ranks"]) == 3, (pooling, combine, out["ranks"])
            assert all("_" in rank for rank in out["ranks"]), out["ranks"]

    print("mars demo ok:", mars(heavy, inject_time=60, dataset="demo")["ranks"])


if __name__ == "__main__":
    _demo()
