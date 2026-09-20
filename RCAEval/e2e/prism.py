"""PRISM: Graph-Free Root Cause Analysis (Pham, arXiv:2601.21359).

PRISM ranks root cause candidates without a dependency graph. Under the paper's
Component-Property Model, every component exposes *internal* properties (local
resource state: cpu, memory, disk I/O, socket count) and *external* properties
(boundary-observable QoS: latency, error rate, request rate). Axioms 2.5-2.6 say
faults originate internally and propagate between components only externally, so
the root cause is the component anomalous in *both* classes, while downstream
components are anomalous only externally -- even when their external anomaly is
the larger one.

Pipeline (Figure 2): deviation-based anomaly scoring (Sec 3.1) -> pooling of
property scores into per-component S^I and S^E (Sec 3.2) -> root cause scoring
and ranking (Sec 3.3). Default configuration is zscore + max + additive, per the
footnote of Table 5.

The paper ships no reference implementation; every value it left unspecified is
marked `# gap N:` below and tabulated in the reproduction notes.
"""

import warnings
from typing import Any, Callable, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from RCAEval.io.time_series import preprocess

# Sec 4.1.1 and Appendix D.1: "Internal properties are local resource states: CPU
# usage, memory utilization, disk I/O, and socket count", classified by USE;
# "External properties are observable metrics at component boundaries including
# response time and error rate", classified by RED (Rate, Errors, Duration).
# Matched against the first token of the property name, so latency-90, lat_90 and
# latency all resolve alike.
INTERNAL_PROPERTIES = frozenset({"cpu", "mem", "memory", "disk", "diskio", "socket", "sockets"})
EXTERNAL_PROPERTIES = frozenset(
    # gap 3: `workload` is RED's Rate; the paper cites RED but omits rate from its
    # explicit external list. Move it to INTERNAL_PROPERTIES to score it as local.
    {"latency", "lat", "latency-90", "error", "errors", "duration", "rt", "workload"}
)

SCORERS = ("zscore", "iqr")
POOLINGS = ("max", "mean", "sum")
COMBINERS = ("additive", "conjunctive", "internal", "external", "marginal")

_POOL_FUNCS = {"max": np.max, "mean": np.mean, "sum": np.sum}


def _split_property(column: str) -> tuple[str, str]:
    """Split an RCAEval column into (component, property).

    `main.py` derives the service as `column.split("_")[0]`, so the component is
    everything before the first underscore and the property is the remainder:
    "adservice_cpu" -> ("adservice", "cpu"), "carts_lat_90" -> ("carts", "lat_90").
    """
    component, _, prop = column.partition("_")
    return component, prop


def _property_class(prop: str) -> Optional[str]:
    """Classify a property as "internal", "external", or None if unrecognised."""
    root = prop.replace("-", "_").split("_")[0].lower()
    if root in INTERNAL_PROPERTIES:
        return "internal"
    if root in EXTERNAL_PROPERTIES:
        return "external"
    return None


def _deviation_scores(
    normal: pd.DataFrame,
    anomal: pd.DataFrame,
    scorer: str,
    time_agg: Callable[..., np.ndarray],
) -> pd.Series:
    """Per-property anomaly scores, Definition 3.1: S(x) = |x - c(theta)| / s(theta).

    `c` and `s` are estimated on the pre-fault reference window (gap 2); the
    post-fault observations are then aggregated over time by `time_agg` (gap 1).
    Returns a Series indexed by column name.
    """
    reference = normal.to_numpy(dtype=float)
    observed = anomal.to_numpy(dtype=float)

    if scorer == "zscore":
        center = reference.mean(axis=0)
        scale = reference.std(axis=0)
    else:  # "iqr", as in BARO: RobustScaler's median / interquartile range
        q25, q50, q75 = np.percentile(reference, [25, 50, 75], axis=0)
        center = q50
        scale = q75 - q25

    # gap 5: Definition 3.1 requires s > 0. A property constant across the
    # reference window has s = 0; fall back to 1.0, as sklearn's StandardScaler
    # and RobustScaler do for zero-variance features (so PRISM and the BARO
    # baseline treat degenerate columns identically).
    scale = np.where(scale > 0, scale, 1.0)

    # gap 8: |x - c|, per Definition 3.1. BARO uses the signed deviation instead.
    scores = np.abs(observed - center) / scale
    return pd.Series(time_agg(scores, axis=0), index=normal.columns)


def _combine(internal_score: float, external_score: float, combine: str) -> float:
    """Root cause score M(S^I, S^E), Section 3.3."""
    if combine == "additive":  # Equation (3)
        total = internal_score + external_score
        return total - np.log1p(total)
    if combine == "conjunctive":  # Equation (4)
        return min(internal_score, external_score)
    if combine == "internal":  # PRISM_Internal ablation, Table 6
        return internal_score
    return external_score  # PRISM_External ablation, Table 6


def _group_by_component(
    scores: pd.Series,
) -> tuple[dict[str, dict[str, list[tuple[str, float]]]], list[str]]:
    """Group property scores by component and class, Section 3.2.

    Returns ({component: {"internal": [(column, score)], "external": [...]}},
    [columns whose property PRISM cannot classify]).
    """
    grouped = {}
    unclassified = []
    for column, score in scores.items():
        component, prop = _split_property(column)
        prop_class = _property_class(prop)
        if prop_class is None:
            unclassified.append(column)
            continue
        entry = grouped.setdefault(component, {"internal": [], "external": []})
        entry[prop_class].append((column, score))
    return grouped, unclassified


def _rank_components(
    grouped: dict[str, dict[str, list[tuple[str, float]]]],
    pool: Callable[..., float],
    combine: str,
) -> list[tuple[str, float]]:
    """Pool into S^I and S^E (Sec 3.2), then score and rank components (Sec 3.3).

    Returns [(witness column, root cause score)], highest score first.
    """
    ranked = []
    for properties in grouped.values():
        everything = properties["internal"] + properties["external"]
        # gap 4: Assumption 2.4 expects both classes instrumented; a component
        # missing one scores 0 there rather than being dropped from the ranking.
        internal_score = pool([s for _, s in properties["internal"]]) if properties["internal"] else 0.0
        external_score = pool([s for _, s in properties["external"]]) if properties["external"] else 0.0

        if combine == "marginal":  # PRISM_Marginal ablation: highest single property score
            root_cause_score = max(s for _, s in everything)
        else:
            root_cause_score = _combine(internal_score, external_score, combine)

        # gap 6: name the component's most deviant property, so RCAEval's
        # service-level and metric-level evaluators both read the rank correctly.
        witness = max(everything, key=lambda x: x[1])[0]
        ranked.append((witness, root_cause_score))

    return sorted(ranked, key=lambda x: x[1], reverse=True)


def prism(
    data: pd.DataFrame,
    inject_time: Optional[int] = None,
    dataset: Optional[str] = None,
    num_loop: Optional[int] = None,
    sli: Optional[str] = None,
    anomalies: Optional[list] = None,
    scorer: str = "zscore",
    pooling: str = "max",
    combine: str = "additive",
    **kwargs: Any,
) -> dict[str, list]:
    """Rank root cause candidates with PRISM (arXiv:2601.21359).

    Args:
        data: metric DataFrame with a "time" column, covering the pre- and
            post-fault windows.
        inject_time: fault injection timestamp, used to split the two windows.
            Ignored when `anomalies` is given.
        dataset: dataset name, forwarded to `preprocess`.
        anomalies: optional `[n]`, splitting after the first `n` rows instead of
            on `inject_time`.
        scorer: anomaly scorer, one of SCORERS. Default "zscore" (Table 5).
        pooling: pooling function phi, one of POOLINGS. Default "max" (Table 5).
        combine: root cause score M, one of COMBINERS. Default "additive",
            Equation (3). "conjunctive" is Equation (4); "internal", "external"
            and "marginal" are the Table 6 ablations.

    Returns:
        {"node_names": [...], "ranks": [...]} where each rank is
        "<component>_<most deviant property of that component>", ordered by
        decreasing root cause score.
    """
    if scorer not in SCORERS:
        raise ValueError(f"{scorer=} must be one of {SCORERS}")
    if pooling not in POOLINGS:
        raise ValueError(f"{pooling=} must be one of {POOLINGS}")
    if combine not in COMBINERS:
        raise ValueError(f"{combine=} must be one of {COMBINERS}")
    time_agg = kwargs.get("time_agg", "max")
    if time_agg not in POOLINGS:
        raise ValueError(f"{time_agg=} must be one of {POOLINGS}")
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"prism expects a metric DataFrame, got {type(data).__name__}")
    if anomalies is None:
        if inject_time is None:
            raise ValueError("prism needs either inject_time or anomalies to split the windows")
        if "time" not in data.columns:
            raise ValueError("prism needs a 'time' column to split on inject_time")
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        normal_df = data.head(anomalies[0])
        anomal_df = data.tail(len(data) - anomalies[0])

    normal_df = preprocess(
        data=normal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )
    anomal_df = preprocess(
        data=anomal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]

    if normal_df.empty or anomal_df.empty or not intersects:
        raise ValueError(
            f"prism needs non-empty pre- and post-fault windows over shared columns "
            f"(got {len(normal_df)} and {len(anomal_df)} rows over {len(intersects)} columns)"
        )

    # Section 3.1: anomaly scoring
    scores = _deviation_scores(normal_df, anomal_df, scorer, _POOL_FUNCS[time_agg])
    grouped, unclassified = _group_by_component(scores)
    ranked = _rank_components(grouped, _POOL_FUNCS[pooling], combine)

    if kwargs.get("verbose") is True:
        for name, score in ranked[:20]:
            print(f"{name}: {score:.2f}")

    # properties PRISM cannot classify are no evidence either way, but stay in the
    # ranking after the scored components so the candidate set is fully covered
    ranks = [name for name, _ in ranked] + sorted(unclassified, key=lambda c: -scores[c])

    return {
        "node_names": intersects,
        "ranks": ranks,
    }


def prism_iqr(
    data: pd.DataFrame,
    inject_time: Optional[int] = None,
    dataset: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, list]:
    """PRISM with the IQR-based scorer instead of the z-score (Table 5)."""
    kwargs.pop("scorer", None)
    return prism(data, inject_time=inject_time, dataset=dataset, scorer="iqr", **kwargs)


def prism_mean(
    data: pd.DataFrame,
    inject_time: Optional[int] = None,
    dataset: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, list]:
    """PRISM pooling property scores with mean instead of max (Table 5)."""
    kwargs.pop("pooling", None)
    return prism(data, inject_time=inject_time, dataset=dataset, pooling="mean", **kwargs)


def prism_sum(
    data: pd.DataFrame,
    inject_time: Optional[int] = None,
    dataset: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, list]:
    """PRISM pooling property scores with sum instead of max (Table 5)."""
    kwargs.pop("pooling", None)
    return prism(data, inject_time=inject_time, dataset=dataset, pooling="sum", **kwargs)


def prism_conjunctive(
    data: pd.DataFrame,
    inject_time: Optional[int] = None,
    dataset: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, list]:
    """PRISM ranking by the conjunctive score min(S^I, S^E), Equation (4)."""
    kwargs.pop("combine", None)
    return prism(data, inject_time=inject_time, dataset=dataset, combine="conjunctive", **kwargs)


def prism_marginal(
    data: pd.DataFrame,
    inject_time: Optional[int] = None,
    dataset: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, list]:
    """PRISM_Marginal: rank by the highest marginal property score (Table 6)."""
    kwargs.pop("combine", None)
    return prism(data, inject_time=inject_time, dataset=dataset, combine="marginal", **kwargs)


def prism_internal(
    data: pd.DataFrame,
    inject_time: Optional[int] = None,
    dataset: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, list]:
    """PRISM_Internal: rank by S^I alone (Table 6)."""
    kwargs.pop("combine", None)
    return prism(data, inject_time=inject_time, dataset=dataset, combine="internal", **kwargs)


def prism_external(
    data: pd.DataFrame,
    inject_time: Optional[int] = None,
    dataset: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, list]:
    """PRISM_External: rank by S^E alone (Table 6)."""
    kwargs.pop("combine", None)
    return prism(data, inject_time=inject_time, dataset=dataset, combine="external", **kwargs)


def _propagation_case(external_amplification: float) -> pd.DataFrame:
    """A fault propagating from `cartservice` to `frontend`.

    `cartservice` is the root cause: an internal (cpu) anomaly plus an external
    (latency) one of the same magnitude. `frontend` is downstream and shows only
    an external anomaly, `external_amplification` times larger -- the case that
    defeats marginal-score ranking (Sec 1, Theorem C.3). `adservice` is unaffected.
    """
    rng = np.random.default_rng(0)
    n, fault = 120, 15.0
    frame = pd.DataFrame({"time": np.arange(n)})
    for component in ("cartservice", "frontend", "adservice"):
        for prop in ("cpu", "mem", "latency", "error"):
            frame[f"{component}_{prop}"] = rng.normal(10, 1.0, n)

    post = frame["time"] >= n // 2
    frame.loc[post, "cartservice_cpu"] += fault  # fault originates internally
    frame.loc[post, "cartservice_latency"] += fault  # and shows at the boundary
    frame.loc[post, "frontend_latency"] += fault * external_amplification
    return frame


def _demo() -> None:
    """The paper's claims, as a runnable check, each under its stated condition."""
    top = lambda fn, frame: fn(frame, inject_time=60, dataset="demo")["ranks"]

    # == Bounded external amplification (the condition of Proposition 3.10) ==
    bounded = _propagation_case(external_amplification=1.7)

    # Theorem 3.13: both scorers rank the root cause above the affected component
    assert top(prism, bounded)[0].startswith("cartservice"), top(prism, bounded)
    assert top(prism_conjunctive, bounded)[0].startswith("cartservice")

    # ...which is exactly what the ablations cannot do: the downstream component
    # carries the larger marginal and external anomaly (Table 6)
    assert top(prism_marginal, bounded)[0].startswith("frontend")
    assert top(prism_external, bounded)[0].startswith("frontend")

    # == Arbitrary external amplification (beyond Proposition 3.10's condition) ==
    arbitrary = _propagation_case(external_amplification=6.0)

    # Proposition 3.11: M_conj is internally bounded with f(s) = s, unconditionally
    assert top(prism_conjunctive, arbitrary)[0].startswith("cartservice")

    # M_add is not: it grows without bound in S^E, so it loses the root cause here.
    # This is the paper's own caveat -- Prop 3.10 holds only under S^E <= a*S^I + b
    # -- and the reason the conjunctive scorer exists (Sec 3.3).
    assert top(prism, arbitrary)[0].startswith("frontend"), top(prism, arbitrary)

    # == Every documented configuration produces a full, valid ranking ==
    for scorer in SCORERS:
        for pooling in POOLINGS:
            for combine in COMBINERS:
                out = prism(
                    bounded, inject_time=60, dataset="demo",
                    scorer=scorer, pooling=pooling, combine=combine,
                )
                assert len(out["ranks"]) == 3, (scorer, pooling, combine, out["ranks"])
                assert all("_" in r for r in out["ranks"]), out["ranks"]

    print("prism demo ok:", top(prism, bounded))


if __name__ == "__main__":
    _demo()
