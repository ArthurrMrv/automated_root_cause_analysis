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

v2 keeps every specified part of the paper identical to `prism.py` and only
revisits the gaps that were distorting the ranking rather than filling it:
the zero-scale fallback (gap 5) no longer scores a property in raw units,
preprocessing runs once over the whole frame instead of once per window, and
pooling is NaN-safe. `diagnostics` in the returned dict reports how often each
of those fired, per case.
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

# gap 5: floor on s(theta), as a fraction of the property's own centre, so a
# near-constant reference window cannot turn a small absolute move into a
# six-figure score. Binds only in that degenerate regime.
SCALE_FLOOR = 0.01
# last resort for a property with no scale information at all (constant at
# zero): kept tiny so it stays visible in `diagnostics` rather than silently
# passing for a real scale.
SCALE_EPSILON = 1e-9

# NaN-safe: a single NaN would otherwise poison the pooled score and put the
# component in an arbitrary place in the ranking
_POOL_FUNCS = {"max": np.nanmax, "mean": np.nanmean, "sum": np.nansum}
# gap 1: the paper does not say how the post-fault window is aggregated over
# time. max is the default; p90 is the same statistic made robust to a single
# spike, and mean/sum are much less window-length dependent.
_TIME_AGGS = dict(_POOL_FUNCS, p90=lambda a, axis: np.nanpercentile(a, 90, axis=axis))


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
    Returns (Series indexed by column name, counts of each scale fallback).
    """
    reference = normal.to_numpy(dtype=float)
    observed = anomal.to_numpy(dtype=float)

    if scorer == "zscore":
        center = np.nanmean(reference, axis=0)
        scale = np.nanstd(reference, axis=0)
    else:  # "iqr", as in BARO: RobustScaler's median / interquartile range
        q25, q50, q75 = np.nanpercentile(reference, [25, 50, 75], axis=0)
        center = q50
        scale = q75 - q25

    # gap 5: Definition 3.1 requires s > 0, and what replaces a zero s is the
    # single largest lever on the ranking. Falling back to 1.0 (sklearn's rule
    # for zero-variance features) scores the property in its raw units, so a
    # byte-valued metric with a flat baseline outranks every properly scaled
    # one. Fall back instead to information still about this property: the
    # reference std when a robust scale collapses, then a floor relative to the
    # property's own centre, then an absolute epsilon.
    std = np.nanstd(reference, axis=0)
    from_std = (scale <= 0) & (std > 0)
    scale = np.where(scale > 0, scale, std)
    floor = SCALE_FLOOR * np.abs(center)
    floored = scale < floor
    scale = np.maximum(scale, floor)
    scaleless = scale <= 0
    scale = np.where(scale > 0, scale, SCALE_EPSILON)

    # gap 8: |x - c|, per Definition 3.1. BARO uses the signed deviation instead.
    scores = np.abs(observed - center) / scale
    counts = {
        "scale_from_std": int(from_std.sum()),
        "scale_floored": int(floored.sum()),
        "scale_epsilon": int(scaleless.sum()),
    }
    return pd.Series(time_agg(scores, axis=0), index=normal.columns), counts


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
) -> list[tuple[str, float, float]]:
    """Pool into S^I and S^E (Sec 3.2), then score and rank components (Sec 3.3).

    Returns [(witness column, root cause score, S^E)], highest score first;
    S^E breaks ties, which `sorted` would otherwise resolve by column order.
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
        ranked.append((witness, root_cause_score, external_score))

    return sorted(ranked, key=lambda x: (x[1], x[2]), reverse=True)


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
        {"node_names": [...], "ranks": [...], "diagnostics": {...}} where each
        rank is "<component>_<most deviant property of that component>",
        ordered by decreasing root cause score, one entry per component.
        `diagnostics` reports the gap-filling that fired on this case: columns
        dropped, unclassified properties, scale fallbacks and NaN columns.
    """
    if scorer not in SCORERS:
        raise ValueError(f"{scorer=} must be one of {SCORERS}")
    if pooling not in POOLINGS:
        raise ValueError(f"{pooling=} must be one of {POOLINGS}")
    if combine not in COMBINERS:
        raise ValueError(f"{combine=} must be one of {COMBINERS}")
    time_agg = kwargs.get("time_agg", "max")
    if time_agg not in _TIME_AGGS:
        raise ValueError(f"{time_agg=} must be one of {tuple(_TIME_AGGS)}")
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"prism expects a metric DataFrame, got {type(data).__name__}")
    if anomalies is None:
        if inject_time is None:
            raise ValueError("prism needs either inject_time or anomalies to split the windows")
        if "time" not in data.columns:
            raise ValueError("prism needs a 'time' column to split on inject_time")
        is_normal = (data["time"] < inject_time).to_numpy()
    else:
        is_normal = np.arange(len(data)) < anomalies[0]

    # Preprocess once over the whole frame, then split. Preprocessing each
    # window separately is asymmetric: `drop_constant` sees a property that is
    # flat before the fault and moves after it as constant in the reference
    # window, so the column intersection then discards it -- the clearest root
    # cause evidence there is.
    frame = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )
    normal_df, anomal_df = frame[is_normal], frame[~is_normal]

    # a property with no observation in a window has no score in that window
    usable = normal_df.notna().any() & anomal_df.notna().any()
    normal_df, anomal_df = normal_df.loc[:, usable], anomal_df.loc[:, usable]
    columns = list(normal_df.columns)

    if normal_df.empty or anomal_df.empty or not columns:
        raise ValueError(
            f"prism needs non-empty pre- and post-fault windows over observed columns "
            f"(got {len(normal_df)} and {len(anomal_df)} rows over {len(columns)} columns)"
        )

    # Section 3.1: anomaly scoring
    scores, scale_counts = _deviation_scores(
        normal_df, anomal_df, scorer, _TIME_AGGS[time_agg]
    )
    if not np.isfinite(scores.to_numpy()).all():
        raise ValueError(
            f"prism produced non-finite scores for "
            f"{list(scores.index[~np.isfinite(scores.to_numpy())])[:5]}"
        )
    grouped, unclassified = _group_by_component(scores)
    ranked = _rank_components(grouped, _POOL_FUNCS[pooling], combine)

    # properties PRISM cannot classify are no evidence either way, but stay in
    # the ranking after the scored components so the candidate set is fully
    # covered -- unless their component is already ranked, which would put the
    # same component in the list twice
    scored = {_split_property(name)[0] for name, _, _ in ranked}
    extra = [c for c in sorted(unclassified, key=lambda c: -scores[c])
             if _split_property(c)[0] not in scored]
    ranks = [name for name, _, _ in ranked] + extra

    diagnostics = {
        "n_columns": len(columns),
        "n_components": len(grouped),
        "n_dropped_by_preprocess": len(data.columns) - 1 - len(frame.columns),
        "n_all_nan_in_a_window": int((~usable).sum()),
        "n_unclassified": len(unclassified),
        "unclassified_properties": sorted({_split_property(c)[1] for c in unclassified}),
        **scale_counts,
    }

    if kwargs.get("verbose") is True:
        print(diagnostics)
        for name, score, _ in ranked[:20]:
            print(f"{name}: {score:.2f}")

    return {
        "node_names": columns,
        "ranks": ranks,
        "diagnostics": diagnostics,
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


# The functions above keep the paper's names, so this file stays diffable
# against prism.py; RCAEval registers them under prismv2* (see e2e/__init__.py)
# so `--method prism` and `--method prismv2` can be run side by side.
prismv2 = prism
prismv2_conjunctive = prism_conjunctive
prismv2_external = prism_external
prismv2_internal = prism_internal
prismv2_iqr = prism_iqr
prismv2_marginal = prism_marginal
prismv2_mean = prism_mean
prismv2_sum = prism_sum


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


def _flat_baseline_case() -> pd.DataFrame:
    """A root cause whose internal property is flat until the fault hits it.

    `cartservice_cpu` has zero variance across the reference window -- the case
    gap 5 has to resolve, and the one per-window preprocessing would throw away.
    `cartservice_gc` is a property PRISM cannot classify, and `adservice_mem`
    carries a missing observation.
    """
    frame = _propagation_case(external_amplification=1.0)
    n = len(frame)
    post = frame["time"] >= n // 2
    frame["cartservice_cpu"] = 5.0
    frame.loc[post, "cartservice_cpu"] = 35.0
    frame["cartservice_gc"] = np.arange(n, dtype=float)
    frame.loc[0, "adservice_mem"] = np.nan
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

    # == The gaps v2 revisits ==
    flat = _flat_baseline_case()
    out = prism(flat, inject_time=60, dataset="demo")
    diagnostics = out["diagnostics"]

    # preprocessing once keeps the flat-then-moving property, and it decides the
    # case; preprocessing per window would have dropped it as constant
    assert out["ranks"][0].startswith("cartservice"), out["ranks"]
    # gap 5 resolved as a floor on the property's own centre, not as raw units
    assert diagnostics["scale_floored"] == 1, diagnostics
    assert diagnostics["scale_epsilon"] == 0, diagnostics
    # unclassified properties are reported, and never duplicate their component
    assert diagnostics["unclassified_properties"] == ["gc"], diagnostics
    components = [r.split("_")[0] for r in out["ranks"]]
    assert len(components) == len(set(components)) == 3, out["ranks"]
    # a missing observation does not poison the pooled score (NaN-safe)
    assert diagnostics["n_all_nan_in_a_window"] == 0, diagnostics

    # == Every documented configuration produces a full, valid ranking ==
    for scorer in SCORERS:
        for pooling in POOLINGS:
            for combine in COMBINERS:
                for time_agg in _TIME_AGGS:
                    out = prism(
                        bounded, inject_time=60, dataset="demo", time_agg=time_agg,
                        scorer=scorer, pooling=pooling, combine=combine,
                    )
                    assert len(out["ranks"]) == 3, (scorer, pooling, combine, out["ranks"])
                    assert all("_" in r for r in out["ranks"]), out["ranks"]

    print("prism demo ok:", top(prism, bounded))


if __name__ == "__main__":
    _demo()
