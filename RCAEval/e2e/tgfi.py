"""PRISM + MARS + TG-FI: Telemetry-Gated Fallback Imputation (report.md, Add-On 2).

PRISM's whole discriminative power comes from S^I: a downstream victim shows a
large external anomaly and a quiet internal one, so the additive score keeps it
below the originating component. When a component's internal metrics are simply
*missing* -- a crashed container agent, an un-instrumented pod -- PRISM reads
S^I = 0 and silently degrades to PRISM_External on that component, which is the
ablation the paper shows ranking victims first.

TG-FI repairs exactly that component and nothing else:

    S^I_hat(C_i) = max(0, S^E(E_caller->C_i) - lambda * max_{C_k in Callees(C_i)} S^E(E_C_i->C_k))

Read as: the part of C_i's boundary anomaly that its downstream cannot explain
must have originated inside C_i. A component whose internal metrics are present
keeps its measured S^I untouched, so on complete telemetry TG-FI is provably a
no-op -- the report's "gated passive protection".

Everything else (windowing, MARS scoring, pooling, ranking) comes from `mars`.
This is also the first stage that needs the call graph, so the trace loading
helpers the topology add-ons share live here.

Values the report leaves unspecified are marked `# gap N:`.
"""

import warnings
from os.path import dirname
from typing import Any, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from RCAEval.e2e.mars import (
    ComponentTable,
    _component_table,
    _emit,
    _mars_scores,
    _order,
    _validate,
    _windows,
)
from RCAEval.e2e.prism import _POOL_FUNCS, _property_class, _split_property
from RCAEval.e2e.rclagent import _build_spans
from RCAEval.utility import read_traces

# gap 1: the report gives no value for lambda. 1.0 charges a component for the
# whole of its worst callee's external anomaly -- the most conservative choice,
# since it is the one least likely to manufacture a virtual internal anomaly out
# of inherited latency. Lower it to make TG-FI more willing to blame C_i.
LAMBDA = 1.0


# ==========================================================================
# Telemetry sources -- shared with the topology and agentic stages
# ==========================================================================


def _metric(data: Any) -> pd.DataFrame:
    """The metric frame, whether the harness passed it bare or in a dict."""
    if isinstance(data, dict):
        metric = data.get("metric")
        if metric is None:
            raise ValueError("needs a 'metric' DataFrame in the input dict")
        return metric
    return data


def _case_dir(kwargs: dict) -> Optional[str]:
    """The failure case's directory, from the `args` namespace `main.py` passes."""
    data_path = getattr(kwargs.get("args"), "data_path", None)
    return dirname(data_path) if data_path else None


def _traces(data: Any, kwargs: dict) -> Optional[pd.DataFrame]:
    """Traces for this case, or None when it ships none (RE1, Sock Shop).

    `RCAEval.utility.read_traces` is used rather than a bare `read_csv`, so both
    the traces.csv and the Hugging Face traces.parquet layouts resolve.
    """
    if isinstance(data, dict) and data.get("traces") is not None:
        return data["traces"]
    if kwargs.get("traces") is not None:
        return kwargs["traces"]
    case_dir = _case_dir(kwargs)
    if case_dir is None:
        return None
    try:
        return read_traces(case_dir)
    except Exception:  # unreadable or malformed traces are "no traces", not a crash
        return None


def _call_edges(
    traces: Optional[pd.DataFrame], candidates: list[str]
) -> list[tuple[str, str, float, float, float]]:
    """Directed caller->callee observations, one per parent/child span pair.

    Returns (caller, callee, caller duration, callee duration, callee start), with
    service names normalized onto `candidates` by `rclagent._match_service` -- the
    trace and metric vocabularies disagree often enough that skipping that step
    drops the right answer out of the candidate set.
    """
    if traces is None or traces.empty:
        return []
    spans = _build_spans(traces, candidates)
    edges = []
    for span in spans.values():
        parent = spans.get(span.parent_id) if span.parent_id else None
        if parent is None or parent.service == span.service:
            continue  # self-calls carry no caller/callee information
        edges.append(
            (parent.service, span.service, parent.duration, span.duration, span.start_time)
        )
    return edges


def _callees(edges: list[tuple[str, str, float, float, float]]) -> dict[str, set[str]]:
    """Callees(C_i) for every observed caller."""
    callees: dict[str, set[str]] = {}
    for caller, callee, *_ in edges:
        callees.setdefault(caller, set()).add(callee)
    return callees


# ==========================================================================
# Add-On 2: the gate and the virtual score
# ==========================================================================


def _instrumented(scores: pd.Series) -> set[str]:
    """Components that actually expose at least one internal property."""
    covered = set()
    for column in scores.index:
        component, prop = _split_property(column)
        if _property_class(prop) == "internal":
            covered.add(component)
    return covered


def _gated_internal(
    table: ComponentTable,
    instrumented: set[str],
    callees: Optional[dict[str, set[str]]],
    lam: float = LAMBDA,
) -> ComponentTable:
    """Impute S^I only for components whose internal telemetry is missing.

    Returns a new table; the input is left untouched.
    """
    # fail-safe: the formula is defined in terms of Callees(C_i). With no call
    # graph at all there is nothing to subtract, and crediting a component with
    # its full external anomaly would promote every victim -- so TG-FI stays
    # passive and the ranking is MARS's, unchanged.
    if callees is None:
        return table

    missing = [component for component in table if component not in instrumented]
    if not missing:
        return table  # the complete-telemetry path: provably zero effect

    repaired = dict(table)
    for component in missing:
        internal, external, witness = table[component]
        # a leaf has no downstream to blame, so max over an empty callee set is
        # 0 and the whole boundary anomaly is charged to the component itself
        downstream = [
            table[callee][1] for callee in callees.get(component, ()) if callee in table
        ]
        divergence = external - lam * (max(downstream) if downstream else 0.0)
        repaired[component] = (max(0.0, divergence), external, witness)
    return repaired


def _scored(
    data: Any,
    inject_time: Optional[int],
    dataset: Optional[str],
    anomalies: Optional[list],
    pooling: str,
    lam: float,
    kwargs: dict,
) -> tuple[ComponentTable, list[str], pd.Series, list[str]]:
    """Stage 1 of the report's pipeline: PRISM core + MARS + TG-FI.

    The shared entry point for every downstream add-on, so LSTR, DTE-RWR and
    PAVE all inherit the same scoring stack.
    """
    metric = _metric(data)
    normal_df, anomal_df, intersects = _windows(
        metric, inject_time, anomalies, dataset, kwargs.get("dk_select_useful", False)
    )
    scores = _mars_scores(normal_df, anomal_df, _POOL_FUNCS[kwargs.get("time_agg", "max")])
    table, unclassified = _component_table(scores, _POOL_FUNCS[pooling])

    traces = _traces(data, kwargs)
    callees = _callees(_call_edges(traces, list(table))) if traces is not None else None
    return _gated_internal(table, _instrumented(scores), callees, lam), unclassified, scores, intersects


def tgfi(
    data: Any,
    inject_time: Optional[int] = None,
    dataset: Optional[str] = None,
    num_loop: Optional[int] = None,
    sli: Optional[str] = None,
    anomalies: Optional[list] = None,
    pooling: str = "max",
    combine: str = "additive",
    lam: float = LAMBDA,
    **kwargs: Any,
) -> dict[str, list]:
    """Rank root cause candidates with PRISM + MARS + TG-FI (report.md, Add-On 2).

    Args:
        data: metric DataFrame, or a dict with "metric" and optionally "traces".
        inject_time: fault injection timestamp; ignored when `anomalies` is given.
        dataset: dataset name, forwarded to `preprocess`.
        lam: the formula's lambda, how much of a callee's external anomaly is
            treated as explaining the caller's.

    Returns:
        {"node_names": [...], "ranks": ["<component>_<witness property>", ...]}.
    """
    _validate(pooling, combine, kwargs.get("time_agg", "max"))
    table, unclassified, scores, intersects = _scored(
        data, inject_time, dataset, anomalies, pooling, lam, kwargs
    )
    ranks = _emit(_order(table, combine), table, unclassified, scores)

    if kwargs.get("verbose") is True:
        for component in _order(table, combine)[:20]:
            internal, external, witness = table[component]
            print(f"{witness}: S^I={internal:.2f} S^E={external:.2f}")

    return {"node_names": intersects, "ranks": ranks}


# ==========================================================================
# Runnable check -- no datasets
# ==========================================================================


def _missing_internal_case(inject_time: int = 60) -> dict:
    """`cartservice` is the root cause, and its internal metrics are gone.

    frontend -> cartservice -> catalogue. The fault raises cartservice's latency;
    frontend inherits it, amplified. cartservice exposes no cpu or memory column,
    so PRISM and MARS see S^I = 0 for it and rank the amplified caller first.
    """
    n = 120
    rng = np.random.default_rng(2)
    metric = pd.DataFrame({"time": np.arange(n)})
    for column in (
        "frontend_cpu", "frontend_mem", "frontend_latency", "frontend_error",
        "cartservice_latency", "cartservice_error",  # no internal properties
        "catalogue_cpu", "catalogue_mem", "catalogue_latency", "catalogue_error",
    ):
        metric[column] = rng.normal(10, 1.0, n)

    post = metric["time"] >= inject_time
    metric.loc[post, "cartservice_latency"] += 15.0
    metric.loc[post, "frontend_latency"] += 15.0 * 1.3  # inherited, amplified

    def span(span_id, parent, service, start, duration):
        return {"traceID": "t1", "spanID": span_id, "parentSpanID": parent,
                "serviceName": service, "operationName": f"{service}.op",
                "startTime": start * 1e6, "duration": duration, "statusCode": 0.0}

    traces = pd.DataFrame([
        span("s1", None, "frontend", inject_time + 10, 1_000_000),
        span("s2", "s1", "cartservice", inject_time + 10, 950_000),
        span("s3", "s2", "catalogue", inject_time + 10, 20_000),
    ])
    return {"metric": metric, "traces": traces}


def _demo() -> None:
    """The add-on's claims, as a runnable check, each under its stated condition."""
    from RCAEval.e2e.mars import mars
    from RCAEval.e2e.prism import _propagation_case

    # == Gated: on complete telemetry TG-FI changes nothing ==
    bounded = _propagation_case(external_amplification=1.7)
    traces = _missing_internal_case()["traces"]
    assert (
        tgfi(bounded, inject_time=60, dataset="demo", traces=traces)["ranks"]
        == mars(bounded, inject_time=60, dataset="demo")["ranks"]
    )

    # == Fires: the component whose internal metrics are missing is recovered ==
    case = _missing_internal_case()
    assert mars(case["metric"], inject_time=60, dataset="demo")["ranks"][0].startswith("frontend")
    assert tgfi(case, inject_time=60, dataset="demo")["ranks"][0].startswith("cartservice")

    # only that component is imputed: the instrumented ones keep their own S^I
    table, _, scores, _ = _scored(case, 60, "demo", None, "max", LAMBDA, {})
    assert _instrumented(scores) == {"frontend", "catalogue"}, _instrumented(scores)
    plain, _ = _component_table(
        _mars_scores(*_windows(case["metric"], 60, None, "demo")[:2], np.max), np.max
    )
    assert table["frontend"] == plain["frontend"] and table["catalogue"] == plain["catalogue"]
    assert table["cartservice"][0] > 0 and plain["cartservice"][0] == 0

    # the callee's share is what gets subtracted, per the formula
    edges = _call_edges(case["traces"], list(table))
    assert _callees(edges) == {"frontend": {"cartservice"}, "cartservice": {"catalogue"}}
    assert np.isclose(
        table["cartservice"][0],
        max(0.0, plain["cartservice"][1] - LAMBDA * plain["catalogue"][1]),
    )

    # == Fail-safe: no call graph, no imputation -- never worse than MARS ==
    assert (
        tgfi(case["metric"], inject_time=60, dataset="demo")["ranks"]
        == mars(case["metric"], inject_time=60, dataset="demo")["ranks"]
    )

    print("tgfi demo ok:", tgfi(case, inject_time=60, dataset="demo")["ranks"])


if __name__ == "__main__":
    _demo()
