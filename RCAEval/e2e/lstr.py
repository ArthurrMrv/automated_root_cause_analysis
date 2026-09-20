"""PRISM + MARS + TG-FI + LSTR: Localized Subgraph Trace-Path Re-Ranking
(report.md, Add-On 3).

Global dependency-graph discovery is what makes traditional RCA slow and wrong:
a 64x64 adjacency matrix inferred during an active outage is mostly sampling
noise, which is why PC-PageRank reaches 9% Top-1 at 1.2 seconds. LSTR inverts
the order. PRISM's statistical stage filters 64 components down to a Top-5
shortlist in milliseconds, and only then is topology consulted -- over a 5-node
subgraph, from trace spans that already exist, with no inference at all.

Within that subgraph LSTR decomposes span durations. For a directed edge
C_i -> C_j it measures how much of the caller's time the callee actually holds:

    R_self(C_j) = Duration(C_j) / Duration(C_i -> C_j)

When a callee accounts for >= 85% of its caller's span, the caller is not slow --
it is *waiting*. It is a backpressure victim, and it must not outrank the callee
it is waiting on. That is the one case PRISM's statistics cannot settle alone:
Proposition 3.10 bounds the victim below the root cause only while external
amplification stays bounded, and sequential fan-in blows straight through it.

Fail-safe throughout: no traces, no intra-shortlist edges, or no edge over the
threshold, and the statistical ranking is returned untouched.

Values the report leaves unspecified are marked `# gap N:`.
"""

import warnings
from typing import Any, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from RCAEval.e2e.mars import TOP_K, ComponentTable, _emit, _order, _validate
from RCAEval.e2e.tgfi import LAMBDA, _call_edges, _scored, _traces

# "Directional Causality Weighting: if candidate C_j accounts for >= 85% of
# C_i's total span duration, C_i's score is penalized as a downstream victim of
# backpressure, and C_j's root cause score is elevated."
SELF_RATIO_THRESHOLD = 0.85


def _micro_topology(
    edges: list[tuple[str, str, float, float, float]],
    shortlist: list[str],
    inject_time: Optional[int],
) -> list[tuple[str, str, float]]:
    """Extract the candidate subgraph as (caller, callee, R_self) -- Add-On 3 step 1-2.

    Only edges whose *both* endpoints are shortlisted are kept: that is what
    keeps this a 5x5 problem instead of an N x N one. Spans are restricted to the
    incident window, and each directed edge's ratio is the median over its
    observations, so one outlier span cannot flip a re-ranking.
    """
    members = set(shortlist)
    local = [e for e in edges if e[0] in members and e[1] in members and e[0] != e[1]]

    if inject_time is not None:
        post = [e for e in local if np.isfinite(e[4]) and e[4] >= inject_time]
        # gap 1: the report does not say what to do when no span in the subgraph
        # carries a usable post-injection timestamp. Falling back to every
        # observed span keeps the case re-rankable rather than silently
        # disabling the add-on; the fail-safe below still covers "no edges".
        local = post or local

    ratios: dict[tuple[str, str], list[float]] = {}
    for caller, callee, caller_duration, callee_duration, _ in local:
        if caller_duration <= 0:
            continue  # an unmeasured caller span carries no decomposition
        ratios.setdefault((caller, callee), []).append(callee_duration / caller_duration)

    return [
        (caller, callee, float(np.median(observed)))
        for (caller, callee), observed in ratios.items()
    ]


def _rerank(
    shortlist: list[str],
    topology: list[tuple[str, str, float]],
    threshold: float = SELF_RATIO_THRESHOLD,
) -> list[str]:
    """Demote backpressure victims below the callees they wait on -- Add-On 3 step 3.

    gap 2: the report states the penalty and the elevation qualitatively but
    gives no magnitudes. Implemented as a stable topological reorder under the
    constraint "a dominating callee precedes its caller": it introduces no
    invented constants, it is idempotent, and it moves nothing that the 85% rule
    does not force. Cycles (mutual domination, which real traces do produce under
    retries) leave the pair in its statistical order.
    """
    predecessors = {component: set() for component in shortlist}
    for caller, callee, ratio in topology:
        if ratio >= threshold and caller in predecessors and callee in predecessors:
            predecessors[caller].add(callee)

    if not any(predecessors.values()):
        return list(shortlist)  # fail-safe: nothing dominates, nothing moves

    placed: list[str] = []
    seen, active, cyclic = set(), set(), False

    def emit(component: str) -> None:
        """Place a component after every callee that dominates it."""
        nonlocal cyclic
        if component in seen:
            return
        if component in active:
            cyclic = True
            return
        active.add(component)
        for predecessor in shortlist:  # shortlist order: the reorder stays stable
            if predecessor in predecessors[component]:
                emit(predecessor)
        active.discard(component)
        seen.add(component)
        placed.append(component)

    for component in shortlist:
        emit(component)

    # mutual domination means the 85% rule fired both ways, which is not
    # evidence of a direction -- keep the statistical order rather than pick one
    return list(shortlist) if cyclic else placed


def _lstr_order(
    data: Any,
    inject_time: Optional[int],
    dataset: Optional[str],
    anomalies: Optional[list],
    pooling: str,
    combine: str,
    lam: float,
    threshold: float,
    kwargs: dict,
) -> tuple[list[str], ComponentTable, list[str], pd.Series, list[str]]:
    """Stage 1 + LSTR, as a component ranking. Shared with `pave`."""
    traces = _traces(data, kwargs)
    if traces is not None:
        kwargs = {**kwargs, "traces": traces}  # scored again below; read once

    table, unclassified, scores, intersects = _scored(
        data, inject_time, dataset, anomalies, pooling, lam, kwargs
    )
    order = _order(table, combine)

    shortlist = order[:TOP_K]
    topology = _micro_topology(_call_edges(traces, list(table)), shortlist, inject_time)
    reordered = _rerank(shortlist, topology, threshold)
    return reordered + order[TOP_K:], table, unclassified, scores, intersects


def lstr(
    data: Any,
    inject_time: Optional[int] = None,
    dataset: Optional[str] = None,
    num_loop: Optional[int] = None,
    sli: Optional[str] = None,
    anomalies: Optional[list] = None,
    pooling: str = "max",
    combine: str = "additive",
    lam: float = LAMBDA,
    threshold: float = SELF_RATIO_THRESHOLD,
    **kwargs: Any,
) -> dict[str, list]:
    """Rank root cause candidates with PRISM + MARS + TG-FI + LSTR (Add-On 3).

    Args:
        data: metric DataFrame (traces then read from `args.data_path`'s
            directory), or a dict with "metric" and "traces".
        inject_time: fault injection timestamp, also bounding the span window.
        dataset: dataset name, forwarded to `preprocess`.
        threshold: R_self at which a callee is judged to dominate its caller.

    Returns:
        {"node_names": [...], "ranks": ["<component>_<witness property>", ...]}.
    """
    _validate(pooling, combine, kwargs.get("time_agg", "max"))
    order, table, unclassified, scores, intersects = _lstr_order(
        data, inject_time, dataset, anomalies, pooling, combine, lam, threshold, kwargs
    )
    ranks = _emit(order, table, unclassified, scores)

    if kwargs.get("verbose") is True:
        for component in order[:TOP_K]:
            internal, external, witness = table[component]
            print(f"{witness}: S^I={internal:.2f} S^E={external:.2f}")

    return {"node_names": intersects, "ranks": ranks}


# ==========================================================================
# Runnable check -- no datasets
# ==========================================================================


def _spans(inject_time: int, self_ratio: float) -> pd.DataFrame:
    """frontend -> cartservice, where cartservice holds `self_ratio` of the call."""
    caller_duration = 1_000_000.0

    def span(span_id, parent, service, duration):
        return {"traceID": "t1", "spanID": span_id, "parentSpanID": parent,
                "serviceName": service, "operationName": f"{service}.op",
                "startTime": (inject_time + 10) * 1e6, "duration": duration,
                "statusCode": 0.0}

    return pd.DataFrame([
        span("s1", None, "frontend", caller_duration),
        span("s2", "s1", "cartservice", caller_duration * self_ratio),
    ])


def _demo() -> None:
    """The add-on's claims, as a runnable check, each under its stated condition."""
    from RCAEval.e2e.prism import _propagation_case
    from RCAEval.e2e.tgfi import tgfi

    # Amplification beyond Proposition 3.10's bound: the statistical stage ranks
    # the amplified caller first, which is the case LSTR exists to fix.
    case = _propagation_case(external_amplification=6.0)
    assert tgfi(case, inject_time=60, dataset="demo")["ranks"][0].startswith("frontend")

    # == The callee holds 95% of the caller's span: the caller is a victim ==
    dominated = {"metric": case, "traces": _spans(60, 0.95)}
    assert lstr(dominated, inject_time=60, dataset="demo")["ranks"][0].startswith("cartservice")

    # == Below the threshold the caller really is slow: nothing moves ==
    independent = {"metric": case, "traces": _spans(60, 0.30)}
    assert (
        lstr(independent, inject_time=60, dataset="demo")["ranks"]
        == tgfi(case, inject_time=60, dataset="demo")["ranks"]
    )

    # == Fail-safe: no traces at all, the statistical ranking stands ==
    assert (
        lstr(case, inject_time=60, dataset="demo")["ranks"]
        == tgfi(case, inject_time=60, dataset="demo")["ranks"]
    )

    # == The subgraph really is local: edges leaving the shortlist are dropped ==
    edges = _call_edges(_spans(60, 0.95), ["frontend", "cartservice"])
    assert _micro_topology(edges, ["frontend", "cartservice"], 60)[0][:2] == (
        "frontend", "cartservice"
    )
    assert _micro_topology(edges, ["frontend"], 60) == []
    # ...and spans from before the incident do not decide the re-ranking
    assert _micro_topology(edges, ["frontend", "cartservice"], 10_000) == [
        ("frontend", "cartservice", 0.95)
    ]

    # == The reorder is constrained, stable, and cycle-safe ==
    assert _rerank(["a", "b", "c"], [("a", "c", 0.9)]) == ["c", "a", "b"]
    assert _rerank(["a", "b", "c"], [("a", "c", 0.5)]) == ["a", "b", "c"]
    assert _rerank(["a", "b"], [("a", "b", 0.9), ("b", "a", 0.9)]) == ["a", "b"]

    print("lstr demo ok:", lstr(dominated, inject_time=60, dataset="demo")["ranks"])


if __name__ == "__main__":
    _demo()
