"""PAVE-RCA: PRISM-Agentic Verification & Explanation (report.md, Stage 2).

Unconstrained agentic RCA reasons over all of an incident's telemetry at once.
`rclagent` is that design, and the report measures where it lands: context
explosion, 79 seconds, 10% Top-1. PAVE-RCA keeps the LLM and throws away the
context. PRISM's statistical stage has already reduced the system to five
candidates in milliseconds, and the agent only ever sees those five.

That bound is what makes the remaining ambiguity tractable. Statistics cannot
separate a service that is slow because it is broken from one that is slow
because it is waiting, when neither shows a resource anomaly -- a null-pointer
exception, a lock contention, a socket leak leave almost no metric signature. A
log line does. The agent's cross-modal sanity rule is the report's: a candidate
whose logs carry an explicit *local* failure (OutOfMemoryError, panic, connection
refused, a stack trace) outranks a candidate showing only generic downstream
timeouts (504 and friends), whatever their statistical order.

Three bounded tools per candidate -- logs, spans, config -- then one call, then a
ranking and a narrative. Every failure path (no API key, an exception, an
unparseable or nonsense response) falls back to the LSTR ranking, so the agent
can only ever improve on the statistical stage, never degrade it.

Values the report leaves unspecified are marked `# gap N:`.
"""

import warnings
from typing import Any, Callable, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from RCAEval.e2e.lstr import SELF_RATIO_THRESHOLD, _lstr_order
from RCAEval.e2e.mars import TOP_K, _emit, _validate
from RCAEval.e2e.tgfi import LAMBDA, _case_dir, _traces
from RCAEval.e2e.rclagent import (
    DEFAULT_MODEL,
    DELTA_SECONDS,
    _build_spans,
    _children_of,
    _default_llm,
    _log_tool,
    _parse_json,
)
from RCAEval.utility import read_logs

# The report's own examples of an explicit local failure, versus the generic
# downstream symptom a victim shows. Named here because the prompt states the
# rule and the self-check exercises it.
EXPLICIT_FAILURE = (
    "outofmemoryerror", "connectionrefused", "connection refused", "panic",
    "stack trace", "traceback", "segmentation fault", "deadlock", "nullpointer",
    "null pointer", "oomkilled", "too many open files",
)
GENERIC_SYMPTOM = ("504", "gateway timeout", "deadline exceeded", "upstream timeout")

# gap 1: the report gives no per-candidate evidence cap. 12 log lines and the
# span summary keep all five candidates inside a single bounded prompt, which is
# the entire point of the stage.
LOG_LIMIT = 12

_PAVE_SYSTEM = (
    "You are the PAVE verification agent for a microservice incident. A fast "
    "statistical engine has already shortlisted the candidates below and ranked "
    "them; you see their anomaly scores plus logs, span timings and configuration "
    "history for those candidates only.\n"
    "Re-rank them by which is the true root cause. Apply this rule: a candidate "
    "whose logs show an explicit local failure (OutOfMemoryError, connection "
    "refused, panic, stack trace, segmentation fault) is more likely the root "
    "cause than one showing only generic downstream symptoms (504 Gateway "
    "Timeout, deadline exceeded), even when the latter's anomaly scores are "
    "larger -- large external latency with a quiet interior is what a victim "
    "looks like. When the evidence adds nothing, keep the statistical order.\n"
    'Reply with JSON only: {"ranking": ["<candidate>", ...], "narrative": '
    '"<two sentences: what failed, and what to do about it>"}. Rank every '
    "candidate you are given, most likely first, spelling names exactly as given."
)


# ==========================================================================
# Bounded tools -- each scoped to one candidate and the incident window
# ==========================================================================


def _logs(data: Any, kwargs: dict) -> Optional[pd.DataFrame]:
    """Logs for this case, or None when it ships none (RE1)."""
    if isinstance(data, dict) and data.get("logs") is not None:
        return data["logs"]
    if kwargs.get("logs") is not None:
        return kwargs["logs"]
    case_dir = _case_dir(kwargs)
    if case_dir is None:
        return None
    try:
        return read_logs(case_dir)
    except Exception:  # unreadable logs are "no logs", not a crash
        return None


def inspect_candidate_logs(
    logs: Optional[pd.DataFrame], service: str, t0: float, delta: int = DELTA_SECONDS
) -> list[str]:
    """Tool 1: error logs, unhandled exceptions and panic stack traces.

    Reuses `rclagent`'s relevance function phi(l) -- severity, keywords, error
    codes -- so both agentic methods read the same log vocabulary.
    """
    if logs is None or logs.empty:
        return []
    return _log_tool(logs, service, t0, delta, limit=LOG_LIMIT)


def inspect_candidate_spans(
    traces: Optional[pd.DataFrame],
    service: str,
    t0: float,
    candidates: list[str],
    delta: int = DELTA_SECONDS,
) -> list[str]:
    """Tool 2: the candidate's own execution time versus its boundary latency.

    A service that is waiting spends its span inside its callees; a service that
    is broken spends it in itself. That ratio is the span-level counterpart of
    PRISM's internal/external split, and it is what separates the two when no
    resource metric moved.
    """
    if traces is None or traces.empty:
        return []
    spans = _build_spans(traces, candidates)
    children = _children_of(spans)

    own, self_fractions = [], []
    for span in spans.values():
        if span.service != service:
            continue
        if np.isfinite(span.start_time) and abs(span.start_time - t0) > delta:
            continue
        callee_time = sum(spans[c].duration for c in children.get(span.span_id, []))
        own.append(span.duration)
        if span.duration > 0:
            self_fractions.append(max(0.0, span.duration - callee_time) / span.duration)

    if not own:
        return []
    line = f"{len(own)} spans, median duration {np.median(own) / 1000:.1f} ms"
    if self_fractions:
        fraction = float(np.median(self_fractions))
        line += (
            f", median {fraction:.0%} of that spent in {service} itself "
            f"({1 - fraction:.0%} waiting on callees)"
        )
    return [line]


def inspect_config_changes(service: str, t0: float, delta: int = DELTA_SECONDS) -> list[str]:
    """Tool 3: recent deployments, feature flags and environment changes.

    gap 2: RCAEval ships no deployment, feature-flag or environment telemetry in
    any suite, so on this benchmark the tool has nothing to read. It reports that
    honestly rather than being dropped from the agent's toolset or, worse,
    inferring a config change from metrics. Wire it to a real change feed and the
    agent uses it without any other modification.
    """
    return ["no configuration telemetry available for this case"]


def _evidence_block(
    shortlist: list[str],
    table: dict[str, tuple[float, float, str]],
    logs: Optional[pd.DataFrame],
    traces: Optional[pd.DataFrame],
    t0: float,
    candidates: list[str],
    delta: int,
) -> str:
    """Gather all three tools for all five candidates into one bounded prompt.

    The shortlist is small and fixed, so every tool can be called up front: the
    agent needs no tool-calling loop, which is one round trip instead of a dozen.
    """
    blocks = []
    for position, service in enumerate(shortlist, start=1):
        internal, external, _ = table[service]
        sections = [
            ("logs", inspect_candidate_logs(logs, service, t0, delta)),
            ("spans", inspect_candidate_spans(traces, service, t0, candidates, delta)),
            ("config", inspect_config_changes(service, t0, delta)),
        ]
        rendered = "\n".join(
            f"  {name}:\n" + ("\n".join(f"    - {line}" for line in lines) or "    - (none)")
            for name, lines in sections
        )
        blocks.append(
            f"Candidate {position}: {service} "
            f"(S^I={internal:.2f}, S^E={external:.2f}, statistical rank {position})\n{rendered}"
        )
    return "\n\n".join(blocks)


# ==========================================================================
# Stage 2: verification and re-ranking
# ==========================================================================


def _verified_order(
    shortlist: list[str], prompt: str, llm: Callable[[str, str], str]
) -> tuple[list[str], str]:
    """One call, then validate. Anything unusable leaves the shortlist as it was."""
    parsed = _parse_json(llm(_PAVE_SYSTEM, prompt))
    ranking = parsed.get("ranking", [])
    if not isinstance(ranking, list):
        return list(shortlist), ""

    seen, ordered = set(), []
    for name in ranking:
        name = str(name)
        if name in shortlist and name not in seen:
            seen.add(name)
            ordered.append(name)
    if not ordered:
        return list(shortlist), ""

    # a partial answer is still useful: candidates the agent did not name keep
    # their statistical order behind the ones it did
    ordered += [c for c in shortlist if c not in seen]
    return ordered, str(parsed.get("narrative", ""))


def pave(
    data: Any,
    inject_time: Optional[int] = None,
    dataset: Optional[str] = None,
    num_loop: Optional[int] = None,
    sli: Optional[str] = None,
    anomalies: Optional[list] = None,
    llm: Optional[Callable[[str, str], str]] = None,
    pooling: str = "max",
    combine: str = "additive",
    lam: float = LAMBDA,
    threshold: float = SELF_RATIO_THRESHOLD,
    **kwargs: Any,
) -> dict[str, list]:
    """Rank root cause candidates with PAVE-RCA (report.md, Stage 2).

    Stage 1 is the full PRISM + MARS + TG-FI + LSTR stack; Stage 2 verifies its
    Top-5 against logs, spans and configuration history.

    Args:
        data: metric DataFrame (logs and traces then read from `args.data_path`'s
            directory), or a dict with "metric", "traces" and "logs".
        inject_time: fault injection timestamp, centering the tools' window.
        dataset: dataset name, forwarded to `preprocess`.
        llm: backbone as a callable (system, user) -> str. Defaults to the
            Claude backbone; pass your own to swap models or to run offline.

    Returns:
        {"node_names": [...], "ranks": [...], "narrative": "..."} -- ranks as
        "<component>_<witness property>", most likely root cause first.
    """
    _validate(pooling, combine, kwargs.get("time_agg", "max"))
    order, table, unclassified, scores, intersects = _lstr_order(
        data, inject_time, dataset, anomalies, pooling, combine, lam, threshold, kwargs
    )
    shortlist = order[:TOP_K]
    narrative = ""

    # Fail-safe: every way the agent can fail ends on the statistical ranking,
    # so PAVE-RCA is bounded below by LSTR rather than by the LLM's worst day.
    try:
        if inject_time is None:
            raise ValueError("pave needs inject_time to window its tools")
        prompt = _evidence_block(
            shortlist,
            table,
            _logs(data, kwargs),
            _traces(data, kwargs),
            float(inject_time),
            list(table),
            int(kwargs.get("delta", DELTA_SECONDS)),
        )
        backbone = llm or _default_llm(kwargs.get("model", DEFAULT_MODEL))
        verified, narrative = _verified_order(shortlist, prompt, backbone)
    except Exception as exc:
        if kwargs.get("verbose") is True:
            print(f"pave: falling back to the statistical ranking ({exc})")
        verified = shortlist

    ranks = _emit(verified + order[TOP_K:], table, unclassified, scores)

    if kwargs.get("verbose") is True:
        print(f"narrative: {narrative or '(none)'}")
        for rank in ranks[:TOP_K]:
            print(f"  {rank}")

    return {"node_names": intersects, "ranks": ranks, "narrative": narrative}


# ==========================================================================
# Runnable check -- no API key, no datasets
# ==========================================================================


def _scripted_llm() -> Callable[[str, str], str]:
    """A stub backbone that applies the report's sanity rule, and only that.

    It reads nothing but the candidate blocks: a candidate whose evidence names
    an explicit local failure rises above one showing a generic symptom. That is
    the minimum an LLM must do here, so the check fails if the tools' evidence
    never reaches the prompt.
    """
    import json

    def call(system: str, user: str) -> str:
        candidates = []
        for block in user.split("Candidate ")[1:]:
            name = block.split(":", 1)[1].split("(")[0].strip()
            lowered = block.lower()
            explicit = any(marker in lowered for marker in EXPLICIT_FAILURE)
            generic = any(marker in lowered for marker in GENERIC_SYMPTOM)
            candidates.append((-int(explicit), int(generic), name))
        ranking = [name for *_, name in sorted(candidates, key=lambda x: x[:2])]
        return json.dumps({"ranking": ranking, "narrative": "explicit local failure found"})

    return call


def _exception_case() -> dict:
    """The victim outranks the culprit statistically; only the logs disagree.

    `cartservice` throws OutOfMemoryError, but the fault is a software one and
    barely moves its resources, while `frontend` -- waiting on it -- shows six
    times the latency anomaly and nothing but 504s in its logs.
    """
    from RCAEval.e2e.prism import _propagation_case

    metric = _propagation_case(external_amplification=6.0)
    logs = pd.DataFrame({
        "time": [65, 66, 67],
        "serviceName": ["frontend", "cartservice", "adservice"],
        "message": [
            "ERROR 504 Gateway Timeout calling cartservice",
            "ERROR java.lang.OutOfMemoryError: Java heap space",
            "WARN cache miss ratio elevated",
        ],
    })
    return {"metric": metric, "logs": logs}


def _demo() -> None:
    """The stage's claims, as a runnable check, each under its stated condition."""
    from RCAEval.e2e.lstr import lstr

    case = _exception_case()
    baseline = lstr(case["metric"], inject_time=60, dataset="demo")["ranks"]

    # the statistical stack ranks the amplified victim first -- correctly, on
    # metrics alone: a software fault leaves almost no resource signature
    assert baseline[0].startswith("frontend"), baseline

    # == Cross-modal sanity: the explicit exception outranks the generic 504 ==
    out = pave(case, inject_time=60, dataset="demo", llm=_scripted_llm())
    assert out["ranks"][0].startswith("cartservice"), out["ranks"]
    assert out["narrative"], out
    # the shortlist is re-ordered, never truncated: Avg@k stays well defined
    assert sorted(out["ranks"]) == sorted(baseline), out["ranks"]

    # == The tools really are bounded to one candidate and one window ==
    logs = case["logs"]
    assert inspect_candidate_logs(logs, "cartservice", 60.0) == [
        "ERROR java.lang.OutOfMemoryError: Java heap space"
    ]
    assert inspect_candidate_logs(logs, "cartservice", 10_000.0) == []
    assert inspect_candidate_logs(None, "cartservice", 60.0) == []
    assert inspect_config_changes("cartservice", 60.0)  # present, and honest

    # spans decompose waiting from working
    from RCAEval.e2e.lstr import _spans
    evidence = inspect_candidate_spans(_spans(60, 0.95), "frontend", 70.0, ["frontend", "cartservice"])
    assert "waiting on callees" in evidence[0], evidence
    assert "5%" in evidence[0], evidence

    # == Fail-safe: every agent failure lands on the statistical ranking ==
    def broken(system: str, user: str) -> str:
        raise RuntimeError("backbone unavailable")

    assert pave(case, inject_time=60, dataset="demo", llm=broken)["ranks"] == baseline
    assert pave(case, inject_time=60, dataset="demo", llm=lambda s, u: "not json")["ranks"] == baseline
    assert pave(case, inject_time=60, dataset="demo",
                llm=lambda s, u: '{"ranking": ["nonexistent"]}')["ranks"] == baseline

    print("pave demo ok:", pave(case, inject_time=60, dataset="demo", llm=_scripted_llm())["ranks"])


if __name__ == "__main__":
    _demo()
