"""RCLAgent: multi-agent recursion-of-thought root cause localization.

Zhang, Jia, Wang, Duan, He, Wang, Peng, Wang, Zhang, Chen, Li -- "Towards
In-Depth Root Cause Localization for Microservices with Multi-Agent
Recursion-of-Thought" (arXiv:2605.14866).

The paper's thesis is that a single LLM prompt over a whole incident suffers
*context explosion* (critical evidence diluted) and *serial reasoning* (no deep
causal exploration). RCLAgent instead assigns one Dedicated Agent per span of the
trace graph, so every agent reasons over a bounded, span-local context, and
organizes the agents recursively along the trace topology with independent
branches running in parallel.

Pipeline (Figure 2 / Section IV):

  IV-A  Data Tools ..... Trace Tool T(s) (Eq. 2), Metric Tool Q(t0, d, C)
                         (Eq. 3-4, n-sigma), Log Tool L(t0, d, C) (Eq. 5).
  IV-B  Multi-Agent RoT  per span: Self-State Verification f_self (Eq. 6) then
                         Evidence Consolidation f_cons (Eq. 7) over the child
                         agents' consolidated evidence.
  IV-C  Synthesizer .... Root-Level Diagnosis Report e_root plus the Global
                         Evidence Graph G_E = (V, E, Phi), jointly reasoned over
                         by F_synth (Eq. 10) into the ranked list R (Eq. 1).

Hyperparameters are the paper's Section V-A4 values: n = 3, delta = 60s, agents
pool K = 100, Claude-3.5-Sonnet backbone.

No reference implementation ships with the paper; everything it leaves
unspecified is marked `# gap N:` below.
"""

import json
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from RCAEval.io.time_series import preprocess

# Section V-A4: "n = 3", "delta = 60s", "K = 100", Claude-3.5-Sonnet backbone.
N_SIGMA = 3.0
DELTA_SECONDS = 60
POOL_CAPACITY = 100
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

# Section V-B: the paper diagnoses "abnormal requests whose latency exceeds 100
# times the normal average", which is how the single trace under diagnosis is
# picked out of the incident window.
ABNORMAL_LATENCY_FACTOR = 100.0

# Log Tool relevance function phi(l), Eq. 5: "severity, keywords, error codes".
# The paper names the three signals but not the vocabulary -- gap 1.
LOG_SEVERITIES = ("error", "fatal", "critical", "severe", "warn")
LOG_KEYWORDS = ("exception", "timeout", "timed out", "refused", "unavailable",
                "failed", "failure", "panic", "deadline", "reset by peer", "oom")
LOG_ERROR_CODES = re.compile(r"\b(4\d{2}|5\d{2})\b|\bcode\s*[=:]\s*\d+")

# shortest shared prefix that counts as the same service when trace and metric
# spellings differ (gap 5)
MIN_SERVICE_PREFIX = 4

SYNTHESIZERS = ("full", "rldr", "geg")


@dataclass(frozen=True)
class Span:
    """One row of traces.csv, as the Trace Tool T(s) returns it (Eq. 2)."""

    span_id: str
    parent_id: Optional[str]
    service: str
    operation: str
    start_time: float  # epoch seconds
    duration: float  # microseconds, as traces.csv records it
    status: float


@dataclass(frozen=True)
class SelfEvidence:
    """e_s = <s_id, s_svc, a_s, k_s, h_s> -- Self-State Verification, Eq. 6."""

    span_id: str
    service: str
    abnormal: bool
    symptoms: str
    hypothesis: str


@dataclass(frozen=True)
class Consolidated:
    """e^_s = <s_id, s_svc, r_s, r_s^reason, r_s^conf> -- Eq. 7."""

    span_id: str
    service: str
    root_cause: str
    reason: str
    confidence: float


# ==========================================================================
# LLM backbone
# ==========================================================================


def _default_llm(model: str = DEFAULT_MODEL) -> Callable[[str, str], str]:
    """Claude-3.5-Sonnet backbone (Section V-A4), via the anthropic SDK.

    Returns a callable (system_prompt, user_prompt) -> text. Pass your own with
    the `llm=` kwarg to swap in any of the paper's other backbones (Qwen,
    DeepSeek-R1, GPT-4, Llama-3.1-70B) or a local stub.
    """
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "rclagent needs an LLM backbone: `pip install anthropic` and set "
            "ANTHROPIC_API_KEY, or pass llm=<callable(system, user) -> str>"
        ) from exc

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError(
            "rclagent needs ANTHROPIC_API_KEY set, or an explicit "
            "llm=<callable(system, user) -> str>"
        )

    client = anthropic.Anthropic()

    def call(system: str, user: str) -> str:
        # gap 2: the paper reports no temperature or max_tokens. 0 is the
        # reproducibility-friendly choice for a diagnostic task; 1024 tokens
        # comfortably holds the small JSON objects of Figures 5-6.
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in message.content if block.type == "text")

    return call


def _parse_json(text: str) -> dict:
    """Read the JSON object the Figure 5/6 prompts ask for.

    Models fence or preface JSON often enough that a bare `json.loads` is not
    safe here; fall back to the outermost braces. Returns {} when the response
    carries no object at all, which the callers treat as "no evidence".
    """
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ==========================================================================
# IV-A: Data Tools
# ==========================================================================


def _match_service(name: str, candidates: list[str]) -> str:
    """Map a span's serviceName onto the harness's component vocabulary.

    gap 5: traces.csv and the metric columns do not spell services alike --
    RE2-OB traces say "frontendservice" where the metrics say "frontend", and
    "-db" suffixes appear on one side only. Unmatched names would be dropped by
    the synthesizer's candidate filter, silently burying a correct answer in the
    tail, so they are matched here: exact, then on the common "service" affix,
    then by longest shared prefix. A name that still matches nothing is kept
    as-is rather than forced onto a wrong candidate.
    """
    if name in candidates:
        return name
    lowered = name.lower()
    by_lower = {c.lower(): c for c in candidates}
    if lowered in by_lower:
        return by_lower[lowered]
    for variant in (lowered.removesuffix("service"), lowered.removeprefix("service"),
                    lowered + "service", "service" + lowered, lowered.removesuffix("-db")):
        if variant in by_lower:
            return by_lower[variant]
    # longest shared prefix, but only when it is long enough to be meaningful:
    # "cart" vs "cartservice" is a match, "ts-a" vs "ts-b" is not
    best, best_len = None, 0
    for candidate in candidates:
        shared = os.path.commonprefix([lowered, candidate.lower()])
        if len(shared) > best_len:
            best, best_len = candidate, len(shared)
    return best if best is not None and best_len >= MIN_SERVICE_PREFIX else name


def _build_spans(traces: pd.DataFrame, candidates: list[str]) -> dict[str, Span]:
    """Parse traces.csv into Spans keyed by span id.

    RCAEval's traces.csv carries traceID, spanID, parentSpanID, serviceName,
    operationName, startTime (microseconds), duration, statusCode. Service names
    are normalized onto `candidates` so every downstream agent, and the final
    ranking, speak the harness's vocabulary.
    """
    spans = {}
    for row in traces.itertuples(index=False):
        span_id = str(getattr(row, "spanID", "") or "")
        if not span_id:
            continue
        parent = getattr(row, "parentSpanID", None)
        parent_id = None if parent is None or pd.isna(parent) else str(parent)
        start = getattr(row, "startTime", np.nan)
        if pd.isna(start):
            start = getattr(row, "startTimeMillis", np.nan) * 1e3
        spans[span_id] = Span(
            span_id=span_id,
            parent_id=parent_id,
            service=_match_service(str(getattr(row, "serviceName", "") or ""), candidates),
            operation=str(getattr(row, "operationName", "") or ""),
            start_time=float(start) / 1e6 if not pd.isna(start) else float("nan"),
            duration=float(getattr(row, "duration", 0.0) or 0.0),
            status=float(getattr(row, "statusCode", 0.0) or 0.0),
        )
    return spans


def _children_of(spans: dict[str, Span]) -> dict[Optional[str], list[str]]:
    """Trace Tool T(s), Eq. 2: the child span set C(s), as an adjacency map."""
    children: dict[Optional[str], list[str]] = {}
    for span in spans.values():
        parent = span.parent_id if span.parent_id in spans else None
        children.setdefault(parent, []).append(span.span_id)
    # deterministic agent order: earliest span first, so reruns rank alike
    for parent, kids in children.items():
        children[parent] = sorted(kids, key=lambda s: (spans[s].start_time, s))
    return children


def _select_trace(traces: pd.DataFrame, inject_time: float) -> pd.DataFrame:
    """Pick the abnormal request to diagnose (Section V-B).

    The paper diagnoses "abnormal requests whose latency exceeds 100 times the
    normal average" and treats one trace per failure episode. Root-span duration
    stands in for request latency; the normal average is taken over the
    pre-injection window.
    """
    if "traceID" not in traces.columns or traces.empty:
        raise ValueError("rclagent needs traces.csv with a traceID column")

    roots = traces[traces["parentSpanID"].isna()] if "parentSpanID" in traces else traces
    if roots.empty:  # trace fragments with no captured entry span
        roots = traces.loc[traces.groupby("traceID")["duration"].idxmax()]

    seconds = roots["startTime"] / 1e6 if "startTime" in roots else roots["time"]
    normal = roots[seconds < inject_time]
    anomal = roots[seconds >= inject_time]
    if anomal.empty:
        raise ValueError("rclagent found no trace after inject_time")

    baseline = float(normal["duration"].mean()) if not normal.empty else 0.0
    abnormal = anomal[anomal["duration"] > ABNORMAL_LATENCY_FACTOR * baseline]
    # gap 3: the paper never says what to diagnose when no request clears 100x
    # (it happens for the non-latency faults RCAEval injects). Falling back to
    # the slowest post-injection request keeps every case diagnosable; drop the
    # fallback to score only the cases the paper's own filter admits.
    pool = abnormal if not abnormal.empty else anomal
    trace_id = pool.loc[pool["duration"].idxmax(), "traceID"]
    return traces[traces["traceID"] == trace_id]


def _metric_tool(
    metric: pd.DataFrame, service: str, t0: float, delta: int, n_sigma: float
) -> list[str]:
    """Metric Tool Q(t0, delta, C), Eq. 3-4.

    n-sigma filter |m(t) - mu_m| > n * sigma_m over [t0 - delta, t0 + delta],
    restricted to the properties of component C. mu_m and sigma_m are the
    baseline moments, taken over everything before the window -- the paper says
    "historical baselines" without pinning the estimation window (gap 4).
    """
    if metric is None or metric.empty or "time" not in metric.columns:
        return []
    columns = [c for c in metric.columns if c != "time" and c.split("_")[0] == service]
    if not columns:
        return []

    window = metric[(metric["time"] >= t0 - delta) & (metric["time"] <= t0 + delta)]
    baseline = metric[metric["time"] < t0 - delta]
    if window.empty:
        return []
    if baseline.empty:  # window covers the whole capture; fall back to it
        baseline = metric

    anomalous = []
    for column in columns:
        mu = float(baseline[column].mean())
        sigma = float(baseline[column].std())
        if not np.isfinite(sigma) or sigma == 0:
            continue
        deviation = (window[column] - mu).abs()
        peak = float(deviation.max())
        if peak > n_sigma * sigma:
            anomalous.append(
                f"{column}: peak deviation {peak / sigma:.1f} sigma "
                f"(baseline mean {mu:.3g}, window mean {float(window[column].mean()):.3g})"
            )
    return anomalous


def _log_tool(
    logs: pd.DataFrame, service: str, t0: float, delta: int, limit: int = 20
) -> list[str]:
    """Log Tool L(t0, delta, C), Eq. 5: logs of C in window, kept when phi(l)=1."""
    if logs is None or logs.empty:
        return []

    service_col = next(
        (c for c in ("serviceName", "service", "container", "cmdb_id", "pod") if c in logs.columns),
        None,
    )
    message_col = next(
        (c for c in ("message", "log", "value", "body", "content") if c in logs.columns), None
    )
    if service_col is None or message_col is None:
        return []

    subset = logs[logs[service_col].astype(str).str.contains(service, case=False, na=False)]
    if "time" in subset.columns:
        seconds = pd.to_numeric(subset["time"], errors="coerce")
        subset = subset[(seconds >= t0 - delta) & (seconds <= t0 + delta)]

    relevant = []
    for message in subset[message_col].astype(str):
        lowered = message.lower()
        if (
            any(sev in lowered for sev in LOG_SEVERITIES)
            or any(kw in lowered for kw in LOG_KEYWORDS)
            or LOG_ERROR_CODES.search(message)
        ):
            relevant.append(message[:300])
        if len(relevant) >= limit:  # bounded span-local context (Section IV-B)
            break
    return relevant


# ==========================================================================
# IV-B: Multi-Agent Recursion-of-Thought
# ==========================================================================

_SELF_SYSTEM = (
    "You are a Dedicated Agent assigned to exactly one span of a microservice "
    "trace. Judge only your own span from its trace attributes, metrics and "
    "logs. Do not speculate about other services. Reply with JSON only: "
    '{"abnormal": true|false, "symptoms": "<key symptoms, one sentence>", '
    '"hypothesis": "<why, one sentence>"}'
)

_CONS_SYSTEM = (
    "You are a Dedicated Agent consolidating your own span evidence with the "
    "consolidated evidence of your child spans. Decide where the root cause "
    "lies. Reply with JSON only: "
    '{"root_cause": "<service name, or \'self\', or \'upstream\' to defer to '
    'your parent>", "reason": "<concise causal rationale>", '
    '"confidence": <0.0-1.0>}'
)

_SYNTH_SYSTEM = (
    "You are the Diagnosis Synthesizer for a microservice incident. Rank the "
    "candidate services by how likely each is the root cause. Reply with JSON "
    'only: {"ranking": ["<service>", ...], "reason": "<one sentence>"}. '
    "Rank every candidate you are given, most likely first, using only the "
    "candidate names exactly as spelled."
)


def _self_state_verification(
    span: Span, metric_lines: list[str], log_lines: list[str], llm: Callable[[str, str], str]
) -> SelfEvidence:
    """f_self: (T(s), Q_log, Q_metric) -> e_s (Eq. 6)."""
    prompt = (
        f"Span: {span.span_id}\n"
        f"Service: {span.service}\n"
        f"Operation: {span.operation}\n"
        f"Duration: {span.duration:.0f} us\n"
        f"Status code: {span.status:.0f}\n"
        f"Anomalous metrics ({N_SIGMA:.0f}-sigma):\n"
        + ("\n".join(f"  - {line}" for line in metric_lines) or "  (none)")
        + "\nRelevant logs:\n"
        + ("\n".join(f"  - {line}" for line in log_lines) or "  (none)")
    )
    parsed = _parse_json(llm(_SELF_SYSTEM, prompt))
    return SelfEvidence(
        span_id=span.span_id,
        service=span.service,
        abnormal=bool(parsed.get("abnormal", False)),
        symptoms=str(parsed.get("symptoms", "")),
        hypothesis=str(parsed.get("hypothesis", "")),
    )


def _evidence_consolidation(
    evidence: SelfEvidence,
    downstream: list[Consolidated],
    llm: Callable[[str, str], str],
) -> Consolidated:
    """f_cons: (e_s, E_down(s)) -> e^_s (Eq. 7)."""
    prompt = (
        f"Your span: {evidence.span_id} ({evidence.service})\n"
        f"Abnormal: {evidence.abnormal}\n"
        f"Symptoms: {evidence.symptoms or '(none)'}\n"
        f"Hypothesis: {evidence.hypothesis or '(none)'}\n"
        "Downstream child evidence:\n"
        + (
            "\n".join(
                f"  - {c.service} (span {c.span_id}): root_cause={c.root_cause}, "
                f"confidence={c.confidence:.2f}, reason={c.reason}"
                for c in downstream
            )
            or "  (leaf span, no children)"
        )
    )
    parsed = _parse_json(llm(_CONS_SYSTEM, prompt))
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return Consolidated(
        span_id=evidence.span_id,
        service=evidence.service,
        root_cause=str(parsed.get("root_cause", "") or ""),
        reason=str(parsed.get("reason", "")),
        confidence=min(max(confidence, 0.0), 1.0),
    )


def _depth_levels(
    spans: dict[str, Span], children: dict[Optional[str], list[str]], root_id: str
) -> list[list[str]]:
    """Group span ids by depth, so the recursion can run bottom-up level by level.

    The paper runs the agents recursively with independent branches in parallel
    under a pool of K agents. Submitting the recursion itself into a bounded
    pool deadlocks -- a parent occupies a worker while waiting on its children --
    so the levels are walked deepest-first and parallelised *within* a level.
    Same evidence flow and same parallelism, no self-starvation.
    """
    levels, frontier, seen = [], [root_id], {root_id}
    while frontier:
        levels.append(frontier)
        nxt = []
        for span_id in frontier:
            for child in children.get(span_id, []):
                if child not in seen:  # cycle guard: malformed parent pointers
                    seen.add(child)
                    nxt.append(child)
        frontier = nxt
    return levels


def _run_agents(
    spans: dict[str, Span],
    children: dict[Optional[str], list[str]],
    root_id: str,
    metric: pd.DataFrame,
    logs: pd.DataFrame,
    inject_time: float,
    llm: Callable[[str, str], str],
    pool_capacity: int,
    delta: int,
    n_sigma: float,
) -> tuple[dict[str, SelfEvidence], dict[str, Consolidated]]:
    """Section IV-B: one Dedicated Agent per span, recursive and parallel."""
    levels = _depth_levels(spans, children, root_id)
    self_evidence: dict[str, SelfEvidence] = {}
    consolidated: dict[str, Consolidated] = {}

    with ThreadPoolExecutor(max_workers=max(1, pool_capacity)) as pool:
        for level in reversed(levels):  # leaves first, root last
            def diagnose(span_id: str) -> tuple[SelfEvidence, Consolidated]:
                span = spans[span_id]
                # t0 is the span's own start: the agent's context is span-local
                t0 = span.start_time if np.isfinite(span.start_time) else inject_time
                evidence = _self_state_verification(
                    span,
                    _metric_tool(metric, span.service, t0, delta, n_sigma),
                    _log_tool(logs, span.service, t0, delta),
                    llm,
                )
                downstream = [
                    consolidated[c] for c in children.get(span_id, []) if c in consolidated
                ]
                return evidence, _evidence_consolidation(evidence, downstream, llm)

            for span_id, (evidence, cons) in zip(level, pool.map(diagnose, level)):
                self_evidence[span_id] = evidence
                consolidated[span_id] = cons

    return self_evidence, consolidated


# ==========================================================================
# IV-C: Diagnosis Synthesizer
# ==========================================================================


def _evidence_graph_lines(
    spans: dict[str, Span], self_evidence: dict[str, SelfEvidence]
) -> list[str]:
    """Global Evidence Graph G_E = (V, E, Phi), Section IV-C2.

    Isomorphic to the trace topology, so it is serialized as its edges carrying
    Phi(v_s) = e_s. Only abnormal nodes are rendered: the graph exists to
    cross-validate the root report against distributed local observations, and
    the normal nodes are what context explosion is made of.
    """
    lines = []
    for span_id, evidence in self_evidence.items():
        if not evidence.abnormal:
            continue
        parent = spans[span_id].parent_id
        edge = f"{spans[parent].service} -> " if parent in spans else ""
        lines.append(f"{edge}{evidence.service} (span {span_id}): {evidence.symptoms} "
                     f"| hypothesis: {evidence.hypothesis}")
    return lines


def _synthesize(
    root_report: Consolidated,
    graph_lines: list[str],
    candidates: list[str],
    llm: Callable[[str, str], str],
    synthesizer: str,
) -> list[str]:
    """F_synth: (e^_root, G_E) -> R (Eq. 10), returning ranked service names.

    `synthesizer` selects the Section V-E ablations: "rldr" withholds the Global
    Evidence Graph, "geg" withholds the Root-Level Diagnosis Report.
    """
    report_block = (
        f"Root-Level Diagnosis Report:\n"
        f"  root cause: {root_report.root_cause}\n"
        f"  service: {root_report.service}\n"
        f"  confidence: {root_report.confidence:.2f}\n"
        f"  reason: {root_report.reason}\n"
    )
    graph_block = "Global Evidence Graph (abnormal spans):\n" + (
        "\n".join(f"  - {line}" for line in graph_lines) or "  (no abnormal spans)"
    )
    blocks = {
        "full": f"{report_block}\n{graph_block}",
        "rldr": report_block,
        "geg": graph_block,
    }[synthesizer]

    prompt = f"{blocks}\n\nCandidate services:\n" + "\n".join(f"  - {c}" for c in candidates)
    parsed = _parse_json(llm(_SYNTH_SYSTEM, prompt))
    ranking = parsed.get("ranking", [])
    if not isinstance(ranking, list):
        return []
    seen, ordered = set(), []
    for name in ranking:
        name = str(name)
        if name in candidates and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


# ==========================================================================
# Entrypoint
# ==========================================================================


def _load_sources(data: Any, kwargs: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resolve (metric, traces, logs) from whatever the harness handed over.

    `main.py` passes the metric DataFrame plus `args.data_path`; the multi-source
    notebook passes a dict already holding traces and logs. Both are accepted,
    so RCLAgent needs no loader changes in `main.py`.
    """
    if isinstance(data, dict):
        metric = data.get("metric")
        if metric is None:
            raise ValueError("rclagent needs a 'metric' DataFrame in the input dict")
        return metric, data.get("traces", pd.DataFrame()), data.get("logs", pd.DataFrame())

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"rclagent expects a DataFrame or dict, got {type(data).__name__}")

    args = kwargs.get("args")
    data_path = getattr(args, "data_path", None)
    if not data_path:
        raise ValueError(
            "rclagent needs traces: pass data as a dict with 'traces'/'logs', or "
            "let main.py supply args.data_path so traces.csv can be read"
        )
    case_dir = os.path.dirname(data_path)

    def read(name: str) -> pd.DataFrame:
        path = os.path.join(case_dir, name)
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    traces = read("traces.csv")
    if traces.empty:
        raise ValueError(
            f"rclagent needs traces.csv in {case_dir} (RE2/RE3 ship it; RE1 does not)"
        )
    return data, traces, read("logs.csv")


def rclagent(
    data: Any,
    inject_time: Optional[int] = None,
    dataset: Optional[str] = None,
    num_loop: Optional[int] = None,
    sli: Optional[str] = None,
    anomalies: Optional[list] = None,
    llm: Optional[Callable[[str, str], str]] = None,
    synthesizer: str = "full",
    **kwargs: Any,
) -> dict[str, list]:
    """Localize the root cause with RCLAgent (arXiv:2605.14866).

    Args:
        data: metric DataFrame (traces/logs then read from `args.data_path`'s
            directory), or a dict with "metric", "traces" and "logs".
        inject_time: fault injection timestamp, used to select the abnormal
            request and split the metric baseline.
        dataset: dataset name, forwarded to `preprocess`.
        llm: backbone as a callable (system, user) -> str. Defaults to
            Claude-3.5-Sonnet (Section V-A4); pass your own to swap backbones.
        synthesizer: one of SYNTHESIZERS. "full" is the paper's method;
            "rldr" and "geg" are the Section V-E ablations.

    Returns:
        {"node_names": [...], "ranks": [...]} with ranks as
        "<service>_<most deviant property>", most likely root cause first.
    """
    if synthesizer not in SYNTHESIZERS:
        raise ValueError(f"{synthesizer=} must be one of {SYNTHESIZERS}")
    if inject_time is None:
        raise ValueError("rclagent needs inject_time to select the abnormal request")

    metric, traces, logs = _load_sources(data, kwargs)
    llm = llm or _default_llm(kwargs.get("model", DEFAULT_MODEL))

    # candidate set: the components the harness scores against, after the same
    # preprocessing every other RCAEval method applies
    columns = preprocess(
        data=metric.copy(), dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    ).columns
    properties = [c for c in columns if c != "time"]
    candidates = sorted({c.split("_")[0] for c in properties})
    if not candidates:
        raise ValueError("rclagent found no candidate components in the metric data")

    trace = _select_trace(traces, inject_time)
    spans = _build_spans(trace, candidates)
    if not spans:
        raise ValueError("rclagent could not parse any span from the selected trace")
    children = _children_of(spans)
    roots = children.get(None, [])
    if not roots:
        raise ValueError("rclagent found no entry span in the selected trace")
    root_id = roots[0]

    self_evidence, consolidated = _run_agents(
        spans,
        children,
        root_id,
        metric,
        logs,
        float(inject_time),
        llm,
        int(kwargs.get("pool_capacity", POOL_CAPACITY)),
        int(kwargs.get("delta", DELTA_SECONDS)),
        float(kwargs.get("n_sigma", N_SIGMA)),
    )

    ranked_services = _synthesize(
        consolidated[root_id],
        _evidence_graph_lines(spans, self_evidence),
        candidates,
        llm,
        synthesizer,
    )

    # the candidate set must be fully covered for Avg@k, so anything the
    # synthesizer left out follows, ordered by its own metric deviation
    deviation = _deviation_by_service(metric, properties, float(inject_time))
    tail = sorted(
        (c for c in candidates if c not in ranked_services),
        key=lambda c: -deviation.get(c, 0.0),
    )

    ranks = [
        f"{service}_{_peak_property(metric, properties, service, float(inject_time))}"
        for service in ranked_services + tail
    ]

    if kwargs.get("verbose") is True:
        print(f"root report: {consolidated[root_id]}")
        for rank in ranks[:10]:
            print(f"  {rank}")

    return {"node_names": properties, "ranks": ranks}


def _service_deviations(
    metric: pd.DataFrame, properties: list[str], inject_time: float
) -> dict[str, float]:
    """Peak |z| of every property, against its pre-injection baseline."""
    normal = metric[metric["time"] < inject_time]
    anomal = metric[metric["time"] >= inject_time]
    scores = {}
    for column in properties:
        if column not in metric.columns or normal.empty or anomal.empty:
            continue
        mu, sigma = float(normal[column].mean()), float(normal[column].std())
        if not np.isfinite(sigma) or sigma == 0:
            scores[column] = 0.0
            continue
        scores[column] = float(((anomal[column] - mu).abs() / sigma).max())
    return scores


def _deviation_by_service(
    metric: pd.DataFrame, properties: list[str], inject_time: float
) -> dict[str, float]:
    scores = _service_deviations(metric, properties, inject_time)
    by_service: dict[str, float] = {}
    for column, score in scores.items():
        service = column.split("_")[0]
        by_service[service] = max(by_service.get(service, 0.0), score)
    return by_service


def _peak_property(
    metric: pd.DataFrame, properties: list[str], service: str, inject_time: float
) -> str:
    """The service's most deviant property, so ranks read "<service>_<property>"."""
    scores = _service_deviations(metric, properties, inject_time)
    owned = {c: s for c, s in scores.items() if c.split("_")[0] == service}
    if not owned:
        return "latency"  # gap 6: paper ranks services, harness keys on a property
    return max(owned, key=owned.get).split("_", 1)[1]


def rclagent_rldr(
    data: Any, inject_time: Optional[int] = None, dataset: Optional[str] = None, **kwargs: Any
) -> dict[str, list]:
    """Ablation (Section V-E): Root-Level Diagnosis Report without the graph."""
    kwargs.pop("synthesizer", None)
    return rclagent(data, inject_time=inject_time, dataset=dataset, synthesizer="rldr", **kwargs)


def rclagent_geg(
    data: Any, inject_time: Optional[int] = None, dataset: Optional[str] = None, **kwargs: Any
) -> dict[str, list]:
    """Ablation (Section V-E): Global Evidence Graph without the root report."""
    kwargs.pop("synthesizer", None)
    return rclagent(data, inject_time=inject_time, dataset=dataset, synthesizer="geg", **kwargs)


# ==========================================================================
# Runnable check -- no API key, no datasets
# ==========================================================================


def _scripted_llm() -> Callable[[str, str], str]:
    """A stub backbone that plays a competent SRE, so the wiring is testable.

    Answers from the prompt text alone: a span is abnormal when its own metrics
    fired, a parent defers to whichever child claimed the cause, and the
    synthesizer trusts the report's named root cause, then the deepest abnormal
    node of the graph. That is the minimum an LLM must do for the recursion to
    mean anything -- so the check fails if the evidence never reaches it.
    """

    def call(system: str, user: str) -> str:
        if system is _SELF_SYSTEM:
            abnormal = "(none)" not in user.split("Anomalous metrics")[1].split("Relevant logs")[0]
            return json.dumps(
                {"abnormal": abnormal, "symptoms": "metrics off baseline" if abnormal else "",
                 "hypothesis": "local resource saturation" if abnormal else ""}
            )
        if system is _CONS_SYSTEM:
            for line in user.splitlines():
                if "root_cause=" in line and "upstream" not in line:
                    child = line.split("root_cause=")[1].split(",")[0]
                    return json.dumps({"root_cause": child, "reason": "child owns it",
                                       "confidence": 0.8})
            mine = "self" if "Abnormal: True" in user else "upstream"
            service = user.splitlines()[0].split("(")[1].rstrip(")")
            return json.dumps(
                {"root_cause": service if mine == "self" else "upstream",
                 "reason": "own metrics anomalous" if mine == "self" else "no local anomaly",
                 "confidence": 0.9 if mine == "self" else 0.1}
            )
        head, tail = user.split("Candidate services:")
        candidates = [line.strip(" -") for line in tail.splitlines() if line.strip()]
        preferred = []
        reported = re.search(r"root cause: (\S+)", head)
        if reported and reported.group(1) in candidates:
            preferred.append(reported.group(1))
        # graph lines are serialized leaves first, so the deepest abnormal node
        # -- the one after the last arrow -- is the first one worth believing
        for line in head.splitlines():
            node = re.search(r"(?:-> )?([\w.-]+) \(span ", line)
            if node and node.group(1) in candidates and node.group(1) not in preferred:
                preferred.append(node.group(1))
        ranking = preferred + [c for c in candidates if c not in preferred]
        return json.dumps({"ranking": ranking, "reason": "evidence names it"})

    return call


def _synthetic_case(inject_time: int = 100) -> dict:
    """A two-hop trace where the deep service is faulty and its callers inherit."""
    times = np.arange(inject_time - 60, inject_time + 60, dtype=float)
    post = times >= inject_time
    metric = pd.DataFrame({"time": times})
    rng = np.random.default_rng(0)
    for service in ("frontend", "cartservice", "redis"):
        metric[f"{service}_latency"] = 100 + rng.normal(0, 1, times.size)
        metric[f"{service}_cpu"] = 10 + rng.normal(0, 1, times.size)
    metric.loc[post, "redis_cpu"] += 80  # the injected fault
    metric.loc[post, "cartservice_latency"] += 40  # propagated
    metric.loc[post, "frontend_latency"] += 45  # propagated further

    def span(span_id, parent, service, start, duration):
        return {"traceID": "t1" if start >= inject_time else "t0", "spanID": span_id,
                "parentSpanID": parent, "serviceName": service, "operationName": f"{service}.op",
                "startTime": start * 1e6, "duration": duration, "statusCode": 0.0}

    traces = pd.DataFrame([
        span("n1", None, "frontend", inject_time - 30, 1_000),
        span("n2", "n1", "cartservice", inject_time - 30, 500),
        span("n3", "n2", "redis", inject_time - 30, 200),
        span("a1", None, "frontend", inject_time + 10, 500_000),
        span("a2", "a1", "cartservice", inject_time + 10, 400_000),
        span("a3", "a2", "redis", inject_time + 10, 350_000),
    ])
    logs = pd.DataFrame({"time": [inject_time + 5], "serviceName": ["redis"],
                         "message": ["ERROR: command timed out"]})
    return {"metric": metric, "traces": traces, "logs": logs}


def _demo() -> None:
    """Every claim this implementation can check without an API key or datasets."""
    case = _synthetic_case()
    llm = _scripted_llm()

    # the trace under diagnosis is the abnormal request, not a normal one (Sec V-B)
    selected = _select_trace(case["traces"], 100)
    assert set(selected["traceID"]) == {"t1"}, selected["traceID"].unique()

    # Trace Tool: the span graph is recovered from parent pointers (Eq. 2)
    spans = _build_spans(selected, ["frontend", "cartservice", "redis"])
    children = _children_of(spans)
    assert children[None] == ["a1"], children
    assert children["a1"] == ["a2"] and children["a2"] == ["a3"], children

    # trace and metric spellings are reconciled, or the right answer would be
    # filtered out of the ranking entirely (gap 5)
    assert _match_service("frontendservice", ["frontend", "redis"]) == "frontend"
    assert _match_service("cart", ["cartservice", "redis"]) == "cartservice"
    assert _match_service("ts-station", ["ts-station-service", "ts-order"]) == "ts-station-service"
    assert _match_service("unrelated", ["frontend", "redis"]) == "unrelated"

    # the recursion walks leaves first, so children are consolidated before parents
    assert _depth_levels(spans, children, "a1") == [["a1"], ["a2"], ["a3"]]

    # Metric Tool: n-sigma fires on the faulty service, stays quiet on a healthy one
    assert _metric_tool(case["metric"], "redis", 110, DELTA_SECONDS, N_SIGMA)
    assert not _metric_tool(case["metric"], "redis", 40, DELTA_SECONDS, N_SIGMA)

    # Log Tool: phi(l) keeps the ERROR line, and only for the right component
    assert _log_tool(case["logs"], "redis", 100, DELTA_SECONDS) == ["ERROR: command timed out"]
    assert _log_tool(case["logs"], "frontend", 100, DELTA_SECONDS) == []

    # end to end: the root cause outranks the services that merely inherited the
    # latency, which is the paper's whole claim about recursion over the graph
    for variant in SYNTHESIZERS:
        out = rclagent(case, inject_time=100, dataset="demo", llm=llm, synthesizer=variant)
        assert out["ranks"][0].startswith("redis"), (variant, out["ranks"])
        # the candidate set stays fully covered, so Avg@k is well defined
        assert len(out["ranks"]) == 3, (variant, out["ranks"])
        assert all("_" in rank for rank in out["ranks"]), out["ranks"]

    # a metric-only call cannot invent traces: it must say so, not guess
    try:
        rclagent(case["metric"], inject_time=100, dataset="demo", llm=llm)
    except ValueError as exc:
        assert "traces" in str(exc)
    else:
        raise AssertionError("expected a ValueError without traces")

    print("rclagent demo ok:", rclagent(case, inject_time=100, dataset="demo", llm=llm)["ranks"])


if __name__ == "__main__":
    _demo()
