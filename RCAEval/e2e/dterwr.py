"""PRISM + MARS + TG-FI + DTE-RWR: Directed Telemetry Edge-Weight Random Walk
(report.md, Add-On 4).

LSTR needs distributed tracing. Plenty of environments do not have it, and RE1
does not ship it. DTE-RWR recovers the same directional information from the
metrics alone: within PRISM's Top-5 shortlist it measures time-lagged
cross-correlation between the candidates' external latency series, builds a 5x5
transition matrix W whose edges encode "C_i moves before C_j", and runs a
personalized random walk with restart seeded by PRISM's internal scores:

    r = (1 - d) p + d W r,   p proportional to S^I

Because propagation takes time, a parallel latency spike that merely coincides
with the incident carries no lead-lag relationship and no edge, which is how the
walk filters out the non-causal look-alikes that defeat purely statistical
ranking on wide systems.

Fail-safe: when no significant lead-lag survives, W is the identity and the walk
is skipped entirely, returning the statistical ranking unchanged.

Values the report leaves unspecified are marked `# gap N:`.
"""

import warnings
from typing import Any, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from RCAEval.e2e.mars import TOP_K, ComponentTable, _emit, _order, _validate
from RCAEval.e2e.prism import _property_class, _split_property
from RCAEval.e2e.tgfi import LAMBDA, _metric, _scored

# The report writes the correlation window as [t_incident - 30s, t_incident].
# That presumes t_incident is a *detection* time, which lags onset, so the window
# lands on the propagation. RCAEval's inject_time is the exact injection instant
# (main.py reads it from inject_time.txt, and --tdelta exists precisely to
# simulate detection lag), so the literal window would hold ~15 pre-fault samples
# at 2s sampling -- no propagation to find, and short enough that a chance
# correlation can still cross the threshold and re-rank on noise. A symmetric
# window carries the propagation whether t is onset or a late alert. Pass
# window=(-30, 0) to reproduce the report literally.
WINDOW = (-30, 30)

# gap 1: none of the walk's constants are given. d = 0.85 is PageRank's
# conventional damping; max_lag = 5 samples is 10s at RCAEval's 2s sampling,
# comfortably longer than intra-cluster propagation; min_rho = 0.5 keeps only
# lead-lag strong enough to be directional rather than coincidental; 8 samples
# is the least that makes a correlation meaningful.
DAMPING = 0.85
MAX_LAG = 5
MIN_RHO = 0.5
MIN_SAMPLES = 8

# "external latency metrics of the Top-5 candidates" -- the external class also
# holds error and rate properties, which do not carry propagation delay
LATENCY_ROOTS = frozenset({"latency", "lat", "duration", "rt"})


def _latency_column(component: str, scores: pd.Series) -> Optional[str]:
    """The component's external latency property, most deviant one first."""
    owned = []
    for column, score in scores.items():
        name, prop = _split_property(column)
        if name != component or _property_class(prop) != "external":
            continue
        root = prop.replace("-", "_").split("_")[0].lower()
        owned.append((root in LATENCY_ROOTS, float(score), column))
    if not owned:
        return None
    return max(owned)[2]


def _window_frame(
    metric: pd.DataFrame, inject_time: Optional[int], window: tuple[int, int]
) -> Optional[pd.DataFrame]:
    """The incident window, or None when it cannot support a correlation."""
    if inject_time is None or "time" not in metric.columns:
        return None
    frame = metric[
        (metric["time"] >= inject_time + window[0]) & (metric["time"] <= inject_time + window[1])
    ]
    return frame if len(frame) >= MIN_SAMPLES else None


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, 0 when it is undefined rather than NaN."""
    if a.size < 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    if a.std() == 0 or b.std() == 0:
        return 0.0
    rho = float(np.corrcoef(a, b)[0, 1])
    return 0.0 if not np.isfinite(rho) else rho


def _lead_lag(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[int, float]:
    """(tau, rho) maximizing the cross-correlation; tau > 0 means x precedes y.

    Taking the argmax over *signed* lags is what makes this a direction test
    rather than a similarity test. Two series that move together score their
    highest correlation at tau = 0 -- the parallel spike the report wants
    filtered out -- while a genuine propagation path peaks at the delay it takes
    to propagate. Antisymmetric by construction: if x leads y, then y does not
    lead x.
    """
    best_lag, best_rho = 0, _corr(x, y)
    for lag in range(1, max_lag + 1):
        if lag >= x.size:
            break
        forward = _corr(x[:-lag], y[lag:])  # x at t against y at t + lag
        if forward > best_rho:
            best_lag, best_rho = lag, forward
        backward = _corr(x[lag:], y[:-lag])  # y leads x by the same amount
        if backward > best_rho:
            best_lag, best_rho = -lag, backward
    return best_lag, best_rho


def _lead_lag_matrix(
    series: dict[str, np.ndarray],
    components: list[str],
    max_lag: int = MAX_LAG,
    min_rho: float = MIN_RHO,
) -> np.ndarray:
    """The candidate transition matrix W, column-normalized.

    W[i][j] is the strength with which C_i precedes C_j, so a walk step moves
    mass from the later component to the earlier one -- from effect toward cause,
    which is what makes S^I the right personalization vector.
    """
    # gap 2: the report says "cross-correlations among the Top-5 components"
    # without saying on what. Correlating the raw latency series would be a
    # trend test, not a lag test: two services both climbing through an incident
    # correlate near 1.0 at every shift, and the argmax lands on noise. First
    # differences are stationary, and their peak sits exactly on the propagation
    # delay -- a step in a service's latency shows up as one spike at its onset.
    differenced = {name: np.diff(values) for name, values in series.items()}

    size = len(components)
    weights = np.zeros((size, size))
    for i, cause in enumerate(components):
        for j, effect in enumerate(components):
            if i == j:
                continue
            lag, rho = _lead_lag(differenced[cause], differenced[effect], max_lag)
            if lag > 0 and rho >= min_rho:
                weights[i, j] = rho

    totals = weights.sum(axis=0)
    np.divide(weights, totals, out=weights, where=totals > 0)
    return weights


def _rwr(
    weights: np.ndarray,
    personalization: np.ndarray,
    damping: float = DAMPING,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> np.ndarray:
    """Personalized PageRank r = (1 - d) p + d W r, by power iteration.

    r is renormalized each step so components with no outgoing edge (a column of
    zeros) leak no mass out of the walk.
    """
    rank = personalization.copy()
    for _ in range(max_iter):
        nxt = (1 - damping) * personalization + damping * (weights @ rank)
        total = nxt.sum()
        if total > 0:
            nxt = nxt / total
        if np.abs(nxt - rank).max() < tol:
            return nxt
        rank = nxt
    return rank


def _personalization(shortlist: list[str], table: ComponentTable) -> np.ndarray:
    """p from PRISM's internal anomaly scores; uniform when all of them are 0."""
    vector = np.array([table[component][0] for component in shortlist], dtype=float)
    total = vector.sum()
    return vector / total if total > 0 else np.full(len(shortlist), 1.0 / len(shortlist))


def _walk_order(
    shortlist: list[str],
    table: ComponentTable,
    scores: pd.Series,
    frame: Optional[pd.DataFrame],
    max_lag: int,
    min_rho: float,
) -> list[str]:
    """Re-rank the shortlist by the walk, or return it untouched (fail-safe)."""
    if frame is None or len(shortlist) < 2:
        return list(shortlist)

    columns = {component: _latency_column(component, scores) for component in shortlist}
    # gap 2: the report assumes every candidate carries an external latency
    # series. A shortlist where one does not cannot be correlated end to end, so
    # the add-on stands down rather than re-ranking on a partial subgraph.
    if any(column is None or column not in frame.columns for column in columns.values()):
        return list(shortlist)

    series = {c: frame[column].to_numpy(dtype=float) for c, column in columns.items()}
    weights = _lead_lag_matrix(series, shortlist, max_lag, min_rho)
    if not weights.any():
        return list(shortlist)  # W defaults to the identity: PRISM's ranking stands

    rank = _rwr(weights, _personalization(shortlist, table))
    # stable: ties keep the statistical order they came in with
    return [shortlist[i] for i in np.argsort(-rank, kind="stable")]


def dterwr(
    data: Any,
    inject_time: Optional[int] = None,
    dataset: Optional[str] = None,
    num_loop: Optional[int] = None,
    sli: Optional[str] = None,
    anomalies: Optional[list] = None,
    pooling: str = "max",
    combine: str = "additive",
    lam: float = LAMBDA,
    window: tuple[int, int] = WINDOW,
    max_lag: int = MAX_LAG,
    min_rho: float = MIN_RHO,
    **kwargs: Any,
) -> dict[str, list]:
    """Rank root cause candidates with PRISM + MARS + TG-FI + DTE-RWR (Add-On 4).

    Metric-only: no traces are read, by design.

    Args:
        data: metric DataFrame, or a dict with "metric".
        inject_time: fault injection timestamp, centering the correlation window.
        dataset: dataset name, forwarded to `preprocess`.
        window: correlation window in seconds relative to `inject_time`.
        max_lag: largest lead-lag shift, in samples.
        min_rho: correlation below which an edge is not directional evidence.

    Returns:
        {"node_names": [...], "ranks": ["<component>_<witness property>", ...]}.
    """
    _validate(pooling, combine, kwargs.get("time_agg", "max"))
    table, unclassified, scores, intersects = _scored(
        data, inject_time, dataset, anomalies, pooling, lam, kwargs
    )
    order = _order(table, combine)
    shortlist = order[:TOP_K]

    frame = _window_frame(_metric(data), inject_time, window)
    reordered = _walk_order(shortlist, table, scores, frame, max_lag, min_rho)
    ranks = _emit(reordered + order[TOP_K:], table, unclassified, scores)

    if kwargs.get("verbose") is True:
        for component in reordered:
            internal, external, witness = table[component]
            print(f"{witness}: S^I={internal:.2f} S^E={external:.2f}")

    return {"node_names": intersects, "ranks": ranks}


# ==========================================================================
# Runnable check -- no datasets
# ==========================================================================


def _lagged_case(inject_time: int = 1000, lag: int = 2, correlated: bool = True) -> pd.DataFrame:
    """A fault in `redis` propagating up the chain, one lag step at a time.

    Sampled every 2s like RCAEval. `redis` ramps first, `cartservice` two samples
    later, `frontend` two after that and amplified -- so the statistical stage
    ranks the amplified caller first and only the lead-lag direction says who
    moved before whom. With `correlated=False` the three ramps start together,
    the parallel-spike case the walk is supposed to reject.
    """
    times = np.arange(inject_time - 240, inject_time + 240, 2, dtype=float)
    rng = np.random.default_rng(3)
    metric = pd.DataFrame({"time": times})
    for service in ("redis", "cartservice", "frontend"):
        for prop in ("cpu", "mem", "latency", "error"):
            metric[f"{service}_{prop}"] = rng.normal(10, 0.2, times.size)

    def step(delay: int, amplitude: float) -> np.ndarray:
        onset = inject_time + (delay if correlated else 0) * 2
        return np.clip((times - onset) / 4.0, 0.0, 1.0) * amplitude

    metric["redis_cpu"] += step(0, 4.0)  # the injected fault: a modest interior
    metric["redis_latency"] += step(0, 4.0)
    metric["cartservice_latency"] += step(lag, 8.0)
    metric["frontend_latency"] += step(2 * lag, 16.0)  # amplified victim
    return metric


def _demo() -> None:
    """The add-on's claims, as a runnable check, each under its stated condition."""
    from RCAEval.e2e.tgfi import tgfi

    case = _lagged_case()

    # == The lead-lag matrix recovers the true propagation direction ==
    frame = _window_frame(case, 1000, WINDOW)
    assert frame is not None and len(frame) >= MIN_SAMPLES, frame
    order = ["redis", "cartservice", "frontend"]
    series = {c: frame[f"{c}_latency"].to_numpy(dtype=float) for c in order}
    weights = _lead_lag_matrix(series, order)
    assert weights[0, 1] > 0 and weights[1, 0] == 0, weights  # redis leads cart
    assert weights[1, 2] > 0 and weights[2, 1] == 0, weights  # cart leads frontend

    # == End to end: the walk moves the root cause past the amplified victim ==
    assert tgfi(case, inject_time=1000, dataset="demo")["ranks"][0].startswith("frontend")
    assert dterwr(case, inject_time=1000, dataset="demo")["ranks"][0].startswith("redis")

    # == Fail-safe: simultaneous spikes are not a propagation path ==
    parallel = _lagged_case(correlated=False)
    assert (
        dterwr(parallel, inject_time=1000, dataset="demo")["ranks"]
        == tgfi(parallel, inject_time=1000, dataset="demo")["ranks"]
    )

    # == Fail-safe: a window too short to correlate leaves the ranking alone ==
    assert _window_frame(case, 1000, (-4, 0)) is None
    assert (
        dterwr(case, inject_time=1000, dataset="demo", window=(-4, 0))["ranks"]
        == tgfi(case, inject_time=1000, dataset="demo")["ranks"]
    )

    # == Why the default window is symmetric: once t is the true onset, the
    # report's literal [t-30s, t] holds only pre-fault noise, and whatever it
    # correlates there is not the propagation chain ==
    pre_fault = _window_frame(case, 1000, (-30, 0))
    pre_series = {c: pre_fault[f"{c}_latency"].to_numpy(dtype=float) for c in order}
    pre_weights = _lead_lag_matrix(pre_series, order)
    assert not (pre_weights[0, 1] > 0 and pre_weights[1, 2] > 0), pre_weights

    # == The walk itself: mass ends up on the node the chain leads back to ==
    rank = _rwr(_lead_lag_matrix(series, order), np.array([0.34, 0.33, 0.33]))
    assert rank.argmax() == 0, rank

    print("dterwr demo ok:", dterwr(case, inject_time=1000, dataset="demo")["ranks"])


if __name__ == "__main__":
    _demo()
