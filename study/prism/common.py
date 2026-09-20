"""Shared setup for the three PRISM experiments in this directory.

Everything the experiments need sits in ONE long table, one row per
(case, metric): `case_table(datasets)`. Every analysis below it is a groupby.

The table stores the *ingredients* of PRISM's score rather than the score
alone. With the default configuration (zscore + max time pooling, Def. 3.1)
a metric's score is

    S = max_t |x_t - c| / s

so keeping `dev_mean = max_t |x_t - mean|`, `dev_med = max_t |x_t - median|`
and every candidate scale (std, MAD, median |x|) lets experiment 2 re-scale
every score -- floors, MAD, caps, logs -- without reloading a byte.

Scoring, property classification and window handling are taken from
`RCAEval.e2e.prism` itself, and the windowing mirrors `main.py`, so the table
describes the same run the benchmark reports.

Ground truth is matched on the exact component name (main.py strips "-db"
from predictions but not from the answer, which makes its -db cases
unhittable; here both sides keep their name).
"""

import argparse
import os
from dataclasses import dataclass
from glob import glob
from os.path import basename, dirname, join
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from RCAEval.e2e.prism import _property_class, _split_property
from RCAEval.io.time_series import preprocess
from RCAEval.utility import (
    download_online_boutique_dataset,
    download_re2ob_dataset,
    download_re2ss_dataset,
    download_re2tt_dataset,
    download_re3_dataset,
    download_sock_shop_2_dataset,
    download_train_ticket_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"

# dataset name -> (download helper, directory relative to the repo root), as in main.py
DATASETS = {
    "re1-ob": (download_online_boutique_dataset, "data/online-boutique"),
    "re1-ss": (download_sock_shop_2_dataset, "data/sock-shop-2"),
    "re1-tt": (download_train_ticket_dataset, "data/train-ticket"),
    "re2-ob": (download_re2ob_dataset, "data/RE2/RE2-OB"),
    "re2-ss": (download_re2ss_dataset, "data/RE2/RE2-SS"),
    "re2-tt": (download_re2tt_dataset, "data/RE2/RE2-TT"),
    "re3-ob": (download_re3_dataset, "data/RE3/RE3-OB"),
    "re3-ss": (download_re3_dataset, "data/RE3/RE3-SS"),
    "re3-tt": (download_re3_dataset, "data/RE3/RE3-TT"),
}
ALL_DATASETS = list(DATASETS)

# the fault tokens main.py evaluates; a "<service>_<fault>" directory whose tail
# is not one of these is not a case directory and is skipped (loudly)
FAULTS = frozenset({"cpu", "mem", "disk", "socket", "delay", "loss"})

# M(S^I, S^E), Sec 3.3, plus the three replacements the study proposes
COMBINERS = {
    "additive": lambda i, e: i + e - np.log1p(i + e),  # Eq (3), PRISM's default
    "min": np.minimum,                                 # Eq (4), conjunctive
    "log": lambda i, e: np.log1p(i) + np.log1p(e),     # additive without the magnitude race
    "geometric": lambda i, e: np.sqrt(i * e),          # smooth min
    "internal": lambda i, e: i,                        # Table 6 ablations
    "external": lambda i, e: e,
}


@dataclass(frozen=True)
class Case:
    dataset: str
    case: str
    gt_service: str
    fault: str
    data: pd.DataFrame  # windowed, with "time"
    inject_time: int


def dataset_dir(name):
    """Directory of `name`, downloading it on first use (helpers write to ./data)."""
    if name not in DATASETS:
        raise ValueError(f"{name=} must be one of {ALL_DATASETS}")
    download, rel = DATASETS[name]
    path = ROOT / rel
    if not path.exists():
        os.chdir(ROOT)
        download()
    if not path.exists():
        raise FileNotFoundError(f"{name} not found at {path} after download")
    return path


def iter_cases(dataset, length=20, limit=None, root=None):
    """Yield every case of `dataset`, windowed exactly as main.py windows it."""
    base = Path(root) if root else dataset_dir(dataset)
    paths = sorted(glob(join(base, "**", "data.csv"), recursive=True))
    if not paths:
        paths = sorted(glob(join(base, "**", "simple_metrics.csv"), recursive=True))
    n = length * 60 // 2

    for path in paths[:limit]:
        service, _, fault = basename(dirname(dirname(path))).rpartition("_")
        if fault not in FAULTS:  # not a case directory; main.py would crash here
            print(f"{dataset}: skipping {path}, not a <service>_<fault> case")
            continue
        with open(join(dirname(path), "inject_time.txt")) as f:
            inject_time = int(f.readlines()[0].strip())

        data = pd.read_csv(path)
        data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
        data = data.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        data = pd.concat(
            [data[data["time"] < inject_time].tail(n), data[data["time"] >= inject_time].head(n)],
            ignore_index=True,
        )
        data = data.rename(
            columns={c: c.replace("_latency-90", "_latency") for c in data.columns}
        )
        yield Case(dataset, f"{dataset}/{service}_{fault}/{basename(dirname(path))}",
                   service, fault, data, inject_time)


def windows(case):
    """PRISM's own pre/post-fault frames: preprocess each side, keep the intersection."""
    normal = preprocess(case.data[case.data["time"] < case.inject_time],
                        dataset=case.dataset, dk_select_useful=False)
    anomal = preprocess(case.data[case.data["time"] >= case.inject_time],
                        dataset=case.dataset, dk_select_useful=False)
    cols = [c for c in normal.columns if c in anomal.columns]
    return normal[cols], anomal[cols]


def _case_rows(case):
    normal, anomal = windows(case)
    if normal.empty or anomal.empty or not len(normal.columns):
        return []

    pre, post = normal.to_numpy(dtype=float), anomal.to_numpy(dtype=float)
    mean, std = pre.mean(axis=0), pre.std(axis=0)
    med = np.median(pre, axis=0)
    mad = np.median(np.abs(pre - med), axis=0)
    absmed = np.median(np.abs(pre), axis=0)
    dev_mean = np.abs(post - mean).max(axis=0)
    dev_med = np.abs(post - med).max(axis=0)

    rows = []
    for j, column in enumerate(normal.columns):
        component, prop = _split_property(column)
        cls = _property_class(prop)
        if cls is None:  # PRISM cannot classify it, so it is no evidence either way
            continue
        rows.append({
            "dataset": case.dataset, "case": case.case, "fault": case.fault,
            "gt_service": case.gt_service, "component": component, "metric": column,
            "prop_type": prop.replace("-", "_").split("_")[0].lower(), "cls": cls,
            "score": dev_mean[j] / (std[j] if std[j] > 0 else 1.0),  # gap 5: s=0 -> 1.0
            "dev_mean": dev_mean[j], "dev_med": dev_med[j],
            "pre_mean": mean[j], "pre_std": std[j], "pre_med": med[j],
            "pre_mad": mad[j], "pre_absmed": absmed[j],
        })
    return rows


def case_table(datasets, length=20, limit=None, cache=True, root=None):
    """The one table: one row per (case, metric), cached under out/."""
    OUT.mkdir(exist_ok=True)
    path = OUT / f"table_{'-'.join(datasets)}_{length}_{limit or 'all'}.csv.gz"
    if cache and path.exists():
        return pd.read_csv(path)

    rows = []
    for dataset in datasets:
        for case in iter_cases(dataset, length=length, limit=limit, root=root):
            rows.extend(_case_rows(case))
        print(f"{dataset}: {len({r['case'] for r in rows})} cases, {len(rows)} rows so far")

    table = pd.DataFrame(rows)
    if cache:
        table.to_csv(path, index=False)
        print("wrote", path)
    return table


def component_scores(table, score_col="score"):
    """Pool metric scores into S^I and S^E per (case, component), Sec 3.2 (max pooling).

    gap 4: a component missing a class scores 0 there rather than being dropped.
    """
    key = ["dataset", "case", "fault", "gt_service", "component"]
    comp = table.pivot_table(index=key, columns="cls", values=score_col, aggfunc="max").reset_index()
    for cls in ("internal", "external"):
        if cls not in comp:
            comp[cls] = 0.0
    comp = comp.rename(columns={"internal": "SI", "external": "SE"})
    comp[["SI", "SE"]] = comp[["SI", "SE"]].fillna(0.0)
    comp["is_root"] = comp["component"] == comp["gt_service"]

    for cls, name in (("internal", "witness_I"), ("external", "witness_E")):
        sub = table[table["cls"] == cls]
        if sub.empty:
            comp[name] = None
            continue
        best = sub.loc[sub.groupby(key)[score_col].idxmax(), key + ["metric"]]
        comp = comp.merge(best.rename(columns={"metric": name}), on=key, how="left")
    return comp


def combine(comp, combiner):
    """Root cause score for every component. "rank" = per-case percentile of each class."""
    if combiner == "rank":
        return (comp.groupby("case")["SI"].rank(pct=True)
                + comp.groupby("case")["SE"].rank(pct=True))
    return pd.Series(COMBINERS[combiner](comp["SI"].to_numpy(), comp["SE"].to_numpy()),
                     index=comp.index)


def root_rank(comp, score):
    """Per case: the ground truth's rank (1 = top, pessimistic on ties) and Top-1 hit.

    A case whose root cause component never reaches the ranking (dropped as
    constant, or carrying no classifiable property) ranks last, i.e. a miss.
    """
    ranked = comp.assign(_score=np.asarray(score, dtype=float))
    ranked["_rank"] = ranked.groupby("case")["_score"].rank(ascending=False, method="max")
    per_case = ranked.groupby(["dataset", "case", "fault"], as_index=False).agg(n=("_score", "size"))
    root = ranked[ranked["is_root"]].groupby("case", as_index=False)["_rank"].min()
    out = per_case.merge(root.rename(columns={"_rank": "rank"}), on="case", how="left")
    out["rank"] = out["rank"].fillna(out["n"] + 1)
    out["hit1"] = out["rank"] == 1
    return out


def mcnemar(hit_a, hit_b):
    """Exact McNemar on paired Top-1 hit/miss -> (a-only wins, b-only wins, p)."""
    a, b = np.asarray(hit_a, dtype=bool), np.asarray(hit_b, dtype=bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = 1.0 if n10 + n01 == 0 else float(stats.binomtest(n10, n10 + n01, 0.5).pvalue)
    return n10, n01, p


def wilcoxon(rank_a, rank_b):
    """Wilcoxon signed-rank p-value on the root cause's rank under two variants."""
    a, b = np.asarray(rank_a, dtype=float), np.asarray(rank_b, dtype=float)
    if not np.any(a != b):
        return 1.0
    return float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)


def report(frame, name, floatfmt="%.3f"):
    """Print a frame and drop it next to the figures."""
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.csv"
    frame.to_csv(path, index=False, float_format=floatfmt)
    print(f"\n== {name} ==")
    print(frame.to_string(index=False))
    print("wrote", path)
    return frame


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("wrote", path)


def argparser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS,
                        help="datasets to run, downloaded on first use")
    parser.add_argument("--length", type=int, default=20, help="window length in minutes")
    parser.add_argument("--limit", type=int, default=None, help="cases per dataset (debug)")
    parser.add_argument("--no-cache", action="store_true", help="rebuild the table")
    parser.add_argument("--root", default=None,
                        help="read cases from this directory instead of the dataset (debug)")
    return parser


def _selftest():
    """One runnable check with no dataset: synthetic cases through the whole setup."""
    import tempfile

    from RCAEval.e2e.prism import _propagation_case

    with tempfile.TemporaryDirectory() as tmp:
        for amplification, service in ((1.7, "cartservice"), (6.0, "cartservice")):
            case_dir = Path(tmp) / f"{service}_cpu" / f"{amplification}"
            case_dir.mkdir(parents=True)
            _propagation_case(amplification).to_csv(case_dir / "data.csv", index=False)
            (case_dir / "inject_time.txt").write_text("60\n")

        table = case_table(["re1-ob"], length=20, cache=False, root=tmp)
        comp = component_scores(table)

        # the table reproduces PRISM's own score, and both classes are populated
        assert (table["score"] - table["dev_mean"] / table["pre_std"].where(table.pre_std > 0, 1.0)
                ).abs().max() < 1e-9
        assert comp["SI"].gt(0).all() and comp["SE"].gt(0).all()

        additive = root_rank(comp, combine(comp, "additive")).set_index("case")
        conjunctive = root_rank(comp, combine(comp, "min")).set_index("case")

        # bounded amplification (Prop 3.10): both scorers find the root cause;
        # arbitrary amplification: only the conjunctive one does (prism._demo)
        bounded, arbitrary = [c for c in additive.index if "1.7" in c][0], \
                             [c for c in additive.index if "6.0" in c][0]
        assert additive.loc[bounded, "hit1"] and conjunctive.loc[bounded, "hit1"]
        assert not additive.loc[arbitrary, "hit1"] and conjunctive.loc[arbitrary, "hit1"]

        hits = [additive.loc[bounded, "hit1"], additive.loc[arbitrary, "hit1"]]
        assert mcnemar(hits, [True, True])[2] <= 1.0
        assert 0.0 <= wilcoxon(additive["rank"], conjunctive["rank"]) <= 1.0

    print("common selftest ok")


if __name__ == "__main__":
    _selftest()
