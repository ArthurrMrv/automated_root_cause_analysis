"""Experiment 2 -- a floor on the scale estimate (plus the noise stress test).

Question: how much of PRISM's accuracy comes from dividing by a near-zero
pre-fault std?

    python study/prism/exp2_floors.py --datasets re2-ob --noise-limit 40

Outputs (study/prism/out/): exp2_scales, exp2_floors, exp2_floor_by_fault,
exp2_accuracy_vs_floor.png, exp2_noise, exp2_accuracy_vs_noise.png.
The floor sweep is recomputed from the cached table; only the noise stress
test reloads the data (bound it with --noise-limit).
"""

import numpy as np
import pandas as pd

from common import (OUT, argparser, case_table, combine, component_scores, iter_cases,
                    report, root_rank, save)
from RCAEval.e2e.baro import baro
from RCAEval.e2e.prism import _property_class, _split_property, prism

RELATIVE_FLOORS = [0.0, 0.01, 0.05, 0.10]
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.20]
CAP = 50.0


def rescale(table, scale, cap=None, log=False, dev="dev_mean"):
    """PRISM's metric score under a different scale estimate, cap or transform."""
    scores = table[dev].to_numpy() / np.where(scale > 0, scale, 1.0)
    if cap is not None:
        scores = np.minimum(scores, cap)
    return np.log1p(scores) if log else scores


def variants(table):
    """(name, floor strength, metric scores) for every scaling variant."""
    std, mad = table["pre_std"].to_numpy(), 1.4826 * table["pre_mad"].to_numpy()
    typical = table["pre_absmed"].to_numpy()
    # a per-type floor: the 10th percentile of std across metrics of that type
    per_type = table.groupby(["dataset", "prop_type"])["pre_std"].transform(
        lambda s: s.quantile(0.10)).to_numpy()

    for floor in RELATIVE_FLOORS:
        yield f"std+{floor:.0%}", floor, rescale(table, np.maximum(std, floor * typical))
        yield f"mad+{floor:.0%}", floor, rescale(
            table, np.maximum(mad, floor * typical), dev="dev_med")
    yield "std+p10-per-type", np.nan, rescale(table, np.maximum(std, per_type))
    yield f"std cap {CAP:.0f}", np.nan, rescale(table, std, cap=CAP)
    yield "std log1p", np.nan, rescale(table, std, log=True)


def sweep(table):
    """Top-1 of PRISM (additive) under every scaling variant, overall and per fault."""
    overall, by_fault = [], []
    for name, floor, scores in variants(table):
        comp = component_scores(table.assign(s=scores), score_col="s")
        ranks = root_rank(comp, combine(comp, "additive"))
        for dataset, sub in ranks.groupby("dataset"):
            overall.append({"dataset": dataset, "variant": name, "floor": floor,
                            "n": len(sub), "top1": sub["hit1"].mean(),
                            "avg_rank": sub["rank"].mean()})
        for (dataset, fault), sub in ranks.groupby(["dataset", "fault"]):
            by_fault.append({"dataset": dataset, "fault": fault, "variant": name,
                             "floor": floor, "n": len(sub), "top1": sub["hit1"].mean()})
    return pd.DataFrame(overall), pd.DataFrame(by_fault)


def plot_floors(by_fault):
    import matplotlib.pyplot as plt

    relative = by_fault[by_fault["variant"].str.startswith("std+") & by_fault["floor"].notna()]
    datasets = sorted(relative["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 3.4),
                             sharey=True, squeeze=False, layout="constrained")
    for ax, dataset in zip(axes[0], datasets):
        for fault, sub in relative[relative["dataset"] == dataset].groupby("fault"):
            sub = sub.sort_values("floor")
            ax.plot(sub["floor"] * 100, sub["top1"], marker="o", label=fault)
        ax.set_title(dataset)
        ax.set_xlabel("relative floor (% of median |x|)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Top-1")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle("PRISM Top-1 against floor strength (flat = robust, steep = scaling artifact)")
    save(fig, "exp2_accuracy_vs_floor.png")


def noisy(case, level, rng):
    """The case with Gaussian noise on its PRE-fault internal metrics.

    sigma = level x the metric's own typical magnitude (median |x| pre-fault),
    so a flat baseline becomes as noisy as a production one.
    """
    data = case.data.copy()
    pre = (data["time"] < case.inject_time).to_numpy()
    for column in data.columns:
        if column == "time" or _property_class(_split_property(column)[1]) != "internal":
            continue
        values = data.loc[pre, column].to_numpy(dtype=float)
        typical = np.median(np.abs(values)) or 1.0
        data.loc[pre, column] = values + rng.normal(0, level * typical, values.size)
    return data


def stress(datasets, length, limit, root=None):
    """Bonus: PRISM and BARO Top-1 against the level of injected baseline noise."""
    rows = []
    for dataset in datasets:
        for case in iter_cases(dataset, length=length, limit=limit, root=root):
            rng = np.random.default_rng(0)
            for level in NOISE_LEVELS:
                data = case.data if level == 0 else noisy(case, level, rng)
                for name, method in (("prism", prism), ("baro", baro)):
                    try:
                        ranks = method(data, inject_time=case.inject_time, dataset=dataset)["ranks"]
                        hit = bool(ranks) and ranks[0].split("_")[0] == case.gt_service
                    except Exception as error:  # a method that fails on a case scores a miss
                        print(f"{name} failed on {case.case} at noise {level}: {error}")
                        hit = False
                    rows.append({"dataset": dataset, "fault": case.fault, "case": case.case,
                                 "method": name, "noise": level, "hit1": hit})
    return pd.DataFrame(rows)


def plot_noise(noise):
    import matplotlib.pyplot as plt

    summary = noise.groupby(["dataset", "method", "noise"])["hit1"].mean().reset_index()
    datasets = sorted(summary["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 3.4),
                             sharey=True, squeeze=False, layout="constrained")
    for ax, dataset in zip(axes[0], datasets):
        for method, sub in summary[summary["dataset"] == dataset].groupby("method"):
            sub = sub.sort_values("noise")
            ax.plot(sub["noise"] * 100, sub["hit1"], marker="o", label=method)
        ax.set_title(dataset)
        ax.set_xlabel("noise on pre-fault internal metrics (%)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Top-1")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Accuracy against injected baseline noise")
    save(fig, "exp2_accuracy_vs_noise.png")
    return summary


def main():
    parser = argparser(__doc__)
    parser.add_argument("--noise-limit", type=int, default=40,
                        help="cases per dataset for the noise stress test, 0 to skip")
    args = parser.parse_args()

    table = case_table(args.datasets, length=args.length, limit=args.limit,
                       cache=not args.no_cache, root=args.root)

    # 1. the scales themselves: which metric types are nearly constant pre-fault?
    table = table.assign(
        degenerate=table["pre_std"] <= 1e-6 * np.maximum(table["pre_absmed"], 1e-12))
    report(table.groupby(["dataset", "prop_type"]).agg(
        n=("pre_std", "size"), std_p10=("pre_std", lambda s: s.quantile(0.10)),
        std_median=("pre_std", "median"), typical_median=("pre_absmed", "median"),
        frac_degenerate=("degenerate", "mean"), max_score=("score", "max"),
    ).reset_index(), "exp2_scales")

    # 2-3. floors, MAD, caps and logs, then Top-1 against floor strength
    overall, by_fault = sweep(table)
    report(overall, "exp2_floors")
    report(by_fault, "exp2_floor_by_fault")
    plot_floors(by_fault)

    # bonus: the production stress test
    if args.noise_limit:
        noise = stress(args.datasets, args.length, args.noise_limit, root=args.root)
        noise.to_csv(OUT / "exp2_noise_raw.csv", index=False)
        report(plot_noise(noise), "exp2_noise")


if __name__ == "__main__":
    main()
