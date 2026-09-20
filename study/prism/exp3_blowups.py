"""Experiment 3 -- how often internal z-scores blow up.

Question: is the paper's "222M mean" S^I a handful of outliers, or the
typical case? And when the axiom fails, which metrics fail it?

    python study/prism/exp3_blowups.py --datasets re2-ob re2-ss

Outputs (study/prism/out/): exp3_distribution, exp3_blowup_drivers,
exp3_conditional_top1, exp3_axiom_violations, exp3_SI_histogram.png.
"""

import numpy as np
import pandas as pd

from common import (argparser, case_table, combine, component_scores, report, root_rank,
                    save)

BLOWUP = 1e3  # a root cause S^I above this is called a blow-up
EXTREME = 1e6


def main():
    args = argparser(__doc__).parse_args()
    table = case_table(args.datasets, length=args.length, limit=args.limit,
                       cache=not args.no_cache, root=args.root)
    comp = component_scores(table)
    comp["role"] = np.where(comp["is_root"], "root cause", "affected")

    # 1. the distributions of S^I, root cause versus affected component
    report(comp.groupby(["dataset", "fault", "role"]).agg(
        n=("SI", "size"), median_SI=("SI", "median"), mean_SI=("SI", "mean"),
        max_SI=("SI", "max"),
        frac_above_1e3=("SI", lambda s: float((s > BLOWUP).mean())),
        frac_above_1e6=("SI", lambda s: float((s > EXTREME).mean())),
    ).reset_index(), "exp3_distribution")
    plot_histogram(comp)

    # 2. which metrics drive the blow-ups: a meaningful deviation, or a 0 ->
    #    nonzero jump on an idle metric (near-zero pre-fault mean and std)?
    internal = table[table["cls"] == "internal"].copy()
    internal["blowup"] = internal["score"] > BLOWUP
    internal["idle"] = internal["pre_absmed"] <= 1e-9
    internal["flat"] = internal["pre_std"] <= 1e-6 * np.maximum(internal["pre_absmed"], 1e-12)
    internal["is_root"] = internal["component"] == internal["gt_service"]
    drivers = internal[internal["blowup"]]
    report(drivers.groupby(["dataset", "prop_type"]).agg(
        n_blowups=("score", "size"), frac_on_root=("is_root", "mean"),
        frac_idle_metric=("idle", "mean"), frac_flat_baseline=("flat", "mean"),
        median_dev=("dev_mean", "median"), median_pre_std=("pre_std", "median"),
        median_score=("score", "median"),
    ).reset_index(), "exp3_blowup_drivers")

    # 3. Top-1 with and without a blow-up on the root cause
    ranks = root_rank(comp, combine(comp, "additive"))
    root_si = comp[comp["is_root"]].groupby("case")["SI"].max()
    ranks["root_SI"] = ranks["case"].map(root_si).fillna(0.0)
    ranks["blowup"] = ranks["root_SI"] > BLOWUP
    report(ranks.groupby(["dataset", "blowup"]).agg(
        n=("case", "size"), top1=("hit1", "mean"), avg_rank=("rank", "mean"),
        median_root_SI=("root_SI", "median"),
    ).reset_index(), "exp3_conditional_top1")
    report(ranks.groupby(["dataset", "fault", "blowup"]).agg(
        n=("case", "size"), top1=("hit1", "mean"),
    ).reset_index(), "exp3_conditional_top1_by_fault")

    # 4. Axiom 2.6: how often does an affected component out-score the root
    #    cause internally, and on which metric?
    report(violations(comp), "exp3_axiom_violations")

    print("\nRead: if Top-1 is high only on the blowup=True rows, PRISM works when the "
          "fault produces an extreme internal spike and not otherwise. The violation "
          "table says which metric types break Axiom 2.6.")


def violations(comp):
    """Cases where an affected component's S^I exceeds the root cause's."""
    root_si = comp[comp["is_root"]].groupby("case")["SI"].max()
    affected = comp[~comp["is_root"]].assign(root_SI=lambda d: d["case"].map(root_si).fillna(0.0))
    worst = affected.loc[affected.groupby("case")["SI"].idxmax()]
    worst = worst.assign(violates=worst["SI"] > worst["root_SI"])
    overall = worst.groupby(["dataset", "fault"]).agg(
        n=("case", "size"), frac_violating=("violates", "mean")).reset_index()
    # the metric behind the violating component's internal score
    culprit = worst[worst["violates"]].copy()
    culprit["culprit_type"] = culprit["witness_I"].fillna("none").map(
        lambda m: str(m).split("_", 1)[-1].replace("-", "_").split("_")[0])
    share = (culprit.groupby(["dataset", "fault", "culprit_type"])["case"].count()
             / culprit.groupby(["dataset", "fault"])["case"].count()).rename("share")
    return overall.merge(share.reset_index(), on=["dataset", "fault"], how="left")


def plot_histogram(comp):
    import matplotlib.pyplot as plt

    faults = sorted(comp["fault"].unique())
    fig, axes = plt.subplots(1, len(faults), figsize=(3.2 * len(faults), 3.2),
                             sharey=True, squeeze=False, layout="constrained")
    bins = np.linspace(-2, 9, 45)
    for ax, fault in zip(axes[0], faults):
        sub = comp[comp["fault"] == fault]
        for role, color in (("root cause", "#d1495b"), ("affected", "#8d99ae")):
            values = np.log10(np.clip(sub.loc[sub["role"] == role, "SI"], 1e-2, None))
            ax.hist(values, bins=bins, alpha=0.6, label=role, color=color, density=True)
        ax.axvline(np.log10(BLOWUP), ls="--", lw=1, color="k")
        ax.set_title(fault)
        ax.set_xlabel("log10 S^I")
    axes[0][0].set_ylabel("density")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Internal score distribution (dashed line = blow-up threshold 10^3)")
    save(fig, "exp3_SI_histogram.png")


if __name__ == "__main__":
    main()
