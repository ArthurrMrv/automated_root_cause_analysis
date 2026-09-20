"""Experiment 1 -- min versus additive scoring.

Question: does the additive score (Eq. 3) win because of the boundedness
argument, or because root cause internal scores are simply huge?

    python study/prism/exp1_combiners.py --datasets re2-ob re2-ss

Outputs (study/prism/out/): exp1_buckets, exp1_bucket_evidence,
exp1_magnitude_race, exp1_emailservice, exp1_combiners.
RE3 carries 30 cases per system; treat its rows as illustrative.
"""

import numpy as np
import pandas as pd

from common import (COMBINERS, argparser, case_table, combine, component_scores, mcnemar,
                    report, root_rank, wilcoxon)

VARIANTS = ["additive", "min", "log", "geometric", "rank", "internal", "external"]


def case_features(comp):
    """Per case: the root's two scores and the largest scores among affected components."""
    pct = comp.assign(
        pct_SI=comp.groupby("case")["SI"].rank(pct=True),
        pct_SE=comp.groupby("case")["SE"].rank(pct=True),
    )
    root = pct[pct["is_root"]].groupby("case").agg(
        root_SI=("SI", "max"), root_SE=("SE", "max"),
        root_pct_SI=("pct_SI", "max"), root_pct_SE=("pct_SE", "max"),
    )
    affected = comp[~comp["is_root"]].groupby("case").agg(
        aff_SE_max=("SE", "max"), aff_SI_max=("SI", "max"),
    )
    features = root.join(affected, how="outer").fillna(0.0)
    # >1 means an affected component out-shouts the root cause on that class
    features["amplification"] = features["aff_SE_max"] / features["root_SE"].replace(0, np.nan)
    features["root_SI_beats_aff_SE"] = features["root_SI"] > features["aff_SE_max"]
    return features.reset_index()


def bucketize(comp):
    """Classify every case: both correct, only additive, only min, neither."""
    additive = root_rank(comp, combine(comp, "additive"))
    conjunctive = root_rank(comp, combine(comp, "min"))[["case", "rank", "hit1"]]
    out = additive.rename(columns={"rank": "rank_add", "hit1": "hit_add"}).merge(
        conjunctive.rename(columns={"rank": "rank_min", "hit1": "hit_min"}), on="case")
    out["bucket"] = np.select(
        [out["hit_add"] & out["hit_min"], out["hit_add"] & ~out["hit_min"],
         ~out["hit_add"] & out["hit_min"]],
        ["both", "only additive", "only min"], default="neither")
    return out


def main():
    args = argparser(__doc__).parse_args()
    table = case_table(args.datasets, length=args.length, limit=args.limit,
                       cache=not args.no_cache, root=args.root)
    comp = component_scores(table)
    cases = bucketize(comp).merge(case_features(comp), on="case")

    # 1. the disagreement buckets
    report(cases.pivot_table(index="dataset", columns="bucket", values="case",
                             aggfunc="count", fill_value=0).reset_index(), "exp1_buckets")

    # 2. what explains each bucket: a weak root external score (the paper's own
    #    explanation for min losing) versus an amplified affected component
    report(cases.groupby("bucket").agg(
        n=("case", "size"),
        root_SE=("root_SE", "median"), root_pct_SE=("root_pct_SE", "median"),
        root_SI=("root_SI", "median"), root_pct_SI=("root_pct_SI", "median"),
        aff_SE_max=("aff_SE_max", "median"), amplification=("amplification", "median"),
    ).reset_index(), "exp1_bucket_evidence")

    # 3. the key test: where additive is correct, is the root's S^I simply bigger
    #    than any affected component's S^E? If so the win is a magnitude race, not
    #    the "affected components lack internal evidence" argument of Axiom 2.6.
    won = cases[cases["hit_add"]]
    report(won.groupby("dataset").agg(
        n_additive_correct=("case", "size"),
        frac_root_SI_beats_aff_SE=("root_SI_beats_aff_SE", "mean"),
        median_root_SI=("root_SI", "median"), median_aff_SE_max=("aff_SE_max", "median"),
    ).assign(median_ratio=lambda d: d["median_root_SI"] / d["median_aff_SE_max"]).reset_index(),
        "exp1_magnitude_race")

    # 4. Table 4 recomputed: does emailservice really score low when it is only
    #    an affected component?
    scored = comp.assign(additive=combine(comp, "additive"))
    scored["rank"] = scored.groupby("case")["additive"].rank(ascending=False, method="max")
    email = scored[(scored["component"] == "emailservice") & (~scored["is_root"])]
    if email.empty:
        print("\n== exp1_emailservice ==\nno emailservice rows in", args.datasets)
    else:
        report(email.groupby(["dataset", "fault"]).agg(
            n=("case", "size"), SI=("SI", "median"), SE=("SE", "median"),
            additive=("additive", "median"), rank=("rank", "median"),
            frac_top1=("rank", lambda r: float((r == 1).mean())),
        ).reset_index(), "exp1_emailservice")

    # 5. better combiners, each against the additive default on the same cases
    rows = []
    for dataset, sub in comp.groupby("dataset"):
        base = root_rank(sub, combine(sub, "additive")).set_index("case")
        for variant in VARIANTS:
            ranks = root_rank(sub, combine(sub, variant)).set_index("case").loc[base.index]
            wins, losses, p = mcnemar(ranks["hit1"], base["hit1"])
            rows.append({
                "dataset": dataset, "variant": variant, "n": len(ranks),
                "top1": ranks["hit1"].mean(), "avg_rank": ranks["rank"].mean(),
                "median_rank": ranks["rank"].median(),
                "wins_vs_additive": wins, "losses_vs_additive": losses,
                "mcnemar_p": p, "wilcoxon_p": wilcoxon(ranks["rank"], base["rank"]),
            })
    report(pd.DataFrame(rows), "exp1_combiners")

    print("\nRead: if 'log' keeps most of additive's Top-1, the internal/external idea is "
          "sound and Eq. (3) merely happened to work. If it collapses while "
          "frac_root_SI_beats_aff_SE is high, the additive score was carried by magnitudes.")


if __name__ == "__main__":
    main()
