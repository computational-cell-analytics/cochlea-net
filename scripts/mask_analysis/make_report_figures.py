"""Make the figures and tables of the mask analysis report from the evaluation outputs."""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mask_common import OUTPUT_ROOT, output_path  # noqa: E402

# Reference palette of the dataviz guidance (light mode): categorical slots 1-3 and text inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SPLIT_COLOR = {"train": SERIES[0], "val": SERIES[1]}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
})


def fig_audit(summary, path):
    summary = summary[summary.split != "failure"].copy()
    kinds = [("halo_cut", "halo cut (neighbour block predicted)"),
             ("mask_cut_candidate", "mask-cut candidate (neighbour block empty)"),
             ("seam_split", "seam split (partner across the face)")]
    for kind, _ in kinds:
        summary[kind] = 100 * summary[f"objects_{kind}"] / summary.n_relevant
    groups = summary.groupby(["class", "reference_rule"])
    rows = [(cls, rule, len(g), *[g[k].median() for k, _ in kinds]) for (cls, rule), g in groups if len(g) >= 2]
    rows.sort(key=lambda r: (r[0], -sum(r[3:])))
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(rows) + 1.4))
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    colors = [SERIES[0], SERIES[1], SERIES[2]]
    for i, (_, label) in enumerate(kinds):
        values = np.array([r[3 + i] for r in rows])
        ax.barh(y, values, left=left, height=0.62, color=colors[i], label=label, edgecolor=SURFACE, linewidth=1.5)
        left += values
    ax.set_yticks(y, [f"{r[0]}  {r[1]}  (n={r[2]})" for r in rows])
    chance = 100 * (summary.expected_face_hits / summary.n_relevant).median()
    ax.axvline(chance, color=INK_2, linestyle="--", linewidth=1)
    ax.annotate(f"chance ~{chance:.1f}%", (chance, len(rows) - 0.45), xytext=(4, 0), textcoords="offset points",
                color=INK_2, fontsize=8, va="bottom")
    ax.set_xlabel("relevant objects ending on a 128-voxel block face (% of objects, median over datasets)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", visible=False)
    fig.suptitle("Objects cut at block faces in existing segmentations, by the mask rule of the reference run",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_critical(critical, scores, mapping, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, cls in zip(axes, ("SGN", "IHC")):
        data = critical[(critical.score == scores[cls]) & (critical.mapping == mapping) & (critical["class"] == cls)]
        strata = data.groupby("stratum").critical_theta.median().sort_values().index.tolist()
        for i, stratum in enumerate(strata):
            for split in ("train", "val"):
                v = data[(data.stratum == stratum) & (data.split == split)].critical_theta
                jitter = -0.12 if split == "train" else 0.12
                ax.scatter(np.full(len(v), i + jitter), v, s=26, color=SPLIT_COLOR[split], edgecolor=SURFACE,
                           linewidth=0.8, label=split if i == 0 else None, zorder=3)
        rule = data[data.split == "train"].critical_theta.min()
        ax.axhline(rule, color=INK_2, linestyle="--", linewidth=1)
        ax.annotate(f"rule = min over train: {rule:.3g}", (len(strata) - 0.5, rule), ha="right", va="top",
                    xytext=(0, -3), textcoords="offset points", color=INK_2, fontsize=8)
        ax.set_ylabel(f"critical threshold / background median ({scores[cls]})")
        ax.set_xticks(range(len(strata)), strata, rotation=50, ha="right")
        ax.set_title(f"{cls}: {scores[cls]}", loc="left")
        ax.legend(frameon=False, loc="upper left")
    fig.suptitle("Highest threshold per dataset that keeps 99.5% of relevant objects complete", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_dilation(fixed, cls, base, threshold, path):
    variants = [(base, "no dilation"), (f"{base}_D6", "face neighbours (D6)"), (f"{base}_D26", "all neighbours (D26)")]
    data = fixed[(fixed["class"] == cls) & (fixed["T"] == threshold) & (fixed.split != "failure")]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    wide = data.pivot_table(index="key", columns="score", values=["kept", "complete"])
    for key in wide.index:
        xs = [100 * wide.loc[key, ("kept", v)] for v, _ in variants]
        ys = [100 * wide.loc[key, ("complete", v)] for v, _ in variants]
        ax.plot(xs, ys, color=GRID, linewidth=1, zorder=1)
    for i, (variant, label) in enumerate(variants):
        d = data[data.score == variant]
        ax.scatter(100 * d.kept, 100 * d.complete, s=28, color=SERIES[i], edgecolor=SURFACE, linewidth=0.8,
                   label=f"{label}: median {100 * d.kept.median():.1f}% blocks", zorder=3)
    ax.axhline(99.5, color=INK_2, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.annotate("target 99.5%", (0.01, 99.5), xycoords=("axes fraction", "data"), xytext=(0, -10),
                textcoords="offset points", color=INK_2, fontsize=8)
    ax.set_xlabel("predicted blocks (% of all 128^3 blocks, log scale)")
    ax.set_ylabel("complete relevant objects (%)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title(f"{cls}: {base} > {threshold} (s3 estimate of the production rule), one line per dataset",
                 loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_validation(results, rules, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, cls in zip(axes, ("SGN", "IHC")):
        for i, (scope, score, mapping) in enumerate(rules[cls]):
            d = results[(results.scope == scope) & (results.score == score) & (results.mapping == mapping)
                        & (results["class"] == cls) & (results.split == "val")]
            if len(d) == 0:
                continue
            ax.scatter(100 * d.kept, 100 * d.complete, s=26, color=SERIES[i], edgecolor=SURFACE, linewidth=0.8,
                       label=f"{score} > {d.theta.iloc[0]:.3g} b: median {100 * d.kept.median():.1f}% blocks",
                       zorder=3)
        ax.axhline(99.5, color=INK_2, linestyle="--", linewidth=1)
        ax.annotate("target 99.5%", (ax.get_xlim()[0], 99.5), xytext=(4, 3), textcoords="offset points",
                    color=INK_2, fontsize=8)
        ax.set_xlabel("predicted blocks (% of all 128^3 blocks)")
        ax.set_title(f"{cls}: validation datasets, rules fitted on train", loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    axes[0].set_ylabel("complete relevant objects (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sgn_score", default="s3p95")
    parser.add_argument("--ihc_score", default="s3p99_D26")
    parser.add_argument("--mapping", default="rel")
    args = parser.parse_args()

    evaluation = os.path.join(OUTPUT_ROOT, "evaluation")
    audit = pd.read_csv(os.path.join(OUTPUT_ROOT, "audit", "summary.csv"))
    fig_audit(audit, output_path("figures", "audit_cutoffs.png"))
    fixed = pd.read_csv(os.path.join(evaluation, "fixed_rules.csv"))
    fig_dilation(fixed, "IHC", "s3p99", 150, output_path("figures", "ihc_dilation.png"))
    scores = {"SGN": args.sgn_score, "IHC": args.ihc_score}
    critical = pd.read_csv(os.path.join(evaluation, "critical_t0.995.csv"))
    fig_critical(critical, scores, args.mapping, output_path("figures", "critical_thresholds.png"))
    results = pd.read_csv(os.path.join(evaluation, "results_t0.995.csv"))
    rules = {"SGN": [("SGN", args.sgn_score, args.mapping), ("SGN", "s3max_r0", args.mapping)],
             "IHC": [("IHC", args.ihc_score, args.mapping), ("IHC", "s3p99_D6", args.mapping)]}
    fig_validation(results, rules, output_path("figures", "validation.png"))
    print("figures ->", os.path.join(OUTPUT_ROOT, "figures"))


if __name__ == "__main__":
    main()
