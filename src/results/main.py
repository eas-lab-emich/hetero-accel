import argparse
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[2]
# RESULTS_ROOT = REPO_ROOT / "run/latest_log_dir"
RESULTS_ROOT = REPO_ROOT / "corrected_seed_pairs_multiplicative_continuous/workload_a"
ITERATIONS = range(126)

STL_SUFFIX = "_stl"
STYLES = {
    "baseline": {"label": "Baseline", "color": "tab:blue", "linestyle": "-"},
    "stl": {"label": "STL", "color": "tab:orange", "linestyle": ":"},
}


def split_run_name(name):
    """'0_stl' -> ('0', 'stl'); '0' -> ('0', 'baseline')."""
    if name.endswith(STL_SUFFIX):
        return name[: -len(STL_SUFFIX)], "stl"
    return name, "baseline"


def seed_sort_key(seed):
    """Numeric seeds sort numerically; anything else sorts after them."""
    return (0, int(seed)) if seed.isdigit() else (1, seed)


def load_datasets(workload, results_root=RESULTS_ROOT):
    """Load accelerator metrics for every seed and optimization variant."""
    # workload_dir = Path(results_root) / f"workload_{workload}"
    workload_dir = Path(results_root)
    datasets = {}

    for run_dir in sorted(path for path in workload_dir.iterdir() if path.is_dir()):
        csv_files = sorted(run_dir.glob("accelerator_metrics*.csv"))
        if not csv_files:
            continue

        seed, variant = split_run_name(run_dir.name)
        datasets[(seed, variant)] = pd.read_csv(csv_files[0])

    if not datasets:
        raise FileNotFoundError(f"No accelerator metrics found in {workload_dir}")

    return datasets


def best_feasible_edp(metrics):
    """Return the cumulative best feasible EDP for iterations 0 through 125."""
    metrics = metrics.copy()
    metrics["edp"] = pd.to_numeric(metrics["edp"], errors="coerce")

    feasible = (
            metrics["evaluation_result"].eq("Success")
            & metrics["edp"].notna()
            & metrics["edp"].ne(float("inf"))
    )

    best = (
        metrics.loc[feasible]
        .set_index("step")["edp"]
        .reindex(ITERATIONS)
        .cummin()
    )

    return best.ffill()


def summarize_datasets(datasets):
    """Build per-run best-EDP and accepted-accuracy summary tables."""
    best_edp_rows = []
    accuracy_rows = []

    for (seed, variant), metrics in sorted(
            datasets.items(),
            key=lambda item: (seed_sort_key(item[0][0]), item[0][1]),
    ):
        metrics = metrics.copy()

        metrics["edp"] = pd.to_numeric(metrics["edp"], errors="coerce")
        metrics["accuracy_loss"] = pd.to_numeric(
            metrics["accuracy_loss"], errors="coerce"
        )

        feasible = (
                metrics["evaluation_result"].eq("Success")
                & metrics["edp"].notna()
                & metrics["edp"].ne(float("inf"))
        )

        best = metrics.loc[feasible].sort_values("edp").iloc[0]
        accepted = metrics.loc[
            metrics["is_accepted"].eq(True),
            "accuracy_loss",
        ]

        best_edp_rows.append(
            {
                "seed": seed,
                "variant": STYLES[variant]["label"],
                "best_edp": best["edp"],
                "iteration": int(best["step"]),
            }
        )

        accuracy_rows.append(
            {
                "seed": seed,
                "variant": STYLES[variant]["label"],
                "accepted_count": len(accepted),
                "average_accuracy_loss": accepted.mean(),
            }
        )

    return pd.DataFrame(best_edp_rows), pd.DataFrame(accuracy_rows)


def print_summary_tables(datasets):
    best_edp, accuracy = summarize_datasets(datasets)

    print("\nBest feasible EDP per run:")
    print(
        best_edp.to_string(
            index=False,
            formatters={"best_edp": "{:.6e}".format},
        )
    )

    print("\nAverage accuracy loss across accepted rows:")
    print(
        accuracy.to_string(
            index=False,
            formatters={"average_accuracy_loss": "{:.4f}".format},
        )
    )


def plot_best_edp(datasets, workload):
    figure, axis = plt.subplots(figsize=(12, 7))

    seeds = sorted({seed for seed, _ in datasets}, key=seed_sort_key)
    colors = {
        seed: plt.get_cmap("tab10")(index % 10)
        for index, seed in enumerate(seeds)
    }

    for (seed, variant), metrics in sorted(datasets.items()):
        style = STYLES[variant]
        best = best_feasible_edp(metrics)

        axis.plot(
            best.index,
            best.values,
            color=colors[seed],
            linestyle=style["linestyle"],
            label=f"seed {seed} ({style['label'].lower()})",
        )

    axis.set_title(f"Workload {workload.upper()}: Best Feasible EDP")
    axis.set_xlabel("Annealing iteration")
    axis.set_ylabel("Best cumulative EDP")
    axis.set_xlim(0, 125)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="small", ncol=2)

    figure.tight_layout()
    return figure, axis


def plot_aggregate_best_edp(datasets, workload):
    figure, axis = plt.subplots(figsize=(12, 7))

    for variant, style in STYLES.items():
        seed_curves = [
            best_feasible_edp(metrics)
            for (seed, dataset_variant), metrics in datasets.items()
            if dataset_variant == variant
        ]

        if not seed_curves:
            continue

        values = pd.concat(seed_curves, axis=1)

        median = values.median(axis=1)
        lower = values.quantile(0.25, axis=1)
        upper = values.quantile(0.75, axis=1)

        axis.plot(
            median.index,
            median.values,
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
        )

        axis.fill_between(
            median.index,
            lower.values,
            upper.values,
            color=style["color"],
            alpha=0.2,
        )

    axis.set_title(
        f"Workload {workload.upper()}: Aggregate Best Feasible EDP"
    )
    axis.set_xlabel("Annealing iteration")
    axis.set_ylabel("Best cumulative EDP")
    axis.set_xlim(0, 125)
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    return figure, axis


# ---------------------------------------------------------------------------
# New figure 1:
# Final paired STL effect relative to the corresponding baseline.
# ---------------------------------------------------------------------------

def paired_edp_changes(datasets):
    """Return one row per paired seed with STL EDP change vs baseline."""
    best_edp, _ = summarize_datasets(datasets)

    pivot = best_edp.pivot(
        index="seed",
        columns="variant",
        values="best_edp",
    ).dropna(subset=["Baseline", "STL"])

    pivot["ratio"] = pivot["STL"] / pivot["Baseline"]
    pivot["percent_change"] = 100.0 * (pivot["ratio"] - 1.0)

    return pivot.reset_index()


def paired_log_ratio_tests(datasets):
    """Compute paired log-ratio effect sizes and two-sided paired tests.

    For each seed, d_i = log(EDP_STL / EDP_Baseline). Negative values favor STL.
    The permutation p-value is exact: for n pairs, all 2**n sign flips are
    enumerated under the null that treatment labels are exchangeable within pairs.
    """
    paired = paired_edp_changes(datasets).copy()
    paired["log_ratio"] = np.log(paired["ratio"])

    log_ratios = paired["log_ratio"].to_numpy(dtype=float)
    if len(log_ratios) == 0:
        raise ValueError("No complete baseline/STL seed pairs were found.")

    observed_mean = float(log_ratios.mean())

    # Exact two-sided paired permutation test using all possible sign flips.
    permutation_means = np.fromiter(
        (
            np.mean(log_ratios * np.asarray(signs))
            for signs in itertools.product((-1.0, 1.0), repeat=len(log_ratios))
        ),
        dtype=float,
        count=2 ** len(log_ratios),
    )
    permutation_p = float(
        np.mean(np.abs(permutation_means) >= abs(observed_mean) - 1e-15)
    )

    # Wilcoxon signed-rank test on the same paired log ratios.
    nonzero = log_ratios[~np.isclose(log_ratios, 0.0)]
    if len(nonzero) == 0:
        wilcoxon_statistic = 0.0
        wilcoxon_p = 1.0
    else:
        wilcoxon_result = wilcoxon(
            nonzero,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        wilcoxon_statistic = float(wilcoxon_result.statistic)
        wilcoxon_p = float(wilcoxon_result.pvalue)

    geometric_mean_ratio = math.exp(observed_mean)

    return paired, {
        "n": len(log_ratios),
        "mean_log_ratio": observed_mean,
        "geometric_mean_ratio": geometric_mean_ratio,
        "geometric_mean_percent_change": 100.0 * (geometric_mean_ratio - 1.0),
        "median_percent_change": float(paired["percent_change"].median()),
        "permutation_p_two_sided": permutation_p,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p_two_sided": wilcoxon_p,
    }


def print_paired_statistical_tests(datasets):
    """Print paired log-ratio effect sizes and inferential tests."""
    paired, stats = paired_log_ratio_tests(datasets)

    print("\nPaired STL / Baseline log-ratio analysis:")
    print(
        paired[["seed", "ratio", "percent_change", "log_ratio"]]
        .sort_values("percent_change")
        .to_string(
            index=False,
            formatters={
                "ratio": "{:.6f}".format,
                "percent_change": "{:+.2f}%".format,
                "log_ratio": "{:+.6f}".format,
            },
        )
    )

    print("\nPaired statistical tests:")
    print(f"  n paired seeds:                 {stats['n']}")
    print(f"  mean log ratio:                 {stats['mean_log_ratio']:+.6f}")
    print(f"  geometric mean STL/base ratio:  {stats['geometric_mean_ratio']:.6f}")
    print(
        "  geometric mean EDP change:      "
        f"{stats['geometric_mean_percent_change']:+.2f}%"
    )
    print(
        "  median paired EDP change:       "
        f"{stats['median_percent_change']:+.2f}%"
    )
    print(
        "  exact permutation p (2-sided):  "
        f"{stats['permutation_p_two_sided']:.6f}"
    )
    print(
        "  Wilcoxon signed-rank W:          "
        f"{stats['wilcoxon_statistic']:.6f}"
    )
    print(
        "  Wilcoxon p (2-sided):            "
        f"{stats['wilcoxon_p_two_sided']:.6f}"
    )


def plot_paired_edp_change(datasets, workload):
    """Horizontal bars showing STL's final EDP change relative to baseline."""
    paired = paired_edp_changes(datasets).sort_values("percent_change")

    figure, axis = plt.subplots(figsize=(10, 6))

    colors = [
        STYLES["stl"]["color"] if value < 0 else "tab:red"
        for value in paired["percent_change"]
    ]

    bars = axis.barh(
        paired["seed"],
        paired["percent_change"],
        color=colors,
        alpha=0.85,
    )

    axis.axvline(0, color="black", linewidth=1)

    for bar, value in zip(bars, paired["percent_change"]):
        x = bar.get_width()

        if value < 0:
            text_x = x - 0.8
            horizontal_alignment = "right"
        else:
            text_x = x + 0.8
            horizontal_alignment = "left"

        axis.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f}%",
            va="center",
            ha=horizontal_alignment,
            )

    axis.set_title(
        f"Workload {workload.upper()}: STL Best-EDP Change by Paired Seed"
    )
    axis.set_xlabel(
        "STL change relative to baseline best feasible EDP (%)"
    )
    axis.set_ylabel("Seed")
    axis.grid(True, axis="x", alpha=0.3)

    figure.tight_layout()
    return figure, axis


# ---------------------------------------------------------------------------
# New figure 2:
# One paired convergence plot per seed.
# ---------------------------------------------------------------------------

def plot_paired_seed_convergence(datasets, workload):
    """Small multiples comparing baseline and STL convergence per seed."""
    seeds = sorted(
        {
            seed
            for seed, _ in datasets
            if (seed, "baseline") in datasets
               and (seed, "stl") in datasets
        },
        key=seed_sort_key,
    )

    columns = 3
    rows = math.ceil(len(seeds) / columns)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, 4 * rows),
        sharex=True,
    )

    axes = axes.flatten()

    for axis, seed in zip(axes, seeds):
        baseline = best_feasible_edp(datasets[(seed, "baseline")])
        stl = best_feasible_edp(datasets[(seed, "stl")])

        axis.plot(
            baseline.index,
            baseline.values,
            color=STYLES["baseline"]["color"],
            linestyle=STYLES["baseline"]["linestyle"],
            label="Baseline",
        )

        axis.plot(
            stl.index,
            stl.values,
            color=STYLES["stl"]["color"],
            linestyle=STYLES["stl"]["linestyle"],
            label="STL",
        )

        baseline_final = baseline.min()
        stl_final = stl.min()

        percent_change = 100.0 * (
                stl_final / baseline_final - 1.0
        )

        axis.set_title(
            f"Seed {seed} ({percent_change:+.1f}%)"
        )

        axis.set_xlim(0, 125)
        axis.grid(True, alpha=0.3)

    for axis in axes[len(seeds):]:
        axis.set_visible(False)

    for axis in axes[-columns:]:
        if axis.get_visible():
            axis.set_xlabel("Annealing iteration")

    for row in range(rows):
        axes[row * columns].set_ylabel("Best cumulative EDP")

    handles, labels = axes[0].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.945),
    )

    figure.suptitle(
        f"Workload {workload.upper()}: Paired Best-Feasible EDP Pairs",
        y=0.985,
        fontsize=14,
    )

    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.95)
    )

    return figure, axes


# ---------------------------------------------------------------------------
# New figure 3:
# Does STL improvement correlate with greater accuracy-budget usage?
# ---------------------------------------------------------------------------

def plot_accuracy_vs_edp_change(datasets, workload):
    """Compare change in accepted accuracy loss with change in best EDP."""
    best_edp, accuracy = summarize_datasets(datasets)

    best_pivot = best_edp.pivot(
        index="seed",
        columns="variant",
        values="best_edp",
    )

    accuracy_pivot = accuracy.pivot(
        index="seed",
        columns="variant",
        values="average_accuracy_loss",
    )

    paired = pd.DataFrame(index=best_pivot.index)

    paired["edp_change"] = 100.0 * (
            best_pivot["STL"] / best_pivot["Baseline"] - 1.0
    )

    paired["accuracy_loss_change"] = (
            accuracy_pivot["STL"] - accuracy_pivot["Baseline"]
    )

    paired = paired.dropna()

    figure, axis = plt.subplots(figsize=(9, 7))

    axis.scatter(
        paired["accuracy_loss_change"],
        paired["edp_change"],
        s=70,
    )

    for seed, row in paired.iterrows():
        axis.annotate(
            str(seed),
            (
                row["accuracy_loss_change"],
                row["edp_change"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize="small",
        )

    axis.axhline(0, color="black", linewidth=1)
    axis.axvline(0, color="black", linewidth=1)

    axis.set_title(
        f"Workload {workload.upper()}: "
        "Accuracy-Budget Use vs Best-EDP Change"
    )

    axis.set_xlabel(
        "Change in mean accepted accuracy loss "
        "(STL - baseline)"
    )

    axis.set_ylabel(
        "Best feasible EDP change (%)"
    )

    axis.grid(True, alpha=0.3)

    figure.tight_layout()
    return figure, axis


def main():
    parser = argparse.ArgumentParser(
        description="Plot cumulative best EDP results."
    )

    parser.add_argument(
        "--workload",
        choices=("a", "b", "c"),
        default="a",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Save the plots instead of displaying them.",
    )

    args = parser.parse_args()

    datasets = load_datasets(args.workload)

    print_summary_tables(datasets)
    print_paired_statistical_tests(datasets)

    figure, _ = plot_best_edp(
        datasets,
        args.workload,
    )

    aggregate_figure, _ = plot_aggregate_best_edp(
        datasets,
        args.workload,
    )

    paired_change_figure, _ = plot_paired_edp_change(
        datasets,
        args.workload,
    )

    paired_convergence_figure, _ = plot_paired_seed_convergence(
        datasets,
        args.workload,
    )

    accuracy_figure, _ = plot_accuracy_vs_edp_change(
        datasets,
        args.workload,
    )

    if args.output:
        figure.savefig(
            args.output,
            dpi=300,
            bbox_inches="tight",
        )

        aggregate_figure.savefig(
            args.output.with_name(
                f"{args.output.stem}_aggregate{args.output.suffix}"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        paired_change_figure.savefig(
            args.output.with_name(
                f"{args.output.stem}_paired_change{args.output.suffix}"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        paired_convergence_figure.savefig(
            args.output.with_name(
                f"{args.output.stem}_paired_convergence{args.output.suffix}"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        accuracy_figure.savefig(
            args.output.with_name(
                f"{args.output.stem}_accuracy_vs_edp{args.output.suffix}"
            ),
            dpi=300,
            bbox_inches="tight",
        )

    else:
        plt.show()


if __name__ == "__main__":
    main()