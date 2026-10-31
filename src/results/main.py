import argparse
import itertools
import math
from enum import Enum
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import wilcoxon, wasserstein_distance

from src.train_test import validate

REPO_ROOT = Path(__file__).resolve().parents[2]
# RESULTS_ROOT = REPO_ROOT / "run/latest_log_dir"
RESULTS_ROOT = REPO_ROOT / "corrected_seed_pairs_multiplicative_continuous_frozen/workload_a"
ITERATIONS = range(126)

STL_SUFFIX = "_stl"
STYLES = {
    "baseline": {"label": "Baseline", "color": "tab:blue", "linestyle": "-"},
    "stl": {"label": "STL", "color": "tab:orange", "linestyle": ":"},
}

design_space = {
    "pe_array_x": [8, 10, 12, 14, 16, 20, 25],
    "pe_array_y": [8, 10, 12, 14, 16, 20, 25],
    "ifmap-spad_size": [8, 12, 16, 24, 32, 40, 48, 64],
    "weights_spad_size": [256, 320, 384, 448, 512, 576, 640],
    "psum_spad_size": [16, 24, 32, 40, 48, 64, 80, 96],
    "sram_size": [45000, 60000, 80000, 108000, 120000, 140000, 180000]
}

def split_run_name(name):
    """'0_stl' -> ('0', 'stl'); '0' -> ('0', 'baseline')."""
    if name.endswith(STL_SUFFIX):
        return name[: -len(STL_SUFFIX)], "stl"
    return name, "baseline"


def seed_sort_key(seed):
    """Numeric seeds sort numerically; anything else sorts after them."""
    return (0, int(seed)) if seed.isdigit() else (1, seed)

class MetricType(Enum):
    ACCELERATOR = "accelerator_metrics*.csv"
    SUBACCELERATOR = "subaccelerator_params_metrics*.csv"


def load_datasets(metric_type: MetricType, results_root=RESULTS_ROOT):
    """Load accelerator metrics for every seed and optimization variant."""
    # workload_dir = Path(results_root) / f"workload_{workload}"
    workload_dir = Path(results_root)
    datasets = {}

    for run_dir in sorted(path for path in workload_dir.iterdir() if path.is_dir()):
        csv_files = sorted(run_dir.glob(metric_type.value))
        if not csv_files:
            continue

        seed, variant = split_run_name(run_dir.name)
        datasets[(seed, variant)] = pd.read_csv(csv_files[0])

    if not datasets:
        raise FileNotFoundError(f"No accelerator metrics found in {workload_dir}")

    return datasets

def load_dataframe(metric_type: MetricType, results_root=RESULTS_ROOT):
    """Load accelerator metrics for every seed and optimization variant."""
    # workload_dir = Path(results_root) / f"workload_{workload}"
    workload_dir = Path(results_root)
    dataframes = []

    for run_dir in sorted(path for path in workload_dir.iterdir() if path.is_dir()):
        csv_files = sorted(run_dir.glob(metric_type.value))
        if not csv_files:
            continue

        seed, variant = split_run_name(run_dir.name)
        df  = pd.read_csv(csv_files[0])
        df["seed"] = seed
        df["variant"] = variant
        dataframes.append(df)

    if not dataframes:
        raise FileNotFoundError(f"No accelerator metrics found in {workload_dir}")

    return pd.concat(dataframes, ignore_index=True)


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
        f"Workload {workload.upper()}: Aggregate Best Feasible EDP (IQR)"
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


def paired_log_ratio_tests_monte_carlo(datasets):
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
    # Monte Carlo two-sided paired permutation test
    rng = np.random.default_rng(42)
    n_permutations = 100_000

    signs = rng.choice(
        [-1.0, 1.0],
        size=(n_permutations, len(log_ratios)),
    )

    permutation_means = (signs * log_ratios).mean(axis=1)

    permutation_p = (
                            np.count_nonzero(
                                np.abs(permutation_means) >= abs(observed_mean) - 1e-15
                            ) + 1
                    ) / (n_permutations + 1)

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
        "  Monte Carlo permutation p (2-sided): "
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

    min_change = paired["percent_change"].min()
    max_change = paired["percent_change"].max()
    padding = (max_change - min_change) * 0.10

    axis.set_xlim(
        min_change - padding,
        max_change + padding,
    )

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
        "Mean Accepted Accuracy Loss vs. Best-EDP Change"
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

def param_index(config_param, param_value):
    return design_space[config_param].index(param_value)

def print_wasserstein(df: DataFrame, accepted_only=False):
    config_params = ["pe_array_x", "pe_array_y", "sram_size", "ifmap-spad_size", "weights_spad_size", "psum_spad_size"]
    print(f"is_accepted_only={accepted_only}")
    if accepted_only:
        keys = ["seed", "variant", "precision"]

        df = df.sort_values(keys + ["step"]).copy()

        # Mark rejected rows as missing, then forward-fill within each group
        value_cols = df.columns.difference(keys + ["step", "is_accepted"])
        accepted = df["is_accepted"].isna() | (df["is_accepted"] == "") | (df["is_accepted"] == True)

        df[value_cols] = (
            df[value_cols]
            .where(accepted)
            .groupby([df[k] for k in keys])
            .ffill()
        )
    print("seed,precision,config_param,distance")
    for (seed, precision), group in df.groupby(["seed", "precision"]):
        for config_param in config_params:
            df[config_param].map(lambda p: param_index(config_param, p))
            variants = {variant: variant_df[variant_df[config_param].notna()]
                        for variant, variant_df in group.groupby("variant")}
            if len(variants) != 2 or variants.get("baseline") is None or variants.get("stl") is None:
                raise RuntimeError(f"Should be 2 variants per seed, 'baseline' and 'stl', got len={len(variants)}")
            baseline = variants["baseline"][config_param].map(lambda p: param_index(config_param, p))
            stl = variants["stl"][config_param].map(lambda p: param_index(config_param, p))
            distance = wasserstein_distance(
                baseline,
                stl
            )
            print(f"{seed},{precision},{config_param},{distance}")
    return


def plot_regional_ifmap_weights_improvements(
        df: DataFrame,
        workload,
        partition_size: int = 2,
):
    params = ["ifmap-spad_size", "weights_spad_size"]
    precisions = [6, 7, 8]
    variants = ["baseline", "stl"]

    group = df.copy()

    # Map hardware values to discrete partition indices.
    for param in params:
        index_lookup = {
            value: i
            for i, value in enumerate(design_space[param])
        }
        group[param + "_region"] = (
                group[param].map(index_lookup) // partition_size
        )

    n_x_regions = math.ceil(
        len(design_space["ifmap-spad_size"]) / partition_size
    )
    n_y_regions = math.ceil(
        len(design_space["weights_spad_size"]) / partition_size
    )

    # Determine new best feasible EDP for each complete
    # accelerator evaluation, independently per run.
    evaluations = (
        group.drop_duplicates(["seed", "variant", "step"])
        [["seed", "variant", "step", "edp", "evaluation_result"]]
        .copy()
        .sort_values(["seed", "variant", "step"])
    )

    evaluations["edp"] = pd.to_numeric(
        evaluations["edp"], errors="coerce"
    )

    feasible = (
            evaluations["evaluation_result"].eq("Success")
            & np.isfinite(evaluations["edp"])
    )

    evaluations["feasible_edp"] = (
        evaluations["edp"].where(feasible, np.inf)
    )

    previous_best = (
        evaluations.groupby(["seed", "variant"])["feasible_edp"]
        .transform(
            lambda values: values.cummin().shift(1).fillna(np.inf)
        )
    )

    evaluations["is_improvement"] = (
            feasible
            & evaluations["edp"].lt(previous_best)
    )

    group = group.merge(
        evaluations[
            ["seed", "variant", "step", "is_improvement"]
        ],
        on=["seed", "variant", "step"],
        how="left",
        validate="many_to_one",
    )

    probabilities = {}
    counts = {}

    for precision in precisions:
        for variant in variants:
            subset = group.loc[
                group["precision"].eq(precision)
                & group["variant"].eq(variant)
                ].dropna(subset=[
                "ifmap-spad_size_region",
                "weights_spad_size_region",
            ])

            total = np.zeros((n_y_regions, n_x_regions))
            improved = np.zeros((n_y_regions, n_x_regions))

            x = subset["ifmap-spad_size_region"].astype(int)
            y = subset["weights_spad_size_region"].astype(int)

            np.add.at(total, (y, x), 1)

            successful = subset["is_improvement"].eq(True)

            np.add.at(
                improved,
                (y[successful], x[successful]),
                1,
            )

            probabilities[(precision, variant)] = np.divide(
                improved,
                total,
                out=np.full_like(total, np.nan),
                where=total > 0,
            )

            counts[(precision, variant)] = {
                "total": total,
                "improved": improved,
            }

    # Shared color scale across precision levels and variants.
    finite_maxima = [
        np.nanmax(matrix)
        for matrix in probabilities.values()
        if np.isfinite(matrix).any()
    ]
    vmax = max(finite_maxima, default=1.0)

    figure, axes = plt.subplots(
        3, 2,
        figsize=(11, 14),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")

    for row, precision in enumerate(precisions):
        for col, variant in enumerate(variants):
            axis = axes[row, col]
            matrix = probabilities[(precision, variant)]

            image = axis.imshow(
                matrix,
                origin="lower",
                cmap=cmap,
                vmin=0,
                vmax=vmax,
            )

            axis.set_title(
                f"{precision}-bit {STYLES[variant]['label']}"
            )
            axis.set_xlabel("IFMAP SPAD partition")
            axis.set_ylabel("Weights SPAD partition")
            axis.set_xticks(range(n_x_regions))
            axis.set_yticks(range(n_y_regions))

            # Annotate improvements / total evaluations.
            for y in range(n_y_regions):
                for x in range(n_x_regions):
                    total = int(
                        counts[(precision, variant)]["total"][y, x]
                    )
                    improved = int(
                        counts[(precision, variant)]["improved"][y, x]
                    )

                    if total:
                        rgba = cmap(matrix[y, x] / vmax)
                        r, g, b = rgba[:3]

                        # Relative luminance (WCAG)
                        def linearize(c):
                            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

                        luminance = (
                                0.2126 * linearize(r)
                                + 0.7152 * linearize(g)
                                + 0.0722 * linearize(b)
                        )

                        axis.text(
                            x, y, f"{improved}/{total}",
                            ha="center",
                            va="center",
                            fontsize=9,
                            fontweight="bold",
                            color="black" if luminance > 0.179 else "white",
                        )

    figure.colorbar(
        image,
        ax=axes,
        shrink=0.65,
        label="Probability of new best feasible EDP",
    )

    figure.suptitle(
        f"Workload {workload}: Regional EDP Improvement Frequency: "
        "IFMAP × Weights",
        fontsize=15,
    )

    return figure, axes


def plot_regional_edp_improvements(
        df: DataFrame,
        workload,
        partition_size: int = 2,
):
    params = ["pe_array_x", "pe_array_y"]
    precisions = [6, 7, 8]
    variants = ["baseline", "stl"]

    group = df.copy()

    # Map physical PE dimensions to discrete partition indices.
    for param in params:
        index_lookup = {
            value: i
            for i, value in enumerate(design_space[param])
        }
        group[param + "_region"] = (
                group[param].map(index_lookup) // partition_size
        )

    n_regions = math.ceil(
        len(design_space["pe_array_x"]) / partition_size
    )

    # Determine whether each proposed complete accelerator
    # configuration establishes a new best feasible EDP.
    #
    # Calculate once per (seed, variant, step), not once
    # per subaccelerator.
    evaluations = (
        group.drop_duplicates(["seed", "variant", "step"])
        [["seed", "variant", "step", "edp", "evaluation_result"]]
        .copy()
        .sort_values(["seed", "variant", "step"])
    )

    evaluations["edp"] = pd.to_numeric(
        evaluations["edp"], errors="coerce"
    )

    feasible = (
            evaluations["evaluation_result"].eq("Success")
            & np.isfinite(evaluations["edp"])
    )

    evaluations["feasible_edp"] = (
        evaluations["edp"].where(feasible, np.inf)
    )

    run_keys = ["seed", "variant"]

    previous_best = (
        evaluations.groupby(["seed", "variant"])["feasible_edp"]
        .transform(lambda values: values.cummin().shift(1).fillna(np.inf))
    )

    evaluations["is_improvement"] = (
            feasible
            & evaluations["edp"].lt(previous_best)
    )

    # Attach improvement flags to the per-subaccelerator rows.
    group = group.merge(
        evaluations[
            ["seed", "variant", "step", "is_improvement"]
        ],
        on=["seed", "variant", "step"],
        how="left",
        validate="many_to_one",
    )

    probabilities = {}
    counts = {}

    for precision in precisions:
        for variant in variants:
            subset = group.loc[
                group["precision"].eq(precision)
                & group["variant"].eq(variant)
                ].dropna(subset=[
                "pe_array_x_region", "pe_array_y_region"
            ])

            total = np.zeros((n_regions, n_regions))
            improved = np.zeros((n_regions, n_regions))

            x = subset["pe_array_x_region"].astype(int)
            y = subset["pe_array_y_region"].astype(int)

            np.add.at(total, (y, x), 1)

            successful = subset["is_improvement"].eq(True)
            np.add.at(
                improved,
                (y[successful], x[successful]),
                1,
            )

            probabilities[(precision, variant)] = np.divide(
                improved,
                total,
                out=np.full_like(total, np.nan),
                where=total > 0,
            )

            counts[(precision, variant)] = {
                "total": total,
                "improved": improved,
            }
    # Shared scale allows direct comparison across all panels.
    finite_maxima = [
        np.nanmax(matrix)
        for matrix in probabilities.values()
        if np.isfinite(matrix).any()
    ]
    vmax = max(finite_maxima, default=1.0)

    figure, axes = plt.subplots(
        3, 2, figsize=(11, 14),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")

    for row, precision in enumerate(precisions):
        for col, variant in enumerate(variants):
            axis = axes[row, col]

            matrix = probabilities[(precision, variant)]

            image = axis.imshow(
                matrix,
                origin="lower",
                cmap=cmap,
                vmin=0,
                vmax=vmax,
            )

            axis.set_title(
                f"{precision}-bit {STYLES[variant]['label']}"
            )
            axis.set_xlabel("PE X partition")
            axis.set_ylabel("PE Y partition")
            axis.set_xticks(range(n_regions))
            axis.set_yticks(range(n_regions))

            # Show sample counts in each occupied region.
            for y in range(n_regions):
                for x in range(n_regions):
                    total = int(counts[(precision, variant)]["total"][y, x])
                    improved = int(counts[(precision, variant)]["improved"][y, x])

                    if total:
                        rgba = cmap(matrix[y, x] / vmax)
                        r, g, b = rgba[:3]

                        # Relative luminance (WCAG)
                        def linearize(c):
                            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

                        luminance = (
                                0.2126 * linearize(r)
                                + 0.7152 * linearize(g)
                                + 0.0722 * linearize(b)
                        )

                        axis.text(
                            x, y, f"{improved}/{total}",
                            ha="center",
                            va="center",
                            fontsize=9,
                            fontweight="bold",
                            color="black" if luminance > 0.179 else "white",
                        )

    figure.colorbar(
        image,
        ax=axes,
        shrink=0.65,
        label="Probability of new best feasible EDP",
    )

    figure.suptitle(
        f"Workload {workload}: Regional EDP Improvement Frequency",
        fontsize=15,
    )

    return figure, axes


def plot_regional_psum_sram_improvements(
        df: DataFrame,
        workload,
        partition_size: int = 2,
):
    params = ["psum_spad_size", "sram_size"]
    precisions = [6, 7, 8]
    variants = ["baseline", "stl"]

    group = df.copy()

    # Map hardware values to discrete partition indices.
    for param in params:
        index_lookup = {
            value: i
            for i, value in enumerate(design_space[param])
        }
        group[param + "_region"] = (
                group[param].map(index_lookup) // partition_size
        )

    n_x_regions = math.ceil(
        len(design_space["psum_spad_size"]) / partition_size
    )
    n_y_regions = math.ceil(
        len(design_space["sram_size"]) / partition_size
    )

    # Identify new best feasible EDP records per complete
    # accelerator evaluation, independently for each run.
    evaluations = (
        group.drop_duplicates(["seed", "variant", "step"])
        [["seed", "variant", "step", "edp", "evaluation_result"]]
        .copy()
        .sort_values(["seed", "variant", "step"])
    )

    evaluations["edp"] = pd.to_numeric(
        evaluations["edp"], errors="coerce"
    )

    feasible = (
            evaluations["evaluation_result"].eq("Success")
            & np.isfinite(evaluations["edp"])
    )

    evaluations["feasible_edp"] = (
        evaluations["edp"].where(feasible, np.inf)
    )

    previous_best = (
        evaluations.groupby(["seed", "variant"])["feasible_edp"]
        .transform(
            lambda values: values.cummin().shift(1).fillna(np.inf)
        )
    )

    evaluations["is_improvement"] = (
            feasible
            & evaluations["edp"].lt(previous_best)
    )

    # Attach system-level improvement flags to subaccelerators.
    group = group.merge(
        evaluations[
            ["seed", "variant", "step", "is_improvement"]
        ],
        on=["seed", "variant", "step"],
        how="left",
        validate="many_to_one",
    )

    probabilities = {}
    counts = {}

    for precision in precisions:
        for variant in variants:
            subset = group.loc[
                group["precision"].eq(precision)
                & group["variant"].eq(variant)
                ].dropna(subset=[
                "psum_spad_size_region",
                "sram_size_region",
            ])

            total = np.zeros((n_y_regions, n_x_regions))
            improved = np.zeros((n_y_regions, n_x_regions))

            x = subset["psum_spad_size_region"].astype(int)
            y = subset["sram_size_region"].astype(int)

            np.add.at(total, (y, x), 1)

            successful = subset["is_improvement"].eq(True)

            np.add.at(
                improved,
                (y[successful], x[successful]),
                1,
            )

            probabilities[(precision, variant)] = np.divide(
                improved,
                total,
                out=np.full_like(total, np.nan),
                where=total > 0,
            )

            counts[(precision, variant)] = {
                "total": total,
                "improved": improved,
            }

    # Shared scale across all six heatmaps.
    finite_maxima = [
        np.nanmax(matrix)
        for matrix in probabilities.values()
        if np.isfinite(matrix).any()
    ]
    vmax = max(finite_maxima, default=1.0)

    figure, axes = plt.subplots(
        3, 2,
        figsize=(11, 14),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")

    for row, precision in enumerate(precisions):
        for col, variant in enumerate(variants):
            axis = axes[row, col]
            matrix = probabilities[(precision, variant)]

            image = axis.imshow(
                matrix,
                origin="lower",
                cmap=cmap,
                vmin=0,
                vmax=vmax,
            )

            axis.set_title(
                f"{precision}-bit {STYLES[variant]['label']}"
            )
            axis.set_xlabel("PSUM SPAD partition")
            axis.set_ylabel("SRAM partition")
            axis.set_xticks(range(n_x_regions))
            axis.set_yticks(range(n_y_regions))

            # Annotate new best EDP records / total evaluations.
            for y in range(n_y_regions):
                for x in range(n_x_regions):
                    total = int(
                        counts[(precision, variant)]["total"][y, x]
                    )
                    improved = int(
                        counts[(precision, variant)]["improved"][y, x]
                    )

                    if total:
                        rgba = cmap(matrix[y, x] / vmax)
                        r, g, b = rgba[:3]

                        # Relative luminance (WCAG)
                        def linearize(c):
                            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

                        luminance = (
                                0.2126 * linearize(r)
                                + 0.7152 * linearize(g)
                                + 0.0722 * linearize(b)
                        )

                        axis.text(
                            x, y, f"{improved}/{total}",
                            ha="center",
                            va="center",
                            fontsize=9,
                            fontweight="bold",
                            color="black" if luminance > 0.179 else "white",
                        )

    figure.colorbar(
        image,
        ax=axes,
        shrink=0.65,
        label="Probability of new best feasible EDP",
    )

    figure.suptitle(
        f"Workload {workload}: Regional EDP Improvement Frequency: PSUM × SRAM",
        fontsize=15,
    )

    return figure, axes


def plot_pe_occupancy(df: DataFrame, seed: str, precision: int,
                      partition_size: int = 2):
    params = ["pe_array_x", "pe_array_y"]
    group = df.loc[
        df["seed"].eq(str(seed))
        & df["precision"].eq(precision)
        ].copy()

    keys = ["seed", "variant", "precision"]
    group = group.sort_values(keys + ["step"])

    # Reconstruct post-decision incumbents.
    accepted = group["is_accepted"].isna() | group["is_accepted"].eq(True)

    group[params] = (
        group[params]
        .where(accepted)
        .groupby([group[k] for k in keys])
        .ffill()
    )

    n_options = len(design_space["pe_array_x"])
    n_regions = math.ceil(n_options / partition_size)

    for param in params:
        group[param + "_region"] = group[param].map(
            lambda value: (
                param_index(param, value) // partition_size
                if pd.notna(value) else np.nan
            )
        )

    occupancy = {}
    for variant in ("baseline", "stl"):
        subset = group.loc[group["variant"].eq(variant)]
        if subset.empty:
            raise ValueError(f"Missing {variant} data for seed {seed}")

        counts = np.zeros((n_regions, n_regions))

        for _, row in subset.dropna(
                subset=["pe_array_x_region", "pe_array_y_region"]
        ).iterrows():
            x = int(row["pe_array_x_region"])
            y = int(row["pe_array_y_region"])
            counts[y, x] += 1

        occupancy[variant] = counts / counts.sum()

    difference = occupancy["stl"] - occupancy["baseline"]
    limit = max(np.abs(difference).max(), 1e-12)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    vmax = max(matrix.max() for matrix in occupancy.values())

    for axis, variant in zip(axes[:2], ("baseline", "stl")):
        image = axis.imshow(
            occupancy[variant],
            origin="lower",
            cmap="Blues",
            vmin=0,
            vmax=vmax,
        )
        axis.set_title(STYLES[variant]["label"])

    image_diff = axes[2].imshow(
        difference,
        origin="lower",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axes[2].set_title("STL - Baseline")

    for axis in axes:
        axis.set_xlabel("PE X partition")
        axis.set_ylabel("PE Y partition")
        axis.set_xticks(range(n_regions))
        axis.set_yticks(range(n_regions))

    figure.colorbar(image, ax=axes[:2], shrink=0.75,
                    label="Occupancy fraction")
    figure.colorbar(image_diff, ax=axes[2], shrink=0.75,
                    label="Occupancy difference")

    figure.suptitle(
        f"Seed {seed}, {precision}-bit PE Array Occupancy"
    )

    tv = 0.5 * np.abs(
        occupancy["stl"] - occupancy["baseline"]
    ).sum()
    print(f"seed={seed}, total_variation_distance={tv}")
    return figure, axes



def plot_aggregate_pe_occupancy(
        df: DataFrame,
        workload,
        partition_size: int = 2,
):
    params = ["pe_array_x", "pe_array_y"]
    keys = ["seed", "variant", "precision"]
    precisions = [6, 7, 8]

    group = df.copy()
    group = group.sort_values(keys + ["step"])

    # Reconstruct post-decision incumbents.
    accepted = (
            group["is_accepted"].isna()
            | group["is_accepted"].eq(True)
    )

    group[params] = (
        group[params]
        .where(accepted)
        .groupby([group[k] for k in keys])
        .ffill()
    )

    n_regions = math.ceil(
        len(design_space["pe_array_x"]) / partition_size
    )

    for param in params:
        index_lookup = {
            value: i
            for i, value in enumerate(design_space[param])
        }
        group[param + "_region"] = (
                group[param].map(index_lookup) // partition_size
        )

    occupancies = {}
    differences = {}
    statistics = {}

    for precision in precisions:
        precision_df = group.loc[
            group["precision"].eq(precision)
        ]

        matrices = {"baseline": [], "stl": []}
        seed_tvs = []

        for seed, seed_df in precision_df.groupby("seed"):
            pair = {}

            for variant in ("baseline", "stl"):
                subset = seed_df.loc[
                    seed_df["variant"].eq(variant)
                ].dropna(subset=[
                    "pe_array_x_region",
                    "pe_array_y_region",
                ])

                if subset.empty:
                    raise ValueError(
                        f"Missing incumbent data: "
                        f"{seed}, {variant}, {precision}-bit"
                    )

                counts = np.zeros((n_regions, n_regions))

                x = subset["pe_array_x_region"].astype(int)
                y = subset["pe_array_y_region"].astype(int)

                np.add.at(counts, (y, x), 1)

                pair[variant] = counts / counts.sum()

            for variant in matrices:
                matrices[variant].append(pair[variant])

            seed_tvs.append(
                0.5 * np.abs(
                    pair["stl"] - pair["baseline"]
                ).sum()
            )
            # Baseline-vs-baseline TV across all unique seed pairs.
        baseline_matrices = matrices["baseline"]

        baseline_tvs = [
            0.5 * np.abs(a - b).sum()
            for a, b in itertools.combinations(baseline_matrices, 2)
        ]

        mean_baseline_tv = np.mean(baseline_tvs)

        occupancy = {
            variant: np.mean(values, axis=0)
            for variant, values in matrices.items()
        }

        difference = (
                occupancy["stl"] - occupancy["baseline"]
        )

        occupancies[precision] = occupancy
        differences[precision] = difference

        statistics[precision] = {
            "aggregate_tv": 0.5 * np.abs(difference).sum(),
            "mean_seed_tv": np.mean(seed_tvs),
            "mean_baseline_tv": mean_baseline_tv,
            "n_seeds": len(seed_tvs),
            "n_baseline_pairs": len(baseline_tvs),
        }

    # Shared scales across all precision levels.
    vmax = max(
        matrix.max()
        for occupancy in occupancies.values()
        for matrix in occupancy.values()
    )

    limit = max(
        max(np.abs(matrix).max() for matrix in differences.values()),
        1e-12,
    )

    figure, axes = plt.subplots(
        3, 3,
        figsize=(15, 14),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for row, precision in enumerate(precisions):
        occupancy = occupancies[precision]

        for col, variant in enumerate(("baseline", "stl")):
            axis = axes[row, col]

            image = axis.imshow(
                occupancy[variant],
                origin="lower",
                cmap="Blues",
                vmin=0,
                vmax=vmax,
            )

            axis.set_title(
                f"{precision}-bit {STYLES[variant]['label']}"
            )

        axis = axes[row, 2]

        image_diff = axis.imshow(
            differences[precision],
            origin="lower",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )

        axis.set_title(
            f"{precision}-bit STL - Baseline"
        )

        for axis in axes[row]:
            axis.set_xlabel("PE X partition")
            axis.set_ylabel("PE Y partition")
            axis.set_xticks(range(n_regions))
            axis.set_yticks(range(n_regions))

    figure.colorbar(
        image,
        ax=axes[:, :2],
        shrink=0.65,
        label="Mean occupancy fraction",
    )

    figure.colorbar(
        image_diff,
        ax=axes[:, 2],
        shrink=0.65,
        label="Mean occupancy difference",
    )

    figure.suptitle(
        f"Workload {workload}: Aggregate PE Array Occupancy",
        fontsize=15,
    )

    print("\nAggregate PE Array Occupancy:")
    print(
        "precision,aggregate_tv,mean_seed_tv,"
        "mean_baseline_tv,n_seeds,n_baseline_pairs"
    )

    for precision, stats in statistics.items():
        print(
            f"{precision},"
            f"{stats['aggregate_tv']:.4f},"
            f"{stats['mean_seed_tv']:.4f},"
            f"{stats['mean_baseline_tv']:.4f},"
            f"{stats['n_seeds']},"
            f"{stats['n_baseline_pairs']}"
        )

    return figure, axes


def plot_aggregate_ifmap_weights_occupancy(
        df: DataFrame,
        workload,
        partition_size: int = 2,
):
    params = ["ifmap-spad_size", "weights_spad_size"]
    x_param, y_param = params
    keys = ["seed", "variant", "precision"]
    precisions = [6, 7, 8]

    group = df.sort_values(keys + ["step"]).copy()

    # Reconstruct post-decision incumbents.
    accepted = (
            group["is_accepted"].isna()
            | group["is_accepted"].eq(True)
    )

    group[params] = (
        group[params]
        .where(accepted)
        .groupby([group[k] for k in keys])
        .ffill()
    )

    n_x_regions = math.ceil(
        len(design_space[x_param]) / partition_size
    )
    n_y_regions = math.ceil(
        len(design_space[y_param]) / partition_size
    )

    for param in params:
        index_lookup = {
            value: i for i, value in enumerate(design_space[param])
        }
        group[param + "_region"] = (
                group[param].map(index_lookup) // partition_size
        )

    occupancies = {}
    differences = {}
    statistics = {}

    for precision in precisions:
        precision_df = group.loc[
            group["precision"].eq(precision)
        ]

        matrices = {"baseline": [], "stl": []}
        seed_tvs = []

        for seed, seed_df in precision_df.groupby("seed"):
            pair = {}

            for variant in ("baseline", "stl"):
                subset = seed_df.loc[
                    seed_df["variant"].eq(variant)
                ].dropna(subset=[
                    x_param + "_region",
                    y_param + "_region",
                    ])

                if subset.empty:
                    raise ValueError(
                        f"Missing incumbent data: "
                        f"{seed}, {variant}, {precision}-bit"
                    )

                counts = np.zeros((n_y_regions, n_x_regions))

                x = subset[x_param + "_region"].astype(int)
                y = subset[y_param + "_region"].astype(int)

                np.add.at(counts, (y, x), 1)
                pair[variant] = counts / counts.sum()

            for variant in matrices:
                matrices[variant].append(pair[variant])

            seed_tvs.append(
                0.5 * np.abs(
                    pair["stl"] - pair["baseline"]
                ).sum()
            )

        occupancy = {
            variant: np.mean(values, axis=0)
            for variant, values in matrices.items()
        }

        difference = occupancy["stl"] - occupancy["baseline"]

        baseline_tvs = [
            0.5 * np.abs(a - b).sum()
            for a, b in itertools.combinations(
                matrices["baseline"], 2
            )
        ]

        occupancies[precision] = occupancy
        differences[precision] = difference

        statistics[precision] = {
            "aggregate_tv": 0.5 * np.abs(difference).sum(),
            "mean_seed_tv": np.mean(seed_tvs),
            "mean_baseline_tv": np.mean(baseline_tvs),
            "n_seeds": len(seed_tvs),
            "n_baseline_pairs": len(baseline_tvs),
        }

    vmax = max(
        matrix.max()
        for occupancy in occupancies.values()
        for matrix in occupancy.values()
    )

    limit = max(
        max(np.abs(matrix).max() for matrix in differences.values()),
        1e-12,
    )

    figure, axes = plt.subplots(
        3, 3,
        figsize=(15, 14),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for row, precision in enumerate(precisions):
        occupancy = occupancies[precision]

        for col, variant in enumerate(("baseline", "stl")):
            axis = axes[row, col]

            image = axis.imshow(
                occupancy[variant],
                origin="lower",
                cmap="Blues",
                vmin=0,
                vmax=vmax,
            )

            axis.set_title(
                f"{precision}-bit {STYLES[variant]['label']}"
            )

        axis = axes[row, 2]

        image_diff = axis.imshow(
            differences[precision],
            origin="lower",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )

        axis.set_title(f"{precision}-bit STL - Baseline")

        for axis in axes[row]:
            axis.set_xlabel("IFMAP SPAD partition")
            axis.set_ylabel("Weights SPAD partition")
            axis.set_xticks(range(n_x_regions))
            axis.set_yticks(range(n_y_regions))

    figure.colorbar(
        image,
        ax=axes[:, :2],
        shrink=0.65,
        label="Mean occupancy fraction",
    )

    figure.colorbar(
        image_diff,
        ax=axes[:, 2],
        shrink=0.65,
        label="Mean occupancy difference",
    )

    figure.suptitle(
        f"Workload {workload}: Aggregate IFMAP × Weights SPAD Occupancy",
        fontsize=15,
    )

    print("\nAggregate IFMAP × Weights Occupancy:")
    print(
        "precision,aggregate_tv,mean_seed_tv,"
        "mean_baseline_tv,n_seeds,n_baseline_pairs"
    )

    for precision, stats in statistics.items():
        print(
            f"{precision},"
            f"{stats['aggregate_tv']:.4f},"
            f"{stats['mean_seed_tv']:.4f},"
            f"{stats['mean_baseline_tv']:.4f},"
            f"{stats['n_seeds']},"
            f"{stats['n_baseline_pairs']}"
        )

    return figure, axes


def plot_aggregate_psum_sram_occupancy(
        df: DataFrame,
        workload,
        partition_size: int = 2,
):
    params = ["psum_spad_size", "sram_size"]
    x_param, y_param = params
    keys = ["seed", "variant", "precision"]
    precisions = [6, 7, 8]

    group = df.sort_values(keys + ["step"]).copy()

    # Reconstruct post-decision incumbents.
    accepted = (
            group["is_accepted"].isna()
            | group["is_accepted"].eq(True)
    )

    group[params] = (
        group[params]
        .where(accepted)
        .groupby([group[k] for k in keys])
        .ffill()
    )

    n_x_regions = math.ceil(
        len(design_space[x_param]) / partition_size
    )
    n_y_regions = math.ceil(
        len(design_space[y_param]) / partition_size
    )

    for param in params:
        index_lookup = {
            value: i for i, value in enumerate(design_space[param])
        }
        group[param + "_region"] = (
                group[param].map(index_lookup) // partition_size
        )

    occupancies = {}
    differences = {}
    statistics = {}

    for precision in precisions:
        precision_df = group.loc[
            group["precision"].eq(precision)
        ]

        matrices = {"baseline": [], "stl": []}
        seed_tvs = []

        for seed, seed_df in precision_df.groupby("seed"):
            pair = {}

            for variant in ("baseline", "stl"):
                subset = seed_df.loc[
                    seed_df["variant"].eq(variant)
                ].dropna(subset=[
                    x_param + "_region",
                    y_param + "_region",
                    ])

                if subset.empty:
                    raise ValueError(
                        f"Missing incumbent data: "
                        f"{seed}, {variant}, {precision}-bit"
                    )

                counts = np.zeros((n_y_regions, n_x_regions))

                x = subset[x_param + "_region"].astype(int)
                y = subset[y_param + "_region"].astype(int)

                np.add.at(counts, (y, x), 1)
                pair[variant] = counts / counts.sum()

            for variant in matrices:
                matrices[variant].append(pair[variant])

            seed_tvs.append(
                0.5 * np.abs(
                    pair["stl"] - pair["baseline"]
                ).sum()
            )

        occupancy = {
            variant: np.mean(values, axis=0)
            for variant, values in matrices.items()
        }

        difference = occupancy["stl"] - occupancy["baseline"]

        baseline_tvs = [
            0.5 * np.abs(a - b).sum()
            for a, b in itertools.combinations(
                matrices["baseline"], 2
            )
        ]

        occupancies[precision] = occupancy
        differences[precision] = difference

        statistics[precision] = {
            "aggregate_tv": 0.5 * np.abs(difference).sum(),
            "mean_seed_tv": np.mean(seed_tvs),
            "mean_baseline_tv": np.mean(baseline_tvs),
            "n_seeds": len(seed_tvs),
            "n_baseline_pairs": len(baseline_tvs),
        }

    vmax = max(
        matrix.max()
        for occupancy in occupancies.values()
        for matrix in occupancy.values()
    )

    limit = max(
        max(np.abs(matrix).max() for matrix in differences.values()),
        1e-12,
    )

    figure, axes = plt.subplots(
        3, 3,
        figsize=(15, 14),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for row, precision in enumerate(precisions):
        occupancy = occupancies[precision]

        for col, variant in enumerate(("baseline", "stl")):
            axis = axes[row, col]

            image = axis.imshow(
                occupancy[variant],
                origin="lower",
                cmap="Blues",
                vmin=0,
                vmax=vmax,
            )

            axis.set_title(
                f"{precision}-bit {STYLES[variant]['label']}"
            )

        axis = axes[row, 2]

        image_diff = axis.imshow(
            differences[precision],
            origin="lower",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )

        axis.set_title(f"{precision}-bit STL - Baseline")

        for axis in axes[row]:
            axis.set_xlabel("PSUM SPAD partition")
            axis.set_ylabel("SRAM partition")
            axis.set_xticks(range(n_x_regions))
            axis.set_yticks(range(n_y_regions))

    figure.colorbar(
        image,
        ax=axes[:, :2],
        shrink=0.65,
        label="Mean occupancy fraction",
    )

    figure.colorbar(
        image_diff,
        ax=axes[:, 2],
        shrink=0.65,
        label="Mean occupancy difference",
    )

    figure.suptitle(
        f"Workload {workload}: Aggregate PSUM SPAD × SRAM Occupancy",
        fontsize=15,
    )

    print("\nAggregate PSUM × SRAM Occupancy:")
    print(
        "precision,aggregate_tv,mean_seed_tv,"
        "mean_baseline_tv,n_seeds,n_baseline_pairs"
    )

    for precision, stats in statistics.items():
        print(
            f"{precision},"
            f"{stats['aggregate_tv']:.4f},"
            f"{stats['mean_seed_tv']:.4f},"
            f"{stats['mean_baseline_tv']:.4f},"
            f"{stats['n_seeds']},"
            f"{stats['n_baseline_pairs']}"
        )

    return figure, axes



def main():
    parser = argparse.ArgumentParser(
        description="Analyze and plot accelerator optimization results."
    )

    parser.add_argument(
        "--workload",
        choices=("a", "b", "c"),
        default="a",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Base output filename for saving all plots instead of displaying them.",
    )

    args = parser.parse_args()
    workload = args.workload

    # Load datasets.
    results_root = (
            REPO_ROOT
            / "corrected_seed_pairs_multiplicative_continuous_frozen"
            / f"workload_{workload}"
    )

    accel_datasets = load_datasets(
        MetricType.ACCELERATOR,
        results_root=results_root,
    )

    left = load_dataframe(
        MetricType.ACCELERATOR,
        results_root=results_root,
    )

    right = load_dataframe(
        MetricType.SUBACCELERATOR,
        results_root=results_root,
    )

    dataset_keys = ["step", "seed", "variant"]

    right = right[
        dataset_keys
        + right.columns.difference(left.columns).tolist()
        ]

    subaccel_datasets = left.merge(
        right,
        on=dataset_keys,
        how="left",
        validate="one_to_many",
    )

    # Print statistical summaries.
    print_wasserstein(subaccel_datasets, accepted_only=True)
    print_summary_tables(accel_datasets)
    print_paired_statistical_tests(accel_datasets)

    # Collect figures for display or saving.
    figures = {}

    # Optimization effectiveness.
    # figures["best_edp"] = plot_best_edp(
    #     accel_datasets, args.workload
    # )[0]

    figures["aggregate_best_edp"] = plot_aggregate_best_edp(
        accel_datasets, args.workload
    )[0]

    figures["paired_edp_change"] = plot_paired_edp_change(
        accel_datasets, args.workload
    )[0]

    figures["paired_convergence"] = plot_paired_seed_convergence(
        accel_datasets, args.workload
    )[0]

    figures["accuracy_vs_edp"] = plot_accuracy_vs_edp_change(
        accel_datasets, args.workload
    )[0]

    # Aggregate regional occupancy.
    figures["occupancy_pe"] = plot_aggregate_pe_occupancy(
        subaccel_datasets,
        workload.upper(),
        partition_size=2,
    )[0]

    figures["occupancy_ifmap_weights"] = (
        plot_aggregate_ifmap_weights_occupancy(
            subaccel_datasets,
            workload.upper(),
            partition_size=2,
        )[0]
    )

    figures["occupancy_psum_sram"] = (
        plot_aggregate_psum_sram_occupancy(
            subaccel_datasets,
            workload.upper(),
            partition_size=2,
        )[0]
    )

    # Regional EDP improvement frequency.
    figures["improvements_pe"] = plot_regional_edp_improvements(
        subaccel_datasets,
        workload.upper(),
        partition_size=2,
    )[0]

    figures["improvements_ifmap_weights"] = (
        plot_regional_ifmap_weights_improvements(
            subaccel_datasets,
            workload.upper(),
            partition_size=2,
        )[0]
    )

    figures["improvements_psum_sram"] = (
        plot_regional_psum_sram_improvements(
            subaccel_datasets,
            workload.upper(),
            partition_size=2,
        )[0]
    )

    # Individual seed PE occupancy.
    seeds = sorted(
        subaccel_datasets["seed"].unique(),
        key=seed_sort_key,
    )

    # for seed in seeds:
    #     for precision in (6, 7, 8):
    #         figures[f"occupancy_pe_seed_{seed}_{precision}bit"] = (
    #             plot_pe_occupancy(
    #                 subaccel_datasets,
    #                 seed=seed,
    #                 precision=precision,
    #                 partition_size=2,
    #             )[0]
    #         )
    # Save or display all figures.
    if args.output:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        for name, figure in figures.items():
            path = args.output.with_name(
                f"{args.output.stem}_{name}{args.output.suffix}"
            )

            figure.savefig(
                path,
                dpi=300,
                bbox_inches="tight",
            )

            print(f"Saved: {path}")
            plt.close(figure)

        print(f"\nSaved {len(figures)} figures.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
