import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "corrected_seed_pairs"
ITERATIONS = range(126)


def load_datasets(workload, results_root=RESULTS_ROOT):
    """Load accelerator metrics for every seed and optimization variant."""
    workload_dir = Path(results_root) / f"workload_{workload}"
    datasets = {}

    for variant_dir in sorted(path for path in workload_dir.glob("*/*") if path.is_dir()):
        csv_files = sorted(variant_dir.glob("accelerator_metrics*.csv"))
        if not csv_files:
            continue
        csv_path = csv_files[0]
        seed = variant_dir.parent.name
        variant = variant_dir.name
        datasets[(seed, variant)] = pd.read_csv(csv_path)

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
        datasets.items(), key=lambda item: (int(item[0][0]), item[0][1])
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
        accepted = metrics.loc[metrics["is_accepted"].eq(True), "accuracy_loss"]

        label = "STL" if variant == "1_15_e15" else "Baseline"
        best_edp_rows.append(
            {
                "seed": seed,
                "variant": label,
                "best_edp": best["edp"],
                "iteration": int(best["step"]),
            }
        )
        accuracy_rows.append(
            {
                "seed": seed,
                "variant": label,
                "accepted_count": len(accepted),
                "average_accuracy_loss": accepted.mean(),
            }
        )

    return pd.DataFrame(best_edp_rows), pd.DataFrame(accuracy_rows)


def print_summary_tables(datasets):
    best_edp, accuracy = summarize_datasets(datasets)
    print("\nBest feasible EDP per run:")
    print(best_edp.to_string(index=False, formatters={"best_edp": "{:.6e}".format}))
    print("\nAverage accuracy loss across accepted rows:")
    print(
        accuracy.to_string(
            index=False,
            formatters={"average_accuracy_loss": "{:.4f}".format},
        )
    )


def plot_best_edp(datasets, workload):
    figure, axis = plt.subplots(figsize=(12, 7))
    seeds = sorted({seed for seed, _ in datasets})
    colors = {
        seed: plt.get_cmap("tab10")(index % 10)
        for index, seed in enumerate(seeds)
    }

    for (seed, variant), metrics in sorted(datasets.items()):
        label = f"seed {seed} ({'STL' if variant == '1_15_e15' else 'baseline'})"
        best = best_feasible_edp(metrics)
        axis.plot(
            best.index,
            best.values,
            color=colors[seed],
            linestyle=":" if variant == "1_15_e15" else "-",
            label=label,
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
    styles = {
        "baseline": {"label": "Baseline", "color": "tab:blue", "linestyle": "-"},
        "1_15_e15": {"label": "STL", "color": "tab:orange", "linestyle": ":"},
    }

    for variant, style in styles.items():
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

    axis.set_title(f"Workload {workload.upper()}: Aggregate Best Feasible EDP")
    axis.set_xlabel("Annealing iteration")
    axis.set_ylabel("Best cumulative EDP")
    axis.set_xlim(0, 125)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure, axis


def main():
    parser = argparse.ArgumentParser(description="Plot cumulative best EDP results.")
    parser.add_argument("--workload", choices=("a", "b"), default="a")
    parser.add_argument("--output", type=Path, help="Save the plot instead of displaying it.")
    args = parser.parse_args()

    datasets = load_datasets(args.workload)
    print_summary_tables(datasets)
    figure, _ = plot_best_edp(datasets, args.workload)
    aggregate_figure, _ = plot_aggregate_best_edp(datasets, args.workload)
    if args.output:
        figure.savefig(args.output, dpi=300)
        aggregate_output = args.output.with_name(
            f"{args.output.stem}_aggregate{args.output.suffix}"
        )
        aggregate_figure.savefig(aggregate_output, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
