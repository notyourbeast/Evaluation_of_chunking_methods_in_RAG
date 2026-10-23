"""Create thesis figures from validated expanded-retrieval analysis outputs.

The figures are descriptive displays of the already-computed results. They do
not rerun retrieval or statistical inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "results" / "expanded_retrieval" / "analysis"
FULL_RUNS_DIR = ROOT / "results" / "expanded_retrieval" / "full"
OUTPUT_DIR = ANALYSIS_DIR / "figures"

DATASETS = ("scifact", "nfcorpus", "arguana")
STRATEGIES = ("fixed", "sentence", "semantic", "topic")
STRATEGY_LABELS = {
    "fixed": "Fixed-size",
    "sentence": "Sentence-based",
    "semantic": "Semantic",
    "topic": "Topic-aware",
}
DATASET_LABELS = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "arguana": "ArguAna"}
COLOURS = {
    "fixed": "#4C78A8",
    "sentence": "#F58518",
    "semantic": "#54A24B",
    "topic": "#B279A2",
}


def _require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")


def read_descriptive_metrics() -> pd.DataFrame:
    source = ANALYSIS_DIR / "descriptive_metrics.csv"
    frame = pd.read_csv(source)
    _require_columns(frame, {"dataset", "strategy", "k", "metric", "mean"}, source)
    expected_rows = len(DATASETS) * len(STRATEGIES) * 4 * 5
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} descriptive rows, found {len(frame)}")
    return frame


def read_significant_comparisons() -> pd.DataFrame:
    source = ANALYSIS_DIR / "pairwise_comparisons.csv"
    frame = pd.read_csv(source)
    _require_columns(
        frame,
        {
            "dataset",
            "k",
            "metric",
            "strategy_a",
            "strategy_b",
            "mean_paired_difference_a_minus_b",
            "bootstrap_ci_95_low",
            "bootstrap_ci_95_high",
            "holm_adjusted_p_value",
            "significant_at_0_05_holm",
        },
        source,
    )
    frame = frame.loc[frame["significant_at_0_05_holm"]].copy()
    if len(frame) != 9 or set(frame["dataset"]) != {"nfcorpus"}:
        raise ValueError("Expected exactly nine Holm-significant NFCorpus comparisons")
    return frame


def read_indexing_footprint() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset in DATASETS:
        for strategy in STRATEGIES:
            source = FULL_RUNS_DIR / dataset / strategy / "run_manifest.json"
            with source.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            counts = manifest["counts"]
            documents = int(counts["documents"])
            chunks = int(counts["chunks"])
            if documents <= 0 or chunks <= 0:
                raise ValueError(f"Invalid document/chunk count in {source}")
            records.append(
                {
                    "dataset": dataset,
                    "strategy": strategy,
                    "documents": documents,
                    "chunks": chunks,
                    "chunks_per_document": chunks / documents,
                }
            )
    return pd.DataFrame.from_records(records)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def save_figure(figure: plt.Figure, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / filename
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def create_ndcg_figure(descriptive: pd.DataFrame) -> Path:
    subset = descriptive.loc[descriptive["metric"] == "ndcg"].copy()
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)

    for axis, dataset in zip(axes, DATASETS, strict=True):
        data = subset.loc[subset["dataset"] == dataset]
        for strategy in STRATEGIES:
            series = data.loc[data["strategy"] == strategy].sort_values("k")
            axis.plot(
                series["k"],
                series["mean"],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                color=COLOURS[strategy],
                label=STRATEGY_LABELS[strategy],
            )
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Rank cut-off (k)")
        axis.set_xticks((1, 3, 5, 10))
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean nDCG")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    figure.suptitle("Document-level nDCG by collection, cut-off, and chunking strategy", y=1.03)
    figure.subplots_adjust(bottom=0.24, wspace=0.1)
    return save_figure(figure, "figure1_ndcg_by_collection.png")


def create_footprint_figure(footprint: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    x_positions = range(len(STRATEGIES))

    for axis, dataset in zip(axes, DATASETS, strict=True):
        data = footprint.loc[footprint["dataset"] == dataset].set_index("strategy")
        values = [data.loc[strategy, "chunks_per_document"] for strategy in STRATEGIES]
        bars = axis.bar(
            x_positions,
            values,
            color=[COLOURS[strategy] for strategy in STRATEGIES],
            width=0.68,
        )
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.05,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xticks(list(x_positions), ["Fixed", "Sentence", "Semantic", "Topic"], rotation=20)
        axis.set_ylim(0, 3.7)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Generated chunks per source document")
    figure.suptitle("Indexing footprint by collection and chunking strategy", y=1.03)
    figure.subplots_adjust(wspace=0.1)
    return save_figure(figure, "figure2_indexing_footprint.png")


def create_significance_figure(significant: pd.DataFrame) -> Path:
    metric_order = {"precision": 0, "recall": 1, "ndcg": 2}
    data = significant.copy()
    data["metric_order"] = data["metric"].map(metric_order)
    data = data.sort_values(["metric_order", "k", "strategy_b"]).reset_index(drop=True)
    labels = [
        f"{row.metric}@{row.k}: fixed - {row.strategy_b}"
        for row in data.itertuples(index=False)
    ]
    colours = ["#4C78A8" if metric == "precision" else "#F58518" if metric == "recall" else "#54A24B" for metric in data["metric"]]
    positions = list(range(len(data)))[::-1]
    figure, axis = plt.subplots(figsize=(9.2, 4.8))

    for position, row, colour in zip(positions, data.itertuples(index=False), colours, strict=True):
        estimate = row.mean_paired_difference_a_minus_b
        low = row.bootstrap_ci_95_low
        high = row.bootstrap_ci_95_high
        axis.errorbar(
            estimate,
            position,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color=colour,
            capsize=3,
            markersize=5.5,
            linewidth=1.4,
        )
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Mean paired difference (fixed-size minus comparator)")
    axis.set_title("Holm-significant NFCorpus comparisons with 95% bootstrap intervals")
    axis.grid(axis="x", alpha=0.25)
    axis.text(
        0.01,
        -0.18,
        "Points left of zero favour the comparator on the reported metric. Three recall intervals cross zero;\n"
        "their significance derives from the paired Wilcoxon test after Holm correction.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.36, bottom=0.22)
    return save_figure(figure, "figure3_holm_significant_comparisons.png")


def main() -> None:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for PNG figures (default: expanded analysis figures directory).",
    )
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir.resolve()

    apply_style()
    outputs = [
        create_ndcg_figure(read_descriptive_metrics()),
        create_footprint_figure(read_indexing_footprint()),
        create_significance_figure(read_significant_comparisons()),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
