"""Create descriptive summaries for the validated expanded efficiency benchmark.

The script deliberately reports distributions rather than significance tests: the
three preparation repetitions and repeated query timings characterise this
machine/configuration, not a population of independent hardware trials.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = PROJECT_ROOT / "results" / "expanded_retrieval" / "efficiency"
OUTPUT_ROOT = INPUT_ROOT / "analysis"
GROUP_COLUMNS = ["dataset", "strategy"]


def distribution_summary(frame: pd.DataFrame, value_column: str, label: str) -> pd.DataFrame:
    """Summarise observed values without implying an inferential comparison."""
    grouped = frame.groupby(GROUP_COLUMNS, observed=True)[value_column]
    return grouped.agg(
        **{
            f"{label}_observations": "count",
            f"{label}_median_seconds": "median",
            f"{label}_p10_seconds": lambda values: values.quantile(0.10),
            f"{label}_p90_seconds": lambda values: values.quantile(0.90),
            f"{label}_mean_seconds": "mean",
        }
    ).reset_index()


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small Markdown table without adding an optional dependency."""
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    manifest = json.loads((input_root / "benchmark_manifest.json").read_text(encoding="utf-8"))
    if len(manifest.get("completed_configurations", [])) != 12:
        raise ValueError("Benchmark manifest does not record all 12 configurations")
    preparation = pd.read_csv(input_root / "preparation_trials.csv")
    latency = pd.read_csv(input_root / "query_latency_trials.csv")
    expected_preparation = 12 * int(manifest["preparation_repetitions"])
    expected_warm = 12 * int(manifest["warm_repetitions"]) * int(manifest["query_sample_size"])
    expected_fresh = 12 * int(manifest["fresh_repetitions"])
    if len(preparation) != expected_preparation:
        raise ValueError(f"Expected {expected_preparation} preparation rows, found {len(preparation)}")
    warm = latency.loc[latency["mode"] == "warm"].copy()
    fresh = latency.loc[latency["mode"] == "fresh_process"].copy()
    if len(warm) != expected_warm or len(fresh) != expected_fresh:
        raise ValueError(f"Expected {expected_warm} warm and {expected_fresh} fresh rows; found {len(warm)} and {len(fresh)}")

    preparation_summary = distribution_summary(preparation, "preparation_total_seconds", "preparation")
    stable_columns = ["documents", "queries"]
    for column in stable_columns:
        counts = preparation.groupby(GROUP_COLUMNS, observed=True)[column].nunique()
        if (counts != 1).any():
            raise ValueError(f"{column} changed across preparation trials: {counts[counts != 1].to_dict()}")
    footprint = preparation.groupby(GROUP_COLUMNS, observed=True).agg(
        documents=("documents", "first"),
        queries=("queries", "first"),
        chunks_median=("chunks", "median"),
        chunks_p10=("chunks", lambda values: values.quantile(0.10)),
        chunks_p90=("chunks", lambda values: values.quantile(0.90)),
        embedding_array_bytes_median=("embedding_array_bytes", "median"),
        serialised_faiss_index_bytes_median=("serialised_faiss_index_bytes", "median"),
    ).reset_index()
    for metric in ("chunk_construction_seconds", "chunk_embedding_seconds", "index_build_seconds", "model_initialisation_seconds"):
        preparation_summary = preparation_summary.merge(
            distribution_summary(preparation, metric, metric.removesuffix("_seconds")), on=GROUP_COLUMNS, how="inner"
        )
    preparation_summary = preparation_summary.merge(footprint, on=GROUP_COLUMNS, how="inner")

    warm_total = distribution_summary(warm, "total_seconds", "warm_total")
    warm_embedding = distribution_summary(warm, "query_embedding_seconds", "warm_embedding")
    warm_search = distribution_summary(warm, "query_retrieval_seconds", "warm_search")
    warm_summary = warm_total.merge(warm_embedding, on=GROUP_COLUMNS).merge(warm_search, on=GROUP_COLUMNS)
    fresh_total = distribution_summary(fresh, "fresh_process_total_seconds", "fresh_process_total")
    fresh_setup = distribution_summary(fresh, "fresh_process_setup_seconds", "fresh_process_setup")
    fresh_summary = fresh_total.merge(fresh_setup, on=GROUP_COLUMNS)
    combined = preparation_summary.merge(warm_summary, on=GROUP_COLUMNS).merge(fresh_summary, on=GROUP_COLUMNS)

    output_root.mkdir(parents=True, exist_ok=True)
    preparation_summary.to_csv(output_root / "preparation_summary.csv", index=False)
    warm_summary.to_csv(output_root / "warm_latency_summary.csv", index=False)
    fresh_summary.to_csv(output_root / "fresh_process_summary.csv", index=False)
    combined.to_csv(output_root / "efficiency_summary.csv", index=False)
    display = combined.loc[:, [
        "dataset", "strategy", "chunks_median", "chunks_p10", "chunks_p90",
        "preparation_median_seconds", "preparation_p10_seconds", "preparation_p90_seconds",
        "serialised_faiss_index_bytes_median", "warm_total_median_seconds",
        "warm_total_p10_seconds", "warm_total_p90_seconds", "fresh_process_total_median_seconds",
        "fresh_process_total_p10_seconds", "fresh_process_total_p90_seconds",
    ]].copy()
    for column in display.columns:
        if column.endswith("_seconds"):
            display[column] = display[column].map(lambda value: f"{value:.6f}")
    (output_root / "efficiency_results_table.md").write_text(
        "# Expanded efficiency benchmark\n\n"
        "Values are observed descriptive summaries on one machine. Preparation uses three complete trials; warm-query latency uses 300 observations (100 deterministic sampled queries × 3 randomised orders); fresh-process latency uses three observations. Fresh-process timing is not hardware-cold timing.\n\n"
        + markdown_table(display) + "\n",
        encoding="utf-8",
    )
    analysis_manifest = {
        "input_manifest": str((input_root / "benchmark_manifest.json").relative_to(PROJECT_ROOT)),
        "completed_configurations": manifest["completed_configurations"],
        "preparation_rows": len(preparation),
        "warm_latency_rows": len(warm),
        "fresh_process_rows": len(fresh),
        "statistics": "median, 10th percentile, 90th percentile, and mean; descriptive only",
        "outputs": ["preparation_summary.csv", "warm_latency_summary.csv", "fresh_process_summary.csv", "efficiency_summary.csv", "efficiency_results_table.md"],
    }
    (output_root / "analysis_manifest.json").write_text(json.dumps(analysis_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote summaries for {len(combined)} dataset-strategy configurations to {output_root}")


if __name__ == "__main__":
    main()
