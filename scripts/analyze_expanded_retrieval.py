"""Analyse the validated expanded BEIR retrieval results without pooling datasets."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = PROJECT_ROOT / "results" / "expanded_retrieval" / "full"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "expanded_retrieval" / "analysis"
DATASETS = ("scifact", "nfcorpus", "arguana")
STRATEGIES = ("fixed", "sentence", "semantic", "topic")
METRICS = ("hit_rate", "recall", "precision", "reciprocal_rank", "ndcg")
CUTOFFS = (1, 3, 5, 10)
BOOTSTRAP_RESAMPLES = 10_000
SEED = 20261022


def load_rows(dataset: str, strategy: str) -> list[dict[str, str]]:
    path = INPUT_ROOT / dataset / strategy / "per_query_metrics.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def load_matrix() -> dict[tuple[str, str, int, str], dict[str, float]]:
    matrix: dict[tuple[str, str, int, str], dict[str, float]] = {}
    for dataset in DATASETS:
        for strategy in STRATEGIES:
            for row in load_rows(dataset, strategy):
                key = (dataset, strategy, int(row["k"]), row["query_id"])
                if key in matrix:
                    raise ValueError(f"Duplicate result row: {key}")
                matrix[key] = {metric: float(row[metric]) for metric in METRICS}
    return matrix


def bootstrap_mean_ci(differences: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile CI from paired query-level resampling, without a huge matrix."""
    size = len(differences)
    estimates = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    batch_size = 250
    for start in range(0, BOOTSTRAP_RESAMPLES, batch_size):
        count = min(batch_size, BOOTSTRAP_RESAMPLES - start)
        indices = rng.integers(0, size, size=(count, size))
        estimates[start : start + count] = differences[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(estimates, (0.025, 0.975)))


def signed_rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    previous = 0.0
    count = len(p_values)
    for position, index in enumerate(order):
        value = min(1.0, (count - position) * p_values[index])
        previous = max(previous, value)
        adjusted[index] = previous
    return [float(value) for value in adjusted]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    matrix = load_matrix()
    rng = np.random.default_rng(SEED)
    descriptive_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []

    for dataset in DATASETS:
        for cutoff in CUTOFFS:
            query_sets = {
                strategy: {
                    query_id
                    for source_dataset, source_strategy, source_cutoff, query_id in matrix
                    if source_dataset == dataset and source_strategy == strategy and source_cutoff == cutoff
                }
                for strategy in STRATEGIES
            }
            expected_queries = query_sets[STRATEGIES[0]]
            if not expected_queries or any(ids != expected_queries for ids in query_sets.values()):
                raise ValueError(f"Unpaired or missing queries for {dataset} at k={cutoff}")
            query_ids = sorted(expected_queries)
            for strategy in STRATEGIES:
                for metric in METRICS:
                    values = np.array([matrix[(dataset, strategy, cutoff, query_id)][metric] for query_id in query_ids])
                    descriptive_rows.append({
                        "dataset": dataset,
                        "strategy": strategy,
                        "k": cutoff,
                        "metric": metric,
                        "query_count": len(query_ids),
                        "mean": float(values.mean()),
                        "standard_deviation": float(values.std(ddof=1)),
                    })
            for strategy_a, strategy_b in combinations(STRATEGIES, 2):
                for metric in METRICS:
                    a = np.array([matrix[(dataset, strategy_a, cutoff, query_id)][metric] for query_id in query_ids])
                    b = np.array([matrix[(dataset, strategy_b, cutoff, query_id)][metric] for query_id in query_ids])
                    differences = a - b
                    nonzero_count = int(np.count_nonzero(differences))
                    if nonzero_count == 0:
                        statistic, p_value = 0.0, 1.0
                    else:
                        test = wilcoxon(differences, zero_method="pratt", alternative="two-sided", method="approx")
                        statistic, p_value = float(test.statistic), float(test.pvalue)
                    ci_low, ci_high = bootstrap_mean_ci(differences, rng)
                    comparison_rows.append({
                        "dataset": dataset,
                        "k": cutoff,
                        "metric": metric,
                        "strategy_a": strategy_a,
                        "strategy_b": strategy_b,
                        "query_count": len(query_ids),
                        "nonzero_paired_differences": nonzero_count,
                        "mean_a": float(a.mean()),
                        "mean_b": float(b.mean()),
                        "mean_paired_difference_a_minus_b": float(differences.mean()),
                        "bootstrap_ci_95_low": ci_low,
                        "bootstrap_ci_95_high": ci_high,
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                        "rank_biserial_correlation": signed_rank_biserial(differences),
                    })

    adjusted = holm_adjust([float(row["p_value"]) for row in comparison_rows])
    for row, adjusted_p in zip(comparison_rows, adjusted):
        row["holm_adjusted_p_value"] = adjusted_p
        row["significant_at_0_05_holm"] = adjusted_p < 0.05

    args.output_root.mkdir(parents=True, exist_ok=True)
    descriptive_fields = list(descriptive_rows[0])
    comparison_fields = list(comparison_rows[0])
    with (args.output_root / "descriptive_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=descriptive_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(descriptive_rows)
    with (args.output_root / "pairwise_comparisons.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparison_rows)
    manifest = {
        "analysis_protocol": "data/manifests/expanded_statistical_analysis_protocol_v1.md",
        "input_root": str(INPUT_ROOT.relative_to(PROJECT_ROOT)),
        "datasets": list(DATASETS),
        "strategies": list(STRATEGIES),
        "metrics": list(METRICS),
        "cutoffs": list(CUTOFFS),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": SEED,
        "pairwise_tests": len(comparison_rows),
        "p_value_adjustment": "Holm across all planned pairwise tests",
        "outputs": ["descriptive_metrics.csv", "pairwise_comparisons.csv"],
    }
    (args.output_root / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(descriptive_rows)} descriptive rows and {len(comparison_rows)} pairwise comparisons to {args.output_root}")


if __name__ == "__main__":
    main()
