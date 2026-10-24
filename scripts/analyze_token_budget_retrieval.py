"""Analyse retrieval effectiveness under common retrieved-token budgets."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = PROJECT_ROOT / "results" / "expanded_retrieval" / "token_budget_v1" / "full"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "expanded_retrieval" / "token_budget_v1" / "analysis"
DATASETS = ("scifact", "nfcorpus", "arguana")
STRATEGIES = ("fixed", "sentence", "semantic", "topic")
BUDGETS = (512, 1024)
METRICS = ("hit_rate", "recall", "precision", "reciprocal_rank", "ndcg")
BOOTSTRAP_RESAMPLES = 10_000
SEED = 20261024


def load_rows(path: Path) -> dict[tuple[int, str], dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[tuple[int, str], dict[str, float]] = {}
    for row in rows:
        key = (int(row["token_budget"]), row["query_id"])
        if key in output:
            raise ValueError(f"Duplicate query/budget row in {path}: {key}")
        output[key] = {
            "retrieved_document_count": float(row["retrieved_document_count"]),
            "retrieved_token_count": float(row["retrieved_token_count"]),
            **{metric: float(row[metric]) for metric in METRICS},
        }
    return output


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    estimates = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for start in range(0, BOOTSTRAP_RESAMPLES, 250):
        count = min(250, BOOTSTRAP_RESAMPLES - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


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
    for position, index in enumerate(order):
        value = min(1.0, (len(p_values) - position) * p_values[index])
        previous = max(previous, value)
        adjusted[index] = previous
    return [float(value) for value in adjusted]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    matrix: dict[tuple[str, str, int, str], dict[str, float]] = {}
    for dataset in DATASETS:
        for strategy in STRATEGIES:
            path = args.input_root / dataset / strategy / "per_query_token_budget_metrics.csv"
            for (budget, query_id), row in load_rows(path).items():
                matrix[(dataset, strategy, budget, query_id)] = row

    rng = np.random.default_rng(SEED)
    descriptive_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for budget in BUDGETS:
            query_sets = {
                strategy: {
                    query_id for source_dataset, source_strategy, source_budget, query_id in matrix
                    if (source_dataset, source_strategy, source_budget) == (dataset, strategy, budget)
                }
                for strategy in STRATEGIES
            }
            query_ids = sorted(query_sets[STRATEGIES[0]])
            if not query_ids or any(ids != set(query_ids) for ids in query_sets.values()):
                raise ValueError(f"Unpaired or missing queries for {dataset} at {budget} tokens")
            for strategy in STRATEGIES:
                values = [matrix[(dataset, strategy, budget, query_id)] for query_id in query_ids]
                for measure in ("retrieved_document_count", "retrieved_token_count", *METRICS):
                    array = np.array([row[measure] for row in values])
                    descriptive_rows.append({
                        "dataset": dataset,
                        "strategy": strategy,
                        "token_budget": budget,
                        "measure": measure,
                        "query_count": len(query_ids),
                        "mean": float(array.mean()),
                        "standard_deviation": float(array.std(ddof=1)),
                    })
            for strategy_a, strategy_b in combinations(STRATEGIES, 2):
                for metric in METRICS:
                    a = np.array([matrix[(dataset, strategy_a, budget, query_id)][metric] for query_id in query_ids])
                    b = np.array([matrix[(dataset, strategy_b, budget, query_id)][metric] for query_id in query_ids])
                    differences = a - b
                    nonzero = int(np.count_nonzero(differences))
                    if nonzero:
                        test = wilcoxon(differences, zero_method="pratt", alternative="two-sided", method="approx")
                        statistic, p_value = float(test.statistic), float(test.pvalue)
                    else:
                        statistic, p_value = 0.0, 1.0
                    ci_low, ci_high = bootstrap_mean_ci(differences, rng)
                    comparison_rows.append({
                        "dataset": dataset,
                        "token_budget": budget,
                        "metric": metric,
                        "strategy_a": strategy_a,
                        "strategy_b": strategy_b,
                        "query_count": len(query_ids),
                        "nonzero_paired_differences": nonzero,
                        "mean_a": float(a.mean()),
                        "mean_b": float(b.mean()),
                        "mean_paired_difference_a_minus_b": float(differences.mean()),
                        "bootstrap_ci_95_low": ci_low,
                        "bootstrap_ci_95_high": ci_high,
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                        "rank_biserial_correlation": signed_rank_biserial(differences),
                    })
    for row, adjusted_p in zip(comparison_rows, holm_adjust([float(row["p_value"]) for row in comparison_rows])):
        row["holm_adjusted_p_value"] = adjusted_p
        row["significant_at_0_05_holm"] = adjusted_p < 0.05

    args.output_root.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("descriptive_metrics.csv", descriptive_rows), ("pairwise_comparisons.csv", comparison_rows)):
        with (args.output_root / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "protocol": "data/manifests/retrieved_token_budget_protocol_v1.md",
        "input_root": str(args.input_root.relative_to(PROJECT_ROOT)),
        "datasets": list(DATASETS),
        "strategies": list(STRATEGIES),
        "budgets": list(BUDGETS),
        "metrics": list(METRICS),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": SEED,
        "pairwise_tests": len(comparison_rows),
        "p_value_adjustment": "Holm across all 180 planned token-budget pairwise tests",
    }
    (args.output_root / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(descriptive_rows)} descriptive rows and {len(comparison_rows)} paired comparisons to {args.output_root}")


if __name__ == "__main__":
    main()
