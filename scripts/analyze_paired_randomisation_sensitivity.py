"""Run paired sign-flip randomisation sensitivity analyses for retrieval results."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "expanded_retrieval"
OUTPUT_ROOT = RESULTS_ROOT / "randomisation_sensitivity_v1"
DATASETS = ("scifact", "nfcorpus", "arguana")
STRATEGIES = ("fixed", "sentence", "semantic", "topic")
METRICS = ("hit_rate", "recall", "precision", "reciprocal_rank", "ndcg")
PRIMARY_CUTOFFS = (1, 3, 5, 10)
TOKEN_BUDGETS = (512, 1024)
DEFAULT_RESAMPLES = 20_000
SEED = 20261024


def read_rows(path: Path, condition_column: str) -> dict[tuple[int, str], dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[tuple[int, str], dict[str, float]] = {}
    for row in rows:
        key = (int(row[condition_column]), row["query_id"])
        if key in output:
            raise ValueError(f"Duplicate query/condition row in {path}: {key}")
        output[key] = {metric: float(row[metric]) for metric in METRICS}
    return output


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    previous = 0.0
    for position, index in enumerate(order):
        value = min(1.0, (len(p_values) - position) * p_values[index])
        previous = max(previous, value)
        adjusted[index] = previous
    return [float(value) for value in adjusted]


def sign_flip_p_value(differences: np.ndarray, resamples: int, seed_sequence: np.random.SeedSequence) -> float:
    observed = abs(float(differences.mean()))
    if not np.any(differences) or observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed_sequence)
    extreme = 0
    batch_size = 500
    for start in range(0, resamples, batch_size):
        count = min(batch_size, resamples - start)
        signs = rng.integers(0, 2, size=(count, len(differences)), dtype=np.int8)
        signed_means = ((signs * 2 - 1) @ differences) / len(differences)
        extreme += int(np.count_nonzero(np.abs(signed_means) >= observed - 1e-15))
    return (extreme + 1) / (resamples + 1)


def build_family(
    name: str,
    root: Path,
    filename: str,
    condition_column: str,
    conditions: tuple[int, ...],
    resamples: int,
    seed: int,
) -> list[dict[str, object]]:
    matrix: dict[tuple[str, str, int, str], dict[str, float]] = {}
    for dataset in DATASETS:
        for strategy in STRATEGIES:
            for (condition, query_id), values in read_rows(root / dataset / strategy / filename, condition_column).items():
                matrix[(dataset, strategy, condition, query_id)] = values
    rows: list[dict[str, object]] = []
    test_index = 0
    for dataset in DATASETS:
        for condition in conditions:
            query_sets = {
                strategy: {
                    query_id for source_dataset, source_strategy, source_condition, query_id in matrix
                    if (source_dataset, source_strategy, source_condition) == (dataset, strategy, condition)
                }
                for strategy in STRATEGIES
            }
            query_ids = sorted(query_sets[STRATEGIES[0]])
            if not query_ids or any(ids != set(query_ids) for ids in query_sets.values()):
                raise ValueError(f"Unpaired or missing queries for {name}, {dataset}, condition={condition}")
            for strategy_a, strategy_b in combinations(STRATEGIES, 2):
                for metric in METRICS:
                    a = np.array([matrix[(dataset, strategy_a, condition, query_id)][metric] for query_id in query_ids])
                    b = np.array([matrix[(dataset, strategy_b, condition, query_id)][metric] for query_id in query_ids])
                    differences = a - b
                    p_value = sign_flip_p_value(
                        differences, resamples, np.random.SeedSequence([seed, 1 if name == "primary_retrieval" else 2, test_index])
                    )
                    rows.append({
                        "family": name,
                        "dataset": dataset,
                        "condition_name": condition_column,
                        "condition": condition,
                        "metric": metric,
                        "strategy_a": strategy_a,
                        "strategy_b": strategy_b,
                        "query_count": len(query_ids),
                        "nonzero_paired_differences": int(np.count_nonzero(differences)),
                        "mean_paired_difference_a_minus_b": float(differences.mean()),
                        "randomisation_p_value": p_value,
                        "resamples": resamples,
                        "base_seed": seed,
                    })
                    test_index += 1
            print(f"Completed {name}: {dataset}, {condition_column}={condition}", flush=True)
    for row, adjusted_p in zip(rows, holm_adjust([float(row["randomisation_p_value"]) for row in rows])):
        row["holm_adjusted_p_value"] = adjusted_p
        row["significant_at_0_05_holm"] = adjusted_p < 0.05
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    args = parser.parse_args()
    if args.resamples <= 0:
        parser.error("--resamples must be positive")
    primary = build_family(
        "primary_retrieval", RESULTS_ROOT / "full", "per_query_metrics.csv", "k", PRIMARY_CUTOFFS, args.resamples, SEED
    )
    token_budget = build_family(
        "token_budget", RESULTS_ROOT / "token_budget_v1" / "full", "per_query_token_budget_metrics.csv", "token_budget", TOKEN_BUDGETS,
        args.resamples, SEED,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_root / "primary_retrieval_randomisation.csv", primary)
    write_rows(args.output_root / "token_budget_randomisation.csv", token_budget)
    manifest = {
        "protocol": "data/manifests/paired_randomisation_sensitivity_protocol_v1.md",
        "test": "two-sided Monte Carlo paired sign-flip randomisation test of the mean paired difference",
        "resamples": args.resamples,
        "base_seed": SEED,
        "families": {"primary_retrieval": len(primary), "token_budget": len(token_budget)},
        "holm_adjustment": "Applied separately within each family.",
    }
    (args.output_root / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(primary)} primary and {len(token_budget)} token-budget randomisation comparisons to {args.output_root}")


if __name__ == "__main__":
    main()
