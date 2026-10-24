"""Analyse the preregistered fixed-window overlap ablation within each BEIR collection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = PROJECT_ROOT / "results" / "expanded_retrieval" / "full"
ABLATION_ROOT = PROJECT_ROOT / "results" / "expanded_retrieval" / "ablations" / "fixed_overlap_v1" / "full"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "expanded_retrieval" / "ablations" / "fixed_overlap_v1" / "analysis"
DATASETS = ("scifact", "nfcorpus", "arguana")
METRICS = ("hit_rate", "recall", "precision", "reciprocal_rank", "ndcg")
CUTOFFS = (1, 3, 5, 10)
BOOTSTRAP_RESAMPLES = 10_000
SEED = 20261024


def read_rows(path: Path) -> dict[tuple[int, str], dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[tuple[int, str], dict[str, float]] = {}
    for row in rows:
        key = (int(row["k"]), row["query_id"])
        if key in output:
            raise ValueError(f"Duplicate query/cutoff row in {path}: {key}")
        output[key] = {metric: float(row[metric]) for metric in METRICS}
    return output


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    estimates = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for start in range(0, BOOTSTRAP_RESAMPLES, 250):
        count = min(250, BOOTSTRAP_RESAMPLES - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


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
    parser.add_argument("--base-root", type=Path, default=BASE_ROOT)
    parser.add_argument("--ablation-root", type=Path, default=ABLATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        baseline = read_rows(args.base_root / dataset / "fixed" / "per_query_metrics.csv")
        no_overlap = read_rows(args.ablation_root / dataset / "fixed_no_overlap" / "per_query_metrics.csv")
        if set(baseline) != set(no_overlap):
            raise ValueError(f"Unpaired baseline/ablation rows for {dataset}")
        for cutoff in CUTOFFS:
            query_ids = sorted(query_id for source_cutoff, query_id in baseline if source_cutoff == cutoff)
            for metric in METRICS:
                baseline_values = np.array([baseline[(cutoff, query_id)][metric] for query_id in query_ids])
                no_overlap_values = np.array([no_overlap[(cutoff, query_id)][metric] for query_id in query_ids])
                differences = no_overlap_values - baseline_values
                nonzero = int(np.count_nonzero(differences))
                if nonzero:
                    test = wilcoxon(differences, zero_method="pratt", alternative="two-sided", method="approx")
                    statistic, p_value = float(test.statistic), float(test.pvalue)
                else:
                    statistic, p_value = 0.0, 1.0
                ci_low, ci_high = bootstrap_mean_ci(differences, rng)
                rows.append({
                    "dataset": dataset,
                    "k": cutoff,
                    "metric": metric,
                    "query_count": len(query_ids),
                    "mean_fixed_overlap_32": float(baseline_values.mean()),
                    "mean_fixed_no_overlap": float(no_overlap_values.mean()),
                    "mean_difference_no_overlap_minus_overlap_32": float(differences.mean()),
                    "bootstrap_ci_95_low": ci_low,
                    "bootstrap_ci_95_high": ci_high,
                    "nonzero_paired_differences": nonzero,
                    "wilcoxon_statistic": statistic,
                    "p_value_unadjusted": p_value,
                })
    for row, adjusted_p in zip(rows, holm_adjust([float(row["p_value_unadjusted"]) for row in rows])):
        row["holm_adjusted_p_value"] = adjusted_p
        row["significant_at_0_05_holm"] = adjusted_p < 0.05
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "overlap_ablation_comparisons.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "protocol": "data/manifests/fixed_overlap_ablation_protocol_v1.md",
        "base_configuration": "config/expanded_retrieval_v1.json",
        "ablation_configuration": "config/fixed_overlap_ablation_v1.json",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": SEED,
        "comparisons": len(rows),
        "p_value_adjustment": "Holm across all 60 planned ablation comparisons",
        "interpretation": "no-overlap minus 32-token-overlap; within-dataset paired comparisons only",
    }
    (args.output_root / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} overlap-ablation comparisons to {args.output_root}")


if __name__ == "__main__":
    main()
