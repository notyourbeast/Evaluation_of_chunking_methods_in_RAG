"""Create thesis-ready tables from validated expanded retrieval analysis outputs."""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = PROJECT_ROOT / "results" / "expanded_retrieval" / "analysis"
OUTPUT_ROOT = ANALYSIS_ROOT / "thesis_tables"
DATASETS = ("scifact", "nfcorpus", "arguana")
STRATEGIES = ("fixed", "sentence", "semantic", "topic")
METRICS = ("hit_rate", "recall", "precision", "reciprocal_rank", "ndcg")
CUTOFFS = (1, 3, 5, 10)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}"


def main() -> None:
    descriptive = read_csv(ANALYSIS_ROOT / "descriptive_metrics.csv")
    comparisons = read_csv(ANALYSIS_ROOT / "pairwise_comparisons.csv")
    lookup = {(row["dataset"], row["strategy"], int(row["k"]), row["metric"]): row for row in descriptive}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    descriptive_rows = []
    markdown = [
        "# Expanded Retrieval Results Tables",
        "",
        "These tables are descriptive macro-averages over queries within each dataset. They are not pooled across datasets.",
        "",
        "## Table 1. Retrieval metrics by dataset, strategy, and cutoff",
        "",
        "| Dataset | Strategy | k | Hit rate | Recall | Precision | MRR | nDCG |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        for strategy in STRATEGIES:
            for cutoff in CUTOFFS:
                row = {"dataset": dataset, "strategy": strategy, "k": cutoff}
                values = []
                for metric in METRICS:
                    value = float(lookup[(dataset, strategy, cutoff, metric)]["mean"])
                    row[metric] = value
                    values.append(fmt(value))
                descriptive_rows.append(row)
                markdown.append(f"| {dataset} | {strategy} | {cutoff} | " + " | ".join(values) + " |")

    significant = [row for row in comparisons if row["significant_at_0_05_holm"] == "True"]
    significant.sort(key=lambda row: (row["dataset"], int(row["k"]), row["metric"], row["strategy_a"], row["strategy_b"]))
    markdown.extend([
        "",
        "## Table 2. Holm-significant paired comparisons",
        "",
        "The difference is `strategy A − strategy B`; a negative value favours strategy B. Holm correction was applied across all 360 planned tests.",
        "",
        "| Dataset | k | Metric | Strategy A | Strategy B | Mean difference | 95% bootstrap CI | Rank-biserial r | Holm p |",
        "| --- | ---: | --- | --- | --- | ---: | --- | ---: | ---: |",
    ])
    significance_rows = []
    for row in significant:
        item = {
            "dataset": row["dataset"], "k": int(row["k"]), "metric": row["metric"],
            "strategy_a": row["strategy_a"], "strategy_b": row["strategy_b"],
            "mean_difference_a_minus_b": float(row["mean_paired_difference_a_minus_b"]),
            "bootstrap_ci_95_low": float(row["bootstrap_ci_95_low"]),
            "bootstrap_ci_95_high": float(row["bootstrap_ci_95_high"]),
            "rank_biserial_correlation": float(row["rank_biserial_correlation"]),
            "holm_adjusted_p_value": float(row["holm_adjusted_p_value"]),
        }
        significance_rows.append(item)
        markdown.append(
            f"| {item['dataset']} | {item['k']} | {item['metric']} | {item['strategy_a']} | {item['strategy_b']} | "
            f"{fmt(item['mean_difference_a_minus_b'])} | [{fmt(item['bootstrap_ci_95_low'])}, {fmt(item['bootstrap_ci_95_high'])}] | "
            f"{fmt(item['rank_biserial_correlation'])} | {item['holm_adjusted_p_value']:.4g} |"
        )

    markdown.extend([
        "",
        "## Reporting note",
        "",
        "All nine Holm-significant tests are from NFCorpus. Three significant recall comparisons have bootstrap confidence intervals for the mean paired difference that cross zero. Report the Wilcoxon and bootstrap results together; do not describe those three as unequivocal mean-recall improvements. The remaining 351 planned comparisons were not Holm-significant and should be described as no statistically significant difference detected under this analysis, not as equivalent performance.",
        "",
    ])

    with (OUTPUT_ROOT / "descriptive_all_cutoffs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "strategy", "k", *METRICS], lineterminator="\n")
        writer.writeheader()
        writer.writerows(descriptive_rows)
    with (OUTPUT_ROOT / "holm_significant_comparisons.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(significance_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(significance_rows)
    (OUTPUT_ROOT / "retrieval_results_tables.md").write_text("\n".join(markdown), encoding="utf-8")
    print(f"Wrote {len(descriptive_rows)} descriptive rows and {len(significance_rows)} significant comparisons to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
