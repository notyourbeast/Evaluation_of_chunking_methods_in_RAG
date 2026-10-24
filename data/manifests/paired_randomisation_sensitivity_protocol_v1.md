# Paired Randomisation Sensitivity Protocol

## Purpose

This sensitivity analysis checks whether conclusions based on paired Wilcoxon signed-rank tests depend materially on rank-based testing when query-level retrieval outcomes are discrete and contain many zero differences.

## Test

For every already reported within-collection strategy comparison, the query-level paired differences are tested with a two-sided Monte Carlo sign-flip randomisation test of the mean difference. Under the sharp null, each paired difference has an independently exchangeable sign. The observed absolute mean difference is compared with 20,000 randomly sign-flipped means. The finite-simulation p-value is `(extreme + 1) / (20,000 + 1)`.

The test is not called exact because the full `2^n` sign space is not enumerated. It is a robustness/sensitivity analysis rather than a replacement for the pre-specified Wilcoxon analysis.

## Families and reporting

The primary retrieval family includes every collection, cutoff (1, 3, 5, and 10), metric (Hit Rate, Recall, Precision, Reciprocal Rank, and nDCG), and unordered pair of the four strategies: 360 tests. The common-token-budget family includes every collection, budget (512 and 1,024 embedding-token units), metric, and unordered strategy pair: 180 tests. Holm adjustment is applied separately within each already defined family.

Results report the observed mean paired difference, the number of non-zero differences, Monte Carlo p-value, Holm-adjusted p-value, resample count, and seed. Because of simulation resolution, very small p-values should be interpreted as bounded by the configured resampling precision rather than as exact probabilities.
