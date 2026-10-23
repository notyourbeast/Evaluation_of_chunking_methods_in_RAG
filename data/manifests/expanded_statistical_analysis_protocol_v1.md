# Expanded Retrieval Statistical Analysis Protocol

## Inputs and unit of analysis

This analysis uses only the validated full-run `per_query_metrics.csv` files
from the expanded BEIR retrieval evaluation. The unit of analysis is a query
within a single dataset and retrieval cutoff. Results are never pooled across
datasets for hypothesis tests.

## Planned comparisons

For each dataset (SciFact, NFCorpus, and ArguAna), cutoff (1, 3, 5, and 10),
and metric (hit rate, recall, precision, reciprocal rank, and nDCG), compare
all six unordered pairs of the four chunking strategies. This produces
3 x 4 x 5 x 6 = 360 paired tests.

The reported paired difference is `strategy_a - strategy_b`, so positive
values favour `strategy_a` for the specified metric.

## Descriptive estimates and uncertainty

For every strategy, report its mean metric value over the dataset's queries.
For every comparison, report the mean paired difference and a 95% percentile
bootstrap confidence interval based on 10,000 resamples of paired query
differences using fixed seed `20261022`.

## Hypothesis test and effect size

Use a two-sided Wilcoxon signed-rank test on the paired query-level metric
differences, with Pratt zero handling. Pratt handling retains zero differences
in rank assignment while excluding them from the signed-rank sum, which is
appropriate for bounded retrieval metrics with many ties. The normal
approximation is used consistently because sample sizes differ across datasets
and can be large.

Report the signed rank-biserial correlation, calculated from ranks of the
non-zero absolute paired differences. It ranges from -1 to 1 and its sign
matches the comparison direction.

## Multiplicity and interpretation

Apply Holm's family-wise error correction across all 360 planned tests. Report
both unadjusted and Holm-adjusted p-values. Statistical significance is based
on adjusted p < 0.05. A non-significant result is described as insufficient
evidence of a difference under this design; it is not evidence of equivalent
retrieval performance.
