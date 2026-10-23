# Captions for expanded retrieval figures

**Figure 1. Document-level nDCG by collection, rank cut-off, and chunking strategy.** Mean nDCG is shown for the SciFact, NFCorpus, and ArguAna test collections at ranks 1, 3, 5, and 10. Lines are descriptive; inferential claims are based on the paired, Holm-adjusted analyses reported separately.

**Figure 2. Indexing footprint by collection and chunking strategy.** Bars show the number of generated chunks divided by the number of source documents for each full run. This is an observed indexing-footprint measure and does not measure runtime, memory use, retrieval latency, or monetary cost.

**Figure 3. Holm-significant NFCorpus comparisons with 95% bootstrap intervals.** Points show the mean paired difference for fixed-size chunking minus the stated comparator; horizontal bars show 95% percentile-bootstrap intervals. All displayed comparisons were significant in paired Wilcoxon signed-rank tests after Holm correction across the full family of 360 tests. Three recall intervals cross zero, so they should not be interpreted as unambiguous non-zero mean differences.
