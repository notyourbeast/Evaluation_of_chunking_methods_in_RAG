# Thesis replacement map for the expanded retrieval study

## Purpose

This map compares the current Word thesis with the completed expanded evaluation. It specifies what must be replaced, retained, or removed if the paper adopts the retrieval-focused scope. It is a planning document only; it does not edit the Word file.

## Evidence boundary for the revised paper

The revised main study evaluates four chunking configurations on document-level retrieval effectiveness and indexing footprint. Its primary evidence consists of three BEIR collections, 17,490 documents, 2,024 evaluable test queries, four retrieval cut-offs, full descriptive results, and dataset-specific paired inference. It does not establish generated-answer correctness, latency differences, memory use, or monetary cost.

## Required section-level changes

| Current location | Current framing | Required action | Revised evidence or wording to use |
|---|---|---|---|
| Title | “Impact of Chunking Strategies on Retrieval Quality in Retrieval-Augmented Generation (RAG) Systems” | Replace | **Evaluating Chunking Strategies for Dense Retrieval Across Heterogeneous Benchmark Collections**. This avoids claiming end-to-end generation effects. |
| Abstract | Reports 50 Wikipedia documents, 30 Natural Questions questions, answer-string metrics, Bonferroni tests, context volume, and latency. | Replace in full | Describe the three BEIR collections, four strategies, shared dense-retrieval setup, document-level metrics at ranks 1/3/5/10, and dataset-specific Holm-adjusted paired analysis. State that nine adjusted comparisons were significant, all on NFCorpus; avoid naming a universal winner. Mention indexing footprint descriptively. |
| Introduction, background paragraphs | General RAG/chunking background and motivation. | Retain with light revision | Retain grounded background and literature review. Remove or soften any statement implying that the study evaluates downstream generated answers or generation cost. |
| Introduction, aim/objectives/RQs paragraph | 50-document Wikipedia design; answer-string metrics; RQ2 generated-answer quality; RQ3 context volume/cost. | Replace in full | Use the aim, four objectives, and two RQs in `revised_research_scope_draft.md`. Remove the RAGAS objective and generated-answer-quality RQ. |
| Methods 2.1 Research design | Same 50 documents / 30 questions / k=5 / answer-string rule. | Replace in full | Describe a controlled, within-query comparison conducted separately for SciFact, NFCorpus, and ArguAna. State the shared MiniLM, normalised embeddings, FAISS exact inner-product search, max-chunk-score document aggregation, and ranks 1/3/5/10. |
| Methods 2.2 Data sources | Wikipedia corpus and Natural Questions first-100 filtering protocol. | Replace in full | Describe the official BEIR releases; full corpora; supplied test queries; inclusion of test queries with at least one relevant document present; no first-N or outcome-based selection. State 17,490 documents and 2,024 evaluable queries, plus the documented five unavailable ArguAna qrel document IDs. |
| Methods 2.3 Chunking configurations | Baseline-only chunk totals and parameter descriptions. | Revise | Retain parameter descriptions only where they match the expanded configuration. Replace baseline chunk totals with the expanded cross-collection counts, or move those counts to Results. State that fixed-size uses 256 tokens with 32-token overlap; all observed chunks were within the validated maximum budget. |
| Methods 2.4 Embedding/indexing/metrics | k=5 chunk ranking and answer-string containment relevance. | Replace in full | Define document-level relevance from official qrels; hit rate, recall, precision, reciprocal rank, and nDCG at 1/3/5/10; retrieve chunk candidates then rank parent documents by their maximum chunk score. Do not describe answer-string containment. |
| Methods 2.5 Statistical analysis | 18 Bonferroni retrieval tests, six context-token tests, failed RAGAS. | Replace in full | State paired query-level comparisons, six unordered strategy pairs, 360 tests across datasets/metrics/cut-offs, Wilcoxon signed-rank with Pratt zero handling, 10,000-resample percentile bootstrap CIs, signed rank-biserial effect sizes, and Holm correction over the complete family. State that collections were analysed separately and non-significance is not equivalence. |
| Results opening and Table 1 | Baseline mean metrics, retrieved context tokens, and single-run latency. | Replace in full | Insert the complete 48-row descriptive table from `descriptive_all_cutoffs.csv`, formatted as a thesis table or split by dataset for readability. Do not retain baseline means as the main study results. |
| Results 3.1 Retrieval effectiveness | Baseline Precision@5, Answer Hit Rate@5, MRR, and 13 estimable tests. | Replace in full | Use the validated retrieval Results draft. Report descriptive patterns by dataset and the nine Holm-significant NFCorpus findings; retain the confidence-interval qualification for semantic-versus-fixed recall results whose mean-difference intervals cross zero. |
| Results 3.2 Context size | Baseline retrieved-context token analysis. | Replace | Present the expanded indexing-footprint table: generated chunks, chunks per document, and token-length distributions. Do not call this retrieval-context volume or computational cost. |
| Results 3.3 Latency and answer-level evaluation | Single-run latency values and untested RQ2/RAGAS. | Remove as a main Results subsection | Do not substitute partial RAGAS outputs for an answer-quality result. At most, mention the exploratory RAGAS work in a limitations note or appendix, with parsing failures and partial-results handling stated explicitly. |
| Figures 3 and 4 | Baseline metric/context/latency figures and Bonferroni test chart. | Retire and replace | Recreate only figures supported by expanded results, such as per-dataset nDCG by strategy/cut-off and an indexing-footprint comparison. Do not retain a latency panel or baseline significance figure. |
| Discussion | Concludes no retrieval effect and a confirmed context-volume effect in the baseline. | Replace in full | Use `retrieval_results_discussion_draft.md`: dataset-specific effects, no universal winner, all adjusted findings on NFCorpus, non-significance is not equivalence, document-level qrel limitation, and no answer-correctness inference. |
| Conclusion | Summarises the 50-document study, context volume, latency, and failed RAGAS. | Replace in full | Summarise the three-collection retrieval study, the conditional NFCorpus finding, the absence of a universal winner, and indexing-footprint differences. Exclude answer-quality and latency conclusions. |
| References | Existing RAG/chunking literature and BEIR citation. | Review and extend | Retain only sources actually cited after revision. Ensure correct citations for BEIR and each collection used (SciFact, NFCorpus, ArguAna), plus the selected embedding model and retrieval tools if required by the dissertation style. Retain RAGAS only if it remains in an appendix or limitations statement. |
| Reproducibility / appendix | No expanded-study protocol or run manifests described. | Add | Provide the expanded corpus/query protocol, configuration, run manifests, analysis protocol, seed values, results tables, and repository/commit reference where permitted. |

## Statements that must not survive into the revised main narrative

- “50 processed Wikipedia documents” or “30 evaluation questions” as the main evaluation dataset.
- Natural Questions first-100 filtering and the answer-string containment relevance rule.
- Claims based on Precision@5, Answer Hit Rate@5, or MRR from the baseline as the main result.
- The conclusion that no retrieval difference was statistically confirmed in the revised expanded study.
- Claims about causal latency differences, retrieval-context volume, input-token cost, memory use, or monetary cost.
- Any RQ or conclusion claiming generated-answer quality, correctness, relevance, or faithfulness.
- The statement that RAGAS “failed” as though it were a null finding; if mentioned, report its actual incomplete/partial status only.

## Materials ready for final integration

- `revised_research_scope_draft.md`: title, aim, objectives, and RQs.
- `retrieval_results_discussion_draft.md`: Results and Discussion prose.
- `descriptive_all_cutoffs.csv`: complete retrieval table source.
- `holm_significant_comparisons.csv`: adjusted-significance table source.
- `indexing_footprint_table.md`: measured RQ2 table and cautious interpretation.
- `expanded_statistical_analysis_protocol_v1.md`: inference protocol.

## Final integration order

1. Replace the title, Abstract, aim, objectives, and RQs as one consistent batch.
2. Replace Methods before Results so the metrics and statistical claims are defined before they appear.
3. Insert replacement tables and regenerated figures.
4. Replace Discussion and Conclusion so every claim maps to the new results.
5. Review citations, appendices, terminology, cross-references, and all remaining baseline numbers before producing the final Word version.
