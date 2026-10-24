# Fixed-Overlap Ablation Protocol

## Purpose

This ablation isolates the effect of the 32-token overlap used in the expanded study's fixed-size configuration. It compares the existing `fixed` configuration (256 embedding-model tokens, 32-token overlap) with `fixed_no_overlap` (256 embedding-model tokens, zero overlap).

## Controls

The corpus inputs, eligible queries, official BEIR document-level qrels, MiniLM embedding model, normalisation, exact FAISS `IndexFlatIP` index, document-score aggregation, query exclusion rule, cutoffs, and random seed are unchanged. The only intended chunking difference is overlap.

## Interpretation

This test estimates the effect of overlap within the fixed-window method. It does not identify a causal effect of semantic or topic boundaries, because those complete configurations still differ in realised chunk lengths and segmentation rules. Results are analysed within collection with paired query-level outcomes; collections are not pooled.
