# Chunking Strategies for Dense Retrieval

This repository evaluates fixed-size, sentence-based, semantic, and topic-aware chunking in a controlled dense-retrieval pipeline. The main study reports document-level retrieval effectiveness and indexing footprint; it does not make claims about generated-answer correctness.

## Expanded evaluation

The completed expanded evaluation uses the SciFact, NFCorpus, and ArguAna BEIR collections:

- 17,490 documents across the three corpora;
- 2,024 evaluable test queries with an available relevant document;
- a shared `sentence-transformers/all-MiniLM-L6-v2` embedding model and exact FAISS inner-product search;
- document ranking by the maximum score among each document's chunks;
- hit rate, recall, precision, reciprocal rank, and nDCG at ranks 1, 3, 5, and 10.

Strategies are compared within each collection using query-level paired Wilcoxon signed-rank tests, percentile-bootstrap confidence intervals, signed rank-biserial correlations, and Holm correction across the complete family of 360 comparisons. Collections are not pooled.

## Reproducing the analysis

The corpus and query-selection protocol is documented in `data/manifests/expanded_study_protocol_v1.md`. The inferential procedure is documented in `data/manifests/expanded_statistical_analysis_protocol_v1.md`.

The efficiency protocol, frozen topic-assignment approach, input/relevance audit, package versions, and final artefact checksums are included under `data/manifests/` and `results/expanded_retrieval/audits/`. Topic-model fitting was not fully deterministic for every collection; the reported topic runs therefore reuse sentence-hash-validated, persisted assignment files rather than claiming deterministic refitting.

After the expanded retrieval runs are available, generate analysis tables with:

```bash
conda run -n rag_chunking python scripts/analyze_expanded_retrieval.py
conda run -n rag_chunking python scripts/create_expanded_retrieval_tables.py
```

Generate the publication figures with:

```bash
MPLCONFIGDIR="$PWD/.cache/matplotlib" conda run -n rag_chunking \
  python scripts/create_expanded_retrieval_figures.py
```

## Results

- Full run outputs: `results/expanded_retrieval/full/`
- Descriptive and inferential outputs: `results/expanded_retrieval/analysis/`
- Reader-facing retrieval tables: `results/expanded_retrieval/analysis/thesis_tables/retrieval_results_tables.md`
- Publication figures: `results/expanded_retrieval/analysis/figures/`
- Efficiency trial summaries: `results/expanded_retrieval/efficiency_v2/analysis/`
- Segmentation length/boundary summaries and deterministic excerpts: `results/expanded_retrieval/chunk_metadata/`
- Input, environment, and artefact-checksum audits: `results/expanded_retrieval/audits/`

The RAGAS scripts and outputs are retained as exploratory material. They are not part of the main retrieval conclusions because the available runs contain parsing failures and partial-result handling.
