# Draft replacement for title aim objectives and research questions

## Scope decision

This version treats the study as an evaluation of retrieval effectiveness and indexing footprint. It does not pose or answer a generated-answer-quality research question. The existing RAGAS outputs may be retained only as exploratory supplementary material, with their failed parses and partial-result status disclosed; they must not support the main conclusions.

## Proposed title

**Evaluating Chunking Strategies for Dense Retrieval Across Heterogeneous Benchmark Collections**

## Proposed aim

This study evaluates how fixed-size, sentence-based, semantic, and topic-aware chunking configurations affect document-level retrieval effectiveness and indexing footprint across heterogeneous benchmark collections in a controlled dense-retrieval pipeline.

## Proposed objectives

1. Apply four chunking configurations to the SciFact, NFCorpus, and ArguAna collections under a shared embedding model, similarity search method, document-ranking rule, and evaluation protocol.
2. Measure document-level retrieval effectiveness at ranks 1, 3, 5, and 10 using hit rate, recall, precision, reciprocal rank, and nDCG.
3. Compare strategies within each collection using paired query-level inference, bootstrap confidence intervals, effect sizes, and Holm correction for the stated multiple-comparison family.
4. Describe indexing footprint using generated chunk count, chunks per document, and chunk-token-length distributions.

## Proposed research questions

**RQ1. How do fixed-size, sentence-based, semantic, and topic-aware chunking configurations affect document-level retrieval effectiveness across the SciFact, NFCorpus, and ArguAna collections under a controlled dense-retrieval pipeline?**

**RQ2. How do these chunking configurations affect indexing footprint, measured by the number and token-length distribution of generated chunks, across the evaluated collections?**

## Required removals or replacements in the current thesis

- Remove the current RQ2 on generated-answer quality; do not renumber it as an answer-quality question.
- Replace the current RQ3 on retrieved-context volume and computational cost with the proposed RQ2 only after removing claims of latency, memory use, or monetary cost unless those outcomes are measured in a controlled expanded experiment.
- Replace phrases such as “answer-string retrieval effectiveness” with “document-level retrieval effectiveness,” because the expanded BEIR evaluation uses official document relevance judgements.
- Replace the stated 50-document Wikipedia / 30-question main-study framing with the expanded three-collection design. The original experiment can remain as an exploratory pilot or development-stage experiment, clearly labelled as such.
