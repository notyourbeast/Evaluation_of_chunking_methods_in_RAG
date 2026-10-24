# Expanded Efficiency Benchmark Protocol

## Scope

This protocol measures the resource characteristics of the four expanded-study chunking configurations. It does not treat generated chunk count as a substitute for latency, memory, or index size.

## Unit and controls

Each dataset-strategy preparation trial starts from the same normalised BEIR input files and versioned retrieval configuration. A trial measures four sequential stages: sentence processing and chunk construction, chunk embedding, FAISS index construction, and index serialisation. The hardware, package versions, thread settings, configuration hash, and random seed are recorded with every trial.

Each query-latency trial uses a prebuilt index and the same embedding model. Query order is a deterministic random permutation, independent of qrels and retrieval outcomes. Query embedding, exact FAISS search, document-score aggregation, and total query time are measured separately. A fixed, deterministic random sample of at most 100 queries per collection is shared across strategies and used for repeated latency trials; the sampled identifiers are saved with the results.

## Warm and fresh-process measurements

Warm trials run after an untimed warm-up query using already-loaded model and index objects. Fresh-process trials run in a new Python process, loading the saved model and FAISS index before measuring one randomly selected query. Fresh-process measurements are labelled as such; they are not claimed to flush operating-system file caches or represent a hardware-cold start.

## Repetitions and randomisation

The default design uses three preparation trials, three warm query-order repetitions, and three fresh-process trials for each dataset-strategy configuration. Dataset-strategy configuration blocks are shuffled with a fixed benchmark seed before execution; the three preparation trials within a block are consecutive. If resource constraints require fewer repetitions, the actual counts and seed must be reported and the result described as a reduced benchmark.

## Reported outcomes

- chunk construction time;
- chunk embedding time;
- FAISS index-build time;
- serialised index size and embedding-array bytes;
- generated chunk count and chunk-token-length distribution;
- warm query embedding, search, aggregation, and total latency distributions;
- fresh-process setup and first-query latency distributions.

Results must report median, 10th percentile, 90th percentile, and all trial-level values. Differences are descriptive unless an additional pre-specified paired timing analysis is run.

## Boundary metadata and segmentation checks

The preparation pass exports metadata for every generated chunk: stable chunk/document identifiers, token count, SHA-256 of chunk text, source-unit spans, and boundary reasons. It exports deterministic examples for semantic threshold boundaries, topic-label-change boundaries, and chunks containing BERTopic outlier label `-1` when present. Example selection follows corpus order and never uses retrieval success.

Semantic sentence embeddings use the study's MiniLM model with L2-normalised vectors; a semantic boundary is eligible when adjacent-sentence cosine similarity is below 0.75 and the current chunk meets the 100-token minimum. Topic-aware chunking fits BERTopic to sentence segments from the full dataset corpus, with the fixed seed and UMAP/HDBSCAN settings in `config/expanded_retrieval_v1.json`. Because repeated full-corpus fits were not always identical, the retrieval study persists sentence-to-topic labels with sentence hashes and reuses the validated assignment cache for any rerun, metadata export, or representative-segment inspection. Outlier label `-1` is retained as a valid label and can trigger a topic-label-change boundary under the same rule as any other topic label.
