# Indexing footprint by collection and chunking configuration

The table reports directly observed output from the 12 full expanded-retrieval runs. “Chunks per document” is a descriptive ratio, calculated as total generated chunks divided by source documents. It is an indexing-footprint measure, not a direct measurement of indexing time, memory use, retrieval latency, or monetary cost.

| Collection | Documents | Strategy | Generated chunks | Chunks per document | Mean chunk tokens | Observed token range |
|---|---:|---|---:|---:|---:|---|
| SciFact | 5,183 | Fixed-size | 10,400 | 2.01 | 182.0 | 1–256 |
| SciFact | 5,183 | Sentence-based | 9,550 | 1.84 | 181.9 | 9–256 |
| SciFact | 5,183 | Semantic | 15,864 | 3.06 | 109.5 | 1–256 |
| SciFact | 5,183 | Topic-aware | 13,147 | 2.54 | 132.1 | 1–256 |
| NFCorpus | 3,633 | Fixed-size | 7,512 | 2.07 | 187.5 | 1–256 |
| NFCorpus | 3,633 | Sentence-based | 7,034 | 1.94 | 183.5 | 4–256 |
| NFCorpus | 3,633 | Semantic | 11,888 | 3.27 | 108.6 | 3–255 |
| NFCorpus | 3,633 | Topic-aware | 10,484 | 2.89 | 123.1 | 2–256 |
| ArguAna | 8,674 | Fixed-size | 12,074 | 1.39 | 156.3 | 1–256 |
| ArguAna | 8,674 | Sentence-based | 11,124 | 1.28 | 161.2 | 2–256 |
| ArguAna | 8,674 | Semantic | 18,910 | 2.18 | 94.8 | 1–228 |
| ArguAna | 8,674 | Topic-aware | 17,222 | 1.99 | 104.1 | 1–256 |

## Grounded interpretation

Semantic and topic-aware configurations generated more chunks than fixed-size and sentence-based configurations in each collection. This supports a descriptive claim about indexing footprint under the specified implementation and parameters. It does not, by itself, establish a difference in runtime, memory consumption, retrieval latency, or cost.

## Source and reproducibility

Values were transcribed from the `counts` fields in the full-run manifests under `results/expanded_retrieval/full/<collection>/<strategy>/run_manifest.json`. All runs use the shared expanded-retrieval configuration and a maximum chunk-token budget of 256 tokens.
