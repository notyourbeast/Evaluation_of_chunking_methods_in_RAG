# Retrieved-Token-Budget Evaluation Protocol

## Purpose

This evaluation compares retrieval effectiveness when every strategy is given the same retrieved-context token allowance. It addresses the fact that top-`k` document retrieval can expose different amounts of chunk text across chunking configurations.

## Retrieval and cost rule

For each query, chunk vectors are ranked by normalised inner-product score. Each document is represented by its highest-scoring retrieved chunk, as in the primary document-ranking evaluation. Documents are ordered by that representative score. For a token budget `B`, the retrieved set is the longest strict rank-preserving prefix for which the cumulative token counts of the representative chunks do not exceed `B`. If the next ranked document would exceed the budget, selection stops; lower-ranked documents are not substituted.

The evaluated budgets are 512 and 1,024 embedding-model tokenizer tokens. The token cost is the actual token count of the representative chunk, excluding special tokens, not a generator-specific tokenizer count.

## Controls and outcomes

Corpus inputs, eligible queries, official BEIR document-level qrels, MiniLM embeddings, normalisation, exact FAISS index, document aggregation, and ArguAna query-document exclusion match the expanded retrieval study. Metrics are Hit Rate, Recall, Precision, Reciprocal Rank, and nDCG. Precision is relevant retrieved documents divided by the number of documents admitted under the budget; nDCG uses that admitted depth.

## Interpretation

This is a document-level retrieval-cost comparison, not an answer-generation or generator-context experiment. Results remain within collection; collections are not pooled.
