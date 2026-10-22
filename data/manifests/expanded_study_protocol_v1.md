# Expanded Study Protocol

## Purpose

This protocol replaces the baseline study's 50-document Wikipedia corpus and
30 corpus-filtered Natural Questions queries for the final retrieval study. It
is defined before data ingestion and evaluation to prevent outcome-driven
dataset selection.

## Research scope

The expanded study evaluates four complete chunking configurations—fixed-size,
sentence-based, semantic, and topic-aware—across three English-language
retrieval benchmarks with published relevance judgements. It tests retrieval
performance across scientific, biomedical, and argument-retrieval domains. It
does not claim that the results generalise to every RAG corpus or generator.

## Benchmark sources

| Dataset | Retrieval task and domain | Expected corpus size | Expected test queries | Official BEIR archive MD5 |
| --- | --- | ---: | ---: | --- |
| SciFact | Scientific claim retrieval | 5,183 | 300 | `5f7d1de60b170fc8027bb7898e2efca1` |
| NFCorpus | Biomedical and nutrition information retrieval | 3,633 | 323 | `a89dba18a62ef92f7d323ec890a0d38d` |
| ArguAna | Counterargument retrieval | 8,674 | 1,401 evaluable of 1,406 supplied | `8ad3e3c2a5867cdced806d6503f29b99` |

The implementation must verify the downloaded archive checksum and record its
URL, retrieval date, archive checksum, and loaded document/query/qrel counts
in a dataset manifest. The source artefacts are the BEIR public test splits;
the expected counts above are validation targets, not replacement data.

## Corpus inclusion rules

1. Load the complete corpus supplied with each BEIR dataset. Do not retain only
   documents linked to a query's relevance judgement.
2. Retain every non-empty document and construct its retrieval text from title
   and body text, separated by two newline characters when both are present.
3. Preserve the original corpus document identifier, query identifier, source
   dataset, title, and text in a normalised data file.
4. For ArguAna, remove a query's source document from that query's candidate
   set when its identifier is present in the corpus, following the benchmark's
   documented self-match precaution.

## Query inclusion rules

1. Use every query in each dataset's official test split that has at least one
   positive relevance judgement whose relevant document is present in that
   dataset's supplied corpus archive.
2. Do not select the first N records, filter queries by exact answer-string
   occurrence, or discard queries because a baseline retriever fails them.
3. Preserve the complete qrels file. For binary measures, a qrel score greater
   than zero is relevant. For graded metrics such as nDCG, preserve the
   original qrel grade.
4. Record loaded, excluded, and eligible query counts by dataset. Any
   exclusion must have an explicit rule and an identifier-level audit trail.

### Source-integrity exception discovered during ingestion

The official ArguAna archive contains five test qrels whose target document
identifiers are absent from its supplied corpus. These qrels are retained
unchanged in the normalised qrels file and their identifiers are recorded in
the ArguAna manifest. Because an absent document cannot be retrieved, the five
affected queries are excluded only from metric denominators; 1,401 of 1,406
ArguAna test queries remain evaluable. This exception is a property of the
downloaded source archive, not a researcher-selected exclusion.

## Controlled retrieval configuration

The expanded retrieval experiment keeps the following controls identical across
all chunking configurations within each dataset:

- embedding model: `sentence-transformers/all-MiniLM-L6-v2`;
- vector index: FAISS `IndexFlatIP` with L2-normalised embeddings;
- initial retrieval depths: k = 1, 3, 5, and 10;
- document text preprocessing and title/body concatenation rule;
- query text, qrels, and document candidate set.

Chunking parameters, seeds, package versions, hardware details, and generated
chunk metadata must be captured in a run manifest. Topic modelling and any
randomised operation must use the fixed seed `20261021`.

## Planned analyses

The final analysis will report per-dataset and pooled results separately. It
will not pool benchmark scores without also showing the dataset-level values.
For every configuration pair and metric, it will report paired effect sizes,
bootstrap confidence intervals for the paired difference, test statistics,
unadjusted p-values, adjusted p-values, and the count of usable paired
observations. A non-significant result will be described as a failure to detect
a difference under the specified design, not as evidence of equivalence.

## Deferred components

This protocol defines data and retrieval evaluation only. Answer-level RAGAS
evaluation, common retrieved-token budgets, controlled overlap/length
ablations, manual relevance adjudication, and repeated efficiency benchmarking
are implemented in later revision steps. Their protocols must be added before
their respective experiments run.
