# RAG Chunking Strategy Evaluation Study

## Research Objective

Investigating how document chunking strategies influence Retrieval-Augmented Generation (RAG) performance.

## Current Stage

Week 1 completed:
- Environment setup
- Local LLM deployment
- Dataset preparation
- Validation pipeline

## Current Dataset

Wikipedia:
- 4 articles (pipeline validation corpus)

Natural Questions:
- 200 evaluation queries

## Planned Work

- Build baseline RAG pipeline
- Compare chunking strategies
- Evaluate using RAGAS metrics

## RAGAS evaluation

`scripts/evaluate_ragas.py` evaluates generated answers with faithfulness,
answer relevancy, context precision, and context recall. It uses the local
`qwen2.5:7b` Ollama model by default and records the question ID alongside
each metric so results are traceable.

Start Ollama, then run a one-row smoke test before a full strategy evaluation:

```bash
ollama serve
conda run -n rag_chunking python scripts/evaluate_ragas.py \
  --strategy fixed --limit 1
```

Run the same command without `--limit` for each strategy (`fixed`, `sentence`,
`semantic`, and `topic`). A rerun replaces that strategy's previous rows in
`results/ragas_results.csv` and regenerates `results/ragas_summary.csv`.
The evaluator stops on missing scores by default; use
`--allow-partial-results` only when investigating a failure, not for final
comparisons.
