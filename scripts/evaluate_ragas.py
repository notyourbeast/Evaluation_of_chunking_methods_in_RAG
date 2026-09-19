"""Evaluate generated RAG answers with RAGAS 0.1.x.

The script deliberately fails when RAGAS cannot score a row. A CSV containing
missing metric values looks complete, but is not a valid strategy comparison.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

os.environ["RAGAS_MAX_WORKERS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "generation_results.json"
DEFAULT_OUTPUT = ROOT / "results" / "ragas_results.csv"
DEFAULT_SUMMARY = ROOT / "results" / "ragas_summary.csv"
METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["fixed", "sentence", "semantic", "topic"], required=True)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N rows (for a smoke test).")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--evaluator-model", default="qwen2.5:7b")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--num-predict", type=int, default=256,
        help="Maximum evaluator tokens for the local evaluator (default: 256).",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Maximum seconds for one RAGAS evaluator request (default: 300).",
    )
    parser.add_argument(
        "--max-workers", type=int, default=1,
        help="Concurrent RAGAS evaluator requests (default: 1 for local Ollama).",
    )
    parser.add_argument(
        "--llm-retries", type=int, default=1,
        help="RAGAS retries for a stalled evaluator call (default: 1).",
    )
    parser.add_argument(
        "--missing-score-retries", type=int, default=0,
        help="Retries for an individual metric that returned no parsable score (default: 0).",
    )
    parser.add_argument(
        "--max-repair-rows", type=int, default=5,
        help="Do not run individual retries when more than this many rows are missing (default: 5).",
    )
    parser.add_argument(
        "--metrics", nargs="+", choices=list(METRICS), default=list(METRICS),
        help="Metrics to calculate; defaults to all four study metrics.",
    )
    parser.add_argument(
        "--allow-partial-results", action="store_true",
        help="Write rows with missing scores after evaluator failures. Avoid this for final experiments.",
    )
    return parser.parse_args()


def load_dataset(input_path, strategy, limit):
    with input_path.open(encoding="utf-8") as handle:
        source = json.load(handle)
    rows = []
    for item in source:
        if item.get("strategy") != strategy:
            continue
        required = ("question_id", "question", "answer", "contexts", "ground_truth")
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"Missing {missing} in a {strategy!r} generation row")
        if not isinstance(item["contexts"], list) or not item["contexts"]:
            raise ValueError(f"Empty or invalid contexts for question_id={item['question_id']}")
        rows.append({key: item[key] for key in required})
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No rows found for strategy={strategy!r}")
    print(f"Evaluating {len(rows)} {strategy} row(s).")
    return Dataset.from_list(rows)


def build_evaluators(model_name, embedding_model, num_predict):
    return (
        Ollama(
            model=model_name,
            temperature=0,
            timeout=1800,
            num_predict=num_predict,
            keep_alive="30m",
        ),
        HuggingFaceEmbeddings(model_name=embedding_model),
    )


def update_summary(result_path, summary_path):
    results = pd.read_csv(result_path)
    metric_columns = [name for name in METRICS if name in results.columns]
    summary = results.groupby("strategy", as_index=False)[metric_columns].agg(["count", "mean", "std"])
    summary.columns = ["strategy" if not column[0] else "_".join(filter(None, column)) for column in summary.columns]
    summary.to_csv(summary_path, index=False)


def retry_missing_scores(dataset, result, metric_columns, llm, embeddings, run_config, attempts, max_rows):
    """Retry only rows/metrics for which RAGAS returned NaN (often parser failures)."""
    for attempt in range(1, attempts + 1):
        missing_rows = result.index[result[metric_columns].isna().any(axis=1)].tolist()
        if not missing_rows:
            return
        if len(missing_rows) > max_rows:
            print(
                f"Skipping individual retries: {len(missing_rows)} rows are missing scores "
                f"(limit: {max_rows})."
            )
            return
        print(f"Retry {attempt}/{attempts}: repairing missing scores in {len(missing_rows)} row(s).")
        for row_index in missing_rows:
            missing_metrics = [name for name in metric_columns if pd.isna(result.at[row_index, name])]
            row_dataset = Dataset.from_list([dataset[row_index]])
            repaired = evaluate(
                row_dataset,
                metrics=[METRICS[name] for name in missing_metrics],
                llm=llm,
                embeddings=embeddings,
                run_config=run_config,
                raise_exceptions=False,
            ).to_pandas()
            for metric in missing_metrics:
                score = repaired.at[0, metric]
                if pd.notna(score):
                    result.at[row_index, metric] = score


def main():
    args = parse_args()
    dataset = load_dataset(args.input, args.strategy, args.limit)
    llm, embeddings = build_evaluators(
        args.evaluator_model, args.embedding_model, args.num_predict
    )
    print(f"Evaluator model: {args.evaluator_model}")
    print(f"Metrics: {', '.join(args.metrics)}")
    run_config = RunConfig(
        timeout=args.timeout,
        max_workers=args.max_workers,
        max_retries=args.llm_retries,
        max_wait=10,
    )
    result = evaluate(
        dataset,
        metrics=[METRICS[name] for name in args.metrics],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=False,
    ).to_pandas()
    result.insert(0, "strategy", args.strategy)
    metric_columns = list(args.metrics)
    retry_missing_scores(
        dataset,
        result,
        metric_columns,
        llm,
        embeddings,
        run_config,
        args.missing_score_retries,
        args.max_repair_rows,
    )
    missing = result[metric_columns].isna().sum()
    if missing.any() and not args.allow_partial_results:
        partial_output = args.output.with_name(
            f"{args.output.stem}_{args.strategy}_partial{args.output.suffix}"
        )
        partial_output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(partial_output, index=False)
        print(f"Saved recoverable partial results: {partial_output}")
        raise RuntimeError(f"RAGAS returned missing scores: {missing[missing > 0].to_dict()}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing = pd.read_csv(args.output)
        # A rerun replaces this strategy, rather than mixing incompatible runs.
        existing = existing[existing["strategy"] != args.strategy]
        result = pd.concat([existing, result], ignore_index=True, sort=False)
    result.to_csv(args.output, index=False)
    update_summary(args.output, args.summary)
    print(f"Saved detailed results: {args.output}")
    print(f"Saved summary: {args.summary}")
    print(result[result["strategy"] == args.strategy][metric_columns].mean().to_string())


if __name__ == "__main__":
    main()
