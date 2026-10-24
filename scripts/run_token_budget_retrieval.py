"""Evaluate expanded retrieval configurations at common retrieved-token budgets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from run_beir_retrieval import (
    CONFIG_PATH,
    DATASETS,
    PROJECT_ROOT,
    RESULTS_ROOT,
    STRATEGIES,
    TOPIC_ASSIGNMENT_CACHE_ROOT,
    create_chunks,
    load_config,
    load_dataset,
    package_versions,
    sha256,
)


OUTPUT_ROOT = RESULTS_ROOT / "token_budget_v1"
BUDGETS = (512, 1024)


def budget_metrics(ranked: list[str], qrels: dict[str, int]) -> dict[str, float]:
    positives = {document_id for document_id, score in qrels.items() if score > 0}
    if not positives:
        raise ValueError("Query has no positive qrel")
    grades = [qrels.get(document_id, 0) for document_id in ranked]
    relevant = sum(grade > 0 for grade in grades)
    first_rank = next((rank for rank, grade in enumerate(grades, 1) if grade > 0), None)
    dcg = sum((2**grade - 1) / np.log2(rank + 1) for rank, grade in enumerate(grades, 1))
    ideal = sorted(qrels.values(), reverse=True)[: len(ranked)]
    idcg = sum((2**grade - 1) / np.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    return {
        "hit_rate": float(relevant > 0),
        "recall": relevant / len(positives),
        "precision": 0.0 if not ranked else relevant / len(ranked),
        "reciprocal_rank": 0.0 if first_rank is None else 1.0 / first_rank,
        "ndcg": 0.0 if idcg == 0 else dcg / idcg,
    }


def representative_ranking(
    dataset: str, query_id: str, scores: np.ndarray, indices: np.ndarray, chunks: list[dict[str, Any]]
) -> list[tuple[str, int]]:
    representatives: dict[str, int] = {}
    for index_id in indices:
        chunk = chunks[int(index_id)]
        document_id = chunk["document_id"]
        if dataset == "arguana" and document_id == query_id:
            continue
        if document_id not in representatives:
            representatives[document_id] = int(chunk["token_count"])
    return list(representatives.items())


def strict_budget_prefix(ranked: list[tuple[str, int]], budget: int) -> tuple[list[str], list[int]]:
    document_ids: list[str] = []
    token_counts: list[int] = []
    used = 0
    for document_id, token_count in ranked:
        if used + token_count > budget:
            break
        document_ids.append(document_id)
        token_counts.append(token_count)
        used += token_count
    return document_ids, token_counts


def write_outputs(output_dir: Path, rows: list[dict[str, object]], manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "strategy", "query_id", "token_budget", "retrieved_document_count", "retrieved_token_count",
        "retrieved_document_ids", "representative_chunk_token_counts", "hit_rate", "recall", "precision", "reciprocal_rank", "ndcg",
    ]
    with (output_dir / "per_query_token_budget_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for budget in BUDGETS:
        subset = [row for row in rows if row["token_budget"] == budget]
        summary.append({
            "token_budget": budget,
            "query_count": len(subset),
            "mean_retrieved_document_count": float(np.mean([row["retrieved_document_count"] for row in subset])),
            "mean_retrieved_token_count": float(np.mean([row["retrieved_token_count"] for row in subset])),
            **{metric: float(np.mean([row[metric] for row in subset])) for metric in ("hit_rate", "recall", "precision", "reciprocal_rank", "ndcg")},
        })
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run(dataset: str, strategy: str, output_root: Path, overwrite: bool) -> Path:
    config = load_config(CONFIG_PATH)
    documents, queries, qrels, source_manifest = load_dataset(dataset)
    output_dir = output_root / "full" / dataset / strategy
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; use --overwrite to replace it")
        shutil.rmtree(output_dir)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["embedding_model"], local_files_only=True)
    tokenizer.model_max_length = 1_000_000
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer(config["embedding_model"], local_files_only=True)
    model.max_seq_length = config["chunking"][strategy]["target_tokens"] + model.tokenizer.num_special_tokens_to_add(pair=False)
    topic_assignment_provenance: dict[str, Any] = {}
    topic_cache_path = None
    if strategy == "topic":
        topic_cache_path = TOPIC_ASSIGNMENT_CACHE_ROOT / f"{dataset}_{sha256(CONFIG_PATH)[:16]}.csv"
        if not topic_cache_path.exists():
            raise FileNotFoundError(f"Missing frozen topic assignments: {topic_cache_path}")
    started = __import__("time").monotonic()
    chunks = create_chunks(
        strategy, documents, tokenizer, nlp, model, config,
        topic_assignment_cache_path=topic_cache_path,
        topic_assignment_provenance=topic_assignment_provenance,
    )
    chunk_embeddings = model.encode([chunk["text"] for chunk in chunks], batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype("float32")
    import faiss

    index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)
    rows: list[dict[str, object]] = []
    for query in queries:
        embedding = model.encode([query["text"]], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, indices = index.search(embedding, index.ntotal)
        ranked = representative_ranking(dataset, query["query_id"], scores[0], indices[0], chunks)
        for budget in BUDGETS:
            document_ids, token_counts = strict_budget_prefix(ranked, budget)
            rows.append({
                "dataset": dataset,
                "strategy": strategy,
                "query_id": query["query_id"],
                "token_budget": budget,
                "retrieved_document_count": len(document_ids),
                "retrieved_token_count": sum(token_counts),
                "retrieved_document_ids": json.dumps(document_ids),
                "representative_chunk_token_counts": json.dumps(token_counts),
                **budget_metrics(document_ids, qrels[query["query_id"]]),
            })
    manifest = {
        "protocol": "data/manifests/retrieved_token_budget_protocol_v1.md",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "strategy": strategy,
        "config_path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
        "config_sha256": sha256(CONFIG_PATH),
        "source_archive_sha256": source_manifest["source"]["sha256"],
        "budgets": list(BUDGETS),
        "budget_rule": "strict rank-preserving prefix of document representative chunks",
        "counts": {"documents": len(documents), "queries": len(queries), "chunks": len(chunks), "per_query_budget_rows": len(rows)},
        "topic_assignment_provenance": topic_assignment_provenance if strategy == "topic" else None,
        "environment": {"python": sys.version, "platform": sys.platform, "packages": package_versions()},
        "elapsed_seconds": __import__("time").monotonic() - started,
    }
    write_outputs(output_dir, rows, manifest)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--strategy", choices=[*STRATEGIES, "all"], default="all")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)
    for dataset in datasets:
        for strategy in strategies:
            print(f"Completed {dataset}/{strategy}: {run(dataset, strategy, args.output_root, args.overwrite)}")


if __name__ == "__main__":
    main()
