"""Benchmark expanded-retrieval preparation and query latency reproducibly.

Use the protocol in data/manifests/expanded_efficiency_benchmark_protocol_v1.md.
The default full matrix is expensive on CPU-only systems; validate with one
dataset/strategy and reduced repetitions before launching it under caffeinate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Must precede NumPy, spaCy, PyTorch, and FAISS imports. This mirrors the
# validated expanded-retrieval runner's macOS OpenMP safeguards.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from run_beir_retrieval import CONFIG_PATH, DATASETS, PROJECT_ROOT, STRATEGIES, create_chunks, load_config, load_dataset, sha256


OUTPUT_ROOT = PROJECT_ROOT / "results" / "expanded_retrieval" / "efficiency"
SEED = 20261023


def now_seconds() -> float:
    return time.perf_counter()


def write_chunk_metadata(path: Path, chunks: list[dict[str, Any]]) -> None:
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["chunk_id", "document_id", "token_count", "text_sha256", "boundary_metadata_json"], lineterminator="\n")
        writer.writeheader()
        for chunk in chunks:
            writer.writerow({
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "token_count": chunk["token_count"],
                "text_sha256": hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest(),
                "boundary_metadata_json": json.dumps(chunk["boundary_metadata"], sort_keys=True),
            })


def rank_documents(index: Any, document_ids: list[str], embedding: np.ndarray) -> None:
    scores, indices = index.search(embedding, index.ntotal)
    best: dict[str, float] = {}
    for score, index_id in zip(scores[0], indices[0]):
        best.setdefault(document_ids[int(index_id)], float(score))
    sorted(best.items(), key=lambda item: item[1], reverse=True)


def prepare(dataset: str, strategy: str, config: dict[str, Any]) -> tuple[dict[str, Any], Any, list[str], SentenceTransformer, list[dict[str, Any]]]:
    # Match the retrieval runner's controls before every measured preparation
    # trial, including BERTopic/UMAP's use of NumPy's global RNG.
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    started = now_seconds()
    documents, queries, _, source_manifest = load_dataset(dataset)
    tokenizer = AutoTokenizer.from_pretrained(config["embedding_model"], local_files_only=True)
    tokenizer.model_max_length = 1_000_000
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer(config["embedding_model"], local_files_only=True)
    model.max_seq_length = config["chunking"][strategy]["target_tokens"] + model.tokenizer.num_special_tokens_to_add(pair=False)
    model_ready = now_seconds()
    chunks = create_chunks(strategy, documents, tokenizer, nlp, model, config)
    chunks_ready = now_seconds()
    embeddings = model.encode([chunk["text"] for chunk in chunks], batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype("float32")
    embeddings_ready = now_seconds()
    # Match the validated retrieval runner: first MiniLM inference precedes
    # FAISS import to avoid a macOS OpenMP shared-memory abort.
    import faiss

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    index_bytes = len(faiss.serialize_index(index))
    index_ready = now_seconds()
    return ({
        "dataset": dataset,
        "strategy": strategy,
        "documents": len(documents),
        "queries": len(queries),
        "chunks": len(chunks),
        "model_initialisation_seconds": model_ready - started,
        "chunk_construction_seconds": chunks_ready - model_ready,
        "chunk_embedding_seconds": embeddings_ready - chunks_ready,
        "index_build_seconds": index_ready - embeddings_ready,
        "preparation_total_seconds": index_ready - started,
        "embedding_array_bytes": int(embeddings.nbytes),
        "serialised_faiss_index_bytes": index_bytes,
        "source_archive_sha256": source_manifest["source"]["sha256"],
    }, index, [chunk["document_id"] for chunk in chunks], model, queries, chunks)


def fresh_once(index_path: Path, document_ids_path: Path, query_path: Path) -> dict[str, float]:
    query = json.loads(query_path.read_text(encoding="utf-8"))
    started = now_seconds()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    embedding_started = now_seconds()
    embedding = model.encode([query["text"]], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    embedding_ready = now_seconds()
    # Import FAISS only after the first MiniLM inference. Importing it first
    # triggers an OpenMP shared-memory abort in this macOS conda environment.
    import faiss

    index = faiss.read_index(str(index_path))
    document_ids = json.loads(document_ids_path.read_text(encoding="utf-8"))
    index_ready = now_seconds()
    rank_documents(index, document_ids, embedding)
    finished = now_seconds()
    return {
        "query_id": query["query_id"],
        "fresh_process_setup_seconds": index_ready - started - (embedding_ready - embedding_started),
        "query_embedding_seconds": embedding_ready - embedding_started,
        "query_retrieval_seconds": finished - embedding_ready,
        "fresh_process_total_seconds": finished - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--strategy", choices=[*STRATEGIES, "all"], default="all")
    parser.add_argument("--preparation-repetitions", type=int, default=3)
    parser.add_argument("--warm-repetitions", type=int, default=3)
    parser.add_argument("--fresh-repetitions", type=int, default=3)
    parser.add_argument("--query-sample-size", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--fresh-process-once", action="store_true")
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--document-ids-path", type=Path)
    parser.add_argument("--query-path", type=Path)
    args = parser.parse_args()

    if args.fresh_process_once:
        if not all((args.index_path, args.document_ids_path, args.query_path)):
            parser.error("fresh-process mode requires --index-path, --document-ids-path, and --query-path")
        print(json.dumps(fresh_once(args.index_path, args.document_ids_path, args.query_path)))
        return
    if min(args.preparation_repetitions, args.warm_repetitions, args.query_sample_size) <= 0 or args.fresh_repetitions < 0:
        parser.error("Preparation/warm repetitions and sample size must be positive; fresh repetitions may be zero")

    output_root = args.output_root.resolve()
    config = load_config(CONFIG_PATH)
    preparation_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)
    configurations = [(dataset, strategy) for dataset in datasets for strategy in strategies]
    order_rng = np.random.default_rng(SEED)
    order_rng.shuffle(configurations)
    sampled_query_ids: dict[str, list[str]] = {}
    for index, dataset in enumerate(datasets):
        _, queries, _, _ = load_dataset(dataset)
        if len(queries) < args.query_sample_size:
            raise ValueError(f"{dataset} has fewer queries than requested sample size")
        sample_rng = np.random.default_rng(SEED + index)
        selected = sample_rng.choice(len(queries), size=args.query_sample_size, replace=False)
        sampled_query_ids[dataset] = [queries[int(query_index)]["query_id"] for query_index in selected]

    def write_checkpoint(completed_configurations: list[str]) -> None:
        """Persist completed trial records so an interrupted run remains auditable."""
        output_root.mkdir(parents=True, exist_ok=True)
        for name, rows in (("preparation_trials.csv", preparation_rows), ("query_latency_trials.csv", query_rows)):
            with (output_root / name).open("w", encoding="utf-8", newline="") as handle:
                if not rows:
                    continue
                writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        (output_root / "benchmark_manifest.json").write_text(json.dumps({
            "protocol": "data/manifests/expanded_efficiency_benchmark_protocol_v1.md",
            "benchmark_seed": SEED,
            "config_sha256": sha256(CONFIG_PATH),
            "preparation_repetitions": args.preparation_repetitions,
            "warm_repetitions": args.warm_repetitions,
            "fresh_repetitions": args.fresh_repetitions,
            "query_sample_size": args.query_sample_size,
            "completed_configurations": completed_configurations,
        }, indent=2) + "\n", encoding="utf-8")

    completed_configurations: list[str] = []

    for configuration_index, (dataset, strategy) in enumerate(configurations):
            final_index = final_document_ids = final_model = final_queries = final_chunks = None
            for repetition in range(args.preparation_repetitions):
                row, index, document_ids, model, queries, chunks = prepare(dataset, strategy, config)
                row["preparation_repetition"] = repetition + 1
                preparation_rows.append(row)
                final_index, final_document_ids, final_model, final_queries, final_chunks = index, document_ids, model, queries, chunks
            assert final_index is not None and final_document_ids is not None and final_model is not None and final_queries is not None and final_chunks is not None
            destination = output_root / dataset / strategy
            destination.mkdir(parents=True, exist_ok=True)
            import faiss
            index_path = destination / "benchmark.index"
            document_ids_path = destination / "chunk_document_ids.json"
            faiss.write_index(final_index, str(index_path))
            document_ids_path.write_text(json.dumps(final_document_ids), encoding="utf-8")
            write_chunk_metadata(destination / "chunk_metadata.csv", final_chunks)

            queries_by_id = {query["query_id"]: query for query in final_queries}
            sampled = [queries_by_id[query_id] for query_id in sampled_query_ids[dataset]]
            (destination / "latency_query_sample.json").write_text(json.dumps(sampled, indent=2), encoding="utf-8")
            warmup = final_model.encode([sampled[0]["text"]], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
            rank_documents(final_index, final_document_ids, warmup)
            for repetition in range(args.warm_repetitions):
                query_order_rng = np.random.default_rng(SEED + configuration_index * args.warm_repetitions + repetition)
                for query in query_order_rng.permutation(sampled):
                    start = now_seconds()
                    embedding = final_model.encode([query["text"]], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
                    embedded = now_seconds()
                    rank_documents(final_index, final_document_ids, embedding)
                    finished = now_seconds()
                    query_rows.append({"dataset": dataset, "strategy": strategy, "mode": "warm", "repetition": repetition + 1, "query_id": query["query_id"], "query_embedding_seconds": embedded - start, "query_retrieval_seconds": finished - embedded, "total_seconds": finished - start})
            for repetition, query in enumerate(sampled[: args.fresh_repetitions], start=1):
                query_path = destination / f"fresh_query_{repetition}.json"
                query_path.write_text(json.dumps(query), encoding="utf-8")
                command = [sys.executable, str(Path(__file__).resolve()), "--fresh-process-once", "--index-path", str(index_path), "--document-ids-path", str(document_ids_path), "--query-path", str(query_path)]
                completed = subprocess.run(command, check=True, capture_output=True, text=True)
                values = json.loads(completed.stdout)
                query_rows.append({"dataset": dataset, "strategy": strategy, "mode": "fresh_process", "repetition": repetition, **values, "total_seconds": values["fresh_process_total_seconds"]})
            completed_configurations.append(f"{dataset}/{strategy}")
            write_checkpoint(completed_configurations)
            print(f"Completed {dataset}/{strategy}", flush=True)


if __name__ == "__main__":
    main()
