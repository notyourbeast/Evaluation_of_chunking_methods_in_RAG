"""Run the preregistered expanded BEIR retrieval evaluation.

The runner ranks *documents* from chunk-level retrieval scores.  A document's
score is the maximum inner-product score of any of its chunks, making the
document-level qrels applicable without using answer-string matching.  It
always writes per-query results and a run manifest; aggregate summaries are
derived from those rows.

Runs with --max-documents or --max-queries are explicitly labelled smoke runs
and must not be used in the thesis or statistical analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Numba and Matplotlib otherwise try to write into non-writable home locations
# in the managed environment. These paths also make topic runs reproducible.
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# This is an exact IndexFlatIP experiment. A single OpenMP thread avoids a
# macOS shared-memory failure observed with this conda FAISS build and keeps
# the run configuration deterministic.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


CONFIG_PATH = PROJECT_ROOT / "config" / "expanded_retrieval_v1.json"
DATA_ROOT = PROJECT_ROOT / "data" / "expanded"
RESULTS_ROOT = PROJECT_ROOT / "results" / "expanded_retrieval"
DATASETS = ("scifact", "nfcorpus", "arguana")
STRATEGIES = ("fixed", "sentence", "semantic", "topic")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path} line {line_number}") from error
    return rows


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset(dataset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]], dict[str, Any]]:
    root = DATA_ROOT / dataset
    required = [root / name for name in ("documents.jsonl", "eligible_test_queries.jsonl", "qrels_test.tsv", "manifest.json")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing normalised {dataset} inputs: {missing}")

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset"] != dataset:
        raise ValueError(f"Dataset manifest mismatch in {root / 'manifest.json'}")
    if manifest["source"]["expected_md5"] != manifest["source"]["verified_md5"]:
        raise ValueError(f"Source archive checksum was not verified for {dataset}")

    documents = read_jsonl(root / "documents.jsonl")
    queries = read_jsonl(root / "eligible_test_queries.jsonl")
    document_ids = {row["document_id"] for row in documents}
    query_ids = {row["query_id"] for row in queries}
    if len(document_ids) != len(documents) or len(query_ids) != len(queries):
        raise ValueError(f"Duplicate document or query ID in normalised {dataset} data")

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with (root / "qrels_test.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            query_id, document_id, score = row["query-id"], row["corpus-id"], int(row["score"])
            # Preserve dangling source qrels in the input file, but they cannot
            # be retrieved and therefore cannot enter an evaluable denominator.
            if query_id in query_ids and document_id in document_ids:
                qrels[query_id][document_id] = score

    for query in queries:
        if not any(score > 0 for score in qrels[query["query_id"]].values()):
            raise ValueError(f"Eligible query has no retrievable positive qrel: {query['query_id']}")
    return documents, queries, qrels, manifest


def deterministic_smoke_subset(
    documents: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    qrels: dict[str, dict[str, int]],
    max_documents: int | None,
    max_queries: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Create a labelled smoke subset while retaining each selected query's positives."""
    if max_queries is None and max_documents is None:
        return documents, queries, qrels
    selected_queries = queries[:max_queries] if max_queries is not None else queries
    required_ids = {
        document_id
        for query in selected_queries
        for document_id, score in qrels[query["query_id"]].items()
        if score > 0
    }
    limit = max_documents if max_documents is not None else len(documents)
    if limit < len(required_ids):
        raise ValueError(
            f"--max-documents={limit} is smaller than the {len(required_ids)} positive documents "
            "needed for the selected smoke queries"
        )
    selected_documents = [document for document in documents if document["document_id"] in required_ids]
    selected_ids = {document["document_id"] for document in selected_documents}
    for document in documents:
        if len(selected_documents) >= limit:
            break
        if document["document_id"] not in selected_ids:
            selected_documents.append(document)
            selected_ids.add(document["document_id"])
    selected_qrels = {
        query["query_id"]: {
            document_id: score
            for document_id, score in qrels[query["query_id"]].items()
            if document_id in selected_ids
        }
        for query in selected_queries
    }
    return selected_documents, selected_queries, selected_qrels


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_long_sentence(tokenizer: Any, sentence: str, limit: int) -> list[str]:
    words = sentence.split()
    segments: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and token_count(tokenizer, candidate) > limit:
            segments.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        segments.append(" ".join(current))
    return segments


def sentence_lists(documents: Iterable[dict[str, Any]], nlp: Any, tokenizer: Any, limit: int) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for document, parsed in zip(documents, nlp.pipe((row["text"] for row in documents), batch_size=64)):
        sentences = []
        for sentence in parsed.sents:
            clean = sentence.text.strip()
            if clean:
                sentences.extend(split_long_sentence(tokenizer, clean, limit))
        output[document["document_id"]] = sentences
    return output


def enforce_token_budget(tokenizer: Any, text: str, budget: int) -> str:
    """Return text that re-tokenises to no more than the declared budget."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    while len(token_ids) > budget:
        text = tokenizer.decode(token_ids[:budget], skip_special_tokens=True).strip()
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    return text


def make_chunk(document_id: str, strategy: str, number: int, text: str, tokenizer: Any, budget: int) -> dict[str, Any]:
    text = enforce_token_budget(tokenizer, text, budget)
    return {
        "chunk_id": f"{document_id}::{strategy}::{number:05d}",
        "document_id": document_id,
        "text": text,
        "token_count": token_count(tokenizer, text),
    }


def fixed_chunks(documents: list[dict[str, Any]], tokenizer: Any, settings: dict[str, Any]) -> list[dict[str, Any]]:
    size, overlap = settings["target_tokens"], settings["overlap_tokens"]
    chunks = []
    for document in documents:
        tokens = tokenizer.encode(document["text"], add_special_tokens=False)
        for number, start in enumerate(range(0, len(tokens), size - overlap)):
            decoded = tokenizer.decode(tokens[start : start + size], skip_special_tokens=True).strip()
            if decoded:
                chunks.append(make_chunk(document["document_id"], "fixed", number, decoded, tokenizer, size))
    return chunks


def grouped_sentence_chunks(
    documents: list[dict[str, Any]],
    tokenizer: Any,
    sentence_map: dict[str, list[str]],
    strategy: str,
    settings: dict[str, Any],
    sentence_embeddings: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    chunks = []
    target, minimum = settings["target_tokens"], settings.get("minimum_chunk_tokens", 0)
    threshold = settings.get("adjacent_sentence_similarity_threshold")
    for document in documents:
        current: list[str] = []
        current_tokens = 0
        number = 0
        sentences = sentence_map[document["document_id"]]
        embeddings = sentence_embeddings.get(document["document_id"]) if sentence_embeddings else None
        for index, sentence in enumerate(sentences):
            sentence_tokens = token_count(tokenizer, sentence)
            boundary = False
            if embeddings is not None and index > 0:
                boundary = float(np.dot(embeddings[index - 1], embeddings[index])) < threshold
            split = current and (
                current_tokens + sentence_tokens > target
                or (boundary and current_tokens >= minimum)
            )
            if split:
                chunks.append(make_chunk(document["document_id"], strategy, number, " ".join(current), tokenizer, target))
                number += 1
                current, current_tokens = [], 0
            current.append(sentence)
            current_tokens += sentence_tokens
        if current:
            chunks.append(make_chunk(document["document_id"], strategy, number, " ".join(current), tokenizer, target))
    return chunks


def semantic_chunks(
    documents: list[dict[str, Any]], tokenizer: Any, nlp: Any, model: SentenceTransformer, settings: dict[str, Any]
) -> list[dict[str, Any]]:
    sentence_map = sentence_lists(documents, nlp, tokenizer, settings["long_sentence_segment_limit_tokens"])
    all_sentences = [sentence for sentences in sentence_map.values() for sentence in sentences]
    encoded = model.encode(all_sentences, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    embeddings: dict[str, np.ndarray] = {}
    cursor = 0
    for document in documents:
        length = len(sentence_map[document["document_id"]])
        embeddings[document["document_id"]] = encoded[cursor : cursor + length]
        cursor += length
    return grouped_sentence_chunks(documents, tokenizer, sentence_map, "semantic", settings, embeddings)


def topic_chunks(
    documents: list[dict[str, Any]], tokenizer: Any, nlp: Any, model: SentenceTransformer, settings: dict[str, Any], seed: int
) -> list[dict[str, Any]]:
    # Deferred imports keep fixed/sentence/semantic runs independent of BERTopic.
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    sentence_map = sentence_lists(documents, nlp, tokenizer, settings["long_sentence_segment_limit_tokens"])
    sentence_records = [(document["document_id"], sentence) for document in documents for sentence in sentence_map[document["document_id"]]]
    if not sentence_records:
        return []
    texts = [sentence for _, sentence in sentence_records]
    embeddings = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    umap_model = UMAP(random_state=seed, **settings["umap"])
    hdbscan_model = HDBSCAN(**settings["hdbscan"])
    topics, _ = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, embedding_model=None, verbose=False).fit_transform(texts, embeddings)
    by_document: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (document_id, sentence), topic in zip(sentence_records, topics):
        by_document[document_id].append((sentence, int(topic)))
    chunks = []
    for document in documents:
        current: list[str] = []
        current_tokens, number, current_topic = 0, 0, None
        for sentence, topic in by_document[document["document_id"]]:
            sentence_tokens = token_count(tokenizer, sentence)
            split = current and (
                current_tokens + sentence_tokens > settings["target_tokens"]
                or (current_tokens >= settings["minimum_chunk_tokens"] and topic != current_topic)
            )
            if split:
                chunks.append(make_chunk(document["document_id"], "topic", number, " ".join(current), tokenizer, settings["target_tokens"]))
                number += 1
                current, current_tokens = [], 0
            current.append(sentence)
            current_tokens += sentence_tokens
            current_topic = topic
        if current:
            chunks.append(make_chunk(document["document_id"], "topic", number, " ".join(current), tokenizer, settings["target_tokens"]))
    return chunks


def create_chunks(strategy: str, documents: list[dict[str, Any]], tokenizer: Any, nlp: Any, model: SentenceTransformer, config: dict[str, Any]) -> list[dict[str, Any]]:
    settings = config["chunking"][strategy]
    if strategy == "fixed":
        return fixed_chunks(documents, tokenizer, settings)
    if strategy == "sentence":
        sentence_map = sentence_lists(documents, nlp, tokenizer, settings["long_sentence_segment_limit_tokens"])
        return grouped_sentence_chunks(documents, tokenizer, sentence_map, "sentence", settings)
    if strategy == "semantic":
        return semantic_chunks(documents, tokenizer, nlp, model, settings)
    return topic_chunks(documents, tokenizer, nlp, model, settings, config["seed"])


def ranking_metrics(ranked_ids: list[str], qrels: dict[str, int], cutoffs: list[int]) -> list[dict[str, float | int]]:
    positives = {document_id for document_id, score in qrels.items() if score > 0}
    if not positives:
        raise ValueError("Attempted to score a query without a positive qrel")
    output = []
    for cutoff in cutoffs:
        retrieved = ranked_ids[:cutoff]
        grades = [qrels.get(document_id, 0) for document_id in retrieved]
        relevant = sum(score > 0 for score in grades)
        first_rank = next((rank for rank, score in enumerate(grades, 1) if score > 0), None)
        dcg = sum((2**score - 1) / np.log2(rank + 1) for rank, score in enumerate(grades, 1))
        ideal = sorted(qrels.values(), reverse=True)[:cutoff]
        idcg = sum((2**score - 1) / np.log2(rank + 1) for rank, score in enumerate(ideal, 1))
        output.append({
            "k": cutoff,
            "hit_rate": int(relevant > 0),
            "recall": relevant / len(positives),
            "precision": relevant / cutoff,
            "reciprocal_rank": 0.0 if first_rank is None else 1.0 / first_rank,
            "ndcg": 0.0 if idcg == 0 else dcg / idcg,
        })
    return output


def retrieve_and_score(dataset: str, chunks: list[dict[str, Any]], queries: list[dict[str, Any]], qrels: dict[str, dict[str, int]], model: SentenceTransformer, config: dict[str, Any]) -> list[dict[str, Any]]:
    if not chunks:
        raise ValueError("Chunking produced no chunks")
    chunk_embeddings = model.encode([chunk["text"] for chunk in chunks], batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype("float32")
    # On the Apple/conda environment, importing FAISS before the first MiniLM
    # inference reproducibly segfaults. Import after embedding initialisation.
    import faiss

    index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)
    cutoffs = config["evaluation"]["cutoffs"]
    rows = []
    for query in queries:
        embedding = model.encode([query["text"]], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        # IndexFlatIP returns every chunk, so taking max score per document is exact.
        scores, indices = index.search(embedding, index.ntotal)
        document_scores: dict[str, float] = {}
        for score, index_id in zip(scores[0], indices[0]):
            document_id = chunks[int(index_id)]["document_id"]
            if dataset == "arguana" and config["evaluation"]["arguana_exclude_query_document"] and document_id == query["query_id"]:
                continue
            document_scores.setdefault(document_id, float(score))
        ranked_ids = [document_id for document_id, _ in sorted(document_scores.items(), key=lambda item: item[1], reverse=True)]
        for metrics in ranking_metrics(ranked_ids, qrels[query["query_id"]], cutoffs):
            rows.append({
                "dataset": dataset,
                "strategy": chunks[0]["chunk_id"].split("::")[1],
                "query_id": query["query_id"],
                "k": metrics.pop("k"),
                "retrieved_document_ids": json.dumps(ranked_ids[: max(cutoffs)]),
                **metrics,
            })
    return rows


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], run_manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "strategy", "query_id", "k", "retrieved_document_ids", "hit_rate", "recall", "precision", "reciprocal_rank", "ndcg"]
    with (output_dir / "per_query_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for cutoff in sorted({row["k"] for row in rows}):
        subset = [row for row in rows if row["k"] == cutoff]
        summaries.append({"k": cutoff, "query_count": len(subset), **{metric: float(np.mean([row[metric] for row in subset])) for metric in ("hit_rate", "recall", "precision", "reciprocal_rank", "ndcg")}})
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["k", "query_count", "hit_rate", "recall", "precision", "reciprocal_rank", "ndcg"])
        writer.writeheader()
        writer.writerows(summaries)
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")


def package_versions() -> dict[str, str]:
    import faiss
    import sentence_transformers
    import transformers

    return {"faiss": faiss.__version__, "numpy": np.__version__, "spacy": spacy.__version__, "sentence_transformers": sentence_transformers.__version__, "transformers": transformers.__version__}


def run(dataset: str, strategy: str, config: dict[str, Any], max_documents: int | None, max_queries: int | None, output_root: Path, overwrite: bool) -> Path:
    documents, queries, qrels, source_manifest = load_dataset(dataset)
    documents, queries, qrels = deterministic_smoke_subset(documents, queries, qrels, max_documents, max_queries)
    smoke = max_documents is not None or max_queries is not None
    output_dir = output_root / ("smoke" if smoke else "full") / dataset / strategy
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; use --overwrite to replace it")
        shutil.rmtree(output_dir)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["embedding_model"], local_files_only=True)
    # The tokenizer is used to divide source documents before embedding; source
    # documents may exceed the encoder's sequence limit, while no emitted chunk
    # may do so. Avoid a misleading warning during this pre-tokenisation step.
    tokenizer.model_max_length = 1_000_000
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer(config["embedding_model"], local_files_only=True)
    embedding_sequence_limit = (
        config["chunking"][strategy]["target_tokens"]
        + model.tokenizer.num_special_tokens_to_add(pair=False)
    )
    model.max_seq_length = embedding_sequence_limit
    started = time.monotonic()
    chunks = create_chunks(strategy, documents, tokenizer, nlp, model, config)
    rows = retrieve_and_score(dataset, chunks, queries, qrels, model, config)
    manifest = {
        "protocol_version": config["protocol_version"],
        "run_kind": "smoke" if smoke else "full",
        "warning": "Smoke outputs are implementation checks only and are not valid study results." if smoke else None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "strategy": strategy,
        "source_dataset_manifest_sha256": sha256(DATA_ROOT / dataset / "manifest.json"),
        "source_archive_sha256": source_manifest["source"]["sha256"],
        "config_path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
        "config_sha256": sha256(CONFIG_PATH),
        "seed": config["seed"],
        "counts": {
            "documents": len(documents),
            "queries": len(queries),
            "chunks": len(chunks),
            "per_query_metric_rows": len(rows),
            "chunk_token_count": {
                "minimum": min(chunk["token_count"] for chunk in chunks),
                "maximum": max(chunk["token_count"] for chunk in chunks),
                "mean": float(np.mean([chunk["token_count"] for chunk in chunks])),
            },
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "packages": package_versions(),
            "runtime_controls": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_CACHE_DIR", "TOKENIZERS_PARALLELISM")
            },
        },
        "embedding_max_seq_length": embedding_sequence_limit,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_outputs(output_dir, rows, manifest)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--strategy", choices=[*STRATEGIES, "all"], default="all")
    parser.add_argument("--max-documents", type=int, help="Smoke-run document limit; never use its outputs as study results.")
    parser.add_argument("--max-queries", type=int, help="Smoke-run query limit; never use its outputs as study results.")
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned full runs without loading models.")
    args = parser.parse_args()
    if (args.max_documents is not None and args.max_documents <= 0) or (args.max_queries is not None and args.max_queries <= 0):
        parser.error("Smoke limits must be positive")
    config = load_config(CONFIG_PATH)
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)
    for dataset in datasets:
        documents, queries, _, _ = load_dataset(dataset)
        for strategy in strategies:
            if args.dry_run:
                print(f"Planned full run: dataset={dataset}, strategy={strategy}, documents={len(documents)}, queries={len(queries)}")
            else:
                output = run(dataset, strategy, config, args.max_documents, args.max_queries, args.output_root, args.overwrite)
                print(f"Completed {dataset}/{strategy}: {output}")


if __name__ == "__main__":
    main()
