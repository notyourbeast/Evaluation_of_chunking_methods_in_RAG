"""Export auditable chunk metadata and deterministic segmentation examples.

This script regenerates chunks from the versioned expanded-retrieval
configuration without rerunning retrieval. Full chunk text is not copied into
the metadata exports; each row includes a text SHA-256 and boundary evidence.
Representative excerpts are exported separately using documented deterministic
selection rules.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from run_beir_retrieval import (
    CONFIG_PATH,
    DATASETS,
    PROJECT_ROOT,
    STRATEGIES,
    TOPIC_ASSIGNMENT_CACHE_ROOT,
    create_chunks,
    load_config,
    load_dataset,
    sha256,
)


OUTPUT_ROOT = PROJECT_ROOT / "results" / "expanded_retrieval" / "chunk_metadata"
SUMMARY_PATH = OUTPUT_ROOT / "chunk_length_distributions.csv"
EXAMPLES_PATH = OUTPUT_ROOT / "representative_segments.csv"
MANIFEST_PATH = OUTPUT_ROOT / "metadata_export_manifest.json"


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def excerpt(text: str, limit: int = 500) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def write_metadata(path: Path, chunks: list[dict[str, Any]]) -> None:
    fields = [
        "chunk_id",
        "document_id",
        "token_count",
        "text_sha256",
        "boundary_metadata_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for chunk in chunks:
            writer.writerow(
                {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "token_count": chunk["token_count"],
                    "text_sha256": text_sha256(chunk["text"]),
                    "boundary_metadata_json": json.dumps(chunk["boundary_metadata"], sort_keys=True),
                }
            )


def chunk_length_summary(dataset: str, strategy: str, chunks: list[dict[str, Any]]) -> dict[str, object]:
    lengths = np.array([chunk["token_count"] for chunk in chunks], dtype=float)
    boundary_counts: dict[str, int] = {}
    outlier_chunks = 0
    for chunk in chunks:
        metadata = chunk["boundary_metadata"]
        if metadata.get("contains_outlier_topic"):
            outlier_chunks += 1
        for reason in metadata["boundary_after"]:
            boundary_counts[reason] = boundary_counts.get(reason, 0) + 1
    return {
        "dataset": dataset,
        "strategy": strategy,
        "chunk_count": len(chunks),
        "minimum_tokens": int(lengths.min()),
        "p10_tokens": float(np.quantile(lengths, 0.10)),
        "p25_tokens": float(np.quantile(lengths, 0.25)),
        "median_tokens": float(np.quantile(lengths, 0.50)),
        "p75_tokens": float(np.quantile(lengths, 0.75)),
        "p90_tokens": float(np.quantile(lengths, 0.90)),
        "maximum_tokens": int(lengths.max()),
        "topic_outlier_chunks": outlier_chunks,
        "boundary_after_counts_json": json.dumps(boundary_counts, sort_keys=True),
    }


def selected_examples(dataset: str, strategy: str, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Choose boundary examples deterministically, never by retrieval outcome."""
    selected: list[tuple[str, dict[str, Any]]] = []
    if strategy == "semantic":
        selected.extend(
            ("first_two_semantic_threshold_boundaries", chunk)
            for chunk in chunks
            if "semantic_similarity_below_threshold" in chunk["boundary_metadata"]["boundary_after"]
        )
    elif strategy == "topic":
        selected.extend(
            ("first_two_topic_label_change_boundaries", chunk)
            for chunk in chunks
            if "topic_label_change" in chunk["boundary_metadata"]["boundary_after"]
        )
        selected.extend(
            ("first_two_chunks_containing_outlier_topic", chunk)
            for chunk in chunks
            if chunk["boundary_metadata"].get("contains_outlier_topic")
        )
    else:
        return []

    limits = {
        "first_two_semantic_threshold_boundaries": 2,
        "first_two_topic_label_change_boundaries": 2,
        "first_two_chunks_containing_outlier_topic": 2,
    }
    counts: dict[str, int] = {}
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for rule, chunk in selected:
        if counts.get(rule, 0) >= limits[rule] or chunk["chunk_id"] in seen:
            continue
        counts[rule] = counts.get(rule, 0) + 1
        seen.add(chunk["chunk_id"])
        rows.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "selection_rule": rule,
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "token_count": str(chunk["token_count"]),
                "boundary_metadata_json": json.dumps(chunk["boundary_metadata"], sort_keys=True),
                "text_excerpt": excerpt(chunk["text"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--strategy", choices=[*STRATEGIES, "all"], default="all")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    summaries: list[dict[str, object]] = []
    examples: list[dict[str, str]] = []
    config = load_config(CONFIG_PATH)
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)

    for dataset in datasets:
        documents, _, _, source_manifest = load_dataset(dataset)
        for strategy in strategies:
            destination = output_root / dataset / strategy / "chunk_metadata.csv"
            if destination.exists() and not args.overwrite:
                raise FileExistsError(f"{destination} already exists; use --overwrite to replace it")
            tokenizer = AutoTokenizer.from_pretrained(config["embedding_model"], local_files_only=True)
            tokenizer.model_max_length = 1_000_000
            model = SentenceTransformer(config["embedding_model"], local_files_only=True)
            model.max_seq_length = config["chunking"][strategy]["target_tokens"] + model.tokenizer.num_special_tokens_to_add(pair=False)
            nlp = spacy.load("en_core_web_sm")
            topic_cache_path = None
            if strategy == "topic":
                topic_cache_path = TOPIC_ASSIGNMENT_CACHE_ROOT / f"{dataset}_{sha256(CONFIG_PATH)[:16]}.csv"
                if not topic_cache_path.exists():
                    raise FileNotFoundError(
                        f"No persisted topic assignments for {dataset}. Run the full topic retrieval evaluation first: {topic_cache_path}"
                    )
            chunks = create_chunks(strategy, documents, tokenizer, nlp, model, config, topic_assignment_cache_path=topic_cache_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            write_metadata(destination, chunks)
            summaries.append(chunk_length_summary(dataset, strategy, chunks))
            examples.extend(selected_examples(dataset, strategy, chunks))
            print(f"Exported {len(chunks)} chunks for {dataset}/{strategy}: {destination}")

    summary_fields = [
        "dataset",
        "strategy",
        "chunk_count",
        "minimum_tokens",
        "p10_tokens",
        "p25_tokens",
        "median_tokens",
        "p75_tokens",
        "p90_tokens",
        "maximum_tokens",
        "topic_outlier_chunks",
        "boundary_after_counts_json",
    ]
    example_fields = [
        "dataset",
        "strategy",
        "selection_rule",
        "chunk_id",
        "document_id",
        "token_count",
        "boundary_metadata_json",
        "text_excerpt",
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / SUMMARY_PATH.name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    with (output_root / EXAMPLES_PATH.name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=example_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(examples)
    topic_assignment_caches = {}
    if "topic" in strategies:
        for dataset in datasets:
            cache_path = TOPIC_ASSIGNMENT_CACHE_ROOT / f"{dataset}_{sha256(CONFIG_PATH)[:16]}.csv"
            topic_assignment_caches[dataset] = {
                "path": str(cache_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(cache_path),
                "manifest_path": str(cache_path.with_suffix(".manifest.json").relative_to(PROJECT_ROOT)),
            }
    manifest = {
        "metadata_schema_version": 1,
        "config_path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
        "config_sha256": sha256(CONFIG_PATH),
        "seed": config["seed"],
        "datasets": list(datasets),
        "strategies": list(strategies),
        "source_archives": {
            dataset: load_dataset(dataset)[3]["source"]["sha256"] for dataset in datasets
        },
        "topic_assignment_caches": topic_assignment_caches,
        "text_policy": "Full chunk text is represented by SHA-256. Excerpts appear only in representative_segments.csv.",
        "example_selection": "First qualifying chunks in deterministic corpus order; no retrieval outcome is used.",
    }
    (output_root / MANIFEST_PATH.name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote summary and examples to {output_root}")


if __name__ == "__main__":
    main()
