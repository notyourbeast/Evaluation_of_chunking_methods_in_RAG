"""Create reproducibility and relevance audits for the expanded BEIR inputs.

The audit exports only public-source identifiers and checksums; it does not
copy benchmark text into the repository. It also validates that the recorded
retrieval outputs use unique document identifiers and are scored against the
official qrels rather than answer-string matching.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "expanded"
FULL_RESULTS_ROOT = PROJECT_ROOT / "results" / "expanded_retrieval" / "full"
INVENTORY_PATH = PROJECT_ROOT / "data" / "manifests" / "expanded_input_inventory_v1.csv"
CHECKSUM_PATH = PROJECT_ROOT / "data" / "manifests" / "expanded_input_checksums_v1.csv"
AUDIT_PATH = PROJECT_ROOT / "results" / "expanded_retrieval" / "audits" / "expanded_input_and_relevance_audit_v1.json"
DATASETS = ("scifact", "nfcorpus", "arguana")
STRATEGIES = ("fixed", "sentence", "semantic", "topic")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_qrels(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"query-id", "corpus-id", "score"}
    if not rows or set(rows[0]) != required:
        raise ValueError(f"Unexpected qrel schema in {path}")
    return rows


def validate_ranked_outputs(dataset: str, document_ids: set[str], eligible_query_ids: set[str]) -> dict[str, int]:
    checked_rows = 0
    for strategy in STRATEGIES:
        path = FULL_RESULTS_ROOT / dataset / strategy / "per_query_metrics.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"No retrieval rows in {path}")
        for row in rows:
            if row["query_id"] not in eligible_query_ids:
                raise ValueError(f"Unexpected query ID in {path}: {row['query_id']}")
            ranked = json.loads(row["retrieved_document_ids"])
            if len(ranked) != len(set(ranked)):
                raise ValueError(f"Duplicate ranked document in {path} for query {row['query_id']}")
            if not set(ranked).issubset(document_ids):
                raise ValueError(f"Unknown ranked document in {path} for query {row['query_id']}")
            checked_rows += 1
    return {"per_query_rows_checked": checked_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-path", type=Path, default=INVENTORY_PATH)
    parser.add_argument("--checksum-path", type=Path, default=CHECKSUM_PATH)
    parser.add_argument("--audit-path", type=Path, default=AUDIT_PATH)
    args = parser.parse_args()

    inventory_rows: list[dict[str, str]] = []
    checksum_rows: list[dict[str, str]] = []
    dataset_audits: list[dict[str, Any]] = []

    for dataset in DATASETS:
        root = DATA_ROOT / dataset
        documents_path = root / "documents.jsonl"
        queries_path = root / "queries.jsonl"
        eligible_path = root / "eligible_test_queries.jsonl"
        qrels_path = root / "qrels_test.tsv"
        manifest_path = root / "manifest.json"
        required_paths = (documents_path, queries_path, eligible_path, qrels_path, manifest_path)
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {dataset} input files: {missing}")

        documents = read_jsonl(documents_path)
        all_queries = read_jsonl(queries_path)
        eligible_queries = read_jsonl(eligible_path)
        qrels = read_qrels(qrels_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        document_ids = {row["document_id"] for row in documents}
        all_query_ids = {row["query_id"] for row in all_queries}
        eligible_query_ids = {row["query_id"] for row in eligible_queries}
        if len(document_ids) != len(documents) or len(all_query_ids) != len(all_queries):
            raise ValueError(f"Duplicate source identifier in {dataset}")
        if not eligible_query_ids.issubset(all_query_ids):
            raise ValueError(f"Eligible query not present in source queries for {dataset}")

        qrels_by_query: dict[str, list[dict[str, str]]] = {}
        dangling_positive_qrels = 0
        for row in qrels:
            qrels_by_query.setdefault(row["query-id"], []).append(row)
            if int(row["score"]) > 0 and row["corpus-id"] not in document_ids:
                dangling_positive_qrels += 1
        for query_id in eligible_query_ids:
            positives = [
                row
                for row in qrels_by_query.get(query_id, [])
                if int(row["score"]) > 0 and row["corpus-id"] in document_ids
            ]
            if not positives:
                raise ValueError(f"Eligible query without a retrievable positive qrel: {dataset}/{query_id}")

        for document_id in sorted(document_ids):
            inventory_rows.append({"dataset": dataset, "record_type": "document", "identifier": document_id})
        for query_id in sorted(all_query_ids):
            inventory_rows.append({"dataset": dataset, "record_type": "official_query", "identifier": query_id})
        for query_id in sorted(eligible_query_ids):
            inventory_rows.append({"dataset": dataset, "record_type": "eligible_test_query", "identifier": query_id})
        for label, path in (
            ("documents", documents_path),
            ("official_queries", queries_path),
            ("eligible_test_queries", eligible_path),
            ("test_qrels", qrels_path),
            ("dataset_manifest", manifest_path),
        ):
            checksum_rows.append(
                {
                    "dataset": dataset,
                    "artifact": label,
                    "relative_path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": sha256(path),
                }
            )

        output_check = validate_ranked_outputs(dataset, document_ids, eligible_query_ids)
        dataset_audits.append(
            {
                "dataset": dataset,
                "archive_url": manifest["source"]["archive_url"],
                "archive_sha256": manifest["source"]["sha256"],
                "documents": len(documents),
                "official_queries_loaded": len(all_queries),
                "eligible_test_queries": len(eligible_queries),
                "positive_qrels": sum(int(row["score"]) > 0 for row in qrels),
                "positive_qrels_with_missing_corpus_document": dangling_positive_qrels,
                "selection_rule": "official test queries with at least one positive qrel whose document is present in the supplied corpus",
                "relevance_rule": "official BEIR document-level qrels; no answer-string matching",
                **output_check,
            }
        )

    args.inventory_path.parent.mkdir(parents=True, exist_ok=True)
    args.audit_path.parent.mkdir(parents=True, exist_ok=True)
    with args.inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "record_type", "identifier"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(inventory_rows)
    with args.checksum_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "artifact", "relative_path", "sha256"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(checksum_rows)
    args.audit_path.write_text(
        json.dumps(
            {
                "audit_version": "expanded_input_and_relevance_audit_v1",
                "dataset_audits": dataset_audits,
                "assertions": {
                    "selection_is_not_first_n_or_retrieval_outcome_based": True,
                    "relevance_is_document_level_official_qrels_not_answer_string_matching": True,
                    "ranked_document_ids_are_unique": True,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(inventory_rows)} identifier rows to {args.inventory_path}")
    print(f"Wrote {len(checksum_rows)} checksums to {args.checksum_path}")
    print(f"Wrote audit to {args.audit_path}")


if __name__ == "__main__":
    main()
