"""Download, validate, and normalise the BEIR datasets used in the expanded study.

The importer deliberately keeps the full BEIR corpus and every test query with
at least one positive qrel. It does not sample by retrieval outcome or filter
on answer-string occurrence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = PROJECT_ROOT / "data" / "raw" / "beir_archives"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "expanded"

DATASETS = {
    "scifact": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "md5": "5f7d1de60b170fc8027bb7898e2efca1",
        "domain": "scientific claim retrieval",
        "self_match_exclusion": False,
    },
    "nfcorpus": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
        "md5": "a89dba18a62ef92f7d323ec890a0d38d",
        "domain": "biomedical and nutrition information retrieval",
        "self_match_exclusion": False,
    },
    "arguana": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip",
        "md5": "8ad3e3c2a5867cdced806d6503f29b99",
        "domain": "counterargument retrieval",
        "self_match_exclusion": True,
    },
}


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_archive(dataset: str, spec: dict, archive_root: Path) -> Path:
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / f"{dataset}.zip"
    if archive.exists() and file_digest(archive, "md5") == spec["md5"]:
        return archive

    temporary = archive.with_suffix(".zip.part")
    temporary.unlink(missing_ok=True)
    print(f"Downloading {dataset} from {spec['url']}")
    try:
        urllib.request.urlretrieve(spec["url"], temporary)
        actual_md5 = file_digest(temporary, "md5")
        if actual_md5 != spec["md5"]:
            raise ValueError(
                f"Checksum mismatch for {dataset}: expected {spec['md5']}, got {actual_md5}"
            )
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive path: {member.filename}")
        bundle.extractall(destination)


def locate_dataset_root(extracted_root: Path) -> Path:
    matches = list(extracted_root.rglob("corpus.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"Expected one corpus.jsonl under {extracted_root}, found {len(matches)}")
    dataset_root = matches[0].parent
    required = [dataset_root / "queries.jsonl", dataset_root / "qrels" / "test.tsv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected BEIR files: {missing}")
    return dataset_root


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path} line {line_number}") from error


def normalise_text(title: str | None, body: str | None) -> str:
    title = (title or "").strip()
    body = (body or "").strip()
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def write_jsonl(path: Path, records) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def ingest_dataset(dataset: str, spec: dict, archive_root: Path, output_root: Path, overwrite: bool) -> None:
    destination = output_root / dataset
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"{destination} already exists; use --overwrite to replace it")
        shutil.rmtree(destination)

    archive = download_archive(dataset, spec, archive_root)
    with tempfile.TemporaryDirectory(prefix=f"beir_{dataset}_") as temporary_directory:
        extracted_root = Path(temporary_directory)
        safe_extract(archive, extracted_root)
        source_root = locate_dataset_root(extracted_root)

        corpus_records = list(read_jsonl(source_root / "corpus.jsonl"))
        query_records = list(read_jsonl(source_root / "queries.jsonl"))
        corpus_ids = {record["_id"] for record in corpus_records}
        query_by_id = {record["_id"]: record for record in query_records}
        if len(corpus_ids) != len(corpus_records):
            raise ValueError(f"Duplicate corpus identifiers in {dataset}")
        if len(query_by_id) != len(query_records):
            raise ValueError(f"Duplicate query identifiers in {dataset}")

        qrels = []
        positive_qrels = defaultdict(list)
        dangling_qrels = []
        qrels_path = source_root / "qrels" / "test.tsv"
        with qrels_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required_columns = {"query-id", "corpus-id", "score"}
            if not required_columns.issubset(reader.fieldnames or []):
                raise ValueError(f"Unexpected qrels header in {qrels_path}: {reader.fieldnames}")
            for row in reader:
                query_id = row["query-id"]
                corpus_id = row["corpus-id"]
                score = int(row["score"])
                if query_id not in query_by_id:
                    raise ValueError(f"Qrel references unknown query {query_id} in {dataset}")
                qrels.append({"query_id": query_id, "document_id": corpus_id, "score": score})
                if corpus_id not in corpus_ids:
                    dangling_qrels.append({"query_id": query_id, "document_id": corpus_id, "score": score})
                elif score > 0:
                    positive_qrels[query_id].append(corpus_id)

        destination.mkdir(parents=True)
        documents_path = destination / "documents.jsonl"
        queries_path = destination / "queries.jsonl"
        eligible_path = destination / "eligible_test_queries.jsonl"
        qrels_output_path = destination / "qrels_test.tsv"

        document_count = write_jsonl(
            documents_path,
            (
                {
                    "dataset": dataset,
                    "document_id": record["_id"],
                    "title": (record.get("title") or "").strip(),
                    "text": normalise_text(record.get("title"), record.get("text")),
                    "metadata": record.get("metadata") or {},
                }
                for record in corpus_records
            ),
        )
        query_count = write_jsonl(
            queries_path,
            (
                {
                    "dataset": dataset,
                    "query_id": record["_id"],
                    "text": (record.get("text") or "").strip(),
                    "metadata": record.get("metadata") or {},
                }
                for record in query_records
            ),
        )
        eligible_count = write_jsonl(
            eligible_path,
            (
                {
                    "dataset": dataset,
                    "query_id": query_id,
                    "text": (query_by_id[query_id].get("text") or "").strip(),
                    "positive_qrel_count": len(positive_qrels[query_id]),
                }
                for query_id in query_by_id
                if positive_qrels[query_id]
            ),
        )
        with qrels_output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["query-id", "corpus-id", "score"], delimiter="\t")
            writer.writeheader()
            writer.writerows(
                {"query-id": row["query_id"], "corpus-id": row["document_id"], "score": row["score"]}
                for row in qrels
            )

        manifest = {
            "dataset": dataset,
            "domain": spec["domain"],
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "archive_url": spec["url"],
                "archive_path": str(archive.relative_to(PROJECT_ROOT)),
                "expected_md5": spec["md5"],
                "verified_md5": file_digest(archive, "md5"),
                "sha256": file_digest(archive, "sha256"),
            },
            "selection_rules": {
                "corpus": "All non-empty documents in the official BEIR corpus.",
                "queries": (
                    "All official test queries with at least one qrel score greater than zero "
                    "whose relevant document is present in the official corpus archive."
                ),
                "qrel_binary_threshold": 1,
                "arguana_self_match_exclusion_at_retrieval": spec["self_match_exclusion"],
            },
            "counts": {
                "loaded_documents": document_count,
                "loaded_queries": query_count,
                "qrels": len(qrels),
                "qrels_with_corpus_document": len(qrels) - len(dangling_qrels),
                "qrels_with_missing_corpus_document": len(dangling_qrels),
                "eligible_test_queries": eligible_count,
                "queries_without_positive_qrel": query_count - eligible_count,
            },
            "source_integrity": {
                "dangling_qrels": dangling_qrels,
                "handling": (
                    "All official qrels are preserved in qrels_test.tsv. Qrels whose document IDs "
                    "are absent from the official corpus cannot be retrieved and are excluded only "
                    "when determining evaluable queries and computing retrieval metrics."
                ),
            },
            "normalised_files": {
                name: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_digest(path, "sha256")}
                for name, path in {
                    "documents": documents_path,
                    "queries": queries_path,
                    "eligible_test_queries": eligible_path,
                    "qrels_test": qrels_output_path,
                }.items()
            },
        }
        with (destination / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    print(
        f"Imported {dataset}: {document_count} documents, {eligible_count}/{query_count} eligible test queries, {len(qrels)} qrels"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), action="append", help="Dataset to ingest; repeat as needed.")
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing normalised dataset directory.")
    args = parser.parse_args()

    selected = args.dataset or list(DATASETS)
    for dataset in selected:
        ingest_dataset(dataset, DATASETS[dataset], args.archive_root, args.output_root, args.overwrite)


if __name__ == "__main__":
    main()
