"""Capture environment and final-artifact checksums for the expanded study."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "expanded_retrieval"
AUDIT_ROOT = RESULTS_ROOT / "audits"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*arguments: str) -> str | None:
    completed = subprocess.run(["git", *arguments], cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else None


def tracked_paths() -> list[Path]:
    config_prefix = sha256(PROJECT_ROOT / "config" / "expanded_retrieval_v1.json")[:16]
    paths = [
        PROJECT_ROOT / "config" / "expanded_retrieval_v1.json",
        PROJECT_ROOT / "data" / "manifests" / "expanded_study_protocol_v1.md",
        PROJECT_ROOT / "data" / "manifests" / "expanded_statistical_analysis_protocol_v1.md",
        PROJECT_ROOT / "data" / "manifests" / "expanded_efficiency_benchmark_protocol_v1.md",
        PROJECT_ROOT / "config" / "fixed_overlap_ablation_v1.json",
        PROJECT_ROOT / "data" / "manifests" / "fixed_overlap_ablation_protocol_v1.md",
        PROJECT_ROOT / "data" / "manifests" / "retrieved_token_budget_protocol_v1.md",
        PROJECT_ROOT / "data" / "manifests" / "paired_randomisation_sensitivity_protocol_v1.md",
        PROJECT_ROOT / "data" / "manifests" / "expanded_input_inventory_v1.csv",
        PROJECT_ROOT / "data" / "manifests" / "expanded_input_checksums_v1.csv",
        AUDIT_ROOT / "expanded_input_and_relevance_audit_v1.json",
        RESULTS_ROOT / "analysis" / "analysis_manifest.json",
        RESULTS_ROOT / "analysis" / "descriptive_metrics.csv",
        RESULTS_ROOT / "analysis" / "pairwise_comparisons.csv",
        RESULTS_ROOT / "efficiency_v2" / "benchmark_manifest.json",
        RESULTS_ROOT / "efficiency_v2" / "preparation_trials.csv",
        RESULTS_ROOT / "efficiency_v2" / "query_latency_trials.csv",
        RESULTS_ROOT / "efficiency_v2" / "analysis" / "analysis_manifest.json",
        RESULTS_ROOT / "efficiency_v2" / "analysis" / "efficiency_summary.csv",
        RESULTS_ROOT / "chunk_metadata" / "chunk_length_distributions.csv",
        RESULTS_ROOT / "chunk_metadata" / "representative_segments.csv",
        RESULTS_ROOT / "chunk_metadata" / "metadata_export_manifest.json",
        RESULTS_ROOT / "ablations" / "fixed_overlap_v1" / "analysis" / "analysis_manifest.json",
        RESULTS_ROOT / "ablations" / "fixed_overlap_v1" / "analysis" / "overlap_ablation_comparisons.csv",
        RESULTS_ROOT / "token_budget_v1" / "analysis" / "analysis_manifest.json",
        RESULTS_ROOT / "token_budget_v1" / "analysis" / "descriptive_metrics.csv",
        RESULTS_ROOT / "token_budget_v1" / "analysis" / "pairwise_comparisons.csv",
        RESULTS_ROOT / "randomisation_sensitivity_v1" / "analysis_manifest.json",
        RESULTS_ROOT / "randomisation_sensitivity_v1" / "primary_retrieval_randomisation.csv",
        RESULTS_ROOT / "randomisation_sensitivity_v1" / "token_budget_randomisation.csv",
        PROJECT_ROOT / "scripts" / "run_beir_retrieval.py",
        PROJECT_ROOT / "scripts" / "analyze_expanded_retrieval.py",
        PROJECT_ROOT / "scripts" / "benchmark_expanded_efficiency.py",
        PROJECT_ROOT / "scripts" / "analyze_expanded_efficiency.py",
        PROJECT_ROOT / "scripts" / "export_expanded_chunk_metadata.py",
        PROJECT_ROOT / "scripts" / "audit_expanded_inputs.py",
        PROJECT_ROOT / "scripts" / "capture_expanded_reproducibility.py",
        PROJECT_ROOT / "scripts" / "analyze_fixed_overlap_ablation.py",
        PROJECT_ROOT / "scripts" / "run_token_budget_retrieval.py",
        PROJECT_ROOT / "scripts" / "analyze_token_budget_retrieval.py",
        PROJECT_ROOT / "scripts" / "analyze_paired_randomisation_sensitivity.py",
    ]
    for dataset in ("scifact", "nfcorpus", "arguana"):
        for strategy in ("fixed", "sentence", "semantic", "topic"):
            paths.extend([
                RESULTS_ROOT / "full" / dataset / strategy / "per_query_metrics.csv",
                RESULTS_ROOT / "full" / dataset / strategy / "summary.csv",
                RESULTS_ROOT / "full" / dataset / strategy / "run_manifest.json",
            ])
        paths.extend([
            RESULTS_ROOT / "topic_assignment_cache" / f"{dataset}_{config_prefix}.csv",
            RESULTS_ROOT / "topic_assignment_cache" / f"{dataset}_{config_prefix}.manifest.json",
            RESULTS_ROOT / "ablations" / "fixed_overlap_v1" / "full" / dataset / "fixed_no_overlap" / "per_query_metrics.csv",
            RESULTS_ROOT / "ablations" / "fixed_overlap_v1" / "full" / dataset / "fixed_no_overlap" / "summary.csv",
            RESULTS_ROOT / "ablations" / "fixed_overlap_v1" / "full" / dataset / "fixed_no_overlap" / "run_manifest.json",
        ])
        for strategy in ("fixed", "sentence", "semantic", "topic"):
            paths.extend([
                RESULTS_ROOT / "token_budget_v1" / "full" / dataset / strategy / "per_query_token_budget_metrics.csv",
                RESULTS_ROOT / "token_budget_v1" / "full" / dataset / strategy / "summary.csv",
                RESULTS_ROOT / "token_budget_v1" / "full" / dataset / strategy / "run_manifest.json",
            ])
    return paths


def main() -> None:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    snapshot = Path(snapshot_download(MODEL_ID, local_files_only=True))
    packages = sorted(
        {distribution.metadata["Name"].lower(): distribution.version for distribution in importlib.metadata.distributions()}.items()
    )
    model_files = [path for path in (snapshot / "config.json", snapshot / "modules.json", snapshot / "sentence_bert_config.json") if path.exists()]
    environment = {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "model": {
            "identifier": MODEL_ID,
            "local_snapshot_revision": snapshot.name,
            "snapshot_file_sha256": {str(path.relative_to(snapshot)): sha256(path) for path in model_files},
        },
        "packages": dict(packages),
        "git_head_at_capture": git_output("rev-parse", "HEAD"),
        "git_status_porcelain_at_capture": git_output("status", "--porcelain"),
        "runtime_environment_variables": {
            key: __import__("os").environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM", "NUMBA_CACHE_DIR")
        },
    }
    (AUDIT_ROOT / "expanded_environment_v1.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    paths = tracked_paths()
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot checksum missing final artifacts: {missing}")
    with (AUDIT_ROOT / "expanded_final_artifact_checksums_v1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"], lineterminator="\n")
        writer.writeheader()
        for path in sorted(paths):
            writer.writerow({"relative_path": str(path.relative_to(PROJECT_ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    print(f"Captured {len(packages)} package versions and checksums for {len(paths)} study artifacts.")


if __name__ == "__main__":
    main()
