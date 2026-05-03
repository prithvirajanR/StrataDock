from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stratadock.core.session import (
    RUN_MANIFEST_NAME,
    duplicate_run_settings,
    extract_session_archive,
    reopen_run,
    run_summary_metadata,
)


@dataclass(frozen=True)
class RunHistoryEntry:
    name: str
    path: Path
    created_at_utc: str
    counts: dict[str, int]
    manifest_path: Path
    results_csv: Path
    summary_txt: Path
    started_at_utc: str = ""
    ended_at_utc: str = ""
    runtime_seconds: int | float | None = None


@dataclass(frozen=True)
class LoadedRun:
    path: Path
    manifest: dict[str, object]
    results: list[dict[str, object]]
    counts: dict[str, int]
    paths: dict[str, Path]
    summary: dict[str, object]


def _read_manifest(run_dir: Path) -> dict[str, object]:
    payload = json.loads((run_dir / RUN_MANIFEST_NAME).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected run manifest object: {run_dir / RUN_MANIFEST_NAME}")
    return payload


def _manifest_output_path(run_dir: Path, manifest: dict[str, object], key: str, fallback: str) -> Path:
    outputs = manifest.get("outputs")
    raw = outputs.get(key) if isinstance(outputs, dict) else None
    path = Path(str(raw or fallback))
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
        candidates.append(run_dir / path.name)
    else:
        candidates.append(run_dir / path)
        candidates.append(run_dir / path.name)
    fallback_path = run_dir / fallback
    candidates.append(fallback_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return fallback_path


def _normalized_counts(manifest: dict[str, object], results: list[dict[str, object]] | None = None) -> dict[str, int]:
    counts = manifest.get("counts")
    if isinstance(counts, dict):
        total = int(counts.get("total", 0))
        success = int(counts.get("success", 0))
        failed = int(counts.get("failed", counts.get("failed_or_skipped", total - success)))
        return {"total": total, "success": success, "failed": failed}
    results = results or []
    success = sum(1 for row in results if row.get("docking_status") == "success")
    return {"total": len(results), "success": success, "failed": len(results) - success}


def list_run_history(runs_dir: Path, *, limit: int | None = None) -> list[RunHistoryEntry]:
    runs_dir = runs_dir.resolve()
    if not runs_dir.exists():
        return []

    entries: list[RunHistoryEntry] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        try:
            manifest = _read_manifest(run_dir)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        summary = run_summary_metadata(run_dir, manifest)
        entries.append(
            RunHistoryEntry(
                name=run_dir.name,
                path=run_dir,
                created_at_utc=str(manifest.get("created_at_utc") or ""),
                counts=_normalized_counts(manifest),
                manifest_path=run_dir / RUN_MANIFEST_NAME,
                results_csv=_manifest_output_path(run_dir, manifest, "results_csv", "results.csv"),
                summary_txt=_manifest_output_path(run_dir, manifest, "summary_txt", "run_summary.txt"),
                started_at_utc=str(summary.get("started_at_utc") or ""),
                ended_at_utc=str(summary.get("ended_at_utc") or ""),
                runtime_seconds=summary.get("runtime_seconds"),
            )
        )
    sorted_entries = sorted(entries, key=lambda entry: (entry.created_at_utc, entry.name), reverse=True)
    return sorted_entries[:limit] if limit is not None else sorted_entries


def load_run(run_dir: Path, *, runs_dir: Path | None = None) -> LoadedRun:
    run_dir = run_dir.resolve()
    if runs_dir is not None:
        root = runs_dir.resolve()
        if run_dir != root and root not in run_dir.parents:
            raise ValueError(f"Run directory is outside runs directory: {run_dir}")
    manifest = _read_manifest(run_dir)
    results_json = _manifest_output_path(run_dir, manifest, "results_json", "results.json")
    reopened = reopen_run(run_dir, runs_dir=runs_dir)
    results: list[dict[str, object]]
    if results_json.exists():
        payload = json.loads(results_json.read_text(encoding="utf-8"))
        results = payload if isinstance(payload, list) else []
    else:
        results = list(reopened["records"])
    paths = dict(reopened["paths"])
    paths["results_json"] = results_json
    return LoadedRun(
        path=run_dir,
        manifest=manifest,
        results=results,
        counts=_normalized_counts(manifest, results),
        paths=paths,
        summary=dict(reopened["summary"]),
    )


def import_session_zip(archive_path: Path, output_dir: Path, *, runs_dir: Path | None = None) -> Path:
    if runs_dir is not None:
        root = runs_dir.resolve()
        target = output_dir.resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Import directory is outside runs directory: {target}")
    return extract_session_archive(archive_path, output_dir)
