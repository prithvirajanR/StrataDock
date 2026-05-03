from __future__ import annotations

import json
import csv
import zipfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath


SESSION_MANIFEST_NAME = "session_manifest.json"
RUN_MANIFEST_NAME = "run_manifest.json"
STOP_FILE_NAME = "STOP"


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_run_manifest(run_dir: Path) -> dict[str, object]:
    manifest_path = run_dir / RUN_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest does not exist: {manifest_path}")
    return _read_json(manifest_path)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve_run_dir(run_dir: Path, runs_dir: Path | None = None) -> Path:
    resolved = run_dir.resolve()
    if runs_dir is None:
        return resolved
    root = runs_dir.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Run directory is outside runs directory: {resolved}")
    return resolved


def _parse_utc(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _manifest_output_path(run_dir: Path, manifest: dict[str, object], key: str, fallback: str) -> Path:
    outputs = manifest.get("outputs")
    raw = outputs.get(key) if isinstance(outputs, dict) else None
    path = Path(str(raw or fallback))
    return path if path.is_absolute() else run_dir / path


def _read_run_events(events_log: Path) -> list[dict[str, object]]:
    if not events_log.exists():
        return []
    events: list[dict[str, object]] = []
    for line in events_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _event_time(events: list[dict[str, object]], names: set[str], *, first: bool) -> str:
    sequence = events if first else list(reversed(events))
    for event in sequence:
        if str(event.get("event") or "") in names and event.get("time_utc"):
            return str(event["time_utc"])
    return ""


def run_summary_metadata(run_dir: Path, manifest: dict[str, object] | None = None) -> dict[str, object]:
    run_dir = run_dir.resolve()
    manifest = manifest or _read_run_manifest(run_dir)
    events_log = _manifest_output_path(run_dir, manifest, "events_log", "run_events.jsonl")
    events = _read_run_events(events_log)
    started_at = (
        _event_time(events, {"run_started", "ensemble_started"}, first=True)
        or str(manifest.get("started_at_utc") or manifest.get("created_at_utc") or "")
    )
    ended_at = (
        _event_time(events, {"run_completed", "ensemble_completed", "run_cancelled", "ensemble_cancelled"}, first=False)
        or str(manifest.get("ended_at_utc") or manifest.get("completed_at_utc") or manifest.get("finished_at_utc") or "")
    )
    runtime = manifest.get("runtime_seconds")
    if runtime is None:
        started = _parse_utc(started_at)
        ended = _parse_utc(ended_at)
        runtime = int((ended - started).total_seconds()) if started and ended else None
    return {
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "runtime_seconds": runtime,
        "events_log": str(events_log),
    }


def _manifest_counts(manifest: dict[str, object], records: list[dict[str, str]] | None = None) -> dict[str, int]:
    counts = manifest.get("counts")
    if isinstance(counts, dict):
        return {str(key): int(value) for key, value in counts.items() if isinstance(value, int | float)}
    records = records or []
    success = sum(1 for record in records if record.get("docking_status") == "success")
    return {"total": len(records), "success": success, "failed_or_skipped": len(records) - success}


def list_run_history(runs_dir: Path) -> list[dict[str, object]]:
    runs_dir = runs_dir.resolve()
    if not runs_dir.exists():
        return []

    records: list[dict[str, object]] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        try:
            manifest = _read_run_manifest(run_dir)
        except (OSError, ValueError, json.JSONDecodeError):
            continue

        results_csv = _manifest_output_path(run_dir, manifest, "results_csv", "results.csv")
        summary_txt = _manifest_output_path(run_dir, manifest, "summary_txt", "run_summary.txt")
        summary = run_summary_metadata(run_dir, manifest)
        records.append(
            {
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "manifest_path": str(run_dir / RUN_MANIFEST_NAME),
                "created_at_utc": str(manifest.get("created_at_utc") or ""),
                "started_at_utc": summary["started_at_utc"],
                "ended_at_utc": summary["ended_at_utc"],
                "runtime_seconds": summary["runtime_seconds"],
                "version": str(manifest.get("version") or ""),
                "counts": _manifest_counts(manifest),
                "results_csv": str(results_csv),
                "summary_txt": str(summary_txt),
                "has_results": results_csv.exists(),
                "has_summary": summary_txt.exists(),
            }
        )
    return sorted(records, key=lambda record: (str(record["created_at_utc"]), str(record["run_id"])), reverse=True)


def reopen_run(run_dir: Path, *, runs_dir: Path | None = None) -> dict[str, object]:
    run_dir = _resolve_run_dir(run_dir, runs_dir)
    manifest = _read_run_manifest(run_dir)
    results_csv = _manifest_output_path(run_dir, manifest, "results_csv", "results.csv")
    summary_txt = _manifest_output_path(run_dir, manifest, "summary_txt", "run_summary.txt")

    records: list[dict[str, str]] = []
    if results_csv.exists():
        with results_csv.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))

    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "records": records,
        "paths": {
            "run_manifest": run_dir / RUN_MANIFEST_NAME,
            "results_csv": results_csv,
            "summary_txt": summary_txt,
        },
        "counts": _manifest_counts(manifest, records),
        "summary": run_summary_metadata(run_dir, manifest),
    }


def duplicate_run_settings(run_dir: Path) -> dict[str, object] | None:
    run_dir = run_dir.resolve()
    try:
        manifest = _read_run_manifest(run_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return {
        "source_run_id": run_dir.name,
        "source_run_dir": str(run_dir),
        "source_created_at_utc": str(manifest.get("created_at_utc") or ""),
        "draft_created_at_utc": _now_utc_iso(),
        "inputs": deepcopy(manifest.get("inputs")) if isinstance(manifest.get("inputs"), dict) else {},
        "parameters": deepcopy(manifest.get("parameters")) if isinstance(manifest.get("parameters"), dict) else {},
    }


def create_stop_file(stop_file: Path, *, reason: str = "stop_requested") -> Path:
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"requested_at_utc": _now_utc_iso(), "reason": reason}
    stop_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return stop_file


def stop_requested(stop_file: Path | None) -> bool:
    return bool(stop_file and stop_file.exists())


def _validate_archive_member_name(name: str) -> PurePosixPath:
    if not name or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    posix_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError(f"Unsafe archive path: {name}")
    if any(part in ("", ".", "..") for part in posix_path.parts):
        raise ValueError(f"Unsafe archive path: {name}")
    return posix_path


def create_session_manifest(run_dir: Path, output_path: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    output_path = output_path or run_dir / SESSION_MANIFEST_NAME
    files = [
        str(path.relative_to(run_dir)).replace("\\", "/")
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.resolve() != output_path.resolve()
    ]
    run_manifest_path = run_dir / RUN_MANIFEST_NAME
    run_manifest = None
    if run_manifest_path.exists():
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))

    payload = {
        "session_version": 1,
        "created_at_utc": _now_utc_iso(),
        "run_dir": str(run_dir),
        "file_count": len(files),
        "files": files,
        "run_manifest": run_manifest,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def create_session_archive(run_dir: Path, output_zip: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    output_zip = output_zip or run_dir.with_suffix(".stratadock-session.zip")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    session_manifest = create_session_manifest(run_dir)
    output_zip_resolved = output_zip.resolve()

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and path.resolve() != output_zip_resolved:
                archive.write(path, path.relative_to(run_dir))
        if not (run_dir / SESSION_MANIFEST_NAME).exists():
            archive.write(session_manifest, SESSION_MANIFEST_NAME)
    return output_zip


def create_selected_session_archive(run_dir: Path, selected_paths: list[str | Path], output_zip: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    output_zip = output_zip or run_dir.with_suffix(".selected.stratadock-session.zip")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    output_zip_resolved = output_zip.resolve()

    archive_paths: list[tuple[Path, PurePosixPath]] = []
    for selected in selected_paths:
        archive_name = _validate_archive_member_name(str(selected).replace("\\", "/"))
        if archive_name.as_posix() == SESSION_MANIFEST_NAME:
            continue
        source = (run_dir / Path(*archive_name.parts)).resolve()
        if run_dir != source and run_dir not in source.parents:
            raise ValueError(f"Selected path is outside run directory: {selected}")
        if not source.exists() or not source.is_file() or source == output_zip_resolved:
            continue
        archive_paths.append((source, archive_name))

    run_manifest_path = run_dir / RUN_MANIFEST_NAME
    run_manifest = _read_json(run_manifest_path) if run_manifest_path.exists() else None
    selected_file_names = sorted({archive_name.as_posix() for _, archive_name in archive_paths})
    session_payload = {
        "session_version": 1,
        "created_at_utc": _now_utc_iso(),
        "run_dir": str(run_dir),
        "file_count": len(selected_file_names),
        "files": selected_file_names,
        "run_manifest": run_manifest,
    }

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(SESSION_MANIFEST_NAME, json.dumps(session_payload, indent=2) + "\n")
        for source, archive_name in archive_paths:
            archive.write(source, archive_name.as_posix())
    return output_zip


def read_session_archive(archive_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        if SESSION_MANIFEST_NAME in names:
            return json.loads(archive.read(SESSION_MANIFEST_NAME).decode("utf-8"))
        if "run_manifest.json" in names:
            run_manifest = json.loads(archive.read("run_manifest.json").decode("utf-8"))
            return {
                "session_version": 0,
                "file_count": len([name for name in names if not name.endswith("/")]),
                "files": sorted(names),
                "run_manifest": run_manifest,
            }
    raise ValueError(f"Archive does not contain {SESSION_MANIFEST_NAME} or run_manifest.json: {archive_path}")


def extract_session_archive(archive_path: Path, output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            _validate_archive_member_name(member.filename.rstrip("/"))
        for member in archive.infolist():
            archive_name = _validate_archive_member_name(member.filename.rstrip("/"))
            target = (output_dir / Path(*archive_name.parts)).resolve()
            if output_dir != target and output_dir not in target.parents:
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
    return output_dir
