import json
import subprocess
import sys
import zipfile
from pathlib import Path

from stratadock.core import session as session_module
from stratadock.core.session import (
    SESSION_MANIFEST_NAME,
    create_session_archive,
    extract_session_archive,
    read_session_archive,
)


ROOT = Path(__file__).resolve().parents[1]


def test_session_archive_contains_manifest_and_run_files(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.csv").write_text("ligand,score\nx,-1.0\n", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps({"counts": {"success": 1}}), encoding="utf-8")

    archive_path = create_session_archive(run_dir)
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())

    assert SESSION_MANIFEST_NAME in names
    assert "results.csv" in names
    manifest = read_session_archive(archive_path)
    assert manifest["file_count"] >= 2
    assert manifest["run_manifest"]["counts"]["success"] == 1

    extracted = extract_session_archive(archive_path, tmp_path / "imported")
    assert (extracted / "results.csv").exists()
    assert (extracted / SESSION_MANIFEST_NAME).exists()


def test_export_session_cli_inspects_archive(tmp_path):
    run_dir = tmp_path / "run_cli"
    run_dir.mkdir()
    (run_dir / "run_summary.txt").write_text("ok\n", encoding="utf-8")
    archive_path = tmp_path / "session.zip"

    export_run = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "user" / "export_session.py"), "--run-dir", str(run_dir), "--out", str(archive_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert export_run.returncode == 0, export_run.stdout + export_run.stderr
    assert archive_path.exists()

    inspect_run = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "user" / "export_session.py"), "--inspect", str(archive_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert inspect_run.returncode == 0, inspect_run.stdout + inspect_run.stderr
    assert "run_summary.txt" in inspect_run.stdout

    extract_dir = tmp_path / "restored"
    extract_run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "user" / "export_session.py"),
            "--inspect",
            str(archive_path),
            "--extract-to",
            str(extract_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert extract_run.returncode == 0, extract_run.stdout + extract_run.stderr
    assert (extract_dir / "run_summary.txt").exists()


def test_stop_file_helper_creates_and_checks_request(tmp_path):
    stop_file = tmp_path / "run" / "STOP"

    assert hasattr(session_module, "create_stop_file")
    assert hasattr(session_module, "stop_requested")
    assert session_module.stop_requested(stop_file) is False

    created = session_module.create_stop_file(stop_file, reason="user_cancelled")

    assert created == stop_file
    assert session_module.stop_requested(stop_file) is True
    payload = json.loads(stop_file.read_text(encoding="utf-8"))
    assert payload["reason"] == "user_cancelled"
    assert "requested_at_utc" in payload
