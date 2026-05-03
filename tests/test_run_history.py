import json
import zipfile

import pytest

from stratadock.core.history import duplicate_run_settings, import_session_zip, list_run_history, load_run


def _write_run(path, *, score=-7.0):
    path.mkdir(parents=True)
    (path / "results.json").write_text(
        json.dumps([{"ligand_name": "aspirin", "docking_status": "success", "vina_score": score}]),
        encoding="utf-8",
    )
    (path / "results.csv").write_text("ligand_name,vina_score\naspirin,-7.0\n", encoding="utf-8")
    (path / "run_summary.txt").write_text("summary\n", encoding="utf-8")
    (path / "run_manifest.json").write_text(
        json.dumps(
            {
                "created_at_utc": "2026-04-30T20:00:00+00:00",
                "inputs": {"receptor_pdb": "r.pdb", "ligand_file": "l.smi"},
                "parameters": {"exhaustiveness": 8, "seed": 1},
                "counts": {"total": 1, "success": 1, "failed_or_skipped": 0},
            }
        ),
        encoding="utf-8",
    )


def test_list_run_history_and_load_run(tmp_path):
    _write_run(tmp_path / "ui_1")
    (tmp_path / "not_a_run").mkdir()

    runs = list_run_history(tmp_path)
    loaded = load_run(runs[0].path)

    assert [run.name for run in runs] == ["ui_1"]
    assert loaded.counts == {"total": 1, "success": 1, "failed": 0}
    assert loaded.results[0]["ligand_name"] == "aspirin"


def test_duplicate_run_settings_returns_manifest_inputs_and_parameters(tmp_path):
    _write_run(tmp_path / "ui_1")

    draft = duplicate_run_settings(tmp_path / "ui_1")

    assert draft["inputs"]["receptor_pdb"] == "r.pdb"
    assert draft["parameters"]["exhaustiveness"] == 8


def test_import_session_zip_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../bad.txt", "oops")

    with pytest.raises(ValueError, match="Unsafe archive path"):
        import_session_zip(archive, tmp_path / "out")


def test_load_run_can_be_constrained_to_runs_directory(tmp_path):
    _write_run(tmp_path / "runs" / "ui_1")
    _write_run(tmp_path / "outside")

    loaded = load_run(tmp_path / "runs" / "ui_1", runs_dir=tmp_path / "runs")

    assert loaded.path == (tmp_path / "runs" / "ui_1").resolve()
    with pytest.raises(ValueError, match="outside runs directory"):
        load_run(tmp_path / "outside", runs_dir=tmp_path / "runs")


def test_run_history_includes_timing_from_events_log(tmp_path):
    run_dir = tmp_path / "ui_1"
    _write_run(run_dir)
    (run_dir / "run_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"time_utc": "2026-04-30T20:00:00+00:00", "event": "run_started"}),
                json.dumps({"time_utc": "2026-04-30T20:02:30+00:00", "event": "run_completed"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    entry = list_run_history(tmp_path)[0]
    loaded = load_run(run_dir)

    assert entry.started_at_utc == "2026-04-30T20:00:00+00:00"
    assert entry.ended_at_utc == "2026-04-30T20:02:30+00:00"
    assert entry.runtime_seconds == 150
    assert loaded.summary["runtime_seconds"] == 150


def test_duplicate_run_settings_returns_independent_draft_metadata(tmp_path):
    run_dir = tmp_path / "ui_1"
    _write_run(run_dir)

    draft = duplicate_run_settings(run_dir)
    draft["parameters"]["seed"] = 99
    second = duplicate_run_settings(run_dir)

    assert second["parameters"]["seed"] == 1
    assert second["source_run_id"] == "ui_1"
    assert "draft_created_at_utc" in second
