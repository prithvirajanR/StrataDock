import csv
import json
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from stratadock.core import batch as batch_module
from stratadock.core.batch import run_batch_screen, run_ensemble_screen
from stratadock.core.boxes import box_from_ligand_pdb, write_box_json
from stratadock.core.docking import DockingResult
from stratadock.core.ligands import LigandPrepOptions, LigandPrepReport
from stratadock.core.ligands import load_ligand_records_with_errors
from stratadock.core.models import DockingBox, NamedDockingBox
from stratadock.tools.binaries import vina_binary


ROOT = Path(__file__).resolve().parents[1]


def _requires_vina() -> None:
    try:
        vina_binary(ROOT)
    except FileNotFoundError as exc:
        pytest.skip(f"AutoDock Vina binary is not installed for integration test: {exc}")


def _write_smiles_batch(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "NC(=N)c1ccccc1 benzamidine",
                "C1CC bad_ligand",
                "CC(=O)OC1=CC=CC=C1C(=O)O aspirin",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_zip_ligands(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("set_a/ligands.smi", "NC(=N)c1ccccc1 benzamidine\nC1CC bad_ring\n")
        archive.writestr("notes/readme.txt", "# ignored comment\nCCO ethanol\n")
    return path


def _install_fast_batch_fakes(monkeypatch):
    state = {"active": 0, "max_active": 0, "vina_calls": 0}
    lock = threading.Lock()

    def fake_prepare_receptor(input_pdb: Path, output_dir: Path, **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        prepared_pdb = output_dir / "receptor.pdb"
        pdbqt = output_dir / "receptor.pdbqt"
        report_json = output_dir / "receptor_report.json"
        report_txt = output_dir / "receptor_report.txt"
        prepared_pdb.write_text("ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n", encoding="utf-8")
        pdbqt.write_text("RECEPTOR\n", encoding="utf-8")
        report_json.write_text("{}\n", encoding="utf-8")
        report_txt.write_text("report\n", encoding="utf-8")
        return SimpleNamespace(prepared_pdb=prepared_pdb, pdbqt=pdbqt, report_json=report_json, report_txt=report_txt)

    def fake_prepare_ligand(record, output_dir: Path, *, options=None, force_field="MMFF94") -> LigandPrepReport:
        options = options or LigandPrepOptions(force_field=force_field)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in record.name) or "ligand"
        if record.source_index > 1:
            safe_name = f"{safe_name}_{record.source_index}"
        prepared_sdf = output_dir / f"{safe_name}.prepared.sdf"
        pdbqt = output_dir / f"{safe_name}.pdbqt"
        prepared_sdf.write_text("sdf\n", encoding="utf-8")
        pdbqt.write_text("ligand\n", encoding="utf-8")
        return LigandPrepReport(
            name=safe_name,
            source_path=record.source_path,
            prepared_sdf=prepared_sdf,
            pdbqt=pdbqt,
            heavy_atoms=record.mol.GetNumHeavyAtoms(),
            embedding_used=False,
            force_field=options.force_field,
            requested_force_field=options.force_field,
            protonation_ph=options.protonation_ph,
            protonation_status="disabled" if options.protonation_ph is None else "obabel_unavailable",
            salt_stripping_status="stripped" if options.strip_salts else "disabled",
            neutralization_status="neutralized" if options.neutralize else "disabled",
            options=options,
        )

    def fake_run_vina(*, ligand_pdbqt: Path, output_pdbqt: Path, **_kwargs) -> DockingResult:
        if "fail_ligand" in ligand_pdbqt.stem:
            raise RuntimeError("synthetic docking failure")
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            state["vina_calls"] += 1
        try:
            time.sleep(0.05)
            score = -float(sum(ord(ch) for ch in ligand_pdbqt.stem) % 100)
            output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
            output_pdbqt.write_text(f"REMARK VINA RESULT: {score:.1f} 0.0 0.0\n", encoding="utf-8")
            return DockingResult(score=score, scores=[score], pose_pdbqt=output_pdbqt, log="fake")
        finally:
            with lock:
                state["active"] -= 1

    def fake_complex(*, output_pdb: Path, **_kwargs) -> Path:
        output_pdb.parent.mkdir(parents=True, exist_ok=True)
        output_pdb.write_text("END\n", encoding="utf-8")
        return output_pdb

    monkeypatch.setattr(batch_module, "prepare_receptor_input", fake_prepare_receptor)
    monkeypatch.setattr(batch_module, "prepare_ligand_record", fake_prepare_ligand)
    monkeypatch.setattr(batch_module, "run_vina", fake_run_vina)
    monkeypatch.setattr(batch_module, "build_complex_pdb", fake_complex)
    monkeypatch.setattr(batch_module, "analyze_interactions", lambda **_kwargs: [])
    return state


def _write_parallel_smiles(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "CCO ethanol",
                "C1CC invalid_ring",
                "CC ethane",
                "CCC fail_ligand",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_batch_screen_continues_after_invalid_ligand(tmp_path):
    _requires_vina()
    case_dir = ROOT / "data" / "validation" / "trypsin_3ptb"
    ligands = _write_smiles_batch(tmp_path / "batch.smi")
    out_dir = tmp_path / "batch_out"

    result = run_batch_screen(
        project_root=ROOT,
        receptor_pdb=case_dir / "3ptb_complex.pdb",
        ligand_file=ligands,
        box=box_from_ligand_pdb(case_dir / "native_ligand.pdb"),
        output_dir=out_dir,
    )

    assert result.results_csv.exists()
    assert result.results_json.exists()
    assert result.summary_txt.exists()
    assert result.box_json.exists()
    assert result.manifest_json.exists()
    assert result.best_by_ligand_csv.exists()
    assert result.best_by_pocket_csv.exists()
    assert result.html_report.exists()
    assert result.pdf_report.exists()
    assert result.events_log.exists()
    assert result.receptor.report_json.exists()
    assert result.receptor.report_txt.exists()
    assert len(result.results) == 3
    assert sum(1 for row in result.results if row.docking_status == "success") == 2
    assert sum(1 for row in result.results if row.prep_status == "load_failed") == 1
    for row in result.results:
        if row.docking_status == "success":
            assert row.complex_pdb and Path(row.complex_pdb).exists()
            assert row.interactions_json and Path(row.interactions_json).exists()
            assert row.interactions_csv and Path(row.interactions_csv).exists()
            assert row.pymol_script and Path(row.pymol_script).exists()
            assert row.viewer_html and Path(row.viewer_html).exists()

    rows = list(csv.DictReader(result.results_csv.open(newline="", encoding="utf-8")))
    assert set(rows[0]) >= {
        "ligand_name",
        "prep_status",
        "docking_status",
        "vina_score",
        "pose_pdbqt",
        "complex_pdb",
        "interactions_json",
        "pymol_script",
        "viewer_html",
        "molecular_weight",
        "qed",
    }


def test_batch_cli_writes_expected_outputs(tmp_path):
    _requires_vina()
    case_dir = ROOT / "data" / "validation" / "trypsin_3ptb"
    ligands = _write_smiles_batch(tmp_path / "batch_cli.smi")
    out_dir = tmp_path / "batch_cli_out"
    box_json = write_box_json(box_from_ligand_pdb(case_dir / "native_ligand.pdb"), tmp_path / "box.json")

    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "user" / "run_batch.py"),
            "--receptor",
            str(case_dir / "3ptb_complex.pdb"),
            "--ligands",
            str(ligands),
            "--box-json",
            str(box_json),
            "--out-dir",
            str(out_dir),
            "--n-jobs",
            "2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert run.returncode == 0, run.stdout + run.stderr
    cli_summary = json.loads(run.stdout)
    assert cli_summary["total"] == 3
    assert cli_summary["success"] == 2
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "results.json").exists()
    assert (out_dir / "run_summary.txt").exists()
    assert Path(cli_summary["best_by_ligand_csv"]).exists()
    assert Path(cli_summary["best_by_pocket_csv"]).exists()
    assert Path(cli_summary["html_report"]).exists()
    assert Path(cli_summary["pdf_report"]).exists()
    assert Path(cli_summary["events_log"]).exists()
    assert Path(cli_summary["box_json"]).exists()
    assert Path(cli_summary["manifest_json"]).exists()
    assert Path(cli_summary["receptor_report_json"]).exists()
    manifest = json.loads(Path(cli_summary["manifest_json"]).read_text())
    assert manifest["counts"]["total"] == 3
    assert manifest["counts"]["success"] == 2
    assert manifest["parameters"]["box"]["size_x"] > 0
    assert len(list((out_dir / "poses").rglob("*_pose.pdbqt"))) == 2


def test_parallel_batch_accepts_n_jobs_and_matches_sequential_shape(tmp_path, monkeypatch):
    state = _install_fast_batch_fakes(monkeypatch)
    ligands = _write_parallel_smiles(tmp_path / "parallel.smi")
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM\n", encoding="utf-8")
    box = DockingBox(0, 0, 0, 10, 10, 10)
    boxes = [
        NamedDockingBox(name="pocket_a", box=box, source="manual"),
        NamedDockingBox(name="pocket_b", box=box, source="manual"),
    ]

    sequential = run_batch_screen(
        project_root=ROOT,
        receptor_pdb=receptor,
        ligand_file=ligands,
        box=box,
        boxes=boxes,
        output_dir=tmp_path / "sequential",
        n_jobs=1,
    )
    parallel = run_batch_screen(
        project_root=ROOT,
        receptor_pdb=receptor,
        ligand_file=ligands,
        box=box,
        boxes=boxes,
        output_dir=tmp_path / "parallel",
        n_jobs=2,
    )

    seq_shape = [(row.pocket_name, row.ligand_name, row.prep_status, row.docking_status) for row in sequential.results]
    par_shape = [(row.pocket_name, row.ligand_name, row.prep_status, row.docking_status) for row in parallel.results]
    assert par_shape == seq_shape
    assert len(parallel.results) == 8
    assert sum(row.prep_status == "load_failed" for row in parallel.results) == 2
    assert sum(row.docking_status == "failed" for row in parallel.results) == 2
    assert state["max_active"] > 1


def test_batch_progress_callback_reports_ligand_level_progress(tmp_path, monkeypatch):
    _install_fast_batch_fakes(monkeypatch)
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n", encoding="utf-8")
    ligands = _write_parallel_smiles(tmp_path / "ligands.smi")
    messages: list[str] = []
    box = DockingBox(0, 0, 0, 12, 12, 12)

    run_batch_screen(
        project_root=ROOT,
        receptor_pdb=receptor,
        ligand_file=ligands,
        box=box,
        output_dir=tmp_path / "progress_out",
        boxes=[NamedDockingBox(name="pocket_a", box=box, source="test")],
        progress_callback=messages.append,
    )

    assert any("Preparing receptor" in message for message in messages)
    assert any("Loaded 3 valid ligand record(s); 1 invalid" in message for message in messages)
    assert any("Docking 1/3" in message and "pocket_a" in message for message in messages)
    assert any("Finished 3/3" in message for message in messages)
    assert any("Writing result tables and reports" in message for message in messages)


def test_batch_screen_can_use_gnina_engine_and_records_cnn_scores(tmp_path, monkeypatch):
    _install_fast_batch_fakes(monkeypatch)
    calls = []

    def fake_run_docking(*, engine, output_pdbqt: Path, gnina_cpu_only, gnina_device, **_kwargs) -> DockingResult:
        calls.append({"engine": engine, "gnina_cpu_only": gnina_cpu_only, "gnina_device": gnina_device})
        output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
        output_pdbqt.write_text("> <minimizedAffinity>\n-8.5\n\n> <CNNscore>\n0.77\n\n> <CNNaffinity>\n6.2\n", encoding="utf-8")
        return DockingResult(
            score=-8.5,
            scores=[-8.5],
            pose_pdbqt=output_pdbqt,
            log="fake gnina",
            engine="gnina",
            cnn_scores=[0.77],
            cnn_affinities=[6.2],
        )

    monkeypatch.setattr(batch_module, "run_docking", fake_run_docking)
    ligands = tmp_path / "one.smi"
    ligands.write_text("CCO ethanol\n", encoding="utf-8")
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM\n", encoding="utf-8")

    result = run_batch_screen(
        project_root=ROOT,
        receptor_pdb=receptor,
        ligand_file=ligands,
        box=DockingBox(0, 0, 0, 10, 10, 10),
        output_dir=tmp_path / "gnina",
        engine="gnina",
        gnina_cpu_only=True,
        gnina_device="0",
    )

    assert calls == [{"engine": "gnina", "gnina_cpu_only": True, "gnina_device": "0"}]
    assert result.results[0].docking_status == "success"
    assert result.results[0].docking_engine == "gnina"
    assert result.results[0].vina_score == -8.5
    assert result.results[0].cnn_score == 0.77
    assert result.results[0].cnn_affinity == 6.2
    rows = list(csv.DictReader(result.results_csv.open(newline="", encoding="utf-8")))
    assert rows[0]["vina_score"] == "-8.5"
    assert rows[0]["docking_score"] == "-8.5"
    assert rows[0]["score_type"] == "gnina_affinity"
    assert rows[0]["score_label"] == "GNINA affinity"
    json_rows = json.loads(result.results_json.read_text(encoding="utf-8"))
    assert json_rows[0]["docking_score"] == -8.5
    assert json_rows[0]["score_type"] == "gnina_affinity"
    best_rows = list(csv.DictReader(result.best_by_ligand_csv.open(newline="", encoding="utf-8")))
    assert best_rows[0]["docking_score"] == "-8.5"
    assert best_rows[0]["score_label"] == "GNINA affinity"
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    assert manifest["parameters"]["engine"] == "gnina"
    assert manifest["parameters"]["gnina_cpu_only"] is True


def test_parallel_batch_resume_reuses_existing_pose_without_redocking(tmp_path, monkeypatch):
    state = _install_fast_batch_fakes(monkeypatch)
    ligands = tmp_path / "one.smi"
    ligands.write_text("CCO ethanol\n", encoding="utf-8")
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM\n", encoding="utf-8")
    box = DockingBox(0, 0, 0, 10, 10, 10)
    out_dir = tmp_path / "resume_parallel"

    first = run_batch_screen(
        project_root=ROOT,
        receptor_pdb=receptor,
        ligand_file=ligands,
        box=box,
        output_dir=out_dir,
        n_jobs=1,
    )
    calls_after_first = state["vina_calls"]
    second = run_batch_screen(
        project_root=ROOT,
        receptor_pdb=receptor,
        ligand_file=ligands,
        box=box,
        output_dir=out_dir,
        n_jobs=2,
        resume=True,
    )

    assert first.results[0].docking_status == "success"
    assert second.results[0].docking_status == "success"
    assert second.results[0].prep_status == "cached"
    assert second.results[0].pose_pdbqt == first.results[0].pose_pdbqt
    assert state["vina_calls"] == calls_after_first


def test_ensemble_screen_accepts_and_passes_n_jobs(tmp_path, monkeypatch):
    _install_fast_batch_fakes(monkeypatch)
    ligands = tmp_path / "one.smi"
    ligands.write_text("CCO ethanol\n", encoding="utf-8")
    receptors = [tmp_path / "receptor_a.pdb", tmp_path / "receptor_b.pdb"]
    for receptor in receptors:
        receptor.write_text("ATOM\n", encoding="utf-8")
    box = DockingBox(0, 0, 0, 10, 10, 10)

    result = run_ensemble_screen(
        project_root=ROOT,
        receptor_pdbs=receptors,
        ligand_file=ligands,
        box=box,
        output_dir=tmp_path / "ensemble_parallel",
        n_jobs=2,
    )

    assert len(result.runs) == 2
    assert len(result.results) == 2
    for run in result.runs:
        manifest = json.loads(run.manifest_json.read_text(encoding="utf-8"))
        assert manifest["parameters"]["n_jobs"] == 2


def test_zip_ligand_library_loads_valid_records_and_errors(tmp_path):
    archive = _write_zip_ligands(tmp_path / "ligands.zip")

    records, errors = load_ligand_records_with_errors(archive)

    assert [record.name for record in records] == ["benzamidine"]
    assert len(errors) == 1
    assert "Invalid SMILES" in errors[0].message


def test_empty_results_csv_still_has_headers(tmp_path):
    path = batch_module.write_results_csv([], tmp_path / "results.csv")

    header = path.read_text(encoding="utf-8").splitlines()[0]

    assert "ligand_name" in header
    assert "docking_status" in header


def test_best_summary_keeps_zero_score_over_positive_score(tmp_path):
    base = batch_module._error_result("ligand", 1, "success", "", receptor_name="r", pocket_name="p")
    zero = replace(base, docking_status="success", vina_score=0.0, vina_scores=[0.0], docking_engine="vina", error=None)
    positive = replace(base, docking_status="success", vina_score=5.0, vina_scores=[5.0], docking_engine="vina", error=None)

    path = batch_module.write_best_summary_csv([zero, positive], tmp_path / "best.csv", group_field="ligand_name")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))

    assert len(rows) == 1
    assert float(rows[0]["vina_score"]) == 0.0


def test_write_summary_counts_unfavorable_positive_scores(tmp_path):
    base = batch_module._error_result("ligand", 1, "success", "", receptor_name="r", pocket_name="p")
    good = replace(base, docking_status="success", vina_score=-7.0, vina_scores=[-7.0], docking_engine="vina", error=None)
    bad = replace(base, docking_status="success", vina_score=3.2, vina_scores=[3.2], docking_engine="vina", error=None)

    path = batch_module.write_summary([good, bad], tmp_path / "summary.txt")

    text = path.read_text(encoding="utf-8")
    assert "Unfavorable non-negative scores: 1" in text
    assert "These completed but should be treated as low-confidence/clashing poses." in text


def test_ensemble_screen_can_use_per_receptor_boxes(tmp_path, monkeypatch):
    _install_fast_batch_fakes(monkeypatch)
    ligands = tmp_path / "one.smi"
    ligands.write_text("CCO ethanol\n", encoding="utf-8")
    receptors = [tmp_path / "receptor_a.pdb", tmp_path / "receptor_b.pdb"]
    for receptor in receptors:
        receptor.write_text("ATOM\n", encoding="utf-8")
    box_a = DockingBox(1, 1, 1, 10, 10, 10)
    box_b = DockingBox(2, 2, 2, 10, 10, 10)

    result = run_ensemble_screen(
        project_root=ROOT,
        receptor_pdbs=receptors,
        ligand_file=ligands,
        box=box_a,
        output_dir=tmp_path / "ensemble_per_receptor_boxes",
        boxes_by_receptor=[
            [NamedDockingBox(name="receptor_a_pocket", box=box_a, source="fpocket", rank=1)],
            [NamedDockingBox(name="receptor_b_pocket", box=box_b, source="fpocket", rank=1)],
        ],
    )

    assert [row.pocket_name for row in result.results] == ["receptor_a_pocket", "receptor_b_pocket"]
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    assert {box["name"] for box in manifest["parameters"]["boxes"]} == {
        "receptor_a:receptor_a_pocket",
        "receptor_b:receptor_b_pocket",
    }


def test_ensemble_screen_combines_multiple_receptors(tmp_path):
    _requires_vina()
    case_dir = ROOT / "data" / "validation" / "trypsin_3ptb"
    ligands = tmp_path / "one.smi"
    ligands.write_text("NC(=N)c1ccccc1 benzamidine\n", encoding="utf-8")
    box = box_from_ligand_pdb(case_dir / "native_ligand.pdb")

    result = run_ensemble_screen(
        project_root=ROOT,
        receptor_pdbs=[case_dir / "3ptb_complex.pdb", case_dir / "receptor.pdb"],
        ligand_file=ligands,
        box=box,
        output_dir=tmp_path / "ensemble",
        exhaustiveness=1,
    )

    assert len(result.runs) == 2
    assert len(result.results) == 2
    assert sum(row.docking_status == "success" for row in result.results) == 2
    assert {row.receptor_name for row in result.results} == {"3ptb_complex", "receptor"}
    assert result.manifest_json.exists()
    assert result.best_by_ligand_csv.exists()
    assert result.html_report.exists()
    assert result.pdf_report.exists()
    assert result.events_log.exists()


def test_batch_resume_reuses_existing_pose(tmp_path):
    _requires_vina()
    case_dir = ROOT / "data" / "validation" / "trypsin_3ptb"
    ligands = tmp_path / "one.smi"
    ligands.write_text("NC(=N)c1ccccc1 benzamidine\n", encoding="utf-8")
    out_dir = tmp_path / "resume"
    box = box_from_ligand_pdb(case_dir / "native_ligand.pdb")

    first = run_batch_screen(
        project_root=ROOT,
        receptor_pdb=case_dir / "3ptb_complex.pdb",
        ligand_file=ligands,
        box=box,
        output_dir=out_dir,
        exhaustiveness=1,
    )
    second = run_batch_screen(
        project_root=ROOT,
        receptor_pdb=case_dir / "3ptb_complex.pdb",
        ligand_file=ligands,
        box=box,
        output_dir=out_dir,
        exhaustiveness=1,
        resume=True,
    )

    assert first.results[0].docking_status == "success"
    assert second.results[0].docking_status == "success"
    assert second.results[0].prep_status == "cached"
    assert second.results[0].pose_pdbqt == first.results[0].pose_pdbqt


def test_batch_stop_file_cancels_before_ligands(tmp_path):
    _requires_vina()
    case_dir = ROOT / "data" / "validation" / "trypsin_3ptb"
    ligands = tmp_path / "one.smi"
    ligands.write_text("NC(=N)c1ccccc1 benzamidine\n", encoding="utf-8")
    stop_file = tmp_path / "STOP"
    stop_file.write_text("stop\n", encoding="utf-8")

    result = run_batch_screen(
        project_root=ROOT,
        receptor_pdb=case_dir / "3ptb_complex.pdb",
        ligand_file=ligands,
        box=box_from_ligand_pdb(case_dir / "native_ligand.pdb"),
        output_dir=tmp_path / "cancelled",
        exhaustiveness=1,
        stop_file=stop_file,
        n_jobs=2,
    )

    assert result.results == []
    assert result.events_log.exists()
    assert "run_cancelled" in result.events_log.read_text(encoding="utf-8")
