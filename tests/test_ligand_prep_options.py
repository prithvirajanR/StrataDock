import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdkit import Chem

from stratadock.core import batch as batch_module
from stratadock.core import ligands as ligand_module
from stratadock.core.batch import run_batch_screen
from stratadock.core.docking import DockingResult
from stratadock.core.ligands import LigandPrepOptions, LigandPrepReport, load_ligand_records_with_errors, load_sdf, prepare_ligand_input
from stratadock.core.models import DockingBox


ROOT = Path(__file__).resolve().parents[1]


def _fake_prepare_ligand_pdbqt(input_sdf: Path, output_pdbqt: Path) -> Path:
    output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    output_pdbqt.write_text("REMARK fake ligand pdbqt\n", encoding="utf-8")
    return output_pdbqt


def test_ligand_prep_defaults_keep_existing_behavior_and_report_options(tmp_path, monkeypatch):
    monkeypatch.setattr(ligand_module, "prepare_ligand_pdbqt", _fake_prepare_ligand_pdbqt)
    smi = tmp_path / "ligands.smi"
    smi.write_text("CCO ethanol\n", encoding="utf-8")

    report = prepare_ligand_input(smi, tmp_path / "out")

    assert report.embedding_used is True
    assert report.force_field == "MMFF94"
    assert report.requested_force_field == "MMFF94"
    assert report.protonation_ph is None
    assert report.protonation_status == "disabled"
    assert report.salt_stripping_status == "disabled"
    assert report.neutralization_status == "disabled"
    assert report.options.as_dict() == {
        "force_field": "MMFF94",
        "protonation_ph": None,
        "strip_salts": False,
        "neutralize": False,
    }
    assert report.pdbqt.exists()
    assert report.name.startswith("00001_")


def test_smiles_loader_preserves_names_with_spaces(tmp_path):
    smi = tmp_path / "names.smi"
    smi.write_text("CCO aspirin sodium salt\nCC ethyl acetate batch 7\n", encoding="utf-8")

    records, errors = load_ligand_records_with_errors(smi)

    assert errors == []
    assert [record.name for record in records] == ["aspirin sodium salt", "ethyl acetate batch 7"]


def test_ligand_prep_supports_force_field_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(ligand_module, "prepare_ligand_pdbqt", _fake_prepare_ligand_pdbqt)
    smi = tmp_path / "ligands.smi"
    smi.write_text("CCO ethanol\n", encoding="utf-8")

    report = prepare_ligand_input(
        smi,
        tmp_path / "out",
        options=LigandPrepOptions(force_field="UFF"),
    )

    assert report.force_field == "UFF"
    assert report.requested_force_field == "UFF"


def test_ligand_prep_rejects_unknown_force_field():
    with pytest.raises(ValueError, match="Unsupported force field"):
        LigandPrepOptions(force_field="BAD")


def test_ligand_prep_can_strip_salts_and_neutralize(tmp_path, monkeypatch):
    monkeypatch.setattr(ligand_module, "prepare_ligand_pdbqt", _fake_prepare_ligand_pdbqt)
    smi = tmp_path / "salt.smi"
    smi.write_text("C[NH3+].[Cl-] methylammonium_chloride\n", encoding="utf-8")

    report = prepare_ligand_input(
        smi,
        tmp_path / "out",
        options=LigandPrepOptions(strip_salts=True, neutralize=True),
    )
    prepared = load_sdf(report.prepared_sdf)[0]

    assert len(Chem.GetMolFrags(prepared)) == 1
    assert Chem.GetFormalCharge(prepared) == 0
    assert report.salt_stripping_status == "stripped"
    assert report.neutralization_status == "neutralized"


def test_ligand_prep_uses_obabel_for_requested_protonation_ph(tmp_path, monkeypatch):
    monkeypatch.setattr(ligand_module, "prepare_ligand_pdbqt", _fake_prepare_ligand_pdbqt)
    monkeypatch.setattr(ligand_module.shutil, "which", lambda name: "obabel" if name == "obabel" else None)
    calls = []

    def fake_run(cmd, capture_output, text, timeout):
        calls.append(cmd)
        output_path = Path(cmd[cmd.index("-O") + 1])
        input_path = Path(cmd[cmd.index("-isdf") + 1])
        output_path.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ligand_module.subprocess, "run", fake_run)
    smi = tmp_path / "ligands.smi"
    smi.write_text("CCO ethanol\n", encoding="utf-8")

    report = prepare_ligand_input(
        smi,
        tmp_path / "out",
        options=LigandPrepOptions(protonation_ph=6.5),
    )

    assert report.protonation_ph == 6.5
    assert report.protonation_status == "obabel_ph_6.5"
    assert calls and "-p" in calls[0]
    assert calls[0][calls[0].index("-p") + 1] == "6.5"


def test_ligand_prep_reports_when_requested_obabel_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(ligand_module, "prepare_ligand_pdbqt", _fake_prepare_ligand_pdbqt)
    monkeypatch.setattr(ligand_module.shutil, "which", lambda name: None)
    smi = tmp_path / "ligands.smi"
    smi.write_text("CCO ethanol\n", encoding="utf-8")

    report = prepare_ligand_input(
        smi,
        tmp_path / "out",
        options=LigandPrepOptions(protonation_ph=7.4),
    )

    assert report.protonation_status == "obabel_unavailable"
    assert report.prepared_sdf.exists()


def test_batch_screen_passes_ligand_prep_options_and_exports_statuses(tmp_path, monkeypatch):
    captured_options = []

    def fake_prepare_receptor(input_pdb: Path, output_dir: Path, **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        prepared_pdb = output_dir / "receptor.pdb"
        pdbqt = output_dir / "receptor.pdbqt"
        report_json = output_dir / "receptor_report.json"
        report_txt = output_dir / "receptor_report.txt"
        prepared_pdb.write_text("ATOM\n", encoding="utf-8")
        pdbqt.write_text("RECEPTOR\n", encoding="utf-8")
        report_json.write_text("{}\n", encoding="utf-8")
        report_txt.write_text("report\n", encoding="utf-8")
        return SimpleNamespace(prepared_pdb=prepared_pdb, pdbqt=pdbqt, report_json=report_json, report_txt=report_txt)

    def fake_prepare_ligand(record, output_dir: Path, *, options=None, force_field="MMFF94") -> LigandPrepReport:
        captured_options.append(options)
        output_dir.mkdir(parents=True, exist_ok=True)
        prepared_sdf = output_dir / f"{record.name}.prepared.sdf"
        pdbqt = output_dir / f"{record.name}.pdbqt"
        prepared_sdf.write_text("sdf\n", encoding="utf-8")
        pdbqt.write_text("ligand\n", encoding="utf-8")
        return LigandPrepReport(
            name=record.name,
            source_path=record.source_path,
            prepared_sdf=prepared_sdf,
            pdbqt=pdbqt,
            heavy_atoms=record.mol.GetNumHeavyAtoms(),
            embedding_used=True,
            force_field="UFF",
            requested_force_field="UFF",
            protonation_ph=6.5,
            protonation_status="obabel_ph_6.5",
            salt_stripping_status="stripped",
            neutralization_status="neutralized",
            options=LigandPrepOptions(force_field="UFF", protonation_ph=6.5, strip_salts=True, neutralize=True),
        )

    def fake_run_vina(*, output_pdbqt: Path, **_kwargs) -> DockingResult:
        output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
        output_pdbqt.write_text("REMARK VINA RESULT: -1.0 0.0 0.0\n", encoding="utf-8")
        return DockingResult(score=-1.0, scores=[-1.0], pose_pdbqt=output_pdbqt, log="fake")

    monkeypatch.setattr(batch_module, "prepare_receptor_input", fake_prepare_receptor)
    monkeypatch.setattr(batch_module, "prepare_ligand_record", fake_prepare_ligand)
    monkeypatch.setattr(batch_module, "run_vina", fake_run_vina)
    def fake_complex(*, output_pdb, **_kwargs):
        output_pdb.parent.mkdir(parents=True, exist_ok=True)
        output_pdb.write_text("END\n", encoding="utf-8")
        return output_pdb

    monkeypatch.setattr(batch_module, "build_complex_pdb", fake_complex)
    monkeypatch.setattr(batch_module, "analyze_interactions", lambda **_kwargs: [])

    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM\n", encoding="utf-8")
    ligands = tmp_path / "ligands.smi"
    ligands.write_text("CCO ethanol\n", encoding="utf-8")
    options = LigandPrepOptions(force_field="UFF", protonation_ph=6.5, strip_salts=True, neutralize=True)

    result = run_batch_screen(
        project_root=ROOT,
        receptor_pdb=receptor,
        ligand_file=ligands,
        box=DockingBox(0, 0, 0, 10, 10, 10),
        output_dir=tmp_path / "batch",
        ligand_prep_options=options,
    )

    row = result.results[0]
    assert captured_options == [options]
    assert row.ligand_force_field == "UFF"
    assert row.ligand_requested_force_field == "UFF"
    assert row.ligand_protonation_ph == 6.5
    assert row.ligand_protonation_status == "obabel_ph_6.5"
    assert row.ligand_salt_stripping_status == "stripped"
    assert row.ligand_neutralization_status == "neutralized"
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    assert manifest["parameters"]["ligand_prep"] == options.as_dict()
