import json
import subprocess
import sys
from pathlib import Path

import pytest

from stratadock.core.boxes import box_from_ligand_pdb
from stratadock.core.ligands import load_sdf, load_smiles, prepare_ligand_input, prepare_ligand_pdbqt
from stratadock.core.pdb import parse_pdb_atoms
from stratadock.core.receptors import clean_receptor_pdb, prepare_receptor_input, prepare_receptor_pdbqt
from stratadock.tools.binaries import vina_binary


ROOT = Path(__file__).resolve().parents[1]


def _requires_vina() -> None:
    try:
        vina_binary(ROOT)
    except FileNotFoundError as exc:
        pytest.skip(f"AutoDock Vina binary is not installed for integration test: {exc}")


def test_all_validation_cases_prepare_ligand_and_receptor(tmp_path):
    index = json.loads((ROOT / "data" / "validation" / "index.json").read_text())

    for case in index["cases"]:
        case_tmp = tmp_path / case["case_id"]
        mols = load_sdf(ROOT / case["files"]["ligand_input_sdf"])
        assert len(mols) == 1

        ligand_pdbqt = prepare_ligand_pdbqt(
            ROOT / case["files"]["ligand_input_sdf"],
            case_tmp / "ligand.pdbqt",
        )
        receptor_pdbqt = prepare_receptor_pdbqt(
            ROOT / case["files"]["receptor_pdb"],
            case_tmp / "receptor.pdbqt",
        )

        assert ligand_pdbqt.exists()
        assert ligand_pdbqt.stat().st_size > 100
        assert receptor_pdbqt.exists()
        assert receptor_pdbqt.stat().st_size > 100


def test_smiles_ligand_prep_embeds_3d_and_writes_pdbqt(tmp_path):
    smi = tmp_path / "ligands.smi"
    smi.write_text("CC(=O)OC1=CC=CC=C1C(=O)O aspirin\n", encoding="utf-8")

    records = load_smiles(smi)
    assert records[0].name == "aspirin"

    report = prepare_ligand_input(smi, tmp_path / "out")
    mols = load_sdf(report.prepared_sdf)

    assert report.embedding_used is True
    assert report.heavy_atoms == 13
    assert report.pdbqt.exists()
    assert report.pdbqt.stat().st_size > 100
    assert mols[0].GetNumConformers() == 1


def test_invalid_smiles_fails_cleanly(tmp_path):
    smi = tmp_path / "bad.smi"
    smi.write_text("C1CC bad_ring\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid SMILES"):
        load_smiles(smi)


def test_receptor_cleaning_removes_water_and_ligand_but_keeps_protein(tmp_path):
    complex_pdb = ROOT / "data" / "validation" / "hiv_protease_1hsg" / "1hsg_complex.pdb"
    cleaned = tmp_path / "cleaned.pdb"
    report = clean_receptor_pdb(complex_pdb, cleaned)
    atoms = parse_pdb_atoms(cleaned.read_text())

    assert report.atoms_kept > 1000
    assert report.hetero_removed >= 45
    assert report.source_atom_records >= report.atoms_kept
    assert report.source_hetatm_records >= report.hetero_removed
    assert report.protein_residue_count > 100
    assert report.chains
    assert "MK1" in report.hetero_residue_counts
    assert cleaned.exists()
    assert all(atom.residue_name != "MK1" for atom in atoms)


def test_all_validation_cases_prepare_through_high_level_api(tmp_path):
    index = json.loads((ROOT / "data" / "validation" / "index.json").read_text())

    for case in index["cases"]:
        case_tmp = tmp_path / "high_level" / case["case_id"]
        ligand = prepare_ligand_input(ROOT / case["files"]["ligand_input_sdf"], case_tmp)
        receptor = prepare_receptor_input(ROOT / case["files"]["complex_pdb"], case_tmp)

        assert ligand.pdbqt.exists()
        assert receptor.pdbqt.exists()
        assert receptor.report_json.exists()
        assert receptor.report_txt.exists()
        assert receptor.prepared_pdb.exists()
        assert receptor.options.keep_metals is True
        assert "clean" in receptor.prep_steps
        assert receptor.clean_report.hetero_removed >= ligand.heavy_atoms


def test_single_run_cli_with_smiles_ligand(tmp_path):
    _requires_vina()
    case_dir = ROOT / "data" / "validation" / "trypsin_3ptb"
    smi = tmp_path / "benzamidine.smi"
    # Benzamidine, matching the BEN validation ligand class.
    smi.write_text("NC(=N)c1ccccc1 benzamidine\n", encoding="utf-8")
    out_dir = tmp_path / "single"
    box = box_from_ligand_pdb(case_dir / "native_ligand.pdb")

    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "user" / "run_single.py"),
            "--receptor",
            str(case_dir / "3ptb_complex.pdb"),
            "--ligand",
            str(smi),
            "--box-center",
            str(box.center_x),
            str(box.center_y),
            str(box.center_z),
            "--box-size",
            str(box.size_x),
            str(box.size_y),
            str(box.size_z),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads((out_dir / "result.json").read_text())
    assert isinstance(result["docking"]["score"], float)
    assert Path(result["docking"]["pose_pdbqt"]).exists()
    assert Path(result["box_json"]).exists()
    assert Path(result["receptor"]["report_json"]).exists()
    assert result["receptor"]["protein_residue_count"] > 100
    assert result["ligand"]["embedding_used"] is True


def test_inspect_receptor_cli_writes_reports(tmp_path):
    receptor = ROOT / "data" / "validation" / "trypsin_3ptb" / "3ptb_complex.pdb"
    out_dir = tmp_path / "receptor_inspect"

    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "user" / "inspect_receptor.py"),
            "--receptor",
            str(receptor),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert run.returncode == 0, run.stdout + run.stderr
    summary = json.loads(run.stdout)
    report_json = Path(summary["report_json"])
    report_txt = Path(summary["report_txt"])
    assert report_json.exists()
    assert report_txt.exists()
    report = json.loads(report_json.read_text())
    assert report["atoms_kept"] > 1000
    assert report["protein_residue_count"] > 100
    assert "BEN" in report["hetero_residue_counts"]
