from pathlib import Path

import pytest

from stratadock.core.interactions import analyze_interactions, build_complex_pdb, write_interactions, write_pymol_script


ROOT = Path(__file__).resolve().parents[1]


def test_complex_interactions_and_pymol_artifacts_from_existing_pose(tmp_path):
    receptor = ROOT / "runs" / "trypsin_3ptb" / "receptor" / "receptor.cleaned.pdb"
    pose = ROOT / "runs" / "trypsin_3ptb" / "pose.pdbqt"
    if not receptor.exists():
        receptor = ROOT / "data" / "validation" / "trypsin_3ptb" / "receptor.pdb"
    if not pose.exists():
        pytest.skip("Generated validation pose is not present; run validation before this artifact test.")

    complex_pdb = build_complex_pdb(receptor_pdb=receptor, pose_pdbqt=pose, output_pdb=tmp_path / "complex.pdb")
    interactions = analyze_interactions(receptor_pdb=receptor, pose_pdbqt=pose)
    interactions_json, interactions_csv = write_interactions(
        interactions,
        tmp_path / "interactions.json",
        tmp_path / "interactions.csv",
    )
    pymol = write_pymol_script(complex_pdb=complex_pdb, output_pml=tmp_path / "view.pml")

    assert complex_pdb.exists()
    assert "HETATM" in complex_pdb.read_text()
    assert interactions
    assert interactions_json.exists()
    assert interactions_csv.exists()
    assert pymol.exists()
    assert "binding_site" in pymol.read_text()
