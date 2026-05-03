import json
from pathlib import Path

import pytest

from stratadock.core.evaluate import heavy_atom_rmsd


ROOT = Path(__file__).resolve().parents[1]


def _project_path(value: str) -> Path:
    return ROOT / value.replace("\\", "/")


def test_existing_validation_runs_match_native_pose_when_present():
    result_paths = sorted((ROOT / "runs").glob("*/result.json"))
    if not result_paths:
        pytest.skip("Run scripts/validation/run_validation_case.py <case_id> first.")

    for result_path in result_paths:
        result = json.loads(result_path.read_text())
        if "case_id" not in result:
            continue
        manifest = json.loads(
            (ROOT / "data" / "validation" / result["case_id"] / "manifest.json").read_text()
        )
        assert isinstance(result["score"], float)
        assert _project_path(result["pose_pdbqt"]).exists()
        rmsd = heavy_atom_rmsd(
            ligand_input_sdf=_project_path(manifest["files"]["ligand_input_sdf"]),
            native_ligand_pdb=_project_path(manifest["files"]["native_ligand_pdb"]),
            docked_pose_pdbqt=_project_path(result["pose_pdbqt"]),
        )
        assert abs(rmsd - result["heavy_atom_rmsd"]) < 0.01
        assert result["passes_rmsd_threshold"] is True
        assert rmsd <= manifest["acceptance"]["pose_rmsd_angstrom_max"]
