import json
from pathlib import Path

from rdkit import Chem

from stratadock.core.pdb import compute_box_from_pdb_ligand, parse_pdb_atoms


ROOT = Path(__file__).resolve().parents[1]


def test_validation_index_exists():
    assert (ROOT / "data" / "validation" / "index.json").exists()


def test_validation_cases_have_receptor_ligand_and_sane_box():
    index = json.loads((ROOT / "data" / "validation" / "index.json").read_text())

    assert index["cases"]
    for case in index["cases"]:
        receptor_path = ROOT / case["files"]["receptor_pdb"]
        ligand_path = ROOT / case["files"]["native_ligand_pdb"]
        ligand_sdf_path = ROOT / case["files"]["ligand_input_sdf"]

        receptor_atoms = parse_pdb_atoms(receptor_path.read_text())
        ligand_atoms = parse_pdb_atoms(ligand_path.read_text())
        box = compute_box_from_pdb_ligand(ligand_path.read_text())

        assert len(receptor_atoms) > 100
        assert len(ligand_atoms) >= 8
        assert box.size_x > 0
        assert box.size_y > 0
        assert box.size_z > 0
        assert ligand_sdf_path.exists()
        assert ligand_sdf_path.read_text().strip().endswith("$$$$")


def test_validation_manifests_are_instance_specific_and_consistent():
    index = json.loads((ROOT / "data" / "validation" / "index.json").read_text())
    seen_case_ids = set()

    for case in index["cases"]:
        assert case["case_id"] not in seen_case_ids
        seen_case_ids.add(case["case_id"])
        assert case["validation_type"] == "co_crystal_redocking"
        assert "models.rcsb.org" in case["ligand_sdf_source"]
        assert case["acceptance"]["pose_rmsd_angstrom_max"] > 0

        ligand_sdf_path = ROOT / case["files"]["ligand_input_sdf"]
        mol = Chem.SDMolSupplier(str(ligand_sdf_path), removeHs=False, sanitize=True)[0]
        assert mol is not None
        assert mol.GetNumConformers() == 1

        native_atoms = parse_pdb_atoms((ROOT / case["files"]["native_ligand_pdb"]).read_text())
        native_heavy = [atom for atom in native_atoms if atom.element.upper() != "H"]
        sdf_heavy = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
        assert len(native_heavy) == len(sdf_heavy)
