from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from stratadock.core.boxes import box_from_ligand_file


def _write_sdf(path: Path) -> Path:
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=1) == 0
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()
    return path


def test_box_from_ligand_file_supports_sdf(tmp_path):
    box = box_from_ligand_file(_write_sdf(tmp_path / "native.sdf"), padding=6.0)

    assert box.size_x > 6.0
    assert box.size_y > 6.0
    assert box.size_z > 6.0
