from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from stratadock.core.pose_images import write_complex_pose_images, write_ligand_2d_images


def _write_ethanol_sdf(path: Path) -> Path:
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 7
    assert AllChem.EmbedMolecule(mol, params) == 0
    AllChem.UFFOptimizeMolecule(mol, maxIters=50)
    mol.SetProp("_Name", "ethanol")
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()
    return path


def _pdb_line(record: str, serial: int, atom: str, residue: str, chain: str, seq: int, x: float, y: float, z: float, element: str) -> str:
    return (
        f"{record:<6}{serial:5d} {atom:^4} {residue:>3} {chain}{seq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{20.00:6.2f}          {element:>2}"
    )


def test_write_ligand_2d_images_creates_png_and_jpg(tmp_path):
    sdf = _write_ethanol_sdf(tmp_path / "ethanol.sdf")

    images = write_ligand_2d_images(
        sdf,
        png_path=tmp_path / "ethanol_2d.png",
        jpg_path=tmp_path / "ethanol_2d.jpg",
        title="ethanol",
    )

    assert images["png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert images["jpg"].read_bytes().startswith(b"\xff\xd8\xff")
    assert images["png"].stat().st_size > 1000
    assert images["jpg"].stat().st_size > 1000


def test_write_complex_pose_images_creates_static_3d_png_and_jpg(tmp_path):
    complex_pdb = tmp_path / "complex.pdb"
    complex_pdb.write_text(
        "\n".join(
            [
                _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
                _pdb_line("ATOM", 2, "CB", "ALA", "A", 1, 1.6, 0.0, 0.0, "C"),
                _pdb_line("ATOM", 3, "O", "ALA", "A", 1, 0.0, 1.5, 0.0, "O"),
                _pdb_line("HETATM", 4, "C1", "LIG", "Z", 1, 2.8, 0.4, 0.2, "C"),
                _pdb_line("HETATM", 5, "C2", "LIG", "Z", 1, 4.1, 0.7, 0.4, "C"),
                _pdb_line("HETATM", 6, "O1", "LIG", "Z", 1, 4.8, 1.8, 0.8, "O"),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    images = write_complex_pose_images(
        complex_pdb,
        png_path=tmp_path / "pose_3d.png",
        jpg_path=tmp_path / "pose_3d.jpg",
        title="ALA / ethanol pose",
    )

    assert images["png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert images["jpg"].read_bytes().startswith(b"\xff\xd8\xff")
    assert images["png"].stat().st_size > 1000
    assert images["jpg"].stat().st_size > 1000
