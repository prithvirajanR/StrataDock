from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from rdkit import Chem

from stratadock.core.models import DockingBox
from stratadock.core.pdb import PdbAtom, compute_box_from_pdb_ligand, parse_pdb_atoms

BOX_FIELDS = ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")


def box_from_ligand_pdb(native_ligand_pdb: Path, padding: float = 8.0) -> DockingBox:
    return compute_box_from_pdb_ligand(
        native_ligand_pdb.read_text(encoding="utf-8", errors="ignore"),
        padding=padding,
    )


def box_from_ligand_file(native_ligand: Path, padding: float = 8.0) -> DockingBox:
    suffix = native_ligand.suffix.lower()
    if suffix == ".pdb":
        return box_from_ligand_pdb(native_ligand, padding=padding)
    if suffix in {".sdf", ".mol"}:
        return _box_from_rdkit_ligand(native_ligand, padding=padding)
    if suffix == ".mol2":
        return _box_from_mol2(native_ligand, padding=padding)
    raise ValueError(f"Unsupported reference ligand file type: {native_ligand.suffix}")


def _box_from_rdkit_ligand(native_ligand: Path, *, padding: float) -> DockingBox:
    if native_ligand.suffix.lower() == ".sdf":
        supplier = Chem.SDMolSupplier(str(native_ligand), removeHs=False, sanitize=False)
        mol = next((item for item in supplier if item is not None), None)
    else:
        mol = Chem.MolFromMolFile(str(native_ligand), removeHs=False, sanitize=False)
    return _box_from_rdkit_mol(mol, native_ligand, padding=padding)


def _box_from_rdkit_mol(mol: Chem.Mol | None, native_ligand: Path, *, padding: float) -> DockingBox:
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Could not read 3D coordinates from reference ligand: {native_ligand}")
    conformer = mol.GetConformer()
    xs = [conformer.GetAtomPosition(index).x for index in range(mol.GetNumAtoms())]
    ys = [conformer.GetAtomPosition(index).y for index in range(mol.GetNumAtoms())]
    zs = [conformer.GetAtomPosition(index).z for index in range(mol.GetNumAtoms())]
    return _box_from_coordinates(xs, ys, zs, padding=padding)


def _box_from_mol2(native_ligand: Path, *, padding: float) -> DockingBox:
    mol = Chem.MolFromMol2File(str(native_ligand), removeHs=False, sanitize=False)
    if mol is not None and mol.GetNumConformers() > 0:
        return _box_from_rdkit_mol(mol, native_ligand, padding=padding)
    obabel = shutil.which("obabel")
    if not obabel:
        raise ValueError("MOL2 reference ligands require RDKit MOL2 support or OpenBabel.")
    with tempfile.TemporaryDirectory() as temp_dir:
        converted = Path(temp_dir) / "reference.sdf"
        result = subprocess.run(
            [obabel, "-imol2", str(native_ligand), "-osdf", "-O", str(converted)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0 or not converted.exists():
            raise RuntimeError(f"OpenBabel could not convert MOL2 reference ligand.\nSTDERR:\n{result.stderr}")
        return _box_from_rdkit_ligand(converted, padding=padding)


def _box_from_coordinates(xs: Sequence[float], ys: Sequence[float], zs: Sequence[float], *, padding: float) -> DockingBox:
    if not xs or not ys or not zs:
        raise ValueError("Reference ligand does not contain coordinates.")
    return validate_box(
        DockingBox(
            center_x=round((min(xs) + max(xs)) / 2, 3),
            center_y=round((min(ys) + max(ys)) / 2, 3),
            center_z=round((min(zs) + max(zs)) / 2, 3),
            size_x=round(max(xs) - min(xs) + padding, 3),
            size_y=round(max(ys) - min(ys) + padding, 3),
            size_z=round(max(zs) - min(zs) + padding, 3),
        )
    )


def box_from_residues_pdb(receptor_pdb: Path, residues: list[str], padding: float = 8.0) -> DockingBox:
    atoms = parse_pdb_atoms(receptor_pdb.read_text(encoding="utf-8", errors="ignore"))
    selected = select_residue_atoms(atoms, residues)
    if not selected:
        raise ValueError(f"No atoms matched residue selector(s): {', '.join(residues)}")
    xs = [atom.x for atom in selected]
    ys = [atom.y for atom in selected]
    zs = [atom.z for atom in selected]
    return validate_box(
        DockingBox(
            center_x=round((min(xs) + max(xs)) / 2, 3),
            center_y=round((min(ys) + max(ys)) / 2, 3),
            center_z=round((min(zs) + max(zs)) / 2, 3),
            size_x=round(max(xs) - min(xs) + padding, 3),
            size_y=round(max(ys) - min(ys) + padding, 3),
            size_z=round(max(zs) - min(zs) + padding, 3),
        )
    )


def select_residue_atoms(atoms: list[PdbAtom], residues: list[str]) -> list[PdbAtom]:
    selectors = {_normalize_residue_selector(item) for item in residues}
    selected: list[PdbAtom] = []
    for atom in atoms:
        chain = atom.chain_id or "_"
        keys = {
            atom.residue_seq,
            f"{chain}:{atom.residue_seq}",
            f"{atom.residue_name.upper()}:{chain}:{atom.residue_seq}",
        }
        if keys & selectors:
            selected.append(atom)
    return selected


def _normalize_residue_selector(selector: str) -> str:
    return selector.strip().upper().replace(" ", "")


def box_from_center_size(center: Sequence[float], size: Sequence[float]) -> DockingBox:
    if len(center) != 3:
        raise ValueError("Box center must have exactly 3 values: x y z.")
    if len(size) != 3:
        raise ValueError("Box size must have exactly 3 values: x y z.")
    return validate_box(
        DockingBox(
            center_x=float(center[0]),
            center_y=float(center[1]),
            center_z=float(center[2]),
            size_x=float(size[0]),
            size_y=float(size[1]),
            size_z=float(size[2]),
        )
    )


def box_from_mapping(data: Mapping[str, object]) -> DockingBox:
    missing = [field for field in BOX_FIELDS if field not in data]
    if missing:
        raise ValueError(f"Docking box JSON is missing required field(s): {', '.join(missing)}.")
    return validate_box(
        DockingBox(
            center_x=float(data["center_x"]),
            center_y=float(data["center_y"]),
            center_z=float(data["center_z"]),
            size_x=float(data["size_x"]),
            size_y=float(data["size_y"]),
            size_z=float(data["size_z"]),
        )
    )


def load_box_json(path: Path) -> DockingBox:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Docking box JSON must contain an object.")
    return box_from_mapping(data)


def write_box_json(box: DockingBox, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validate_box(box).as_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def validate_box(box: DockingBox) -> DockingBox:
    values = box.as_dict()
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")
    if box.size_x <= 0 or box.size_y <= 0 or box.size_z <= 0:
        raise ValueError("Docking box sizes must be positive.")
    if box.size_x > 80 or box.size_y > 80 or box.size_z > 80:
        raise ValueError("Docking box is suspiciously large; check units/input.")
    return box
