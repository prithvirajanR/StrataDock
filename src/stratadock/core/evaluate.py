from __future__ import annotations

from pathlib import Path

from meeko import PDBQTMolecule, RDKitMolCreate
from rdkit import Chem


def _native_heavy_coords(native_ligand_pdb: Path) -> list[tuple[float, float, float]]:
    coords: list[tuple[float, float, float]] = []
    for line in native_ligand_pdb.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        element = (line[76:78].strip() or line[12:16].strip()[0]).upper()
        if element == "H":
            continue
        coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return coords


def native_pose_mol(ligand_input_sdf: Path, native_ligand_pdb: Path) -> Chem.Mol:
    mol = Chem.SDMolSupplier(str(ligand_input_sdf), removeHs=False, sanitize=True)[0]
    if mol is None:
        raise ValueError(f"Could not load ligand SDF: {ligand_input_sdf}")
    if mol.GetNumConformers():
        return mol

    coords = _native_heavy_coords(native_ligand_pdb)
    heavy_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if len(coords) != len(heavy_atoms):
        raise ValueError(
            f"Native heavy atom count mismatch: PDB={len(coords)} SDF={len(heavy_atoms)}"
        )

    conf = mol.GetConformer()
    for atom_idx, xyz in zip(heavy_atoms, coords):
        conf.SetAtomPosition(atom_idx, xyz)
    return mol


def docked_pose_mol(pose_pdbqt: Path) -> Chem.Mol:
    pdbqt = PDBQTMolecule.from_file(str(pose_pdbqt), skip_typing=True)
    mols = RDKitMolCreate.from_pdbqt_mol(pdbqt)
    if not mols:
        raise ValueError(f"Could not recover RDKit molecule from {pose_pdbqt}")
    return mols[0]


def heavy_atom_rmsd(
    *,
    ligand_input_sdf: Path,
    native_ligand_pdb: Path,
    docked_pose_pdbqt: Path,
) -> float:
    native = Chem.RemoveHs(native_pose_mol(ligand_input_sdf, native_ligand_pdb))
    docked = Chem.RemoveHs(docked_pose_mol(docked_pose_pdbqt))
    return _best_pose_space_rmsd(native, docked)


def _best_pose_space_rmsd(native: Chem.Mol, docked: Chem.Mol) -> float:
    if native.GetNumAtoms() != docked.GetNumAtoms():
        raise ValueError(
            f"Cannot compute RMSD: native has {native.GetNumAtoms()} heavy atoms, "
            f"docked pose has {docked.GetNumAtoms()}."
        )
    matches = docked.GetSubstructMatches(native, uniquify=False)
    if not matches:
        raise ValueError("Cannot compute RMSD: docked pose does not match native ligand graph.")

    native_conf = native.GetConformer()
    docked_conf = docked.GetConformer()
    best: float | None = None
    for match in matches:
        if len(match) != native.GetNumAtoms():
            continue
        squared = 0.0
        for native_idx, docked_idx in enumerate(match):
            native_pos = native_conf.GetAtomPosition(native_idx)
            docked_pos = docked_conf.GetAtomPosition(docked_idx)
            squared += (
                (native_pos.x - docked_pos.x) ** 2
                + (native_pos.y - docked_pos.y) ** 2
                + (native_pos.z - docked_pos.z) ** 2
            )
        rmsd = (squared / native.GetNumAtoms()) ** 0.5
        best = rmsd if best is None or rmsd < best else best
    if best is None:
        raise ValueError("Cannot compute RMSD: no complete atom mapping found.")
    return float(best)
