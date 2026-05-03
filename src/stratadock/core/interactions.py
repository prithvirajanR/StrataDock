from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rdkit import Chem

from stratadock.core.evaluate import docked_pose_mol
from stratadock.core.pdb import PdbAtom, parse_pdb_atoms


@dataclass(frozen=True)
class Interaction:
    residue_name: str
    chain_id: str
    residue_seq: str
    receptor_atom: str
    ligand_atom_index: int
    ligand_element: str
    distance_angstrom: float
    interaction_type: str


POLAR_ELEMENTS = {"N", "O", "S"}
POSITIVE_RESIDUES = {"ARG", "LYS", "HIS", "HIP"}
NEGATIVE_RESIDUES = {"ASP", "GLU"}
AROMATIC_RESIDUES = {"PHE", "TYR", "TRP", "HIS", "HID", "HIE", "HIP"}


def build_complex_pdb(*, receptor_pdb: Path, pose_pdbqt: Path, output_pdb: Path, ligand_resname: str = "LIG") -> Path:
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    receptor_lines = [
        line
        for line in receptor_pdb.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.startswith(("ATOM", "HETATM", "TER"))
    ]
    ligand = docked_pose_mol(pose_pdbqt)
    ligand_block = Chem.MolToPDBBlock(ligand)
    ligand_lines: list[str] = []
    for idx, line in enumerate(ligand_block.splitlines(), start=1):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atom_name = line[12:16]
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        element = (line[76:78].strip() or atom_name.strip()[:1]).rjust(2)
        ligand_lines.append(
            f"HETATM{idx:5d} {atom_name:<4s} {ligand_resname:>3s} Z   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element}"
        )
    output_pdb.write_text("\n".join(receptor_lines + ["TER"] + ligand_lines + ["END"]) + "\n", encoding="utf-8")
    return output_pdb


def analyze_interactions(*, receptor_pdb: Path, pose_pdbqt: Path, cutoff: float = 4.0) -> list[Interaction]:
    receptor_atoms = parse_pdb_atoms(receptor_pdb.read_text(encoding="utf-8", errors="ignore"))
    ligand = Chem.RemoveHs(docked_pose_mol(pose_pdbqt))
    conf = ligand.GetConformer()
    interactions: list[Interaction] = []

    for ligand_idx, ligand_atom in enumerate(ligand.GetAtoms()):
        ligand_element = ligand_atom.GetSymbol().upper()
        ligand_pos = conf.GetAtomPosition(ligand_idx)
        for receptor_atom in receptor_atoms:
            dist = (
                (ligand_pos.x - receptor_atom.x) ** 2
                + (ligand_pos.y - receptor_atom.y) ** 2
                + (ligand_pos.z - receptor_atom.z) ** 2
            ) ** 0.5
            if dist > cutoff:
                continue
            interaction_type = classify_interaction(receptor_atom, ligand_element, dist)
            if interaction_type is None:
                continue
            interactions.append(
                Interaction(
                    residue_name=receptor_atom.residue_name,
                    chain_id=receptor_atom.chain_id,
                    residue_seq=receptor_atom.residue_seq,
                    receptor_atom=receptor_atom.atom_name,
                    ligand_atom_index=ligand_idx,
                    ligand_element=ligand_element,
                    distance_angstrom=round(dist, 3),
                    interaction_type=interaction_type,
                )
            )
    interactions.sort(key=lambda item: (item.distance_angstrom, item.residue_name, item.residue_seq))
    return interactions


def classify_interaction(receptor_atom: PdbAtom, ligand_element: str, distance: float) -> str | None:
    receptor_element = receptor_atom.element.upper()
    residue = receptor_atom.residue_name.upper()
    if ((residue in POSITIVE_RESIDUES and ligand_element in {"O", "S"}) or (residue in NEGATIVE_RESIDUES and ligand_element in {"N"})) and distance <= 4.0:
        return "salt_bridge_candidate"
    if residue in AROMATIC_RESIDUES and ligand_element == "C" and distance <= 4.0:
        return "aromatic_contact"
    if receptor_element in POLAR_ELEMENTS and ligand_element in POLAR_ELEMENTS and distance <= 3.5:
        return "polar_contact"
    if receptor_element == "C" and ligand_element == "C" and distance <= 4.0:
        return "hydrophobic"
    return None


def write_interactions(interactions: list[Interaction], json_path: Path, csv_path: Path) -> tuple[Path, Path]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in interactions]
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else list(Interaction.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def write_pymol_script(*, complex_pdb: Path, output_pml: Path) -> Path:
    output_pml.parent.mkdir(parents=True, exist_ok=True)
    output_pml.write_text(
        "\n".join(
            [
                f"load {complex_pdb.as_posix()}, complex",
                "hide everything",
                "show cartoon, polymer.protein",
                "show sticks, resn LIG",
                "color slate, polymer.protein",
                "color yellow, resn LIG",
                "select binding_site, byres (polymer.protein within 4.0 of resn LIG)",
                "show sticks, binding_site",
                "color cyan, binding_site",
                "zoom resn LIG, 8",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output_pml
