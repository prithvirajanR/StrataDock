from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stratadock.core.models import DockingBox


WATER_NAMES = {"HOH", "WAT", "SOL"}
COMMON_BUFFER_NAMES = {"SO4", "PO4", "EDO", "GOL", "PEG", "DMS", "ACT"}
COMMON_IONS = {
    "NA", "K", "CL", "CA", "MG", "MN", "ZN", "FE", "CU", "CO", "NI", "CD",
}


@dataclass(frozen=True)
class PdbAtom:
    line: str
    record: str
    atom_name: str
    residue_name: str
    chain_id: str
    residue_seq: str
    x: float
    y: float
    z: float
    element: str

    @property
    def residue_key(self) -> tuple[str, str, str]:
        return (self.residue_name, self.chain_id, self.residue_seq)


def parse_pdb_atoms(pdb_text: str) -> list[PdbAtom]:
    atoms: list[PdbAtom] = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            atoms.append(
                PdbAtom(
                    line=line,
                    record=line[0:6].strip(),
                    atom_name=line[12:16].strip(),
                    residue_name=line[17:20].strip(),
                    chain_id=line[21:22].strip(),
                    residue_seq=line[22:26].strip(),
                    x=float(line[30:38]),
                    y=float(line[38:46]),
                    z=float(line[46:54]),
                    element=line[76:78].strip() or line[12:16].strip()[0],
                )
            )
        except ValueError:
            continue
    return atoms


def choose_primary_ligand(atoms: list[PdbAtom], min_atoms: int = 8) -> tuple[str, str, str]:
    counts: dict[tuple[str, str, str], int] = {}
    for atom in atoms:
        if atom.record != "HETATM":
            continue
        residue_name = atom.residue_name.upper()
        if residue_name in WATER_NAMES or residue_name in COMMON_BUFFER_NAMES or residue_name in COMMON_IONS:
            continue
        counts[atom.residue_key] = counts.get(atom.residue_key, 0) + 1

    candidates = [(key, count) for key, count in counts.items() if count >= min_atoms]
    if not candidates:
        raise ValueError("No ligand-like HETATM residue found in PDB.")
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][0]


def split_receptor_and_ligand(
    pdb_text: str,
    ligand_key: tuple[str, str, str] | None = None,
) -> tuple[str, str, tuple[str, str, str]]:
    atoms = parse_pdb_atoms(pdb_text)
    if ligand_key is None:
        ligand_key = choose_primary_ligand(atoms)

    receptor_lines: list[str] = []
    ligand_lines: list[str] = []

    for line in pdb_text.splitlines():
        if line.startswith("ATOM"):
            receptor_lines.append(line)
            continue
        if not line.startswith("HETATM"):
            continue

        resname = line[17:20].strip()
        chain = line[21:22].strip()
        resseq = line[22:26].strip()
        if (resname, chain, resseq) == ligand_key:
            ligand_lines.append(line)

    if not receptor_lines:
        raise ValueError("No receptor ATOM records found.")
    if not ligand_lines:
        raise ValueError(f"Ligand {ligand_key} was not found.")

    return "\n".join(receptor_lines) + "\nEND\n", "\n".join(ligand_lines) + "\nEND\n", ligand_key


def compute_box_from_pdb_ligand(ligand_pdb_text: str, padding: float = 8.0) -> DockingBox:
    atoms = parse_pdb_atoms(ligand_pdb_text)
    if not atoms:
        raise ValueError("Cannot compute box: ligand has no parseable atoms.")

    xs = [atom.x for atom in atoms]
    ys = [atom.y for atom in atoms]
    zs = [atom.z for atom in atoms]

    return DockingBox(
        center_x=round((min(xs) + max(xs)) / 2, 3),
        center_y=round((min(ys) + max(ys)) / 2, 3),
        center_z=round((min(zs) + max(zs)) / 2, 3),
        size_x=round(max(xs) - min(xs) + padding, 3),
        size_y=round(max(ys) - min(ys) + padding, 3),
        size_z=round(max(zs) - min(zs) + padding, 3),
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

