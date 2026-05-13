from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from rdkit import RDLogger
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from stratadock.core.pdb import PdbAtom, is_ligand_like_hetatm, parse_pdb_atoms


ATOM_COLORS: dict[str, tuple[int, int, int]] = {
    "C": (126, 216, 113),
    "N": (91, 145, 255),
    "O": (255, 88, 94),
    "S": (244, 201, 78),
    "P": (255, 151, 76),
    "F": (113, 227, 185),
    "CL": (83, 207, 127),
    "BR": (180, 92, 58),
    "I": (150, 94, 196),
    "H": (220, 220, 220),
}

COVALENT_RADII: dict[str, float] = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "CL": 1.02,
    "BR": 1.20,
    "I": 1.39,
}


def write_ligand_2d_images(
    ligand_sdf: Path,
    *,
    png_path: Path,
    jpg_path: Path | None = None,
    title: str | None = None,
    size: tuple[int, int] = (900, 650),
) -> dict[str, Path]:
    """Write a clean 2D ligand depiction from the prepared SDF."""
    mol = _first_sdf_mol(ligand_sdf)
    drawing_mol = Chem.RemoveHs(Chem.Mol(mol))
    if drawing_mol.GetNumAtoms() == 0:
        raise ValueError(f"No drawable atoms found in {ligand_sdf}")
    AllChem.Compute2DCoords(drawing_mol)
    image = Draw.MolToImage(drawing_mol, size=size, legend=title or _mol_title(mol, ligand_sdf))
    return _write_image_variants(image, png_path=png_path, jpg_path=jpg_path)


def write_complex_pose_images(
    complex_pdb: Path,
    *,
    png_path: Path,
    jpg_path: Path | None = None,
    title: str | None = None,
    size: tuple[int, int] = (1200, 820),
    receptor_cutoff: float = 5.0,
) -> dict[str, Path]:
    """Write a static 3D-style binding pose image from a merged complex PDB."""
    atoms = parse_pdb_atoms(complex_pdb.read_text(encoding="utf-8", errors="ignore"))
    ligand_key = _choose_ligand_key(atoms)
    ligand_atoms = [
        atom
        for atom in atoms
        if atom.record == "HETATM" and atom.residue_key == ligand_key and atom.element.upper() != "H"
    ]
    if not ligand_atoms:
        raise ValueError(f"No ligand atoms found in {complex_pdb}")

    receptor_atoms = _nearby_receptor_atoms(atoms, ligand_key=ligand_key, ligand_atoms=ligand_atoms, cutoff=receptor_cutoff)
    image = _render_pose_image(
        ligand_atoms=ligand_atoms,
        receptor_atoms=receptor_atoms,
        title=title or f"{ligand_key[0]} pose",
        size=size,
    )
    return _write_image_variants(image, png_path=png_path, jpg_path=jpg_path)


def _first_sdf_mol(path: Path) -> Chem.Mol:
    RDLogger.DisableLog("rdApp.warning")
    try:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
        for mol in supplier:
            if mol is not None:
                return mol
        raise ValueError(f"No valid molecule found in {path}")
    finally:
        RDLogger.EnableLog("rdApp.warning")


def _mol_title(mol: Chem.Mol, path: Path) -> str:
    if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
        return mol.GetProp("_Name").strip()
    return path.stem


def _write_image_variants(image: Image.Image, *, png_path: Path, jpg_path: Path | None) -> dict[str, Path]:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path, format="PNG")
    written = {"png": png_path}
    if jpg_path is not None:
        jpg_path.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(jpg_path, format="JPEG", quality=92, optimize=True)
        written["jpg"] = jpg_path
    return written


def _choose_ligand_key(atoms: list[PdbAtom]) -> tuple[str, str, str]:
    counts: dict[tuple[str, str, str], int] = {}
    for atom in atoms:
        if atom.record != "HETATM":
            continue
        if atom.residue_name.upper() == "LIG":
            counts[atom.residue_key] = counts.get(atom.residue_key, 0) + 1
    if not counts:
        for atom in atoms:
            if is_ligand_like_hetatm(atom):
                counts[atom.residue_key] = counts.get(atom.residue_key, 0) + 1
    if not counts:
        raise ValueError("No ligand-like HETATM residue found in complex PDB.")
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _nearby_receptor_atoms(
    atoms: list[PdbAtom],
    *,
    ligand_key: tuple[str, str, str],
    ligand_atoms: list[PdbAtom],
    cutoff: float,
    max_atoms: int = 420,
) -> list[PdbAtom]:
    ligand_coords = np.array([[atom.x, atom.y, atom.z] for atom in ligand_atoms], dtype=float)
    nearby: list[tuple[float, PdbAtom]] = []
    for atom in atoms:
        if atom.record == "HETATM" and atom.residue_key == ligand_key:
            continue
        if atom.element.upper() == "H":
            continue
        coord = np.array([atom.x, atom.y, atom.z], dtype=float)
        min_dist = float(np.min(np.linalg.norm(ligand_coords - coord, axis=1)))
        if min_dist <= cutoff:
            nearby.append((min_dist, atom))
    nearby.sort(key=lambda item: item[0])
    return [atom for _distance, atom in nearby[:max_atoms]]


def _render_pose_image(
    *,
    ligand_atoms: list[PdbAtom],
    receptor_atoms: list[PdbAtom],
    title: str,
    size: tuple[int, int],
) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, (29, 29, 29))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()

    _draw_background_grid(draw, size)
    ligand_coords = np.array([[atom.x, atom.y, atom.z] for atom in ligand_atoms], dtype=float)
    receptor_coords = np.array([[atom.x, atom.y, atom.z] for atom in receptor_atoms], dtype=float) if receptor_atoms else np.empty((0, 3))
    axes = _projection_axes(ligand_coords)
    center = ligand_coords.mean(axis=0)
    ligand_projected = _project(ligand_coords, center, axes)
    receptor_projected = _project(receptor_coords, center, axes) if len(receptor_coords) else np.empty((0, 3))
    projector = _pixel_projector(ligand_projected, receptor_projected, size=size)

    if receptor_atoms:
        for atom, projected in sorted(zip(receptor_atoms, receptor_projected), key=lambda item: item[1][2]):
            x, y = projector(projected)
            radius = 3
            color = _receptor_color(atom.element)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    for i, j in _infer_ligand_bonds(ligand_atoms):
        p1 = ligand_projected[i]
        p2 = ligand_projected[j]
        x1, y1 = projector(p1)
        x2, y2 = projector(p2)
        draw.line((x1, y1, x2, y2), fill=(230, 230, 230, 230), width=5)
        draw.line((x1, y1, x2, y2), fill=(95, 210, 125, 245), width=3)

    depth_values = ligand_projected[:, 2]
    depth_min = float(depth_values.min()) if len(depth_values) else 0.0
    depth_span = max(float(depth_values.max() - depth_min), 0.001) if len(depth_values) else 1.0
    for atom, projected in sorted(zip(ligand_atoms, ligand_projected), key=lambda item: item[1][2]):
        x, y = projector(projected)
        depth_factor = 0.85 + 0.25 * ((float(projected[2]) - depth_min) / depth_span)
        radius = int(11 * depth_factor)
        color = _atom_color(atom.element, depth_factor)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, 255), outline=(245, 245, 245, 220), width=2)
        if atom.element.upper() != "C":
            label = atom.element.upper()
            draw.text((x - 4, y - 5), label, fill=(18, 18, 18, 240), font=font)

    draw.rectangle((0, 0, width, 72), fill=(24, 24, 24, 215))
    draw.text((28, 22), title, fill=(245, 245, 245, 255), font=title_font)
    subtitle = f"{len(ligand_atoms)} ligand atoms"
    if receptor_atoms:
        subtitle += f" | {len(receptor_atoms)} nearby receptor atoms"
    draw.text((28, 44), subtitle, fill=(160, 160, 160, 255), font=font)
    _draw_legend(draw, width=width, font=font)
    return image


def _draw_background_grid(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    width, height = size
    for x in range(0, width, 80):
        draw.line((x, 0, x, height), fill=(255, 255, 255, 8), width=1)
    for y in range(0, height, 80):
        draw.line((0, y, width, y), fill=(255, 255, 255, 8), width=1)


def _projection_axes(coords: np.ndarray) -> np.ndarray:
    if coords.shape[0] >= 3:
        centered = coords - coords.mean(axis=0)
        try:
            _u, _s, vh = np.linalg.svd(centered, full_matrices=True)
            axis_x = vh[0]
            axis_y = vh[1] if vh.shape[0] > 1 else np.array([0.0, 1.0, 0.0])
            axis_z = np.cross(axis_x, axis_y)
            if np.linalg.norm(axis_z) < 1e-6:
                axis_z = np.array([0.0, 0.0, 1.0])
            axis_z = axis_z / np.linalg.norm(axis_z)
            axis_y = np.cross(axis_z, axis_x)
            axis_y = axis_y / np.linalg.norm(axis_y)
            return np.vstack([axis_x / np.linalg.norm(axis_x), axis_y, axis_z])
        except np.linalg.LinAlgError:
            pass
    return np.eye(3)


def _project(coords: np.ndarray, center: np.ndarray, axes: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return np.empty((0, 3))
    return (coords - center) @ axes.T


def _pixel_projector(ligand_projected: np.ndarray, receptor_projected: np.ndarray, *, size: tuple[int, int]):
    width, height = size
    all_xy = ligand_projected[:, :2]
    if len(receptor_projected):
        all_xy = np.vstack([all_xy, receptor_projected[:, :2]])
    min_x, min_y = all_xy.min(axis=0) - 1.5
    max_x, max_y = all_xy.max(axis=0) + 1.5
    span_x = max(float(max_x - min_x), 1.0)
    span_y = max(float(max_y - min_y), 1.0)
    margin_x = 70
    top = 100
    bottom = 50
    scale = min((width - margin_x * 2) / span_x, (height - top - bottom) / span_y)
    used_w = span_x * scale
    used_h = span_y * scale
    offset_x = (width - used_w) / 2
    offset_y = top + ((height - top - bottom) - used_h) / 2

    def to_pixel(projected: np.ndarray) -> tuple[float, float]:
        x = offset_x + (float(projected[0]) - min_x) * scale
        y = offset_y + (max_y - float(projected[1])) * scale
        return x, y

    return to_pixel


def _infer_ligand_bonds(atoms: list[PdbAtom]) -> list[tuple[int, int]]:
    bonds: list[tuple[int, int]] = []
    for i, atom_i in enumerate(atoms):
        for j in range(i + 1, len(atoms)):
            atom_j = atoms[j]
            dist = math.dist((atom_i.x, atom_i.y, atom_i.z), (atom_j.x, atom_j.y, atom_j.z))
            cutoff = _bond_cutoff(atom_i.element, atom_j.element)
            if 0.35 <= dist <= cutoff:
                bonds.append((i, j))
    return bonds


def _bond_cutoff(element_a: str, element_b: str) -> float:
    radius_a = COVALENT_RADII.get(element_a.upper(), 0.77)
    radius_b = COVALENT_RADII.get(element_b.upper(), 0.77)
    return min(radius_a + radius_b + 0.45, 2.35)


def _atom_color(element: str, depth_factor: float = 1.0) -> tuple[int, int, int]:
    base = ATOM_COLORS.get(element.upper(), (190, 190, 190))
    return tuple(max(0, min(255, int(channel * depth_factor))) for channel in base)


def _receptor_color(element: str) -> tuple[int, int, int, int]:
    if element.upper() in {"N", "O", "S"}:
        return (120, 145, 170, 95)
    return (135, 135, 135, 70)


def _draw_legend(draw: ImageDraw.ImageDraw, *, width: int, font: ImageFont.ImageFont) -> None:
    items = [("Ligand", (95, 210, 125)), ("Nearby protein", (135, 135, 135))]
    x = width - 260
    y = 22
    for label, color in items:
        draw.ellipse((x, y + 1, x + 12, y + 13), fill=(*color, 220))
        draw.text((x + 20, y), label, fill=(190, 190, 190, 255), font=font)
        y += 22
