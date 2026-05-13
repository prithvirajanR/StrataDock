from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rdkit import RDLogger
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

CONTACT_COLORS: dict[str, tuple[int, int, int]] = {
    "polar_contact": (84, 205, 255),
    "hydrophobic": (245, 185, 77),
    "aromatic_contact": (190, 125, 255),
    "salt_bridge_candidate": (255, 96, 96),
}


@dataclass(frozen=True)
class PoseContact:
    interaction_type: str
    residue_name: str
    chain_id: str
    residue_seq: str
    receptor_atom: str
    ligand_atom_index: int
    ligand_element: str
    distance_angstrom: float
    residue_label: str
    label: str


@dataclass(frozen=True)
class NearbyResidue:
    residue_name: str
    chain_id: str
    residue_seq: str
    residue_label: str
    closest_atom: str
    distance_angstrom: float
    atom_count: int


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
    interactions: list[object] | None = None,
    score: float | None = None,
    size: tuple[int, int] = (1600, 1000),
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
    contacts = summarize_pose_contacts(interactions or [])
    image = _render_pose_image(
        ligand_atoms=ligand_atoms,
        receptor_atoms=receptor_atoms,
        contacts=contacts,
        title=title or f"{ligand_key[0]} pose",
        score=score,
        size=size,
    )
    return _write_image_variants(image, png_path=png_path, jpg_path=jpg_path)


def summarize_pose_contacts(interactions: list[object], *, limit: int = 8) -> list[PoseContact]:
    contacts: list[PoseContact] = []
    for row in interactions:
        distance = _float_value(row, "distance_angstrom")
        ligand_index = _int_value(row, "ligand_atom_index")
        residue_name = str(_row_value(row, "residue_name") or "").strip()
        chain_id = str(_row_value(row, "chain_id") or "").strip()
        residue_seq = str(_row_value(row, "residue_seq") or "").strip()
        receptor_atom = str(_row_value(row, "receptor_atom") or "").strip()
        ligand_element = str(_row_value(row, "ligand_element") or "").strip()
        interaction_type = str(_row_value(row, "interaction_type") or _row_value(row, "type") or "contact").strip()
        if distance is None or ligand_index is None or not residue_name or not residue_seq:
            continue
        residue_label = _residue_label(residue_name, chain_id, residue_seq)
        type_label = interaction_type.replace("_", " ")
        type_label = type_label[:1].upper() + type_label[1:]
        ligand_label = f"{ligand_element or 'Lig'}{ligand_index}"
        label = f"{type_label} | {ligand_label} -> {residue_label} {receptor_atom} | {distance:.2f} A"
        contacts.append(
            PoseContact(
                interaction_type=interaction_type,
                residue_name=residue_name,
                chain_id=chain_id,
                residue_seq=residue_seq,
                receptor_atom=receptor_atom,
                ligand_atom_index=ligand_index,
                ligand_element=ligand_element,
                distance_angstrom=distance,
                residue_label=residue_label,
                label=label,
            )
        )
    contacts.sort(key=lambda item: (item.distance_angstrom, item.residue_label, item.receptor_atom))
    return contacts[:limit]


def summarize_nearby_residues(
    receptor_atoms: list[PdbAtom],
    ligand_atoms: list[PdbAtom],
    *,
    limit: int = 8,
) -> list[NearbyResidue]:
    if not receptor_atoms or not ligand_atoms:
        return []
    ligand_coords = np.array([[atom.x, atom.y, atom.z] for atom in ligand_atoms], dtype=float)
    residue_map: dict[tuple[str, str, str], dict[str, object]] = {}
    for atom in receptor_atoms:
        coord = np.array([atom.x, atom.y, atom.z], dtype=float)
        min_dist = round(float(np.min(np.linalg.norm(ligand_coords - coord, axis=1))), 3)
        key = atom.residue_key
        current = residue_map.get(key)
        if current is None:
            residue_map[key] = {"distance": min_dist, "closest_atom": atom.atom_name, "count": 1, "atom": atom}
            continue
        current["count"] = int(current["count"]) + 1
        if min_dist < float(current["distance"]):
            current["distance"] = min_dist
            current["closest_atom"] = atom.atom_name
            current["atom"] = atom
    residues: list[NearbyResidue] = []
    for key, value in residue_map.items():
        atom = value["atom"]
        assert isinstance(atom, PdbAtom)
        residues.append(
            NearbyResidue(
                residue_name=key[0],
                chain_id=key[1],
                residue_seq=key[2],
                residue_label=_residue_label(key[0], key[1], key[2]),
                closest_atom=str(value["closest_atom"]),
                distance_angstrom=float(value["distance"]),
                atom_count=int(value["count"]),
            )
        )
    residues.sort(key=lambda residue: (residue.distance_angstrom, residue.residue_label))
    return residues[:limit]


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
    contacts: list[PoseContact],
    title: str,
    score: float | None,
    size: tuple[int, int],
) -> Image.Image:
    width, height = size
    panel_width = 520
    image = Image.new("RGB", size, (29, 29, 29))
    draw = ImageDraw.Draw(image, "RGBA")
    font = _load_font(13)
    title_font = _load_font(17, bold=True)

    _draw_background_grid(draw, size)
    ligand_coords = np.array([[atom.x, atom.y, atom.z] for atom in ligand_atoms], dtype=float)
    receptor_coords = np.array([[atom.x, atom.y, atom.z] for atom in receptor_atoms], dtype=float) if receptor_atoms else np.empty((0, 3))
    nearby_residues = summarize_nearby_residues(receptor_atoms, ligand_atoms)
    axes = _projection_axes(ligand_coords)
    center = ligand_coords.mean(axis=0)
    ligand_projected = _project(ligand_coords, center, axes)
    receptor_projected = _project(receptor_coords, center, axes) if len(receptor_coords) else np.empty((0, 3))
    projector = _pixel_projector(ligand_projected, receptor_projected, size=size, right_panel_width=panel_width)
    receptor_pixel_map = _atom_pixel_map(receptor_atoms, receptor_projected, projector)
    ligand_pixels = [projector(projected) for projected in ligand_projected]

    if receptor_atoms:
        for atom, projected in sorted(zip(receptor_atoms, receptor_projected), key=lambda item: item[1][2]):
            x, y = projector(projected)
            radius = 4 if _atom_has_contact(atom, contacts) else 3
            color = _receptor_color(atom.element)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    _draw_contacts(draw, contacts=contacts, receptor_pixel_map=receptor_pixel_map, ligand_pixels=ligand_pixels, font=font, plot_right=width - panel_width - 18)

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
    if contacts:
        subtitle += f" | {len(contacts)} highlighted contacts"
    draw.text((28, 44), subtitle, fill=(160, 160, 160, 255), font=font)
    _draw_contact_panel(
        draw,
        contacts=contacts,
        nearby_residues=nearby_residues,
        ligand_atom_count=len(ligand_atoms),
        receptor_atom_count=len(receptor_atoms),
        score=score,
        x0=width - panel_width,
        width=panel_width,
        height=height,
        font=font,
    )
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


def _pixel_projector(
    ligand_projected: np.ndarray,
    receptor_projected: np.ndarray,
    *,
    size: tuple[int, int],
    right_panel_width: int,
):
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
    plot_width = width - right_panel_width
    scale = min((plot_width - margin_x * 2) / span_x, (height - top - bottom) / span_y)
    used_w = span_x * scale
    used_h = span_y * scale
    offset_x = (plot_width - used_w) / 2
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
    items = [("Ligand", (95, 210, 125)), ("Contact", (84, 205, 255)), ("Nearby protein", (135, 135, 135))]
    x = width - 300
    y = 22
    for label, color in items:
        draw.ellipse((x, y + 1, x + 12, y + 13), fill=(*color, 220))
        draw.text((x + 20, y), label, fill=(190, 190, 190, 255), font=font)
        y += 22


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _row_value(row: object, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, None)


def _float_value(row: object, key: str) -> float | None:
    value = _row_value(row, key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_value(row: object, key: str) -> int | None:
    value = _row_value(row, key)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _residue_label(residue_name: str, chain_id: str, residue_seq: str) -> str:
    chain_part = f" {chain_id}{residue_seq}" if chain_id else f" {residue_seq}"
    return f"{residue_name}{chain_part}"


def _contact_color(interaction_type: str) -> tuple[int, int, int]:
    return CONTACT_COLORS.get(interaction_type, (210, 210, 210))


def _atom_key(atom: PdbAtom) -> tuple[str, str, str, str]:
    return (atom.residue_name, atom.chain_id, atom.residue_seq, atom.atom_name)


def _contact_atom_key(contact: PoseContact) -> tuple[str, str, str, str]:
    return (contact.residue_name, contact.chain_id, contact.residue_seq, contact.receptor_atom)


def _atom_pixel_map(
    atoms: list[PdbAtom],
    projected: np.ndarray,
    projector,
) -> dict[tuple[str, str, str, str], tuple[float, float]]:
    return {_atom_key(atom): projector(point) for atom, point in zip(atoms, projected)}


def _atom_has_contact(atom: PdbAtom, contacts: list[PoseContact]) -> bool:
    key = _atom_key(atom)
    return any(_contact_atom_key(contact) == key for contact in contacts)


def _draw_contacts(
    draw: ImageDraw.ImageDraw,
    *,
    contacts: list[PoseContact],
    receptor_pixel_map: dict[tuple[str, str, str, str], tuple[float, float]],
    ligand_pixels: list[tuple[float, float]],
    font: ImageFont.ImageFont,
    plot_right: int,
) -> None:
    for index, contact in enumerate(contacts[:6]):
        receptor_pixel = receptor_pixel_map.get(_contact_atom_key(contact))
        if receptor_pixel is None or contact.ligand_atom_index >= len(ligand_pixels):
            continue
        ligand_pixel = ligand_pixels[contact.ligand_atom_index]
        color = _contact_color(contact.interaction_type)
        _draw_dashed_line(draw, ligand_pixel, receptor_pixel, fill=(*color, 210), width=3)
        rx, ry = receptor_pixel
        lx, ly = ligand_pixel
        draw.ellipse((rx - 7, ry - 7, rx + 7, ry + 7), outline=(*color, 245), width=2)
        draw.ellipse((lx - 6, ly - 6, lx + 6, ly + 6), outline=(*color, 230), width=2)
        if index < 5:
            label = f"{contact.residue_label}  {contact.distance_angstrom:.1f} A"
            label_x = rx + 10 if rx < plot_right - 130 else rx - 130
            label_y = ry - 12
            _draw_label(draw, (label_x, label_y), label, fill=color, font=font)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: tuple[int, int, int, int],
    width: int,
    dash: int = 8,
    gap: int = 6,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = max(math.hypot(x2 - x1, y2 - y1), 0.001)
    step = dash + gap
    distance = 0.0
    while distance < length:
        segment_end = min(distance + dash, length)
        start_ratio = distance / length
        end_ratio = segment_end / length
        sx = x1 + (x2 - x1) * start_ratio
        sy = y1 + (y2 - y1) * start_ratio
        ex = x1 + (x2 - x1) * end_ratio
        ey = y1 + (y2 - y1) * end_ratio
        draw.line((sx, sy, ex, ey), fill=fill, width=width)
        distance += step


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    fill: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 4
    draw.rounded_rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        radius=5,
        fill=(18, 18, 18, 215),
        outline=(*fill, 180),
    )
    draw.text((x, y), text, fill=(245, 245, 245, 255), font=font)


def _draw_contact_panel(
    draw: ImageDraw.ImageDraw,
    *,
    contacts: list[PoseContact],
    nearby_residues: list[NearbyResidue],
    ligand_atom_count: int,
    receptor_atom_count: int,
    score: float | None,
    x0: int,
    width: int,
    height: int,
    font: ImageFont.ImageFont,
) -> None:
    heading_font = _load_font(15, bold=True)
    metric_font = _load_font(18, bold=True)
    small_font = _load_font(12)
    table_font = _load_font(12)
    draw.rectangle((x0, 72, x0 + width, height), fill=(22, 22, 22, 235))
    draw.line((x0, 72, x0, height), fill=(255, 255, 255, 24), width=1)
    x = x0 + 24
    y = 100
    draw.text((x, y), "POSE SUMMARY", fill=(245, 245, 245, 255), font=heading_font)
    y += 34

    metrics = [
        ("Docking score", f"{score:.3f}" if score is not None else "n/a", (150, 235, 166)),
        ("Contacts", str(len(contacts)), (84, 205, 255)),
        ("Residues", str(len(nearby_residues)), (245, 185, 77)),
    ]
    card_w = (width - 74) // 3
    for index, (label, value, color) in enumerate(metrics):
        card_x = x + index * (card_w + 12)
        _draw_metric_card(draw, card_x, y, card_w, label, value, color, small_font=small_font, metric_font=metric_font)
    y += 78

    draw.text(
        (x, y),
        f"Ligand heavy atoms: {ligand_atom_count}    nearby protein atoms: {receptor_atom_count}",
        fill=(170, 170, 170, 255),
        font=small_font,
    )
    y += 26

    draw.text((x, y), "Color key", fill=(170, 170, 170, 255), font=small_font)
    y += 22
    key_x = x
    for label, color in [
        ("Ligand", (95, 210, 125)),
        ("Polar", CONTACT_COLORS["polar_contact"]),
        ("Hydrophobic", CONTACT_COLORS["hydrophobic"]),
        ("Aromatic", CONTACT_COLORS["aromatic_contact"]),
        ("Protein atoms", (135, 135, 135)),
    ]:
        chip_w = _draw_chip(draw, key_x, y, label, color, font=small_font)
        key_x += chip_w + 8
        if key_x > x0 + width - 115:
            key_x = x
            y += 28
    y += 34

    if contacts:
        draw.text((x, y), "Interaction mix", fill=(170, 170, 170, 255), font=small_font)
        y += 22
        chip_x = x
        for interaction_type, count in _interaction_type_counts(contacts).items():
            chip_w = _draw_chip(draw, chip_x, y, f"{_contact_type_label(interaction_type)} {count}", _contact_color(interaction_type), font=small_font)
            chip_x += chip_w + 8
            if chip_x > x0 + width - 130:
                chip_x = x
                y += 28
        y += 34

    if nearby_residues:
        draw.text((x, y), "Closest binding-site residues", fill=(245, 245, 245, 255), font=heading_font)
        y += 28
        cell_w = (width - 76) // 2
        shown = nearby_residues[:6]
        for idx, residue in enumerate(shown):
            row_x = x + (idx % 2) * (cell_w + 14)
            row_y = y + (idx // 2) * 30
            text = f"{residue.residue_label} {residue.closest_atom}  {residue.distance_angstrom:.2f} A"
            draw.rounded_rectangle((row_x, row_y, row_x + cell_w, row_y + 23), radius=8, fill=(255, 255, 255, 10))
            draw.text((row_x + 8, row_y + 5), _truncate_text(text, 25), fill=(220, 220, 220, 255), font=small_font)
        y += 30 * ((len(shown) + 1) // 2) + 24

    draw.text((x, y), "Top contacts", fill=(245, 245, 245, 255), font=heading_font)
    y += 28
    if not contacts:
        draw.text((x, y), "No classified contacts detected.", fill=(170, 170, 170, 255), font=font)
        return

    headers = [("Type", 0), ("Lig", 118), ("Residue", 174), ("A", width - 68)]
    draw.rounded_rectangle((x, y, x0 + width - 22, y + 28), radius=8, fill=(12, 12, 12, 170))
    for label, offset in headers:
        draw.text((x + offset, y + 8), label, fill=(150, 150, 150, 255), font=small_font)
    y += 32
    for idx, contact in enumerate(contacts[:8]):
        color = _contact_color(contact.interaction_type)
        row_h = 38
        row_y = y
        if idx % 2 == 0:
            draw.rounded_rectangle((x, row_y, x0 + width - 22, row_y + row_h), radius=7, fill=(255, 255, 255, 10))
        draw.rounded_rectangle((x, row_y + 10, x + 9, row_y + 19), radius=4, fill=(*color, 235))
        draw.text((x + 16, row_y + 8), _contact_type_label(contact.interaction_type), fill=(230, 230, 230, 255), font=table_font)
        draw.text((x + 118, row_y + 8), f"{contact.ligand_element}{contact.ligand_atom_index}", fill=(220, 220, 220, 255), font=table_font)
        residue_atom = f"{contact.residue_label} {contact.receptor_atom}"
        draw.text((x + 174, row_y + 8), _truncate_text(residue_atom, 18), fill=(220, 220, 220, 255), font=table_font)
        distance = f"{contact.distance_angstrom:.2f}"
        dist_bbox = draw.textbbox((0, 0), distance, font=table_font)
        draw.text((x0 + width - 30 - (dist_bbox[2] - dist_bbox[0]), row_y + 8), distance, fill=(220, 220, 220, 255), font=table_font)
        y += row_h


def _wrap_text(text: str, *, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines or [text]


def _contact_type_label(interaction_type: str) -> str:
    labels = {
        "polar_contact": "Polar",
        "hydrophobic": "Hydrophobic",
        "aromatic_contact": "Aromatic",
        "salt_bridge_candidate": "Salt bridge",
    }
    return labels.get(interaction_type, interaction_type.replace("_", " ").capitalize())


def _interaction_type_counts(contacts: list[PoseContact]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for contact in contacts:
        counts[contact.interaction_type] = counts.get(contact.interaction_type, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], _contact_type_label(item[0]))))


def _draw_metric_card(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    label: str,
    value: str,
    color: tuple[int, int, int],
    *,
    small_font: ImageFont.ImageFont,
    metric_font: ImageFont.ImageFont,
) -> None:
    draw.rounded_rectangle((x, y, x + width, y + 58), radius=10, fill=(255, 255, 255, 10), outline=(255, 255, 255, 18))
    draw.text((x + 12, y + 10), label, fill=(160, 160, 160, 255), font=small_font)
    draw.text((x + 12, y + 30), value, fill=(*color, 255), font=metric_font)


def _draw_chip(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int],
    *,
    font: ImageFont.ImageFont,
) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0] + 28
    draw.rounded_rectangle((x, y, x + width, y + 22), radius=11, fill=(*color, 42), outline=(*color, 170))
    draw.ellipse((x + 8, y + 7, x + 14, y + 13), fill=(*color, 230))
    draw.text((x + 20, y + 4), text, fill=(235, 235, 235, 255), font=font)
    return width


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "."
