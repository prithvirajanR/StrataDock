from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from rdkit.Chem import Draw

from stratadock.core.ligands import load_ligand_records_with_errors


@dataclass(frozen=True)
class LigandPreview:
    png: bytes
    labels: list[str]
    total_mols: int
    shown_mols: int
    omitted_count: int
    errors: list[str]


def ligand_preview(ligand_file: Path, *, max_mols: int = 12) -> LigandPreview:
    if max_mols <= 0:
        raise ValueError("max_mols must be greater than zero")

    records, errors = load_ligand_records_with_errors(ligand_file)
    if not records:
        raise ValueError(f"No valid ligands found for preview: {ligand_file}")
    shown = records[:max_mols]
    labels = [f"{record.source_index}: {record.name}" for record in shown]
    image = Draw.MolsToGridImage(
        [record.mol for record in shown],
        molsPerRow=4,
        subImgSize=(220, 180),
        legends=labels,
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return LigandPreview(
        png=buffer.getvalue(),
        labels=labels,
        total_mols=len(records),
        shown_mols=len(shown),
        omitted_count=max(len(records) - len(shown), 0),
        errors=[error.message for error in errors],
    )


def ligand_preview_png(ligand_file: Path, *, max_mols: int = 12) -> bytes:
    return ligand_preview(ligand_file, max_mols=max_mols).png
