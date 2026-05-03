from __future__ import annotations

import io
import zipfile
from collections.abc import Iterable
from pathlib import Path


def normalize_uploaded_receptors(uploads: Iterable[dict[str, bytes]]) -> list[dict[str, bytes]]:
    receptors: list[dict[str, bytes]] = []
    for upload in uploads:
        name = str(upload.get("name") or "").strip()
        data = upload.get("bytes")
        if not name or not isinstance(data, bytes) or not data:
            continue
        receptors.append({"name": name, "bytes": data})
    return receptors


def normalize_uploaded_ligands(uploads: Iterable[dict[str, bytes]]) -> dict[str, bytes] | None:
    ligands: list[dict[str, bytes]] = []
    for upload in uploads:
        name = str(upload.get("name") or "").strip()
        data = upload.get("bytes")
        if not name or not isinstance(data, bytes) or not data:
            continue
        ligands.append({"name": Path(name).name, "bytes": data})
    if not ligands:
        return None
    if len(ligands) == 1:
        return ligands[0]

    buffer = io.BytesIO()
    used_names: set[str] = set()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, ligand in enumerate(ligands, start=1):
            name = str(ligand["name"])
            archive_name = name
            if archive_name in used_names:
                path = Path(name)
                archive_name = f"{path.stem}_{index}{path.suffix}"
            used_names.add(archive_name)
            archive.writestr(archive_name, ligand["bytes"])
    return {"name": "stratadock_ligands.zip", "bytes": buffer.getvalue()}


def selected_pocket_names(rows: list[dict[str, object]], selected_ranks: list[int]) -> set[str]:
    if not selected_ranks:
        return {str(row["name"]) for row in rows if row.get("name")}
    ranks = {int(rank) for rank in selected_ranks}
    return {
        str(row["name"])
        for row in rows
        if row.get("name") and row.get("rank") is not None and int(row["rank"]) in ranks
    }
