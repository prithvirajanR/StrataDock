from __future__ import annotations

import io
import csv
import tempfile
import zipfile
from pathlib import Path

from stratadock.core.session import create_selected_session_archive, create_session_manifest


def zip_run_outputs(output_dir: Path) -> bytes:
    create_session_manifest(output_dir)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir))
    return buffer.getvalue()


def zip_selected_run_outputs(output_dir: Path, selected_paths: list[str | Path]) -> bytes:
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = create_selected_session_archive(output_dir, selected_paths, Path(temp_dir) / "selected.zip")
        return archive_path.read_bytes()


HIT_ARTIFACT_COLUMNS = [
    "pose_pdbqt",
    "pose_path",
    "complex_pdb",
    "complex_path",
    "interactions_json",
    "interactions_csv",
    "viewer_html",
    "pymol_script",
    "prepared_sdf",
    "ligand_pdbqt",
]


def _selected_archive_name(output_dir: Path | None, value: object) -> str:
    raw = str(value).replace("\\", "/")
    if output_dir is None:
        return raw
    path = Path(str(value))
    if not path.is_absolute():
        return raw
    root = output_dir.resolve()
    resolved = path.resolve()
    if resolved == root or root in resolved.parents:
        return str(resolved.relative_to(root)).replace("\\", "/")
    return raw


def selected_hit_paths(
    rows: list[dict[str, object]], *, include_summary: bool = True, output_dir: Path | None = None
) -> list[str]:
    paths: list[str] = []
    if include_summary:
        paths.extend(["run_manifest.json", "results.csv"])
    for row in rows:
        for column in HIT_ARTIFACT_COLUMNS:
            value = row.get(column)
            if value:
                paths.append(_selected_archive_name(output_dir, value))
    seen: set[str] = set()
    ordered: list[str] = []
    for path in paths:
        normalized = path.replace("\\", "/")
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def zip_selected_hit_outputs(output_dir: Path, rows: list[dict[str, object]], *, include_summary: bool = True) -> bytes:
    return zip_selected_run_outputs(
        output_dir,
        selected_hit_paths(rows, include_summary=include_summary, output_dir=output_dir),
    )


def csv_bytes(rows: list[dict[str, object]]) -> bytes:
    if not rows:
        return b""
    buffer = io.StringIO()
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")
