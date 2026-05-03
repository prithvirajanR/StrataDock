from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from stratadock.core.boxes import validate_box
from stratadock.core.models import NamedDockingBox
from stratadock.core.pdb import compute_box_from_pdb_ligand


def fpocket_available() -> bool:
    return shutil.which("fpocket") is not None or _bundled_fpocket_binary() is not None


def parse_fpocket_output(output_dir: Path, *, padding: float = 8.0, top_n: int = 5) -> list[NamedDockingBox]:
    pockets_dir = output_dir / "pockets"
    if not pockets_dir.exists():
        raise ValueError(f"fpocket output is missing pockets directory: {pockets_dir}")

    scores = _parse_fpocket_scores(output_dir)
    boxes: list[NamedDockingBox] = []
    pocket_files = sorted(
        pockets_dir.glob("pocket*_atm.pdb"),
        key=lambda path: _pocket_rank(path.name),
    )
    for path in pocket_files:
        rank = _pocket_rank(path.name)
        box = validate_box(compute_box_from_pdb_ligand(path.read_text(encoding="utf-8", errors="ignore"), padding=padding))
        score_data = scores.get(rank, {})
        boxes.append(
            NamedDockingBox(
                name=f"pocket_{rank}",
                box=box,
                source="fpocket",
                rank=rank,
                score=score_data.get("score"),
                druggability_score=score_data.get("druggability_score"),
            )
        )
    if not boxes:
        raise ValueError(f"No fpocket pocket PDB files found in {pockets_dir}")
    boxes.sort(key=lambda item: item.rank or 999)
    return boxes[:top_n]


def suggest_pockets_with_fpocket(
    receptor_pdb: Path,
    output_dir: Path,
    *,
    padding: float = 8.0,
    top_n: int = 5,
) -> list[NamedDockingBox]:
    if not fpocket_available():
        raise RuntimeError(
            "fpocket was not found on PATH and no bundled WSL fpocket was available. "
            "Install fpocket to use automatic pocket detection."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = receptor_fingerprint(receptor_pdb)
    expected_dir = _cached_fpocket_output_dir(output_dir, receptor_fingerprint=fingerprint)
    if expected_dir is None:
        before = {path.resolve() for path in output_dir.glob("*_out")}
        run = _run_fpocket(receptor_pdb=receptor_pdb, output_dir=output_dir)
        if run.returncode != 0:
            raise RuntimeError(f"fpocket failed with code {run.returncode}.\nSTDOUT:\n{run.stdout}\nSTDERR:\n{run.stderr}")

        expected_dir = output_dir / f"{receptor_pdb.stem}_out"
        if not expected_dir.exists():
            candidates = sorted(path for path in output_dir.glob("*_out") if path.resolve() not in before)
            if not candidates:
                raise RuntimeError(f"fpocket completed but no *_out directory was produced in {output_dir}")
            if len(candidates) > 1:
                raise RuntimeError(f"fpocket produced multiple unexpected *_out directories in {output_dir}")
            expected_dir = candidates[0]
        _write_fpocket_cache_manifest(
            output_dir=output_dir,
            receptor_pdb=receptor_pdb,
            receptor_fingerprint=fingerprint,
            fpocket_output_dir=expected_dir,
        )
    pockets = parse_fpocket_output(expected_dir, padding=padding, top_n=top_n)
    write_pockets_json(pockets, output_dir / "pockets.json")
    write_pocket_metadata_table(fpocket_metadata_table(expected_dir, pockets), output_dir / "pockets_table.json")
    return pockets


def receptor_fingerprint(receptor_pdb: Path) -> str:
    digest = hashlib.sha256()
    with receptor_pdb.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fpocket_metadata_table(output_dir: Path, pockets: list[NamedDockingBox] | None = None) -> list[dict[str, object]]:
    metadata = _parse_fpocket_metadata(output_dir)
    if pockets is None:
        pockets = parse_fpocket_output(output_dir, top_n=999)
    rows: list[dict[str, object]] = []
    for pocket in pockets:
        rank = pocket.rank or _pocket_rank(pocket.name)
        values = metadata.get(rank, {})
        rows.append(
            {
                "rank": rank,
                "name": pocket.name,
                "source": pocket.source,
                "score": values.get("score", pocket.score),
                "volume": values.get("volume"),
                "druggability_score": values.get("druggability_score", pocket.druggability_score),
                "center_x": pocket.box.center_x,
                "center_y": pocket.box.center_y,
                "center_z": pocket.box.center_z,
                "size_x": pocket.box.size_x,
                "size_y": pocket.box.size_y,
                "size_z": pocket.box.size_z,
            }
        )
    return rows


def write_pocket_metadata_table(rows: list[dict[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        fieldnames = list(rows[0]) if rows else [
            "rank",
            "name",
            "source",
            "score",
            "volume",
            "druggability_score",
            "center_x",
            "center_y",
            "center_z",
            "size_x",
            "size_y",
            "size_z",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return path


def _run_fpocket(*, receptor_pdb: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    receptor_pdb = receptor_pdb.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_receptor = output_dir / receptor_pdb.name
    if receptor_pdb.resolve() != output_receptor.resolve():
        shutil.copy2(receptor_pdb, output_receptor)
    native_fpocket = shutil.which("fpocket")
    if native_fpocket:
        return subprocess.run(
            [native_fpocket, "-f", str(output_receptor)],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )
    binary = _bundled_fpocket_binary()
    if binary:
        if not shutil.which("wsl"):
            return subprocess.run(
                [str(binary), "-f", str(output_receptor)],
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )
        return subprocess.run(
            [
                "wsl",
                "--cd",
                _wsl_path(output_dir),
                _wsl_path(binary),
                "-f",
                _wsl_path(output_receptor),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
    raise RuntimeError(
        "fpocket was not found on PATH and bundled WSL fpocket could not be launched. "
        "Install fpocket or run inside Ubuntu/WSL."
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _bundled_fpocket_binary() -> Path | None:
    binary = _project_root() / "tools" / "fpocket_src" / "bin" / "fpocket"
    return binary if binary.is_file() and os.access(binary, os.X_OK) else None


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    if resolved.drive:
        drive = resolved.drive.rstrip(":").lower()
        parts = [part for part in resolved.parts[1:] if part not in {"\\", "/"}]
        return "/mnt/" + drive + "/" + "/".join(parts)
    run = subprocess.run(
        ["wsl", "wslpath", "-a", str(resolved)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if run.returncode != 0:
        raise RuntimeError(f"Could not convert path to WSL path: {path}\n{run.stderr}")
    return run.stdout.strip()


def write_pockets_json(pockets: list[NamedDockingBox], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([pocket.as_dict() for pocket in pockets], indent=2) + "\n", encoding="utf-8")
    return path


def _fpocket_cache_manifest(output_dir: Path) -> Path:
    return output_dir / "fpocket_cache.json"


def _cached_fpocket_output_dir(output_dir: Path, *, receptor_fingerprint: str) -> Path | None:
    manifest_path = _fpocket_cache_manifest(output_dir)
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if manifest.get("receptor_fingerprint") != receptor_fingerprint:
        return None
    raw_output_dir = manifest.get("fpocket_output_dir")
    if not raw_output_dir:
        return None
    cached_dir = Path(str(raw_output_dir))
    if not cached_dir.is_absolute():
        cached_dir = output_dir / cached_dir
    pockets_dir = cached_dir / "pockets"
    if not pockets_dir.exists() or not any(pockets_dir.glob("pocket*_atm.pdb")):
        return None
    return cached_dir


def _write_fpocket_cache_manifest(
    *,
    output_dir: Path,
    receptor_pdb: Path,
    receptor_fingerprint: str,
    fpocket_output_dir: Path,
) -> Path:
    try:
        output_value = str(fpocket_output_dir.relative_to(output_dir))
    except ValueError:
        output_value = str(fpocket_output_dir)
    payload = {
        "receptor_pdb": str(receptor_pdb),
        "receptor_fingerprint": receptor_fingerprint,
        "fpocket_output_dir": output_value,
    }
    path = _fpocket_cache_manifest(output_dir)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _pocket_rank(name: str) -> int:
    match = re.search(r"pocket(\d+)", name)
    return int(match.group(1)) if match else 999


def _parse_fpocket_scores(output_dir: Path) -> dict[int, dict[str, float]]:
    return {
        rank: {key: value for key, value in values.items() if key in {"score", "druggability_score"}}
        for rank, values in _parse_fpocket_metadata(output_dir).items()
    }


def _parse_fpocket_metadata(output_dir: Path) -> dict[int, dict[str, float]]:
    info_files = sorted({*output_dir.glob("*_info.txt"), *output_dir.glob("*.txt")})
    if not info_files:
        return {}
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in info_files)
    scores: dict[int, dict[str, float]] = {}
    current: int | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = re.match(r"Pocket\s+(\d+)", line, re.IGNORECASE)
        if match:
            current = int(match.group(1))
            scores.setdefault(current, {})
            continue
        if current is None:
            continue
        value_match = re.search(r":\s*([-+]?\d+(?:\.\d+)?)", line)
        if not value_match:
            continue
        value = float(value_match.group(1))
        lower = line.lower()
        if "druggability" in lower:
            scores[current]["druggability_score"] = value
        elif "volume" in lower:
            scores[current]["volume"] = value
        elif lower.startswith("score"):
            scores[current]["score"] = value
    return scores
