from __future__ import annotations

import re
import urllib.request
from pathlib import Path


def validate_pdb_id(pdb_id: str) -> str:
    normalized = pdb_id.strip().lower()
    if not re.fullmatch(r"[0-9][a-z0-9]{3}", normalized):
        raise ValueError(f"Invalid PDB ID: {pdb_id!r}")
    return normalized


def download_pdb(pdb_id: str, output_dir: Path) -> Path:
    normalized = validate_pdb_id(pdb_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{normalized}.pdb"
    url = f"https://files.rcsb.org/download/{normalized.upper()}.pdb"
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8", errors="replace")
    if "ATOM" not in text:
        raise ValueError(f"Downloaded file for {normalized.upper()} does not look like a PDB structure.")
    out_path.write_text(text, encoding="utf-8")
    return out_path
