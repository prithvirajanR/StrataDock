from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stratadock.core.pdb_download import download_pdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a PDB structure from RCSB.")
    parser.add_argument("pdb_id", help="Four-character PDB ID, e.g. 3PTB.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "pdb", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = download_pdb(args.pdb_id, args.out_dir)
    print(json.dumps({"pdb_id": args.pdb_id.upper(), "pdb": str(path)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from None
