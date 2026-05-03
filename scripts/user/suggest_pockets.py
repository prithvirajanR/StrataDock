from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stratadock.core.pockets import suggest_pockets_with_fpocket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suggest docking boxes with fpocket.")
    parser.add_argument("--receptor", required=True, type=Path, help="Input receptor PDB.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--top", type=int, default=5, help="Number of pockets to return.")
    parser.add_argument("--padding", type=float, default=8.0, help="Padding added around pocket atoms.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pockets = suggest_pockets_with_fpocket(
        args.receptor,
        args.out_dir,
        padding=args.padding,
        top_n=args.top,
    )
    print(json.dumps([pocket.as_dict() for pocket in pockets], indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from None
