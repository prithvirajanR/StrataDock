from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stratadock.core.boxes import box_from_ligand_file, box_from_residues_pdb, write_box_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reusable StrataDock box JSON.")
    parser.add_argument("--out", required=True, type=Path, help="Output box JSON.")
    parser.add_argument("--from-ligand", type=Path, help="Reference ligand PDB/SDF/MOL/MOL2.")
    parser.add_argument("--from-ligand-pdb", type=Path, help="Backward-compatible alias for reference ligand PDB.")
    parser.add_argument("--receptor", type=Path, help="Receptor PDB for residue-based box.")
    parser.add_argument("--residues", nargs="+", help="Residue selectors: 45, A:45, or ASP:A:45.")
    parser.add_argument("--padding", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_ligand = args.from_ligand or args.from_ligand_pdb
    if reference_ligand and (args.receptor or args.residues):
        raise ValueError("Use either --from-ligand or --receptor with --residues, not both.")
    if reference_ligand:
        box = box_from_ligand_file(reference_ligand, padding=args.padding)
    elif args.receptor and args.residues:
        box = box_from_residues_pdb(args.receptor, args.residues, padding=args.padding)
    else:
        raise ValueError("Provide --from-ligand or --receptor with --residues.")
    write_box_json(box, args.out)
    print(json.dumps({"box_json": str(args.out), "box": box.as_dict()}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from None
