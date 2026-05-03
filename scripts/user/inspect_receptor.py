from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stratadock.core.receptors import prepare_receptor_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and inspect a receptor without docking.")
    parser.add_argument("--receptor", required=True, type=Path, help="Input receptor/complex PDB.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--keep-waters", action="store_true", help="Keep crystallographic water HETATM records.")
    parser.add_argument("--keep-heteroatoms", action="store_true", help="Keep non-protein, non-metal HETATM records.")
    parser.add_argument("--remove-metals", action="store_true", help="Remove metal/ion HETATM records.")
    parser.add_argument("--default-altloc", default="A", help="Alternate location to retain.")
    parser.add_argument("--add-hydrogens-ph", type=float, help="Add receptor hydrogens at pH using OpenBabel.")
    parser.add_argument("--repair-pdbfixer", action="store_true", help="Repair missing atoms/residues with PDBFixer.")
    parser.add_argument("--minimize-openmm", action="store_true", help="Energy-minimize receptor with OpenMM.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_receptor_input(
        args.receptor,
        args.out_dir,
        remove_waters=not args.keep_waters,
        remove_non_protein_heteroatoms=not args.keep_heteroatoms,
        keep_metals=not args.remove_metals,
        default_altloc=args.default_altloc,
        add_hydrogens_ph=args.add_hydrogens_ph,
        repair_with_pdbfixer_enabled=args.repair_pdbfixer,
        minimize_with_openmm_enabled=args.minimize_openmm,
    )
    clean = report.clean_report
    print(
        json.dumps(
            {
                "cleaned_pdb": str(clean.cleaned_pdb),
                "pdbqt": str(report.pdbqt),
                "report_json": str(report.report_json),
                "report_txt": str(report.report_txt),
                "prepared_pdb": str(report.prepared_pdb),
                "prep_steps": list(report.prep_steps),
                "options": report.options.as_dict(),
                "source_atom_records": clean.source_atom_records,
                "source_hetatm_records": clean.source_hetatm_records,
                "atoms_kept": clean.atoms_kept,
                "waters_removed": clean.waters_removed,
                "waters_kept": clean.waters_kept,
                "hetero_removed": clean.hetero_removed,
                "hetero_kept": clean.hetero_kept,
                "metals_kept": clean.metals_kept,
                "metals_removed": clean.metals_removed,
                "chains": list(clean.chains),
                "protein_residue_count": clean.protein_residue_count,
                "warnings": list(clean.warnings),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from None
