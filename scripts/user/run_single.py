from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stratadock.core.boxes import box_from_center_size, box_from_ligand_file, box_from_residues_pdb, load_box_json, validate_box, write_box_json
from stratadock.core.docking import run_docking
from stratadock.core.ligands import LigandPrepOptions, prepare_ligand_input
from stratadock.core.models import DockingBox
from stratadock.core.receptors import prepare_receptor_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one StrataDock v 1.6.01 docking job.")
    parser.add_argument("--receptor", required=True, type=Path, help="Input receptor PDB.")
    parser.add_argument("--ligand", required=True, type=Path, help="Input ligand SDF/SMI/SMILES/TXT.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--box-from-ligand", type=Path, help="Native/reference ligand PDB/SDF/MOL/MOL2 for box.")
    parser.add_argument("--box-from-ligand-pdb", type=Path, help="Backward-compatible alias for reference ligand PDB.")
    parser.add_argument("--box-json", type=Path, help="JSON object with center_x/y/z and size_x/y/z.")
    parser.add_argument("--box-center", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Manual box center.")
    parser.add_argument("--box-size", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Manual box size.")
    parser.add_argument("--box-residues", nargs="+", help="Residue selectors for residue-based box: 45, A:45, ASP:A:45.")
    parser.add_argument("--exhaustiveness", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-modes", type=int, default=1)
    parser.add_argument("--energy-range", type=float, default=3.0)
    parser.add_argument("--scoring", choices=["vina", "vinardo"], default="vina")
    parser.add_argument("--engine", choices=["vina", "gnina"], default="vina", help="Docking engine. GNINA requires a user-installed gnina executable.")
    parser.add_argument("--gnina-cpu-only", action="store_true", help="Run GNINA with --no_gpu.")
    parser.add_argument("--gnina-device", help="GNINA CUDA device index, for example 0. Ignored with --gnina-cpu-only.")
    parser.add_argument("--add-hydrogens-ph", type=float, help="Add receptor hydrogens at pH using OpenBabel.")
    parser.add_argument("--repair-pdbfixer", action="store_true", help="Repair receptor with PDBFixer.")
    parser.add_argument("--minimize-openmm", action="store_true", help="Energy-minimize receptor with OpenMM.")
    parser.add_argument("--ligand-force-field", choices=["MMFF94", "MMFF94s", "UFF"], default="MMFF94", help="Ligand minimization force field.")
    parser.add_argument("--ligand-ph", type=float, help="Protonate ligand at pH using OpenBabel when obabel is available.")
    parser.add_argument("--ligand-strip-salts", action="store_true", help="Keep the largest ligand fragment before 3D preparation.")
    parser.add_argument("--ligand-neutralize", action="store_true", help="Neutralize simple formal charges before 3D preparation.")
    return parser.parse_args()


def load_box(args: argparse.Namespace) -> DockingBox:
    manual_requested = args.box_center is not None or args.box_size is not None
    reference_ligand = args.box_from_ligand or args.box_from_ligand_pdb
    modes = [bool(reference_ligand), bool(args.box_json), manual_requested, bool(args.box_residues)]
    if sum(modes) != 1:
        raise SystemExit("Provide exactly one box source: --box-from-ligand, --box-json, --box-center with --box-size, or --box-residues.")
    if reference_ligand:
        return validate_box(box_from_ligand_file(reference_ligand))
    if args.box_json:
        return load_box_json(args.box_json)
    if args.box_residues:
        return box_from_residues_pdb(args.receptor, args.box_residues)
    if args.box_center is None or args.box_size is None:
        raise SystemExit("Manual box mode requires both --box-center and --box-size.")
    return box_from_center_size(args.box_center, args.box_size)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prep_dir = args.out_dir / "prepared"
    box = load_box(args)
    box_json = write_box_json(box, args.out_dir / "box.json")
    ligand_prep_options = LigandPrepOptions(
        force_field=args.ligand_force_field,
        protonation_ph=args.ligand_ph,
        strip_salts=args.ligand_strip_salts,
        neutralize=args.ligand_neutralize,
    )

    ligand = prepare_ligand_input(args.ligand, prep_dir, options=ligand_prep_options)
    receptor = prepare_receptor_input(
        args.receptor,
        prep_dir,
        add_hydrogens_ph=args.add_hydrogens_ph,
        repair_with_pdbfixer_enabled=args.repair_pdbfixer,
        minimize_with_openmm_enabled=args.minimize_openmm,
    )
    result = run_docking(
        engine=args.engine,
        project_root=ROOT,
        receptor_pdbqt=receptor.pdbqt,
        ligand_pdbqt=ligand.pdbqt,
        box=box,
        output_pdbqt=args.out_dir / "pose.pdbqt",
        exhaustiveness=args.exhaustiveness,
        seed=args.seed,
        num_modes=args.num_modes,
        energy_range=args.energy_range,
        scoring=args.scoring,
        gnina_cpu_only=args.gnina_cpu_only,
        gnina_device=args.gnina_device,
    )

    summary = {
        "ligand": {
            "name": ligand.name,
            "heavy_atoms": ligand.heavy_atoms,
            "embedding_used": ligand.embedding_used,
            "force_field": ligand.force_field,
            "requested_force_field": ligand.requested_force_field,
            "protonation_ph": ligand.protonation_ph,
            "protonation_status": ligand.protonation_status,
            "salt_stripping_status": ligand.salt_stripping_status,
            "neutralization_status": ligand.neutralization_status,
            "options": ligand.options.as_dict(),
            "prepared_sdf": str(ligand.prepared_sdf),
            "pdbqt": str(ligand.pdbqt),
        },
        "receptor": {
            "cleaned_pdb": str(receptor.clean_report.cleaned_pdb),
            "pdbqt": str(receptor.pdbqt),
            "report_json": str(receptor.report_json),
            "report_txt": str(receptor.report_txt),
            "source_atom_records": receptor.clean_report.source_atom_records,
            "source_hetatm_records": receptor.clean_report.source_hetatm_records,
            "atoms_kept": receptor.clean_report.atoms_kept,
            "waters_removed": receptor.clean_report.waters_removed,
            "hetero_removed": receptor.clean_report.hetero_removed,
            "metals_kept": receptor.clean_report.metals_kept,
            "altlocs_normalized": receptor.clean_report.altlocs_normalized,
            "altlocs_dropped": receptor.clean_report.altlocs_dropped,
            "chains": list(receptor.clean_report.chains),
            "protein_residue_count": receptor.clean_report.protein_residue_count,
            "warnings": list(receptor.clean_report.warnings),
            "prepared_pdb": str(receptor.prepared_pdb),
            "prep_steps": list(receptor.prep_steps),
            "options": receptor.options.as_dict(),
        },
        "box": box.as_dict(),
        "box_json": str(box_json),
        "docking": {
            "engine": result.engine,
            "score": result.score,
            "scores": result.scores,
            "cnn_score": result.cnn_scores[0] if result.cnn_scores else None,
            "cnn_affinity": result.cnn_affinities[0] if result.cnn_affinities else None,
            "cnn_scores": result.cnn_scores,
            "cnn_affinities": result.cnn_affinities,
            "pose_pdbqt": str(result.pose_pdbqt),
            "num_modes": args.num_modes,
            "energy_range": args.energy_range,
            "scoring": args.scoring,
            "gnina_cpu_only": args.gnina_cpu_only,
            "gnina_device": args.gnina_device,
        },
    }
    (args.out_dir / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from None
