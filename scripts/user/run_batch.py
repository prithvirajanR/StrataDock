from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stratadock.core.batch import run_batch_screen, run_ensemble_screen
from stratadock.core.boxes import box_from_center_size, box_from_ligand_file, box_from_residues_pdb, load_box_json, validate_box
from stratadock.core.ligands import LigandPrepOptions
from stratadock.core.models import NamedDockingBox
from stratadock.core.pockets import suggest_pockets_with_fpocket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a StrataDock v 1.6.01 ligand batch screen.")
    parser.add_argument("--receptor", required=True, type=Path, nargs="+", help="Input receptor PDB(s).")
    parser.add_argument("--ligands", required=True, type=Path, help="Multi-ligand SDF or SMILES file.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--box-from-ligand", type=Path, help="Native/reference ligand PDB/SDF/MOL/MOL2 for box.")
    parser.add_argument("--box-from-ligand-pdb", type=Path, help="Backward-compatible alias for reference ligand PDB.")
    parser.add_argument("--box-json", type=Path, help="JSON object with center_x/y/z and size_x/y/z.")
    parser.add_argument("--box-center", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Manual box center.")
    parser.add_argument("--box-size", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Manual box size.")
    parser.add_argument("--box-residues", nargs="+", help="Residue selectors for residue-based box: 45, A:45, ASP:A:45.")
    parser.add_argument("--suggest-pockets", action="store_true", help="Use fpocket to suggest docking pockets.")
    parser.add_argument("--top-pockets", type=int, default=5, help="Number of fpocket pockets to dock.")
    parser.add_argument("--exhaustiveness", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-modes", type=int, default=1, help="Number of poses to write.")
    parser.add_argument("--energy-range", type=float, default=3.0, help="Vina energy range.")
    parser.add_argument("--scoring", choices=["vina", "vinardo"], default="vina", help="Vina scoring function.")
    parser.add_argument("--engine", choices=["vina", "gnina"], default="vina", help="Docking engine. GNINA is installed by the Linux setup script.")
    parser.add_argument("--gnina-cpu-only", action="store_true", help="Run GNINA with --no_gpu.")
    parser.add_argument("--gnina-device", help="GNINA CUDA device index, for example 0. Ignored with --gnina-cpu-only.")
    parser.add_argument("--keep-waters", action="store_true", help="Keep crystallographic waters during receptor cleaning.")
    parser.add_argument("--keep-heteroatoms", action="store_true", help="Keep non-protein heteroatoms/cofactors during receptor cleaning.")
    parser.add_argument("--remove-metals", action="store_true", help="Remove metals/simple ions instead of keeping them.")
    parser.add_argument("--default-altloc", default="A", help="Alternate location ID to keep when PDB altlocs exist.")
    parser.add_argument("--add-hydrogens-ph", type=float, help="Add receptor hydrogens at pH using OpenBabel.")
    parser.add_argument("--repair-pdbfixer", action="store_true", help="Repair receptor with PDBFixer.")
    parser.add_argument("--minimize-openmm", action="store_true", help="Energy-minimize receptor with OpenMM.")
    parser.add_argument("--ligand-force-field", choices=["MMFF94", "MMFF94s", "UFF"], default="MMFF94", help="Ligand minimization force field.")
    parser.add_argument("--ligand-ph", type=float, help="Protonate ligands at pH using OpenBabel when obabel is available.")
    parser.add_argument("--ligand-strip-salts", action="store_true", help="Keep the largest ligand fragment before 3D preparation.")
    parser.add_argument("--ligand-neutralize", action="store_true", help="Neutralize simple formal charges before 3D preparation.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing pose files when rerunning the same output directory.")
    parser.add_argument("--stop-file", type=Path, help="If this file exists, stop cleanly before the next ligand/receptor.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of independent ligand/pocket docking jobs to run in parallel.")
    return parser.parse_args()


def load_boxes(args: argparse.Namespace) -> list[NamedDockingBox]:
    manual_requested = args.box_center is not None or args.box_size is not None
    residue_requested = bool(args.box_residues)
    reference_ligand = args.box_from_ligand or args.box_from_ligand_pdb
    modes = [bool(reference_ligand), bool(args.box_json), manual_requested, residue_requested, args.suggest_pockets]
    if sum(modes) != 1:
        raise SystemExit(
            "Provide exactly one box source: --box-from-ligand, --box-json, "
            "--box-center with --box-size, --box-residues, or --suggest-pockets."
        )
    if reference_ligand:
        return [NamedDockingBox(name="reference_ligand", box=validate_box(box_from_ligand_file(reference_ligand)), source="reference_ligand")]
    if args.box_json:
        return [NamedDockingBox(name=args.box_json.stem, box=load_box_json(args.box_json), source="box_json")]
    if args.box_residues:
        return [NamedDockingBox(name="residue_box", box=box_from_residues_pdb(args.receptor[0], args.box_residues), source="residues")]
    if args.box_center is None or args.box_size is None:
        if args.suggest_pockets:
            return suggest_pockets_with_fpocket(args.receptor[0], args.out_dir / "fpocket", top_n=args.top_pockets)
        raise SystemExit("Manual box mode requires both --box-center and --box-size.")
    return [NamedDockingBox(name="manual", box=box_from_center_size(args.box_center, args.box_size), source="manual")]


def load_boxes_by_receptor(args: argparse.Namespace) -> list[list[NamedDockingBox]] | None:
    if len(args.receptor) <= 1:
        return None
    boxes_by_receptor: list[list[NamedDockingBox]] = []
    for index, receptor in enumerate(args.receptor, start=1):
        if args.suggest_pockets:
            pocket_dir = args.out_dir / f"receptor_{index}_{receptor.stem}" / "fpocket"
            receptor_boxes = suggest_pockets_with_fpocket(receptor, pocket_dir, top_n=args.top_pockets)
        elif args.box_residues:
            receptor_boxes = [NamedDockingBox(name="residue_box", box=box_from_residues_pdb(receptor, args.box_residues), source="residues")]
        else:
            return None
        if not receptor_boxes:
            raise SystemExit(f"No docking boxes were produced for receptor: {receptor}")
        boxes_by_receptor.append(receptor_boxes)
    return boxes_by_receptor


def main() -> None:
    args = parse_args()
    boxes_by_receptor = load_boxes_by_receptor(args)
    if boxes_by_receptor is None:
        boxes = load_boxes(args)
    else:
        boxes = [
            NamedDockingBox(
                name=f"{receptor.stem}:{box.name}",
                box=box.box,
                source=box.source,
                rank=box.rank,
                score=box.score,
                druggability_score=box.druggability_score,
            )
            for receptor, receptor_boxes in zip(args.receptor, boxes_by_receptor, strict=False)
            for box in receptor_boxes
        ]
    ligand_prep_options = LigandPrepOptions(
        force_field=args.ligand_force_field,
        protonation_ph=args.ligand_ph,
        strip_salts=args.ligand_strip_salts,
        neutralize=args.ligand_neutralize,
    )
    if len(args.receptor) == 1:
        batch = run_batch_screen(
            project_root=ROOT,
            receptor_pdb=args.receptor[0],
            ligand_file=args.ligands,
            box=boxes[0].box,
            boxes=boxes,
            output_dir=args.out_dir,
            exhaustiveness=args.exhaustiveness,
            seed=args.seed,
            num_modes=args.num_modes,
            energy_range=args.energy_range,
            scoring=args.scoring,
            engine=args.engine,
            gnina_cpu_only=args.gnina_cpu_only,
            gnina_device=args.gnina_device,
            remove_waters=not args.keep_waters,
            remove_non_protein_heteroatoms=not args.keep_heteroatoms,
            keep_metals=not args.remove_metals,
            default_altloc=args.default_altloc,
            add_hydrogens_ph=args.add_hydrogens_ph,
            repair_with_pdbfixer=args.repair_pdbfixer,
            minimize_with_openmm=args.minimize_openmm,
            resume=args.resume,
            stop_file=args.stop_file,
            n_jobs=args.n_jobs,
            ligand_prep_options=ligand_prep_options,
        )
        receptor_reports = [str(batch.receptor.report_json)]
        box_json = str(batch.box_json)
    else:
        batch = run_ensemble_screen(
            project_root=ROOT,
            receptor_pdbs=args.receptor,
            ligand_file=args.ligands,
            box=boxes[0].box,
            boxes=boxes,
            boxes_by_receptor=boxes_by_receptor,
            output_dir=args.out_dir,
            exhaustiveness=args.exhaustiveness,
            seed=args.seed,
            num_modes=args.num_modes,
            energy_range=args.energy_range,
            scoring=args.scoring,
            engine=args.engine,
            gnina_cpu_only=args.gnina_cpu_only,
            gnina_device=args.gnina_device,
            remove_waters=not args.keep_waters,
            remove_non_protein_heteroatoms=not args.keep_heteroatoms,
            keep_metals=not args.remove_metals,
            default_altloc=args.default_altloc,
            add_hydrogens_ph=args.add_hydrogens_ph,
            repair_with_pdbfixer=args.repair_pdbfixer,
            minimize_with_openmm=args.minimize_openmm,
            resume=args.resume,
            stop_file=args.stop_file,
            n_jobs=args.n_jobs,
            ligand_prep_options=ligand_prep_options,
        )
        receptor_reports = [str(run.receptor.report_json) for run in batch.runs]
        box_json = None
    print(
        json.dumps(
            {
                "results_csv": str(batch.results_csv),
                "results_json": str(batch.results_json),
                "summary_txt": str(batch.summary_txt),
                "best_by_ligand_csv": str(batch.best_by_ligand_csv),
                "best_by_pocket_csv": str(batch.best_by_pocket_csv),
                "html_report": str(batch.html_report),
                "pdf_report": str(batch.pdf_report),
                "events_log": str(batch.events_log),
                "box_json": box_json,
                "boxes_json": str(batch.boxes_json),
                "manifest_json": str(batch.manifest_json),
                "engine": args.engine,
                "gnina_cpu_only": args.gnina_cpu_only,
                "gnina_device": args.gnina_device,
                "receptor_report_json": receptor_reports[0] if receptor_reports else None,
                "receptor_report_jsons": receptor_reports,
                "total": len(batch.results),
                "success": sum(1 for result in batch.results if result.docking_status == "success"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from None
