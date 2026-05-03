from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdkit import RDLogger

from stratadock.core.docking import run_vina
from stratadock.core.evaluate import heavy_atom_rmsd
from stratadock.core.ligands import load_sdf, prepare_ligand_pdbqt
from stratadock.core.models import DockingBox
from stratadock.core.receptors import prepare_receptor_input


def main() -> None:
    RDLogger.DisableLog("rdApp.warning")
    case_id = sys.argv[1] if len(sys.argv) > 1 else "hiv_protease_1hsg"
    manifest_path = ROOT / "data" / "validation" / case_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["files"]
    work = ROOT / "runs" / case_id
    work.mkdir(parents=True, exist_ok=True)

    ligand_sdf = ROOT / files["ligand_input_sdf"]
    receptor_pdb = ROOT / files["receptor_pdb"]
    mols = load_sdf(ligand_sdf)
    if not mols:
        raise SystemExit(f"No molecules loaded from {ligand_sdf}")

    ligand_pdbqt = prepare_ligand_pdbqt(ligand_sdf, work / "ligand.pdbqt")
    receptor = prepare_receptor_input(receptor_pdb, work / "receptor")
    box = DockingBox(**manifest["box"])
    result = run_vina(
        project_root=ROOT,
        receptor_pdbqt=receptor.pdbqt,
        ligand_pdbqt=ligand_pdbqt,
        box=box,
        output_pdbqt=work / "pose.pdbqt",
        exhaustiveness=8,
        num_modes=9,
        seed=1,
    )
    rmsd = heavy_atom_rmsd(
        ligand_input_sdf=ligand_sdf,
        native_ligand_pdb=ROOT / files["native_ligand_pdb"],
        docked_pose_pdbqt=result.pose_pdbqt,
    )
    out = {
        "case_id": case_id,
        "score": result.score,
        "heavy_atom_rmsd": round(rmsd, 3),
        "rmsd_angstrom": round(rmsd, 3),
        "passes_rmsd_threshold": rmsd <= manifest["acceptance"]["pose_rmsd_angstrom_max"],
        "pose_pdbqt": str(result.pose_pdbqt.relative_to(ROOT)),
        "native_ligand_pdb": files["native_ligand_pdb"],
        "receptor_report_json": str(receptor.report_json.relative_to(ROOT)),
        "validation_docking_params": {
            "exhaustiveness": 8,
            "num_modes": 9,
            "seed": 1,
        },
    }
    (work / "result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
