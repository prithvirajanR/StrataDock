from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stratadock.core.pdb import compute_box_from_pdb_ligand, parse_pdb_atoms


def main() -> None:
    index_path = ROOT / "data" / "validation" / "index.json"
    if not index_path.exists():
        raise SystemExit("Validation index missing. Run scripts/install/download_validation_data.py first.")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    for case in index["cases"]:
        files = case["files"]
        receptor_path = ROOT / files["receptor_pdb"]
        ligand_path = ROOT / files["native_ligand_pdb"]
        ligand_sdf_path = ROOT / files["ligand_input_sdf"]
        receptor_atoms = parse_pdb_atoms(receptor_path.read_text(encoding="utf-8"))
        ligand_atoms = parse_pdb_atoms(ligand_path.read_text(encoding="utf-8"))
        box = compute_box_from_pdb_ligand(ligand_path.read_text(encoding="utf-8"))

        assert receptor_atoms, f"{case['case_id']}: receptor has no atoms"
        assert ligand_atoms, f"{case['case_id']}: ligand has no atoms"
        assert ligand_sdf_path.exists(), f"{case['case_id']}: ligand SDF missing"
        assert box.size_x > 0 and box.size_y > 0 and box.size_z > 0
        print(
            f"OK {case['case_id']}: receptor_atoms={len(receptor_atoms)} "
            f"ligand_atoms={len(ligand_atoms)} box={box.as_dict()}"
        )


if __name__ == "__main__":
    main()
