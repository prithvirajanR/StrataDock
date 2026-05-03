from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stratadock.core.pdb import compute_box_from_pdb_ligand, split_receptor_and_ligand, write_text


CASES = [
    {
        "case_id": "hiv_protease_1hsg",
        "pdb_id": "1HSG",
        "description": "HIV-1 protease with MK1 inhibitor; useful compact redocking case.",
    },
    {
        "case_id": "egfr_1m17",
        "pdb_id": "1M17",
        "description": "EGFR kinase domain with AQ4 inhibitor; kinase-like validation case.",
    },
    {
        "case_id": "abl_kinase_1iep",
        "pdb_id": "1IEP",
        "description": "ABL kinase with imatinib-like inhibitor; tests larger kinase inhibitor handling.",
    },
    {
        "case_id": "trypsin_3ptb",
        "pdb_id": "3PTB",
        "description": "Trypsin with benzamidine-like ligand; tests small charged ligand handling.",
    },
    {
        "case_id": "hsv_tk_1kim",
        "pdb_id": "1KIM",
        "description": "HSV thymidine kinase with thymidine; tests nucleoside-like ligand handling.",
    },
]


def download_pdb(pdb_id: str) -> str:
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    with urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def download_ligand_sdf(pdb_id: str, ligand_key: tuple[str, str, str]) -> tuple[str, str]:
    # Exact ligand-instance coordinates from the deposited structure. This is better
    # for redocking validation than generic CCD ideal coordinates.
    url = (
        f"https://models.rcsb.org/v1/{pdb_id.upper()}/ligand"
        f"?auth_asym_id={ligand_key[1]}&auth_seq_id={ligand_key[2]}&encoding=sdf"
    )
    with urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8"), url


def build_case(case: dict[str, str]) -> dict[str, object]:
    case_dir = ROOT / "data" / "validation" / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)

    pdb_text = download_pdb(case["pdb_id"])
    receptor_text, ligand_text, ligand_key = split_receptor_and_ligand(pdb_text)
    ligand_sdf_text, ligand_sdf_source = download_ligand_sdf(case["pdb_id"], ligand_key)
    box = compute_box_from_pdb_ligand(ligand_text, padding=8.0)

    complex_path = case_dir / f"{case['pdb_id'].lower()}_complex.pdb"
    receptor_path = case_dir / "receptor.pdb"
    ligand_path = case_dir / "native_ligand.pdb"
    ligand_sdf_path = case_dir / "ligand_input.sdf"

    write_text(complex_path, pdb_text)
    write_text(receptor_path, receptor_text)
    write_text(ligand_path, ligand_text)
    write_text(ligand_sdf_path, ligand_sdf_text)

    manifest = {
        "case_id": case["case_id"],
        "pdb_id": case["pdb_id"],
        "description": case["description"],
        "source": f"https://files.rcsb.org/download/{case['pdb_id'].upper()}.pdb",
        "ligand_sdf_source": ligand_sdf_source,
        "validation_type": "co_crystal_redocking",
        "ligand": {
            "resname": ligand_key[0],
            "chain": ligand_key[1],
            "resseq": ligand_key[2],
        },
        "files": {
            "complex_pdb": str(complex_path.relative_to(ROOT)),
            "receptor_pdb": str(receptor_path.relative_to(ROOT)),
            "native_ligand_pdb": str(ligand_path.relative_to(ROOT)),
            "ligand_input_sdf": str(ligand_sdf_path.relative_to(ROOT)),
        },
        "box": box.as_dict(),
        "acceptance": {
            "pose_rmsd_angstrom_max": 2.5,
            "score_must_be_float": True,
            "pose_file_must_exist": True,
        },
    }
    (case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    all_manifests = [build_case(case) for case in CASES]
    index = {
        "source": "RCSB PDB",
        "purpose": "Small co-crystal redocking validation set for StrataDock v 1.6.01.",
        "cases": all_manifests,
    }
    index_path = ROOT / "data" / "validation" / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    for manifest in all_manifests:
        ligand = manifest["ligand"]
        print(
            f"{manifest['case_id']}: {manifest['pdb_id']} "
            f"ligand={ligand['resname']} chain={ligand['chain']} resseq={ligand['resseq']}"
        )
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
