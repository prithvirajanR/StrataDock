from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stratadock.core.batch import run_batch_screen
from stratadock.core.boxes import box_from_ligand_pdb
from stratadock.core.docking import run_vina
from stratadock.core.ligands import prepare_ligand_pdbqt
from stratadock.core.models import DockingBox
from stratadock.core.pockets import suggest_pockets_with_fpocket
from stratadock.core.receptors import prepare_receptor_input


OUT_DIR = ROOT / "runs" / "comprehensive_settings"
CASE_DIR = ROOT / "data" / "validation" / "trypsin_3ptb"
RDLogger.DisableLog("rdApp.warning")


def timed_case(name: str, fn, *, expected: str, rows: list[dict[str, object]]) -> object | None:
    started = time.perf_counter()
    try:
        actual = fn()
        rows.append(
            {
                "case": name,
                "status": "PASS",
                "expected": expected,
                "actual": _short_actual(actual),
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
        return actual
    except Exception as exc:
        rows.append(
            {
                "case": name,
                "status": "FAIL",
                "expected": expected,
                "actual": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
        return None


def expected_failure_case(name: str, fn, *, expected_error: str, rows: list[dict[str, object]]) -> None:
    started = time.perf_counter()
    try:
        fn()
    except Exception as exc:
        actual = f"{type(exc).__name__}: {exc}"
        status = "PASS" if expected_error.lower() in actual.lower() else "FAIL"
        rows.append(
            {
                "case": name,
                "status": status,
                "expected": f"Expected error containing {expected_error!r}",
                "actual": actual,
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
        return
    rows.append(
        {
            "case": name,
            "status": "FAIL",
            "expected": f"Expected error containing {expected_error!r}",
            "actual": "No error raised",
            "seconds": round(time.perf_counter() - started, 3),
        }
    )


def dependency_case(name: str, dependency_available: bool, fn, *, rows: list[dict[str, object]]) -> None:
    started = time.perf_counter()
    try:
        actual = fn()
        rows.append(
            {
                "case": name,
                "status": "PASS" if dependency_available else "FAIL",
                "expected": "Succeeds when dependency is installed; otherwise clear dependency error",
                "actual": _short_actual(actual),
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
    except Exception as exc:
        actual = f"{type(exc).__name__}: {exc}"
        is_dependency_error = any(token in actual.lower() for token in ["not installed", "not found", "cannot"])
        rows.append(
            {
                "case": name,
                "status": "PASS" if (not dependency_available and is_dependency_error) else "FAIL",
                "expected": "Succeeds when dependency is installed; otherwise clear dependency error",
                "actual": actual,
                "seconds": round(time.perf_counter() - started, 3),
            }
        )


def _short_actual(value: object) -> str:
    if value is None:
        return "ok"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if hasattr(value, "score"):
        return json.dumps(
            {
                "score": getattr(value, "score"),
                "scores": getattr(value, "scores"),
                "pose_pdbqt": str(getattr(value, "pose_pdbqt")),
            },
            sort_keys=True,
        )
    return str(value)


def write_smiles(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("NC(=N)c1ccccc1 benzamidine\n", encoding="utf-8")
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    manifest = json.loads((CASE_DIR / "manifest.json").read_text(encoding="utf-8"))
    receptor_pdb = CASE_DIR / "3ptb_complex.pdb"
    ligand_sdf = CASE_DIR / "ligand_input.sdf"
    native_ligand = CASE_DIR / "native_ligand.pdb"
    ligand_smi = write_smiles(OUT_DIR / "inputs" / "one_ligand.smi")
    box = DockingBox(**manifest["box"])

    receptor = timed_case(
        "prep_default_clean_only",
        lambda: prepare_receptor_input(receptor_pdb, OUT_DIR / "prepared" / "default_receptor"),
        expected="Clean receptor and prepare PDBQT",
        rows=rows,
    )
    ligand_pdbqt = timed_case(
        "ligand_sdf_prepare",
        lambda: prepare_ligand_pdbqt(ligand_sdf, OUT_DIR / "prepared" / "ligand.pdbqt"),
        expected="Prepare validation SDF ligand as PDBQT",
        rows=rows,
    )

    if receptor is None or ligand_pdbqt is None:
        write_reports(rows)
        raise SystemExit("Core receptor/ligand preparation failed; aborting matrix.")

    docking_matrix = [
        ("vina_default", {"scoring": "vina", "exhaustiveness": 4, "num_modes": 1, "energy_range": 3.0, "seed": 1}),
        ("scoring_vinardo", {"scoring": "vinardo", "exhaustiveness": 4, "num_modes": 1, "energy_range": 3.0, "seed": 1}),
        ("exhaustiveness_min_1", {"scoring": "vina", "exhaustiveness": 1, "num_modes": 1, "energy_range": 3.0, "seed": 1}),
        ("exhaustiveness_mid_8", {"scoring": "vina", "exhaustiveness": 8, "num_modes": 1, "energy_range": 3.0, "seed": 1}),
        ("exhaustiveness_max_32", {"scoring": "vina", "exhaustiveness": 32, "num_modes": 1, "energy_range": 3.0, "seed": 1}),
        ("num_modes_3", {"scoring": "vina", "exhaustiveness": 4, "num_modes": 3, "energy_range": 3.0, "seed": 1}),
        ("num_modes_max_9", {"scoring": "vina", "exhaustiveness": 4, "num_modes": 9, "energy_range": 3.0, "seed": 1}),
        ("energy_range_min_1", {"scoring": "vina", "exhaustiveness": 4, "num_modes": 3, "energy_range": 1.0, "seed": 1}),
        ("energy_range_max_10", {"scoring": "vina", "exhaustiveness": 4, "num_modes": 3, "energy_range": 10.0, "seed": 1}),
        ("seed_zero", {"scoring": "vina", "exhaustiveness": 4, "num_modes": 1, "energy_range": 3.0, "seed": 0}),
        ("seed_large", {"scoring": "vina", "exhaustiveness": 4, "num_modes": 1, "energy_range": 3.0, "seed": 999999}),
    ]
    repeat_results: list[object] = []
    docking_matrix.extend(
        [
            ("seed_repeat_a", {"scoring": "vina", "exhaustiveness": 4, "num_modes": 3, "energy_range": 3.0, "seed": 123}),
            ("seed_repeat_b", {"scoring": "vina", "exhaustiveness": 4, "num_modes": 3, "energy_range": 3.0, "seed": 123}),
        ]
    )

    for name, params in docking_matrix:
        result = timed_case(
            f"vina_parameter_{name}",
            lambda params=params, name=name: run_vina(
                project_root=ROOT,
                receptor_pdbqt=receptor.pdbqt,
                ligand_pdbqt=ligand_pdbqt,
                box=box,
                output_pdbqt=OUT_DIR / "poses" / f"{name}.pdbqt",
                **params,
            ),
            expected=f"Vina run succeeds with {params}",
            rows=rows,
        )
        if name.startswith("seed_repeat"):
            repeat_results.append(result)

    if len(repeat_results) == 2 and all(repeat_results):
        a, b = repeat_results
        rows.append(
            {
                "case": "seed_repeatability_same_seed",
                "status": "PASS" if a.score == b.score and a.scores == b.scores else "FAIL",
                "expected": "Same seed and same parameters produce identical parsed Vina score list",
                "actual": json.dumps({"a": a.scores, "b": b.scores}, sort_keys=True),
                "seconds": 0,
            }
        )

    expected_failure_case(
        "vina_invalid_num_modes_zero",
        lambda: run_vina(
            project_root=ROOT,
            receptor_pdbqt=receptor.pdbqt,
            ligand_pdbqt=ligand_pdbqt,
            box=box,
            output_pdbqt=OUT_DIR / "poses" / "invalid_modes.pdbqt",
            num_modes=0,
        ),
        expected_error="num_modes",
        rows=rows,
    )
    expected_failure_case(
        "vina_invalid_energy_range_zero",
        lambda: run_vina(
            project_root=ROOT,
            receptor_pdbqt=receptor.pdbqt,
            ligand_pdbqt=ligand_pdbqt,
            box=box,
            output_pdbqt=OUT_DIR / "poses" / "invalid_energy.pdbqt",
            energy_range=0,
        ),
        expected_error="energy_range",
        rows=rows,
    )
    expected_failure_case(
        "vina_invalid_scoring",
        lambda: run_vina(
            project_root=ROOT,
            receptor_pdbqt=receptor.pdbqt,
            ligand_pdbqt=ligand_pdbqt,
            box=box,
            output_pdbqt=OUT_DIR / "poses" / "invalid_scoring.pdbqt",
            scoring="bad",
        ),
        expected_error="scoring",
        rows=rows,
    )

    ref_box = timed_case(
        "box_reference_ligand_mode",
        lambda: run_batch_screen(
            project_root=ROOT,
            receptor_pdb=receptor_pdb,
            ligand_file=ligand_smi,
            box=box_from_ligand_pdb(native_ligand),
            output_dir=OUT_DIR / "batch_reference_box",
            exhaustiveness=1,
            seed=1,
        ),
        expected="Reference ligand box batch succeeds and writes artifacts",
        rows=rows,
    )
    if ref_box:
        assert_batch_outputs("box_reference_ligand_mode_outputs", ref_box, rows)

    manual_batch = timed_case(
        "box_manual_mode",
        lambda: run_batch_screen(
            project_root=ROOT,
            receptor_pdb=receptor_pdb,
            ligand_file=ligand_smi,
            box=box,
            output_dir=OUT_DIR / "batch_manual_box",
            exhaustiveness=1,
            seed=1,
        ),
        expected="Manual center/size box batch succeeds and writes artifacts",
        rows=rows,
    )
    if manual_batch:
        assert_batch_outputs("box_manual_mode_outputs", manual_batch, rows)

    fpocket_boxes = timed_case(
        "box_fpocket_top3_suggestion",
        lambda: suggest_pockets_with_fpocket(receptor_pdb, OUT_DIR / "fpocket_top3", top_n=3),
        expected="fpocket returns 1 to 3 named pockets",
        rows=rows,
    )
    if fpocket_boxes:
        rows.append(
            {
                "case": "box_fpocket_top3_count",
                "status": "PASS" if 1 <= len(fpocket_boxes) <= 3 else "FAIL",
                "expected": "1 <= pocket count <= 3",
                "actual": str(len(fpocket_boxes)),
                "seconds": 0,
            }
        )
        fpocket_batch = timed_case(
            "box_fpocket_top3_batch",
            lambda: run_batch_screen(
                project_root=ROOT,
                receptor_pdb=receptor_pdb,
                ligand_file=ligand_smi,
                box=fpocket_boxes[0].box,
                boxes=fpocket_boxes,
                output_dir=OUT_DIR / "batch_fpocket_top3",
                exhaustiveness=1,
                seed=1,
            ),
            expected="fpocket top-3 multi-pocket batch docks one ligand per pocket",
            rows=rows,
        )
        if fpocket_batch:
            assert_batch_outputs("box_fpocket_top3_outputs", fpocket_batch, rows, expected_min_success=1)

    resume_first = timed_case(
        "resume_first_run",
        lambda: run_batch_screen(
            project_root=ROOT,
            receptor_pdb=receptor_pdb,
            ligand_file=ligand_smi,
            box=box,
            output_dir=OUT_DIR / "batch_resume",
            exhaustiveness=1,
            seed=1,
        ),
        expected="Initial run for resume test succeeds",
        rows=rows,
    )
    resume_second = timed_case(
        "resume_second_cached_run",
        lambda: run_batch_screen(
            project_root=ROOT,
            receptor_pdb=receptor_pdb,
            ligand_file=ligand_smi,
            box=box,
            output_dir=OUT_DIR / "batch_resume",
            exhaustiveness=1,
            seed=1,
            resume=True,
        ),
        expected="Resume reuses cached pose and marks prep_status cached",
        rows=rows,
    )
    if resume_first and resume_second:
        rows.append(
            {
                "case": "resume_cached_status",
                "status": "PASS" if resume_second.results and resume_second.results[0].prep_status == "cached" else "FAIL",
                "expected": "Second resume run has prep_status cached",
                "actual": resume_second.results[0].prep_status if resume_second.results else "no results",
                "seconds": 0,
            }
        )

    dependency_case(
        "prep_add_hydrogens_obabel",
        shutil.which("obabel") is not None,
        lambda: prepare_receptor_input(receptor_pdb, OUT_DIR / "prepared" / "obabel_h", add_hydrogens_ph=7.4),
        rows=rows,
    )
    dependency_case(
        "prep_repair_pdbfixer",
        import_available("pdbfixer") and import_available("openmm"),
        lambda: prepare_receptor_input(receptor_pdb, OUT_DIR / "prepared" / "pdbfixer", repair_with_pdbfixer_enabled=True),
        rows=rows,
    )
    dependency_case(
        "prep_minimize_openmm",
        import_available("openmm"),
        lambda: prepare_receptor_input(receptor_pdb, OUT_DIR / "prepared" / "openmm_min", minimize_with_openmm_enabled=True),
        rows=rows,
    )

    write_reports(rows)
    failed = [row for row in rows if row["status"] != "PASS"]
    print(json.dumps({"total": len(rows), "passed": len(rows) - len(failed), "failed": len(failed), "report": str(OUT_DIR / "settings_matrix.json")}, indent=2))
    if failed:
        raise SystemExit(1)


def assert_batch_outputs(name: str, batch, rows: list[dict[str, object]], *, expected_min_success: int = 1) -> None:
    successes = [result for result in batch.results if result.docking_status == "success"]
    required_paths = [
        batch.results_csv,
        batch.results_json,
        batch.summary_txt,
        batch.boxes_json,
        batch.best_by_ligand_csv,
        batch.best_by_pocket_csv,
        batch.html_report,
        batch.pdf_report,
        batch.events_log,
        batch.manifest_json,
    ]
    artifacts_ok = all(path.exists() for path in required_paths)
    pose_artifacts_ok = all(
        result.pose_pdbqt
        and Path(result.pose_pdbqt).exists()
        and result.complex_pdb
        and Path(result.complex_pdb).exists()
        and result.interactions_csv
        and Path(result.interactions_csv).exists()
        and result.viewer_html
        and Path(result.viewer_html).exists()
        for result in successes
    )
    rows.append(
        {
            "case": name,
            "status": "PASS" if len(successes) >= expected_min_success and artifacts_ok and pose_artifacts_ok else "FAIL",
            "expected": "Batch outputs and per-pose artifacts exist",
            "actual": json.dumps(
                {
                    "successes": len(successes),
                    "required_paths": artifacts_ok,
                    "pose_artifacts": pose_artifacts_ok,
                    "outputs": {path.name: path.exists() for path in required_paths},
                },
                sort_keys=True,
            ),
            "seconds": 0,
        }
    )


def import_available(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def write_reports(rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "settings_matrix.json"
    csv_path = OUT_DIR / "settings_matrix.csv"
    txt_path = OUT_DIR / "settings_matrix.txt"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "status", "expected", "actual", "seconds"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "StrataDock comprehensive settings test",
        "=" * 40,
        f"Total: {len(rows)}",
        f"Passed: {sum(row['status'] == 'PASS' for row in rows)}",
        f"Failed: {sum(row['status'] != 'PASS' for row in rows)}",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['case']}: {row['status']} | expected: {row['expected']} | actual: {row['actual']}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
