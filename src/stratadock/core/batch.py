from __future__ import annotations

import csv
import hashlib
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from rdkit import Chem

from stratadock.core.admet import compute_basic_admet
from stratadock.core.boxes import write_box_json
from stratadock.core.docking import DockingResult, parse_gnina_scores_from_pose, parse_vina_scores_from_pose, run_docking, run_vina
from stratadock.core.interactions import analyze_interactions, build_complex_pdb, write_interactions, write_pymol_script
from stratadock.core.ligands import (
    LigandLoadError,
    LigandPrepOptions,
    LigandRecord,
    load_ligand_records_with_errors,
    load_sdf,
    prepare_ligand_record,
)
from stratadock.core.models import DockingBox, NamedDockingBox
from stratadock.core.pockets import write_pockets_json
from stratadock.core.reports import write_run_html_report, write_run_pdf_report
from stratadock.core.receptors import ReceptorPrepReport, prepare_receptor_input
from stratadock.core.visualization import write_3dmol_viewer_html


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class BatchLigandResult:
    receptor_name: str
    pocket_name: str
    ligand_name: str
    source_index: int
    prep_status: str
    docking_status: str
    vina_score: float | None
    vina_scores: list[float] | None
    docking_engine: str | None
    cnn_score: float | None
    cnn_affinity: float | None
    cnn_scores: list[float | None] | None
    cnn_affinities: list[float | None] | None
    pose_pdbqt: str | None
    complex_pdb: str | None
    interactions_json: str | None
    interactions_csv: str | None
    pymol_script: str | None
    viewer_html: str | None
    prepared_sdf: str | None
    ligand_pdbqt: str | None
    ligand_embedding_used: bool | None
    ligand_force_field: str | None
    ligand_requested_force_field: str | None
    ligand_protonation_ph: float | None
    ligand_protonation_status: str | None
    ligand_salt_stripping_status: str | None
    ligand_neutralization_status: str | None
    error: str | None
    molecular_weight: float | None
    logp: float | None
    tpsa: float | None
    hbd: int | None
    hba: int | None
    qed: float | None
    lipinski_failures: int | None
    rotatable_bonds: int | None
    heavy_atom_count: int | None
    aromatic_rings: int | None
    formal_charge: int | None
    fraction_csp3: float | None
    molar_refractivity: float | None
    rule_of_five_pass: bool | None
    rule_of_five_classification: str | None
    veber_pass: bool | None
    bbb_penetration: str | None
    herg_risk: str | None
    hepatotoxicity_risk: str | None
    mutagenicity_risk: str | None
    pains_alert_count: int | None
    brenk_alert_count: int | None
    structural_alert_count: int | None


@dataclass(frozen=True)
class BatchRunResult:
    output_dir: Path
    receptor: ReceptorPrepReport
    results: list[BatchLigandResult]
    results_csv: Path
    results_json: Path
    summary_txt: Path
    box_json: Path
    boxes_json: Path
    best_by_ligand_csv: Path
    best_by_pocket_csv: Path
    html_report: Path
    pdf_report: Path
    events_log: Path
    manifest_json: Path


@dataclass(frozen=True)
class EnsembleRunResult:
    output_dir: Path
    runs: list[BatchRunResult]
    results: list[BatchLigandResult]
    results_csv: Path
    results_json: Path
    summary_txt: Path
    boxes_json: Path
    best_by_ligand_csv: Path
    best_by_pocket_csv: Path
    html_report: Path
    pdf_report: Path
    events_log: Path
    manifest_json: Path


@dataclass(frozen=True)
class _LigandJob:
    project_root: Path
    record: LigandRecord
    receptor: ReceptorPrepReport
    receptor_name: str
    named_box: NamedDockingBox
    prepared_dir: Path
    poses_dir: Path
    artifacts_dir: Path
    exhaustiveness: int
    seed: int
    num_modes: int
    energy_range: float
    scoring: str
    engine: str
    gnina_cpu_only: bool
    gnina_device: int | str | None
    resume: bool
    ligand_prep_options: LigandPrepOptions

    def run(self) -> BatchLigandResult:
        return _run_one_ligand(
            project_root=self.project_root,
            record=self.record,
            receptor=self.receptor,
            receptor_name=self.receptor_name,
            named_box=self.named_box,
            prepared_dir=self.prepared_dir,
            poses_dir=self.poses_dir,
            artifacts_dir=self.artifacts_dir,
            exhaustiveness=self.exhaustiveness,
            seed=self.seed,
            num_modes=self.num_modes,
            energy_range=self.energy_range,
            scoring=self.scoring,
            engine=self.engine,
            gnina_cpu_only=self.gnina_cpu_only,
            gnina_device=self.gnina_device,
            resume=self.resume,
            ligand_prep_options=self.ligand_prep_options,
        )


@dataclass(frozen=True)
class _LigandJobRunResult:
    results: list[BatchLigandResult]
    cancelled: bool = False


def _admet_fields(mol: Chem.Mol) -> dict[str, float | int | bool | str | None]:
    return compute_basic_admet(mol).as_dict(include_extended=True)


def _empty_admet_fields() -> dict[str, float | int | bool | str | None]:
    return {
        "molecular_weight": None,
        "logp": None,
        "tpsa": None,
        "hbd": None,
        "hba": None,
        "qed": None,
        "lipinski_failures": None,
        "rotatable_bonds": None,
        "heavy_atom_count": None,
        "aromatic_rings": None,
        "formal_charge": None,
        "fraction_csp3": None,
        "molar_refractivity": None,
        "rule_of_five_pass": None,
        "rule_of_five_classification": None,
        "veber_pass": None,
        "bbb_penetration": None,
        "herg_risk": None,
        "hepatotoxicity_risk": None,
        "mutagenicity_risk": None,
        "pains_alert_count": None,
        "brenk_alert_count": None,
        "structural_alert_count": None,
    }


def _load_error_result(error: LigandLoadError, *, receptor_name: str, pocket_name: str) -> BatchLigandResult:
    return _error_result(
        name=f"record_{error.source_index}",
        source_index=error.source_index,
        prep_status="load_failed",
        message=error.message,
        receptor_name=receptor_name,
        pocket_name=pocket_name,
    )


def _normalize_n_jobs(n_jobs: int) -> int:
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1.")
    return n_jobs


def _result_sort_key(result: BatchLigandResult) -> tuple[bool, float, str, str, int, str]:
    return (
        result.vina_score is None,
        result.vina_score if result.vina_score is not None else 0.0,
        result.receptor_name,
        result.pocket_name,
        result.source_index,
        result.ligand_name,
    )


def _log_ligand_completed(events_log: Path, result: BatchLigandResult) -> None:
    append_run_event(
        events_log,
        "ligand_completed",
        {
            "receptor_name": result.receptor_name,
            "pocket_name": result.pocket_name,
            "ligand_name": result.ligand_name,
            "prep_status": result.prep_status,
            "docking_status": result.docking_status,
            "vina_score": result.vina_score,
            "error": result.error,
        },
    )


def _emit_progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _job_label(job: _LigandJob) -> str:
    return f"{job.receptor_name} / {job.named_box.name} / {job.record.name}"


def _stop_requested(stop_file: Path | None) -> bool:
    return bool(stop_file and stop_file.exists())


def _run_ligand_jobs(
    jobs: list[_LigandJob],
    *,
    n_jobs: int,
    stop_file: Path | None,
    events_log: Path,
    progress_callback: ProgressCallback | None = None,
) -> _LigandJobRunResult:
    results: list[BatchLigandResult] = []
    cancelled = False
    total_jobs = len(jobs)
    completed_jobs = 0

    def note_cancelled() -> None:
        nonlocal cancelled
        if cancelled:
            return
        cancelled = True
        append_run_event(events_log, "run_cancelled", {"reason": "stop_file", "stop_file": str(stop_file)})
        _emit_progress(progress_callback, "> Stop requested. No new ligand jobs will be started.")

    if n_jobs == 1:
        for index, job in enumerate(jobs, start=1):
            if _stop_requested(stop_file):
                note_cancelled()
                break
            _emit_progress(progress_callback, f"> Docking {index}/{total_jobs}: {_job_label(job)}")
            result = job.run()
            completed_jobs += 1
            results.append(result)
            _log_ligand_completed(events_log, result)
            score = f"{result.vina_score:.3f}" if result.vina_score is not None else "n/a"
            _emit_progress(
                progress_callback,
                f"> Finished {completed_jobs}/{total_jobs}: {result.ligand_name} ({result.docking_status}, score {score})",
            )
        return _LigandJobRunResult(results=results, cancelled=cancelled)

    job_iter = iter(jobs)
    futures: dict[Future[BatchLigandResult], _LigandJob] = {}
    submitted_jobs = 0

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        nonlocal submitted_jobs
        if _stop_requested(stop_file):
            note_cancelled()
            return False
        try:
            job = next(job_iter)
        except StopIteration:
            return False
        submitted_jobs += 1
        _emit_progress(progress_callback, f"> Queued {submitted_jobs}/{total_jobs}: {_job_label(job)}")
        futures[executor.submit(job.run)] = job
        return True

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for _ in range(n_jobs):
            if not submit_next(executor):
                break
        while futures:
            done, _pending = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                result = future.result()
                completed_jobs += 1
                results.append(result)
                _log_ligand_completed(events_log, result)
                score = f"{result.vina_score:.3f}" if result.vina_score is not None else "n/a"
                _emit_progress(
                    progress_callback,
                    f"> Finished {completed_jobs}/{total_jobs}: {result.ligand_name} ({result.docking_status}, score {score})",
                )
            for _ in done:
                submit_next(executor)
    return _LigandJobRunResult(results=results, cancelled=cancelled)


def _error_result(
    name: str,
    source_index: int,
    prep_status: str,
    message: str,
    *,
    receptor_name: str = "",
    pocket_name: str = "",
) -> BatchLigandResult:
    return BatchLigandResult(
        receptor_name=receptor_name,
        pocket_name=pocket_name,
        ligand_name=name,
        source_index=source_index,
        prep_status=prep_status,
        docking_status="skipped",
        vina_score=None,
        vina_scores=None,
        docking_engine=None,
        cnn_score=None,
        cnn_affinity=None,
        cnn_scores=None,
        cnn_affinities=None,
        pose_pdbqt=None,
        complex_pdb=None,
        interactions_json=None,
        interactions_csv=None,
        pymol_script=None,
        viewer_html=None,
        prepared_sdf=None,
        ligand_pdbqt=None,
        ligand_embedding_used=None,
        ligand_force_field=None,
        ligand_requested_force_field=None,
        ligand_protonation_ph=None,
        ligand_protonation_status=None,
        ligand_salt_stripping_status=None,
        ligand_neutralization_status=None,
        error=message,
        **_empty_admet_fields(),
    )


def run_batch_screen(
    *,
    project_root: Path,
    receptor_pdb: Path,
    ligand_file: Path,
    box: DockingBox,
    output_dir: Path,
    boxes: list[NamedDockingBox] | None = None,
    boxes_by_receptor: list[list[NamedDockingBox]] | None = None,
    exhaustiveness: int = 4,
    seed: int = 1,
    num_modes: int = 1,
    energy_range: float = 3.0,
    scoring: str = "vina",
    engine: str = "vina",
    gnina_cpu_only: bool = False,
    gnina_device: int | str | None = None,
    remove_waters: bool = True,
    remove_non_protein_heteroatoms: bool = True,
    keep_metals: bool = True,
    default_altloc: str = "A",
    add_hydrogens_ph: float | None = None,
    repair_with_pdbfixer: bool = False,
    minimize_with_openmm: bool = False,
    resume: bool = False,
    stop_file: Path | None = None,
    n_jobs: int = 1,
    ligand_prep_options: LigandPrepOptions | None = None,
    progress_callback: ProgressCallback | None = None,
) -> BatchRunResult:
    n_jobs = _normalize_n_jobs(n_jobs)
    ligand_prep_options = ligand_prep_options or LigandPrepOptions()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = output_dir / "prepared"
    poses_dir = output_dir / "poses"
    events_log = output_dir / "run_events.jsonl"
    if events_log.exists() and not resume:
        events_log.unlink()
    append_run_event(
        events_log,
        "run_started",
        {
            "receptor_pdb": str(receptor_pdb),
            "ligand_file": str(ligand_file),
            "engine": engine,
            "gnina_cpu_only": gnina_cpu_only,
            "gnina_device": gnina_device,
            "exhaustiveness": exhaustiveness,
            "num_modes": num_modes,
        },
    )
    _emit_progress(progress_callback, f"> Preparing receptor: {receptor_pdb.name}")
    prepared_dir.mkdir(parents=True, exist_ok=True)
    poses_dir.mkdir(parents=True, exist_ok=True)

    receptor = prepare_receptor_input(
        receptor_pdb,
        prepared_dir / "receptor",
        remove_waters=remove_waters,
        remove_non_protein_heteroatoms=remove_non_protein_heteroatoms,
        keep_metals=keep_metals,
        default_altloc=default_altloc,
        add_hydrogens_ph=add_hydrogens_ph,
        repair_with_pdbfixer_enabled=repair_with_pdbfixer,
        minimize_with_openmm_enabled=minimize_with_openmm,
    )
    _emit_progress(progress_callback, "> Receptor preparation complete.")
    named_boxes = boxes or [NamedDockingBox(name="box_1", box=box, source="single")]
    receptor_name = receptor_pdb.stem
    records, load_errors = load_ligand_records_with_errors(ligand_file)
    _emit_progress(progress_callback, f"> Loaded {len(records)} valid ligand record(s); {len(load_errors)} invalid.")
    results: list[BatchLigandResult] = []
    batch_cancelled = False
    for error in load_errors:
        append_run_event(events_log, "ligand_load_failed", {"source_index": error.source_index, "error": error.message})
        _emit_progress(progress_callback, f"> Skipped invalid ligand record {error.source_index}: {error.message}")

    jobs: list[_LigandJob] = []
    for named_box in named_boxes:
        if _stop_requested(stop_file):
            batch_cancelled = True
            append_run_event(events_log, "run_cancelled", {"reason": "stop_file", "stop_file": str(stop_file)})
            break
        results.extend(
            _load_error_result(error, receptor_name=receptor_name, pocket_name=named_box.name)
            for error in load_errors
        )
        for record in records:
            jobs.append(
                _LigandJob(
                    project_root=project_root,
                    record=record,
                    receptor=receptor,
                    receptor_name=receptor_name,
                    named_box=named_box,
                    prepared_dir=prepared_dir / named_box.name,
                    poses_dir=poses_dir / named_box.name,
                    artifacts_dir=output_dir / "artifacts" / named_box.name,
                    exhaustiveness=exhaustiveness,
                    seed=seed,
                    num_modes=num_modes,
                    energy_range=energy_range,
                    scoring=scoring,
                    engine=engine,
                    gnina_cpu_only=gnina_cpu_only,
                    gnina_device=gnina_device,
                    resume=resume,
                    ligand_prep_options=ligand_prep_options,
                )
            )

    _emit_progress(progress_callback, f"> Starting {len(jobs)} ligand docking job(s) across {len(named_boxes)} pocket(s).")
    job_run = _run_ligand_jobs(
        jobs,
        n_jobs=n_jobs,
        stop_file=stop_file,
        events_log=events_log,
        progress_callback=progress_callback,
    )
    results.extend(job_run.results)
    _emit_progress(progress_callback, "> Writing result tables and reports.")
    results.sort(key=_result_sort_key)
    results_csv = output_dir / "results.csv"
    results_json = output_dir / "results.json"
    summary_txt = output_dir / "run_summary.txt"
    box_json = write_box_json(named_boxes[0].box, output_dir / "box.json")
    boxes_json = write_pockets_json(named_boxes, output_dir / "boxes.json")
    write_results_csv(results, results_csv)
    write_results_json(results, results_json)
    write_summary(results, summary_txt)
    best_by_ligand_csv = write_best_summary_csv(results, output_dir / "best_by_ligand.csv", group_field="ligand_name")
    best_by_pocket_csv = write_best_summary_csv_for_fields(
        results,
        output_dir / "best_by_pocket.csv",
        group_fields=["receptor_name", "pocket_name"],
    )
    manifest_json = write_run_manifest(
        path=output_dir / "run_manifest.json",
        receptor_pdb=receptor_pdb,
        ligand_file=ligand_file,
        boxes=named_boxes,
        receptor=receptor,
        results_csv=results_csv,
        results_json=results_json,
        summary_txt=summary_txt,
        box_json=box_json,
        boxes_json=boxes_json,
        best_by_ligand_csv=best_by_ligand_csv,
        best_by_pocket_csv=best_by_pocket_csv,
        events_log=events_log,
        exhaustiveness=exhaustiveness,
        seed=seed,
        num_modes=num_modes,
        energy_range=energy_range,
        scoring=scoring,
        engine=engine,
        gnina_cpu_only=gnina_cpu_only,
        gnina_device=gnina_device,
        remove_waters=remove_waters,
        remove_non_protein_heteroatoms=remove_non_protein_heteroatoms,
        keep_metals=keep_metals,
        default_altloc=default_altloc,
        n_jobs=n_jobs,
        ligand_prep_options=ligand_prep_options,
        total=len(results),
        success=sum(1 for result in results if result.docking_status == "success"),
    )
    html_report = write_run_html_report(
        title="StrataDock Batch Report",
        results=[_result_export_row(result) for result in results],
        output_path=output_dir / "report.html",
        manifest_json=manifest_json,
        summary_txt=summary_txt,
        receptor_prep_summary=_receptor_prep_summary(receptor),
        fpocket_pocket_summary=_pocket_summary(named_boxes),
        receptor_prep_warnings=_receptor_prep_warnings(receptor),
    )
    pdf_report = write_run_pdf_report(
        title="StrataDock Batch Report",
        results=[_result_export_row(result) for result in results],
        output_path=output_dir / "report.pdf",
        manifest_json=manifest_json,
        summary_txt=summary_txt,
        receptor_prep_summary=_receptor_prep_summary(receptor),
        fpocket_pocket_summary=_pocket_summary(named_boxes),
        receptor_prep_warnings=_receptor_prep_warnings(receptor),
    )
    terminal_event = (
        "run_cancelled"
        if batch_cancelled or job_run.cancelled or (_stop_requested(stop_file) and len(job_run.results) < len(jobs))
        else "run_completed"
    )
    append_run_event(
        events_log,
        terminal_event,
        {"total": len(results), "success": sum(1 for result in results if result.docking_status == "success")},
    )
    _emit_progress(progress_callback, "> Reports and session manifest written.")
    return BatchRunResult(
        output_dir=output_dir,
        receptor=receptor,
        results=results,
        results_csv=results_csv,
        results_json=results_json,
        summary_txt=summary_txt,
        box_json=box_json,
        boxes_json=boxes_json,
        best_by_ligand_csv=best_by_ligand_csv,
        best_by_pocket_csv=best_by_pocket_csv,
        html_report=html_report,
        pdf_report=pdf_report,
        events_log=events_log,
        manifest_json=manifest_json,
    )


def run_ensemble_screen(
    *,
    project_root: Path,
    receptor_pdbs: list[Path],
    ligand_file: Path,
    box: DockingBox,
    output_dir: Path,
    boxes: list[NamedDockingBox] | None = None,
    boxes_by_receptor: list[list[NamedDockingBox]] | None = None,
    exhaustiveness: int = 4,
    seed: int = 1,
    num_modes: int = 1,
    energy_range: float = 3.0,
    scoring: str = "vina",
    engine: str = "vina",
    gnina_cpu_only: bool = False,
    gnina_device: int | str | None = None,
    remove_waters: bool = True,
    remove_non_protein_heteroatoms: bool = True,
    keep_metals: bool = True,
    default_altloc: str = "A",
    add_hydrogens_ph: float | None = None,
    repair_with_pdbfixer: bool = False,
    minimize_with_openmm: bool = False,
    resume: bool = False,
    stop_file: Path | None = None,
    n_jobs: int = 1,
    ligand_prep_options: LigandPrepOptions | None = None,
    progress_callback: ProgressCallback | None = None,
) -> EnsembleRunResult:
    n_jobs = _normalize_n_jobs(n_jobs)
    ligand_prep_options = ligand_prep_options or LigandPrepOptions()
    if not receptor_pdbs:
        raise ValueError("At least one receptor PDB is required.")
    output_dir.mkdir(parents=True, exist_ok=True)
    events_log = output_dir / "run_events.jsonl"
    if events_log.exists() and not resume:
        events_log.unlink()
    append_run_event(events_log, "ensemble_started", {"receptor_count": len(receptor_pdbs), "ligand_file": str(ligand_file)})
    _emit_progress(progress_callback, f"> Starting ensemble docking for {len(receptor_pdbs)} receptor(s).")
    named_boxes = boxes or [NamedDockingBox(name="box_1", box=box, source="single")]
    if boxes_by_receptor is not None and len(boxes_by_receptor) != len(receptor_pdbs):
        raise ValueError("boxes_by_receptor must have one box list per receptor.")
    runs: list[BatchRunResult] = []
    results: list[BatchLigandResult] = []
    for idx, receptor_pdb in enumerate(receptor_pdbs, start=1):
        if stop_file and stop_file.exists():
            append_run_event(events_log, "ensemble_cancelled", {"reason": "stop_file", "stop_file": str(stop_file)})
            _emit_progress(progress_callback, "> Stop requested before next receptor.")
            break
        _emit_progress(progress_callback, f"> Receptor {idx}/{len(receptor_pdbs)}: {receptor_pdb.name}")
        run_dir = output_dir / f"receptor_{idx}_{receptor_pdb.stem}"
        receptor_boxes = boxes_by_receptor[idx - 1] if boxes_by_receptor is not None else named_boxes
        if not receptor_boxes:
            raise ValueError(f"No docking boxes available for receptor: {receptor_pdb}")
        run = run_batch_screen(
            project_root=project_root,
            receptor_pdb=receptor_pdb,
            ligand_file=ligand_file,
            box=receptor_boxes[0].box,
            output_dir=run_dir,
            boxes=receptor_boxes,
            exhaustiveness=exhaustiveness,
            seed=seed,
            num_modes=num_modes,
            energy_range=energy_range,
            scoring=scoring,
            engine=engine,
            gnina_cpu_only=gnina_cpu_only,
            gnina_device=gnina_device,
            remove_waters=remove_waters,
            remove_non_protein_heteroatoms=remove_non_protein_heteroatoms,
            keep_metals=keep_metals,
            default_altloc=default_altloc,
            add_hydrogens_ph=add_hydrogens_ph,
            repair_with_pdbfixer=repair_with_pdbfixer,
            minimize_with_openmm=minimize_with_openmm,
            resume=resume,
            stop_file=stop_file,
            n_jobs=n_jobs,
            ligand_prep_options=ligand_prep_options,
            progress_callback=progress_callback,
        )
        runs.append(run)
        results.extend(run.results)
        append_run_event(
            events_log,
            "receptor_completed",
            {
                "receptor_pdb": str(receptor_pdb),
                "output_dir": str(run.output_dir),
                "total": len(run.results),
                "success": sum(result.docking_status == "success" for result in run.results),
            },
        )
        _emit_progress(progress_callback, f"> Receptor {idx}/{len(receptor_pdbs)} complete.")

    results.sort(key=_result_sort_key)
    results_csv = output_dir / "results.csv"
    results_json = output_dir / "results.json"
    summary_txt = output_dir / "run_summary.txt"
    manifest_boxes = _ensemble_manifest_boxes(receptor_pdbs, boxes_by_receptor) if boxes_by_receptor is not None else named_boxes
    boxes_json = write_pockets_json(manifest_boxes, output_dir / "boxes.json")
    write_results_csv(results, results_csv)
    write_results_json(results, results_json)
    write_summary(results, summary_txt)
    best_by_ligand_csv = write_best_summary_csv(results, output_dir / "best_by_ligand.csv", group_field="ligand_name")
    best_by_pocket_csv = write_best_summary_csv_for_fields(
        results,
        output_dir / "best_by_pocket.csv",
        group_fields=["receptor_name", "pocket_name"],
    )
    manifest_json = write_ensemble_manifest(
        path=output_dir / "run_manifest.json",
        receptor_pdbs=receptor_pdbs,
        ligand_file=ligand_file,
        boxes=manifest_boxes,
        runs=runs,
        results_csv=results_csv,
        results_json=results_json,
        summary_txt=summary_txt,
        boxes_json=boxes_json,
        best_by_ligand_csv=best_by_ligand_csv,
        best_by_pocket_csv=best_by_pocket_csv,
        events_log=events_log,
        exhaustiveness=exhaustiveness,
        seed=seed,
        num_modes=num_modes,
        energy_range=energy_range,
        scoring=scoring,
        engine=engine,
        gnina_cpu_only=gnina_cpu_only,
        gnina_device=gnina_device,
        remove_waters=remove_waters,
        remove_non_protein_heteroatoms=remove_non_protein_heteroatoms,
        keep_metals=keep_metals,
        default_altloc=default_altloc,
        n_jobs=n_jobs,
        ligand_prep_options=ligand_prep_options,
        total=len(results),
        success=sum(1 for result in results if result.docking_status == "success"),
    )
    html_report = write_run_html_report(
        title="StrataDock Ensemble Report",
        results=[_result_export_row(result) for result in results],
        output_path=output_dir / "report.html",
        manifest_json=manifest_json,
        summary_txt=summary_txt,
        fpocket_pocket_summary=_pocket_summary(manifest_boxes),
        receptor_prep_warnings=[warning for run in runs for warning in _receptor_prep_warnings(run.receptor)],
    )
    pdf_report = write_run_pdf_report(
        title="StrataDock Ensemble Report",
        results=[_result_export_row(result) for result in results],
        output_path=output_dir / "report.pdf",
        manifest_json=manifest_json,
        summary_txt=summary_txt,
        fpocket_pocket_summary=_pocket_summary(manifest_boxes),
        receptor_prep_warnings=[warning for run in runs for warning in _receptor_prep_warnings(run.receptor)],
    )
    append_run_event(
        events_log,
        "ensemble_completed",
        {"total": len(results), "success": sum(result.docking_status == "success" for result in results)},
    )
    _emit_progress(progress_callback, "> Ensemble reports and session manifest written.")
    return EnsembleRunResult(
        output_dir=output_dir,
        runs=runs,
        results=results,
        results_csv=results_csv,
        results_json=results_json,
        summary_txt=summary_txt,
        boxes_json=boxes_json,
        best_by_ligand_csv=best_by_ligand_csv,
        best_by_pocket_csv=best_by_pocket_csv,
        html_report=html_report,
        pdf_report=pdf_report,
        events_log=events_log,
        manifest_json=manifest_json,
    )


def _run_one_ligand(
    *,
    project_root: Path,
    record: LigandRecord,
    receptor: ReceptorPrepReport,
    receptor_name: str,
    named_box: NamedDockingBox,
    prepared_dir: Path,
    poses_dir: Path,
    artifacts_dir: Path,
    exhaustiveness: int,
    seed: int,
    num_modes: int,
    energy_range: float,
    scoring: str,
    engine: str,
    gnina_cpu_only: bool,
    gnina_device: int | str | None,
    resume: bool,
    ligand_prep_options: LigandPrepOptions,
) -> BatchLigandResult:
    admet = _admet_fields(record.mol)
    try:
        ligand = prepare_ligand_record(record, prepared_dir, options=ligand_prep_options)
        prepared_mols = load_sdf(ligand.prepared_sdf)
        if prepared_mols:
            admet = _admet_fields(prepared_mols[0])
    except Exception as exc:
        return BatchLigandResult(
            receptor_name=receptor_name,
            pocket_name=named_box.name,
            ligand_name=record.name,
            source_index=record.source_index,
            prep_status="failed",
            docking_status="skipped",
            vina_score=None,
            vina_scores=None,
            docking_engine=None,
            cnn_score=None,
            cnn_affinity=None,
            cnn_scores=None,
            cnn_affinities=None,
            pose_pdbqt=None,
            complex_pdb=None,
            interactions_json=None,
            interactions_csv=None,
            pymol_script=None,
            viewer_html=None,
            prepared_sdf=None,
            ligand_pdbqt=None,
            ligand_embedding_used=None,
            ligand_force_field=None,
            ligand_requested_force_field=None,
            ligand_protonation_ph=None,
            ligand_protonation_status=None,
            ligand_salt_stripping_status=None,
            ligand_neutralization_status=None,
            error=str(exc),
            **admet,
        )

    pose_path = poses_dir / f"{ligand.name}_pose.pdbqt"
    cache_meta_path = _pose_cache_metadata_path(pose_path)
    cache_payload = _pose_cache_payload(
        receptor_pdbqt=receptor.pdbqt,
        ligand_pdbqt=ligand.pdbqt,
        box=named_box.box,
        exhaustiveness=exhaustiveness,
        seed=seed,
        num_modes=num_modes,
        energy_range=energy_range,
        scoring=scoring,
        engine=engine,
        gnina_cpu_only=gnina_cpu_only,
        gnina_device=gnina_device,
        ligand_prep_options=ligand_prep_options,
    )
    used_cache = False
    try:
        if resume and pose_path.exists() and _pose_cache_is_valid(cache_meta_path, cache_payload):
            docked = _cached_docking_result(pose_path, engine=engine)
            used_cache = True
        else:
            docked = _execute_docking(
                engine=engine,
                project_root=project_root,
                receptor_pdbqt=receptor.pdbqt,
                ligand_pdbqt=ligand.pdbqt,
                box=named_box.box,
                output_pdbqt=pose_path,
                exhaustiveness=exhaustiveness,
                seed=seed,
                num_modes=num_modes,
                energy_range=energy_range,
                scoring=scoring,
                gnina_cpu_only=gnina_cpu_only,
                gnina_device=gnina_device,
            )
            _write_pose_cache_metadata(cache_meta_path, cache_payload)
    except Exception as exc:
        return BatchLigandResult(
            receptor_name=receptor_name,
            pocket_name=named_box.name,
            ligand_name=ligand.name,
            source_index=record.source_index,
            prep_status="success",
            docking_status="failed",
            vina_score=None,
            vina_scores=None,
            docking_engine=engine,
            cnn_score=None,
            cnn_affinity=None,
            cnn_scores=None,
            cnn_affinities=None,
            pose_pdbqt=None,
            complex_pdb=None,
            interactions_json=None,
            interactions_csv=None,
            pymol_script=None,
            viewer_html=None,
            prepared_sdf=str(ligand.prepared_sdf),
            ligand_pdbqt=str(ligand.pdbqt),
            ligand_embedding_used=ligand.embedding_used,
            ligand_force_field=ligand.force_field,
            ligand_requested_force_field=ligand.requested_force_field,
            ligand_protonation_ph=ligand.protonation_ph,
            ligand_protonation_status=ligand.protonation_status,
            ligand_salt_stripping_status=ligand.salt_stripping_status,
            ligand_neutralization_status=ligand.neutralization_status,
            error=str(exc),
            **admet,
        )

    safe_ligand = ligand.name
    try:
        complex_pdb = build_complex_pdb(
            receptor_pdb=receptor.prepared_pdb,
            pose_pdbqt=docked.pose_pdbqt,
            output_pdb=artifacts_dir / f"{safe_ligand}_complex.pdb",
        )
        interactions = analyze_interactions(receptor_pdb=receptor.prepared_pdb, pose_pdbqt=docked.pose_pdbqt)
        interactions_json, interactions_csv = write_interactions(
            interactions,
            artifacts_dir / f"{safe_ligand}_interactions.json",
            artifacts_dir / f"{safe_ligand}_interactions.csv",
        )
        pymol_script = write_pymol_script(complex_pdb=complex_pdb, output_pml=artifacts_dir / f"{safe_ligand}_visualize.pml")
        viewer_html = write_3dmol_viewer_html(
            complex_pdb=complex_pdb,
            output_html=artifacts_dir / f"{safe_ligand}_viewer.html",
            title=f"{receptor_name} / {named_box.name} / {ligand.name}",
        )
    except Exception as exc:
        return BatchLigandResult(
            receptor_name=receptor_name,
            pocket_name=named_box.name,
            ligand_name=ligand.name,
            source_index=record.source_index,
            prep_status="cached" if used_cache else "success",
            docking_status="artifact_failed",
            vina_score=docked.score,
            vina_scores=docked.scores,
            docking_engine=docked.engine,
            cnn_score=docked.cnn_scores[0] if docked.cnn_scores else None,
            cnn_affinity=docked.cnn_affinities[0] if docked.cnn_affinities else None,
            cnn_scores=docked.cnn_scores,
            cnn_affinities=docked.cnn_affinities,
            pose_pdbqt=str(docked.pose_pdbqt),
            complex_pdb=None,
            interactions_json=None,
            interactions_csv=None,
            pymol_script=None,
            viewer_html=None,
            prepared_sdf=str(ligand.prepared_sdf),
            ligand_pdbqt=str(ligand.pdbqt),
            ligand_embedding_used=ligand.embedding_used,
            ligand_force_field=ligand.force_field,
            ligand_requested_force_field=ligand.requested_force_field,
            ligand_protonation_ph=ligand.protonation_ph,
            ligand_protonation_status=ligand.protonation_status,
            ligand_salt_stripping_status=ligand.salt_stripping_status,
            ligand_neutralization_status=ligand.neutralization_status,
            error=f"Artifact generation failed: {exc}",
            **admet,
        )

    return BatchLigandResult(
        receptor_name=receptor_name,
        pocket_name=named_box.name,
        ligand_name=ligand.name,
        source_index=record.source_index,
        prep_status="cached" if used_cache else "success",
        docking_status="success",
        vina_score=docked.score,
        vina_scores=docked.scores,
        docking_engine=docked.engine,
        cnn_score=docked.cnn_scores[0] if docked.cnn_scores else None,
        cnn_affinity=docked.cnn_affinities[0] if docked.cnn_affinities else None,
        cnn_scores=docked.cnn_scores,
        cnn_affinities=docked.cnn_affinities,
        pose_pdbqt=str(docked.pose_pdbqt),
        complex_pdb=str(complex_pdb),
        interactions_json=str(interactions_json),
        interactions_csv=str(interactions_csv),
        pymol_script=str(pymol_script),
        viewer_html=str(viewer_html),
        prepared_sdf=str(ligand.prepared_sdf),
        ligand_pdbqt=str(ligand.pdbqt),
        ligand_embedding_used=ligand.embedding_used,
        ligand_force_field=ligand.force_field,
        ligand_requested_force_field=ligand.requested_force_field,
        ligand_protonation_ph=ligand.protonation_ph,
        ligand_protonation_status=ligand.protonation_status,
        ligand_salt_stripping_status=ligand.salt_stripping_status,
        ligand_neutralization_status=ligand.neutralization_status,
        error=None,
        **admet,
    )


def _cached_docking_result(pose_path: Path, *, engine: str) -> DockingResult:
    engine = engine.lower()
    if engine == "vina":
        scores = parse_vina_scores_from_pose(pose_path)
        if not scores:
            raise RuntimeError(f"Cached pose is missing Vina score: {pose_path}")
        return DockingResult(score=scores[0], scores=scores, pose_pdbqt=pose_path, log="cached", engine="vina")
    if engine == "gnina":
        gnina_scores = parse_gnina_scores_from_pose(pose_path)
        if not gnina_scores:
            raise RuntimeError(f"Cached pose is missing GNINA scores: {pose_path}")
        scores = [
            score.affinity if score.affinity is not None else score.cnn_affinity if score.cnn_affinity is not None else score.cnn_score
            for score in gnina_scores
        ]
        if any(score is None for score in scores):
            raise RuntimeError(f"Cached pose has incomplete GNINA scores: {pose_path}")
        numeric_scores = [float(score) for score in scores if score is not None]
        return DockingResult(
            score=numeric_scores[0],
            scores=numeric_scores,
            pose_pdbqt=pose_path,
            log="cached",
            engine="gnina",
            cnn_scores=[score.cnn_score for score in gnina_scores],
            cnn_affinities=[score.cnn_affinity for score in gnina_scores],
        )
    raise ValueError("engine must be 'vina' or 'gnina'.")


def _pose_cache_metadata_path(pose_path: Path) -> Path:
    return pose_path.with_suffix(pose_path.suffix + ".cache.json")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pose_cache_payload(
    *,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    box: DockingBox,
    exhaustiveness: int,
    seed: int,
    num_modes: int,
    energy_range: float,
    scoring: str,
    engine: str,
    gnina_cpu_only: bool,
    gnina_device: int | str | None,
    ligand_prep_options: LigandPrepOptions,
) -> dict[str, object]:
    return {
        "version": 1,
        "receptor_sha256": _file_digest(receptor_pdbqt),
        "ligand_sha256": _file_digest(ligand_pdbqt),
        "box": box.as_dict(),
        "exhaustiveness": exhaustiveness,
        "seed": seed,
        "num_modes": num_modes,
        "energy_range": energy_range,
        "scoring": scoring,
        "engine": engine.lower(),
        "gnina_cpu_only": gnina_cpu_only,
        "gnina_device": str(gnina_device) if gnina_device is not None else None,
        "ligand_prep_options": ligand_prep_options.as_dict(),
    }


def _pose_cache_is_valid(path: Path, payload: dict[str, object]) -> bool:
    if not path.exists():
        return False
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return existing == payload


def _write_pose_cache_metadata(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _execute_docking(
    *,
    engine: str,
    project_root: Path,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    box: DockingBox,
    output_pdbqt: Path,
    exhaustiveness: int,
    seed: int,
    num_modes: int,
    energy_range: float,
    scoring: str,
    gnina_cpu_only: bool,
    gnina_device: int | str | None,
) -> DockingResult:
    if engine.lower() == "vina":
        return run_vina(
            project_root=project_root,
            receptor_pdbqt=receptor_pdbqt,
            ligand_pdbqt=ligand_pdbqt,
            box=box,
            output_pdbqt=output_pdbqt,
            exhaustiveness=exhaustiveness,
            seed=seed,
            num_modes=num_modes,
            energy_range=energy_range,
            scoring=scoring,
        )
    return run_docking(
        engine=engine,
        project_root=project_root,
        receptor_pdbqt=receptor_pdbqt,
        ligand_pdbqt=ligand_pdbqt,
        box=box,
        output_pdbqt=output_pdbqt,
        exhaustiveness=exhaustiveness,
        seed=seed,
        num_modes=num_modes,
        energy_range=energy_range,
        scoring=scoring,
        gnina_cpu_only=gnina_cpu_only,
        gnina_device=gnina_device,
    )


def write_results_csv(results: list[BatchLigandResult], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(_result_export_row(results[0]).keys()) if results else _result_export_fieldnames()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(_result_export_row(result))
    return path


def write_results_json(results: list[BatchLigandResult], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([_result_export_row(result) for result in results], indent=2), encoding="utf-8")
    return path


def write_summary(results: list[BatchLigandResult], path: Path) -> Path:
    total = len(results)
    docked = [result for result in results if result.docking_status == "success"]
    failed = total - len(docked)
    lines = [
        "StrataDock v 1.6.01 batch summary",
        "=" * 32,
        f"Total records: {total}",
        f"Docked successfully: {len(docked)}",
        f"Failed/skipped: {failed}",
    ]
    unfavorable = [
        result
        for result in docked
        if result.vina_score is not None and result.vina_score >= 0
    ]
    if unfavorable:
        lines.extend(
            [
                f"Unfavorable non-negative scores: {len(unfavorable)}",
                "These completed but should be treated as low-confidence/clashing poses.",
            ]
        )
    if docked:
        best = min(docked, key=lambda result: result.vina_score if result.vina_score is not None else 999.0)
        engine = best.docking_engine or "vina"
        score_label = "Best GNINA affinity" if engine == "gnina" else "Best Vina score"
        lines.extend(
            [
                "",
                f"Best ligand: {best.ligand_name}",
                f"{score_label}: {best.vina_score}",
            ]
        )
        if engine == "gnina":
            lines.append(f"Best GNINA CNNscore: {best.cnn_score}")
            lines.append(f"Best GNINA CNNaffinity: {best.cnn_affinity}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_best_summary_csv(results: list[BatchLigandResult], path: Path, *, group_field: str) -> Path:
    return write_best_summary_csv_for_fields(results, path, group_fields=[group_field])


def write_best_summary_csv_for_fields(results: list[BatchLigandResult], path: Path, *, group_fields: list[str]) -> Path:
    successes = [result for result in results if result.docking_status == "success" and result.vina_score is not None]
    best: dict[tuple[str, ...], BatchLigandResult] = {}
    for result in successes:
        key = tuple(str(getattr(result, field)) for field in group_fields)
        current = best.get(key)
        current_score = current.vina_score if current is not None and current.vina_score is not None else 999.0
        if current is None or (result.vina_score is not None and result.vina_score < current_score):
            best[key] = result
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(_result_export_row(successes[0]).keys()) if successes else _result_export_fieldnames()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(best):
            writer.writerow(_result_export_row(best[key]))
    return path


def _score_label(engine: str | None) -> str:
    return "GNINA affinity" if (engine or "").lower() == "gnina" else "Vina score"


def _score_type(engine: str | None) -> str:
    return "gnina_affinity" if (engine or "").lower() == "gnina" else "vina_score"


def _result_export_row(result: BatchLigandResult) -> dict[str, object]:
    row = asdict(result)
    row["docking_score"] = result.vina_score
    row["docking_scores"] = result.vina_scores
    row["score_type"] = _score_type(result.docking_engine)
    row["score_label"] = _score_label(result.docking_engine)
    return row


def _result_export_fieldnames() -> list[str]:
    return list(BatchLigandResult.__dataclass_fields__.keys()) + [
        "docking_score",
        "docking_scores",
        "score_type",
        "score_label",
    ]


def append_run_event(path: Path, event: str, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "time_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _ensemble_manifest_boxes(
    receptor_pdbs: list[Path],
    boxes_by_receptor: list[list[NamedDockingBox]] | None,
) -> list[NamedDockingBox]:
    if boxes_by_receptor is None:
        return []
    boxes: list[NamedDockingBox] = []
    for receptor_pdb, receptor_boxes in zip(receptor_pdbs, boxes_by_receptor, strict=False):
        for named_box in receptor_boxes:
            boxes.append(
                NamedDockingBox(
                    name=f"{receptor_pdb.stem}:{named_box.name}",
                    box=named_box.box,
                    source=named_box.source,
                    rank=named_box.rank,
                    score=named_box.score,
                    druggability_score=named_box.druggability_score,
                )
            )
    return boxes


def _receptor_prep_summary(receptor: ReceptorPrepReport) -> dict[str, object]:
    if not hasattr(receptor, "clean_report"):
        return {}
    report = receptor.clean_report
    return {
        "atoms_kept": report.atoms_kept,
        "waters_removed": report.waters_removed,
        "hetero_removed": report.hetero_removed,
        "metals_kept": report.metals_kept,
        "residue_count": report.residue_count,
        "protein_residue_count": report.protein_residue_count,
        "prep_steps": ", ".join(receptor.prep_steps),
    }


def _receptor_prep_warnings(receptor: ReceptorPrepReport) -> list[str]:
    if not hasattr(receptor, "clean_report"):
        return []
    return list(receptor.clean_report.warnings)


def _pocket_summary(boxes: list[NamedDockingBox]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for box in boxes:
        if box.source != "fpocket":
            continue
        rows.append(
            {
                "name": box.name,
                "rank": box.rank,
                "score": box.score,
                "druggability_score": box.druggability_score,
                "center_x": box.box.center_x,
                "center_y": box.box.center_y,
                "center_z": box.box.center_z,
            }
        )
    return rows


def write_run_manifest(
    *,
    path: Path,
    receptor_pdb: Path,
    ligand_file: Path,
    boxes: list[NamedDockingBox],
    receptor: ReceptorPrepReport,
    results_csv: Path,
    results_json: Path,
    summary_txt: Path,
    box_json: Path,
    boxes_json: Path,
    best_by_ligand_csv: Path,
    best_by_pocket_csv: Path,
    events_log: Path,
    exhaustiveness: int,
    seed: int,
    num_modes: int,
    energy_range: float,
    scoring: str,
    engine: str,
    gnina_cpu_only: bool,
    gnina_device: int | str | None,
    remove_waters: bool,
    remove_non_protein_heteroatoms: bool,
    keep_metals: bool,
    default_altloc: str,
    n_jobs: int,
    ligand_prep_options: LigandPrepOptions,
    total: int,
    success: int,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v 1.6.01",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": {
            "receptor_pdb": str(receptor_pdb),
            "ligand_file": str(ligand_file),
        },
        "parameters": {
            "box": boxes[0].box.as_dict(),
            "boxes": [box.as_dict() for box in boxes],
            "engine": engine,
            "exhaustiveness": exhaustiveness,
            "seed": seed,
            "num_modes": num_modes,
            "energy_range": energy_range,
            "scoring": scoring,
            "gnina_cpu_only": gnina_cpu_only,
            "gnina_device": gnina_device,
            "remove_waters": remove_waters,
            "remove_non_protein_heteroatoms": remove_non_protein_heteroatoms,
            "keep_metals": keep_metals,
            "default_altloc": default_altloc,
            "n_jobs": n_jobs,
            "ligand_prep": ligand_prep_options.as_dict(),
        },
        "outputs": {
            "results_csv": str(results_csv),
            "results_json": str(results_json),
            "summary_txt": str(summary_txt),
            "box_json": str(box_json),
            "boxes_json": str(boxes_json),
            "best_by_ligand_csv": str(best_by_ligand_csv),
            "best_by_pocket_csv": str(best_by_pocket_csv),
            "html_report": str(path.parent / "report.html"),
            "pdf_report": str(path.parent / "report.pdf"),
            "events_log": str(events_log),
            "receptor_report_json": str(receptor.report_json),
            "receptor_report_txt": str(receptor.report_txt),
        },
        "counts": {
            "total": total,
            "success": success,
            "failed_or_skipped": total - success,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_ensemble_manifest(
    *,
    path: Path,
    receptor_pdbs: list[Path],
    ligand_file: Path,
    boxes: list[NamedDockingBox],
    runs: list[BatchRunResult],
    results_csv: Path,
    results_json: Path,
    summary_txt: Path,
    boxes_json: Path,
    best_by_ligand_csv: Path,
    best_by_pocket_csv: Path,
    events_log: Path,
    exhaustiveness: int,
    seed: int,
    num_modes: int,
    energy_range: float,
    scoring: str,
    engine: str,
    gnina_cpu_only: bool,
    gnina_device: int | str | None,
    remove_waters: bool,
    remove_non_protein_heteroatoms: bool,
    keep_metals: bool,
    default_altloc: str,
    n_jobs: int,
    ligand_prep_options: LigandPrepOptions,
    total: int,
    success: int,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v 1.6.01",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": {
            "receptor_pdbs": [str(path) for path in receptor_pdbs],
            "ligand_file": str(ligand_file),
        },
        "parameters": {
            "boxes": [box.as_dict() for box in boxes],
            "engine": engine,
            "exhaustiveness": exhaustiveness,
            "seed": seed,
            "num_modes": num_modes,
            "energy_range": energy_range,
            "scoring": scoring,
            "gnina_cpu_only": gnina_cpu_only,
            "gnina_device": gnina_device,
            "remove_waters": remove_waters,
            "remove_non_protein_heteroatoms": remove_non_protein_heteroatoms,
            "keep_metals": keep_metals,
            "default_altloc": default_altloc,
            "n_jobs": n_jobs,
            "ligand_prep": ligand_prep_options.as_dict(),
        },
        "outputs": {
            "results_csv": str(results_csv),
            "results_json": str(results_json),
            "summary_txt": str(summary_txt),
            "boxes_json": str(boxes_json),
            "best_by_ligand_csv": str(best_by_ligand_csv),
            "best_by_pocket_csv": str(best_by_pocket_csv),
            "html_report": str(path.parent / "report.html"),
            "pdf_report": str(path.parent / "report.pdf"),
            "events_log": str(events_log),
            "per_receptor_runs": [
                {
                    "output_dir": str(run.output_dir),
                    "results_csv": str(run.results_csv),
                    "manifest_json": str(run.manifest_json),
                    "receptor_report_json": str(run.receptor.report_json),
                }
                for run in runs
            ],
        },
        "counts": {
            "receptors": len(receptor_pdbs),
            "boxes": len(boxes),
            "total": total,
            "success": success,
            "failed_or_skipped": total - success,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
