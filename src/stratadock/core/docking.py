from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from stratadock.core.gnina import GninaScore, parse_gnina_scores, run_gnina
from stratadock.core.models import DockingBox
from stratadock.tools.binaries import vina_binary


@dataclass(frozen=True)
class DockingResult:
    score: float
    scores: list[float]
    pose_pdbqt: Path
    log: str
    engine: str = "vina"
    cnn_scores: list[float | None] | None = None
    cnn_affinities: list[float | None] | None = None
    command: list[str] | None = None


def run_vina(
    *,
    project_root: Path,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    box: DockingBox,
    output_pdbqt: Path,
    exhaustiveness: int = 4,
    seed: int = 1,
    num_modes: int = 1,
    energy_range: float = 3.0,
    scoring: str = "vina",
) -> DockingResult:
    if num_modes < 1:
        raise ValueError("num_modes must be at least 1.")
    if energy_range <= 0:
        raise ValueError("energy_range must be positive.")
    if scoring not in {"vina", "vinardo"}:
        raise ValueError("scoring must be 'vina' or 'vinardo'.")
    output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    vina = vina_binary(project_root)
    cmd = [
        str(vina),
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--center_x",
        str(box.center_x),
        "--center_y",
        str(box.center_y),
        "--center_z",
        str(box.center_z),
        "--size_x",
        str(box.size_x),
        "--size_y",
        str(box.size_y),
        "--size_z",
        str(box.size_z),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        str(num_modes),
        "--energy_range",
        str(energy_range),
        "--scoring",
        scoring,
        "--seed",
        str(seed),
        "--out",
        str(output_pdbqt),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    log = result.stdout + "\n" + result.stderr
    if result.returncode != 0 or not output_pdbqt.exists():
        raise RuntimeError(f"Vina failed with code {result.returncode}.\n{log}")

    text = output_pdbqt.read_text(errors="ignore")
    scores = parse_vina_scores_from_pose(output_pdbqt)
    if not scores:
        log_match = re.search(r"^\s*1\s+([-\d.]+)\s+", log, re.MULTILINE)
        if log_match:
            scores = [float(log_match.group(1))]
    if not scores:
        raise RuntimeError(f"Could not parse Vina score.\n{log}")

    return DockingResult(score=scores[0], scores=scores, pose_pdbqt=output_pdbqt, log=log, engine="vina", command=cmd)


def parse_vina_scores_from_pose(pose_pdbqt: Path) -> list[float]:
    text = pose_pdbqt.read_text(errors="ignore")
    return [float(value) for value in re.findall(r"REMARK VINA RESULT:\s+([-\d.]+)", text)]


def run_docking(
    *,
    engine: str = "vina",
    project_root: Path,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    box: DockingBox,
    output_pdbqt: Path,
    exhaustiveness: int = 4,
    seed: int = 1,
    num_modes: int = 1,
    energy_range: float = 3.0,
    scoring: str = "vina",
    gnina_cpu_only: bool = False,
    gnina_device: int | str | None = None,
) -> DockingResult:
    engine = engine.lower()
    if engine == "vina":
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
    if engine != "gnina":
        raise ValueError("engine must be 'vina' or 'gnina'.")

    result = run_gnina(
        project_root=project_root,
        receptor=receptor_pdbqt,
        ligand=ligand_pdbqt,
        box=box,
        output=output_pdbqt,
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
        energy_range=energy_range,
        seed=seed,
        device=gnina_device,
        cpu_only=gnina_cpu_only,
    )
    return _gnina_result_to_docking_result(result.scores, result.output, result.log, result.command)


def parse_gnina_scores_from_pose(pose_pdbqt: Path) -> list[GninaScore]:
    return parse_gnina_scores(pose_pdbqt.read_text(errors="ignore"))


def _gnina_result_to_docking_result(
    scores: list[GninaScore],
    pose_pdbqt: Path,
    log: str,
    command: list[str] | None = None,
) -> DockingResult:
    affinities = [_gnina_primary_score(score) for score in scores]
    return DockingResult(
        score=affinities[0],
        scores=affinities,
        pose_pdbqt=pose_pdbqt,
        log=log,
        engine="gnina",
        cnn_scores=[score.cnn_score for score in scores],
        cnn_affinities=[score.cnn_affinity for score in scores],
        command=command,
    )


def _gnina_primary_score(score: GninaScore) -> float:
    if score.affinity is not None:
        return score.affinity
    if score.cnn_affinity is not None:
        return score.cnn_affinity
    if score.cnn_score is not None:
        return score.cnn_score
    raise RuntimeError("GNINA score entry did not contain a numeric score.")
