from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from stratadock.core.models import DockingBox


@dataclass(frozen=True)
class GninaScore:
    mode: int | None = None
    affinity: float | None = None
    cnn_score: float | None = None
    cnn_affinity: float | None = None


@dataclass(frozen=True)
class GninaResult:
    score: float
    scores: list[GninaScore]
    output: Path
    log: str
    command: list[str]


def locate_gnina(*, project_root: Path | None = None, executable: str | Path | None = None) -> Path:
    if executable is not None:
        candidate = Path(executable)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"GNINA executable not found: {candidate}")

    suffix = ".exe" if sys.platform.startswith("win") else ""
    if project_root is not None:
        local_bin = project_root / "tools" / "bin"
        for name in (f"gnina{suffix}", "gnina"):
            candidate = local_bin / name
            if candidate.exists():
                return candidate
        for candidate in sorted(local_bin.glob(f"gnina*{suffix}")):
            if candidate.is_file():
                return candidate

    found = shutil.which("gnina") or shutil.which(f"gnina{suffix}")
    if found:
        return Path(found)
    raise FileNotFoundError("GNINA executable not found. Install gnina or pass executable=.")


def build_gnina_command(
    *,
    executable: str | Path,
    receptor: str | Path,
    ligand: str | Path,
    box: DockingBox,
    output: str | Path,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    energy_range: float | None = None,
    seed: int | None = None,
    device: int | str | None = None,
    cpu_only: bool = False,
) -> list[str]:
    if exhaustiveness < 1:
        raise ValueError("exhaustiveness must be at least 1.")
    if num_modes < 1:
        raise ValueError("num_modes must be at least 1.")

    command = [
        str(executable),
        "--receptor",
        str(receptor),
        "--ligand",
        str(ligand),
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
    ]
    if energy_range is not None and energy_range <= 0:
        raise ValueError("energy_range must be positive.")
    if seed is not None:
        command.extend(["--seed", str(seed)])
    if cpu_only:
        command.append("--no_gpu")
    elif device is not None:
        command.extend(["--device", str(device)])
    command.extend(["--out", str(output)])
    return command


def run_gnina(
    *,
    receptor: str | Path,
    ligand: str | Path,
    box: DockingBox,
    output: str | Path,
    project_root: Path | None = None,
    executable: str | Path | None = None,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    energy_range: float | None = None,
    seed: int | None = None,
    device: int | str | None = None,
    cpu_only: bool = False,
    timeout: float = 600,
) -> GninaResult:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gnina = locate_gnina(project_root=project_root, executable=executable)
    command = build_gnina_command(
        executable=gnina,
        receptor=receptor,
        ligand=ligand,
        box=box,
        output=output_path,
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
        energy_range=energy_range,
        seed=seed,
        device=device,
        cpu_only=cpu_only,
    )
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    log = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        raise RuntimeError(f"GNINA failed with code {result.returncode}.\n{log}")
    if not output_path.exists():
        raise RuntimeError(f"GNINA did not create output file: {output_path}\n{log}")

    output_text = output_path.read_text(errors="ignore")
    scores = parse_gnina_scores(log + "\n" + output_text)
    if not scores:
        raise RuntimeError(f"Could not parse GNINA scores.\n{log}")

    primary_score = _primary_score(scores[0])
    return GninaResult(score=primary_score, scores=scores, output=output_path, log=log, command=command)


def parse_gnina_scores(text: str) -> list[GninaScore]:
    remark_scores = _parse_remark_scores(text)
    if remark_scores:
        return remark_scores
    scores = _parse_sdf_scores(text)
    table_scores = _parse_table_scores(text)
    return table_scores or scores


def _parse_table_scores(text: str) -> list[GninaScore]:
    scores: list[GninaScore] = []
    for line in text.splitlines():
        values = re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?", line)
        if len(values) < 4:
            continue
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            continue
        mode = int(float(values[0]))
        affinity = float(values[1])
        if affinity > 0 and "cnn" not in stripped.lower():
            continue
        scores.append(
            GninaScore(
                mode=mode,
                affinity=affinity,
                cnn_score=float(values[2]),
                cnn_affinity=float(values[3]),
            )
        )
    return scores


def _parse_remark_scores(text: str) -> list[GninaScore]:
    blocks = re.split(r"(?=^\s*MODEL\s+\d+)", text, flags=re.MULTILINE)
    scores: list[GninaScore] = []
    for block in blocks:
        if "REMARK" not in block:
            continue
        mode_match = re.search(r"^\s*MODEL\s+(\d+)", block, re.MULTILINE)
        mode = int(mode_match.group(1)) if mode_match else len(scores) + 1
        affinity = _remark_float(block, "minimizedAffinity")
        if affinity is None:
            affinity = _remark_float(block, "affinity")
        cnn_score = _remark_float(block, "CNNscore")
        cnn_affinity = _remark_float(block, "CNNaffinity")
        if affinity is None and cnn_score is None and cnn_affinity is None:
            continue
        scores.append(GninaScore(mode=mode, affinity=affinity, cnn_score=cnn_score, cnn_affinity=cnn_affinity))
    return scores


def _parse_sdf_scores(text: str) -> list[GninaScore]:
    blocks = text.split("$$$$") if "$$$$" in text else [text]
    scores: list[GninaScore] = []
    for block in blocks:
        affinity = _tag_float(block, "minimizedAffinity")
        if affinity is None:
            affinity = _tag_float(block, "affinity")
        cnn_score = _tag_float(block, "CNNscore")
        cnn_affinity = _tag_float(block, "CNNaffinity")
        if affinity is None and cnn_score is None and cnn_affinity is None:
            continue
        scores.append(GninaScore(mode=len(scores) + 1, affinity=affinity, cnn_score=cnn_score, cnn_affinity=cnn_affinity))
    return scores


def _remark_float(text: str, tag: str) -> float | None:
    match = re.search(rf"^\s*REMARK\s+{re.escape(tag)}\s+([-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?)", text, re.MULTILINE)
    if match:
        return float(match.group(1))
    return None


def _tag_float(text: str, tag: str) -> float | None:
    match = re.search(rf">\s*<{re.escape(tag)}>\s*\n\s*([-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?)", text)
    if match:
        return float(match.group(1))
    return None


def _primary_score(score: GninaScore) -> float:
    if score.affinity is not None:
        return score.affinity
    if score.cnn_affinity is not None:
        return score.cnn_affinity
    if score.cnn_score is not None:
        return score.cnn_score
    raise RuntimeError("GNINA score entry did not contain a numeric score.")
