from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


DEFAULT_ESMFOLD_ENDPOINT = "https://api.esmatlas.com/foldSequence/v1/pdb/"
CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class RemotePredictionResponse:
    status_code: int
    text: str


@dataclass(frozen=True)
class PredictionConfig:
    endpoint_url: str | None = field(
        default_factory=lambda: os.environ.get("STRATADOCK_ESMFOLD_ENDPOINT", DEFAULT_ESMFOLD_ENDPOINT)
    )
    timeout_seconds: int = 120
    allow_network: bool = False
    reuse_cache: bool = True
    report_path: Path | None = None


@dataclass(frozen=True)
class PredictionResult:
    ok: bool
    status: str
    output_path: Path
    report_path: Path
    sequence_length: int
    cached: bool
    mean_plddt: float | None
    endpoint_url: str | None
    error: str | None
    report: dict[str, object]


PostFn = Callable[[str, str, int], RemotePredictionResponse]


def validate_amino_acid_sequence(sequence_text: str) -> str:
    """Normalize FASTA/plain protein text and require canonical residues."""
    sequence_lines = [
        line.strip()
        for line in sequence_text.splitlines()
        if line.strip() and not line.lstrip().startswith(">")
    ]
    sequence = re.sub(r"\s+", "", "".join(sequence_lines)).upper()
    if not sequence:
        raise ValueError("Protein sequence is empty.")

    invalid = sorted(set(sequence) - CANONICAL_AMINO_ACIDS)
    if invalid:
        bad = ", ".join(invalid)
        raise ValueError(f"Protein sequence contains non-canonical amino acid residue(s): {bad}.")
    return sequence


def predict_structure_from_sequence(
    sequence_text: str,
    output_path: Path | str,
    *,
    config: PredictionConfig | None = None,
    post_fn: PostFn | None = None,
) -> PredictionResult:
    cfg = config or PredictionConfig()
    out_path = Path(output_path)
    report_path = cfg.report_path or out_path.with_suffix(".json")

    try:
        sequence = validate_amino_acid_sequence(sequence_text)
    except ValueError as exc:
        return _result(
            ok=False,
            status="invalid_sequence",
            output_path=out_path,
            report_path=report_path,
            sequence_length=0,
            cached=False,
            mean_plddt=None,
            endpoint_url=cfg.endpoint_url,
            error=str(exc),
        )

    if cfg.reuse_cache and _looks_like_pdb(out_path):
        pdb_text = out_path.read_text(encoding="utf-8", errors="replace")
        return _result(
            ok=True,
            status="cached",
            output_path=out_path,
            report_path=report_path,
            sequence_length=len(sequence),
            cached=True,
            mean_plddt=_mean_plddt_from_pdb(pdb_text),
            endpoint_url=cfg.endpoint_url,
            error=None,
        )

    if not cfg.allow_network:
        return _result(
            ok=False,
            status="network_disabled",
            output_path=out_path,
            report_path=report_path,
            sequence_length=len(sequence),
            cached=False,
            mean_plddt=None,
            endpoint_url=cfg.endpoint_url,
            error="Network access is disabled; enable PredictionConfig.allow_network to call a folding endpoint.",
        )

    if not cfg.endpoint_url:
        return _result(
            ok=False,
            status="endpoint_missing",
            output_path=out_path,
            report_path=report_path,
            sequence_length=len(sequence),
            cached=False,
            mean_plddt=None,
            endpoint_url=None,
            error="No folding endpoint URL is configured.",
        )

    try:
        poster = post_fn or _post_sequence_to_endpoint
        response = poster(sequence, cfg.endpoint_url, cfg.timeout_seconds)
    except Exception as exc:
        return _result(
            ok=False,
            status="remote_error",
            output_path=out_path,
            report_path=report_path,
            sequence_length=len(sequence),
            cached=False,
            mean_plddt=None,
            endpoint_url=cfg.endpoint_url,
            error=f"Folding endpoint request failed: {exc}",
        )

    if response.status_code != 200:
        detail = response.text[:200].strip()
        message = f"Folding endpoint returned HTTP {response.status_code}"
        if detail:
            message = f"{message}: {detail}"
        return _result(
            ok=False,
            status="remote_error",
            output_path=out_path,
            report_path=report_path,
            sequence_length=len(sequence),
            cached=False,
            mean_plddt=None,
            endpoint_url=cfg.endpoint_url,
            error=message,
        )

    if not _looks_like_pdb_text(response.text):
        return _result(
            ok=False,
            status="invalid_pdb",
            output_path=out_path,
            report_path=report_path,
            sequence_length=len(sequence),
            cached=False,
            mean_plddt=None,
            endpoint_url=cfg.endpoint_url,
            error="Folding endpoint response did not look like a PDB structure.",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(response.text, encoding="utf-8")
    return _result(
        ok=True,
        status="success",
        output_path=out_path,
        report_path=report_path,
        sequence_length=len(sequence),
        cached=False,
        mean_plddt=_mean_plddt_from_pdb(response.text),
        endpoint_url=cfg.endpoint_url,
        error=None,
    )


def _post_sequence_to_endpoint(sequence: str, endpoint_url: str, timeout_seconds: int) -> RemotePredictionResponse:
    request = urllib.request.Request(
        endpoint_url,
        data=sequence.encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            text = response.read().decode("utf-8", errors="replace")
            status_code = getattr(response, "status", response.getcode())
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return RemotePredictionResponse(status_code=exc.code, text=text)
    return RemotePredictionResponse(status_code=int(status_code), text=text)


def _result(
    *,
    ok: bool,
    status: str,
    output_path: Path,
    report_path: Path,
    sequence_length: int,
    cached: bool,
    mean_plddt: float | None,
    endpoint_url: str | None,
    error: str | None,
) -> PredictionResult:
    report = {
        "schema_version": 1,
        "ok": ok,
        "status": status,
        "model": "esmfold",
        "sequence_length": sequence_length,
        "output_path": str(output_path),
        "report_path": str(report_path),
        "cached": cached,
        "endpoint_url": endpoint_url,
        "mean_plddt": mean_plddt,
        "confidence": _confidence_label(mean_plddt),
        "error": error,
    }
    _write_report(report_path, report)
    return PredictionResult(
        ok=ok,
        status=status,
        output_path=output_path,
        report_path=report_path,
        sequence_length=sequence_length,
        cached=cached,
        mean_plddt=mean_plddt,
        endpoint_url=endpoint_url,
        error=error,
        report=report,
    )


def _write_report(report_path: Path, report: dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _looks_like_pdb(path: Path) -> bool:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        return _looks_like_pdb_text(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return False


def _looks_like_pdb_text(text: str) -> bool:
    return any(line.startswith(("ATOM", "HETATM")) for line in text.splitlines())


def _mean_plddt_from_pdb(pdb_text: str) -> float | None:
    values: list[float] = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 66:
            try:
                values.append(float(line[60:66].strip()))
            except ValueError:
                continue
    if not values:
        return None
    if max(values) <= 1.0:
        values = [value * 100 for value in values]
    return round(sum(values) / len(values), 1)


def _confidence_label(mean_plddt: float | None) -> str | None:
    if mean_plddt is None:
        return None
    if mean_plddt >= 70:
        return "high"
    if mean_plddt >= 50:
        return "medium"
    return "low"
