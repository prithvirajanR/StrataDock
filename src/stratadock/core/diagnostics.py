from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from stratadock.core.gnina import locate_gnina
from stratadock.core.hardware import detect_hardware
from stratadock.tools.binaries import script_binary, vina_binary


@dataclass(frozen=True)
class DiagnosticResult:
    name: str
    status: str
    severity: str
    message: str
    details: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
        }


def run_environment_diagnostics(
    *,
    project_root: Path,
    working_path: Path | None = None,
    platform_name: str | None = None,
    wsl_release_text: str | None = None,
) -> list[DiagnosticResult]:
    platform_name = platform_name or sys.platform
    working_path = working_path or Path.cwd()
    return [
        _check_vina(project_root),
        _check_gnina(project_root),
        _check_meeko_script("mk_prepare_ligand"),
        _check_meeko_script("mk_prepare_receptor"),
        _check_openbabel(),
        _check_fpocket(),
        _check_hardware_backend(),
        _check_wsl_linux(platform_name=platform_name, wsl_release_text=wsl_release_text),
        _check_mnt_path(working_path=working_path, platform_name=platform_name),
    ]


def environment_diagnostics(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root)
    checks: list[dict[str, str]] = []

    def add(key: str, label: str, status: str, detail: str) -> None:
        checks.append({"key": key, "label": label, "status": status, "detail": detail})

    for key, binary in [("vina", "vina"), ("meeko_ligand", "mk_prepare_ligand"), ("meeko_receptor", "mk_prepare_receptor")]:
        try:
            path = vina_binary(root) if binary == "vina" else script_binary(binary)
            add(key, binary, "ok", str(path))
        except Exception as exc:
            add(key, binary, "error", str(exc))

    try:
        gnina = locate_gnina(project_root=root)
        add("gnina", "GNINA", "ok", str(gnina))
    except FileNotFoundError as exc:
        add("gnina", "GNINA", "error", f"Required; {exc}")

    hardware_summary = detect_hardware()
    add("hardware_backend", "Hardware backend", "ok", hardware_summary.recommendation)

    obabel = shutil.which("obabel")
    add("openbabel", "OpenBabel", "ok" if obabel else "warn", obabel or "Optional; needed for pH/protonation steps.")

    fpocket = shutil.which("fpocket")
    bundled_fpocket = root / "tools" / "fpocket_src" / "bin" / "fpocket"
    if fpocket:
        add("fpocket", "fpocket", "ok", fpocket)
    elif bundled_fpocket.exists():
        add("fpocket", "fpocket", "ok", str(bundled_fpocket))
    else:
        add("fpocket", "fpocket", "error", "Missing; automatic pocket detection will fail.")

    system = platform.system()
    add("platform", "Platform", "ok" if system in {"Linux", "Windows"} else "warn", system)
    root_text = str(root).replace("\\", "/")
    add(
        "wsl_mount",
        "WSL mount path",
        "warn" if root_text.startswith("/mnt/") else "ok",
        "Running from /mnt can be slower than Linux-native ~/ paths." if root_text.startswith("/mnt/") else root_text,
    )

    summary = {
        "ok": sum(item["status"] == "ok" for item in checks),
        "warn": sum(item["status"] == "warn" for item in checks),
        "error": sum(item["status"] == "error" for item in checks),
    }
    return {"checks": checks, "summary": summary}


def _check_vina(project_root: Path) -> DiagnosticResult:
    try:
        path = vina_binary(project_root)
    except FileNotFoundError as exc:
        return DiagnosticResult(
            name="vina",
            status="missing",
            severity="error",
            message=str(exc),
        )
    return DiagnosticResult(
        name="vina",
        status="available",
        severity="info",
        message="AutoDock Vina is available.",
        details={"path": str(path)},
    )


def _check_gnina(project_root: Path) -> DiagnosticResult:
    try:
        path = locate_gnina(project_root=project_root)
    except FileNotFoundError as exc:
        return DiagnosticResult(
            name="gnina",
            status="missing",
            severity="error",
            message=f"GNINA is required; {exc}",
        )
    return DiagnosticResult(
        name="gnina",
        status="available",
        severity="info",
        message="GNINA is available.",
        details={"path": str(path)},
    )


def _check_meeko_script(script_name: str) -> DiagnosticResult:
    name = f"meeko_{script_name}"
    try:
        path = script_binary(script_name)
    except FileNotFoundError as exc:
        return DiagnosticResult(
            name=name,
            status="missing",
            severity="error",
            message=str(exc),
        )
    return DiagnosticResult(
        name=name,
        status="available",
        severity="info",
        message=f"Meeko script {script_name} is available.",
        details={"path": str(path)},
    )


def _check_openbabel() -> DiagnosticResult:
    path = shutil.which("obabel")
    if not path:
        return DiagnosticResult(
            name="openbabel",
            status="missing",
            severity="warning",
            message="OpenBabel 'obabel' was not found; receptor hydrogen addition will be unavailable.",
        )
    return DiagnosticResult(
        name="openbabel",
        status="available",
        severity="info",
        message="OpenBabel is available.",
        details={"path": path},
    )


def _check_fpocket() -> DiagnosticResult:
    if fpocket_available():
        return DiagnosticResult(
            name="fpocket",
            status="available",
            severity="info",
            message="fpocket is available.",
        )
    return DiagnosticResult(
        name="fpocket",
        status="missing",
        severity="warning",
        message="fpocket was not found; automatic pocket detection will be unavailable.",
    )


def _check_hardware_backend() -> DiagnosticResult:
    summary = detect_hardware()
    return DiagnosticResult(
        name="hardware_backend",
        status=summary.recommended_backend,
        severity="info",
        message=summary.recommendation,
        details=summary.as_dict(),
    )


def _check_wsl_linux(*, platform_name: str, wsl_release_text: str | None) -> DiagnosticResult:
    if platform_name.startswith("win"):
        wsl = shutil.which("wsl")
        if wsl:
            return DiagnosticResult(
                name="wsl_linux",
                status="available",
                severity="info",
                message="WSL is available on Windows.",
                details={"path": wsl},
            )
        return DiagnosticResult(
            name="wsl_linux",
            status="missing",
            severity="warning",
            message="WSL was not found; bundled Linux tools cannot be launched from Windows.",
        )

    release = wsl_release_text if wsl_release_text is not None else platform.release()
    is_wsl = "microsoft" in release.lower() or "wsl" in release.lower()
    return DiagnosticResult(
        name="wsl_linux",
        status="wsl" if is_wsl else "linux",
        severity="info",
        message="Running inside WSL/Linux." if is_wsl else "Running on Linux.",
        details={"release": release},
    )


def fpocket_available() -> bool:
    from stratadock.core.pockets import fpocket_available as check_fpocket_available

    return check_fpocket_available()


def _check_mnt_path(*, working_path: Path, platform_name: str) -> DiagnosticResult:
    text = str(working_path).replace("\\", "/")
    if platform_name.startswith("linux") and text.startswith("/mnt/"):
        return DiagnosticResult(
            name="mnt_path",
            status="mounted_windows_path",
            severity="warning",
            message="The working path is under /mnt; docking workloads may be faster in the Linux filesystem.",
            details={"path": text},
        )
    return DiagnosticResult(
        name="mnt_path",
        status="ok",
        severity="info",
        message="The working path does not look like a mounted Windows path.",
        details={"path": text},
    )
