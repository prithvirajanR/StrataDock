from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass


PROBE_TIMEOUT_SECONDS = 1.5


@dataclass(frozen=True)
class HardwareDevice:
    kind: str
    available: bool
    name: str | None = None
    cores: int | None = None
    architecture: str | None = None
    memory_mb: int | None = None
    driver: str | None = None
    source: str | None = None
    message: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "available": self.available,
            "name": self.name,
            "cores": self.cores,
            "architecture": self.architecture,
            "memory_mb": self.memory_mb,
            "driver": self.driver,
            "source": self.source,
            "message": self.message,
        }


@dataclass(frozen=True)
class HardwareSummary:
    cpu: HardwareDevice
    nvidia: HardwareDevice
    amd: HardwareDevice
    recommended_backend: str
    recommendation: str

    def as_dict(self) -> dict[str, object]:
        return {
            "cpu": self.cpu.as_dict(),
            "nvidia": self.nvidia.as_dict(),
            "amd": self.amd.as_dict(),
            "recommended_backend": self.recommended_backend,
            "recommendation": self.recommendation,
        }


def detect_hardware() -> HardwareSummary:
    cpu = _detect_cpu()
    nvidia = _detect_nvidia()
    amd = _detect_amd()
    backend, recommendation = _recommend_backend(cpu=cpu, nvidia=nvidia, amd=amd)
    return HardwareSummary(
        cpu=cpu,
        nvidia=nvidia,
        amd=amd,
        recommended_backend=backend,
        recommendation=recommendation,
    )


def _detect_cpu() -> HardwareDevice:
    architecture = platform.machine() or None
    name = platform.processor() or architecture or "CPU"
    return HardwareDevice(
        kind="cpu",
        available=True,
        name=name,
        cores=os.cpu_count(),
        architecture=architecture,
        source="platform",
        message="CPU baseline is available.",
    )


def _detect_nvidia() -> HardwareDevice:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return HardwareDevice(kind="nvidia", available=False, message="nvidia-smi was not found.")

    command = [
        nvidia_smi,
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = _run_probe(command)
    if result.returncode != 0 or not result.stdout.strip():
        return HardwareDevice(
            kind="nvidia",
            available=False,
            source="nvidia-smi",
            message=(result.stderr or "nvidia-smi did not report a GPU.").strip(),
        )

    name, memory_mb, driver = _parse_nvidia_smi(result.stdout)
    return HardwareDevice(
        kind="nvidia",
        available=True,
        name=name,
        memory_mb=memory_mb,
        driver=driver,
        source="nvidia-smi",
        message="NVIDIA GPU detected.",
    )


def _detect_amd() -> HardwareDevice:
    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi:
        result = _run_probe([rocm_smi, "--showproductname", "--showmeminfo", "vram"])
        if result.returncode == 0 and result.stdout.strip():
            name, memory_mb = _parse_rocm_smi(result.stdout)
            if name or memory_mb is not None:
                return HardwareDevice(
                    kind="amd",
                    available=True,
                    name=name,
                    memory_mb=memory_mb,
                    source="rocm-smi",
                    message="AMD ROCm GPU detected.",
                )

    rocminfo = shutil.which("rocminfo")
    if rocminfo:
        result = _run_probe([rocminfo])
        if result.returncode == 0 and _looks_like_amd(result.stdout):
            return HardwareDevice(
                kind="amd",
                available=True,
                name=_parse_device_name(result.stdout) or "AMD GPU",
                source="rocminfo",
                message="AMD ROCm device detected.",
            )

    clinfo = shutil.which("clinfo")
    if clinfo:
        result = _run_probe([clinfo])
        if result.returncode == 0 and _looks_like_amd(result.stdout):
            return HardwareDevice(
                kind="amd",
                available=True,
                name=_parse_device_name(result.stdout) or "AMD GPU",
                source="clinfo",
                message="AMD OpenCL device detected.",
            )

    return HardwareDevice(kind="amd", available=False, message="No AMD ROCm/OpenCL GPU probe reported a device.")


def _run_probe(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            shell=False,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(command, 1, stdout="", stderr=str(exc))


def _parse_nvidia_smi(output: str) -> tuple[str | None, int | None, str | None]:
    first_line = output.strip().splitlines()[0]
    parts = [part.strip() for part in first_line.split(",")]
    name = parts[0] if parts else None
    memory_mb = _parse_int(parts[1]) if len(parts) > 1 else None
    driver = parts[2] if len(parts) > 2 else None
    return name, memory_mb, driver


def _parse_rocm_smi(output: str) -> tuple[str | None, int | None]:
    name = None
    memory_mb = None
    for line in output.splitlines():
        if "Card series:" in line:
            name = line.split("Card series:", 1)[1].strip()
        if "VRAM Total Memory" in line:
            match = re.search(r"(\d+)\s*$", line)
            if match:
                memory_mb = int(match.group(1)) // (1024 * 1024)
    return name, memory_mb


def _looks_like_amd(output: str) -> bool:
    text = output.lower()
    return "advanced micro devices" in text or "amd" in text or "gfx" in text


def _parse_device_name(output: str) -> str | None:
    for line in output.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("device name"):
            return stripped.split(":", 1)[1].strip()
        if "marketing name:" in lower or "name:" in lower:
            return stripped.split(":", 1)[1].strip()
        if "gfx" in lower:
            return stripped
    return None


def _parse_int(value: str) -> int | None:
    try:
        return int(value.strip())
    except ValueError:
        return None


def _recommend_backend(*, cpu: HardwareDevice, nvidia: HardwareDevice, amd: HardwareDevice) -> tuple[str, str]:
    if nvidia.available:
        label = nvidia.name or "NVIDIA GPU"
        return "nvidia", f"Use NVIDIA CUDA acceleration with {label} for GNINA when GNINA is installed."
    if amd.available:
        label = amd.name or "AMD GPU"
        return "cpu", f"{label} detected, but official GNINA GPU acceleration is NVIDIA CUDA-only. Use Vina or GNINA CPU mode."
    label = cpu.name or "CPU"
    cores = f" ({cpu.cores} cores)" if cpu.cores else ""
    return "cpu", f"Use the CPU baseline with {label}{cores}; no NVIDIA CUDA GPU was detected."
