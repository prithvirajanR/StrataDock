import subprocess

from stratadock.core import hardware


def test_detect_hardware_reports_cpu_baseline(monkeypatch):
    monkeypatch.setattr(hardware.platform, "processor", lambda: "AMD Ryzen 9")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(hardware.shutil, "which", lambda _name: None)

    summary = hardware.detect_hardware()

    assert summary.cpu.available is True
    assert summary.cpu.name == "AMD Ryzen 9"
    assert summary.cpu.cores == 16
    assert summary.cpu.architecture == "x86_64"
    assert summary.nvidia.available is False
    assert summary.amd.available is False
    assert summary.recommended_backend == "cpu"
    assert "CPU" in summary.recommendation
    assert summary.as_dict()["recommended_backend"] == "cpu"


def test_detect_hardware_prefers_nvidia_when_nvidia_smi_reports_gpu(monkeypatch):
    calls = []

    def fake_which(name):
        return {"nvidia-smi": "/usr/bin/nvidia-smi"}.get(name)

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        assert command == [
            "/usr/bin/nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
        assert kwargs["timeout"] <= 2.0
        assert kwargs["shell"] is False
        return subprocess.CompletedProcess(command, 0, stdout="NVIDIA RTX 4090, 24564, 550.54\n", stderr="")

    monkeypatch.setattr(hardware.shutil, "which", fake_which)
    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    summary = hardware.detect_hardware()

    assert calls
    assert summary.nvidia.available is True
    assert summary.nvidia.name == "NVIDIA RTX 4090"
    assert summary.nvidia.memory_mb == 24564
    assert summary.nvidia.driver == "550.54"
    assert summary.recommended_backend == "nvidia"
    assert "NVIDIA RTX 4090" in summary.recommendation
    assert "GNINA" in summary.recommendation
    assert "CUDA" in summary.recommendation


def test_detect_hardware_reports_amd_from_rocm_smi(monkeypatch):
    def fake_which(name):
        return {"rocm-smi": "/opt/rocm/bin/rocm-smi"}.get(name)

    def fake_run(command, **_kwargs):
        assert command == ["/opt/rocm/bin/rocm-smi", "--showproductname", "--showmeminfo", "vram"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="GPU[0] : Card series: AMD Radeon RX 7900 XTX\nGPU[0] : VRAM Total Memory (B): 25753026560\n",
            stderr="",
        )

    monkeypatch.setattr(hardware.shutil, "which", fake_which)
    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    summary = hardware.detect_hardware()

    assert summary.amd.available is True
    assert summary.amd.name == "AMD Radeon RX 7900 XTX"
    assert summary.amd.memory_mb == 24560
    assert summary.recommended_backend == "cpu"
    assert "AMD Radeon RX 7900 XTX" in summary.recommendation
    assert "GNINA CPU mode" in summary.recommendation
    assert "CUDA-only" in summary.recommendation


def test_detect_hardware_falls_back_to_rocminfo_or_clinfo_for_amd(monkeypatch):
    commands = []

    def fake_which(name):
        return {"rocminfo": "/opt/rocm/bin/rocminfo", "clinfo": "/usr/bin/clinfo"}.get(name)

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command[0].endswith("rocminfo"):
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="rocminfo failed")
        return subprocess.CompletedProcess(command, 0, stdout="Device Name: gfx1100\nVendor: Advanced Micro Devices\n", stderr="")

    monkeypatch.setattr(hardware.shutil, "which", fake_which)
    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    summary = hardware.detect_hardware()

    assert commands == [["/opt/rocm/bin/rocminfo"], ["/usr/bin/clinfo"]]
    assert summary.amd.available is True
    assert summary.amd.name == "gfx1100"
    assert summary.amd.source == "clinfo"
    assert summary.recommended_backend == "cpu"
    assert "GNINA CPU mode" in summary.recommendation
