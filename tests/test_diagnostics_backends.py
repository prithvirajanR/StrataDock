from types import SimpleNamespace

from stratadock.core import diagnostics
from stratadock.core.diagnostics import environment_diagnostics, run_environment_diagnostics


def test_run_environment_diagnostics_reports_required_gnina_and_hardware(tmp_path, monkeypatch):
    hardware_summary = SimpleNamespace(
        recommended_backend="nvidia",
        recommendation="Use NVIDIA CUDA acceleration with RTX 4090 for GNINA when GNINA is installed.",
        as_dict=lambda: {
            "recommended_backend": "nvidia",
            "recommendation": "Use NVIDIA CUDA acceleration with RTX 4090 for GNINA when GNINA is installed.",
            "nvidia": {"available": True, "name": "RTX 4090"},
        },
    )

    monkeypatch.setattr(diagnostics, "vina_binary", lambda _root: tmp_path / "vina")
    monkeypatch.setattr(diagnostics, "script_binary", lambda name: tmp_path / name)
    monkeypatch.setattr(diagnostics, "fpocket_available", lambda: True)
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: "/usr/bin/obabel" if name == "obabel" else None)
    monkeypatch.setattr(diagnostics, "locate_gnina", lambda **_kwargs: (_ for _ in ()).throw(FileNotFoundError("GNINA not installed")))
    monkeypatch.setattr(diagnostics, "detect_hardware", lambda: hardware_summary)

    results = run_environment_diagnostics(project_root=tmp_path, platform_name="linux", wsl_release_text="Linux")

    by_name = {result.name: result for result in results}
    assert by_name["gnina"].status == "missing"
    assert by_name["gnina"].severity == "error"
    assert "required" in by_name["gnina"].message.lower()
    assert by_name["hardware_backend"].status == "nvidia"
    assert by_name["hardware_backend"].severity == "info"
    assert by_name["hardware_backend"].details["recommended_backend"] == "nvidia"


def test_environment_diagnostics_reports_gnina_and_hardware_backend(tmp_path, monkeypatch):
    hardware_summary = SimpleNamespace(
        recommended_backend="cpu",
        recommendation="Use the CPU baseline.",
        as_dict=lambda: {"recommended_backend": "cpu", "recommendation": "Use the CPU baseline."},
    )

    monkeypatch.setattr(diagnostics, "vina_binary", lambda _root: tmp_path / "vina")
    monkeypatch.setattr(diagnostics, "script_binary", lambda name: tmp_path / name)
    monkeypatch.setattr(diagnostics, "locate_gnina", lambda **_kwargs: tmp_path / "gnina")
    monkeypatch.setattr(diagnostics, "detect_hardware", lambda: hardware_summary)
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: f"/usr/bin/{name}" if name in {"obabel", "fpocket"} else None)
    monkeypatch.setattr(diagnostics.platform, "system", lambda: "Linux")

    report = environment_diagnostics(project_root=tmp_path)

    by_key = {item["key"]: item for item in report["checks"]}
    assert by_key["gnina"]["status"] == "ok"
    assert by_key["gnina"]["detail"] == str(tmp_path / "gnina")
    assert by_key["hardware_backend"]["status"] == "ok"
    assert by_key["hardware_backend"]["detail"] == "Use the CPU baseline."


def test_environment_diagnostics_treats_missing_gnina_as_error(tmp_path, monkeypatch):
    hardware_summary = SimpleNamespace(
        recommended_backend="cpu",
        recommendation="Use the CPU baseline.",
        as_dict=lambda: {"recommended_backend": "cpu", "recommendation": "Use the CPU baseline."},
    )

    monkeypatch.setattr(diagnostics, "vina_binary", lambda _root: tmp_path / "vina")
    monkeypatch.setattr(diagnostics, "script_binary", lambda name: tmp_path / name)
    monkeypatch.setattr(diagnostics, "locate_gnina", lambda **_kwargs: (_ for _ in ()).throw(FileNotFoundError("GNINA not installed")))
    monkeypatch.setattr(diagnostics, "detect_hardware", lambda: hardware_summary)
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: f"/usr/bin/{name}" if name in {"obabel", "fpocket"} else None)
    monkeypatch.setattr(diagnostics.platform, "system", lambda: "Linux")

    report = environment_diagnostics(project_root=tmp_path)

    by_key = {item["key"]: item for item in report["checks"]}
    assert by_key["gnina"]["status"] == "error"
    assert "Required" in by_key["gnina"]["detail"]
    assert report["summary"]["error"] == 1
