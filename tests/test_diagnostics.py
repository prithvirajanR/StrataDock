from pathlib import Path

import pytest

pytest.importorskip("rdkit")

from stratadock.core import diagnostics
from stratadock.core.diagnostics import environment_diagnostics, run_environment_diagnostics


def test_environment_diagnostics_reports_tools_and_path_warning(monkeypatch):
    monkeypatch.setattr("stratadock.core.diagnostics.shutil.which", lambda name: f"/usr/bin/{name}" if name == "obabel" else None)
    monkeypatch.setattr("stratadock.core.diagnostics.script_binary", lambda name: f"/fake/{name}")
    monkeypatch.setattr("stratadock.core.diagnostics.vina_binary", lambda root: root / "tools" / "vina")
    monkeypatch.setattr("stratadock.core.diagnostics.platform.system", lambda: "Linux")

    report = environment_diagnostics(project_root="/mnt/f/StrataDock/stratadock_v2")

    by_key = {item["key"]: item for item in report["checks"]}
    assert by_key["meeko_ligand"]["status"] == "ok"
    assert by_key["openbabel"]["status"] == "ok"
    assert by_key["wsl_mount"]["status"] == "warn"
    assert report["summary"]["warn"] >= 1


def test_run_environment_diagnostics_returns_structured_tool_statuses(tmp_path, monkeypatch):
    def fake_which(name):
        return {
            "obabel": "/usr/bin/obabel",
            "wsl": "/usr/bin/wsl",
        }.get(name)

    def fake_vina_binary(_project_root):
        raise FileNotFoundError("Vina binary not found")

    def fake_script_binary(name):
        if name == "mk_prepare_ligand":
            raise FileNotFoundError("missing ligand script")
        return Path(f"/venv/bin/{name}")

    monkeypatch.setattr(diagnostics.shutil, "which", fake_which)
    monkeypatch.setattr(diagnostics, "vina_binary", fake_vina_binary)
    monkeypatch.setattr(diagnostics, "script_binary", fake_script_binary)
    monkeypatch.setattr(diagnostics, "fpocket_available", lambda: False)

    results = run_environment_diagnostics(
        project_root=tmp_path,
        working_path=Path("/mnt/c/users/project"),
        platform_name="linux",
        wsl_release_text="Ubuntu 22.04 WSL2",
    )

    by_name = {result.name: result for result in results}
    assert by_name["vina"].severity == "error"
    assert by_name["meeko_mk_prepare_ligand"].severity == "error"
    assert by_name["meeko_mk_prepare_receptor"].severity == "info"
    assert by_name["openbabel"].severity == "info"
    assert by_name["fpocket"].severity == "warning"
    assert by_name["wsl_linux"].severity == "info"
    assert by_name["mnt_path"].severity == "warning"
    assert by_name["vina"].as_dict()["severity"] == "error"
