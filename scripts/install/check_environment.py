from __future__ import annotations

import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stratadock.tools.binaries import script_binary, vina_binary
from stratadock.core.gnina import locate_gnina


def module_status(name: str) -> dict[str, str | bool]:
    try:
        module = importlib.import_module(name)
        return {"ok": True, "version": str(getattr(module, "__version__", "unknown"))}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> None:
    status: dict[str, object] = {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "modules": {
            name: module_status(name)
            for name in ["rdkit", "meeko", "gemmi", "scipy", "numpy", "pandas"]
        },
        "executables": {},
    }
    for name in ["mk_prepare_ligand", "mk_prepare_receptor"]:
        try:
            status["executables"][name] = str(script_binary(name))
        except Exception as exc:
            status["executables"][name] = {"error": str(exc)}
    try:
        status["executables"]["vina"] = str(vina_binary(ROOT))
    except Exception as exc:
        status["executables"]["vina"] = {"error": str(exc)}
    try:
        gnina = locate_gnina(project_root=ROOT)
        result = subprocess.run([str(gnina), "--help"], capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError((result.stdout + "\n" + result.stderr).strip())
        status["executables"]["gnina"] = str(gnina)
    except Exception as exc:
        status["executables"]["gnina"] = {"error": str(exc)}

    print(json.dumps(status, indent=2))
    failures = [
        key for key, value in status["modules"].items()
        if isinstance(value, dict) and not value.get("ok")
    ]
    exe_failures = [
        key for key, value in status["executables"].items()
        if isinstance(value, dict) and value.get("error")
    ]
    if failures or exe_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
