from __future__ import annotations

import shutil
import sys
from pathlib import Path


def script_binary(name: str) -> Path:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    script_dir = Path(sys.executable).parent
    for candidate_name in [name + suffix, name + ".py"]:
        candidate = script_dir / candidate_name
        if candidate.exists():
            return candidate

    found = shutil.which(name) or shutil.which(name + suffix) or shutil.which(name + ".py")
    if found:
        return Path(found)
    raise FileNotFoundError(f"Could not find executable: {name}")


def vina_binary(project_root: Path) -> Path:
    exe = "vina_1.2.7_win.exe" if sys.platform.startswith("win") else "vina_1.2.7_linux_x86_64"
    local = project_root / "tools" / "bin" / exe
    if local.exists():
        return local

    found = shutil.which("vina")
    if found:
        return Path(found)
    raise FileNotFoundError(
        f"Vina binary not found. Expected {local}. Run scripts/install/download_vina.py."
    )
