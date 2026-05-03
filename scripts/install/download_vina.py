from __future__ import annotations

import json
import platform
import stat
import sys
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
BIN_DIR = ROOT / "tools" / "bin"
RELEASE_API = "https://api.github.com/repos/ccsb-scripps/AutoDock-Vina/releases/latest"


def asset_name() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows":
        return "vina_1.2.7_win.exe"
    if system == "linux" and machine in {"x86_64", "amd64"}:
        return "vina_1.2.7_linux_x86_64"
    raise SystemExit(f"No bundled Vina asset rule for {platform.system()} {platform.machine()}")


def main() -> None:
    req = Request(RELEASE_API, headers={"User-Agent": "stratadock-v1.6.01"})
    release = json.loads(urlopen(req, timeout=60).read().decode("utf-8"))
    wanted = asset_name()
    assets = {asset["name"]: asset["browser_download_url"] for asset in release["assets"]}
    if wanted not in assets:
        raise SystemExit(f"Could not find {wanted} in Vina release {release['tag_name']}")

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    out = BIN_DIR / wanted
    print(f"Downloading {wanted} from AutoDock Vina {release['tag_name']}...")
    with urlopen(assets[wanted], timeout=120) as response:
        out.write_bytes(response.read())
    if not platform.system().lower().startswith("win"):
        out.chmod(out.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(out)


if __name__ == "__main__":
    main()
