from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
BIN_DIR = ROOT / "tools" / "bin"
CACHE_DIR = ROOT / "tools" / "cache"
GNINA_VERSION = "v1.0.3"
GNINA_URL = "https://github.com/gnina/gnina/releases/download/v1.0.3/gnina"
GNINA_SHA256 = "03d345ff55d2d26460cd431bc80b4341589d6408c8ecfa25a20bc983b41118a4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "stratadock-v1.6.01"})
    with urlopen(req, timeout=300) as response:
        with output.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)


def verify_executable(path: Path) -> None:
    result = subprocess.run([str(path), "--help"], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise SystemExit(f"GNINA did not start cleanly.\n{result.stdout}\n{result.stderr}")
    if "receptor" not in (result.stdout + result.stderr).lower():
        raise SystemExit("GNINA help output did not look valid.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the tested StrataDock GNINA Linux binary.")
    parser.add_argument("--force", action="store_true", help="Redownload even if the cached asset exists.")
    args = parser.parse_args()

    if not sys.platform.startswith("linux"):
        raise SystemExit("This GNINA installer is for Linux/WSL x86_64.")
    if os.uname().machine not in {"x86_64", "amd64"}:
        raise SystemExit(f"Unsupported architecture for bundled GNINA: {os.uname().machine}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"gnina-{GNINA_VERSION}"
    target = BIN_DIR / "gnina"

    if args.force or not cache_path.exists():
        print(f"Downloading GNINA {GNINA_VERSION} from official gnina/gnina release...")
        download(GNINA_URL, cache_path)

    actual = sha256(cache_path)
    if actual != GNINA_SHA256:
        raise SystemExit(f"GNINA checksum mismatch.\nExpected: {GNINA_SHA256}\nActual:   {actual}")

    shutil.copy2(cache_path, target)
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    verify_executable(target)
    print(target)


if __name__ == "__main__":
    main()
