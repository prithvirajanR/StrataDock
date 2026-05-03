from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stratadock.core.session import create_session_archive, extract_session_archive, read_session_archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export or inspect a StrataDock run session archive.")
    parser.add_argument("--run-dir", type=Path, help="Run directory to export.")
    parser.add_argument("--out", type=Path, help="Output .zip path.")
    parser.add_argument("--inspect", type=Path, help="Inspect an existing session archive instead of exporting.")
    parser.add_argument("--extract-to", type=Path, help="Extract --inspect archive into this directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inspect:
        if args.extract_to:
            extracted = extract_session_archive(args.inspect, args.extract_to)
            print(json.dumps({"extracted_to": str(extracted), "session": read_session_archive(args.inspect)}, indent=2))
            return
        print(json.dumps(read_session_archive(args.inspect), indent=2))
        return
    if not args.run_dir:
        raise SystemExit("Provide --run-dir to export, or --inspect to inspect an archive.")
    archive = create_session_archive(args.run_dir, args.out)
    print(json.dumps({"session_archive": str(archive)}, indent=2))


if __name__ == "__main__":
    main()
