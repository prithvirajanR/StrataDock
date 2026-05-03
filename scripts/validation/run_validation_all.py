from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stratadock.core.reports import write_validation_reports


def main() -> None:
    index = json.loads((ROOT / "data" / "validation" / "index.json").read_text())
    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for case in index["cases"]:
        case_id = case["case_id"]
        print(f"=== {case_id} ===", flush=True)
        run = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("run_validation_case.py")), case_id],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if run.stdout:
            print(run.stdout.strip())
        if run.stderr:
            print(run.stderr.strip(), file=sys.stderr)

        result_path = ROOT / "runs" / case_id / "result.json"
        if run.returncode != 0 or not result_path.exists():
            failures.append(
                {
                    "case_id": case_id,
                    "returncode": run.returncode,
                    "stdout": run.stdout,
                    "stderr": run.stderr,
                }
            )
            continue
        result = json.loads(result_path.read_text())
        result["expected_rmsd_max"] = case["acceptance"]["pose_rmsd_angstrom_max"]
        results.append(result)

    summary = {
        "total": len(index["cases"]),
        "passed": sum(1 for result in results if result.get("passes_rmsd_threshold")),
        "failed": len(failures) + sum(1 for result in results if not result.get("passes_rmsd_threshold")),
        "results": results,
        "failures": failures,
    }
    write_validation_reports(summary, ROOT / "runs")
    print(json.dumps(summary, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
