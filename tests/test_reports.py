import csv
import json

from stratadock.core.reports import write_run_html_report, write_run_pdf_report, write_validation_reports


def test_write_validation_reports_creates_json_csv_and_txt(tmp_path):
    summary = {
        "total": 1,
        "passed": 1,
        "failed": 0,
        "results": [
            {
                "case_id": "demo_case",
                "score": -7.5,
                "heavy_atom_rmsd": 0.8,
                "expected_rmsd_max": 2.5,
                "passes_rmsd_threshold": True,
                "pose_pdbqt": "runs/demo/pose.pdbqt",
                "native_ligand_pdb": "data/demo/native_ligand.pdb",
                "receptor_report_json": "runs/demo/receptor/receptor_report.json",
            }
        ],
        "failures": [],
    }

    paths = write_validation_reports(summary, tmp_path)

    assert paths["json"].exists()
    assert paths["csv"].exists()
    assert paths["txt"].exists()

    json_report = json.loads(paths["json"].read_text())
    assert json_report["passed"] == 1
    assert "validation_report_csv" in json_report

    rows = list(csv.DictReader(paths["csv"].open(newline="", encoding="utf-8")))
    assert rows[0]["case_id"] == "demo_case"
    assert rows[0]["passes_rmsd_threshold"] == "True"

    text = paths["txt"].read_text()
    assert "Expected vs actual" in text
    assert "demo_case" in text


def test_write_run_html_report_contains_top_results(tmp_path):
    manifest = tmp_path / "manifest.json"
    summary = tmp_path / "summary.txt"
    manifest.write_text("{}", encoding="utf-8")
    summary.write_text("summary", encoding="utf-8")

    report = write_run_html_report(
        title="Demo Report",
        results=[
            {
                "receptor_name": "rec",
                "pocket_name": "pocket_1",
                "ligand_name": "lig",
                "docking_status": "success",
                "vina_score": -7.0,
                "molecular_weight": 100.0,
                "qed": 0.5,
                "interactions_json": "interactions.json",
            }
        ],
        output_path=tmp_path / "report.html",
        manifest_json=manifest,
        summary_txt=summary,
    )

    text = report.read_text()
    assert "Demo Report" in text
    assert "pocket_1" in text
    assert "-7.0" in text


def test_run_reports_use_generic_docking_score_for_gnina_rows(tmp_path):
    manifest = tmp_path / "manifest.json"
    summary = tmp_path / "summary.txt"
    manifest.write_text("{}", encoding="utf-8")
    summary.write_text("summary", encoding="utf-8")
    rows = [
        {
            "receptor_name": "rec",
            "pocket_name": "pocket_1",
            "ligand_name": "gnina_lig",
            "docking_status": "success",
            "vina_score": -1.0,
            "docking_score": -8.5,
            "score_label": "GNINA affinity",
            "docking_engine": "gnina",
        }
    ]

    html_report = write_run_html_report(
        title="GNINA Report",
        results=rows,
        output_path=tmp_path / "report.html",
        manifest_json=manifest,
        summary_txt=summary,
    )
    pdf_report = write_run_pdf_report(
        title="GNINA Report",
        results=rows,
        output_path=tmp_path / "report.pdf",
        manifest_json=manifest,
        summary_txt=summary,
    )

    html_text = html_report.read_text(encoding="utf-8")
    pdf_bytes = pdf_report.read_bytes()
    assert "Docking Score" in html_text
    assert "GNINA affinity" in html_text
    assert "-8.5" in html_text
    assert "-1.0" not in html_text
    assert b"-8.500" in pdf_bytes
    assert b"-1.000" not in pdf_bytes


def test_run_html_report_warns_about_unfavorable_positive_scores(tmp_path):
    manifest = tmp_path / "manifest.json"
    summary = tmp_path / "summary.txt"
    manifest.write_text("{}", encoding="utf-8")
    summary.write_text("summary", encoding="utf-8")

    report = write_run_html_report(
        title="Docking Report",
        results=[
            {
                "receptor_name": "rec",
                "pocket_name": "pocket_bad",
                "ligand_name": "lig",
                "docking_status": "success",
                "docking_score": 12.5,
                "vina_score": 12.5,
            }
        ],
        output_path=tmp_path / "report.html",
        manifest_json=manifest,
        summary_txt=summary,
    )

    text = report.read_text(encoding="utf-8")
    assert "Unfavorable docking scores" in text
    assert "lig / pocket_bad: non-negative docking score 12.5" in text


def test_write_run_html_report_includes_available_warnings_and_admet_flags(tmp_path):
    manifest = tmp_path / "manifest.json"
    summary = tmp_path / "summary.txt"
    manifest.write_text("{}", encoding="utf-8")
    summary.write_text("summary", encoding="utf-8")

    report = write_run_html_report(
        title="Demo Report",
        results=[
            {
                "receptor_name": "rec",
                "pocket_name": "pocket_1",
                "ligand_name": "alert_ligand",
                "docking_status": "success",
                "vina_score": -8.1,
                "molecular_weight": 650.0,
                "qed": 0.2,
                "lipinski_failures": 2,
                "rule_of_five_pass": False,
                "structural_alert_count": 1,
            }
        ],
        output_path=tmp_path / "report.html",
        manifest_json=manifest,
        summary_txt=summary,
        receptor_prep_warnings=["Metal/ion HETATM records were kept."],
        fpocket_warnings=["fpocket metadata was missing volume values."],
    )

    text = report.read_text(encoding="utf-8")
    assert "Warnings" in text
    assert "Metal/ion HETATM records were kept." in text
    assert "fpocket metadata was missing volume values." in text
    assert "ADMET" in text
    assert "alert_ligand" in text
    assert "Lipinski failures: 2" in text


def test_run_reports_include_summaries_failures_and_warnings(tmp_path):
    manifest = tmp_path / "manifest.json"
    summary = tmp_path / "summary.txt"
    manifest.write_text("{}", encoding="utf-8")
    summary.write_text("summary", encoding="utf-8")

    html_report = write_run_html_report(
        title="Detailed Report",
        results=[
            {
                "receptor_name": "rec",
                "pocket_name": "pocket_1",
                "ligand_name": "lig",
                "docking_status": "failed",
                "vina_score": None,
                "failure_reason": "Vina timeout",
                "qed": 0.8,
                "rule_of_five_pass": True,
                "veber_pass": True,
            }
        ],
        output_path=tmp_path / "report.html",
        manifest_json=manifest,
        summary_txt=summary,
        receptor_prep_summary={"atoms_kept": 1250, "waters_removed": 12},
        fpocket_pocket_summary=[{"name": "pocket_1", "score": 44.2, "volume": 310.0}],
        warnings=["Manual warning"],
        failures=["Docking binary returned no pose"],
    )
    pdf_report = write_run_pdf_report(
        title="Detailed Report",
        results=[],
        output_path=tmp_path / "report.pdf",
        manifest_json=manifest,
        summary_txt=summary,
        admet_summary={"pass": 1, "warn": 0, "fail": 0, "total": 1},
        receptor_prep_summary={"atoms_kept": 1250},
        fpocket_pocket_summary=[{"name": "pocket_1", "score": 44.2}],
        warnings=["Manual warning"],
        failures=["Docking binary returned no pose"],
    )

    html_text = html_report.read_text(encoding="utf-8")
    pdf_bytes = pdf_report.read_bytes()
    assert "ADMET Summary" in html_text
    assert "Receptor Prep Summary" in html_text
    assert "fpocket Pocket Summary" in html_text
    assert "atoms_kept" in html_text
    assert "pocket_1" in html_text
    assert "Manual warning" in html_text
    assert "Docking binary returned no pose" in html_text
    assert b"ADMET Summary" in pdf_bytes
    assert b"Receptor Prep Summary" in pdf_bytes
    assert b"fpocket Pocket Summary" in pdf_bytes


def test_write_run_pdf_report_writes_pdf(tmp_path):
    pdf_path = write_run_pdf_report(
        title="Batch",
        results=[{"docking_status": "success", "ligand_name": "lig1", "vina_score": -7.0}],
        output_path=tmp_path / "report.pdf",
        manifest_json=tmp_path / "run_manifest.json",
        summary_txt=tmp_path / "summary.txt",
    )

    assert pdf_path.read_bytes().startswith(b"%PDF-1.4")
