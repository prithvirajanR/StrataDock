from __future__ import annotations

import csv
import html
import json
from pathlib import Path


VALIDATION_COLUMNS = [
    "case_id",
    "score",
    "heavy_atom_rmsd",
    "expected_rmsd_max",
    "passes_rmsd_threshold",
    "pose_pdbqt",
    "native_ligand_pdb",
    "receptor_report_json",
]


def write_validation_reports(summary: dict[str, object], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "validation_summary.json"
    csv_path = output_dir / "validation_report.csv"
    txt_path = output_dir / "validation_report.txt"

    json_payload = dict(summary)
    json_payload["validation_report_csv"] = str(csv_path)
    json_payload["validation_report_txt"] = str(txt_path)
    json_path.write_text(json.dumps(json_payload, indent=2) + "\n", encoding="utf-8")

    rows = list(summary.get("results", []))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=VALIDATION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in VALIDATION_COLUMNS})

    lines = [
        "StrataDock validation report",
        "=" * 29,
        f"Total cases: {summary.get('total', 0)}",
        f"Passed: {summary.get('passed', 0)}",
        f"Failed: {summary.get('failed', 0)}",
        "",
        "Expected vs actual:",
    ]
    for row in rows:
        status = "PASS" if row.get("passes_rmsd_threshold") else "FAIL"
        lines.append(
            f"- {row.get('case_id')}: {status}; "
            f"RMSD {row.get('heavy_atom_rmsd')} A <= {row.get('expected_rmsd_max')} A; "
            f"score {row.get('score')}"
        )
    failures = list(summary.get("failures", []))
    if failures:
        lines.append("")
        lines.append("Execution failures:")
        for failure in failures:
            lines.append(f"- {failure.get('case_id')}: return code {failure.get('returncode')}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"json": json_path, "csv": csv_path, "txt": txt_path}


def write_run_html_report(
    *,
    title: str,
    results: list[dict[str, object]],
    output_path: Path,
    manifest_json: Path,
    summary_txt: Path,
    receptor_prep_warnings: list[str] | None = None,
    fpocket_warnings: list[str] | None = None,
    admet_summary: dict[str, object] | None = None,
    receptor_prep_summary: dict[str, object] | None = None,
    fpocket_pocket_summary: list[dict[str, object]] | None = None,
    warnings: list[str] | None = None,
    failures: list[str] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = [row for row in results if row.get("docking_status") == "success"]
    best = sorted(success, key=lambda row: _row_score(row) if _row_score(row) is not None else 999.0)[:25]
    warning_sections = _html_warning_sections(
        receptor_prep_warnings=receptor_prep_warnings or [],
        fpocket_warnings=fpocket_warnings or [],
        docking_score_warnings=_docking_score_warning_lines(results),
        admet_warnings=_admet_warning_lines(results),
        warnings=warnings or [],
    )
    summary_sections = _html_summary_sections(
        admet_summary=admet_summary or _admet_summary(results),
        receptor_prep_summary=receptor_prep_summary,
        fpocket_pocket_summary=fpocket_pocket_summary,
        failures=failures or _failure_lines(results),
    )
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('receptor_name', '')))}</td>"
        f"<td>{html.escape(str(row.get('pocket_name', '')))}</td>"
        f"<td>{html.escape(str(row.get('ligand_name', '')))}</td>"
        f"<td>{html.escape(str(row.get('docking_score', row.get('vina_score', ''))))}</td>"
        f"<td>{html.escape(str(row.get('score_label', _score_label_for_row(row))))}</td>"
        f"<td>{html.escape(str(row.get('molecular_weight', '')))}</td>"
        f"<td>{html.escape(str(row.get('qed', '')))}</td>"
        f"<td>{html.escape(str(row.get('interactions_json', '')))}</td>"
        "</tr>"
        for row in best
    )
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f2f2f2; }}
    .meta {{ color: #555; margin-bottom: 20px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="meta">
    <div>Total records: {len(results)}</div>
    <div>Successful dockings: {len(success)}</div>
    <div>Manifest: {html.escape(str(manifest_json))}</div>
    <div>Summary: {html.escape(str(summary_txt))}</div>
  </div>
  {summary_sections}
  {warning_sections}
  <h2>Top Docking Results</h2>
  <table>
    <thead>
      <tr><th>Receptor</th><th>Pocket</th><th>Ligand</th><th>Docking Score</th><th>Score Type</th><th>MW</th><th>QED</th><th>Interactions</th></tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )
    return output_path


def write_run_pdf_report(
    *,
    title: str,
    results: list[dict[str, object]],
    output_path: Path,
    manifest_json: Path,
    summary_txt: Path,
    receptor_prep_warnings: list[str] | None = None,
    fpocket_warnings: list[str] | None = None,
    admet_summary: dict[str, object] | None = None,
    receptor_prep_summary: dict[str, object] | None = None,
    fpocket_pocket_summary: list[dict[str, object]] | None = None,
    warnings: list[str] | None = None,
    failures: list[str] | None = None,
) -> Path:
    success = [row for row in results if row.get("docking_status") == "success"]
    failed = len(results) - len(success)
    best = sorted(success, key=lambda row: _row_score(row) if _row_score(row) is not None else 999.0)[:24]
    best_score = ""
    if best and _row_score(best[0]) is not None:
        best_score = _format_number(_row_score(best[0]), digits=3)
    warning_lines = _plain_warning_lines(
        receptor_prep_warnings=receptor_prep_warnings or [],
        fpocket_warnings=fpocket_warnings or [],
        docking_score_warnings=_docking_score_warning_lines(results),
        admet_warnings=_admet_warning_lines(results),
        warnings=warnings or [],
    )
    failure_lines = failures or _failure_lines(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_run_pdf(
        output_path,
        title=title,
        metadata={"Manifest": manifest_json, "Summary": summary_txt},
        metrics=[
            ("Total Jobs", len(results)),
            ("Successful", len(success)),
            ("Failed", failed),
            ("Best Docking Score", best_score or "n/a"),
        ],
        admet_summary=admet_summary or _admet_summary(results),
        receptor_prep_summary=receptor_prep_summary or {},
        fpocket_pocket_summary=fpocket_pocket_summary or [],
        warning_lines=warning_lines,
        failure_lines=failure_lines,
        best_rows=best,
    )
    return output_path


def _admet_warning_lines(results: list[dict[str, object]]) -> list[str]:
    warnings: list[str] = []
    for row in _unique_ligand_rows(results):
        ligand = str(row.get("ligand_name", "ligand"))
        lipinski = _as_int(row.get("lipinski_failures"))
        structural_alerts = _as_int(row.get("structural_alert_count"))
        if lipinski and lipinski > 0:
            warnings.append(f"{ligand}: Lipinski failures: {lipinski}")
        if structural_alerts and structural_alerts > 0:
            warnings.append(f"{ligand}: structural alerts: {structural_alerts}")
    return warnings


def _docking_score_warning_lines(results: list[dict[str, object]]) -> list[str]:
    warnings: list[str] = []
    for row in results:
        if str(row.get("docking_status") or "").lower() != "success":
            continue
        score = _row_score(row)
        if score in (None, ""):
            continue
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            continue
        if numeric_score >= 0:
            ligand = str(row.get("ligand_name") or "ligand")
            pocket = str(row.get("pocket_name") or "pocket")
            warnings.append(f"{ligand} / {pocket}: non-negative docking score {numeric_score:g}")
    return warnings


def _row_score(row: dict[str, object]) -> object:
    return row.get("docking_score", row.get("vina_score"))


def _score_label_for_row(row: dict[str, object]) -> str:
    label = row.get("score_label")
    if label:
        return str(label)
    return "GNINA affinity" if str(row.get("docking_engine") or "").lower() == "gnina" else "Vina score"


def _admet_summary(results: list[dict[str, object]]) -> dict[str, object]:
    rows = [
        row
        for row in _unique_ligand_rows(results)
        if any(key in row for key in ["rule_of_five_pass", "veber_pass", "qed", "lipinski_failures", "structural_alert_count"])
    ]
    if not rows:
        return {}
    passed = 0
    warned = 0
    failed = 0
    qeds: list[float] = []
    for row in rows:
        rule_pass = row.get("rule_of_five_pass")
        veber_pass = row.get("veber_pass")
        alerts = _as_int(row.get("structural_alert_count")) or 0
        lipinski = _as_int(row.get("lipinski_failures")) or 0
        if row.get("qed") not in (None, ""):
            try:
                qeds.append(float(row["qed"]))
            except (TypeError, ValueError):
                pass
        if rule_pass is False or veber_pass is False or lipinski > 0:
            failed += 1
        elif alerts > 0:
            warned += 1
        else:
            passed += 1
    summary: dict[str, object] = {"total": len(rows), "pass": passed, "warn": warned, "fail": failed}
    if qeds:
        summary["mean_qed"] = round(sum(qeds) / len(qeds), 3)
    return summary


def _unique_ligand_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[object, object]] = set()
    rows: list[dict[str, object]] = []
    for row in results:
        key = (row.get("source_index"), row.get("ligand_name"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def _failure_lines(results: list[dict[str, object]]) -> list[str]:
    failures: list[str] = []
    for row in results:
        status = str(row.get("docking_status") or "").lower()
        reason = row.get("failure_reason") or row.get("error") or row.get("message")
        if status not in {"success", ""} or reason:
            ligand = str(row.get("ligand_name") or "ligand")
            failures.append(f"{ligand}: {reason or status}")
    return failures


def _html_summary_sections(
    *,
    admet_summary: dict[str, object] | None,
    receptor_prep_summary: dict[str, object] | None,
    fpocket_pocket_summary: list[dict[str, object]] | None,
    failures: list[str],
) -> str:
    sections = [
        _html_key_value_section("ADMET Summary", admet_summary or {}),
        _html_key_value_section("Receptor Prep Summary", receptor_prep_summary or {}),
        _html_table_section("fpocket Pocket Summary", fpocket_pocket_summary or []),
        _html_list_section("Warnings And Failures", failures),
    ]
    return "\n".join(section for section in sections if section)


def _html_key_value_section(title: str, values: dict[str, object]) -> str:
    if not values:
        return ""
    rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in values.items()
    )
    return f"<h2>{html.escape(title)}</h2><table><tbody>{rows}</tbody></table>"


def _html_table_section(title: str, rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns) + "</tr>"
        for row in rows
    )
    return f"<h2>{html.escape(title)}</h2><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _html_list_section(title: str, values: list[str]) -> str:
    if not values:
        return ""
    body = "".join(f"<li>{html.escape(value)}</li>" for value in values)
    return f"<h2>{html.escape(title)}</h2><ul>{body}</ul>"


def _html_warning_sections(
    *,
    receptor_prep_warnings: list[str],
    fpocket_warnings: list[str],
    docking_score_warnings: list[str],
    admet_warnings: list[str],
    warnings: list[str],
) -> str:
    body = "\n".join(
        section
        for section in [
            _html_warning_section("Receptor Prep Warnings", receptor_prep_warnings),
            _html_warning_section("fpocket Warnings", fpocket_warnings),
            _html_warning_section("Unfavorable docking scores", docking_score_warnings),
            _html_warning_section("ADMET Warnings", admet_warnings),
            _html_warning_section("General Warnings", warnings),
        ]
        if section
    )
    if not body:
        return ""
    return f"<h2>Warnings</h2>\n{body}"


def _html_warning_section(title: str, warnings: list[str]) -> str:
    if not warnings:
        return ""
    rows = "".join(f"<li>{html.escape(warning)}</li>" for warning in warnings)
    return f"<h3>{html.escape(title)}</h3><ul>{rows}</ul>"


def _plain_warning_lines(
    *,
    receptor_prep_warnings: list[str],
    fpocket_warnings: list[str],
    docking_score_warnings: list[str],
    admet_warnings: list[str],
    warnings: list[str],
) -> list[str]:
    lines: list[str] = []
    for label, warnings in [
        ("Receptor prep", receptor_prep_warnings),
        ("fpocket", fpocket_warnings),
        ("Unfavorable docking score", docking_score_warnings),
        ("ADMET", admet_warnings),
        ("General", warnings),
    ]:
        for warning in warnings:
            lines.append(f"- {label}: {warning}")
    return lines


def _plain_summary_lines(
    *,
    admet_summary: dict[str, object] | None,
    receptor_prep_summary: dict[str, object] | None,
    fpocket_pocket_summary: list[dict[str, object]] | None,
    failures: list[str],
) -> list[str]:
    lines: list[str] = []
    if admet_summary:
        lines.append("ADMET Summary:")
        lines.extend(f"- {key}: {value}" for key, value in admet_summary.items())
    if receptor_prep_summary:
        lines.append("Receptor Prep Summary:")
        lines.extend(f"- {key}: {value}" for key, value in receptor_prep_summary.items())
    if fpocket_pocket_summary:
        lines.append("fpocket Pocket Summary:")
        for row in fpocket_pocket_summary[:10]:
            label = row.get("name") or row.get("rank") or "pocket"
            score = row.get("score", "")
            volume = row.get("volume", "")
            lines.append(f"- {label}: score {score}; volume {volume}")
    if failures:
        lines.append("Warnings And Failures:")
        lines.extend(f"- {failure}" for failure in failures)
    return lines


def _as_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: object, *, digits: int = 2) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _write_run_pdf(
    path: Path,
    *,
    title: str,
    metadata: dict[str, object],
    metrics: list[tuple[str, object]],
    admet_summary: dict[str, object],
    receptor_prep_summary: dict[str, object],
    fpocket_pocket_summary: list[dict[str, object]],
    warning_lines: list[str],
    failure_lines: list[str],
    best_rows: list[dict[str, object]],
) -> None:
    writer = _PdfReportWriter(title)
    writer.header(title)
    writer.metadata(metadata)
    writer.metric_row(metrics)
    writer.key_value_section("ADMET Summary", admet_summary)
    writer.key_value_section("Receptor Prep Summary", receptor_prep_summary)
    writer.pocket_section(fpocket_pocket_summary)
    writer.list_section("Warnings", warning_lines)
    writer.list_section("Warnings And Failures", failure_lines)
    writer.results_table(best_rows)
    writer.save(path)


class _PdfReportWriter:
    page_width = 612
    page_height = 792
    margin = 42
    bottom_margin = 52

    def __init__(self, title: str) -> None:
        self.title = title
        self.pages: list[list[str]] = []
        self.ops: list[str] = []
        self.y = self.page_height - self.margin
        self.page_number = 0
        self._new_page()

    def _new_page(self) -> None:
        if self.ops:
            self._footer()
            self.pages.append(self.ops)
        self.page_number += 1
        self.ops = []
        self.y = self.page_height - self.margin
        if self.page_number > 1:
            self._text(self.margin, self.y, self.title, font="F2", size=10, color=(0.35, 0.35, 0.35))
            self.y -= 18

    def save(self, path: Path) -> None:
        self._footer()
        self.pages.append(self.ops)
        _write_pdf_objects(path, self.pages)

    def header(self, title: str) -> None:
        self._rect(0, self.page_height - 100, self.page_width, 100, fill=(0.12, 0.12, 0.12))
        self._text(self.margin, self.page_height - 52, title, font="F2", size=22, color=(1, 1, 1))
        self._text(self.margin, self.page_height - 76, "StrataDock v 1.6.01 docking report", font="F1", size=10, color=(0.78, 0.78, 0.78))
        self.y = self.page_height - 126

    def metadata(self, values: dict[str, object]) -> None:
        rows = [(key, str(value)) for key, value in values.items() if value not in (None, "")]
        if not rows:
            return
        self._ensure(44 + 13 * len(rows))
        self._text(self.margin, self.y, "Run Files", font="F2", size=10, color=(0.18, 0.18, 0.18))
        self.y -= 16
        for key, value in rows:
            self._text(self.margin, self.y, f"{key}:", font="F2", size=8.5, color=(0.35, 0.35, 0.35))
            for idx, line in enumerate(_wrap_text(value, 82)):
                self._text(self.margin + 58, self.y - (idx * 10), line, font="F1", size=8.5, color=(0.35, 0.35, 0.35))
            self.y -= max(13, 10 * len(_wrap_text(value, 82)))
        self.y -= 8

    def metric_row(self, metrics: list[tuple[str, object]]) -> None:
        if not metrics:
            return
        self._ensure(76)
        gap = 8
        width = (self.page_width - 2 * self.margin - gap * (len(metrics) - 1)) / len(metrics)
        height = 54
        for idx, (label, value) in enumerate(metrics):
            x = self.margin + idx * (width + gap)
            self._rect(x, self.y - height, width, height, fill=(0.96, 0.96, 0.96), stroke=(0.82, 0.82, 0.82))
            self._text(x + 10, self.y - 21, str(value), font="F2", size=15, color=(0.16, 0.16, 0.16))
            self._text(x + 10, self.y - 39, str(label).upper(), font="F2", size=7.5, color=(0.45, 0.45, 0.45))
        self.y -= height + 20

    def key_value_section(self, title: str, values: dict[str, object]) -> None:
        if not values:
            return
        rows = [(str(key), str(value)) for key, value in values.items()]
        self._section_title(title)
        row_h = 20
        table_w = self.page_width - 2 * self.margin
        key_w = 160
        for key, value in rows:
            self._ensure(row_h + 8)
            y0 = self.y - row_h
            self._rect(self.margin, y0, table_w, row_h, stroke=(0.86, 0.86, 0.86))
            self._rect(self.margin, y0, key_w, row_h, fill=(0.95, 0.95, 0.95), stroke=(0.86, 0.86, 0.86))
            self._text(self.margin + 8, self.y - 13, key, font="F2", size=8.3, color=(0.25, 0.25, 0.25))
            self._text(self.margin + key_w + 8, self.y - 13, value[:70], font="F1", size=8.3, color=(0.25, 0.25, 0.25))
            self.y -= row_h
        self.y -= 14

    def pocket_section(self, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        self._section_title("fpocket Pocket Summary")
        columns = [("Pocket", 120), ("Score", 80), ("Volume", 90), ("Center", 220)]
        self._table_header(columns)
        for row in rows[:8]:
            center = ", ".join(
                _format_number(row.get(key), digits=1)
                for key in ("center_x", "center_y", "center_z")
                if row.get(key) not in (None, "")
            )
            values = [
                str(row.get("name") or row.get("rank") or "pocket"),
                _format_number(row.get("score"), digits=2),
                _format_number(row.get("volume"), digits=1),
                center,
            ]
            self._table_row(columns, values)
        self.y -= 12

    def list_section(self, title: str, values: list[str]) -> None:
        if not values:
            return
        self._section_title(title)
        for value in values[:12]:
            lines = _wrap_text(str(value), 92)
            self._ensure(13 * len(lines) + 4)
            self._text(self.margin + 4, self.y, "-", font="F2", size=9, color=(0.45, 0.45, 0.45))
            for idx, line in enumerate(lines):
                self._text(self.margin + 18, self.y - idx * 12, line, font="F1", size=8.5, color=(0.25, 0.25, 0.25))
            self.y -= 13 * len(lines)
        self.y -= 10

    def results_table(self, rows: list[dict[str, object]]) -> None:
        self._section_title("Top Docking Results")
        if not rows:
            self._text(self.margin, self.y, "No successful docking rows available.", size=9, color=(0.35, 0.35, 0.35))
            self.y -= 18
            return
        columns = [
            ("Receptor", 105),
            ("Pocket", 68),
            ("Ligand", 120),
            ("Score", 56),
            ("MW", 56),
            ("QED", 46),
            ("Status", 73),
        ]
        self._table_header(columns)
        for row in rows:
            values = [
                str(row.get("receptor_name", "")),
                str(row.get("pocket_name", "")),
                str(row.get("ligand_name", "")),
                _format_number(_row_score(row), digits=3),
                _format_number(row.get("molecular_weight"), digits=1),
                _format_number(row.get("qed"), digits=3),
                str(row.get("docking_status", "")),
            ]
            self._table_row(columns, values)

    def _section_title(self, title: str) -> None:
        self._ensure(34)
        self._text(self.margin, self.y, title, font="F2", size=12, color=(0.12, 0.12, 0.12))
        self.y -= 16

    def _table_header(self, columns: list[tuple[str, int]]) -> None:
        self._ensure(24)
        self._rect(self.margin, self.y - 20, sum(width for _, width in columns), 20, fill=(0.14, 0.14, 0.14))
        x = self.margin
        for label, width in columns:
            self._text(x + 5, self.y - 13, label, font="F2", size=7.8, color=(1, 1, 1))
            x += width
        self.y -= 20

    def _table_row(self, columns: list[tuple[str, int]], values: list[str]) -> None:
        row_h = 21
        self._ensure(row_h + 8)
        x = self.margin
        fill = (0.985, 0.985, 0.985) if int((self.page_height - self.y) / row_h) % 2 else None
        if fill:
            self._rect(self.margin, self.y - row_h, sum(width for _, width in columns), row_h, fill=fill)
        for (_, width), value in zip(columns, values):
            self._rect(x, self.y - row_h, width, row_h, stroke=(0.86, 0.86, 0.86))
            self._text(x + 5, self.y - 13, _fit_text(str(value), max(4, int(width / 4.8))), font="F1", size=7.5, color=(0.18, 0.18, 0.18))
            x += width
        self.y -= row_h

    def _ensure(self, height: float) -> None:
        if self.y - height < self.bottom_margin:
            self._new_page()

    def _footer(self) -> None:
        self._line(self.margin, 38, self.page_width - self.margin, 38, color=(0.82, 0.82, 0.82))
        self._text(self.margin, 24, "StrataDock v 1.6.01", size=7.5, color=(0.50, 0.50, 0.50))
        self._text(self.page_width - self.margin - 48, 24, f"Page {self.page_number}", size=7.5, color=(0.50, 0.50, 0.50))

    def _text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        font: str = "F1",
        size: float = 9,
        color: tuple[float, float, float] = (0, 0, 0),
    ) -> None:
        safe = _escape_pdf_text(str(text))
        self.ops.append(
            f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg BT /{font} {size:.1f} Tf {x:.1f} {y:.1f} Td ({safe}) Tj ET"
        )

    def _rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: tuple[float, float, float] | None = None,
        stroke: tuple[float, float, float] | None = None,
    ) -> None:
        if fill:
            self.ops.append(f"{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg {x:.1f} {y:.1f} {w:.1f} {h:.1f} re f")
        if stroke:
            self.ops.append(f"{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG {x:.1f} {y:.1f} {w:.1f} {h:.1f} re S")

    def _line(self, x1: float, y1: float, x2: float, y2: float, *, color: tuple[float, float, float]) -> None:
        self.ops.append(f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} RG {x1:.1f} {y1:.1f} m {x2:.1f} {y2:.1f} l S")


def _wrap_text(value: str, width: int) -> list[str]:
    words = str(value).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        if len(word) > width:
            if current:
                lines.append(current)
                current = ""
            lines.extend(word[idx : idx + width] for idx in range(0, len(word), width))
        elif not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _fit_text(value: str, width: int) -> str:
    value = str(value)
    if len(value) <= width:
        return value
    return value[: max(1, width - 3)] + "..."


def _write_pdf_objects(path: Path, pages: list[list[str]]) -> None:
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ]
    page_object_ids: list[int] = []
    for ops in pages:
        page_id = len(objects) + 1
        content_id = page_id + 1
        page_object_ids.append(page_id)
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_id} 0 R >>".encode(
                "ascii"
            )
        )
        stream = "\n".join(ops).encode("latin-1", errors="replace")
        objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
    kids = " ".join(f"{page_id} 0 R" for page_id in page_object_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii")

    body = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body.extend(f"{idx} 0 obj\n".encode("ascii"))
        body.extend(obj)
        body.extend(b"\nendobj\n")
    xref_offset = len(body)
    body.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    body.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    body.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(body)


def _write_simple_pdf(path: Path, lines: list[str]) -> None:
    content_lines = ["BT", "/F1 10 Tf", "14 TL", "50 790 Td"]
    for line in lines:
        content_lines.append(f"({_escape_pdf_text(line[:110])}) Tj")
        content_lines.append("T*")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    body = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body.extend(f"{idx} 0 obj\n".encode("ascii"))
        body.extend(obj)
        body.extend(b"\nendobj\n")
    xref_offset = len(body)
    body.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    body.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    body.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(body)


def _escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
