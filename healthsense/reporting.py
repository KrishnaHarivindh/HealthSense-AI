from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from healthsense.config import REPORTS_ROOT, ensure_platform_dirs


def build_prediction_report_payload(module: str, input_summary: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": module,
        "input_summary": input_summary,
        "prediction": prediction,
    }


def build_prediction_report_html(module: str, input_summary: dict[str, Any], prediction: dict[str, Any]) -> str:
    rows = []
    for section_name, section_payload in [("Input Summary", input_summary), ("Prediction", prediction)]:
        rows.append(f"<tr><th colspan='2'>{escape(section_name)}</th></tr>")
        for key, value in section_payload.items():
            rows.append(f"<tr><td>{escape(str(key))}</td><td>{escape(str(value))}</td></tr>")
    row_markup = "".join(rows)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>HealthSense AI Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2937; }}
    h1 {{ color: #0f766e; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 10px; text-align: left; }}
    th {{ background: #ecfeff; }}
    .note {{ margin-top: 20px; color: #475569; }}
  </style>
</head>
<body>
  <h1>HealthSense AI {escape(module.title())} Report</h1>
  <table>{row_markup}</table>
  <p class="note">Academic decision-support output only. Not for clinical diagnosis.</p>
</body>
</html>
"""


def write_report_files(report_name: str, json_payload: dict[str, Any], html_payload: str) -> dict[str, Path]:
    ensure_platform_dirs()
    report_dir = REPORTS_ROOT / "generated"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{report_name}.json"
    html_path = report_dir / f"{report_name}.html"
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    html_path.write_text(html_payload, encoding="utf-8")
    return {"json": json_path, "html": html_path}


def build_session_report_html(session_name: str, items: list[dict[str, Any]]) -> str:
    sections = []
    for item in items:
        sections.append(f"<h2>{escape(str(item.get('module', 'Unknown')).title())}</h2>")
        sections.append("<ul>")
        for key, value in item.items():
            sections.append(f"<li><strong>{escape(str(key))}:</strong> {escape(str(value))}</li>")
        sections.append("</ul>")
    joined_sections = "".join(sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>HealthSense AI Session Report</title>
</head>
<body>
  <h1>HealthSense AI Session Report: {escape(session_name)}</h1>
  {joined_sections}
  <p>Academic decision-support output only. Not for clinical diagnosis.</p>
</body>
</html>
"""
