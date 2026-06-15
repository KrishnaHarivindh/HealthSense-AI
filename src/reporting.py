from __future__ import annotations

import json
from html import escape


def build_prediction_payload(
    result: dict,
    metrics: dict,
    weights: dict[str, float],
) -> dict:
    return {
        "prediction": {
            "record_id": result.get("record_id"),
            "created_at": result.get("created_at"),
            "image_name": result.get("image_name"),
            "target_label": result.get("target_label"),
            "patient": {
                "age": result.get("patient_age"),
                "gender": result.get("patient_gender"),
                "view_position": result.get("view_position"),
                "follow_up": result.get("follow_up"),
            },
            "model_outputs": {
                "cnn_probability": result.get("cnn_probability"),
                "random_forest_probability": result.get("random_forest_probability"),
                "svm_probability": result.get("svm_probability"),
                "final_probability": result.get("final_probability"),
            },
            "decision": {
                "threshold": result.get("threshold"),
                "diagnosis": result.get("diagnosis"),
                "severity": result.get("severity"),
            },
        },
        "model_context": {
            "ensemble_weights": weights,
            "ensemble_metrics": metrics.get("metrics", {}).get("ensemble", {}),
            "cnn_metrics": metrics.get("metrics", {}).get("cnn", {}),
        },
    }


def build_prediction_report_json(
    result: dict,
    metrics: dict,
    weights: dict[str, float],
) -> bytes:
    payload = build_prediction_payload(result, metrics, weights)
    return json.dumps(payload, indent=2).encode("utf-8")


def build_prediction_report_html(
    result: dict,
    metrics: dict,
    weights: dict[str, float],
) -> str:
    ensemble_metrics = metrics.get("metrics", {}).get("ensemble", {})
    rows = [
        ("Record ID", result.get("record_id")),
        ("Created At", result.get("created_at")),
        ("Image Name", result.get("image_name")),
        ("Target Label", result.get("target_label")),
        ("Patient Age", result.get("patient_age")),
        ("Gender", result.get("patient_gender")),
        ("View Position", result.get("view_position")),
        ("Follow-up Count", result.get("follow_up")),
        ("CNN Probability", f"{float(result.get('cnn_probability', 0.0)) * 100:.2f}%"),
        ("Random Forest Probability", f"{float(result.get('random_forest_probability', 0.0)) * 100:.2f}%"),
        ("SVM Probability", f"{float(result.get('svm_probability', 0.0)) * 100:.2f}%"),
        ("Final Ensemble Probability", f"{float(result.get('final_probability', 0.0)) * 100:.2f}%"),
        ("Decision Threshold", f"{float(result.get('threshold', 0.5)):.2f}"),
        ("Diagnosis", result.get("diagnosis")),
        ("Severity", result.get("severity")),
        ("Ensemble ROC-AUC", f"{float(ensemble_metrics.get('roc_auc', 0.0)):.3f}"),
        ("Ensemble F1 Score", f"{float(ensemble_metrics.get('f1_score', 0.0)):.3f}"),
        (
            "Ensemble Weights",
            ", ".join(f"{escape(name)}={value:.2f}" for name, value in weights.items()),
        ),
    ]

    row_markup = "".join(
        f"<tr><th>{escape(str(label))}</th><td>{escape(str(value))}</td></tr>"
        for label, value in rows
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>HealthSense AI Prediction Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1d2733; }}
    h1 {{ color: #0d9488; margin-bottom: 8px; }}
    p {{ line-height: 1.5; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
    th, td {{ border: 1px solid #dbe4ea; padding: 10px; text-align: left; }}
    th {{ background: #f0fdfa; width: 32%; }}
    .note {{ margin-top: 24px; color: #475569; }}
  </style>
</head>
<body>
  <h1>HealthSense AI Prediction Report</h1>
  <p>Automated academic demo report generated from the chest X-ray ensemble workflow.</p>
  <table>{row_markup}</table>
  <p class="note">
    This output is for academic demonstration only and should not be treated as a clinical diagnosis.
  </p>
</body>
</html>
"""
