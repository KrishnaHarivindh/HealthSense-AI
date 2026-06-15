from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from healthsense.config import ARTIFACTS_ROOT, get_module_spec


def load_app_styles(project_root: Path) -> None:
    css_path = project_root / "app" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def format_probability(value: Any) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value) * 100:.1f}%"


def format_metric(value: Any) -> str:
    if value is None or pd.isna(value):
        return "--"
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return str(value)


def prettify_feature_name(name: str) -> str:
    text = str(name)
    for token in ["numeric__", "categorical__", "remainder__", "onehotencoder__", "standardscaler__", "simpleimputer__"]:
        text = text.replace(token, "")
    return text.replace("_", " ")


def module_ready(module: str) -> bool:
    artifact_dir = ARTIFACTS_ROOT / module
    return (artifact_dir / "config.json").exists() or (artifact_dir / "model_config.joblib").exists()


def load_metrics_payload(module: str) -> dict[str, Any]:
    path = ARTIFACTS_ROOT / module / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_module_config(module: str) -> dict[str, Any]:
    path = ARTIFACTS_ROOT / module / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def selected_model_label(module: str) -> str:
    config = load_module_config(module)
    if not config:
        return "Unavailable"
    if module in {"chest", "skin"}:
        backbone = str(config.get("backbone", "model")).replace("_", " ").title()
        variant = str(config.get("selected_variant", "base")).replace("_", " ").title()
        return f"{backbone} / {variant}"
    return str(config.get("selected_model", "model")).title()


def headline_metric(module: str) -> tuple[str, str]:
    payload = load_metrics_payload(module)
    metrics = payload.get("metrics", {})
    if not metrics:
        return "Status", "Pending"
    if module in {"chest", "skin"}:
        return "Macro ROC-AUC", format_metric(metrics.get("macro_roc_auc"))
    return "ROC-AUC", format_metric(metrics.get("roc_auc"))


def summarize_metrics(module: str, metrics: dict[str, Any]) -> list[tuple[str, str]]:
    if module == "chest":
        pneumonia_auc = None
        per_label = metrics.get("per_label_roc_auc", {})
        if isinstance(per_label, dict):
            pneumonia_auc = per_label.get("Pneumonia")
        return [
            ("Macro ROC-AUC", format_metric(metrics.get("macro_roc_auc"))),
            ("Micro ROC-AUC", format_metric(metrics.get("micro_roc_auc"))),
            ("Macro F1", format_metric(metrics.get("macro_f1"))),
            ("Pneumonia ROC-AUC", format_metric(pneumonia_auc)),
        ]
    if module == "skin":
        return [
            ("Accuracy", format_metric(metrics.get("accuracy"))),
            ("Macro F1", format_metric(metrics.get("macro_f1"))),
            ("Macro ROC-AUC", format_metric(metrics.get("macro_roc_auc"))),
            ("Macro PR-AUC", format_metric(metrics.get("macro_pr_auc"))),
        ]
    return [
        ("ROC-AUC", format_metric(metrics.get("roc_auc"))),
        ("PR-AUC", format_metric(metrics.get("pr_auc"))),
        ("Recall", format_metric(metrics.get("recall"))),
        ("F1", format_metric(metrics.get("f1"))),
    ]


def probability_frame(probabilities: dict[str, float], top_n: int | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        [{"Label": label, "Probability": float(probability)} for label, probability in probabilities.items()]
    ).sort_values("Probability", ascending=False)
    if top_n is not None:
        frame = frame.head(top_n)
    return frame.reset_index(drop=True)


def contribution_frame(explanations: dict[str, float], top_n: int = 12) -> pd.DataFrame:
    frame = pd.DataFrame(
        [{"Feature": prettify_feature_name(feature), "Contribution": float(value)} for feature, value in explanations.items()]
    )
    frame["Direction"] = frame["Contribution"].apply(lambda value: "Raises risk" if value >= 0 else "Lowers risk")
    frame["AbsContribution"] = frame["Contribution"].abs()
    frame = frame.sort_values("AbsContribution", ascending=False).head(top_n)
    return frame.sort_values("Contribution").reset_index(drop=True)


def probability_chart(frame: pd.DataFrame, highlight_label: str | None = None) -> None:
    chart_frame = frame.copy()
    chart_frame["Group"] = "Other"
    if highlight_label is not None:
        chart_frame.loc[chart_frame["Label"] == highlight_label, "Group"] = "Focus"
    chart = (
        alt.Chart(chart_frame)
        .mark_bar(cornerRadiusEnd=8)
        .encode(
            x=alt.X("Probability:Q", axis=alt.Axis(format="%"), title="Probability"),
            y=alt.Y("Label:N", sort="-x", title=None),
            color=alt.Color("Group:N", scale=alt.Scale(domain=["Focus", "Other"], range=["#c05621", "#0f766e"]), legend=None),
            tooltip=[alt.Tooltip("Label:N"), alt.Tooltip("Probability:Q", format=".2%")],
        )
        .properties(height=max(260, 42 * len(chart_frame)))
    )
    st.altair_chart(chart, use_container_width=True)


def contribution_chart(frame: pd.DataFrame) -> None:
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=8)
        .encode(
            x=alt.X("Contribution:Q", title="Feature contribution"),
            y=alt.Y("Feature:N", sort=None, title=None),
            color=alt.Color("Direction:N", scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=["#c05621", "#0f766e"]), legend=None),
            tooltip=[alt.Tooltip("Feature:N"), alt.Tooltip("Contribution:Q", format=".4f"), alt.Tooltip("Direction:N")],
        )
        .properties(height=max(280, 36 * len(frame)))
    )
    st.altair_chart(chart, use_container_width=True)


def risk_band(probability: float, threshold: float = 0.5) -> tuple[str, str, str]:
    if probability >= max(threshold + 0.20, 0.75):
        return "High Priority Review", "high", "The score is materially above the active threshold."
    if probability >= threshold:
        return "Review Recommended", "medium", "The score crossed the threshold and should be reviewed carefully."
    return "Lower Model Risk", "low", "The score is below the active decision threshold."


def render_html_card(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


def render_page_banner(title: str, eyebrow: str, subtitle: str) -> None:
    render_html_card(
        f"""
        <div class="page-banner">
          <div class="hero-kicker">{escape(eyebrow)}</div>
          <div class="page-title">{escape(title)}</div>
          <div class="page-copy">{escape(subtitle)}</div>
        </div>
        """
    )


def render_stat_card(title: str, value: str, caption: str, tone: str = "default") -> None:
    render_html_card(
        f"""
        <div class="soft-card stat-card {escape(tone)}">
          <div class="card-label">{escape(title)}</div>
          <div class="card-value">{escape(value)}</div>
          <div class="card-copy">{escape(caption)}</div>
        </div>
        """
    )


def render_training_hint(module: str, command: str) -> None:
    spec = get_module_spec(module)
    render_html_card(
        f"""
        <div class="soft-card warning-card">
          <div class="card-label">Artifacts not available</div>
          <div class="card-value">{escape(spec.display_name)}</div>
          <div class="card-copy">Train this module first, then reload the page.</div>
        </div>
        """
    )
    st.code(command, language="powershell")


def render_probability_table(frame: pd.DataFrame) -> None:
    st.dataframe(frame.style.format({"Probability": "{:.2%}"}), hide_index=True, use_container_width=True)


def render_feature_table(values: dict[str, Any]) -> None:
    frame = pd.DataFrame([{"Field": prettify_feature_name(key), "Value": value} for key, value in values.items()])
    st.dataframe(frame, hide_index=True, use_container_width=True)


def render_prediction_downloads(result: dict[str, Any]) -> None:
    report_paths = result.get("report_paths", {})
    if not report_paths:
        return
    json_path = Path(report_paths["json"])
    html_path = Path(report_paths["html"])
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download JSON Report",
            data=json_path.read_bytes(),
            file_name=json_path.name,
            mime="application/json",
            use_container_width=True,
        )
    with right:
        st.download_button(
            "Download HTML Report",
            data=html_path.read_bytes(),
            file_name=html_path.name,
            mime="text/html",
            use_container_width=True,
        )


def render_result_banner(module: str, primary_probability: float, threshold: float, decision: str, model_version: str, note: str) -> None:
    band_label, band_tone, band_copy = risk_band(primary_probability, threshold)
    render_html_card(
        f"""
        <div class="result-banner">
          <div>
            <div class="hero-kicker">{escape(get_module_spec(module).display_name)} Summary</div>
            <div class="result-title">{escape(decision)}</div>
            <div class="page-copy">{escape(note)}</div>
          </div>
          <div class="result-side">
            <span class="risk-chip {escape(band_tone)}">{escape(band_label)}</span>
            <div class="result-meta">Model version: {escape(model_version)}</div>
            <div class="result-meta">{escape(band_copy)}</div>
          </div>
        </div>
        """
    )


def render_chip_row(title: str, values: list[str], empty_text: str) -> None:
    if not values:
        render_html_card(
            f"""
            <div class="soft-card">
              <div class="card-label">{escape(title)}</div>
              <div class="card-copy">{escape(empty_text)}</div>
            </div>
            """
        )
        return
    chips = "".join([f"<span class='tag-chip'>{escape(value)}</span>" for value in values])
    render_html_card(
        f"""
        <div class="soft-card">
          <div class="card-label">{escape(title)}</div>
          <div class="chip-row">{chips}</div>
        </div>
        """
    )


def render_module_card(module: str, summary: str, modality: str) -> None:
    spec = get_module_spec(module)
    status = "Ready" if module_ready(module) else "Pending"
    status_class = "ready" if status == "Ready" else "pending"
    metric_label, metric_value = headline_metric(module)
    render_html_card(
        f"""
        <div class="module-card">
          <div class="card-topline">
            <span class="status-chip {status_class}">{escape(status)}</span>
            <span class="module-tag">{escape(modality)}</span>
          </div>
          <div class="module-title">{escape(spec.display_name)}</div>
          <div class="module-copy">{escape(summary)}</div>
          <div class="module-grid">
            <div>
              <div class="card-label">{escape(metric_label)}</div>
              <div class="card-value small">{escape(metric_value)}</div>
            </div>
            <div>
              <div class="card-label">Selected model</div>
              <div class="card-value small">{escape(selected_model_label(module))}</div>
            </div>
          </div>
        </div>
        """
    )
