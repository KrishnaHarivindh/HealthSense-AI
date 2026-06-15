from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from healthsense.config import get_module_spec
from healthsense.reporting import build_session_report_html
from healthsense.service import evaluate_module, load_module_bundle, predict_module
from healthsense.storage import create_session, get_session_predictions, init_storage, list_experiments, list_prediction_history

from term_guides import TERM_GUIDES
from ui_helpers import (
    contribution_chart,
    contribution_frame,
    format_metric,
    format_probability,
    load_app_styles,
    load_metrics_payload,
    module_ready,
    probability_chart,
    probability_frame,
    render_chip_row,
    render_feature_table,
    render_html_card,
    render_page_banner,
    render_prediction_downloads,
    render_probability_table,
    render_result_banner,
    render_stat_card,
    render_training_hint,
    summarize_metrics,
)


st.set_page_config(page_title="HealthSense AI v2", page_icon=":hospital:", layout="wide", initial_sidebar_state="collapsed")

MODULES = ["chest", "skin", "diabetes", "heart"]
PAGE_OPTIONS = ["Chest X-ray", "Skin Cancer", "Diabetes", "Heart Disease", "Evaluation Lab", "History"]
HEART_AGE_OPTIONS = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"]
HEART_RACE_OPTIONS = ["White", "Black", "Asian", "American Indian/Alaskan Native", "Other", "Hispanic"]
HEART_DIABETIC_OPTIONS = ["No", "Yes", "No, borderline diabetes", "Yes (during pregnancy)"]
LOCALIZATION_OPTIONS = ["back", "lower extremity", "trunk", "upper extremity", "abdomen", "face", "chest", "foot", "hand", "neck", "scalp", "ear", "genital", "acral", "unknown"]
TRAIN_COMMANDS = {
    "chest": r".\.venv\Scripts\python -m healthsense.train --module chest --mode smoke",
    "skin": r".\.venv\Scripts\python -m healthsense.train --module skin --mode smoke",
    "diabetes": r".\.venv\Scripts\python -m healthsense.train --module diabetes --mode full",
    "heart": r".\.venv\Scripts\python -m healthsense.train --module heart --mode full",
}
MODULE_UI = {
    "chest": {
        "eyebrow": "Image + metadata screening",
        "summary": "ChestXray14 multilabel screening with a pneumonia-focused summary, confidence table, and Grad-CAM overlay.",
        "modality": "X-ray image + demographics",
    },
    "skin": {
        "eyebrow": "Dermoscopy triage",
        "summary": "HAM10000 7-class lesion screening with malignant-risk proxy, explainability heatmap, and ranked scores.",
        "modality": "Dermoscopic image + lesion context",
    },
    "diabetes": {
        "eyebrow": "Clinical risk estimation",
        "summary": "Structured diabetes prediction with calibrated probabilities and feature-level explanation for every case.",
        "modality": "Tabular clinical measurements",
    },
    "heart": {
        "eyebrow": "Population health risk model",
        "summary": "Heart disease risk estimation from demographic and lifestyle factors with CatBoost explanations.",
        "modality": "Tabular lifestyle + health survey",
    },
}


def load_bundle_safe(module: str):
    try:
        return load_module_bundle(module)
    except FileNotFoundError:
        return None


def render_bmi_and_score_guide(prefix: str, include_health_scores: bool) -> None:
    with st.expander("BMI and score guide", expanded=False):
        guide_left, guide_right = st.columns([0.95, 1.05])
        with guide_left:
            st.markdown("#### BMI calculator")
            height_cm = st.number_input(
                "Height (cm)",
                min_value=80.0,
                max_value=250.0,
                value=170.0,
                step=1.0,
                key=f"{prefix}_height_cm",
            )
            weight_kg = st.number_input(
                "Weight (kg)",
                min_value=20.0,
                max_value=300.0,
                value=70.0,
                step=0.5,
                key=f"{prefix}_weight_kg",
            )
            height_m = height_cm / 100.0
            bmi_value = weight_kg / max(height_m * height_m, 1e-6)
            if bmi_value < 18.5:
                bmi_band = "Underweight"
            elif bmi_value < 25:
                bmi_band = "Normal range"
            elif bmi_value < 30:
                bmi_band = "Overweight"
            else:
                bmi_band = "Obesity range"
            render_stat_card("Calculated BMI", f"{bmi_value:.1f}", f"Formula: weight (kg) / height (m)^2. Category: {bmi_band}.")
        with guide_right:
            st.markdown("#### What these fields mean")
            st.caption("`BMI` is Body Mass Index. You can calculate it here and copy the value into the BMI field.")
            if include_health_scores:
                st.caption(
                    "`PhysicalHealth` means the number of days in the last 30 days when physical health was not good. "
                    "Count days with illness, pain, injury, fatigue, or other physical problems."
                )
                st.caption(
                    "`MentalHealth` means the number of days in the last 30 days when mental health was not good. "
                    "Count days with stress, anxiety, depression, emotional strain, or poor mental wellbeing."
                )
                st.caption(
                    "Both scores are usually entered as whole numbers from `0` to `30`."
                    " Example: if someone felt mentally unwell on 6 days last month, enter `6`."
                )


def render_term_guide(page_key: str) -> None:
    guide = TERM_GUIDES.get(page_key, {})
    if not guide:
        return
    with st.expander("What these fields and terms mean", expanded=False):
        for section_title, section_terms in guide.items():
            st.markdown(f"#### {section_title}")
            for term, meaning in section_terms.items():
                st.markdown(f"**{term}**: {meaning}")


def render_top_navigation() -> tuple[str, int | None]:
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
        st.session_state.current_session_name = None

    render_html_card(
        """
        <div class="top-shell">
          <div class="top-title">HealthSense AI v2</div>
          <div class="top-copy">Select the analysis you want from the top bar and keep the rest of the interface out of the way.</div>
        </div>
        """
    )
    page = st.radio("Select analysis", options=PAGE_OPTIONS, horizontal=True, label_visibility="collapsed")

    if st.session_state.current_session_id:
        rows = get_session_predictions(int(st.session_state.current_session_id))
        session_text = f"Session #{int(st.session_state.current_session_id)} · {st.session_state.current_session_name} · {len(rows)} saved predictions"
    else:
        session_text = "No active session. Create one only if you want to combine predictions into a single report."

    render_html_card(f"<div class='compact-note'>{escape(session_text)}</div>")

    with st.expander("Session and workspace options", expanded=False):
        with st.form("session_form", clear_on_submit=False):
            left, right = st.columns([1.4, 0.8])
            with left:
                session_name = st.text_input(
                    "Session name",
                    value=st.session_state.current_session_name or "",
                    placeholder="Example: Demo Ward Round",
                )
            with right:
                submitted = st.form_submit_button("Create / Switch Session", use_container_width=True)
        if submitted and session_name.strip():
            session_id = create_session(session_name.strip())
            st.session_state.current_session_id = session_id
            st.session_state.current_session_name = session_name.strip()

        status_markup = "".join(
            [
                f"<span class='status-chip {'ready' if module_ready(module) else 'pending'}'>{escape(get_module_spec(module).display_name)}</span>"
                for module in MODULES
            ]
        )
        render_html_card(f"<div class='chip-row'>{status_markup}</div>")

    return page, st.session_state.current_session_id


def render_image_module_page(module: str, session_id: int | None) -> None:
    bundle = load_bundle_safe(module)
    render_page_banner(
        get_module_spec(module).display_name,
        MODULE_UI[module]["eyebrow"],
        MODULE_UI[module]["summary"],
    )
    render_term_guide(module)
    if bundle is None:
        render_training_hint(module, TRAIN_COMMANDS[module])
        return

    left, right = st.columns([1.1, 0.9])
    uploader_label = "Upload chest X-ray image" if module == "chest" else "Upload lesion image"
    uploader_types = ["png", "jpg", "jpeg"]
    uploader_key = f"{module}_uploader"

    with left:
        st.markdown("### Image Input")
        uploaded = st.file_uploader(uploader_label, type=uploader_types, key=uploader_key)
        if uploaded is not None:
            preview = Image.open(uploaded).convert("RGB")
            st.image(preview, caption="Uploaded image", use_container_width=True)
            if hasattr(uploaded, "seek"):
                uploaded.seek(0)
        else:
            render_html_card(
                """
                <div class="soft-card empty-state">
                  <div class="card-label">Awaiting image</div>
                  <div class="card-copy">Upload a clear study image to unlock the guided prediction flow and visual explanation output.</div>
                </div>
                """
            )

    with right:
        st.markdown("### Case Context")
        form_name = f"{module}_form"
        with st.form(form_name, clear_on_submit=False):
            if module == "chest":
                age = st.slider("Patient age", 1, 100, 45)
                gender = st.selectbox("Patient gender", ["M", "F", "Unknown"])
                view_position = st.selectbox("View position", ["PA", "AP", "Unknown"])
                follow_up = st.number_input("Follow-up count", min_value=0, max_value=20, value=0)
            else:
                age = st.number_input("Age", min_value=0, max_value=120, value=45)
                gender = st.selectbox("Sex", ["male", "female", "unknown"])
                view_position = st.selectbox("Localization", LOCALIZATION_OPTIONS)
                follow_up = None
            submitted = st.form_submit_button(f"Run {get_module_spec(module).display_name.lower()} diagnosis", type="primary", use_container_width=True)

        if submitted:
            if uploaded is None:
                st.error("Upload an image first.")
            else:
                if hasattr(uploaded, "seek"):
                    uploaded.seek(0)
                image = Image.open(uploaded).convert("RGB")
                if module == "chest":
                    result = predict_module(
                        module,
                        bundle=bundle,
                        session_id=session_id,
                        image=image,
                        patient_age=age,
                        patient_gender=gender,
                        view_position=view_position,
                        follow_up=int(follow_up or 0),
                    )
                else:
                    result = predict_module(
                        module,
                        bundle=bundle,
                        session_id=session_id,
                        image=image,
                        age=int(age),
                        sex=gender,
                        localization=view_position,
                    )
                st.session_state[f"last_{module}_result"] = result
                st.session_state[f"last_{module}_image"] = image.copy()

    result = st.session_state.get(f"last_{module}_result")
    if not result:
        return

    if module == "chest":
        primary_probability = float(result["metadata"]["pneumonia_probability"])
        threshold = float(result["threshold"].get("Pneumonia", 0.5))
        note = f"Pneumonia score: {format_probability(primary_probability)}"
        focus_label = "Pneumonia"
        focus_values = result["metadata"]["predicted_labels"]
        empty_text = "No label crossed its tuned threshold."
        metric_cards = [
            ("Pneumonia probability", format_probability(primary_probability), "Calibrated score for the pneumonia label."),
            ("Active threshold", format_metric(threshold), "Threshold used for the pneumonia decision."),
            ("Predicted findings", str(len(focus_values)), "Labels crossing tuned thresholds."),
        ]
        overlay_caption = "Grad-CAM overlay for pneumonia"
    else:
        primary_probability = float(result["metadata"]["malignant_risk"])
        threshold = float(result["threshold"])
        note = f"Malignant-risk proxy: {format_probability(primary_probability)}"
        focus_label = result["primary_decision"]
        focus_values = [result["primary_decision"]]
        empty_text = "No class summary is available."
        metric_cards = [
            ("Top class", str(result["primary_decision"]), "Highest-probability lesion class."),
            ("Top probability", format_probability(result["metadata"]["top_probability"]), "Confidence for the leading class."),
            ("Malignant-risk proxy", format_probability(primary_probability), "Sum of selected malignant lesion probabilities."),
        ]
        overlay_caption = "Grad-CAM overlay for the top class"

    render_result_banner(module, primary_probability, threshold, result["primary_decision"], str(result["model_version"]), note)

    cols = st.columns(3)
    for column, (title, value, caption) in zip(cols, metric_cards, strict=False):
        with column:
            render_stat_card(title, value, caption)

    visual, scores = st.columns([1.05, 0.95])
    with visual:
        st.image(result["explanations"]["gradcam_overlay"], caption=overlay_caption, use_container_width=True)
        with st.expander("Original uploaded image", expanded=False):
            st.image(st.session_state.get(f"last_{module}_image"), caption="Uploaded image", use_container_width=True)
    with scores:
        render_chip_row("Decision highlights", focus_values, empty_text)
        probability_chart(probability_frame(result["probabilities"], top_n=8), highlight_label=focus_label)

    with st.expander("Detailed probability table", expanded=False):
        render_probability_table(probability_frame(result["probabilities"]))
    with st.expander("Reports", expanded=False):
        render_prediction_downloads(result)


def render_diabetes_page(session_id: int | None) -> None:
    render_page_banner("Diabetes Risk Module", MODULE_UI["diabetes"]["eyebrow"], MODULE_UI["diabetes"]["summary"])
    render_term_guide("diabetes")
    bundle = load_bundle_safe("diabetes")
    if bundle is None:
        render_training_hint("diabetes", TRAIN_COMMANDS["diabetes"])
        return

    render_bmi_and_score_guide("diabetes", include_health_scores=False)

    with st.form("diabetes_form", clear_on_submit=False):
        left, right = st.columns(2)
        with left:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
            glucose = st.number_input("Glucose", min_value=0, max_value=250, value=120)
            blood_pressure = st.number_input("BloodPressure", min_value=0, max_value=150, value=70)
            skin_thickness = st.number_input("SkinThickness", min_value=0, max_value=100, value=20)
        with right:
            insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
            bmi = st.number_input(
                "BMI",
                min_value=0.0,
                max_value=80.0,
                value=28.0,
                help="Body Mass Index. Use the BMI calculator above if needed.",
            )
            dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=3.0, value=0.47)
            age = st.number_input("Age", min_value=1, max_value=120, value=35)
        st.caption("The preprocessing pipeline handles medically implausible zero values for the relevant clinical fields.")
        submitted = st.form_submit_button("Run diabetes risk", type="primary", use_container_width=True)

    if submitted:
        st.session_state["last_diabetes_result"] = predict_module(
            "diabetes",
            bundle=bundle,
            session_id=session_id,
            Pregnancies=int(pregnancies),
            Glucose=float(glucose),
            BloodPressure=float(blood_pressure),
            SkinThickness=float(skin_thickness),
            Insulin=float(insulin),
            BMI=float(bmi),
            DiabetesPedigreeFunction=float(dpf),
            Age=float(age),
        )

    result = st.session_state.get("last_diabetes_result")
    if not result:
        return

    probability = float(result["probabilities"]["Diabetes"])
    threshold = float(result["threshold"])
    render_result_banner("diabetes", probability, threshold, result["primary_decision"], str(result["model_version"]), f"Calibrated diabetes probability: {format_probability(probability)}")
    cols = st.columns(3)
    with cols[0]:
        render_stat_card("Diabetes probability", format_probability(probability), "Calibrated probability for the positive class.")
    with cols[1]:
        render_stat_card("Decision threshold", format_metric(threshold), "Threshold selected on validation performance.")
    with cols[2]:
        render_stat_card("Primary decision", str(result["primary_decision"]), "Thresholded result returned by the pipeline.")

    frame = contribution_frame(result["explanations"], top_n=12)
    contribution_chart(frame)
    with st.expander("Feature contribution table", expanded=False):
        st.dataframe(frame[["Feature", "Contribution", "Direction"]], hide_index=True, use_container_width=True)
    with st.expander("Input values", expanded=False):
        render_feature_table(result["metadata"]["feature_values"])
    with st.expander("Reports", expanded=False):
        render_prediction_downloads(result)


def render_heart_page(session_id: int | None) -> None:
    render_page_banner("Heart Disease Risk Module", MODULE_UI["heart"]["eyebrow"], MODULE_UI["heart"]["summary"])
    render_term_guide("heart")
    bundle = load_bundle_safe("heart")
    if bundle is None:
        render_training_hint("heart", TRAIN_COMMANDS["heart"])
        return

    render_bmi_and_score_guide("heart", include_health_scores=True)

    with st.form("heart_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=80.0,
                value=26.0,
                help="Body Mass Index. Use the BMI calculator above if needed.",
            )
            smoking = st.selectbox("Smoking", ["Yes", "No"])
            alcohol = st.selectbox("AlcoholDrinking", ["Yes", "No"])
            stroke = st.selectbox("Stroke", ["Yes", "No"])
            physical_health = st.number_input(
                "PhysicalHealth",
                min_value=0,
                max_value=30,
                value=5,
                help="Number of days in the last 30 days when physical health was not good.",
            )
            mental_health = st.number_input(
                "MentalHealth",
                min_value=0,
                max_value=30,
                value=2,
                help="Number of days in the last 30 days when mental health was not good.",
            )
        with col2:
            diff_walking = st.selectbox("DiffWalking", ["Yes", "No"])
            sex = st.selectbox("Sex", ["Male", "Female"])
            age_category = st.selectbox("AgeCategory", HEART_AGE_OPTIONS)
            race = st.selectbox("Race", HEART_RACE_OPTIONS)
            diabetic = st.selectbox("Diabetic", HEART_DIABETIC_OPTIONS)
            physical_activity = st.selectbox("PhysicalActivity", ["Yes", "No"])
        with col3:
            gen_health = st.selectbox("GenHealth", ["Poor", "Fair", "Good", "Very good", "Excellent"])
            sleep_time = st.number_input("SleepTime", min_value=1.0, max_value=24.0, value=7.0)
            asthma = st.selectbox("Asthma", ["Yes", "No"])
            kidney = st.selectbox("KidneyDisease", ["Yes", "No"])
            skin_cancer = st.selectbox("SkinCancer", ["Yes", "No"])
        submitted = st.form_submit_button("Run heart disease risk", type="primary", use_container_width=True)

    if submitted:
        st.session_state["last_heart_result"] = predict_module(
            "heart",
            bundle=bundle,
            session_id=session_id,
            BMI=float(bmi),
            Smoking=smoking,
            AlcoholDrinking=alcohol,
            Stroke=stroke,
            PhysicalHealth=float(int(physical_health)),
            MentalHealth=float(int(mental_health)),
            DiffWalking=diff_walking,
            Sex=sex,
            AgeCategory=age_category,
            Race=race,
            Diabetic=diabetic,
            PhysicalActivity=physical_activity,
            GenHealth=gen_health,
            SleepTime=float(sleep_time),
            Asthma=asthma,
            KidneyDisease=kidney,
            SkinCancer=skin_cancer,
        )

    result = st.session_state.get("last_heart_result")
    if not result:
        return

    probability = float(result["probabilities"]["Heart Disease"])
    threshold = float(result["threshold"])
    render_result_banner("heart", probability, threshold, result["primary_decision"], str(result["model_version"]), f"Calibrated heart-disease probability: {format_probability(probability)}")
    cols = st.columns(3)
    with cols[0]:
        render_stat_card("Heart disease probability", format_probability(probability), "Calibrated probability for the positive class.")
    with cols[1]:
        render_stat_card("Decision threshold", format_metric(threshold), "Threshold selected on validation performance.")
    with cols[2]:
        render_stat_card("Primary decision", str(result["primary_decision"]), "Thresholded result returned by the pipeline.")

    frame = contribution_frame(result["explanations"], top_n=14)
    contribution_chart(frame)
    with st.expander("Feature contribution table", expanded=False):
        st.dataframe(frame[["Feature", "Contribution", "Direction"]], hide_index=True, use_container_width=True)
    with st.expander("Input values", expanded=False):
        render_feature_table(result["metadata"]["feature_values"])
    with st.expander("Reports", expanded=False):
        render_prediction_downloads(result)


def render_evaluation_page() -> None:
    render_page_banner("Evaluation Lab", "Metrics, plots, and ablation review", "Inspect saved metrics, plots, candidate scores, and experiment history from the unified training pipeline.")
    render_term_guide("evaluation")
    module = st.selectbox("Module", MODULES, format_func=lambda name: get_module_spec(name).display_name)
    payload = load_metrics_payload(module)
    if not payload:
        render_training_hint(module, TRAIN_COMMANDS[module])
        return

    metrics = payload.get("metrics", {})
    cols = st.columns(4)
    for column, (label, value) in zip(cols, summarize_metrics(module, metrics), strict=False):
        with column:
            render_stat_card(label, value, "Saved evaluation metric")

    candidate_scores = payload.get("candidate_scores", {})
    if candidate_scores:
        with st.expander("Model comparison", expanded=False):
            rows = []
            for name, score in candidate_scores.items():
                if isinstance(score, dict):
                    for metric_name, metric_value in score.items():
                        rows.append({"Candidate": name, "Metric": metric_name, "Value": metric_value})
                else:
                    rows.append({"Candidate": name, "Metric": "score", "Value": score})
            frame = pd.DataFrame(rows)
            st.dataframe(frame.style.format({"Value": "{:.4f}"}), hide_index=True, use_container_width=True)

    artifact_dir = PROJECT_ROOT / "artifacts" / module / "plots"
    if artifact_dir.exists():
        with st.expander("Evaluation plots", expanded=False):
            plot_paths = sorted(artifact_dir.glob("*.png"))
            for index in range(0, len(plot_paths), 2):
                columns = st.columns(2)
                for offset, plot_path in enumerate(plot_paths[index : index + 2]):
                    with columns[offset]:
                        st.image(str(plot_path), caption=plot_path.stem.replace("_", " ").title(), use_container_width=True)

    with st.expander("Raw metrics payload", expanded=False):
        st.json(evaluate_module(module))

    experiments = list_experiments(limit=50)
    filtered = experiments[experiments["module"] == module] if not experiments.empty else experiments
    if not filtered.empty:
        with st.expander("Experiment history", expanded=False):
            st.dataframe(filtered, hide_index=True, use_container_width=True)


def render_history_page() -> None:
    render_page_banner("Prediction History", "Saved cases and combined reports", "Review saved predictions, filter by session or module, and build a combined HTML session report.")
    render_term_guide("history")
    history = list_prediction_history(limit=100)
    if history.empty:
        st.info("No predictions have been recorded yet.")
        return

    def simplify(row: pd.Series) -> dict[str, str]:
        prediction = json.loads(row["prediction_json"])
        return {
            "ID": str(row["id"]),
            "Session": row["session_name"] or "No session",
            "Module": get_module_spec(row["module"]).display_name,
            "Created": row["created_at"],
            "Decision": str(prediction.get("primary_decision")),
            "Model": str(row["model_version"]),
        }

    table = pd.DataFrame([simplify(row) for _, row in history.iterrows()])
    filters = st.columns([1.0, 1.2, 0.8])
    with filters[0]:
        module_filter = st.selectbox("Filter by module", ["All"] + [get_module_spec(module).display_name for module in MODULES])
    with filters[1]:
        session_filter = st.text_input("Search session or decision", placeholder="Type a session name or decision")
    with filters[2]:
        session_default = int(st.session_state.get("current_session_id") or 1)
        session_id = st.number_input("Session ID for report", min_value=1, value=session_default, step=1)

    if module_filter != "All":
        table = table[table["Module"] == module_filter]
    if session_filter.strip():
        token = session_filter.strip().lower()
        table = table[
            table["Session"].astype(str).str.lower().str.contains(token)
            | table["Decision"].astype(str).str.lower().str.contains(token)
        ]

    st.dataframe(table, hide_index=True, use_container_width=True)
    with st.expander("Combined session report", expanded=False):
        if st.button("Build session report", use_container_width=True):
            session_predictions = get_session_predictions(int(session_id))
            if session_predictions.empty:
                st.warning("No predictions found for that session.")
            else:
                items = []
                for _, row in session_predictions.iterrows():
                    payload = json.loads(row["prediction_json"])
                    payload["module"] = row["module"]
                    items.append(payload)
                html = build_session_report_html(f"Session {session_id}", items)
                st.download_button(
                    "Download session report",
                    data=html,
                    file_name=f"session_{session_id}_report.html",
                    mime="text/html",
                    use_container_width=True,
                )


def main() -> None:
    init_storage()
    load_app_styles(PROJECT_ROOT)
    page, session_id = render_top_navigation()
    if page == "Chest X-ray":
        render_image_module_page("chest", session_id)
    elif page == "Skin Cancer":
        render_image_module_page("skin", session_id)
    elif page == "Diabetes":
        render_diabetes_page(session_id)
    elif page == "Heart Disease":
        render_heart_page(session_id)
    elif page == "Evaluation Lab":
        render_evaluation_page()
    elif page == "History":
        render_history_page()

    st.markdown(
        """
        <div class="footer-note">
          Academic decision-support demo only. Predictions should be reviewed by a qualified clinician before any real-world use.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
