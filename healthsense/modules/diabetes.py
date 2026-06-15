from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from healthsense.config import ModuleSpec
from healthsense.explainability import compute_tree_shap_values
from healthsense.metrics import compute_binary_metrics, optimize_binary_threshold, plot_binary_evaluation, save_json
from healthsense.modules.base import BaseModule
from healthsense.modules.tabular_common import (
    build_binary_candidates,
    build_catboost_candidate,
    select_best_binary_model,
    split_binary_frame,
)


class DiabetesModule(BaseModule):
    name = "diabetes"

    def _load_frame(self, spec: ModuleSpec) -> pd.DataFrame:
        frame = pd.read_csv(spec.dataset_root / "diabetes.csv")
        invalid_zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for column in invalid_zero_columns:
            frame.loc[frame[column] == 0, column] = pd.NA
        return frame

    def train(self, spec: ModuleSpec, mode: str, **kwargs) -> dict[str, Any]:
        artifact_dir = spec.artifact_dir
        plots_dir = artifact_dir / "plots"
        data_dir = artifact_dir / "data"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        frame = self._load_frame(spec)
        train, val, test = split_binary_frame(
            frame,
            spec.target_column or "Outcome",
            smoke_rows=spec.smoke_rows if mode == "smoke" else None,
        )

        numeric_columns = [column for column in frame.columns if column != spec.target_column]
        categorical_columns: list[str] = []
        class_weight_scale = float((train[spec.target_column] == 0).sum() / max((train[spec.target_column] == 1).sum(), 1))

        candidates = build_binary_candidates(numeric_columns, categorical_columns, class_weight_scale)
        catboost = build_catboost_candidate(categorical_columns)
        best_name, best_model, candidate_scores, calibrator = select_best_binary_model(
            candidates=candidates,
            train_frame=train[numeric_columns],
            train_target=train[spec.target_column],
            val_frame=val[numeric_columns],
            val_target=val[spec.target_column],
            catboost_candidate=catboost,
            categorical_columns=categorical_columns,
            output_dir=artifact_dir / "candidates",
        )

        val_raw = best_model.predict_proba(val[numeric_columns])[:, 1]
        test_raw = best_model.predict_proba(test[numeric_columns])[:, 1]
        val_prob = calibrator.predict_proba(val_raw)
        test_prob = calibrator.predict_proba(test_raw)
        threshold = optimize_binary_threshold(val[spec.target_column].to_numpy(), val_prob)
        metrics = compute_binary_metrics(test[spec.target_column].to_numpy(), test_prob, threshold)
        plots = plot_binary_evaluation(test[spec.target_column].to_numpy(), test_prob, threshold, plots_dir, spec.display_name)

        config_payload = {
            "module": spec.name,
            "display_name": spec.display_name,
            "task_type": spec.task_type,
            "selected_model": best_name,
            "feature_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "threshold": threshold,
            "labels": spec.labels,
            "mode": mode,
        }
        joblib.dump(best_model, artifact_dir / "selected_model.joblib")
        joblib.dump(calibrator, artifact_dir / "calibrator.joblib")
        joblib.dump(train[numeric_columns].sample(min(128, len(train)), random_state=42), artifact_dir / "background_frame.joblib")
        train.to_csv(data_dir / "train.csv", index=False)
        val.to_csv(data_dir / "val.csv", index=False)
        test.to_csv(data_dir / "test.csv", index=False)
        save_json({"candidate_scores": candidate_scores, "metrics": metrics, "plots": {k: str(v) for k, v in plots.items()}}, artifact_dir / "metrics.json")
        save_json(config_payload, artifact_dir / "config.json")
        return {"module": spec.name, "metrics": metrics, "selected_model": best_name, "artifact_dir": str(artifact_dir)}

    def evaluate(self, spec: ModuleSpec) -> dict[str, Any]:
        metrics_path = spec.artifact_dir / "metrics.json"
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    def load_bundle(self, spec: ModuleSpec) -> dict[str, Any]:
        config = json.loads((spec.artifact_dir / "config.json").read_text(encoding="utf-8"))
        return {
            "model": joblib.load(spec.artifact_dir / "selected_model.joblib"),
            "calibrator": joblib.load(spec.artifact_dir / "calibrator.joblib"),
            "background": joblib.load(spec.artifact_dir / "background_frame.joblib"),
            "config": config,
            "metrics": json.loads((spec.artifact_dir / "metrics.json").read_text(encoding="utf-8")),
        }

    def predict(self, spec: ModuleSpec, bundle: dict[str, Any], **kwargs) -> dict[str, Any]:
        feature_columns = bundle["config"]["feature_columns"]
        frame = pd.DataFrame([{column: kwargs[column] for column in feature_columns}])
        raw_prob = bundle["model"].predict_proba(frame)[:, 1]
        probability = float(bundle["calibrator"].predict_proba(raw_prob)[0])
        threshold = float(bundle["config"]["threshold"])
        explanations = compute_tree_shap_values(bundle["model"], frame, bundle["background"])
        return {
            "module": spec.name,
            "probabilities": {spec.labels[1]: probability, spec.labels[0]: 1 - probability},
            "primary_decision": spec.labels[1] if probability >= threshold else spec.labels[0],
            "threshold": threshold,
            "explanations": explanations,
            "model_version": bundle["config"]["selected_model"],
            "metadata": {"feature_values": frame.iloc[0].to_dict()},
        }
