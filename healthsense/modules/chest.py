from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier

from healthsense.config import CACHE_ROOT, CHEST_LABELS, ModuleSpec
from healthsense.explainability import generate_gradcam, overlay_heatmap
from healthsense.metrics import compute_multilabel_metrics, optimize_multilabel_thresholds, plot_multilabel_evaluation, save_json
from healthsense.modules.base import BaseModule
from healthsense.modules.vision_common import (
    FusionVisionModel,
    build_dataloaders,
    build_metadata_processor,
    fit_vision_model,
    get_device,
    get_image_transforms,
    load_vision_bundle,
    predict_vision_probabilities,
    save_vision_bundle,
    to_dense_float32,
)


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "Image Index": "image_index",
            "Finding Labels": "finding_labels",
            "Follow-up #": "follow_up",
            "Patient ID": "patient_id",
            "Patient Age": "patient_age",
            "Patient Gender": "patient_gender",
            "View Position": "view_position",
        }
    )


class ChestModule(BaseModule):
    name = "chest"

    def _build_image_index(self, spec: ModuleSpec) -> dict[str, str]:
        cache_file = CACHE_ROOT / "chest_image_index.joblib"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.exists():
            return joblib.load(cache_file)
        image_map = {}
        for image_path in spec.dataset_root.glob("images_*/images/*.png"):
            image_map[image_path.name] = str(image_path.resolve())
        joblib.dump(image_map, cache_file)
        return image_map

    def _load_frame(self, spec: ModuleSpec) -> pd.DataFrame:
        frame = pd.read_csv(spec.dataset_root / "Data_Entry_2017.csv")
        frame = _clean_columns(frame)
        image_map = self._build_image_index(spec)
        frame["image_path"] = frame["image_index"].map(image_map)
        frame = frame.dropna(subset=["image_path"]).copy()
        frame["patient_age"] = pd.to_numeric(frame["patient_age"], errors="coerce").fillna(0).clip(lower=0)
        frame["follow_up"] = pd.to_numeric(frame["follow_up"], errors="coerce").fillna(0).clip(lower=0)
        frame["patient_gender"] = frame["patient_gender"].fillna("Unknown")
        frame["view_position"] = frame["view_position"].fillna("Unknown")
        label_matrix = []
        for labels in frame["finding_labels"].fillna("No Finding"):
            values = set(str(labels).split("|"))
            label_matrix.append([1 if label in values else 0 for label in CHEST_LABELS])
        frame["targets"] = label_matrix
        return frame

    def _split_frame(self, frame: pd.DataFrame, spec: ModuleSpec, mode: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_val_names = {line.strip() for line in (spec.dataset_root / "train_val_list.txt").read_text(encoding="utf-8").splitlines() if line.strip()}
        test_names = {line.strip() for line in (spec.dataset_root / "test_list.txt").read_text(encoding="utf-8").splitlines() if line.strip()}

        test = frame[frame["image_index"].isin(test_names)].copy()
        train_val = frame[frame["image_index"].isin(train_val_names)].copy()
        test_patient_ids = set(test["patient_id"].unique().tolist())
        train_val = train_val[~train_val["patient_id"].isin(test_patient_ids)].copy()

        patient_summary = train_val.groupby("patient_id").agg(any_positive=("targets", lambda rows: int(np.asarray(list(rows)).sum() > 0))).reset_index()
        positive_patients = patient_summary[patient_summary["any_positive"] == 1]["patient_id"].tolist()
        negative_patients = patient_summary[patient_summary["any_positive"] == 0]["patient_id"].tolist()
        rng = np.random.default_rng(42)
        rng.shuffle(positive_patients)
        rng.shuffle(negative_patients)
        val_patient_count = max(1, int(0.15 * len(patient_summary)))
        val_patients = set(positive_patients[: val_patient_count // 2] + negative_patients[: max(1, val_patient_count - (val_patient_count // 2))])

        val = train_val[train_val["patient_id"].isin(val_patients)].copy()
        train = train_val[~train_val["patient_id"].isin(val_patients)].copy()

        if mode == "smoke" and spec.smoke_rows is not None:
            train = train.sample(min(int(spec.smoke_rows * 0.65), len(train)), random_state=42)
            val = val.sample(min(int(spec.smoke_rows * 0.175), len(val)), random_state=42)
            test = test.sample(min(int(spec.smoke_rows * 0.175), len(test)), random_state=42)

        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    def train(self, spec: ModuleSpec, mode: str, **kwargs) -> dict[str, Any]:
        artifact_dir = spec.artifact_dir
        plots_dir = artifact_dir / "plots"
        data_dir = artifact_dir / "data"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        frame = self._load_frame(spec)
        train, val, test = self._split_frame(frame, spec, mode)
        metadata_columns = ["patient_age", "patient_gender", "view_position", "follow_up"]
        metadata_processor = build_metadata_processor(["patient_age", "follow_up"], ["patient_gender", "view_position"])
        numeric_meta = metadata_processor.fit_transform(train[metadata_columns]).astype(np.float32)
        metadata_dim = numeric_meta.shape[1]

        train_loader_image, val_loader_image, test_loader_image = build_dataloaders(
            train_frame=train,
            val_frame=val,
            test_frame=test,
            image_path_column="image_path",
            target_column="targets",
            metadata_processor=build_metadata_processor(["patient_age", "follow_up"], ["patient_gender", "view_position"]),
            metadata_columns=metadata_columns,
            image_size=spec.image_size,
            task_type="multilabel",
            batch_size=spec.batch_size,
        )
        train_loader_fusion, val_loader_fusion, test_loader_fusion = build_dataloaders(
            train_frame=train,
            val_frame=val,
            test_frame=test,
            image_path_column="image_path",
            target_column="targets",
            metadata_processor=build_metadata_processor(["patient_age", "follow_up"], ["patient_gender", "view_position"]),
            metadata_columns=metadata_columns,
            image_size=spec.image_size,
            task_type="multilabel",
            batch_size=spec.batch_size,
        )

        y_train = np.stack(train["targets"].to_list())
        pos_counts = y_train.sum(axis=0)
        neg_counts = len(train) - pos_counts
        pos_weight = np.where(pos_counts > 0, neg_counts / np.maximum(pos_counts, 1), 1.0)

        device = get_device()
        image_model = FusionVisionModel(spec.backbone or "densenet121", len(spec.labels), metadata_dim=0)
        image_model, image_history = fit_vision_model(
            image_model,
            train_loader_image,
            val_loader_image,
            task_type="multilabel",
            class_weights=pos_weight,
            epochs=spec.smoke_epochs if mode == "smoke" else spec.full_epochs,
            device=device,
            primary_metric=spec.primary_metric,
        )
        image_val_prob, image_val_true, _ = predict_vision_probabilities(image_model, val_loader_image, "multilabel", device)
        image_score = float(roc_auc_score(image_val_true, image_val_prob, average="macro"))

        fusion_model = FusionVisionModel(spec.backbone or "densenet121", len(spec.labels), metadata_dim=metadata_dim)
        fusion_model, fusion_history = fit_vision_model(
            fusion_model,
            train_loader_fusion,
            val_loader_fusion,
            task_type="multilabel",
            class_weights=pos_weight,
            epochs=spec.smoke_epochs if mode == "smoke" else spec.full_epochs,
            device=device,
            primary_metric=spec.primary_metric,
        )
        fusion_val_prob, fusion_val_true, _ = predict_vision_probabilities(fusion_model, val_loader_fusion, "multilabel", device)
        fusion_score = float(roc_auc_score(fusion_val_true, fusion_val_prob, average="macro"))

        meta_processor = build_metadata_processor(["patient_age", "follow_up"], ["patient_gender", "view_position"])
        train_meta = meta_processor.fit_transform(train[metadata_columns])
        val_meta = meta_processor.transform(val[metadata_columns])
        metadata_baseline = OneVsRestClassifier(LogisticRegression(max_iter=2000))
        metadata_baseline.fit(train_meta, np.stack(train["targets"].to_list()))
        metadata_score = float(roc_auc_score(np.stack(val["targets"].to_list()), metadata_baseline.predict_proba(val_meta), average="macro"))

        chosen_variant = "fusion" if fusion_score >= image_score else "image_only"
        chosen_model = fusion_model if chosen_variant == "fusion" else image_model
        chosen_loader = test_loader_fusion if chosen_variant == "fusion" else test_loader_image
        chosen_val_prob = fusion_val_prob if chosen_variant == "fusion" else image_val_prob
        thresholds = optimize_multilabel_thresholds(np.stack(val["targets"].to_list()), chosen_val_prob, spec.labels)
        test_prob, test_true, _ = predict_vision_probabilities(chosen_model, chosen_loader, "multilabel", device)
        metrics = compute_multilabel_metrics(test_true, test_prob, thresholds, spec.labels)
        plots = plot_multilabel_evaluation(test_true, test_prob, spec.labels, plots_dir, spec.display_name)

        chosen_processor = build_metadata_processor(["patient_age", "follow_up"], ["patient_gender", "view_position"])
        save_vision_bundle(
            chosen_model,
            chosen_processor.fit(train[metadata_columns]),
            artifact_dir,
            history=fusion_history if chosen_variant == "fusion" else image_history,
            config_payload={
                "module": spec.name,
                "display_name": spec.display_name,
                "selected_variant": chosen_variant,
                "backbone": spec.backbone,
                "labels": spec.labels,
                "metadata_columns": metadata_columns,
                "image_size": list(spec.image_size),
                "thresholds": thresholds,
                "mode": mode,
                "metadata_dim": metadata_dim if chosen_variant == "fusion" else 0,
            },
        )
        torch.save(
            {"image_only": image_model.state_dict(), "fusion": fusion_model.state_dict()},
            artifact_dir / "ablation_models.pt",
        )
        joblib.dump(metadata_baseline, artifact_dir / "metadata_only_baseline.joblib")
        train.to_csv(data_dir / "train.csv", index=False)
        val.to_csv(data_dir / "val.csv", index=False)
        test.to_csv(data_dir / "test.csv", index=False)
        save_json(
            {
                "candidate_scores": {
                    "image_only_macro_roc_auc": image_score,
                    "fusion_macro_roc_auc": fusion_score,
                    "metadata_only_macro_roc_auc": metadata_score,
                },
                "metrics": metrics,
                "plots": {k: str(v) for k, v in plots.items()},
            },
            artifact_dir / "metrics.json",
        )
        return {
            "module": spec.name,
            "selected_variant": chosen_variant,
            "metrics": metrics,
            "artifact_dir": str(artifact_dir),
        }

    def evaluate(self, spec: ModuleSpec) -> dict[str, Any]:
        return json.loads((spec.artifact_dir / "metrics.json").read_text(encoding="utf-8"))

    def load_bundle(self, spec: ModuleSpec) -> dict[str, Any]:
        vision_bundle = load_vision_bundle(spec.artifact_dir)
        config = vision_bundle["config"]
        metadata_dim = int(config["metadata_dim"])
        model = FusionVisionModel(config["backbone"], len(config["labels"]), metadata_dim=metadata_dim)
        state_dict = torch.load(spec.artifact_dir / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return {
            "model": model,
            "metadata_processor": vision_bundle["metadata_processor"],
            "history": vision_bundle["history"],
            "config": config,
            "metrics": json.loads((spec.artifact_dir / "metrics.json").read_text(encoding="utf-8")),
        }

    def predict(self, spec: ModuleSpec, bundle: dict[str, Any], **kwargs) -> dict[str, Any]:
        image = kwargs["image"]
        metadata_frame = pd.DataFrame(
            [
                {
                    "patient_age": kwargs.get("patient_age", 0),
                    "patient_gender": kwargs.get("patient_gender", "Unknown"),
                    "view_position": kwargs.get("view_position", "Unknown"),
                    "follow_up": kwargs.get("follow_up", 0),
                }
            ]
        )
        metadata_tensor = torch.tensor(to_dense_float32(bundle["metadata_processor"].transform(metadata_frame)))
        tensor_image = get_image_transforms(tuple(bundle["config"]["image_size"]), train=False)(image.convert("RGB")).unsqueeze(0)
        logits = bundle["model"](tensor_image, metadata_tensor if bundle["config"]["selected_variant"] == "fusion" else None)
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()[0]
        thresholds = bundle["config"]["thresholds"]
        label_probs = {label: float(probabilities[index]) for index, label in enumerate(spec.labels)}
        predicted_labels = [label for label, probability in label_probs.items() if probability >= thresholds[label]]
        target_index = spec.labels.index("Pneumonia")
        heatmap = generate_gradcam(
            bundle["model"],
            tensor_image,
            metadata_tensor if bundle["config"]["selected_variant"] == "fusion" else None,
            target_index=target_index,
        )
        overlay = overlay_heatmap(image, heatmap)
        return {
            "module": spec.name,
            "probabilities": label_probs,
            "primary_decision": ", ".join(predicted_labels) if predicted_labels else "No finding above threshold",
            "threshold": thresholds,
            "explanations": {
                "focus_label": "Pneumonia",
                "gradcam_overlay": overlay,
                "heatmap": heatmap.tolist(),
            },
            "model_version": bundle["config"]["selected_variant"],
            "metadata": {
                "predicted_labels": predicted_labels,
                "pneumonia_probability": label_probs["Pneumonia"],
            },
        }
