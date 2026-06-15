from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

from healthsense.config import CACHE_ROOT, ModuleSpec, SKIN_LABELS
from healthsense.explainability import generate_gradcam, overlay_heatmap
from healthsense.metrics import compute_multiclass_metrics, plot_multiclass_evaluation, save_json
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


class SkinModule(BaseModule):
    name = "skin"

    def _build_image_index(self, spec: ModuleSpec) -> dict[str, str]:
        cache_file = CACHE_ROOT / "skin_image_index.joblib"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.exists():
            return joblib.load(cache_file)
        image_map = {}
        for image_path in list((spec.dataset_root / "HAM10000_images_part_1").glob("*.jpg")) + list((spec.dataset_root / "HAM10000_images_part_2").glob("*.jpg")):
            image_map[image_path.stem] = str(image_path.resolve())
        joblib.dump(image_map, cache_file)
        return image_map

    def _load_frame(self, spec: ModuleSpec) -> pd.DataFrame:
        frame = pd.read_csv(spec.dataset_root / "HAM10000_metadata.csv")
        image_map = self._build_image_index(spec)
        frame["image_path"] = frame["image_id"].map(image_map)
        frame = frame.dropna(subset=["image_path"]).copy()
        frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
        frame["sex"] = frame["sex"].fillna("unknown")
        frame["localization"] = frame["localization"].fillna("unknown")
        label_to_index = {label: index for index, label in enumerate(SKIN_LABELS)}
        frame["target_index"] = frame["dx"].map(label_to_index)
        return frame

    def _split_frame(self, frame: pd.DataFrame, spec: ModuleSpec, mode: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        lesion_ids = frame["lesion_id"].unique()
        rng = np.random.default_rng(42)
        rng.shuffle(lesion_ids)
        test_cut = int(0.15 * len(lesion_ids))
        val_cut = int(0.15 * len(lesion_ids))
        test_ids = set(lesion_ids[:test_cut])
        val_ids = set(lesion_ids[test_cut : test_cut + val_cut])
        train_ids = set(lesion_ids[test_cut + val_cut :])

        train = frame[frame["lesion_id"].isin(train_ids)].copy()
        val = frame[frame["lesion_id"].isin(val_ids)].copy()
        test = frame[frame["lesion_id"].isin(test_ids)].copy()
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
        metadata_columns = ["age", "sex", "localization"]

        image_loader_train, image_loader_val, image_loader_test = build_dataloaders(
            train_frame=train,
            val_frame=val,
            test_frame=test,
            image_path_column="image_path",
            target_column="target_index",
            metadata_processor=build_metadata_processor(["age"], ["sex", "localization"]),
            metadata_columns=metadata_columns,
            image_size=spec.image_size,
            task_type="multiclass",
            batch_size=spec.batch_size,
        )
        fusion_loader_train, fusion_loader_val, fusion_loader_test = build_dataloaders(
            train_frame=train,
            val_frame=val,
            test_frame=test,
            image_path_column="image_path",
            target_column="target_index",
            metadata_processor=build_metadata_processor(["age"], ["sex", "localization"]),
            metadata_columns=metadata_columns,
            image_size=spec.image_size,
            task_type="multiclass",
            batch_size=spec.batch_size,
        )

        class_counts = train["target_index"].value_counts().sort_index()
        class_weights = np.asarray([len(train) / max(class_counts.get(index, 1), 1) for index in range(len(spec.labels))], dtype=np.float32)
        device = get_device()

        image_model = FusionVisionModel(spec.backbone or "efficientnet_b0", len(spec.labels), metadata_dim=0)
        image_model, image_history = fit_vision_model(
            image_model,
            image_loader_train,
            image_loader_val,
            task_type="multiclass",
            class_weights=class_weights,
            epochs=spec.smoke_epochs if mode == "smoke" else spec.full_epochs,
            device=device,
            primary_metric=spec.primary_metric,
        )
        image_val_prob, image_val_true, _ = predict_vision_probabilities(image_model, image_loader_val, "multiclass", device)
        image_score = float(f1_score(image_val_true, np.argmax(image_val_prob, axis=1), average="macro", zero_division=0))

        meta_processor = build_metadata_processor(["age"], ["sex", "localization"])
        train_meta = meta_processor.fit_transform(train[metadata_columns])
        val_meta = meta_processor.transform(val[metadata_columns])
        metadata_baseline = CatBoostClassifier(
            iterations=350,
            depth=7,
            learning_rate=0.05,
            loss_function="MultiClass",
            verbose=False,
            random_seed=42,
        )
        metadata_baseline.fit(train_meta, train["target_index"])
        metadata_score = float(f1_score(val["target_index"], metadata_baseline.predict(val_meta), average="macro", zero_division=0))

        metadata_dim = train_meta.shape[1]
        fusion_model = FusionVisionModel(spec.backbone or "efficientnet_b0", len(spec.labels), metadata_dim=metadata_dim)
        fusion_model, fusion_history = fit_vision_model(
            fusion_model,
            fusion_loader_train,
            fusion_loader_val,
            task_type="multiclass",
            class_weights=class_weights,
            epochs=spec.smoke_epochs if mode == "smoke" else spec.full_epochs,
            device=device,
            primary_metric=spec.primary_metric,
        )
        fusion_val_prob, fusion_val_true, _ = predict_vision_probabilities(fusion_model, fusion_loader_val, "multiclass", device)
        fusion_score = float(f1_score(fusion_val_true, np.argmax(fusion_val_prob, axis=1), average="macro", zero_division=0))

        chosen_variant = "fusion" if fusion_score >= image_score else "image_only"
        chosen_model = fusion_model if chosen_variant == "fusion" else image_model
        chosen_loader = fusion_loader_test if chosen_variant == "fusion" else image_loader_test
        test_prob, test_true, _ = predict_vision_probabilities(chosen_model, chosen_loader, "multiclass", device)
        metrics = compute_multiclass_metrics(test_true, test_prob, spec.labels)
        plots = plot_multiclass_evaluation(test_true, test_prob, spec.labels, plots_dir, spec.display_name)

        chosen_processor = build_metadata_processor(["age"], ["sex", "localization"])
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
                "mode": mode,
                "metadata_dim": metadata_dim if chosen_variant == "fusion" else 0,
            },
        )
        joblib.dump(metadata_baseline, artifact_dir / "metadata_only_baseline.joblib")
        train.to_csv(data_dir / "train.csv", index=False)
        val.to_csv(data_dir / "val.csv", index=False)
        test.to_csv(data_dir / "test.csv", index=False)
        save_json(
            {
                "candidate_scores": {
                    "image_only_macro_f1": image_score,
                    "fusion_macro_f1": fusion_score,
                    "metadata_only_macro_f1": metadata_score,
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
        model = FusionVisionModel(config["backbone"], len(config["labels"]), metadata_dim=int(config["metadata_dim"]))
        model.load_state_dict(torch.load(spec.artifact_dir / "model.pt", map_location="cpu"))
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
                    "age": kwargs.get("age"),
                    "sex": kwargs.get("sex", "unknown"),
                    "localization": kwargs.get("localization", "unknown"),
                }
            ]
        )
        metadata_tensor = torch.tensor(to_dense_float32(bundle["metadata_processor"].transform(metadata_frame)))
        tensor_image = get_image_transforms(tuple(bundle["config"]["image_size"]), train=False)(image.convert("RGB")).unsqueeze(0)
        logits = bundle["model"](tensor_image, metadata_tensor if bundle["config"]["selected_variant"] == "fusion" else None)
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        label_probs = {label: float(probabilities[index]) for index, label in enumerate(spec.labels)}
        top_index = int(np.argmax(probabilities))
        top_label = spec.labels[top_index]
        heatmap = generate_gradcam(
            bundle["model"],
            tensor_image,
            metadata_tensor if bundle["config"]["selected_variant"] == "fusion" else None,
            target_index=top_index,
        )
        overlay = overlay_heatmap(image, heatmap)
        return {
            "module": spec.name,
            "probabilities": label_probs,
            "primary_decision": top_label,
            "threshold": 1.0 / len(spec.labels),
            "explanations": {
                "focus_label": top_label,
                "gradcam_overlay": overlay,
                "heatmap": heatmap.tolist(),
            },
            "model_version": bundle["config"]["selected_variant"],
            "metadata": {
                "top_probability": label_probs[top_label],
                "malignant_risk": float(label_probs.get("mel", 0.0) + label_probs.get("bcc", 0.0) + label_probs.get("akiec", 0.0)),
            },
        }
