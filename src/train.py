from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cnn_model import predict_cnn_probabilities, train_cnn
from src.data_loader import DatasetConfig, build_balanced_subset, load_metadata
from src.ensemble import DEFAULT_WEIGHTS, combine_probabilities, save_ensemble_config
from src.evaluate import compute_metrics, find_best_threshold, save_confusion_matrix_plot, save_metrics_json
from src.feature_extractor import extract_feature_vectors, save_feature_vectors
from src.ml_models import predict_ml_probabilities, train_ml_models
from src.preprocessing import build_image_array, build_tabular_frame, create_group_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HealthSense AI on the NIH Chest X-ray dataset.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--archive-dir", type=Path, default=None)
    parser.add_argument("--target-label", type=str, default="Pneumonia")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--max-positive-samples", type=int, default=500)
    parser.add_argument("--negative-multiplier", type=float, default=1.5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def _save_manifest(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    config = DatasetConfig.from_project_root(
        project_root=args.project_root,
        archive_dir=args.archive_dir,
        target_label=args.target_label,
        image_size=(args.image_size, args.image_size),
        max_positive_samples=args.max_positive_samples,
        negative_multiplier=args.negative_multiplier,
    )

    processed_dir = config.project_root / "data" / "processed"
    cnn_dir = config.project_root / "models" / "cnn"
    ensemble_dir = config.project_root / "models" / "ensemble"

    print("Loading metadata...")
    metadata = load_metadata(config)
    dataset = build_balanced_subset(metadata, config)
    train_df, val_df, test_df = create_group_splits(dataset, random_state=config.random_state)

    print(f"Balanced dataset size: {len(dataset)}")
    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    _save_manifest(train_df, processed_dir / "train_manifest.csv")
    _save_manifest(val_df, processed_dir / "val_manifest.csv")
    _save_manifest(test_df, processed_dir / "test_manifest.csv")

    print("Preparing image tensors...")
    train_images = build_image_array(train_df["image_path"].tolist(), config.image_size)
    val_images = build_image_array(val_df["image_path"].tolist(), config.image_size)
    test_images = build_image_array(test_df["image_path"].tolist(), config.image_size)

    y_train = train_df["target"].to_numpy(dtype=np.int32)
    y_val = val_df["target"].to_numpy(dtype=np.int32)
    y_test = test_df["target"].to_numpy(dtype=np.int32)

    print("Training CNN...")
    cnn_model, history = train_cnn(
        train_images=train_images,
        y_train=y_train,
        val_images=val_images,
        y_val=y_val,
        output_dir=cnn_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("Training structured-data models...")
    train_tabular = build_tabular_frame(train_df)
    val_tabular = build_tabular_frame(val_df)
    test_tabular = build_tabular_frame(test_df)
    ml_models = train_ml_models(train_tabular, pd.Series(y_train), ensemble_dir)

    print("Generating predictions...")
    cnn_val_probs = predict_cnn_probabilities(cnn_model, val_images)
    cnn_test_probs = predict_cnn_probabilities(cnn_model, test_images)

    ml_val_probs = {
        name: np.asarray(values)
        for name, values in predict_ml_probabilities(ml_models, val_tabular).items()
    }
    ml_test_probs = {
        name: np.asarray(values)
        for name, values in predict_ml_probabilities(ml_models, test_tabular).items()
    }

    ensemble_val_probs = combine_probabilities({"cnn": cnn_val_probs, **ml_val_probs}, DEFAULT_WEIGHTS)
    ensemble_test_probs = combine_probabilities({"cnn": cnn_test_probs, **ml_test_probs}, DEFAULT_WEIGHTS)
    best_threshold = find_best_threshold(y_val, ensemble_val_probs)

    cnn_features = extract_feature_vectors(cnn_model, test_images)
    save_feature_vectors(cnn_features, processed_dir / "test_cnn_features.npy")
    save_ensemble_config(ensemble_dir / "ensemble_config.json", DEFAULT_WEIGHTS)

    metrics_payload = {
        "target_label": config.target_label,
        "image_size": list(config.image_size),
        "split_sizes": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "class_balance": {
            "train_positive_rate": float(y_train.mean()),
            "validation_positive_rate": float(y_val.mean()),
            "test_positive_rate": float(y_test.mean()),
        },
        "best_threshold": float(best_threshold),
        "ensemble_weights": DEFAULT_WEIGHTS,
        "cnn_history": history,
        "metrics": {
            "cnn": compute_metrics(y_test, cnn_test_probs, threshold=0.5),
            "random_forest": compute_metrics(y_test, ml_test_probs["random_forest"], threshold=0.5),
            "svm": compute_metrics(y_test, ml_test_probs["svm"], threshold=0.5),
            "ensemble": compute_metrics(y_test, ensemble_test_probs, threshold=best_threshold),
        },
    }

    save_metrics_json(metrics_payload, ensemble_dir / "metrics.json")
    save_confusion_matrix_plot(
        y_true=y_test,
        y_prob=ensemble_test_probs,
        output_path=ensemble_dir / "confusion_matrix.png",
        threshold=best_threshold,
    )

    runtime_metadata = {
        "target_label": config.target_label,
        "image_size": list(config.image_size),
        "tabular_columns": ["patient_age", "patient_gender", "view_position", "follow_up"],
        "threshold": float(best_threshold),
        "weights": DEFAULT_WEIGHTS,
    }
    (ensemble_dir / "runtime_metadata.json").write_text(
        json.dumps(runtime_metadata, indent=2),
        encoding="utf-8",
    )

    print("Training complete.")
    print(json.dumps(metrics_payload["metrics"]["ensemble"], indent=2))


if __name__ == "__main__":
    main()
