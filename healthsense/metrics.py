from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def optimize_binary_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.1, 0.9, 81):
        predictions = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, predictions, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def optimize_multilabel_thresholds(y_true: np.ndarray, y_prob: np.ndarray, labels: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for index, label in enumerate(labels):
        thresholds[label] = optimize_binary_threshold(y_true[:, index], y_prob[:, index])
    return thresholds


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "threshold": float(threshold),
    }


def compute_multiclass_metrics(y_true: np.ndarray, y_prob: np.ndarray, labels: list[str]) -> dict[str, float]:
    predictions = np.argmax(y_prob, axis=1)
    one_hot = np.eye(len(labels))[y_true]
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "macro_roc_auc": float(roc_auc_score(one_hot, y_prob, multi_class="ovr", average="macro")),
        "macro_pr_auc": float(average_precision_score(one_hot, y_prob, average="macro")),
    }


def compute_multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: dict[str, float], labels: list[str]) -> dict[str, Any]:
    threshold_vector = np.asarray([thresholds[label] for label in labels])
    predictions = (y_prob >= threshold_vector).astype(int)
    per_label_auc = {}
    for index, label in enumerate(labels):
        if np.unique(y_true[:, index]).size > 1:
            per_label_auc[label] = float(roc_auc_score(y_true[:, index], y_prob[:, index]))
        else:
            per_label_auc[label] = float("nan")
    return {
        "macro_roc_auc": float(np.nanmean(list(per_label_auc.values()))),
        "micro_roc_auc": float(roc_auc_score(y_true, y_prob, average="micro")),
        "macro_pr_auc": float(average_precision_score(y_true, y_prob, average="macro")),
        "micro_pr_auc": float(average_precision_score(y_true, y_prob, average="micro")),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, predictions, average="micro", zero_division=0)),
        "per_label_roc_auc": per_label_auc,
        "thresholds": thresholds,
    }


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_binary_evaluation(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, output_dir: Path, title: str) -> dict[str, Path]:
    ensure_dir(output_dir)
    predictions = (y_prob >= threshold).astype(int)
    paths: dict[str, Path] = {}

    cm_path = output_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = confusion_matrix(y_true, predictions)
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Negative", "Positive"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"{title} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)
    paths["confusion_matrix"] = cm_path

    roc_path = output_dir / "roc_curve.png"
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_prob):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title} ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=200)
    plt.close(fig)
    paths["roc_curve"] = roc_path

    pr_path = output_dir / "pr_curve.png"
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AP={average_precision_score(y_true, y_prob):.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title} Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(pr_path, dpi=200)
    plt.close(fig)
    paths["pr_curve"] = pr_path

    cal_path = output_dir / "calibration_curve.png"
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mean_pred, fraction_pos, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"{title} Calibration Curve")
    fig.tight_layout()
    fig.savefig(cal_path, dpi=200)
    plt.close(fig)
    paths["calibration_curve"] = cal_path
    return paths


def plot_multiclass_evaluation(y_true: np.ndarray, y_prob: np.ndarray, labels: list[str], output_dir: Path, title: str) -> dict[str, Path]:
    ensure_dir(output_dir)
    predictions = np.argmax(y_prob, axis=1)
    paths: dict[str, Path] = {}

    cm_path = output_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    matrix = confusion_matrix(y_true, predictions, labels=np.arange(len(labels)))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{title} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)
    paths["confusion_matrix"] = cm_path

    support_path = output_dir / "class_support.png"
    support = pd.Series(y_true).map(lambda idx: labels[idx]).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    support.plot(kind="bar", ax=ax, color="#0f766e")
    ax.set_title(f"{title} Class Support")
    ax.set_xlabel("Class")
    ax.set_ylabel("Samples")
    fig.tight_layout()
    fig.savefig(support_path, dpi=200)
    plt.close(fig)
    paths["class_support"] = support_path
    return paths


def plot_multilabel_evaluation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
    output_dir: Path,
    title: str,
) -> dict[str, Path]:
    ensure_dir(output_dir)
    auc_values = []
    for index, label in enumerate(labels):
        if np.unique(y_true[:, index]).size > 1:
            auc_values.append({"label": label, "roc_auc": roc_auc_score(y_true[:, index], y_prob[:, index])})
        else:
            auc_values.append({"label": label, "roc_auc": np.nan})

    auc_df = pd.DataFrame(auc_values).sort_values("roc_auc", ascending=False)
    roc_bar_path = output_dir / "per_label_roc_auc.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=auc_df, x="label", y="roc_auc", ax=ax, color="#2563eb")
    ax.set_ylim(0, 1)
    ax.set_title(f"{title} Per-label ROC-AUC")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(roc_bar_path, dpi=200)
    plt.close(fig)

    support_path = output_dir / "label_support.png"
    label_support = pd.DataFrame(y_true, columns=labels).sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    label_support.plot(kind="bar", ax=ax, color="#dc2626")
    ax.set_title(f"{title} Label Support")
    ax.set_ylabel("Positive Samples")
    fig.tight_layout()
    fig.savefig(support_path, dpi=200)
    plt.close(fig)

    return {
        "per_label_roc_auc": roc_bar_path,
        "label_support": support_path,
    }
