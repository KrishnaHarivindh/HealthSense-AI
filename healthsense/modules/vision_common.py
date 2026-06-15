from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import torch
import json
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_dense_float32(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def build_metadata_processor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


@dataclass(slots=True)
class VisionBatch:
    images: torch.Tensor
    metadata: torch.Tensor
    targets: torch.Tensor
    image_paths: list[str]


class MultimodalImageDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_column: str,
        target_column: str,
        transform,
        metadata_matrix: np.ndarray,
        task_type: Literal["multilabel", "multiclass"],
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_column = image_column
        self.target_column = target_column
        self.transform = transform
        self.metadata_matrix = to_dense_float32(metadata_matrix)
        self.task_type = task_type

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row[self.image_column]) as image:
            image = image.convert("RGB")
            tensor_image = self.transform(image)
        metadata = torch.tensor(self.metadata_matrix[index], dtype=torch.float32)
        if self.task_type == "multilabel":
            target = torch.tensor(row[self.target_column], dtype=torch.float32)
        else:
            target = torch.tensor(int(row[self.target_column]), dtype=torch.long)
        return tensor_image, metadata, target, row[self.image_column]


class FusionVisionModel(nn.Module):
    def __init__(self, backbone_name: str, num_outputs: int, metadata_dim: int = 0, dropout: float = 0.25) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.metadata_dim = metadata_dim
        self.feature_extractor, feature_dim, target_layer = self._build_backbone(backbone_name)
        self.target_layer = target_layer
        self.metadata_head = None
        fusion_dim = feature_dim
        if metadata_dim > 0:
            self.metadata_head = nn.Sequential(
                nn.Linear(metadata_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            fusion_dim += 64
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_outputs),
        )

    def _build_backbone(self, backbone_name: str):
        if backbone_name == "densenet121":
            backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            feature_dim = backbone.classifier.in_features
            target_layer = backbone.features[-1]
            backbone.classifier = nn.Identity()
            return backbone, feature_dim, target_layer
        if backbone_name == "efficientnet_b0":
            backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            feature_dim = backbone.classifier[1].in_features
            target_layer = backbone.features[-1]
            backbone.classifier = nn.Identity()
            return backbone, feature_dim, target_layer
        raise ValueError(f"Unsupported backbone '{backbone_name}'")

    def forward(self, images: torch.Tensor, metadata: torch.Tensor | None = None) -> torch.Tensor:
        features = self.feature_extractor(images)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        if self.metadata_head is not None and metadata is not None:
            metadata_features = self.metadata_head(metadata)
            features = torch.cat([features, metadata_features], dim=1)
        return self.head(features)


def get_image_transforms(image_size: tuple[int, int], train: bool):
    size = image_size[0]
    if train:
        return transforms.Compose(
            [
                transforms.Resize((size + 12, size + 12)),
                transforms.CenterCrop((size, size)),
                transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.03, hue=0.01),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_dataloaders(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    image_path_column: str,
    target_column: str,
    metadata_processor: ColumnTransformer,
    metadata_columns: list[str],
    image_size: tuple[int, int],
    task_type: Literal["multilabel", "multiclass"],
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    numeric_columns = [col for col in metadata_columns if train_frame[col].dtype.kind in "if"]
    categorical_columns = [col for col in metadata_columns if col not in numeric_columns]
    train_meta = to_dense_float32(metadata_processor.fit_transform(train_frame[metadata_columns]))
    val_meta = to_dense_float32(metadata_processor.transform(val_frame[metadata_columns]))
    test_meta = to_dense_float32(metadata_processor.transform(test_frame[metadata_columns]))

    train_dataset = MultimodalImageDataset(
        train_frame,
        image_path_column,
        target_column,
        transform=get_image_transforms(image_size, train=True),
        metadata_matrix=train_meta,
        task_type=task_type,
    )
    val_dataset = MultimodalImageDataset(
        val_frame,
        image_path_column,
        target_column,
        transform=get_image_transforms(image_size, train=False),
        metadata_matrix=val_meta,
        task_type=task_type,
    )
    test_dataset = MultimodalImageDataset(
        test_frame,
        image_path_column,
        target_column,
        transform=get_image_transforms(image_size, train=False),
        metadata_matrix=test_meta,
        task_type=task_type,
    )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def fit_vision_model(
    model: FusionVisionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task_type: Literal["multilabel", "multiclass"],
    class_weights: np.ndarray | None,
    epochs: int,
    device: torch.device,
    primary_metric: str,
) -> tuple[FusionVisionModel, dict[str, list[float]]]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    if task_type == "multilabel":
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None
        )
    else:
        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None
        )

    best_metric = -float("inf")
    best_state = None
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_metric": []}

    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        for images, metadata, targets, _ in train_loader:
            images = images.to(device)
            metadata = metadata.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(images, metadata)
                loss = loss_fn(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * images.size(0)

        val_loss, val_metric = evaluate_vision_model(model, val_loader, task_type, loss_fn, device, primary_metric)
        history["train_loss"].append(running_loss / max(len(train_loader.dataset), 1))
        history["val_loss"].append(val_loss)
        history["val_metric"].append(val_metric)
        if val_metric > best_metric:
            best_metric = val_metric
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_vision_model(
    model: FusionVisionModel,
    data_loader: DataLoader,
    task_type: Literal["multilabel", "multiclass"],
    loss_fn,
    device: torch.device,
    primary_metric: str,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for images, metadata, targets, _ in data_loader:
            images = images.to(device)
            metadata = metadata.to(device)
            targets = targets.to(device)
            logits = model(images, metadata)
            loss = loss_fn(logits, targets)
            running_loss += float(loss.item()) * images.size(0)
            if task_type == "multilabel":
                probs = torch.sigmoid(logits).cpu().numpy()
                target_np = targets.cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                target_np = targets.cpu().numpy()
            all_probs.append(probs)
            all_targets.append(target_np)

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    if task_type == "multilabel":
        metric = float(roc_auc_score(y_true, y_prob, average="macro"))
    else:
        predictions = np.argmax(y_prob, axis=1)
        metric = float(f1_score(y_true, predictions, average="macro", zero_division=0))
    return running_loss / max(len(data_loader.dataset), 1), metric


def predict_vision_probabilities(model: FusionVisionModel, data_loader: DataLoader, task_type: Literal["multilabel", "multiclass"], device: torch.device):
    model.eval()
    all_probs = []
    all_targets = []
    all_paths: list[str] = []
    with torch.no_grad():
        for images, metadata, targets, image_paths in data_loader:
            images = images.to(device)
            metadata = metadata.to(device)
            logits = model(images, metadata)
            if task_type == "multilabel":
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
            all_paths.extend(image_paths)
    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0), all_paths


def save_vision_bundle(
    model: FusionVisionModel,
    metadata_processor: ColumnTransformer,
    artifact_dir: Path,
    history: dict[str, list[float]],
    config_payload: dict[str, Any],
) -> tuple[Path, Path, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pt"
    processor_path = artifact_dir / "metadata_processor.joblib"
    history_path = artifact_dir / "training_history.joblib"
    torch.save(model.state_dict(), model_path)
    joblib.dump(metadata_processor, processor_path)
    joblib.dump(history, history_path)
    joblib.dump(config_payload, artifact_dir / "model_config.joblib")
    (artifact_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    return model_path, processor_path, history_path


def load_vision_bundle(artifact_dir: Path) -> dict[str, Any]:
    config_path = artifact_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = joblib.load(artifact_dir / "model_config.joblib")
    return {
        "metadata_processor": joblib.load(artifact_dir / "metadata_processor.joblib"),
        "history": joblib.load(artifact_dir / "training_history.joblib"),
        "config": config,
    }
