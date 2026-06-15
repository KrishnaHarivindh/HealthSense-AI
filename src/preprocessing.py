from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold


TABULAR_COLUMNS = [
    "patient_age",
    "patient_gender",
    "view_position",
    "follow_up",
]


def load_grayscale_image(image_path: str, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(image_path) as image:
        processed = image.convert("L").resize(image_size, Image.BILINEAR)
        array = np.asarray(processed, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=-1)


def load_uploaded_image(file_bytes: bytes, image_size: tuple[int, int]) -> tuple[np.ndarray, Image.Image]:
    original = Image.open(BytesIO(file_bytes)).convert("L")
    preview = original.copy()
    processed = original.resize(image_size, Image.BILINEAR)
    array = np.asarray(processed, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=-1), preview


def build_image_array(image_paths: list[str], image_size: tuple[int, int]) -> np.ndarray:
    images = [load_grayscale_image(path, image_size) for path in image_paths]
    return np.stack(images, axis=0).astype(np.float32)


def build_tabular_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[TABULAR_COLUMNS].copy()


def create_group_splits(
    df: pd.DataFrame,
    label_column: str = "target",
    group_column: str = "patient_id",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    folds = list(splitter.split(df, df[label_column], df[group_column]))
    if len(folds) < 2:
        raise ValueError("Not enough grouped folds to create train, validation, and test splits.")

    val_indices = folds[0][1]
    test_indices = folds[1][1]
    holdout_indices = set(val_indices).union(set(test_indices))
    train_indices = np.array([idx for idx in range(len(df)) if idx not in holdout_indices])

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    return train_df, val_df, test_df


def severity_from_probability(probability: float) -> str:
    if probability >= 0.8:
        return "High"
    if probability >= 0.5:
        return "Moderate"
    if probability >= 0.25:
        return "Low"
    return "Minimal"
