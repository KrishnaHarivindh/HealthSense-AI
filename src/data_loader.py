from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class DatasetConfig:
    project_root: Path
    archive_dir: Path
    metadata_csv: Path
    target_label: str = "Pneumonia"
    image_size: tuple[int, int] = (96, 96)
    random_state: int = 42
    max_positive_samples: int | None = 500
    negative_multiplier: float = 1.5

    @classmethod
    def from_project_root(
        cls,
        project_root: Path | None = None,
        archive_dir: Path | None = None,
        target_label: str = "Pneumonia",
        image_size: tuple[int, int] = (96, 96),
        max_positive_samples: int | None = 500,
        negative_multiplier: float = 1.5,
    ) -> "DatasetConfig":
        resolved_root = (project_root or Path(__file__).resolve().parents[1]).resolve()
        resolved_archive = (archive_dir or resolved_root.parent / "archive").resolve()
        metadata_csv = resolved_archive / "Data_Entry_2017.csv"
        return cls(
            project_root=resolved_root,
            archive_dir=resolved_archive,
            metadata_csv=metadata_csv,
            target_label=target_label,
            image_size=image_size,
            max_positive_samples=max_positive_samples,
            negative_multiplier=negative_multiplier,
        )


def _clean_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Image Index": "image_index",
        "Finding Labels": "finding_labels",
        "Follow-up #": "follow_up",
        "Patient ID": "patient_id",
        "Patient Age": "patient_age",
        "Patient Gender": "patient_gender",
        "View Position": "view_position",
        "OriginalImage[Width": "image_width",
        "Height]": "image_height",
        "OriginalImagePixelSpacing[x": "pixel_spacing_x",
        "y]": "pixel_spacing_y",
    }
    cleaned = df.rename(columns=rename_map).copy()
    if "Unnamed: 11" in cleaned.columns:
        cleaned = cleaned.drop(columns=["Unnamed: 11"])
    return cleaned


@lru_cache(maxsize=4)
def build_image_index(archive_dir: str) -> dict[str, str]:
    image_map: dict[str, str] = {}
    archive_path = Path(archive_dir)
    for image_path in archive_path.glob("images_*/images/*.png"):
        image_map[image_path.name] = str(image_path.resolve())
    return image_map


def load_metadata(config: DatasetConfig) -> pd.DataFrame:
    if not config.metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {config.metadata_csv}")

    df = pd.read_csv(config.metadata_csv)
    df = _clean_metadata_columns(df)

    image_map = build_image_index(str(config.archive_dir))
    df["image_path"] = df["image_index"].map(image_map)
    df = df.dropna(subset=["image_path"]).copy()

    df["patient_age"] = pd.to_numeric(df["patient_age"], errors="coerce").fillna(0).clip(lower=0)
    df["follow_up"] = pd.to_numeric(df["follow_up"], errors="coerce").fillna(0).clip(lower=0)
    df["patient_gender"] = df["patient_gender"].fillna("Unknown").astype(str)
    df["view_position"] = df["view_position"].fillna("Unknown").astype(str)
    df["finding_labels"] = df["finding_labels"].fillna("No Finding").astype(str)
    df["target"] = df["finding_labels"].str.contains(config.target_label, regex=False).astype(int)
    return df


def build_balanced_subset(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    positives = df[df["target"] == 1].copy()
    negatives = df[df["target"] == 0].copy()

    if config.max_positive_samples is not None and len(positives) > config.max_positive_samples:
        positives = positives.sample(
            n=config.max_positive_samples,
            random_state=config.random_state,
        )

    negative_count = min(len(negatives), int(len(positives) * config.negative_multiplier))
    negatives = negatives.sample(n=negative_count, random_state=config.random_state)

    subset = (
        pd.concat([positives, negatives], ignore_index=True)
        .sample(frac=1.0, random_state=config.random_state)
        .reset_index(drop=True)
    )
    return subset
