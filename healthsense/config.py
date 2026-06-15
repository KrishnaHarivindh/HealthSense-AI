from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


TaskType = Literal["binary", "multiclass", "multilabel"]
InputType = Literal["image", "tabular", "image_tabular"]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
REPORTS_ROOT = PROJECT_ROOT / "reports"
STORAGE_ROOT = PROJECT_ROOT / "storage"
SUBMISSION_ROOT = PROJECT_ROOT / "submission"
DATABASE_PATH = STORAGE_ROOT / "healthsense.db"
CACHE_ROOT = PROJECT_ROOT / "cache"


CHEST_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]

SKIN_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


@dataclass(slots=True)
class ModuleSpec:
    name: str
    display_name: str
    task_type: TaskType
    input_type: InputType
    dataset_root: Path
    labels: list[str]
    artifact_dir: Path
    metadata_columns: list[str] = field(default_factory=list)
    image_column: str | None = None
    target_column: str | None = None
    image_size: tuple[int, int] = (224, 224)
    smoke_rows: int | None = None
    full_epochs: int = 8
    smoke_epochs: int = 1
    batch_size: int = 16
    threshold_metric: str = "f1"
    primary_metric: str = "roc_auc"
    backbone: str | None = None


def get_module_specs() -> dict[str, ModuleSpec]:
    return {
        "chest": ModuleSpec(
            name="chest",
            display_name="Chest X-ray",
            task_type="multilabel",
            input_type="image_tabular",
            dataset_root=DATASETS_ROOT / "archive",
            labels=CHEST_LABELS,
            artifact_dir=ARTIFACTS_ROOT / "chest",
            metadata_columns=["Patient Age", "Patient Gender", "View Position", "Follow-up #"],
            image_column="Image Index",
            target_column="Finding Labels",
            image_size=(224, 224),
            smoke_rows=2400,
            smoke_epochs=1,
            full_epochs=8,
            batch_size=12,
            threshold_metric="f1",
            primary_metric="macro_roc_auc",
            backbone="densenet121",
        ),
        "skin": ModuleSpec(
            name="skin",
            display_name="Skin Cancer",
            task_type="multiclass",
            input_type="image_tabular",
            dataset_root=DATASETS_ROOT / "Skin Cancer",
            labels=SKIN_LABELS,
            artifact_dir=ARTIFACTS_ROOT / "skin",
            metadata_columns=["age", "sex", "localization"],
            image_column="image_id",
            target_column="dx",
            image_size=(224, 224),
            smoke_rows=2100,
            smoke_epochs=1,
            full_epochs=10,
            batch_size=16,
            threshold_metric="f1",
            primary_metric="macro_f1",
            backbone="efficientnet_b0",
        ),
        "diabetes": ModuleSpec(
            name="diabetes",
            display_name="Diabetes",
            task_type="binary",
            input_type="tabular",
            dataset_root=DATASETS_ROOT / "Diabetes",
            labels=["No Diabetes", "Diabetes"],
            artifact_dir=ARTIFACTS_ROOT / "diabetes",
            target_column="Outcome",
            smoke_rows=500,
            threshold_metric="f1",
            primary_metric="roc_auc",
        ),
        "heart": ModuleSpec(
            name="heart",
            display_name="Heart Disease",
            task_type="binary",
            input_type="tabular",
            dataset_root=DATASETS_ROOT / "Heart Disease",
            labels=["No Heart Disease", "Heart Disease"],
            artifact_dir=ARTIFACTS_ROOT / "heart",
            target_column="HeartDisease",
            smoke_rows=20000,
            threshold_metric="f1",
            primary_metric="roc_auc",
        ),
    }


def get_module_spec(module: str) -> ModuleSpec:
    specs = get_module_specs()
    if module not in specs:
        raise KeyError(f"Unknown module '{module}'. Available modules: {', '.join(specs)}")
    return specs[module]


def ensure_platform_dirs() -> None:
    for path in [ARTIFACTS_ROOT, REPORTS_ROOT, STORAGE_ROOT, SUBMISSION_ROOT, CACHE_ROOT]:
        path.mkdir(parents=True, exist_ok=True)
