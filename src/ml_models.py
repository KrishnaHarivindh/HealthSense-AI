from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


NUMERIC_COLUMNS = ["patient_age", "follow_up"]
CATEGORICAL_COLUMNS = ["patient_gender", "view_position"]


def _build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def build_ml_models() -> dict[str, Pipeline]:
    random_forest = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=8,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    svm = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    C=2.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    return {
        "random_forest": random_forest,
        "svm": svm,
    }


def train_ml_models(
    train_frame: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path,
) -> dict[str, Pipeline]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = build_ml_models()
    for name, model in models.items():
        model.fit(train_frame, y_train)
        joblib.dump(model, output_dir / f"{name}.joblib")
    return models


def load_ml_models(model_dir: str | Path) -> dict[str, Pipeline]:
    directory = Path(model_dir)
    return {
        "random_forest": joblib.load(directory / "random_forest.joblib"),
        "svm": joblib.load(directory / "svm.joblib"),
    }


def predict_ml_probabilities(models: dict[str, Pipeline], frame: pd.DataFrame) -> dict[str, list[float]]:
    probabilities: dict[str, list[float]] = {}
    for name, model in models.items():
        probabilities[name] = model.predict_proba(frame)[:, 1].tolist()
    return probabilities
