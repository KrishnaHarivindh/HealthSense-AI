from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


@dataclass(slots=True)
class PlattCalibrator:
    slope: float = 1.0
    intercept: float = 0.0

    def fit(self, scores: np.ndarray, targets: np.ndarray) -> "PlattCalibrator":
        scores = np.asarray(scores).reshape(-1, 1)
        targets = np.asarray(targets)
        model = LogisticRegression(max_iter=2000)
        model.fit(scores, targets)
        self.slope = float(model.coef_[0][0])
        self.intercept = float(model.intercept_[0])
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores).reshape(-1)
        logits = self.slope * scores + self.intercept
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return probabilities


def split_binary_frame(
    df: pd.DataFrame,
    target_column: str,
    smoke_rows: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = df.copy()
    if smoke_rows is not None and len(frame) > smoke_rows:
        frame, _ = train_test_split(
            frame,
            train_size=smoke_rows,
            stratify=frame[target_column],
            random_state=random_state,
        )

    train_val, test = train_test_split(
        frame,
        test_size=0.15,
        stratify=frame[target_column],
        random_state=random_state,
    )
    train, val = train_test_split(
        train_val,
        test_size=0.1764705882,
        stratify=train_val[target_column],
        random_state=random_state,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def build_binary_candidates(
    numeric_columns: list[str],
    categorical_columns: list[str],
    class_weight_scale: float,
) -> dict[str, Any]:
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    ratio = max(class_weight_scale, 1.0)
    return {
        "logistic": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(class_weight="balanced", max_iter=2000)),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=350,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=ratio,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=350,
                        learning_rate=0.05,
                        num_leaves=63,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def build_catboost_candidate(categorical_columns: list[str]) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=450,
        depth=7,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        random_seed=42,
        auto_class_weights="Balanced",
    )


def select_best_binary_model(
    candidates: dict[str, Any],
    train_frame: pd.DataFrame,
    train_target: pd.Series,
    val_frame: pd.DataFrame,
    val_target: pd.Series,
    catboost_candidate: CatBoostClassifier | None,
    categorical_columns: list[str],
    output_dir: Path,
) -> tuple[str, Any, dict[str, Any], PlattCalibrator]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Any] = {}

    for name, model in candidates.items():
        model.fit(train_frame, train_target)
        probabilities = model.predict_proba(val_frame)[:, 1]
        scores[name] = {"roc_auc": float(roc_auc_score(val_target, probabilities))}
        fitted_models[name] = model

    if catboost_candidate is not None:
        catboost_candidate.fit(
            train_frame,
            train_target,
            cat_features=categorical_columns if categorical_columns else None,
        )
        probabilities = catboost_candidate.predict_proba(val_frame)[:, 1]
        scores["catboost"] = {"roc_auc": float(roc_auc_score(val_target, probabilities))}
        fitted_models["catboost"] = catboost_candidate

    best_name = max(scores, key=lambda name: scores[name]["roc_auc"])
    best_model = fitted_models[best_name]
    raw_probabilities = best_model.predict_proba(val_frame)[:, 1]
    calibrator = PlattCalibrator().fit(raw_probabilities, val_target.to_numpy())

    joblib.dump(best_model, output_dir / f"{best_name}.joblib")
    joblib.dump(calibrator, output_dir / "calibrator.joblib")
    joblib.dump(scores, output_dir / "candidate_scores.joblib")
    return best_name, best_model, scores, calibrator
