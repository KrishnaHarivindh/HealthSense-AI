from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PredictionResult:
    module: str
    probabilities: dict[str, float]
    primary_decision: str
    threshold: float | dict[str, float]
    explanations: dict[str, Any]
    model_version: str
    report_id: int | None = None
    session_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingArtifacts:
    module: str
    artifact_dir: Path
    model_path: Path
    metrics_path: Path
    config_path: Path
    plots_dir: Path


@dataclass(slots=True)
class EvaluationBundle:
    module: str
    metrics: dict[str, Any]
    plots: dict[str, Path]
    selected_model: str
