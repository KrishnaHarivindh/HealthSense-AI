from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DEFAULT_WEIGHTS = {
    "cnn": 0.5,
    "random_forest": 0.25,
    "svm": 0.25,
}


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Ensemble weights must sum to a positive value.")
    return {name: value / total for name, value in weights.items()}


def combine_probabilities(
    probability_map: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    resolved_weights = normalize_weights(weights or DEFAULT_WEIGHTS)
    ensemble = np.zeros_like(next(iter(probability_map.values())), dtype=float)
    for name, probabilities in probability_map.items():
        ensemble += np.asarray(probabilities, dtype=float) * resolved_weights.get(name, 0.0)
    return ensemble


def save_ensemble_config(output_path: str | Path, weights: dict[str, float] | None = None) -> None:
    resolved = normalize_weights(weights or DEFAULT_WEIGHTS)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"weights": resolved}, indent=2), encoding="utf-8")


def load_ensemble_config(config_path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return normalize_weights(payload.get("weights", DEFAULT_WEIGHTS))
