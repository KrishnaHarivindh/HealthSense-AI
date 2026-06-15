from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf


def build_feature_model(model: tf.keras.Model) -> tf.keras.Model:
    return tf.keras.Model(inputs=model.input, outputs=model.get_layer("feature_vector").output)


def extract_feature_vectors(model: tf.keras.Model, images: np.ndarray) -> np.ndarray:
    feature_model = build_feature_model(model)
    features = feature_model.predict(images, verbose=0)
    return np.asarray(features, dtype=np.float32)


def save_feature_vectors(feature_vectors: np.ndarray, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, feature_vectors)
