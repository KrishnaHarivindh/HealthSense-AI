from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras


def build_cnn(input_shape: tuple[int, int, int]) -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="xray_image")
    x = keras.layers.RandomRotation(0.03)(inputs)
    x = keras.layers.RandomZoom(0.05)(x)

    for filters in (16, 32, 64):
        x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Conv2D(96, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation="relu", name="feature_vector")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="pneumonia_probability")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="healthsense_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def train_cnn(
    train_images: np.ndarray,
    y_train: np.ndarray,
    val_images: np.ndarray,
    y_val: np.ndarray,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 32,
) -> tuple[keras.Model, dict[str, list[float]]]:
    tf.keras.utils.set_random_seed(42)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_cnn(input_shape=train_images.shape[1:])
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(label): float(weight) for label, weight in zip(classes, weights)}

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=2,
            mode="max",
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_images,
        y_train,
        validation_data=(val_images, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    model.save(output_dir / "pneumonia_cnn.keras")
    return model, history.history


def load_cnn_model(model_path: str | Path) -> keras.Model:
    return keras.models.load_model(model_path)


def predict_cnn_probabilities(model: keras.Model, images: np.ndarray) -> np.ndarray:
    probabilities = model.predict(images, verbose=0).reshape(-1)
    return probabilities.astype(float)
