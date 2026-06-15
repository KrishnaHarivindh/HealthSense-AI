from __future__ import annotations

import numpy as np
import tensorflow as tf
from matplotlib import cm
from PIL import Image


def _find_last_conv_layer_name(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model for Grad-CAM generation.")


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    layer_name: str | None = None,
) -> np.ndarray:
    target_layer_name = layer_name or _find_last_conv_layer_name(model)
    target_layer = model.get_layer(target_layer_name)
    grad_model = tf.keras.models.Model(model.inputs, [target_layer.output, model.output])

    image_tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor, training=False)
        target_score = predictions[:, 0]

    gradients = tape.gradient(target_score, conv_outputs)
    if gradients is None:
        raise RuntimeError("Gradients could not be computed for the selected layer.")

    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))
    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients[tf.newaxis, tf.newaxis, :], axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if float(max_value) > 0:
        heatmap = heatmap / max_value

    return heatmap.numpy().astype(np.float32)


def overlay_heatmap_on_image(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.35,
) -> Image.Image:
    base_image = image.convert("RGB")
    resized_heatmap = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0)).resize(
        base_image.size,
        Image.BILINEAR,
    )
    colored_heatmap = cm.get_cmap("jet")(np.asarray(resized_heatmap, dtype=np.float32) / 255.0)[..., :3]
    heatmap_image = Image.fromarray(np.uint8(colored_heatmap * 255.0))
    return Image.blend(base_image, heatmap_image, alpha=alpha)


def heatmap_to_image(heatmap: np.ndarray, target_size: tuple[int, int]) -> Image.Image:
    grayscale = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0))
    return grayscale.resize(target_size, Image.BILINEAR)
