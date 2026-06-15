from __future__ import annotations

from typing import Any

import numpy as np
import shap
import torch
from matplotlib import cm
from PIL import Image
from sklearn.pipeline import Pipeline


def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    metadata_tensor: torch.Tensor | None,
    target_index: int,
) -> np.ndarray:
    model.eval()
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    target_layer = getattr(model, "target_layer", None)
    if target_layer is None:
        raise ValueError("Model does not expose a target_layer for Grad-CAM.")

    def forward_hook(_, __, output):
        if not isinstance(output, torch.Tensor):
            raise ValueError("Grad-CAM target layer must return a tensor output.")
        cloned = output.clone()
        activations.append(cloned.detach())
        cloned.register_hook(lambda grad: gradients.append(grad.detach()))
        return cloned

    forward_handle = target_layer.register_forward_hook(forward_hook)

    try:
        outputs = model(input_tensor, metadata_tensor)
        if outputs.ndim == 2:
            target_score = outputs[:, target_index].sum()
        else:
            target_score = outputs[target_index]
        model.zero_grad(set_to_none=True)
        target_score.backward()
        activation = activations[-1][0]
        gradient = gradients[-1][0]
        weights = gradient.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activation).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()
    finally:
        forward_handle.remove()


def overlay_heatmap(base_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.35) -> Image.Image:
    image = base_image.convert("RGB")
    resized = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0)).resize(image.size, Image.BILINEAR)
    colored = cm.get_cmap("jet")(np.asarray(resized, dtype=np.float32) / 255.0)[..., :3]
    heatmap_image = Image.fromarray(np.uint8(colored * 255.0))
    return Image.blend(image, heatmap_image, alpha=alpha)


def compute_tree_shap_values(model: Any, frame, background=None) -> dict[str, float]:
    if isinstance(model, Pipeline):
        preprocessor = model.named_steps.get("preprocessor")
        estimator = model.named_steps.get("model")
        if preprocessor is None or estimator is None:
            raise ValueError("Pipeline model must expose 'preprocessor' and 'model' steps for SHAP.")
        background_frame = background if background is not None else frame
        transformed_background = preprocessor.transform(background_frame)
        transformed_frame = preprocessor.transform(frame)
        feature_names = list(preprocessor.get_feature_names_out())
        explainer = shap.Explainer(estimator, transformed_background)
        explanation = explainer(transformed_frame)
        values = explanation.values[0]
        if np.ndim(values) > 1:
            values = values[..., -1]
        return {
            str(name): float(value)
            for name, value in zip(feature_names, np.asarray(values).reshape(-1))
        }

    background_frame = background if background is not None else frame
    model_module = type(model).__module__.lower()
    if "catboost" in model_module:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        explanation = explainer(frame.iloc[[0]])
    else:
        try:
            explainer = shap.Explainer(model, background_frame)
            explanation = explainer(frame.iloc[[0]])
        except Exception:
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            explanation = explainer(frame.iloc[[0]])
    values = explanation.values[0]
    if np.ndim(values) > 1:
        values = values[..., -1]
    return {
        str(name): float(value)
        for name, value in zip(frame.columns, np.asarray(values).reshape(-1))
    }
