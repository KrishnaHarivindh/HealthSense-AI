from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from healthsense.config import REPORTS_ROOT, get_module_spec
from healthsense.modules import get_module_handler
from healthsense.reporting import (
    build_prediction_report_html,
    build_prediction_report_payload,
    write_report_files,
)
from healthsense.storage import record_prediction, record_report


def train_module(module: str, mode: str = "smoke", **kwargs) -> dict[str, Any]:
    spec = get_module_spec(module)
    handler = get_module_handler(module)
    return handler.train(spec, mode=mode, **kwargs)


def evaluate_module(module: str) -> dict[str, Any]:
    spec = get_module_spec(module)
    handler = get_module_handler(module)
    return handler.evaluate(spec)


def load_module_bundle(module: str) -> dict[str, Any]:
    spec = get_module_spec(module)
    handler = get_module_handler(module)
    return handler.load_bundle(spec)


def predict_module(module: str, bundle: dict[str, Any] | None = None, session_id: int | None = None, **kwargs) -> dict[str, Any]:
    spec = get_module_spec(module)
    handler = get_module_handler(module)
    resolved_bundle = bundle or handler.load_bundle(spec)
    prediction = handler.predict(spec, resolved_bundle, **kwargs)

    report_name = f"{module}_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"
    input_summary = {key: value for key, value in kwargs.items() if key != "image"}
    serializable_prediction = {
        "module": prediction["module"],
        "probabilities": prediction["probabilities"],
        "primary_decision": prediction["primary_decision"],
        "threshold": prediction["threshold"],
        "model_version": prediction["model_version"],
        "metadata": prediction.get("metadata", {}),
    }
    report_payload = build_prediction_report_payload(module, input_summary, serializable_prediction)
    report_html = build_prediction_report_html(module, input_summary, serializable_prediction)
    report_paths = write_report_files(report_name, report_payload, report_html)

    prediction_id = record_prediction(
        module=module,
        input_summary=input_summary,
        prediction_payload=serializable_prediction,
        model_version=str(prediction["model_version"]),
        session_id=session_id,
        report_path=str(report_paths["html"]),
    )
    report_id = record_report("html", str(report_paths["html"]), prediction_id=prediction_id, session_id=session_id)
    prediction["report_id"] = report_id
    prediction["prediction_id"] = prediction_id
    prediction["report_paths"] = {name: str(path) for name, path in report_paths.items()}
    return prediction
