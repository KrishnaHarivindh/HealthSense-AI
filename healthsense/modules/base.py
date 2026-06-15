from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from healthsense.config import ModuleSpec


class BaseModule(ABC):
    name: str

    @abstractmethod
    def train(self, spec: ModuleSpec, mode: str, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, spec: ModuleSpec) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_bundle(self, spec: ModuleSpec) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, spec: ModuleSpec, bundle: dict[str, Any], **kwargs) -> dict[str, Any]:
        raise NotImplementedError
