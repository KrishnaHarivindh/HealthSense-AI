from __future__ import annotations

from healthsense.modules.chest import ChestModule
from healthsense.modules.diabetes import DiabetesModule
from healthsense.modules.heart import HeartModule
from healthsense.modules.skin import SkinModule

MODULE_REGISTRY = {
    "chest": ChestModule(),
    "skin": SkinModule(),
    "diabetes": DiabetesModule(),
    "heart": HeartModule(),
}


def get_module_handler(name: str):
    if name not in MODULE_REGISTRY:
        raise KeyError(f"Unknown module '{name}'.")
    return MODULE_REGISTRY[name]
