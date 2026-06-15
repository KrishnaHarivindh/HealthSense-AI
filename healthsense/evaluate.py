from __future__ import annotations

import argparse
import json

from healthsense.config import ensure_platform_dirs
from healthsense.service import evaluate_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained HealthSense AI module.")
    parser.add_argument("--module", required=True, choices=["chest", "skin", "diabetes", "heart"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_platform_dirs()
    result = evaluate_module(args.module)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
