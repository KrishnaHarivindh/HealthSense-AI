from __future__ import annotations

import argparse
import json

from healthsense.config import ensure_platform_dirs
from healthsense.service import train_module
from healthsense.storage import init_storage, record_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HealthSense AI modules.")
    parser.add_argument("--module", required=True, choices=["chest", "skin", "diabetes", "heart"])
    parser.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_platform_dirs()
    init_storage()
    result = train_module(args.module, mode=args.mode)
    record_experiment(
        module=args.module,
        run_mode=args.mode,
        artifact_version=str(result.get("selected_model", result.get("selected_variant", "default"))),
        status="completed",
        metrics_path=str(result.get("artifact_dir", "")),
        config=result,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
