from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dino_detect.config import TrainConfig
from dino_detect.training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINO-Detect.")
    parser.add_argument("--config", type=str, required=True, help="Path to a JSON training config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig.from_json(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
