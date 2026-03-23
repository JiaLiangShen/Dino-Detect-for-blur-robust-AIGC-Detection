from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dino_detect.config import EvalConfig
from dino_detect.evaluation import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DINO-Detect.")
    parser.add_argument("--config", type=str, required=True, help="Path to a JSON evaluation config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvalConfig.from_json(args.config)
    run_evaluation(config)


if __name__ == "__main__":
    main()
