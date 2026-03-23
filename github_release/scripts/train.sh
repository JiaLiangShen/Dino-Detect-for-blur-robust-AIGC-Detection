#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/train_example.json}"
python train.py --config "${CONFIG_PATH}"
