#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/eval_example.json}"
python test.py --config "${CONFIG_PATH}"
