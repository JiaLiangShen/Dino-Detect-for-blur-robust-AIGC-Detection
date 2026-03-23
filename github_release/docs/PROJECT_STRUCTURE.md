# Project Structure

## Design Goal

This release reorganizes the original research implementation into a paper-friendly codebase with explicit module boundaries.

## Module Map

- `src/dino_detect/config.py`
  Central JSON-backed config definitions for training and evaluation.

- `src/dino_detect/data/`
  Dataset readers, preprocessing transforms, and blur augmentation utilities.

- `src/dino_detect/models/`
  DINOv3 adapter and teacher-student detector definitions.

- `src/dino_detect/training/`
  Multi-stage training loop and loss functions.

- `src/dino_detect/evaluation/`
  Benchmark evaluation, metric computation, and report export.

- `src/dino_detect/utils/`
  Checkpointing, distributed setup, seeding, and JSON helpers.

## Philosophy

The release code aims to make three things easy:

1. Swap datasets without editing the Python source.
2. Reproduce teacher/student experiments with explicit configs.
3. Read the code in the same conceptual order as the paper.
