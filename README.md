# DINO-Detect

Official-style code release layout for **DINO-Detect: Towards Blur-Robust AI-Generated Image Detection**.

This directory is a cleaned, modularized, public-facing reorganization of the original project code. It keeps the paper story explicit:

- A high-capacity **DINOv3 teacher** is trained on sharp images.
- A **student detector** learns from blurred counterparts through feature distillation.
- **Sharp-blur alignment** is enforced with classification, distillation, feature alignment, and contrastive objectives.
- The evaluation entry supports both **clean** and **blurred** settings for cross-dataset benchmarking.

## Why This Release Folder Exists

The original research code in the repository root was written for rapid experimentation. The `github_release/` folder turns that implementation into a structure that is easier to read, extend, and publish on GitHub without overwriting the original scripts.

## Repository Layout

```text
github_release/
├── configs/                 # Example JSON configs for training and evaluation
├── docs/                    # Data layout and structure notes
├── scripts/                 # Minimal shell entrypoints
├── src/dino_detect/
│   ├── data/                # Dataset readers, transforms, blur augmentation
│   ├── evaluation/          # Evaluation pipeline
│   ├── models/              # DINOv3 adapter and teacher-student network
│   ├── training/            # Losses and training loop
│   └── utils/               # Checkpointing, distributed helpers, misc utilities
├── train.py                 # Public training entry
├── test.py                  # Public evaluation entry
└── pyproject.toml           # Installable package metadata
```

Additional notes are available in [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) and [DATA_LAYOUT.md](docs/DATA_LAYOUT.md).

## Method Overview

The release matches the core narrative described in the paper:

1. Train a **teacher** on clean images with a frozen DINOv3 backbone and a lightweight trainable adapter head.
2. Train a **student** on blurred views while keeping the teacher as the semantic reference.
3. Optimize the student with:
   - supervised classification loss
   - teacher-to-student logit distillation
   - feature alignment in embedding space
   - pairwise sharp-blur contrastive alignment

The current implementation exposes motion blur, Gaussian blur, box blur, radial blur, bokeh blur, JPEG degradation, and optional CCMBA-backed blur replacement.

## Installation

```bash
cd github_release
pip install -e .
```

## Training

Edit [train_example.json](configs/train_example.json) with your local paths, then run:

```bash
python train.py --config configs/train_example.json
```

For multi-GPU training:

```bash
torchrun --nproc_per_node=4 train.py --config configs/train_example.json
```

## Evaluation

Edit [eval_example.json](configs/eval_example.json), then run:

```bash
python test.py --config configs/eval_example.json
```

The evaluator stores a structured JSON report under `outputs/<experiment_name>/results/`.

## Data Convention

Training expects a category-wise layout:

```text
train_root/
├── ProGAN/
│   ├── 0_real/
│   └── 1_fake/
├── StyleGAN/
│   ├── 0_real/
│   └── 1_fake/
└── ...
```

Evaluation supports both:

- simple binary datasets: `0_real/` and `1_fake/`
- benchmark-style multi-category datasets: `<generator_name>/0_real` and `<generator_name>/1_fake`

## Release Notes

- The original research scripts at the repository root are preserved as-is for traceability.
- This release folder is the polished version intended for public GitHub presentation.
- Paths are intentionally moved into config files so the public code no longer depends on hard-coded local directories.

## Citation

```bibtex
@article{dinodetect2026,
  title   = {DINO-Detect: Towards Blur-Robust AI-Generated Image Detection},
  author  = {Anonymous},
  journal = {},
  year    = {2026}
}
```
