# DINO-Detect paper-parity training and rebuttal evaluation

Run every command below from the repository root.

## Runtime layout

The scripts in this directory are stable entry points. They import the canonical
implementation from `blur_generalization_suite`, so fixes are not duplicated
between two diverging copies of the same training code.

- `train_paper_backbones.py`: paper-style teacher/student training for all backbones
- `train_motion_only_rebuttal.py`: strict motion-only wrapper
- `eval_motion_only_corruptions.py`: unseen-corruption evaluation for the rebuttal table
- `eval_teacher_student_aigc.py`: existing common AIGC benchmark evaluation

Existing backbone and dataset paths are preserved. The backbone checkpoint is
reloaded from its original path; output checkpoints contain the compact teacher
and student heads plus full experiment provenance, not a second copy of the large
frozen encoder. As in the historical training code, `best_*` means the epoch with
the highest training accuracy; it is not a validation-selected checkpoint.

## Environment smoke checks

Run these in the actual training environment before launching a long job:

```bash
python -c "import transformers; from packaging.version import Version; assert Version(transformers.__version__) >= Version('4.55'); print(transformers.__version__)"
python -c "from transformers import Aimv2VisionModel, CLIPVisionModel; print('transformers vision loaders OK')"
python -c "import timm; print(timm.__version__)"
python -m py_compile \
  blur_generalization_suite/model_zoo.py \
  blur_generalization_suite/data_utils.py \
  blur_generalization_suite/train_paper_backbones.py \
  blur_generalization_suite/eval_motion_only_corruptions.py
```

For all commands below, change `--nproc_per_node=8` only to match the available
GPU count. The defaults preserve the paper's effective batch size of 128 per GPU:
teacher `4 x 32`, student `1 x 128`. If a backbone requires a smaller micro-batch,
increase accumulation so each product remains 128.

## Three backbone runs

These commands use the existing SDv1.4 data preset and the full DINO-Detect
teacher/student objective: focal classification, logit KL, projected-feature
alignment, and ordinal contrastive learning. The frozen encoder remains in eval
mode throughout head training.

### CLIP ViT-bigG-14-1.84B

```bash
torchrun --nproc_per_node=8 0424_modelzoo_comparations/train_paper_backbones.py \
  --backbone-preset clip_vit_bigg_1_84b \
  --data-preset sdv14 \
  --training-profile paper \
  --experiment-mode dino_detect \
  --teacher-epochs 4 --student-epochs 15 \
  --teacher-batch-size 4 --teacher-accumulation-steps 32 \
  --student-batch-size 1 --student-accumulation-steps 128 \
  --teacher-learning-rate 1e-4 --student-learning-rate 5e-5 \
  --weight-decay 1e-4 --max-grad-norm 0 \
  --seed 3407
```

Best student checkpoint:

```text
blur_generalization_suite/outputs/paper_parity/clip_vit_bigg_1_84b_sdv14_dino_detect_paper_blur01/best_student_model.pth
```

### EVA-GIANT-1.1B

```bash
torchrun --nproc_per_node=8 0424_modelzoo_comparations/train_paper_backbones.py \
  --backbone-preset eva_giant_1_1b \
  --data-preset sdv14 \
  --training-profile paper \
  --experiment-mode dino_detect \
  --teacher-epochs 4 --student-epochs 15 \
  --teacher-batch-size 4 --teacher-accumulation-steps 32 \
  --student-batch-size 1 --student-accumulation-steps 128 \
  --teacher-learning-rate 1e-4 --student-learning-rate 5e-5 \
  --weight-decay 1e-4 --max-grad-norm 0 \
  --seed 3407
```

Best student checkpoint:

```text
blur_generalization_suite/outputs/paper_parity/eva_giant_1_1b_sdv14_dino_detect_paper_blur01/best_student_model.pth
```

### AIMv2-Huge comparison tier

```bash
torchrun --nproc_per_node=8 0424_modelzoo_comparations/train_paper_backbones.py \
  --backbone-preset aimv2_huge_840m \
  --data-preset sdv14 \
  --training-profile paper \
  --experiment-mode dino_detect \
  --teacher-epochs 4 --student-epochs 15 \
  --teacher-batch-size 4 --teacher-accumulation-steps 32 \
  --student-batch-size 1 --student-accumulation-steps 128 \
  --teacher-learning-rate 1e-4 --student-learning-rate 5e-5 \
  --weight-decay 1e-4 --max-grad-norm 0 \
  --seed 3407
```

Best student checkpoint:

```text
blur_generalization_suite/outputs/paper_parity/aimv2_huge_840m_sdv14_dino_detect_paper_blur01/best_student_model.pth
```

`apple/aimv2-huge-patch14-336` is the checkpoint used for the paper's 840M
comparison tier. Its public vision encoder is also commonly counted as roughly
682M parameters, so the manuscript should state that distinction explicitly.

## Common AIGC evaluation

Run the same evaluator for each best student checkpoint:

```bash
python 0424_modelzoo_comparations/eval_teacher_student_aigc.py \
  --model-path PATH_TO_BEST_STUDENT_MODEL \
  --branch student \
  --dataset-group aigc_benchmark \
  --blur-mode both \
  --blur-type motion \
  --blur-min 0.1 --blur-max 0.3 \
  --batch-size 16 --num-workers 4
```

The evaluator restores each checkpoint's native input size and normalization.
This matters for CLIP/AIMv2, whose statistics differ from DINOv3/EVA.

## Strict motion-only rebuttal controls

For the formal zero-shot corruption study, both runs below must use the same GPU
count, data preset, seed, student schedule, micro-batch, accumulation, blur
probability, and strength range. The strict profile automatically disables CCMBA,
Gaussian/defocus co-degradation, sensor noise, JPEG augmentation, and down/up
resizing. Required model input resize/crop is retained and is not counted as a
synthetic resize corruption.

The classification-only control trains the same frozen DINOv3-7B encoder and the
same student head on exactly the same clean/motion input stream. Its only training
objective is focal classification; it is not the untrained teacher branch.

### Classification-only DINOv3-7B

```bash
torchrun --nproc_per_node=8 0424_modelzoo_comparations/train_motion_only_rebuttal.py \
  --backbone-preset dinov3_vit7b \
  --data-preset original_motion \
  --experiment-mode classification_only \
  --blur-prob 0.1 --blur-min 0.1 --blur-max 0.3 \
  --student-epochs 15 \
  --student-batch-size 1 --student-accumulation-steps 128 \
  --student-learning-rate 5e-5 --weight-decay 1e-4 \
  --max-grad-norm 0 --seed 3407
```

```text
blur_generalization_suite/outputs/paper_parity/dinov3_vit7b_original_motion_classification_only_strict_motion_blur01/best_classification_only_model.pth
```

### Motion-only DINO-Detect

```bash
torchrun --nproc_per_node=8 0424_modelzoo_comparations/train_motion_only_rebuttal.py \
  --backbone-preset dinov3_vit7b \
  --data-preset original_motion \
  --experiment-mode dino_detect \
  --blur-prob 0.1 --blur-min 0.1 --blur-max 0.3 \
  --teacher-epochs 4 --student-epochs 15 \
  --teacher-batch-size 4 --teacher-accumulation-steps 32 \
  --student-batch-size 1 --student-accumulation-steps 128 \
  --teacher-learning-rate 1e-4 --student-learning-rate 5e-5 \
  --weight-decay 1e-4 --max-grad-norm 0 \
  --seed 3407
```

```text
blur_generalization_suite/outputs/paper_parity/dinov3_vit7b_original_motion_dino_detect_strict_motion_blur01/best_student_model.pth
```

## Cross-corruption evaluation

```bash
python 0424_modelzoo_comparations/eval_motion_only_corruptions.py \
  --classification-checkpoint \
    blur_generalization_suite/outputs/paper_parity/dinov3_vit7b_original_motion_classification_only_strict_motion_blur01/best_classification_only_model.pth \
  --dino-detect-checkpoint \
    blur_generalization_suite/outputs/paper_parity/dinov3_vit7b_original_motion_dino_detect_strict_motion_blur01/best_student_model.pth \
  --wildrf-root /data/app.e0016372/WildRF/test \
  --platforms reddit facebook twitter \
  --profile paper3 \
  --batch-size 16 --num-workers 4 \
  --output-dir blur_generalization_suite/outputs/reviewer_cross_corruption
```

Do not add `--allow-unverified-checkpoints` when producing rebuttal numbers. The
evaluator checks that both checkpoints are strict motion-only runs and rejects
differences in backbone, data, seed, input transform, blur schedule, optimization,
or effective student batch size. Each invocation creates a non-overwriting run
subdirectory named with a timestamp and both checkpoint hashes. A diagnostic run
that uses `--allow-unverified-checkpoints` is forced under `UNVERIFIED/` and its
paper-table files are prefixed with `UNVERIFIED_`.

The `paper3` severity profile is:

| Corruption | Severity values |
| --- | --- |
| Gaussian blur | strength 0.05, 0.10, 0.15 |
| Defocus blur | strength 0.05, 0.10, 0.15 |
| Box blur | strength 0.05, 0.10, 0.15 |
| Radial blur | strength 0.05, 0.10, 0.15 |
| Gaussian noise | sigma 0.02, 0.05, 0.10 |
| Shot noise | Poisson peak 250, 100, 25 |
| JPEG compression | quality 95, 80, 60 |

All corruptions are applied deterministically in RGB pixel space after resize and
center crop, then normalized for DINOv3. Radial blur follows the rotational
sampling operator used by the existing Table 9 experiment rather than zoom blur;
defocus uses a softened disk/bokeh PSF. The blur strength mapping is inherited
from Table 9, but the operator is applied at the checkpoint's 448-pixel crop, so
its physical footprint can be milder than an older lower-resolution evaluation.
The noise and JPEG severity sets are new rebuttal protocol choices, not Table 9
settings, and must be listed in the table caption or appendix.

Primary outputs:

- `paper_table.md` and `paper_table.csv`: the six requested rebuttal rows
- `severity_macro.csv`: severity-wise platform-macro results
- `corruption_summary.csv`: severity-averaged summaries
- `per_dataset_severity.csv`: each platform/corruption/severity result
- `per_image_predictions.csv`: auditable predictions and fake probabilities
- `sample_manifest.csv`: exact paths, file metadata, and SHA-256 for every test image
- `results.json`: checkpoints, provenance, aggregation rules, and all metrics

The table reports accuracy. It first macro-averages Reddit/Facebook/Twitter at
each severity, then averages severities. The combined Box/radial row is the
arithmetic mean of the separately evaluated Box and radial summaries.
