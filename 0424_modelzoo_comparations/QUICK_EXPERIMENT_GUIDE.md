# 实验快速运行说明

以下命令均在项目根目录执行。将 `--nproc_per_node=8` 改成实际 GPU 数量。

## 1. 环境检查

```bash
python -c "import transformers; print(transformers.__version__)"
python -c "from transformers import Aimv2VisionModel, CLIPVisionModel; import timm; print('environment OK')"
```

需要 `transformers>=4.55`。模型和数据继续使用代码中的已有路径。

## 2. 实验一：三个对比 backbone

依次训练 CLIP ViT-bigG、EVA-GIANT 和 AIMv2-Huge：

```bash
for BACKBONE in clip_vit_bigg_1_84b eva_giant_1_1b aimv2_huge_840m; do
  torchrun --nproc_per_node=8 0424_modelzoo_comparations/train_paper_backbones.py \
    --backbone-preset ${BACKBONE} \
    --data-preset sdv14 \
    --training-profile paper \
    --experiment-mode dino_detect \
    --teacher-batch-size 4 --teacher-accumulation-steps 32 \
    --student-batch-size 1 --student-accumulation-steps 128 \
    --max-grad-norm 0 --seed 3407
done
```

checkpoint 位于：

```text
blur_generalization_suite/outputs/paper_parity/<backbone>_sdv14_dino_detect_paper_blur01/best_student_model.pth
```

常规 AIGC 测试：

```bash
python 0424_modelzoo_comparations/eval_teacher_student_aigc.py \
  --model-path PATH_TO_BEST_STUDENT_MODEL \
  --branch student --dataset-group aigc_benchmark \
  --blur-mode both --batch-size 16
```

重点查看评测目录中的汇总 CSV/JSON，对比 clean、motion blur 的 Accuracy 和 BAcc。`best_student_model.pth` 按训练准确率选择，并非验证集最优。

## 3. 实验二：motion-only 严格跨退化实验

先训练 classification-only 对照：

```bash
torchrun --nproc_per_node=8 0424_modelzoo_comparations/train_motion_only_rebuttal.py \
  --backbone-preset dinov3_vit7b --data-preset original_motion \
  --experiment-mode classification_only \
  --blur-prob 0.1 --blur-min 0.1 --blur-max 0.3 \
  --student-batch-size 1 --student-accumulation-steps 128 \
  --max-grad-norm 0 --seed 3407
```

再训练 motion-only DINO-Detect：

```bash
torchrun --nproc_per_node=8 0424_modelzoo_comparations/train_motion_only_rebuttal.py \
  --backbone-preset dinov3_vit7b --data-preset original_motion \
  --experiment-mode dino_detect \
  --blur-prob 0.1 --blur-min 0.1 --blur-max 0.3 \
  --teacher-batch-size 4 --teacher-accumulation-steps 32 \
  --student-batch-size 1 --student-accumulation-steps 128 \
  --max-grad-norm 0 --seed 3407
```

生成 Gaussian/defocus/box/radial/noise/JPEG 表格：

```bash
python 0424_modelzoo_comparations/eval_motion_only_corruptions.py \
  --classification-checkpoint \
    blur_generalization_suite/outputs/paper_parity/dinov3_vit7b_original_motion_classification_only_strict_motion_blur01/best_classification_only_model.pth \
  --dino-detect-checkpoint \
    blur_generalization_suite/outputs/paper_parity/dinov3_vit7b_original_motion_dino_detect_strict_motion_blur01/best_student_model.pth \
  --wildrf-root /data/app.e0016372/WildRF/test \
  --profile paper3 --batch-size 16
```

结果位于：

```text
blur_generalization_suite/outputs/reviewer_cross_corruption/<时间与checkpoint哈希>/
```

重点文件：

- `paper_table.md`：回复表格所需的 Accuracy（%）。
- `severity_macro.csv`：每种退化、每个强度的结果。
- `corruption_summary.csv`：不同强度平均后的结果。
- `per_image_predictions.csv`：逐图预测，用于排查异常。
- `results.json`：checkpoint、配置及 provenance。

主表先对 Reddit、Facebook、Twitter 等权平均，再对三个 severity 平均。正式结果不要添加 `--allow-unverified-checkpoints`。
