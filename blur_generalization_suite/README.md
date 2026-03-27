# Blur Generalization Suite

这个目录是基于 `train_motion.py`、`0125_compensation/motion_direct_train.py` 和 `0125_compensation/test_for_wildrf.py` 拆出来的一组新实验脚本，目标是把你这次要补的三条实验线单独放好：

1. `CLIP + LoRA` 和 `EVA-Giant + LoRA` 的固定 backbone 模糊增强训练。
2. `DINOv3 ViT-Large-300M / ViT-Huge-840M` 的 teacher-student backbone sweep。
3. `DINOv3` 在不同模糊强度下的特征一致性分析与可视化。

## 文件说明

- `common.py`
  - 分布式训练、指标统计、checkpoint 读取、JSON 保存等通用工具。
- `data_utils.py`
  - 训练/测试数据集读取，CCMBA 模糊数据读取，motion/gaussian blur 处理，以及通用 transform 构建。
  - 现在补回了 teacher-student 原脚本里的 `enable_strong_aug` 入口，用于 0.4 概率切到带轻量 JPEG 增强的 strong transform。
- `dataset_configs.py`
  - 直接复用了 `test_for_wildrf.py` 的 cross-dataset 测试集配置，并补了 AIGCBenchmark 子集和 Table 4/5/6/7 的导出定义。
- `model_zoo.py`
  - `CLIP/EVA-Giant + LoRA` 分类器。
  - `DINOv3` teacher/student 网络。
  - `FocalLoss`、distill loss、SimCLR loss 等训练组件。
- `train_lora_blur.py`
  - 用固定 backbone + LoRA 的方式训练 `CLIP` 或 `ViT-Large`。
  - 数据增强逻辑沿用 direct-train 思路，不在训练阶段做测试。
- `eval_cross_dataset.py`
  - 加载 `train_lora_blur.py` 产出的 checkpoint。
  - 按 `test_for_wildrf.py` 的 clean / blur 双模式，在 cross-dataset 上统一评估。
  - 会额外导出 `table4_clean_accuracy.csv`、`table5_blur_accuracy.csv`、`table6_clean_f1.csv`、`table7_blur_f1.csv`。
- `train_teacher_student_backbones.py`
  - 沿用 `train_motion.py` 的 teacher-student 两阶段框架。
  - teacher clean、student blur、distill / feature / SimCLR loss 保持不变。
  - 只把 DINOv3 backbone 改成可配置输入，方便补 `ViT-Large-300M` 和 `ViT-Huge-840M`。
  - 默认 `data_preset=original_motion`，也就是更贴近原版 `train_motion.py` 的路径语义；如果要跑这次补的 SDV1.4 实验，用 `--data-preset sdv14`。
- `eval_teacher_student_aigc.py`
  - 加载 teacher-student checkpoint，并按 `teacher` 或 `student` 分支单独评估。
  - 默认数据集组是 `aigc_benchmark`。
- `analyze_dinov3_blur_consistency.py`
  - 计算 DINOv3 patch-level feature 的 cosine-similarity matrix。
  - 比较原图与不同 blur 强度下的相关性，输出逐图 JSON、平均 JSON 和趋势图。

## 默认路径和预设

### LoRA 训练

- 默认训练集：`/data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4`
- 默认 CCMBA 路径：`/home/work/xueyunqi/11ar_datasets/progan_ccmba_train`

### Teacher-student 训练

脚本里做了两个数据预设：

- `original_motion`：对齐原 `train_motion.py`
  - `train_root=/home/work/xueyunqi/11ar_datasets/extracted`
  - `ccmba_data_dir=/home/work/xueyunqi/11ar_datasets/progan_ccmba_train`
- `sdv14`：用于这次要补的 SDV1.4 实验
  - `train_root=/data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4`
  - `ccmba_data_dir=/data/app.e0016372/imagenet_tmp/ccmba_processed_sdv44`

当前默认是 `original_motion`。如果你服务器上的实际路径不同，直接改命令行参数即可，不需要改脚本内部。

### DINOv3 consistency

- 默认模型：`/nas_train/app.e0016372/models/dinov3-vit7b16-pretrain-lvd1689m`

## 运行入口示例

### 1. CLIP + LoRA 训练

```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_lora_blur.py \
  --model-family clip_lora \
  --backbone-path /nas_train/app.e0016372/models/blur_generalization_hf_backbones/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
  --blur-mode mixed \
  --blur-type motion \
  --blur-prob 0.1
```

### 2. EVA-Giant + LoRA 训练

```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_lora_blur.py \
  --model-family eva_giant_lora \
  --backbone-path /nas_train/app.e0016372/models/blur_generalization_hf_backbones/timm/eva_giant_patch14_336.m30m_ft_in22k_in1k \
  --blur-mode mixed \
  --blur-type motion \
  --blur-prob 0.1
```

### 3. LoRA cross-dataset clean / blur 测试

```bash
python blur_generalization_suite/eval_cross_dataset.py \
  --model-path blur_generalization_suite/outputs/lora_train/.../best_lora_model.pth \
  --dataset-group all \
  --blur-mode both
```

### 4. Teacher-student backbone sweep

如果你想按“原版路径语义”运行：

```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_teacher_student_backbones.py \
  --backbone-preset dinov3_vitl300m
```

如果你想按“这次补实验的 SDV1.4 数据”运行：

```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_teacher_student_backbones.py \
  --backbone-preset dinov3_vitl300m \
  --data-preset sdv14
```

或

```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_teacher_student_backbones.py \
  --backbone-preset dinov3_vith840m \
  --data-preset sdv14
```

### 5. Teacher / Student 在 AIGCBenchmark 上评估

```bash
python blur_generalization_suite/eval_teacher_student_aigc.py \
  --model-path blur_generalization_suite/outputs/teacher_student/.../best_student_model.pth \
  --branch student \
  --dataset-group aigc_benchmark
```

### 6. DINOv3 blur consistency 分析

```bash
python blur_generalization_suite/analyze_dinov3_blur_consistency.py \
  --data-root /data/app.e0016372/11ar_datasets/test/wukong/1_fake \
  --blur-type motion \
  --max-images 64
```

## 几个注意事项

- LoRA 这条线我按“固定 backbone + trainable LoRA + trainable classifier head”写的，没有把整个 backbone 解冻。
- Teacher-student 这条线现在已经对齐了原版的几个关键默认行为：
  - 默认 `local_files_only=True`
  - 默认数据预设回到原版路径语义
  - 补回了 `enable_strong_aug=True` 这层入口
- Table 4/5/6/7 的导出我做成了四个 csv 视图，方便你后续再按论文版式拼表；如果你的最终表头需要别的字段，再加一层汇总脚本就行。
- DINOv3 consistency 这里用的是 patch-level cosine similarity matrix correlation，思路对齐你仓库里 `1019_6models_cosine.py` 的那条分析线，但代码更轻量、只保留 DINOv3 本身。
