# 运行指南
---
本文档为当前模糊泛化实验提供实操运行手册。以下所有命令均需在代码仓库根目录执行：
`E:/.../Dino-Detect-for-blur-robust-AIGC-Detection-main`
若使用 PowerShell，单行完整书写每条命令同样可以正常运行。

---
## 1. 环境检查清单
请确保运行环境已安装以下依赖库：
- `torch`
- `torchvision`
- `transformers`
- `timm`
- `scikit-learn`
- `opencv-python`
- `Pillow`
- `matplotlib`
- `numpy`

---
## 2. 实验一：CLIP ViT-bigG / EVA-Giant + LoRA
### 实验目标
冻结骨干网络，仅训练 LoRA 参数与分类头。训练阶段加入模糊增强，随后在干净数据集与模糊基准数据集上完成评估测试。

### 结果记录规则
你可以自主选择实验记录指标的依据：
- 训练过程最优精度 `best`
- 最后一轮 epoch 精度 `last`

切换参数配置：
- `--report-checkpoint best`
- `--report-checkpoint last`

训练脚本会自动保存以下模型文件：`best_lora_model.pth`、`last_lora_model.pth`、`latest_lora_model.pth` 以及 `selected_lora_model.pth`。如需模型文件与记录规则保持一致，后续可使用 `selected_lora_model.pth`。

### 默认模糊配置
默认参数与原始运动模糊训练逻辑保持统一：
- `blur_prob = 0.1`
- `blur_mode = mixed`
- `blur_type = motion`
- `mixed_mode_ratio = 0.5`

---
### 训练 CLIP ViT-bigG（记录最优模型 best）
```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_lora_blur.py \
  --model-family clip_lora \
  --backbone-path /nas_train/app.e0016372/models/blur_generalization_hf_backbones/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
  --train-root /data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4 \
  --ccmba-data-dir /home/work/xueyunqi/11ar_datasets/progan_ccmba_train \
  --blur-mode mixed \
  --blur-type motion \
  --blur-prob 0.1 \
  --blur-min 0.1 \
  --blur-max 0.3 \
  --mixed-mode-ratio 0.5 \
  --report-checkpoint best \
  --local-files-only
```

---
### 训练 CLIP ViT-bigG（记录最后一轮模型 last）
```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_lora_blur.py \
  --model-family clip_lora \
  --backbone-path /nas_train/app.e0016372/models/blur_generalization_hf_backbones/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
  --train-root /data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4 \
  --ccmba-data-dir /home/work/xueyunqi/11ar_datasets/progan_ccmba_train \
  --blur-mode mixed \
  --blur-type motion \
  --blur-prob 0.1 \
  --blur-min 0.1 \
  --blur-max 0.3 \
  --mixed-mode-ratio 0.5 \
  --report-checkpoint last \
  --local-files-only
```

---
### 训练 EVA-Giant
```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_lora_blur.py \
  --model-family eva_giant_lora \
  --backbone-path /nas_train/app.e0016372/models/blur_generalization_hf_backbones/timm/eva_giant_patch14_336.m30m_ft_in22k_in1k \
  --train-root /data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4 \
  --ccmba-data-dir /home/work/xueyunqi/11ar_datasets/progan_ccmba_train \
  --blur-mode mixed \
  --blur-type motion \
  --blur-prob 0.1 \
  --blur-min 0.1 \
  --blur-max 0.3 \
  --mixed-mode-ratio 0.5 \
  --report-checkpoint best \
  --local-files-only
```

---
### 训练输出文件
每个 LoRA 实验目录将生成以下文件：
- `best_lora_model.pth`
- `last_lora_model.pth`
- `latest_lora_model.pth`
- `selected_lora_model.pth`
- `training_history.json`

`training_history.json` 新增记录字段：
- `train_acc`
- `train_bacc`
- `train_real_acc`
- `train_fake_acc`

---
## 3. 实验一评估：跨数据集 干净/模糊 测试
```bash
python blur_generalization_suite/eval_cross_dataset.py \
  --model-path blur_generalization_suite/outputs/lora_train/.../selected_lora_model.pth \
  --dataset-group all \
  --blur-mode both \
  --blur-type motion \
  --blur-min 0.1 \
  --blur-max 0.3
```

---
### 评估输出文件
评估目录将生成：
- `cross_dataset_eval_<timestamp>.json`
- `cross_dataset_summary_<timestamp>.csv`
- `table4_clean_accuracy.csv`
- `table5_blur_accuracy.csv`
- `table6_clean_f1.csv`
- `table7_blur_f1.csv`
- `clean_bacc.csv`
- `blur_bacc.csv`

---
## 4. 实验二：师生骨干网络遍历实验
### SimCLR / 对比对齐
学生阶段的对比损失权重可通过 `--alpha-simclr` 显式设置。默认值保持 `0.3`，与原始 `train_motion.py` 学生阶段配置一致。

### 训练 DINOv3 ViT-Large-300M
```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_teacher_student_backbones.py \
  --backbone-preset dinov3_vitl300m \
  --data-preset sdv14 \
  --blur-mode mixed \
  --blur-type motion \
  --blur-prob 0.1 \
  --alpha-simclr 0.3
```

### 训练 DINOv3 ViT-Huge-840M
```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_teacher_student_backbones.py \
  --backbone-preset dinov3_vith840m \
  --data-preset sdv14 \
  --blur-mode mixed \
  --blur-type motion \
  --blur-prob 0.1 \
  --alpha-simclr 0.3
```

### 在 AIGC 基准数据集评估学生模型
```bash
python blur_generalization_suite/eval_teacher_student_aigc.py \
  --model-path blur_generalization_suite/outputs/teacher_student/.../best_student_model.pth \
  --branch student \
  --dataset-group aigc_benchmark \
  --blur-mode both \
  --blur-type motion
```

师生模型评估流程现已支持输出与 LoRA 评估一致的平衡准确率指标，包含 `clean_bacc.csv` 和 `blur_bacc.csv`。

---
## 5. 实验三：DINOv3 模糊一致性分析
### 默认分析参数范围
- `min_blur = 0.0`
- `max_blur = 0.5`
- `step = 0.1`

### 运行一致性分析脚本
```bash
python blur_generalization_suite/analyze_dinov3_blur_consistency.py \
  --model-path /nas_train/app.e0016372/models/dinov3-vit7b16-pretrain-lvd1689m \
  --data-root /data/app.e0016372/11ar_datasets/test/wukong/1_fake \
  --blur-type motion \
  --min-blur 0.0 \
  --max-blur 0.5 \
  --step 0.1 \
  --max-images 64 \
  --local-files-only
```

### 输出文件
- `per_image_consistency.json`
- `average_consistency.json`
- `dinov3_consistency_curve.png`

---
## 6. 新增实验：去模糊 -> 检测模型链路
### 实验目标
对比三类检测器在无模糊、模糊、模糊后去模糊三种场景下的性能：
- `DINOv3 student`
- `CLIP LoRA`
- `EVA-Giant LoRA`

### 基准数据集默认目录结构
- `own_benchmark/0_real`
- `own_benchmark/1_fake`

### 运行去模糊基准测试
```bash
python blur_generalization_suite/eval_deblur_benchmark.py \
  --dataset-name own_benchmark \
  --dinov3-student-path blur_generalization_suite/outputs/teacher_student/.../best_student_model.pth \
  --clip-lora-path blur_generalization_suite/outputs/lora_train/.../selected_lora_model.pth \
  --eva-lora-path blur_generalization_suite/outputs/lora_train/.../selected_lora_model.pth \
  --pipelines no_blur blur blur_then_deblur \
  --blur-type motion \
  --blur-strength 0.3 \
  --deblur-regularization 0.01
```

### 使用自定义数据集目录
```bash
python blur_generalization_suite/eval_deblur_benchmark.py \
  --real-folder /your/path/0_real \
  --fake-folder /your/path/1_fake \
  --dinov3-student-path ... \
  --clip-lora-path ... \
  --eva-lora-path ...
```

### 输出文件
- `deblur_benchmark_eval_<timestamp>.json`
- `deblur_benchmark_summary_<timestamp>.csv`

---
## 7. 指标说明补充
平衡准确率定义规则：
- `real_accuracy`：真实样本识别准确率
- `fake_accuracy`：伪造样本识别准确率
- `bacc = (real_accuracy + fake_accuracy) / 2`
- `balanced_accuracy_half_gap = |real_accuracy - fake_accuracy| / 2`

标准报告格式：
- `bacc +- balanced_accuracy_half_gap`

---
## 8. 最终规范建议
为保证开源发布与实验报告统一规范，请遵守以下要求：
1. 实验一统一使用 `selected_lora_model.pth` 作为评估模型；
2. 运行命令务必显式标注 `--report-checkpoint best` 或 `--report-checkpoint last`；
3. 师生网络实验强制写明 `--alpha-simclr 0.3` 参数；
4. 模糊一致性曲线图注需标注参数范围：`0.0 -> 0.5`，步长 `0.1`；
5. 去模糊实验说明中需注明：当前实现基于经典 Wiener 去模糊基线算法。