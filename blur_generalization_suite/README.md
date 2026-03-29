# 模糊泛化工具套件
该文件夹包含整理优化后的实验代码，基于原始的`train_motion.py` 和`0125_compensation/test_for_wildrf.py` 两套流程开发而来。
本工具套件目前包含四大实验组：
1. 骨干网络冻结：`CLIP ViT-bigG + LoRA` 和`EVA02-Large/14-448 + LoRA`
2. `DINOv3 ViT-Large-300M / ViT-Huge-840M` 师生骨干网络对照实验
3. 不同模糊强度的`DINOv3` 特征一致性可视化分析
4. 基于 `own_benchmark` 数据集的「去模糊 -> 检测模型」对比实验
## 新增功能说明
### LoRA 训练模块
- `train_lora_blur.py` 现已支持参数 `--report-checkpoint best|last`
- 自动保存模型：`best_lora_model.pth`、`last_lora_model.pth`、`latest_lora_model.pth`、`selected_lora_model.pth`
- `selected_lora_model.pth` 会严格跟随你选定的记录规则保存
- 训练日志新增记录：`train_bacc`、`train_real_acc`、`train_fake_acc`

### 评估指标扩展
`eval_cross_dataset.py` 与`eval_teacher_student_aigc.py` 现已支持导出以下指标：
- `accuracy`
- `bacc`
- `real_accuracy`
- `fake_accuracy`
- `balanced_accuracy_half_gap`
- `precision`
- `recall`
- `f1_score`

同时会自动生成导出文件：`clean_bacc.csv` 与`blur_bacc.csv` 

### 师生对齐优化
`train_teacher_student_backbones.py` 会将 `alpha_simclr` 参数显式存入配置文件中，

默认值与原版学生训练阶段保持一致：`alpha_simclr = 0.3`

### 模糊一致性分析
默认分析区间已设定为：`0.0 ~ 0.5`，步长`step = 0.1`

### 去模糊基线算法
新增脚本 `eval_deblur_benchmark.py`，为 **模糊 -> 去模糊-> 检测模型** 流程补充经典 `Wiener deconvolution` 维纳反卷积基线算法。
## 核心文件说明
- `common.py`：分布式通用工具、权重加载、指标计算、通用函数
- `data_utils.py`：数据集加载、模糊变换、归一化工具与维纳去模糊算法
- `dataset_configs.py`：跨数据集配置与 `own_benchmark` 自定义数据集配置
- `model_zoo.py`：LoRA 骨干网络与 DINOv3 师生网络定义
- `train_lora_blur.py`：第一组实验训练脚本
- `eval_cross_dataset.py`：第一组实验评估脚本
- `train_teacher_student_backbones.py`：第二组实验训练脚本
- `eval_teacher_student_aigc.py`：第二组实验评估脚本
- `analyze_dinov3_blur_consistency.py`：第三组一致性分析脚本
- `eval_deblur_benchmark.py`：第四组去模糊对比实验脚本
## 默认路径配置
### LoRA 实验路径
- 训练数据集根目录：`/data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4`
- CCMBA 数据集根目录：`/home/work/xueyunqi/11ar_datasets/progan_ccmba_train`
- 本地 HF 模型缓存根目录：`/nas_train/app.e0016372/models/blur_generalization_hf_backbones`

你也可以通过环境变量 `BLUR_GENERALIZATION_HF_BACKBONE_ROOT` 自定义修改骨干模型根路径。

### 师生实验预设路径
- `original_motion`：`/home/work/xueyunqi/11ar_datasets/extracted` + `/home/work/xueyunqi/11ar_datasets/progan_ccmba_train`
- `sdv14`：`/data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4` + `/data/app.e0016372/imagenet_tmp/ccmba_processed_sdv44`

### 去模糊基准数据集
默认数据集目录结构：
- `own_benchmark/0_real`
- `own_benchmark/1_fake`

## 快速启动命令
### CLIP ViT-bigG + LoRA 训练
```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_lora_blur.py \
  --model-family clip_lora \
  --blur-mode mixed \
  --blur-type motion \
  --blur-prob 0.1 \
  --report-checkpoint best
```

### EVA02-Large + LoRA 训练
```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_lora_blur.py \
  --model-family eva02_large_lora \
  --blur-mode mixed \
  --blur-type motion \
  --blur-prob 0.1 \
  --report-checkpoint best
```

### LoRA 跨数据集评估
```bash
python blur_generalization_suite/eval_cross_dataset.py \
  --model-path blur_generalization_suite/outputs/lora_train/.../selected_lora_model.pth \
  --dataset-group all \
  --blur-mode both
```

### 师生骨干网络对照训练
```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_teacher_student_backbones.py \
  --backbone-preset dinov3_vitl300m \
  --data-preset sdv14 \
  --blur-prob 0.1 \
  --alpha-simclr 0.3
```

### DINOv3 模糊一致性分
```bash
python blur_generalization_suite/analyze_dinov3_blur_consistency.py \
  --data-root /data/app.e0016372/11ar_datasets/test/wukong/1_fake \
  --blur-type motion \
  --min-blur 0.0 \
  --max-blur 0.5 \
  --step 0.1
```

### 去模糊基准实验
```bash
python blur_generalization_suite/eval_deblur_benchmark.py \
  --dataset-name own_benchmark \
  --dinov3-student-path blur_generalization_suite/outputs/teacher_student/.../best_student_model.pth \
  --clip-lora-path blur_generalization_suite/outputs/lora_train/.../selected_lora_model.pth \
  --eva-lora-path blur_generalization_suite/outputs/lora_train/.../selected_lora_model.pth \
  --pipelines no_blur blur blur_then_deblur \
  --blur-type motion \
  --blur-strength 0.3
```

## 指标定义
平衡准确率计算公式：
- `real_accuracy`：真实样本检测准确率
- `fake_accuracy`：伪造样本检测准确率
- `bacc = (real_accuracy + fake_accuracy) / 2`
- `balanced_accuracy_half_gap = |real_accuracy - fake_accuracy| / 2`

标准书写报告格式：
- `bacc +- balanced_accuracy_half_gap`

## 补充说明
目前去模糊实验仅采用最简易经典的 Wiener 基线算法
该模块仅作为干净公平的对比基准使用，**并非深度学习前沿最优去模糊模型**。

