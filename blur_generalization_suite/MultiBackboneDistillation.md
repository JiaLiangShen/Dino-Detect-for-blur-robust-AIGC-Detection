# 多骨干师生蒸馏实验说明（SigLIP 2 / AIMv2 vs DINOv3）

本文档只描述本次**新增的"多家族骨干蒸馏对照"实验**，包括选型动机、权重放置、训练命令、评估命令、checkpoint 内容以及结果解读建议。不涉及原有的 `CLIP ViT-bigG + LoRA`、`EVA02-Large + LoRA`、模糊一致性分析与去模糊基线这四组旧实验（这些请继续参考 [README.md](README.md) 与 [HowToRun.md](HowToRun.md)）。

---

## 1. 为什么做这个补充实验

原有的"师生骨干对照"只覆盖了 DINOv3 家族（ViT-L/300M、ViT-H+/840M），结论**只能解释"DINO 自监督信号对模糊鲁棒性的影响"，不能回答以下问题**：

1. 同样在 300M / 840M 量级上，**图文对比预训练**（CLIP 家族）和**纯 ViT 预训练**（DINO / AIM）之间，模糊泛化能力哪个更优？
2. DINOv3 在该任务上的优势是**骨干架构本身**带来的，还是**DINO 特有的自监督训练范式**带来的？
3. 当把其他家族的骨干按同一套师生蒸馏框架（带 blur-aware 增强 + SimCLR 对比对齐）训练后，AIGC 检测的 clean / blur bacc 表现是否会明显拉开差距？

为此在**完全不改动损失函数、训练节奏、超参**的前提下，新增了 4 个 2024-11 ~ 2025-02 发布的骨干预设，在师生框架里作为 **teacher 同时 student 的共享特征提取器**参与 apples-to-apples 对照：

| 档位 | DINO 基线（原有） | CLIP 家族（新增） | 纯 ViT 家族（新增） |
|---|---|---|---|
| **~300M** | `dinov3_vitl300m` — ViT-L/16 (~300M) | `siglip2_vitl300m` — SigLIP 2 Large/16@256 (~303M 视觉塔, Google 2025-02) | `aimv2_vitl300m` — AIMv2 Large/14@224 (~308M, Apple 2024-11) |
| **~840M** | `dinov3_vith840m` — ViT-H+/16 (~840M) | `siglip2_giantopt_1b` — SigLIP 2 Giant-OPT/16@256 (~1.1B 视觉塔, 2025-02，840M 的**上界**) | `aimv2_vith680m` — AIMv2 Huge/14@336 (~682M, 2024-11，840M 的**下界**) |
| 超大 | `dinov3_vit7b` — ViT-7B (~6.7B) | — | — |

> 840M 档在开源 CLIP/ViT 里没有严格对齐的权重，AIMv2 Huge (682M) 和 SigLIP 2 Giant-OPT (1.1B) 正好把 DINOv3 840M 夹住；论文里同时报两个即可。

---

## 2. 权重下载与放置

所有新骨干继续使用和 LoRA 实验相同的**本地 HF 镜像根目录**，由环境变量 `BLUR_GENERALIZATION_HF_BACKBONE_ROOT` 控制（默认 `/nas_train/app.e0016372/models/blur_generalization_hf_backbones`）。从 HuggingFace 下载后目录结构保持如下：

```
${BLUR_GENERALIZATION_HF_BACKBONE_ROOT}/
  google/siglip2-large-patch16-256/          # siglip2_vitl300m
  google/siglip2-giant-opt-patch16-256/      # siglip2_giantopt_1b
  apple/aimv2-large-patch14-224/             # aimv2_vitl300m
  apple/aimv2-huge-patch14-336/              # aimv2_vith680m
```

每个目录下需要包含 `config.json`、`preprocessor_config.json`（如有）、`model.safetensors` 或 `pytorch_model.bin`。**AIMv2 需要 `trust_remote_code=True`，请确认目录下的自定义 `modeling_*.py` 已保留**。

下载示例（三选一）：

```bash
# 方案 A: huggingface-cli
huggingface-cli download google/siglip2-large-patch16-256 --local-dir ${BLUR_GENERALIZATION_HF_BACKBONE_ROOT}/google/siglip2-large-patch16-256
huggingface-cli download google/siglip2-giant-opt-patch16-256 --local-dir ${BLUR_GENERALIZATION_HF_BACKBONE_ROOT}/google/siglip2-giant-opt-patch16-256
huggingface-cli download apple/aimv2-large-patch14-224 --local-dir ${BLUR_GENERALIZATION_HF_BACKBONE_ROOT}/apple/aimv2-large-patch14-224
huggingface-cli download apple/aimv2-huge-patch14-336 --local-dir ${BLUR_GENERALIZATION_HF_BACKBONE_ROOT}/apple/aimv2-huge-patch14-336

# 方案 B: git lfs clone
git lfs clone https://huggingface.co/google/siglip2-large-patch16-256      ${BLUR_GENERALIZATION_HF_BACKBONE_ROOT}/google/siglip2-large-patch16-256
git lfs clone https://huggingface.co/apple/aimv2-huge-patch14-336          ${BLUR_GENERALIZATION_HF_BACKBONE_ROOT}/apple/aimv2-huge-patch14-336
# ...其余同理
```

---

## 3. 代码改动速查（只列与本实验有关）

| 文件 | 关键改动 |
|---|---|
| [model_zoo.py](model_zoo.py) | 新增 `DistillationBackboneSpec` + `DISTILLATION_BACKBONE_SPECS`；`DEFAULT_PREPROCESS` 注册 `siglip2` / `aimv2` / `clip` 三种默认归一化；新增 `_load_distillation_backbone` / `_extract_backbone_features` 处理四种家族；`ImprovedDinoV3Adapter` 与 `TeacherStudentNetwork` 支持可选的 `backbone_family`（默认 `"dinov3"`，向后兼容）。 |
| [train_teacher_student_backbones.py](train_teacher_student_backbones.py) | `--backbone-preset` 扩展为 7 个预设；新增 `--backbone-family` 覆盖；`build_dino_transforms` 按 preset 自动取对应家族的 resize / 归一化；checkpoint config 写入 `backbone_family` 与 `backbone_preset`。 |
| [eval_teacher_student_aigc.py](eval_teacher_student_aigc.py) / [eval_deblur_benchmark.py](eval_deblur_benchmark.py) / [own_batch_eval.py](own_batch_eval.py) | 加载 checkpoint 时读 `config["backbone_family"]`（默认 `"dinov3"`），一路透传给 `TeacherStudentNetwork`；导出的 metadata 里同步记录 `backbone_family` / `backbone_preset`。 |

**所有损失（CE + KL 蒸馏 + 特征对齐 + SimCLR）、温度 `T=0.07`、默认 `alpha_simclr=0.3`、模糊采样策略、teacher/student 两阶段节奏都未改动**，确保四家族对照的可比性。

---

## 4. 训练（四条完整命令）

保持和 DINOv3 学生阶段一致的模糊设置：`blur_mode=mixed`、`blur_type=motion`、`blur_prob=0.1`、`blur_min=0.1`、`blur_max=0.3`、`mixed_mode_ratio=0.5`、`alpha_simclr=0.3`。

### 4.1 SigLIP 2 Large（~300M · CLIP 家族）
```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_teacher_student_backbones.py \
  --backbone-preset siglip2_vitl300m \
  --data-preset sdv14 \
  --blur-mode mixed \
  --blur-type motion \
  --blur-prob 0.1 \
  --alpha-simclr 0.3
```

### 4.2 AIMv2 Large（~300M · 纯 ViT 家族）
```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_teacher_student_backbones.py \
  --backbone-preset aimv2_vitl300m \
  --data-preset sdv14 \
  --blur-mode mixed \
  --blur-type motion \
  --blur-prob 0.1 \
  --alpha-simclr 0.3
```

### 4.3 AIMv2 Huge（~840M 档 · ViT 家族，682M 为 840M 的下界）
```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_teacher_student_backbones.py \
  --backbone-preset aimv2_vith680m \
  --data-preset sdv14 \
  --blur-mode mixed \
  --blur-type motion \
  --blur-prob 0.1 \
  --alpha-simclr 0.3
```

### 4.4 SigLIP 2 Giant-OPT（~840M 档 · CLIP 家族，1.1B 为 840M 的上界）
```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_teacher_student_backbones.py \
  --backbone-preset siglip2_giantopt_1b \
  --data-preset sdv14 \
  --blur-mode mixed \
  --blur-type motion \
  --blur-prob 0.1 \
  --alpha-simclr 0.3
```

### 训练默认保存
输出目录：`blur_generalization_suite/outputs/teacher_student/<backbone_name>_<data_preset>_teacher_student_blur<prob>/`
产物：
- `best_teacher_model.pth` / `latest_teacher_model.pth` — teacher 阶段投影/分类头
- `best_student_model.pth` / `latest_teacher_student_model.pth` — student 阶段投影/分类头（**评估就用这个**）
- `training_history.json` — 两阶段完整 loss / acc 历史

### 常用覆盖参数
- `--resize-size` / `--crop-size`：覆盖 preset 默认分辨率（如想把 SigLIP 2 Large 从 256 拉到 384，传 `--resize-size 384 --crop-size 384`，前提是你下载的是 `siglip2-large-patch16-384`）。
- `--backbone-family`：一般不用；当传的是非预设的自定义路径时用来告诉脚本骨干家族。
- `--teacher-batch-size` / `--student-batch-size`：对 840M 档显存吃紧时，建议 `--student-batch-size 16 --teacher-batch-size 32` 起步。
- `--teacher-epochs` / `--student-epochs`：默认 8 / 15，保持与 DINOv3 一致。

---

## 5. 评估

三个评估脚本会**自动**从 checkpoint `config["backbone_family"]` 读出骨干家族并重建网络，**无需手动再指定 preset**。

### 5.1 AIGCBenchmark 跨数据集评估（主表数据）
```bash
# SigLIP 2 Large 学生
python blur_generalization_suite/eval_teacher_student_aigc.py \
  --model-path blur_generalization_suite/outputs/teacher_student/siglip2-large-patch16-256_sdv14_teacher_student_blur01/best_student_model.pth \
  --branch student \
  --dataset-group aigc_benchmark \
  --blur-mode both \
  --blur-type motion

# AIMv2 Huge 学生
python blur_generalization_suite/eval_teacher_student_aigc.py \
  --model-path blur_generalization_suite/outputs/teacher_student/aimv2-huge-patch14-336_sdv14_teacher_student_blur01/best_student_model.pth \
  --branch student \
  --dataset-group aigc_benchmark \
  --blur-mode both \
  --blur-type motion
```

其余两个骨干把 `--model-path` 换成对应目录即可。每次会在 checkpoint 旁边生成：
- `aigc_eval_<timestamp>.json` — 原始指标 + 完整 config
- `aigc_eval_summary_<timestamp>.csv` — 每数据集 × 每 blur_mode 一行，含 accuracy / bacc / real_acc / fake_acc / gap / precision / recall / f1
- `clean_bacc.csv` / `blur_bacc.csv` — 和 LoRA 实验对齐的平衡准确率导出

### 5.2 去模糊基线对照（可选，用于 own_benchmark）
```bash
python blur_generalization_suite/eval_deblur_benchmark.py \
  --dataset-name own_benchmark \
  --dinov3-student-path blur_generalization_suite/outputs/teacher_student/<任何骨干目录>/best_student_model.pth \
  --clip-lora-path     blur_generalization_suite/outputs/lora_train/.../selected_lora_model.pth \
  --eva-lora-path      blur_generalization_suite/outputs/lora_train/.../selected_lora_model.pth \
  --pipelines no_blur blur blur_then_deblur \
  --blur-type motion \
  --blur-strength 0.3
```
> 注意：`--dinov3-student-path` 这个参数名是历史遗留，其实可以传**任意家族**的师生 checkpoint；脚本会按 checkpoint 里的 `backbone_family` 自动重建对应骨干。

### 5.3 own_benchmark 批量评估
```bash
python blur_generalization_suite/own_batch_eval.py \
  --model-paths \
    blur_generalization_suite/outputs/teacher_student/siglip2-large-patch16-256_sdv14_teacher_student_blur01/best_student_model.pth \
    blur_generalization_suite/outputs/teacher_student/aimv2-large-patch14-224_sdv14_teacher_student_blur01/best_student_model.pth \
    blur_generalization_suite/outputs/teacher_student/aimv2-huge-patch14-336_sdv14_teacher_student_blur01/best_student_model.pth \
    blur_generalization_suite/outputs/teacher_student/siglip2-giant-opt-patch16-256_sdv14_teacher_student_blur01/best_student_model.pth \
    blur_generalization_suite/outputs/teacher_student/dinov3-vitl16-pretrain-lvd1689m_sdv14_teacher_student_blur01/best_student_model.pth \
    blur_generalization_suite/outputs/teacher_student/dinov3-vith16plus-pretrain-lvd1689m_sdv14_teacher_student_blur01/best_student_model.pth
```
（具体参数请参考 `own_batch_eval.py` 的 argparse；不同家族的 checkpoint 可以一起传，脚本会各自按对应 preprocess 构造 eval transform。）

---

## 6. checkpoint 内容说明

每个 `best_student_model.pth` 里的 `config` 字段现在包含：

| 字段 | 含义 |
|---|---|
| `dinov3_model_id` | 骨干权重的本地路径（历史遗留字段名，实际对所有家族通用） |
| `backbone_family` | `"dinov3"` / `"siglip2"` / `"aimv2"` / `"clip"` — 评估脚本据此重建网络 |
| `backbone_preset` | 训练时的 `--backbone-preset` 值，例如 `"siglip2_vitl300m"` |
| `projection_dim` | 投影头维度（默认 512） |
| `transform_config` | `{resize_size, crop_size, mean, std}` — 评估时的图像预处理 |
| `alpha_simclr` | SimCLR 对比对齐权重（默认 0.3） |
| `blur_mode` / `blur_type` / `blur_prob` / `blur_strength_range` / `mixed_mode_ratio` | 模糊增强配置 |
| `local_files_only` | 是否强制本地加载（默认 True） |

**旧的 DINOv3 checkpoint 没有 `backbone_family` 字段不会有任何影响**——所有评估脚本在缺省时默认按 `"dinov3"` 处理。

---

## 7. 结果解读建议

建议按如下维度整理实验结果（均为 `student` 分支、SDv1.4 训练、AIGCBenchmark 评估）：

### 7.1 主表：300M 档四选一 + 840M 档四选一
| Backbone | Clean bacc ± gap/2 | Motion-blur bacc ± gap/2 | Δ (clean - blur) |
|---|---|---|---|
| DINOv3 ViT-L/300M | … | … | … |
| SigLIP 2 Large/300M | … | … | … |
| AIMv2 Large/308M | … | … | … |
| DINOv3 ViT-H+/840M | … | … | … |
| AIMv2 Huge/682M | … | … | … |
| SigLIP 2 Giant-OPT/1.1B | … | … | … |

`bacc ± gap/2` 的取数直接来自 `aigc_eval_summary_<timestamp>.csv` 的 `bacc` 和 `balanced_accuracy_half_gap` 两列。

### 7.2 关键对照角度
1. **同量级跨家族**（300M 档横向对比 DINO vs SigLIP 2 vs AIMv2）：回答"图文对比 vs 纯 ViT vs DINO 自监督哪种预训练在 AIGC 检测上更鲁棒"。
2. **同家族跨量级**（例如 AIMv2 Large 308M → Huge 682M 的 Δ）：回答"同一预训练范式下参数量增加是否带来模糊鲁棒性红利"。
3. **Δ (clean - blur)**：这个差值越小，说明骨干对模糊的**本质鲁棒性**越强（而不是靠 clean 高 baseline 刷上去的）。
4. **real_accuracy 与 fake_accuracy 的 gap**（`balanced_accuracy_half_gap`）：衡量骨干在真假决策边界上的偏置。

---

## 8. 常见问题

**Q1：SigLIP 2 Giant-OPT 显存爆了怎么办？**
A：先把 `--student-batch-size` 降到 8 或 4；如果仍然不行，加 `--resize-size 224 --crop-size 224` 把分辨率降下来（精度会略掉，但架构和家族对照依然有效）。

**Q2：我传了 `--backbone-preset aimv2_vitl300m` 但脚本报 `trust_remote_code` 相关错误？**
A：AIMv2 的 `modeling_*.py` 没被同步下来。用 `huggingface-cli download` 时务必保留全部 `.py` 文件，或者直接 `git lfs clone` 整个仓库。

**Q3：旧的 DINOv3 checkpoint 还能用吗？**
A：能。所有评估脚本在 `config` 中找不到 `backbone_family` 时默认按 `"dinov3"` 加载，与原版行为完全一致。

**Q4：我想加自己的骨干（比如 MetaCLIP 2 / PE Core / DFN CLIP）？**
A：只要满足"HF AutoModel 能加载 + 输出含 `pooler_output` 或 `last_hidden_state`"，在 [model_zoo.py](model_zoo.py) 的 `DISTILLATION_BACKBONE_SPECS` 字典里增加一个 `DistillationBackboneSpec` 条目（填 `local_dir` / `backbone_family` / `preprocess` / `description`）即可，训练脚本会自动识别新 preset，不用改其他文件。`backbone_family` 在 `"dinov3"` / `"siglip2"` / `"aimv2"` / `"clip"` 中选一个最接近的语义即可（主要影响加载路径与特征提取池化方式）。

**Q5：蒸馏框架里的"师生"和"teacher 大模型蒸馏到 student 小模型"是一回事吗？**
A：**不是**。本项目里 teacher 与 student **共享同一个冻结骨干**，区别在于 student 额外经过 blur-aware 增强 + SimCLR 对齐训练了自己的投影头和分类头。所以这里对 CLIP/ViT 骨干做"蒸馏"指的是用**同一个 SigLIP 2 / AIMv2** 同时扮演 teacher 和 student backbone，评估的是它作为**模糊鲁棒特征提取器**的能力，而不是真的把它蒸到更小的网络里。
