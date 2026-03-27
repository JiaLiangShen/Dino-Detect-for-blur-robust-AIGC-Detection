下面我按“真的要把这 3 组实验跑起来”的视角，完整带你走一遍。

先说两个总原则。

1. 最好不要直接改脚本里的默认地址，优先用命令行参数覆盖。
2. 真正需要改文件的地方，主要只有两个：
   - 测试集路径不一致时，改 [dataset_configs.py](E:/薛云起研究生阶段其他文件/2025暑期实习/最终百度实习大模型/dino-ditect_all_codes/Dino-Detect-for-blur-robust-AIGC-Detection-main/blur_generalization_suite/dataset_configs.py)
   - 你想永久改默认训练预设时，改 [train_teacher_student_backbones.py](E:/薛云起研究生阶段其他文件/2025暑期实习/最终百度实习大模型/dino-ditect_all_codes/Dino-Detect-for-blur-robust-AIGC-Detection-main/blur_generalization_suite/train_teacher_student_backbones.py#L43)

我下面默认你是在仓库根目录运行命令：
`E:/薛云起研究生阶段其他文件/2025暑期实习/最终百度实习大模型/dino-ditect_all_codes/Dino-Detect-for-blur-robust-AIGC-Detection-main`

下面命令按 `bash/Linux` 写。如果你在 `PowerShell` 里跑，把换行反斜杠 `\` 去掉，写成一行就行。

---

**先准备环境**
1. 确保环境里有这些包：`torch`、`torchvision`、`transformers`、`scikit-learn`、`opencv-python`、`Pillow`、`matplotlib`、`numpy`。
2. 确保你能 `torchrun --nproc_per_node=8` 正常起分布式训练。如果你不是 8 卡，就把 `8` 改成你实际 GPU 数。
3. 先检查模型和数据目录是否存在。

---

**实验 1：CLIP + LoRA、ViT-Large + LoRA，固定 backbone，只做 blur augmentation，不做 distill**

参数入口在 [train_lora_blur.py](E:/薛云起研究生阶段其他文件/2025暑期实习/最终百度实习大模型/dino-ditect_all_codes/Dino-Detect-for-blur-robust-AIGC-Detection-main/blur_generalization_suite/train_lora_blur.py#L260) 和 [eval_cross_dataset.py](E:/薛云起研究生阶段其他文件/2025暑期实习/最终百度实习大模型/dino-ditect_all_codes/Dino-Detect-for-blur-robust-AIGC-Detection-main/blur_generalization_suite/eval_cross_dataset.py#L91)。

1. 你要准备的数据和模型
   - 训练集根目录。
   - 当前脚本默认是：`/data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4`
   - 这个目录要满足“分类子目录 -> `0_real/1_fake`”或“分类子目录 -> `nature/ai`”结构。
   - CCMBA 模糊增强目录。
   - 当前默认是：`/home/work/xueyunqi/11ar_datasets/progan_ccmba_train`
   - 这个目录要满足“分类子目录 -> `nature` 或 `ai` -> `blurred_images` / `blur_masks` / `metadata`”结构。
   - CLIP backbone 本地目录。
   - 当前默认在 [model_zoo.py](E:/薛云起研究生阶段其他文件/2025暑期实习/最终百度实习大模型/dino-ditect_all_codes/Dino-Detect-for-blur-robust-AIGC-Detection-main/blur_generalization_suite/model_zoo.py) 里是：`/nas_train/app.e0016372/models/clip-vit-large-patch14`
   - ViT-Large backbone 本地目录。
   - 当前默认是：`/nas_train/app.e0016372/models/vit-large-patch16-224-in21k`
   - cross-dataset 测试集路径。
   - 这些都写死在 [dataset_configs.py](E:/薛云起研究生阶段其他文件/2025暑期实习/最终百度实习大模型/dino-ditect_all_codes/Dino-Detect-for-blur-robust-AIGC-Detection-main/blur_generalization_suite/dataset_configs.py) 里。

2. 你需要改什么地址
   - 如果训练集不是 `/data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4`，训练时传 `--train-root`。
   - 如果 CCMBA 不是 `/home/work/xueyunqi/11ar_datasets/progan_ccmba_train`，训练时传 `--ccmba-data-dir`。
   - 如果 CLIP / ViT-Large 的本地模型目录不同，训练时传 `--backbone-path`。
   - 如果测试集路径不对，只能改 [dataset_configs.py](E:/薛云起研究生阶段其他文件/2025暑期实习/最终百度实习大模型/dino-ditect_all_codes/Dino-Detect-for-blur-robust-AIGC-Detection-main/blur_generalization_suite/dataset_configs.py)。

3. 你要不要改 blur 配置
   - 当前 LoRA 脚本默认是 `mixed + motion + blur_prob=0.2 + range=(0.1,0.3) + mixed_ratio=0.5`。
   - 如果你要“严格跟原 `train_motion.py` 的 motion teacher-student 模糊比例一致”，建议把 `--blur-prob 0.1` 显式写上。
   - 如果你要“跟你 0125 的 direct-train 风格一致”，就用默认 `0.2`。
   - 你这组实验最重要的是：`CLIP` 和 `ViT-Large` 必须用同一套 blur 参数，不要一个 0.1 一个 0.2。

4. 推荐命令
   - 如果你要对齐原 `train_motion.py` 风格，我建议这组都用 `blur_prob=0.1`。

```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_lora_blur.py \
  --model-family clip_lora \
  --backbone-path /nas_train/app.e0016372/models/clip-vit-large-patch14 \
  --train-root /data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4 \
  --ccmba-data-dir /home/work/xueyunqi/11ar_datasets/progan_ccmba_train \
  --blur-mode mixed \
  --blur-type motion \
  --blur-prob 0.1 \
  --blur-min 0.1 \
  --blur-max 0.3 \
  --mixed-mode-ratio 0.5 \
  --local-files-only
```

```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_lora_blur.py \
  --model-family vit_large_lora \
  --backbone-path /nas_train/app.e0016372/models/vit-large-patch16-224-in21k \
  --train-root /data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4 \
  --ccmba-data-dir /home/work/xueyunqi/11ar_datasets/progan_ccmba_train \
  --blur-mode mixed \
  --blur-type motion \
  --blur-prob 0.1 \
  --blur-min 0.1 \
  --blur-max 0.3 \
  --mixed-mode-ratio 0.5 \
  --local-files-only
```

5. 训练完成后去哪找模型
   - 输出目录默认在：
   - `blur_generalization_suite/outputs/lora_train/`
   - 每个实验会生成一个子目录，命名规则类似：
   - `clip_lora_clip-vit-large-patch14_blur01`
   - `vit_large_lora_vit-large-patch16-224-in21k_blur01`
   - 里面重点看：
   - `best_lora_model.pth`
   - `training_history.json`

6. 怎么测试 clean / blur 和导出 Table 4/5/6/7
   - 用 [eval_cross_dataset.py](E:/薛云起研究生阶段其他文件/2025暑期实习/最终百度实习大模型/dino-ditect_all_codes/Dino-Detect-for-blur-robust-AIGC-Detection-main/blur_generalization_suite/eval_cross_dataset.py#L91)

```bash
python blur_generalization_suite/eval_cross_dataset.py \
  --model-path blur_generalization_suite/outputs/lora_train/clip_lora_clip-vit-large-patch14_blur01/best_lora_model.pth \
  --dataset-group all \
  --blur-mode both \
  --blur-type motion \
  --blur-min 0.1 \
  --blur-max 0.3
```

```bash
python blur_generalization_suite/eval_cross_dataset.py \
  --model-path blur_generalization_suite/outputs/lora_train/vit_large_lora_vit-large-patch16-224-in21k_blur01/best_lora_model.pth \
  --dataset-group all \
  --blur-mode both \
  --blur-type motion \
  --blur-min 0.1 \
  --blur-max 0.3
```

7. 评估完成后看哪些结果
   - 每个模型的评估输出会在该 checkpoint 同级目录下新建 `cross_dataset_eval`
   - 里面重点看：
   - `cross_dataset_eval_时间戳.json`
   - `cross_dataset_summary_时间戳.csv`
   - `table4_clean_accuracy.csv`
   - `table5_blur_accuracy.csv`
   - `table6_clean_f1.csv`
   - `table7_blur_f1.csv`
   - 这 4 个 csv 就是你扩展 Table 4/5/6/7 的直接材料。
   - 注意：脚本是“每个模型各导一套 csv”，最终论文总表你还需要把多个模型结果汇总到一张表里。

---

**实验 2：DINOv3 ViT-Large-300M / ViT-Huge-840M，teacher-student 框架不变，只换 backbone，在 SDV1.4 训练，在 AIGCBenchmark 测试**

参数入口在 [train_teacher_student_backbones.py](E:/薛云起研究生阶段其他文件/2025暑期实习/最终百度实习大模型/dino-ditect_all_codes/Dino-Detect-for-blur-robust-AIGC-Detection-main/blur_generalization_suite/train_teacher_student_backbones.py#L478) 和 [eval_teacher_student_aigc.py](E:/薛云起研究生阶段其他文件/2025暑期实习/最终百度实习大模型/dino-ditect_all_codes/Dino-Detect-for-blur-robust-AIGC-Detection-main/blur_generalization_suite/eval_teacher_student_aigc.py#L83)。

1. 你要准备的数据和模型
   - DINOv3 ViT-Large-300M 本地目录。
   - 默认是：`/nas_train/app.e0016372/models/dinov3-vitl16-pretrain-lvd1689m`
   - DINOv3 ViT-Huge-840M 本地目录。
   - 默认是：`/nas_train/app.e0016372/models/dinov3-vith16plus-pretrain-lvd1689m`
   - 训练集。
   - 这次你要的是 SDV1.4，所以建议直接用 `--data-preset sdv14`
   - 它对应的训练根目录是 `/data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4`
   - 对应的 CCMBA 是 `/data/app.e0016372/imagenet_tmp/ccmba_processed_sdv44`
   - AIGCBenchmark 测试路径。
   - 还是来自 [dataset_configs.py](E:/薛云起研究生阶段其他文件/2025暑期实习/最终百度实习大模型/dino-ditect_all_codes/Dino-Detect-for-blur-robust-AIGC-Detection-main/blur_generalization_suite/dataset_configs.py)

2. 你需要改什么地址
   - 如果大模型本地目录不同，训练时传 `--dinov3-model-id`，或者换 `--backbone-preset`。
   - 如果你 SDV1.4 训练目录不一样，直接传 `--train-root`。
   - 如果 SDV1.4 对应的 CCMBA 目录不一样，直接传 `--ccmba-data-dir`。
   - 如果 AIGCBenchmark 测试路径不一致，改 [dataset_configs.py](E:/薛云起研究生阶段其他文件/2025暑期实习/最终百度实习大模型/dino-ditect_all_codes/Dino-Detect-for-blur-robust-AIGC-Detection-main/blur_generalization_suite/dataset_configs.py)。

3. 这一组最关键的设置
   - 你说“策略和训练情况不用动”，所以推荐这样理解：
   - teacher-student 两阶段不变
   - teacher clean / student blur 不变
   - strong augmentation 不变
   - `local_files_only=True` 保持原版
   - 只换 backbone
   - 但有一个细节你要自己定一下：
   - 原 `train_motion.py` 的 `BLUR_PROB` 是 `0.1`
   - 当前新脚本默认是 `0.2`
   - 如果你要最严格地“和原 teacher-student 策略对齐”，建议这里显式传 `--blur-prob 0.1`

4. 推荐命令
   - ViT-Large-300M

```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_teacher_student_backbones.py \
  --backbone-preset dinov3_vitl300m \
  --data-preset sdv14 \
  --blur-mode mixed \
  --blur-type motion \
  --blur-prob 0.1
```

   - ViT-Huge-840M

```bash
torchrun --nproc_per_node=8 blur_generalization_suite/train_teacher_student_backbones.py \
  --backbone-preset dinov3_vith840m \
  --data-preset sdv14 \
  --blur-mode mixed \
  --blur-type motion \
  --blur-prob 0.1
```

5. 训练完成后去哪找模型
   - 输出目录默认在：
   - `blur_generalization_suite/outputs/teacher_student/`
   - 子目录名称类似：
   - `dinov3-vitl16-pretrain-lvd1689m_sdv14_teacher_student_blur01`
   - `dinov3-vith16plus-pretrain-lvd1689m_sdv14_teacher_student_blur01`
   - 里面重点看：
   - `best_teacher_model.pth`
   - `best_student_model.pth`
   - `training_history.json`

6. 怎么评估 student 在 AIGCBenchmark 上的 generalization
   - 一般正文里你更应该看 `student`

```bash
python blur_generalization_suite/eval_teacher_student_aigc.py \
  --model-path blur_generalization_suite/outputs/teacher_student/dinov3-vitl16-pretrain-lvd1689m_sdv14_teacher_student_blur01/best_student_model.pth \
  --branch student \
  --dataset-group aigc_benchmark \
  --blur-mode both \
  --blur-type motion
```

```bash
python blur_generalization_suite/eval_teacher_student_aigc.py \
  --model-path blur_generalization_suite/outputs/teacher_student/dinov3-vith16plus-pretrain-lvd1689m_sdv14_teacher_student_blur01/best_student_model.pth \
  --branch student \
  --dataset-group aigc_benchmark \
  --blur-mode both \
  --blur-type motion
```

7. 要不要顺手测 teacher
   - 可以。
   - 如果你想做附录或分析 teacher/student 差异，再跑一遍 `--branch teacher`

```bash
python blur_generalization_suite/eval_teacher_student_aigc.py \
  --model-path blur_generalization_suite/outputs/teacher_student/dinov3-vitl16-pretrain-lvd1689m_sdv14_teacher_student_blur01/best_teacher_model.pth \
  --branch teacher \
  --dataset-group aigc_benchmark \
  --blur-mode both \
  --blur-type motion
```

8. 评估完成后看哪些结果
   - 每个 checkpoint 同级会生成：
   - `aigc_eval_student/` 或 `aigc_eval_teacher/`
   - 里面重点看：
   - `aigc_eval_时间戳.json`
   - `aigc_eval_summary_时间戳.csv`

---

**实验 3：不同 blur 强度下 DINOv3 特征稳定性可视化 / 测试**

参数入口在 [analyze_dinov3_blur_consistency.py](E:/薛云起研究生阶段其他文件/2025暑期实习/最终百度实习大模型/dino-ditect_all_codes/Dino-Detect-for-blur-robust-AIGC-Detection-main/blur_generalization_suite/analyze_dinov3_blur_consistency.py#L85)。

1. 你要准备的数据和模型
   - DINOv3 模型目录。
   - 默认就是你前面说的那个系列里的 `vit7b`：
   - `/nas_train/app.e0016372/models/dinov3-vit7b16-pretrain-lvd1689m`
   - 一批待分析的图片目录。
   - 这里只需要图片，不需要标签。
   - 你可以直接给一个真实图目录，或一个伪造图目录。
   - 如果你想更像 Figure 4 最后一列那种图，建议固定一个子集，不要每次混很多不同来源。

2. 你需要改什么地址
   - 如果 DINOv3 路径不同，传 `--model-path`
   - 图片目录必须显式传 `--data-root`
   - 不需要改任何训练脚本
   - 不需要改 `dataset_configs.py`

3. 推荐怎么选 `data-root`
   - 如果你想看 fake 图的稳定性，可以用：
   - `/data/app.e0016372/11ar_datasets/test/wukong/1_fake`
   - 如果你想看 real 图的稳定性，可以用：
   - `/data/app.e0016372/11ar_datasets/test/wukong/0_real`
   - 如果你想两类都看，最简单的办法是分两次跑。
   - 这个脚本一次只接收一个目录。

4. 推荐命令
   - 先跑 fake 子集

```bash
python blur_generalization_suite/analyze_dinov3_blur_consistency.py \
  --model-path /nas_train/app.e0016372/models/dinov3-vit7b16-pretrain-lvd1689m \
  --data-root /data/app.e0016372/11ar_datasets/test/wukong/1_fake \
  --blur-type motion \
  --min-blur 0.0 \
  --max-blur 0.3 \
  --step 0.05 \
  --max-images 64 \
  --local-files-only
```

   - 再跑 real 子集

```bash
python blur_generalization_suite/analyze_dinov3_blur_consistency.py \
  --model-path /nas_train/app.e0016372/models/dinov3-vit7b16-pretrain-lvd1689m \
  --data-root /data/app.e0016372/11ar_datasets/test/wukong/0_real \
  --blur-type motion \
  --min-blur 0.0 \
  --max-blur 0.3 \
  --step 0.05 \
  --max-images 64 \
  --local-files-only
```

5. 跑完后看哪些结果
   - 输出目录默认在：
   - `blur_generalization_suite/outputs/dinov3_consistency/`
   - 里面重点看：
   - `per_image_consistency.json`
   - `average_consistency.json`
   - `dinov3_consistency_curve.png`

6. 怎么理解结果
   - 如果你的预期成立，`dinov3_consistency_curve.png` 应该随着 blur strength 增大略有下降，但不会掉得特别夸张。
   - 如果你想让曲线更细，可以把 `--step 0.05` 改成 `0.02`
   - 如果你想让统计更稳，可以把 `--max-images 64` 提高到 `128` 或 `256`

---

**哪些地方你最可能真的要手动改文件**
1. 如果测试集挂载路径和我现在写的不一样，改 [dataset_configs.py](E:/薛云起研究生阶段其他文件/2025暑期实习/最终百度实习大模型/dino-ditect_all_codes/Dino-Detect-for-blur-robust-AIGC-Detection-main/blur_generalization_suite/dataset_configs.py)
2. 如果你想永久把 teacher-student 默认训练预设改成 `sdv14`，改 [train_teacher_student_backbones.py](E:/薛云起研究生阶段其他文件/2025暑期实习/最终百度实习大模型/dino-ditect_all_codes/Dino-Detect-for-blur-robust-AIGC-Detection-main/blur_generalization_suite/train_teacher_student_backbones.py#L43) 和 [train_teacher_student_backbones.py](E:/薛云起研究生阶段其他文件/2025暑期实习/最终百度实习大模型/dino-ditect_all_codes/Dino-Detect-for-blur-robust-AIGC-Detection-main/blur_generalization_suite/train_teacher_student_backbones.py#L480)
3. 如果你想永久改 LoRA 默认 backbone 地址，改 [model_zoo.py](E:/薛云起研究生阶段其他文件/2025暑期实习/最终百度实习大模型/dino-ditect_all_codes/Dino-Detect-for-blur-robust-AIGC-Detection-main/blur_generalization_suite/model_zoo.py)

---

**我给你的最推荐执行顺序**
1. 先完成实验 1 的 `CLIP + LoRA`
2. 再完成实验 1 的 `ViT-Large + LoRA`
3. 跑两次 `eval_cross_dataset.py`，先把 Table 4/5/6/7 材料拿到
4. 再跑实验 2 的 `ViT-Large-300M teacher-student`
5. 再跑实验 2 的 `ViT-Huge-840M teacher-student`
6. 跑 `eval_teacher_student_aigc.py` 拿到 generalization 结果
7. 最后跑实验 3 的 consistency 图

如果你愿意，我下一条可以直接帮你把这三组实验整理成一份“可直接复制执行的命令清单”，我会按你的服务器路径写成最终版。