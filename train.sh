#!/bin/bash

# ============================================================================
# 1. CLIP-LoRA 训练 (CLIP-ViT-bigG-14)
# ============================================================================

# 训练1: 保存最佳检查点
torchrun --nproc_per_node=8 \
    blur_generalization_suite/train_lora_blur.py \
    --model-family clip_lora \
    --backbone-path /nas_train/app.e0016372/models/CLIP-ViT-bigG-14-laion2B-39B-b160k \
    --train-root /data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4 \
    --ccmba-data-dir /data/app.e0016372/imagenet_tmp/ccmba_processed_sdv44 \
    --blur-mode mixed \
    --blur-type motion \
    --blur-prob 0.1 \
    --blur-min 0.1 \
    --blur-max 0.3 \
    --mixed-mode-ratio 0.5 \
    --report-checkpoint best \
    --local-files-only

# 训练2: 保存最后一个检查点
torchrun --nproc_per_node=8 \
    blur_generalization_suite/train_lora_blur.py \
    --model-family clip_lora \
    --backbone-path /nas_train/app.e0016372/models/CLIP-ViT-bigG-14-laion2B-39B-b160k \
    --train-root /data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4 \
    --ccmba-data-dir /data/app.e0016372/imagenet_tmp/ccmba_processed_sdv44 \
    --blur-mode mixed \
    --blur-type motion \
    --blur-prob 0.1 \
    --blur-min 0.1 \
    --blur-max 0.3 \
    --mixed-mode-ratio 0.5 \
    --report-checkpoint last \
    --local-files-only

# ============================================================================
# 2. EVA-Giant-LoRA 训练
# ============================================================================

torchrun --nproc_per_node=8 \
    blur_generalization_suite/train_lora_blur.py \
    --model-family eva_giant_lora \
    --backbone-path /nas_train/app.e0016372/models/eva_giant_patch14_336.m30m_ft_in22k_in1k \
    --train-root /data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4 \
    --ccmba-data-dir /data/app.e0016372/imagenet_tmp/ccmba_processed_sdv44 \
    --blur-mode mixed \
    --blur-type motion \
    --blur-prob 0.1 \
    --blur-min 0.1 \
    --blur-max 0.3 \
    --mixed-mode-ratio 0.5 \
    --report-checkpoint best \
    --local-files-only

# ============================================================================
# 3. 跨数据集评估 (LoRA 模型)
# ============================================================================

# 评估 CLIP-LoRA (best)
python blur_generalization_suite/eval_cross_dataset.py \
    --model-path blur_generalization_suite/outputs/lora_train/clip_lora_CLIP-ViT-bigG-14-laion2B-39B-b160k_blur01_best/selected_lora_model.pth \
    --dataset-group all \
    --blur-mode both \
    --blur-type motion \
    --blur-min 0.1 \
    --blur-max 0.3

# 评估 CLIP-LoRA (last)
python blur_generalization_suite/eval_cross_dataset.py \
    --model-path blur_generalization_suite/outputs/lora_train/clip_lora_CLIP-ViT-bigG-14-laion2B-39B-b160k_blur01_last/selected_lora_model.pth \
    --dataset-group all \
    --blur-mode both \
    --blur-type motion \
    --blur-min 0.1 \
    --blur-max 0.3

# 评估 EVA-Giant-LoRA
python blur_generalization_suite/eval_cross_dataset.py \
    --model-path blur_generalization_suite/outputs/lora_train/eva_giant_lora_eva_giant_patch14_336.m30m_ft_in22k_in1k_blur01/selected_lora_model.pth \
    --dataset-group all \
    --blur-mode both \
    --blur-type motion \
    --blur-min 0.1 \
    --blur-max 0.3

# ============================================================================
# 4. 教师-学生模型训练 (DINOv3)
# ============================================================================

# 训练 DINOv3-ViT-L
torchrun --nproc_per_node=8 blur_generalization_suite/train_teacher_student_backbones.py \
    --backbone-preset dinov3_vitl300m \
    --data-preset sdv14 \
    --blur-mode mixed \
    --blur-type motion \
    --blur-prob 0.1 \
    --alpha-simclr 0.3

# 训练 DINOv3-ViT-H
torchrun --nproc_per_node=8 blur_generalization_suite/train_teacher_student_backbones.py \
    --backbone-preset dinov3_vith840m \
    --data-preset sdv14 \
    --blur-mode mixed \
    --blur-type motion \
    --blur-prob 0.1 \
    --alpha-simclr 0.3

# ============================================================================
# 5. 评估教师-学生模型 (AIGC 基准)
# ============================================================================

# 评估 DINOv3-ViT-L 学生模型
python blur_generalization_suite/eval_teacher_student_aigc.py \
    --model-path blur_generalization_suite/outputs/teacher_student/dinov3-vitl16-pretrain-lvd1689m_sdv14_teacher_student_blur01/best_student_model.pth \
    --branch student \
    --dataset-group aigc_benchmark \
    --blur-mode both \
    --blur-type motion

# 评估 DINOv3-ViT-H 学生模型
python blur_generalization_suite/eval_teacher_student_aigc.py \
    --model-path blur_generalization_suite/outputs/teacher_student/dinov3-vith16plus-pretrain-lvd1689m_sdv14_teacher_student_blur01/best_student_model.pth \
    --branch student \
    --dataset-group aigc_benchmark \
    --blur-mode both \
    --blur-type motion

# ============================================================================
# 6. 分析 DINOv3 模糊一致性
# ============================================================================

python blur_generalization_suite/analyze_dinov3_blur_consistency.py \
    --model-path /nas_train/app.e0016372/models/dinov3-vit7b16-pretrain-lvd1689m \
    --data-root /data/app.e0016372/11ar_datasets/test/wukong/1_fake \
    --blur-type motion \
    --min-blur 0.0 \
    --max-blur 0.5 \
    --step 0.1 \
    --max-images 64 \
    --local-files-only

# ============================================================================
# 7. 自有基准批量评估
# ============================================================================

python blur_generalization_suite/own_batch_eval.py \
    --data_root /data/app.e0016372/11ar_datasets/test/own_benchmark \
    --clip_lora_path blur_generalization_suite/outputs/lora_train/clip_lora_CLIP-ViT-bigG-14-laion2B-39B-b160k_blur01_best/best_lora_model.pth \
    --eva02_lora_path blur_generalization_suite/outputs/lora_train/eva_giant_lora_eva_giant_patch14_336.m30m_ft_in22k_in1k_blur01/best_lora_model.pth \
    --student_model_path teacher_mixed_blur_training/dinov3-vit7b16-pretrain-lvd1689m_blur01/best_teacher_blur_model.pth \
    --use_deblur \
    --batch_size 64 \
    --output_dir blur_generalization_suite/outputs/own_batch_eval
