#!/bin/bash

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

# python blur_generalization_suite/eval_cross_dataset.py \
#     --model-path blur_generalization_suite/outputs/lora_train/.../selected_lora_model.pth \
#     --dataset-group all \
#     --blur-mode both \
#     --blur-type motion \
#     --blur-min 0.1 \
#     --blur-max 0.3
