#!/bin/bash

models=(
    "/nas_train/app.e0016372/models/dinov3-vith16plus-pretrain-lvd1689m"
    # "/nas_train/app.e0016372/models/dinov3-vit7b16-pretrain-lvd1689m"
    # "/nas_train/app.e0016372/models/dinov3-vitl16-pretrain-lvd1689m"
)

probs=(0.0 0.1 0.2)

for model in "${models[@]}"; do
    for prob in "${probs[@]}"; do
        torchrun --nproc_per_node=8 0125_compensation/gaussian_direct_train.py \
            --dinov3-model-id "$model" \
            --blur-prob "$prob"
        
        if [[ "$prob" != "0.0" ]]; then
            torchrun --nproc_per_node=8 0125_compensation/motion_direct_train.py \
                --dinov3-model-id "$model" \
                --blur-prob "$prob"
        fi
    done
done
echo "All runs completed!"
