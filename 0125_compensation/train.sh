#!/bin/bash

models=(
    "/nas_train/app.e0016372/models/dinov3-vit7b16-pretrain-lvd1689m"
    "/nas_train/app.e0016372/models/dinov3-vitl16-pretrain-lvd1689m"
)

probs=(0.0 0.1 0.2)

for model in "${models[@]}"; do
    for prob in "${probs[@]}"; do
        echo "========================================"
        echo "Running training with:"
        echo "Model: $model"
        echo "Blur probability: $prob"
        echo "========================================"
        
        # torchrun --nproc_per_node=8 0125_compensation/gaussian_direct_train.py \
        torchrun --nproc_per_node=8 0125_compensation/motion_direct_train.py \
            --dinov3-model-id "$model" \
            --blur-prob "$prob"
        
        echo "========================================"
        echo "Completed: Model=$model, Prob=$prob"
        echo "========================================"
        
        # 等待1分钟（60秒）
        echo "Waiting for 1 minute before next run..."
        sleep 60
        echo ""
    done
done
echo "All runs completed!"
