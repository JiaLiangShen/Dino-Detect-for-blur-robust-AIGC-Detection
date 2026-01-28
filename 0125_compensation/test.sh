#!/bin/bash

# 定义模型ID数组
model_ids=(
    "/nas_train/app.e0016372/models/dinov3-vith16plus-pretrain-lvd1689m"
    "/nas_train/app.e0016372/models/dinov3-vit7b16-pretrain-lvd1689m"
    "/nas_train/app.e0016372/models/dinov3-vitl16-pretrain-lvd1689m"
)

# 定义路径模板数组
path_templates=(
    "teacher_blur_training/{model_name}_blur00/best_teacher_blur_model.pth"
    "teacher_blur_training/{model_name}_blur01/best_teacher_blur_model.pth"
    "teacher_blur_training/{model_name}_blur02/best_teacher_blur_model.pth"
    "teacher_mixed_blur_training/{model_name}_blur01/best_teacher_blur_model.pth"
    "teacher_mixed_blur_training/{model_name}_blur02/best_teacher_blur_model.pth"
)

# 主循环
for model_id in "${model_ids[@]}"; do
    # 从完整路径中提取模型名称（如 dinov3-vitl16-pretrain-lvd1689m）
    model_name=$(basename "$model_id")
    
    # 遍历所有路径模板
    for template in "${path_templates[@]}"; do
        # 替换模板中的占位符为实际模型名称
        model_path="${template//\{model_name\}/$model_name}"
        
        # 执行 Python 脚本
        python 0125_compensation/test_for_wildrf.py \
            --model_path "$model_path" \
            --dinov3_model_id "$model_id"
    done
done
echo "All runs completed!"

export https_proxy="http://172.19.92.23:13128"
git add .
git commit -m "Auto-commit: DINOv3 model testing"
git push
