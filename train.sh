#!/bin/bash

set -e

SCRIPT=blur_generalization_suite/eval_teacher_student_aigc.py
BASE_DIR=blur_generalization_suite/outputs/teacher_student

run() {
    GPU=$1
    MODEL=$2

    echo "[INFO] GPU $GPU running $MODEL"

    CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT \
      --model-path $BASE_DIR/$MODEL/best_student_model.pth \
      --branch student \
      --dataset-group aigc_benchmark \
      --blur-mode both \
      --blur-type motion \
      > logs/${MODEL}.log 2>&1 &
}

mkdir -p logs

run 0 aimv2-huge-patch14-336_sdv14_teacher_student_blur01
run 1 aimv2-large-patch14-224_sdv14_teacher_student_blur01
run 2 siglip2-giant-opt-patch16-256_sdv14_teacher_student_blur01
run 3 siglip2-large-patch16-256_sdv14_teacher_student_blur01

wait

echo "[INFO] All jobs finished."
