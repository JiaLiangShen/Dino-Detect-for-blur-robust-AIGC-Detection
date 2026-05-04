#!/bin/bash

set -e

SCRIPT=blur_generalization_suite/eval_teacher_student_aigc.py
BASE_DIR=blur_generalization_suite/outputs/teacher_student

run() {
    GPU=$1
    MODEL=$2
    BRANCH=$3

    echo "[INFO] GPU $GPU running $MODEL ($BRANCH)"

    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT \
      --model-path $BASE_DIR/$MODEL/best_${BRANCH}_model.pth \
      --branch $BRANCH \
      --dataset-group all \
      --blur-mode both \
      --blur-type motion \
      > logs/${MODEL}_${BRANCH}.log 2>&1 &
}

mkdir -p logs

run 0 aimv2-huge-patch14-336_sdv14_teacher_student_blur01 student
run 1 aimv2-large-patch14-224_sdv14_teacher_student_blur01 student
run 2 siglip2-giant-opt-patch16-256_sdv14_teacher_student_blur01 student
run 3 siglip2-large-patch16-256_sdv14_teacher_student_blur01 student

run 4 aimv2-huge-patch14-336_sdv14_teacher_student_blur01 teacher
run 5 aimv2-large-patch14-224_sdv14_teacher_student_blur01 teacher
run 6 siglip2-giant-opt-patch16-256_sdv14_teacher_student_blur01 teacher
run 7 siglip2-large-patch16-256_sdv14_teacher_student_blur01 teacher

wait

echo "[INFO] All jobs finished."
