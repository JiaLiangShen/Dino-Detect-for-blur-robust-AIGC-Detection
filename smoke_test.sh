#!/usr/bin/env bash
#
# smoke_test.sh — 下班前快速验证所有实验阶段能跑通(非正式结果)
#
# 目的:确认训练→评测全链路在所有 backbone 上都能起来并产出 checkpoint,
#       避免挂机后某个实验跑到一半才崩。验证产物写独立目录,不污染正式输出。
#
# 用法:
#   ./smoke_test.sh                  # 跑全部验证(默认)
#   ./smoke_test.sh env              # 仅环境+路径检查
#   ./smoke_test.sh exp1 exp2        # 只验证指定实验
#
# 常用环境变量:
#   NPROC=8                  GPU 数量
#   SAMPLES=512              训练每类样本数(默认 512,需满足 2*SAMPLES/NPROC>=BATCH)
#   BATCH=128                GPU micro-batch(与正式实验 run_experiments.sh 一致,验证真实显存)
#   ACCUM=1                  梯度累积(与正式实验一致)
#   EVAL_BATCH=16            评测 batch(与正式实验一致)
#   EVAL_WORKERS=16          评测 dataloader workers(机器224核,正式实验用32;smoke数据少用16)
#   SMOKE_DIR=smoke_outputs  验证产物目录(默认 smoke_outputs,与正式 outputs 隔离)
#
# ⚠️ batch/accum 与正式实验保持一致,这样验证才能暴露真实的显存压力(OOM)。
#    快速化只靠 --max-samples-per-class + 1 epoch,不靠缩小 batch。
# ⚠️ DDP 下数据被 DistributedSampler 按卡切分,每卡样本 = 2*SAMPLES/NPROC,必须 >= BATCH。
#    所以 SAMPLES 下限 = BATCH * NPROC / 2(8卡 batch128 → 每类至少512)。
#
# 每个阶段:能起来 + 产出文件 = PASS;起不来或报错 = FAIL(立即停下,方便排查)
# 预计总耗时:~15-20 分钟(大 batch 真实显存 + 模型加载开销)

set -euo pipefail

# ---------------- 配置 ----------------
NPROC="${NPROC:-8}"
BATCH="${BATCH:-128}"
ACCUM="${ACCUM:-1}"
# DDP 下每卡样本 = 2*SAMPLES/NPROC 必须 >= BATCH,故 SAMPLES 下限 = BATCH*NPROC/2
SAMPLES_MIN=$(( (BATCH * NPROC + 1) / 2 ))
SAMPLES="${SAMPLES:-$SAMPLES_MIN}"
EVAL_BATCH="${EVAL_BATCH:-16}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
SMOKE_DIR="${SMOKE_DIR:-smoke_outputs}"
SEED="${SEED:-3407}"
WILDRF_ROOT="${WILDRF_ROOT:-/data/app.e0016372/WildRF/test}"

# 模型权重实际路径(与 run_experiments.sh 一致)
declare -A MODEL_PATH=(
    [clip_vit_bigg_1_84b]="/nas_train/app.e0016372/models/temp/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    [eva_giant_1_1b]="/nas_train/app.e0016372/models/timm/eva_giant_patch14_336.m30m_ft_in22k_in1k"
    [aimv2_huge_840m]="/nas_train/app.e0016372/models/apple/aimv2-huge-patch14-336"
    [dinov3_vit7b]="/nas_train/app.e0016372/models/facebook/dinov3-vit7b16-pretrain-lvd1689m"
)
# sdv14 数据(实验一 paper profile + 实验二 strict_motion 都用它)
SDV14_TRAIN_ROOT="/data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4"
SDV14_CCMBA_DIR="/data/app.e0016372/imagenet_tmp/ccmba_processed_sdv44"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 独立日志
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/smoke_test_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# ---------------- 工具 ----------------
log()  { printf '\n\033[1;34m[%s] === %s ===\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
info() { printf '\033[0;36m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
pass() { printf '\033[1;32m[%s] ✅ PASS: %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '\n\033[1;31m[%s] ❌ FAIL: %s\033[0m\n' "$(date +%H:%M:%S)" "$*" >&2; }

# 失败即停:smoke test 的目的就是发现哪里跑不通,第一个失败就停下排查
trap 'ec=$?; if [ "$ec" -ne 0 ]; then printf "\n\033[1;31m[%s] 验证中断,退出码 %s。日志: %s\033[0m\n请先解决上述问题再下班挂机。\n" "$(date +%H:%M:%S)" "$ec" "$LOG_FILE" >&2; fi' EXIT

# ---------------- 环境与路径检查 ----------------
run_env() {
    log "阶段 0:环境 + 路径检查"
    python -c "import transformers; from packaging.version import Version; \
assert Version(transformers.__version__) >= Version('4.55'); print('transformers', transformers.__version__)"
    python -c "from transformers import Aimv2VisionModel, CLIPVisionModel; print('vision loaders OK')"
    python -c "import timm; print('timm', timm.__version__)"
    python -m py_compile \
        blur_generalization_suite/model_zoo.py \
        blur_generalization_suite/data_utils.py \
        blur_generalization_suite/train_paper_backbones.py \
        blur_generalization_suite/eval_motion_only_corruptions.py \
        blur_generalization_suite/eval_teacher_student_aigc.py

    # 关键路径存在性(任一缺失会直接报错)
    local missing=0
    for bb in clip_vit_bigg_1_84b eva_giant_1_1b aimv2_huge_840m dinov3_vit7b; do
        if [[ -d "${MODEL_PATH[$bb]}" ]]; then info "模型 OK: ${MODEL_PATH[$bb]}"; else fail "模型缺失: ${MODEL_PATH[$bb]}"; missing=1; fi
    done
    [[ -d "$SDV14_TRAIN_ROOT" ]] && info "训练数据 OK: $SDV14_TRAIN_ROOT" || { fail "训练数据缺失: $SDV14_TRAIN_ROOT"; missing=1; }
    [[ -d "$SDV14_CCMBA_DIR"  ]] && info "CCMBA 数据 OK: $SDV14_CCMBA_DIR"  || { fail "CCMBA 数据缺失: $SDV14_CCMBA_DIR";  missing=1; }
    [[ -d "$WILDRF_ROOT"      ]] && info "WildRF OK: $WILDRF_ROOT"          || { fail "WildRF 缺失: $WILDRF_ROOT";          missing=1; }
    [[ "$missing" -eq 0 ]] || { fail "存在路径缺失,无法继续"; exit 1; }

    # drop_last=True + DDP:DistributedSampler 把 2*SAMPLES 按卡切分,
    # 每卡样本 = 2*SAMPLES/NPROC,必须 >= BATCH 才能凑出完整 batch
    local per_gpu=$(( 2 * SAMPLES / NPROC ))
    if (( per_gpu < BATCH )); then
        fail "SAMPLES=$SAMPLES 太小:DDP $NPROC 卡下每卡仅 $per_gpu 张 < BATCH=$BATCH,drop_last 会让 DataLoader 为空。"
        fail "请调大 SAMPLES(至少 $SAMPLES_MIN): SAMPLES=$SAMPLES_MIN ./smoke_test.sh ..."
        exit 1
    fi
    info "batch=$BATCH accum=$ACCUM(与正式一致) | 每类 $SAMPLES 张 → 每卡 $per_gpu 张 → 每 epoch $((per_gpu/BATCH)) 个 batch/卡"
    pass "环境与路径检查"
}

# ---------------- 训练验证(单 backbone) ----------------
# 用法: smoke_train <preset> <experiment_mode> [extra args...]
smoke_train() {
    local bb="$1"; shift
    local mode="$1"; shift
    local label="$bb / $mode"
    local ckpt=""
    {  # 函数体输出重定向到 stderr(仍 tee 到日志),只留最后 echo 路径到 stdout
    log "训练验证: $label(每类 ${SAMPLES} 张,1 epoch)"

    local extra=()
    if [[ "$mode" == "dino_detect" ]]; then
        extra=(--teacher-epochs 1 --student-epochs 1
               --teacher-batch-size "$BATCH" --teacher-accumulation-steps "$ACCUM"
               --student-batch-size "$BATCH" --student-accumulation-steps "$ACCUM")
    else
        extra=(--student-epochs 1
               --student-batch-size "$BATCH" --student-accumulation-steps "$ACCUM")
    fi

    # 实验二走 rebuttal wrapper(强制 strict_motion);实验一直接用 paper 训练脚本
    local script
    if [[ "$mode" == "classification_only" || "$mode" == "dino_detect_strict" ]]; then
        script="0424_modelzoo_comparations/train_motion_only_rebuttal.py"
        [[ "$mode" == "dino_detect_strict" ]] && mode="dino_detect"
    else
        script="0424_modelzoo_comparations/train_paper_backbones.py"
    fi

    torchrun --nproc_per_node="$NPROC" "$script" \
        --backbone-preset "$bb" \
        --dinov3-model-id "${MODEL_PATH[$bb]}" \
        --data-preset sdv14 \
        --training-profile "${TRAIN_PROFILE:-paper}" \
        --experiment-mode "$mode" \
        --max-samples-per-class "$SAMPLES" \
        --output-dir "$SMOKE_DIR/paper_parity" \
        "${extra[@]}" "$@"

    # 确认 checkpoint 落盘(这是最容易出问题的环节)
    if [[ "$mode" == "classification_only" ]]; then
        ckpt="$SMOKE_DIR/paper_parity/${bb}_sdv14_classification_only_strict_motion_blur01/best_classification_only_model.pth"
    else
        ckpt="$SMOKE_DIR/paper_parity/${bb}_sdv14_dino_detect_${TRAIN_PROFILE:-paper}_blur01/best_student_model.pth"
    fi
    [[ -f "$ckpt" ]] && pass "训练产出 checkpoint: $ckpt" || { fail "训练未产出 checkpoint: $ckpt"; exit 1; }
    } >&2
    echo "$ckpt"  # 唯一打到 stdout,供 ckpt=$(smoke_train ...) 捕获
}

# ---------------- 实验一验证:3 backbone 训练 + AIGC 评测 ----------------
run_exp1() {
    log "实验一验证:三个 backbone 训练 + AIGC 评测"
    local backbones=(clip_vit_bigg_1_84b eva_giant_1_1b aimv2_huge_840m)
    local bb ckpt
    for bb in "${backbones[@]}"; do
        ckpt=$(TRAIN_PROFILE=paper smoke_train "$bb" dino_detect)
        # AIGC 评测验证:单数据集,确认 checkpoint 能被加载并推理
        log "AIGC 评测验证: $bb(单数据集 dalle2)"
        python 0424_modelzoo_comparations/eval_teacher_student_aigc.py \
            --model-path "$ckpt" \
            --branch student --dataset-group dalle2 \
            --blur-mode both --blur-type motion --blur-min 0.1 --blur-max 0.3 \
            --batch-size "$EVAL_BATCH" --num-workers "$EVAL_WORKERS" \
            --output-dir "$SMOKE_DIR/aigc_eval/$bb" \
            || { fail "AIGC 评测失败: $bb"; exit 1; }
        pass "AIGC 评测通过: $bb"
    done
    pass "实验一全链路验证通过(3 backbone × 训练+评测)"
}

# ---------------- 实验二验证:2 训练 + 跨退化评测 ----------------
run_exp2() {
    log "实验二验证:motion-only 严格对照 + 跨退化评测"
    local cls_ckpt dino_ckpt
    cls_ckpt=$(TRAIN_PROFILE=strict_motion smoke_train dinov3_vit7b classification_only)
    dino_ckpt=$(TRAIN_PROFILE=strict_motion smoke_train dinov3_vit7b dino_detect_strict)

    # 跨退化评测验证:单平台 + 单退化,确认两个 checkpoint 能通过校验并推理
    log "跨退化评测验证(单平台 reddit + 单退化 gaussian)"
    python 0424_modelzoo_comparations/eval_motion_only_corruptions.py \
        --classification-checkpoint "$cls_ckpt" \
        --dino-detect-checkpoint "$dino_ckpt" \
        --wildrf-root "$WILDRF_ROOT" \
        --platforms reddit \
        --profile paper3 --corruptions gaussian \
        --batch-size "$EVAL_BATCH" --num-workers "$EVAL_WORKERS" \
        --output-dir "$SMOKE_DIR/cross_corruption" \
        || { fail "跨退化评测失败(注意:可能两 checkpoint 配置不一致被校验拒绝)"; exit 1; }
    pass "跨退化评测通过(含 checkpoint 一致性校验)"
    pass "实验二全链路验证通过(2 训练 + 跨退化评测)"
}

# ---------------- 调度 ----------------
usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \?//' | sed '/^set -euo/d'; exit 1; }

if [[ $# -eq 0 ]]; then STAGES=(env exp1 exp2); else STAGES=("$@"); fi
for s in "${STAGES[@]}"; do
    case "$s" in
        env)  run_env ;;
        exp1) run_exp1 ;;
        exp2) run_exp2 ;;
        -h|--help|help) usage ;;
        *) echo "未知阶段: $s" >&2; usage ;;
    esac
done

log "🎉 全部验证通过!可以放心下班挂机跑正式实验:"
log "    nohup ./run_experiments.sh all > /dev/null 2>&1 &"
log "    nohup ./auto_commit.sh > /dev/null 2>&1 &  (自动备份)"
log "验证产物(可删): $SMOKE_DIR/  ;  日志: $LOG_FILE"
