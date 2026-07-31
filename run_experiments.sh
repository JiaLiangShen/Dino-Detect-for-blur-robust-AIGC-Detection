#!/usr/bin/env bash
#
# run_experiments.sh — DINO-Detect 论文复现 + rebuttal 一键实验脚本
#
# 内容对应:
#   0424_modelzoo_comparations/QUICK_EXPERIMENT_GUIDE.md (精简版)
#   0424_modelzoo_comparations/PAPER_PARITY_AND_REBUTTAL_RUNBOOK.md (完整版)
# 训练参数取 runbook 的完整写法,与 QUICK guide 的精简默认值等价。
#
# 用法:
#   ./run_experiments.sh [stage ...]        # 不传参数 = all
#
# stage:
#   env    仅环境检查(版本断言 + py_compile),不训练
#   exp1   实验一:三个 backbone 训练 + AIGC 评测
#   exp2   实验二:motion-only 严格对照训练 + 跨退化评测
#   all    env + exp1 + exp2(默认)
#   可一次传多个,如: ./run_experiments.sh exp1 exp2
#
# 常用环境变量(在命令前覆盖即可):
#   NPROC=8                              GPU 数量(对应 --nproc_per_node)
#   WILDRF_ROOT=/data/.../WildRF/test    跨退化评测的 WildRF 测试集路径
#   SEED=3407                            随机种子
#   FORCE_TRAIN=1                        即使 checkpoint 已存在也重新训练
#   FRESH=1                              从头跑:启动时清空本次实验输出(paper_parity + reviewer_cross_corruption)
#   LOG_DIR=logs                         日志目录
#
# 示例:
#   NPROC=4 ./run_experiments.sh all
#   ./run_experiments.sh env
#   ./run_experiments.sh exp1
#   FORCE_TRAIN=1 ./run_experiments.sh exp2
#
# 特性:
#   - 断点续跑:checkpoint 已存在则自动跳过对应训练,失败后修复再跑即可续上
#   - 全程日志:同时输出到终端与 logs/experiment_<时间戳>.log
#   - 失败即停(set -euo pipefail),避免在异常状态下浪费 GPU
#
# 注意:
#   - best_*_model.pth 按训练准确率选择,并非验证集最优(与历史代码一致)
#   - 正式 rebuttal 数据不要加 --allow-unverified-checkpoints,本脚本也未加

set -euo pipefail

# ---------------- 配置(可被环境变量覆盖) ----------------
NPROC="${NPROC:-8}"
WILDRF_ROOT="${WILDRF_ROOT:-/data/app.e0016372/WildRF/test}"
SEED="${SEED:-3407}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FRESH="${FRESH:-0}"           # FRESH=1:启动时清空本次实验输出(paper_parity + reviewer_cross_corruption),从头跑
LOG_DIR="${LOG_DIR:-logs}"

# 模型权重实际路径(代码 model_zoo.py 里的默认路径在本机不存在,用 --dinov3-model-id 覆盖)
# 已核实 4 个目录均含 config.json + 权重文件
declare -A MODEL_PATH=(
    [clip_vit_bigg_1_84b]="/nas_train/app.e0016372/models/temp/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    [eva_giant_1_1b]="/nas_train/app.e0016372/models/timm/eva_giant_patch14_336.m30m_ft_in22k_in1k"
    [aimv2_huge_840m]="/nas_train/app.e0016372/models/apple/aimv2-huge-patch14-336"
    [dinov3_vit7b]="/nas_train/app.e0016372/models/facebook/dinov3-vit7b16-pretrain-lvd1689m"
)

# sdv14 训练数据(实验一 paper profile + 实验二 strict_motion 都用它)
# 已核实本机存在:train/{nature,ai} + val/{nature,ai}
SDV14_TRAIN_ROOT="/data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4"
SDV14_CCMBA_DIR="/data/app.e0016372/imagenet_tmp/ccmba_processed_sdv44"

# 固定路径
PARITY_DIR="blur_generalization_suite/outputs/paper_parity"
CROSS_CORRUPTION_DIR="blur_generalization_suite/outputs/reviewer_cross_corruption"

# 自动切到仓库根目录(本脚本所在目录)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 时间戳日志,同时写终端与文件
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/experiment_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
TEE_PID="${!:-}"

# ---------------- 工具函数 ----------------
log()  { printf '\n\033[1;34m[%s] === %s ===\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
info() { printf '\033[0;36m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\n\033[1;33m[%s] WARNING: %s\033[0m\n' "$(date +%H:%M:%S)" "$*" >&2; }

# 失败时打印日志位置(实际退出码由 set -e 自动保留)
trap 'ec=$?; if [ "$ec" -ne 0 ]; then printf "\n\033[1;31m[%s] 脚本中断,退出码 %s\033[0m\n完整日志: %s\n" "$(date +%H:%M:%S)" "$ec" "$LOG_FILE" >&2; fi' EXIT

# should_train:checkpoint 不存在或 FORCE_TRAIN=1 时返回 0(真→需要训练)
should_train() {
    local ckpt="$1"
    if [[ "$FORCE_TRAIN" == "1" ]]; then return 0; fi
    if [[ -f "$ckpt" ]]; then return 1; fi
    return 0
}

require_file() { [[ -f "$1" ]] || warn "缺失文件: $1"; }

# 从头跑:清空本次实验输出(只删 paper_parity + reviewer_cross_corruption,不碰 lora_train/teacher_student 等)
fresh_clean() {
    if [[ "$FRESH" != "1" ]]; then return 0; fi
    log "FRESH=1:清空本次实验输出目录,从头跑"
    local removed=0
    for d in "$PARITY_DIR" "$CROSS_CORRUPTION_DIR"; do
        if [[ -d "$d" ]]; then
            rm -rf "${d:?}/"* 2>/dev/null || true
            info "已清空: $d"
            removed=1
        fi
    done
    [[ "$removed" -eq 1 ]] || info "(本次实验输出目录本就为空)"
    warn "已清空 checkpoint,断点续跑将失效,所有训练会重跑。如需保留续跑能力,用 FRESH=0(默认)"
}

# ---------------- 环境检查 ----------------
run_env() {
    fresh_clean
    log "环境检查(transformers>=4.55 / timm / 视觉加载器 / py_compile)"
    python -c "import transformers; from packaging.version import Version; \
assert Version(transformers.__version__) >= Version('4.55'); \
print('transformers', transformers.__version__)"
    python -c "from transformers import Aimv2VisionModel, CLIPVisionModel; \
print('transformers vision loaders OK')"
    python -c "import timm; print('timm', timm.__version__)"
    python -m py_compile \
        blur_generalization_suite/model_zoo.py \
        blur_generalization_suite/data_utils.py \
        blur_generalization_suite/train_paper_backbones.py \
        blur_generalization_suite/train_motion_only_rebuttal.py \
        blur_generalization_suite/eval_motion_only_corruptions.py \
        blur_generalization_suite/eval_teacher_student_aigc.py
    info "环境检查通过"
}

# ---------------- 实验一:三个对比 backbone ----------------
run_exp1() {
    log "实验一:三个对比 backbone(CLIP-bigG / EVA-GIANT / AIMv2-Huge)"

    # 路径自检:模型权重 + sdv14 训练数据(paper profile 会用 ccmba)
    local backbones=(clip_vit_bigg_1_84b eva_giant_1_1b aimv2_huge_840m)
    local bb ckpt
    for bb in "${backbones[@]}"; do
        [[ -d "${MODEL_PATH[$bb]}" ]] || warn "模型路径不存在: ${MODEL_PATH[$bb]}"
    done
    [[ -d "$SDV14_TRAIN_ROOT" ]] || warn "sdv14 训练数据不存在: $SDV14_TRAIN_ROOT"
    [[ -d "$SDV14_CCMBA_DIR" ]] || warn "sdv14 ccmba 数据不存在: $SDV14_CCMBA_DIR"

    for bb in "${backbones[@]}"; do
        ckpt="${PARITY_DIR}/${bb}_sdv14_dino_detect_paper_blur01/best_student_model.pth"

        if should_train "$ckpt"; then
            if [[ "$FORCE_TRAIN" == "1" ]]; then warn "FORCE_TRAIN=1,强制重新训练 $bb"; fi
            log "训练 $bb(SDv1.4 + DINO-Detect,paper profile)"
            torchrun --nproc_per_node="$NPROC" \
                0424_modelzoo_comparations/train_paper_backbones.py \
                --backbone-preset "$bb" \
                --dinov3-model-id "${MODEL_PATH[$bb]}" \
                --data-preset sdv14 \
                --training-profile paper \
                --experiment-mode dino_detect \
                --teacher-epochs 3 --student-epochs 5 \
                --teacher-batch-size 32 --teacher-accumulation-steps 4 \
                --student-batch-size 32 --student-accumulation-steps 4 \
                --teacher-learning-rate 1e-4 --student-learning-rate 5e-5 \
                --weight-decay 1e-4 --max-grad-norm 0 \
                --seed "$SEED"
        else
            info "跳过训练 $bb:checkpoint 已存在"
        fi

        # AIGC 评测(clean + motion blur)
        if [[ -f "$ckpt" ]]; then
            log "AIGC 评测 $bb(clean + motion blur)"
            python 0424_modelzoo_comparations/eval_teacher_student_aigc.py \
                --model-path "$ckpt" \
                --branch student --dataset-group all \
                --blur-mode both --blur-type motion --blur-min 0.1 --blur-max 0.3 \
                --batch-size 16 --num-workers 32
        else
            warn "缺少 checkpoint,跳过 $bb 的 AIGC 评测: $ckpt"
        fi
        info "checkpoint: $ckpt"
    done

    log "实验一完成 — 汇总对比 clean vs motion blur 的 Accuracy / BAcc"
    info "所有 checkpoint 目录: ${PARITY_DIR}/"
}

# ---------------- 实验二:motion-only 严格跨退化 ----------------
run_exp2() {
    log "实验二:motion-only 严格对照(rebuttal)"

    # 路径自检:DINOv3 模型 + sdv14 训练数据 + WildRF
    [[ -d "${MODEL_PATH[dinov3_vit7b]}" ]] || warn "模型路径不存在: ${MODEL_PATH[dinov3_vit7b]}"
    [[ -d "$SDV14_TRAIN_ROOT" ]] || warn "sdv14 训练数据不存在: $SDV14_TRAIN_ROOT"
    [[ -d "$WILDRF_ROOT" ]] || warn "WildRF 测试集不存在: $WILDRF_ROOT"

    local cls_ckpt="${PARITY_DIR}/dinov3_vit7b_sdv14_classification_only_strict_motion_blur01/best_classification_only_model.pth"
    local dino_ckpt="${PARITY_DIR}/dinov3_vit7b_sdv14_dino_detect_strict_motion_blur01/best_student_model.pth"

    # (1) classification-only 对照(仅 focal 分类损失)
    if should_train "$cls_ckpt"; then
        if [[ "$FORCE_TRAIN" == "1" ]]; then warn "FORCE_TRAIN=1,强制重新训练 classification-only"; fi
        log "训练 classification-only 对照(DINOv3-7B,严格 motion-only,sdv14)"
        torchrun --nproc_per_node="$NPROC" \
            0424_modelzoo_comparations/train_motion_only_rebuttal.py \
            --backbone-preset dinov3_vit7b \
            --dinov3-model-id "${MODEL_PATH[dinov3_vit7b]}" \
            --data-preset sdv14 \
            --experiment-mode classification_only \
            --blur-prob 0.1 --blur-min 0.1 --blur-max 0.3 \
            --student-epochs 5 \
            --student-batch-size 32 --student-accumulation-steps 4 \
            --student-learning-rate 5e-5 --weight-decay 1e-4 \
            --max-grad-norm 0 --seed "$SEED"
    else
        info "跳过训练 classification-only:checkpoint 已存在"
    fi

    # (2) motion-only DINO-Detect(完整蒸馏目标)
    if should_train "$dino_ckpt"; then
        if [[ "$FORCE_TRAIN" == "1" ]]; then warn "FORCE_TRAIN=1,强制重新训练 motion-only DINO-Detect"; fi
        log "训练 motion-only DINO-Detect(DINOv3-7B,严格 motion-only,sdv14)"
        torchrun --nproc_per_node="$NPROC" \
            0424_modelzoo_comparations/train_motion_only_rebuttal.py \
            --backbone-preset dinov3_vit7b \
            --dinov3-model-id "${MODEL_PATH[dinov3_vit7b]}" \
            --data-preset sdv14 \
            --experiment-mode dino_detect \
            --blur-prob 0.1 --blur-min 0.1 --blur-max 0.3 \
            --teacher-epochs 3 --student-epochs 5 \
            --teacher-batch-size 32 --teacher-accumulation-steps 4 \
            --student-batch-size 32 --student-accumulation-steps 4 \
            --teacher-learning-rate 1e-4 --student-learning-rate 5e-5 \
            --weight-decay 1e-4 --max-grad-norm 0 \
            --seed "$SEED"
    else
        info "跳过训练 motion-only DINO-Detect:checkpoint 已存在"
    fi

    # (3) 跨退化 zero-shot 评测(Gaussian/defocus/box/radial/noise/JPEG)
    if [[ -f "$cls_ckpt" && -f "$dino_ckpt" ]]; then
        log "跨退化评测(6 种未见退化,paper3 profile)"
        python 0424_modelzoo_comparations/eval_motion_only_corruptions.py \
            --classification-checkpoint "$cls_ckpt" \
            --dino-detect-checkpoint "$dino_ckpt" \
            --wildrf-root "$WILDRF_ROOT" \
            --platforms reddit facebook twitter \
            --profile paper3 \
            --batch-size 16 --num-workers 32 \
            --output-dir "$CROSS_CORRUPTION_DIR"
    else
        warn "缺少 checkpoint,跳过跨退化评测:"
        require_file "$cls_ckpt"
        require_file "$dino_ckpt"
    fi

    log "实验二完成 — rebuttal 表见 ${CROSS_CORRUPTION_DIR}/<时间戳+哈希>/paper_table.md"
    info "重点产物: paper_table.md / severity_macro.csv / corruption_summary.csv / per_image_predictions.csv / results.json"
    info "  classification-only: $cls_ckpt"
    info "  motion-only DINO-Detect: $dino_ckpt"
}

# ---------------- 用法 ----------------
usage() {
    sed -n '2,/^set -euo/p' "$0" | sed 's/^# \?//' | sed '/^set -euo/d'
    exit 1
}

# ---------------- 调度 ----------------
declare -A STAGE_FN=(
    [env]=run_env
    [exp1]=run_exp1
    [exp2]=run_exp2
)

if [[ $# -eq 0 ]]; then
    STAGES=(env exp1 exp2)
else
    STAGES=("$@")
fi

for s in "${STAGES[@]}"; do
    case "$s" in
        all) run_env; run_exp1; run_exp2 ;;
        env|exp1|exp2) "${STAGE_FN[$s]}" ;;
        -h|--help|help) usage ;;
        *) echo "未知 stage: $s" >&2; usage ;;
    esac
done

log "全部完成。日志: $LOG_FILE"
[[ -n "${TEE_PID:-}" ]] && wait "$TEE_PID" 2>/dev/null || true
