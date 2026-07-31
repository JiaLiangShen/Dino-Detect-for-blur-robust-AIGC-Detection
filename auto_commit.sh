#!/usr/bin/env bash
#
# auto_commit.sh — 定时检测 git 变更并自动提交推送(实验跑着时用来备份产物)
#
# 用法:
#   ./auto_commit.sh                 # 立即检测一次,之后每 1 小时一次,循环不断
#   INTERVAL=1800 ./auto_commit.sh   # 自定义间隔(秒),如半小时
#   ./auto_commit.sh --once          # 只检测一次后退出(测试用)
#   PROXY=http://1.2.3.4:port ./auto_commit.sh   # 覆盖代理地址
#   TMUX_SESSION=1 ./auto_commit.sh  # 训练所在的 tmux 会话(默认 1)
#
# 行为:
#   - 每轮:先捕获 tmux 训练输出快照 → git add -A → 有暂存变更则 commit+push
#   - .pth / __pycache__ / 本脚本自身日志已 gitignore;训练输出快照会提交
#   - 自动跳过 >SIZE_LIMIT_MB(默认50MB)的单文件,避免被 GitHub 拒收
#   - push 失败不退出,下一轮自动重试
#   - 日志同时输出到终端与 logs/auto_commit.log(已 gitignore,不触发死循环提交)
#
# 建议:在 tmux/nohup 里后台跑
#   nohup ./auto_commit.sh > /dev/null 2>&1 &
#   tail -f logs/auto_commit.log    # 实时看

set -uo pipefail

# ---------------- 代理(访问 GitHub 必需) ----------------
PROXY="${PROXY:-http://172.19.92.23:13128}"
export https_proxy="$PROXY"
export http_proxy="$PROXY"

# ---------------- 配置(可被环境变量覆盖) ----------------
INTERVAL="${INTERVAL:-3600}"          # 检测间隔(秒),默认 1 小时
SIZE_LIMIT_MB="${SIZE_LIMIT_MB:-50}"  # 单文件大小上限(MB),超过则跳过该文件
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/auto_commit.log"

# tmux 训练日志捕获(等效于 ~/.bashrc 里的 tmux-capture 函数)
# 每轮提交前抓取指定 pane 最近 TMUX_CAPTURE_LINES 行,写入 TMUX_CAPTURE_OUT
TMUX_SESSION="${TMUX_SESSION:-1}"          # tmux 会话名/编号(run_experiments.sh 所在)
TMUX_PANE="${TMUX_PANE:-0}"               # 窗口内的 pane 编号
TMUX_CAPTURE_LINES="${TMUX_CAPTURE_LINES:-10000}"
TMUX_CAPTURE_OUT="$LOG_DIR/tmux_run_output.log"  # 可提交的训练输出快照

# 切到仓库根目录(本脚本所在目录)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$LOG_DIR"
# 同时写终端与日志文件
exec > >(tee -a "$LOG_FILE") 2>&1

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(ts)" "$*"; }
info() { printf '\033[0;36m[%s]\033[0m %s\n' "$(ts)" "$*"; }
warn() { printf '\033[1;33m[%s] WARNING: %s\033[0m\n' "$(ts)" "$*" >&2; }

ONCE=0
[[ "${1:-}" == "--once" ]] && ONCE=1

trap 'log "收到退出信号,停止自动提交。"; exit 0' INT TERM

# ---------------- 单轮检测 ----------------
do_check() {
    log "检测 git 变更..."

    # 先捕获 tmux 训练输出快照(等效 tmux-capture <session> <lines>),
    # 写入可提交的日志文件;tmux 不可用或会话不存在时跳过,不阻断主流程
    if command -v tmux >/dev/null 2>&1; then
        if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
            tmux capture-pane -t "${TMUX_SESSION}:${TMUX_PANE}" -p \
                -S -"${TMUX_CAPTURE_LINES}" > "$TMUX_CAPTURE_OUT" 2>/dev/null || true
            info "已捕获 tmux 会话 ${TMUX_SESSION} 最近 ${TMUX_CAPTURE_LINES} 行 → ${TMUX_CAPTURE_OUT}"
        else
            warn "tmux 会话 '${TMUX_SESSION}' 不存在,跳过捕获(可用 TMUX_SESSION= 覆盖)"
        fi
    else
        warn "未安装 tmux,跳过训练输出捕获"
    fi

    # 暂存所有变更(被 .gitignore 忽略的 .pth/__pycache__/本日志不会进来)
    git add -A

    # 跳过过大的文件,防止 GitHub 拒收(>100MB 会硬失败,这里 50MB 预警)
    local limit=$((SIZE_LIMIT_MB * 1024 * 1024))
    local f size
    while IFS= read -r f; do
        [ -f "$f" ] || continue   # 删除的文件无大小,跳过
        size=$(wc -c < "$f" 2>/dev/null || echo 0)
        if [ "$size" -gt "$limit" ]; then
            git reset -q -- "$f"
            local hsize
            hsize=$(numfmt --to=iec "$size" 2>/dev/null || echo "${size}B")
            warn "跳过大文件(${hsize} > ${SIZE_LIMIT_MB}MB),不提交: $f"
        fi
    done < <(git diff --cached --name-only)

    # 暂存区是否还有内容
    if git diff --cached --quiet; then
        info "无变更,跳过本轮"
        return 0
    fi

    local n shortstat
    n=$(git diff --cached --name-only | wc -l)
    shortstat=$(git diff --cached --shortstat | sed 's/^ *//')
    [ -z "$shortstat" ] && shortstat="(统计不可用)"

    # 提交
    local commit_rc
    git commit -q \
        -m "chore: auto-commit @ $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        -m "${n} paths; ${shortstat}" >/dev/null 2>&1
    commit_rc=$?
    if [ "$commit_rc" -ne 0 ]; then
        warn "提交失败(exit=${commit_rc}),跳过推送。可能原因:无变更/身份未配置/钩子拒绝"
        return 1
    fi
    log "已提交(${n} 个文件,${shortstat})"

    # 推送
    local branch push_out push_rc
    branch=$(git branch --show-current 2>/dev/null || echo "main")
    push_out=$(git push origin "$branch" 2>&1)
    push_rc=$?
    # 过滤空行后回显推送输出
    echo "$push_out" | grep -vE '^[[:space:]]*$' || true
    if [ "$push_rc" -eq 0 ]; then
        log "已推送到 origin/${branch}"
    else
        warn "推送失败(exit=${push_rc}),下一轮重试。代理=${PROXY}"
    fi
}

# ---------------- 主循环 ----------------
log "自动提交已启动 | 间隔=${INTERVAL}s($((INTERVAL/60))分钟) | 代理=${PROXY} | 大文件上限=${SIZE_LIMIT_MB}MB"
log "首次检测立即执行。Ctrl+C 停止。"

while true; do
    do_check || true
    if [ "$ONCE" -eq 1 ]; then
        log "--once 模式,运行完毕退出"
        break
    fi
    log "等待 ${INTERVAL}s 后再次检测..."
    sleep "$INTERVAL"
done
