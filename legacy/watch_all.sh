#!/usr/bin/env bash
# 一个窗口盯住全部评测轮次（含预检索阶段的进度）。
#
#   bash watch_all.sh                       # 自动盯所有 *-v3 轮次
#   bash watch_all.sh L3-slow-v3 L2-slow-v3 # 指定轮次
#
# Ctrl-C 只退出看板，不影响后台评测。

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=.venv-pangu/bin/python
INTERVAL="${WATCH_INTERVAL:-3}"

runs=("$@")
if [[ ${#runs[@]} -eq 0 ]]; then
  # 自动盯正式轮次；冒烟轮次（smoke-*）不混进来看板
  mapfile -t runs < <(for d in eval/runs/*-v*/ eval/runs/*-4x/; do
    name=$(basename "$d")
    [[ "$name" == smoke-* ]] && continue
    [[ -d "$d" ]] && echo "$name"
  done)
fi
if [[ ${#runs[@]} -eq 0 ]]; then
  echo "没有找到任何 *-v3 轮次；也可以手动指定：bash watch_all.sh <run-id> …" >&2
  exit 1
fi

trap 'echo; echo "已退出看板（后台评测仍在跑）。"; exit 0' INT
while true; do
  clear
  echo "════════════════════════════════════════════════════════════════════════════════"
  echo " 评测总览  $(date '+%F %T')   轮次：${runs[*]}"
  echo "════════════════════════════════════════════════════════════════════════════════"
  echo " 进程：$(pgrep -fc "[r]un_eval" 2>/dev/null || echo 0) 个 run_eval 在跑"
  echo " NPU ：$(npu-smi info 2>/dev/null | awk -F'|' '/910B2C/ {gsub(/^ +| +$/,"",$3); print $3}' | head -1)"
  echo " 磁盘：$(df -h /data | awk 'NR==2 {print $3" 已用 / "$2"，剩 "$4}')"
  # 预检索阶段看板不显示，这里补上
  for run in "${runs[@]}"; do
    ctx="eval/runs/${run}/contexts.jsonl"
    if [[ -f "$ctx" && ! -f "eval/runs/${run}/predictions.jsonl" ]]; then
      echo " ${run}：预检索已写 $(wc -l < "$ctx") 行（这一阶段要看能否到 312 行）"
    fi
  done
  echo
  "${PY}" eval/watch.py --run-ids "${runs[@]}" --once
  sleep "${INTERVAL}"
done
