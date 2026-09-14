#!/usr/bin/env bash
# 实时看某次运行（默认看最新一次）：bash eval/scripts/watch.sh [run-id]
#
# run-id 可以直接写时间戳前缀（唯一即可），不用记全名：
#   bash eval/scripts/watch.sh 20260914T1520
# 想看全部在跑的轮次（多轮总览）：
#   bash eval/scripts/watch.sh --all
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=.venv-pangu/bin/python

# --all：多轮总览（进程数 / NPU / 磁盘 / 每轮进度条）
if [[ "${1:-}" == "--all" ]]; then
  runs=()
  for d in eval/runs/*/; do
    [[ -f "${d}progress.json" ]] || continue
    stage=$(grep -o '"stage": *"[^"]*"' "${d}progress.json" 2>/dev/null | head -1 | cut -d'"' -f4)
    [[ "${stage}" == "done" ]] && continue
    # 只认「最近 3 分钟还在更新」的轮次，否则停掉的旧轮次会被当成在跑
    if [[ -n "$(find "${d}progress.json" -mmin +3 2>/dev/null)" ]]; then continue; fi
    runs+=("$(basename "${d}")")
  done
  if [[ ${#runs[@]} -eq 0 ]]; then
    runs=($(ls -1dt eval/runs/*/ 2>/dev/null | head -3 | xargs -r -n1 basename))
  fi
  trap 'echo; echo "已退出看板（后台评测仍在跑）。"; exit 0' INT
  while true; do
    clear
    echo "════════════════════════════════════════════════════════════════════"
    echo " 测评总览  $(date '+%F %T')   在跑：${runs[*]}"
    echo "════════════════════════════════════════════════════════════════════"
    echo " 评测进程：$(pgrep -fc '[r]un_eval|[r]un_agent_batch' 2>/dev/null || echo 0)"
    echo " NPU ：$(npu-smi info 2>/dev/null | awk -F'|' '/910B2C/ {gsub(/^ +| +$/,"",$3); print $3}' | head -1)"
    echo " 磁盘：$(df -h /data | awk 'NR==2 {print $3" 已用 / "$2"，剩 "$4}')"
    echo
    "${PY}" eval/watch.py --run-ids "${runs[@]}" --once
    sleep 3
  done
fi

latest() {
  ls -1dt eval/runs/*/ 2>/dev/null | head -1 | xargs -r basename
}

run_id="${1:-}"
if [[ -z "${run_id}" ]]; then
  run_id="$(latest)"
  echo "（未指定 run-id，看最新一次：${run_id}）"
elif [[ ! -d "eval/runs/${run_id}" ]]; then
  matches=$(ls -1d eval/runs/*"${run_id}"*/ 2>/dev/null | head -3)
  if [[ -z "${matches}" ]]; then
    echo "找不到匹配的 run：${run_id}" >&2
    exit 2
  fi
  run_id=$(basename "${matches}" | head -1)
  echo "（按前缀匹配到：${run_id}）"
fi

echo "日志：eval/runs/${run_id}/harness.log（tail -f 看原始输出）"
exec .venv-pangu/bin/python eval/watch.py --run-id "${run_id}"
