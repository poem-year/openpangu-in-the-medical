#!/usr/bin/env bash
# 停掉正在跑的评测（不影响已完成的结果与报告）。
#
#   bash eval/scripts/stop.sh          # 只停评测进程
#   bash eval/scripts/stop.sh --all    # 连模型服务一起停
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

stopped=0
for pattern in "[r]un_agent_batch" "[r]un_agent_eval" "[r]un_eval.py" "[h]arness.runner"; do
  pids=$(pgrep -f "${pattern}" || true)
  if [[ -n "${pids}" ]]; then
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true
    echo "已停：${pattern}（PID ${pids//$'\n'/ }）"
    stopped=1
  fi
done
[[ ${stopped} -eq 0 ]] && echo "没有正在跑的评测进程"

if [[ "${1:-}" == "--all" ]]; then
  bash scripts/serve_pangu.sh stop || true
fi

echo
echo "已完成的结果与报告没有被改动。继续跑用同一条 run 命令（框架会新开一个 run 目录）。"
