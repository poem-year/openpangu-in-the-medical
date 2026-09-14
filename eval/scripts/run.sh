#!/usr/bin/env bash
# 长期测评框架的唯一入口：跑一次实验（结果永远写进新目录，不覆盖历史）。
#
#   bash eval/scripts/run.sh --list                      # 看有哪些实验配置
#   bash eval/scripts/run.sh eval/configs/agent-4x.yml   # 跑一个实验
#   bash eval/scripts/run.sh eval/configs/agent-smoke.yml --limit 8 --no-judge
#   bash eval/scripts/run.sh eval/configs/l3-hybrid.yml --dry-run   # 只看计划
#
# 想加新实验：在 eval/configs/ 里加一个 yml，不用改代码。

set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
# harness 是 eval/ 下的包，跑之前把 eval 加进模块搜索路径
export PYTHONPATH="eval:${PYTHONPATH:-}"

if [[ $# -eq 0 ]]; then
  echo "用法：bash eval/scripts/run.sh <配置> [覆盖项...]" >&2
  echo "      bash eval/scripts/run.sh --list" >&2
  exit 2
fi
if [[ "$1" == "--list" ]]; then
  exec .venv/bin/python -m harness.runner --list
fi
if [[ ! -f "$1" ]]; then
  echo "找不到配置：$1（用 --list 看可用配置）" >&2
  exit 2
fi
config="$1"; shift
exec .venv/bin/python -m harness.runner --config "${config}" "$@"
