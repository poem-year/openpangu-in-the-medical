#!/usr/bin/env bash
# 实时看评测进度（另开一个终端跑，不影响评测本身）
#
#   bash /data/openpangu/watch.sh              # 默认看 L0-slow-b32
#   bash /data/openpangu/watch.sh smoke-quick  # 看指定 run
#   bash /data/openpangu/watch.sh L0-slow-b32 1   # 刷新间隔 1 秒

set -uo pipefail
RUN_ID="${1:-L0-slow-b32}"
INTERVAL="${2:-2}"
exec /data/openpangu/.venv-pangu/bin/python /data/openpangu/eval/watch.py \
     --run-id "${RUN_ID}" --interval "${INTERVAL}"
