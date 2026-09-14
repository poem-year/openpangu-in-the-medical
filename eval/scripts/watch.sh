#!/usr/bin/env bash
# 实时看某次运行（默认看最新一次）：bash eval/scripts/watch.sh [run-id]
#
# run-id 可以直接写时间戳前缀（唯一即可），不用记全名：
#   bash eval/scripts/watch.sh 20260914T1520
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

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
