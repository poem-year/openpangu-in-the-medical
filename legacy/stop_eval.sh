#!/usr/bin/env bash
# 停止正在跑的评测（一键脚本 / 单独跑的 run_eval.py 都能停）
#
#   bash /data/openpangu/stop_eval.sh

set -uo pipefail

# 用 python 精确匹配进程的可执行文件与脚本参数，避免误伤"命令行里恰好含 run_eval.sh"的其他进程
pids=$(/usr/local/python3.11.15/bin/python3 - <<'PY'
import os
import subprocess

me = {os.getpid(), os.getppid()}
targets = []
out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
for line in out.splitlines()[1:]:
    pid, _, args = line.strip().partition(" ")
    if not pid.isdigit() or int(pid) in me:
        continue
    parts = args.split()
    if not parts:
        continue
    exe = os.path.basename(parts[0])
    if exe in ("bash", "sh") and len(parts) > 1 and os.path.basename(parts[1]) == "run_eval.sh":
        targets.append(pid)
    elif exe.startswith("python") and any(
        os.path.basename(p) in ("run_eval.py", "watch.py", "judge.py", "ai_judge.py")
        for p in parts[1:3]
    ):
        targets.append(pid)
print(" ".join(targets))
PY
)
if [[ -z "${pids}" ]]; then
  echo "没有正在运行的评测进程。"
  exit 0
fi

echo "要停止的进程："
ps -o pid,etime,args -p $(echo "${pids}" | tr '\n' ',' | sed 's/,$//') 2>/dev/null
for pid in ${pids}; do kill -TERM "${pid}" 2>/dev/null; done
sleep 5

left=$(ps -eo pid,args | awk '$0 ~ /run_eval\.py|run_eval\.sh/ && $0 !~ /stop_eval|awk/ {print $1}')
if [[ -n "${left}" ]]; then
  echo "仍有进程未退出，强制结束：${left}"
  for pid in ${left}; do kill -9 "${pid}" 2>/dev/null; done
  sleep 2
fi

echo "已停止。已完成的部分保留在 eval/runs/ 下，下次重跑会自动续跑。"
