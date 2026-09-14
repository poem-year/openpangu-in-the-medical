#!/usr/bin/env bash
# 启停 openPangu 的 OpenAI 兼容服务（B 线智能体连的就是它）
#
#   bash scripts/serve_pangu.sh start     # 起服务（后台，日志在 logs/pangu_server.log）
#   bash scripts/serve_pangu.sh status    # 看状态 + 健康检查
#   bash scripts/serve_pangu.sh stop      # 停服务
#   bash scripts/serve_pangu.sh restart
#
# 说明：本机不允许用 systemctl / Docker（见 AGENTS.md），所以用 setsid+nohup 常驻，
# PID 记在 logs/pangu_server.pid。CANN 会在当前目录写临时文件，这里固定 cd 到项目根目录。

set -uo pipefail

PROJECT_DIR="/data/openpangu"
PY="${PROJECT_DIR}/.venv-pangu/bin/python"
LOG_DIR="${PROJECT_DIR}/logs"
PID_FILE="${LOG_DIR}/pangu_server.pid"
LOG_FILE="${LOG_DIR}/pangu_server.log"
HOST="127.0.0.1"
PORT="8000"

mkdir -p "${LOG_DIR}"

is_running() {
  [[ -f "${PID_FILE}" ]] || return 1
  local pid
  pid="$(cat "${PID_FILE}" 2>/dev/null)"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

health_ok() {
  "${PY}" -c "import sys,urllib.request,json;url=f'http://${HOST}:${PORT}/health';print(json.loads(urllib.request.urlopen(url,timeout=3).read())['status'])" 2>/dev/null
}

start() {
  if is_running; then
    echo "服务已在运行（PID $(cat "${PID_FILE}")），端口 ${PORT}"
    return 0
  fi
  if [[ -f "${PID_FILE}" ]]; then
    echo "清理陈旧的 PID 文件（进程已不在）"
    rm -f "${PID_FILE}"
  fi
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/scripts/pangu_env.sh"
  cd "${PROJECT_DIR}"
  setsid nohup "${PY}" scripts/pangu_server.py --host "${HOST}" --port "${PORT}" \
    >> "${LOG_FILE}" 2>&1 < /dev/null &
  echo $! > "${PID_FILE}"
  echo "启动中（PID $(cat "${PID_FILE}")），日志 ${LOG_FILE}"
  for _ in $(seq 1 90); do
    if health_ok >/dev/null 2>&1; then
      echo "✅ 服务就绪：http://${HOST}:${PORT}/v1"
      echo "   B 线用法：OPENAI_BASE_URL=http://${HOST}:${PORT}/v1 MODEL_NAME=openpangu-7b .venv/bin/python -m agent.cli"
      return 0
    fi
    if ! is_running; then
      echo "✗ 进程已退出，看日志：tail -40 ${LOG_FILE}" >&2
      return 1
    fi
    sleep 1
  done
  echo "⚠ 90 秒内没等到健康检查通过，看日志：tail -40 ${LOG_FILE}" >&2
  return 1
}

stop() {
  if ! is_running; then
    echo "PID 文件指向的进程不在，检查是否有残留"
    pkill -f "scripts/pangu_server.py --host ${HOST}" 2>/dev/null && echo "已清理残留进程"
    rm -f "${PID_FILE}"
    return 0
  fi
  local pid
  pid="$(cat "${PID_FILE}")"
  echo "停止 PID ${pid} …"
  kill "${pid}" 2>/dev/null
  for _ in $(seq 1 20); do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 0.5
  done
  if kill -0 "${pid}" 2>/dev/null; then
    echo "进程没退出，强制结束"
    kill -9 "${pid}" 2>/dev/null
  fi
  # 兜底：模型加载时可能 fork 出辅助进程，按命令行再清一遍
  pkill -f "scripts/pangu_server.py --host ${HOST}" 2>/dev/null
  rm -f "${PID_FILE}"
  echo "已停止"
}

status() {
  if is_running; then
    echo "运行中：PID $(cat "${PID_FILE}")，端口 ${PORT}"
    if health_ok >/dev/null 2>&1; then
      echo "健康检查：通过"
      tail -n 3 "${LOG_FILE}" 2>/dev/null
    else
      echo "健康检查：未通过（进程在但 /health 无响应）"
    fi
  else
    echo "未运行"
  fi
}

case "${1:-status}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; start ;;
  status)  status ;;
  *)       echo "用法：bash scripts/serve_pangu.sh {start|stop|restart|status}"; exit 2 ;;
esac
