#!/usr/bin/env bash
# 一键评测：openPangu-R-7B-2512 在《正式测评集》上跑 L0 基线
#
# 默认一次跑两种模式（慢思考 + 快思考）并给出对比：
#   bash run_eval.sh              # 312 题 × 两模式 + AI 判分，约 20 分钟
#   bash run_eval.sh --watch      # 同上，前台实时看板（推荐）
#
# 其它用法：
#   bash run_eval.sh --quick         # 冒烟：每维度 2 题、快思考，约 2–3 分钟
#   bash run_eval.sh --slow-only     # 只跑慢思考
#   bash run_eval.sh --fast-only     # 只跑快思考
#   bash run_eval.sh --no-judge      # 不调 AI，只出规则判分
#   bash run_eval.sh --estimate      # 只看范围与预估耗时
#   bash run_eval.sh --batch 16      # 改批大小
#
# 中断了直接重跑同一条命令：已完成的题会按配置指纹跳过。

set -uo pipefail

PROJECT_DIR="/data/openpangu"
DATASET_DIR="${PROJECT_DIR}/正式测评集"
PY="${PROJECT_DIR}/.venv-pangu/bin/python"
BATCH=32
QUICK=0
NO_JUDGE=0
ESTIMATE_ONLY=0
WATCH=0
SLOW_ONLY=0
FAST_ONLY=0
JUDGE_WORKERS=6
USER_RUN_ID=""
EXTRA_ARGS=()

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch)      WATCH=1; shift ;;
    --quick)      QUICK=1; shift ;;
    --slow-only)  SLOW_ONLY=1; shift ;;
    --fast-only)  FAST_ONLY=1; shift ;;
    --no-judge)   NO_JUDGE=1; shift ;;
    --estimate)   ESTIMATE_ONLY=1; shift ;;
    --batch)      BATCH="$2"; shift 2 ;;
    --run-id)     USER_RUN_ID="$2"; shift 2 ;;
    --workers)    JUDGE_WORKERS="$2"; shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *)            EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# ---------- 决定跑哪些模式、每个模式的 run id ----------
if [[ ${QUICK} -eq 1 ]]; then
  MODES=(fast); PER_DIM=2
elif [[ ${SLOW_ONLY} -eq 1 ]]; then
  MODES=(slow); PER_DIM=0
elif [[ ${FAST_ONLY} -eq 1 ]]; then
  MODES=(fast); PER_DIM=0
else
  MODES=(slow fast); PER_DIM=0
fi

RUN_IDS=()
for mode in "${MODES[@]}"; do
  if [[ -n "${USER_RUN_ID}" ]]; then
    if [[ ${#MODES[@]} -eq 1 ]]; then RUN_IDS+=("${USER_RUN_ID}"); else RUN_IDS+=("${USER_RUN_ID}-${mode}"); fi
  else
    RUN_IDS+=("L0-${mode}-b${BATCH}")
  fi
done

# --watch：把整个流水线丢后台，前台用多 run 看板刷新
if [[ ${WATCH} -eq 1 ]]; then
  REARGS=(--batch "${BATCH}" --workers "${JUDGE_WORKERS}")
  [[ ${QUICK} -eq 1 ]] && REARGS+=(--quick)
  [[ ${SLOW_ONLY} -eq 1 ]] && REARGS+=(--slow-only)
  [[ ${FAST_ONLY} -eq 1 ]] && REARGS+=(--fast-only)
  [[ ${NO_JUDGE} -eq 1 ]] && REARGS+=(--no-judge)
  [[ ${ESTIMATE_ONLY} -eq 1 ]] && REARGS+=(--estimate)
  [[ -n "${USER_RUN_ID}" ]] && REARGS+=(--run-id "${USER_RUN_ID}")
  # 透传未知参数（例如 --limit、--dimensions），否则后台子进程会丢掉它们
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then REARGS+=("${EXTRA_ARGS[@]}"); fi
  LOG_FILE="${PROJECT_DIR}/eval/runs/run_$(date +%m%d_%H%M%S).log"
  mkdir -p "${PROJECT_DIR}/eval/runs"
  setsid nohup bash "$0" "${REARGS[@]}" > "${LOG_FILE}" 2>&1 < /dev/null &
  PIPELINE_PID=$!
  echo "评测已在后台启动（模式：${MODES[*]}），日志 ${LOG_FILE}"
  echo "下面显示实时看板，Ctrl-C 只退出看板、不影响评测。"
  trap 'echo; echo "已退出看板；评测仍在后台运行（PID ${PIPELINE_PID}）。"; \
        echo "要停止评测：bash ${PROJECT_DIR}/stop_eval.sh（或 kill ${PIPELINE_PID}）"; exit 0' INT
  sleep 3
  while kill -0 "${PIPELINE_PID}" 2>/dev/null; do
    clear
    "${PY}" "${PROJECT_DIR}/eval/watch.py" --run-ids "${RUN_IDS[@]}" --once
    sleep 2
  done
  clear
  "${PY}" "${PROJECT_DIR}/eval/watch.py" --run-ids "${RUN_IDS[@]}" --once
  echo
  for rid in "${RUN_IDS[@]}"; do
    if [[ -f "${PROJECT_DIR}/eval/runs/${rid}/summary.json" ]]; then
      "${PY}" "${PROJECT_DIR}/eval/show_report.py" --run-dir "${PROJECT_DIR}/eval/runs/${rid}"
    else
      echo "✗ ${rid} 没有生成报告，日志：${LOG_FILE}" >&2
      tail -n 20 "${LOG_FILE}" >&2
    fi
  done
  if [[ ${#RUN_IDS[@]} -ge 2 ]]; then
    "${PY}" "${PROJECT_DIR}/eval/compare.py" --run-ids "${RUN_IDS[@]}" || true
  fi
  echo "完整日志：${LOG_FILE}"
  exit 0
fi

echo "════════════════════════════════════════════════════════════════════"
echo " 一键评测 · openPangu-R-7B-2512 × 正式测评集（332 题 / 13 维度）"
printf " 模式：%s ｜ 批大小 %s ｜ AI 判分 %s\n" \
       "${MODES[*]}" "${BATCH}" "$([[ ${NO_JUDGE} -eq 1 ]] && echo 关闭 || echo 开启)"
printf " 运行 ID：%s\n" "${RUN_IDS[*]}"
echo "════════════════════════════════════════════════════════════════════"

# ---------- 前置检查 ----------
missing=0
for path in "${PY}" "${PROJECT_DIR}/scripts/pangu_env.sh" \
            "/data/models/openPangu-R-7B-2512/config.json" \
            "/data/Ascend/ascend-toolkit/latest/opp" "${DATASET_DIR}"; do
  if [[ ! -e "${path}" ]]; then
    echo "✗ 缺少 ${path}" >&2
    missing=1
  fi
done
if [[ ${missing} -eq 1 ]]; then
  echo "  模型或环境未就绪，先看 ${PROJECT_DIR}/模型部署说明.md" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/pangu_env.sh"
cd "${PROJECT_DIR}"

START_TS=$(date +%s)
step() { echo; echo "───────── $* ─────────"; }

for idx in "${!MODES[@]}"; do
  mode="${MODES[$idx]}"
  rid="${RUN_IDS[$idx]}"
  run_dir="${PROJECT_DIR}/eval/runs/${rid}"
  COMMON=(--dataset-dir "${DATASET_DIR}" --run-id "${rid}" --layer L0 \
          --thinking "${mode}" --batch-size "${BATCH}" --per-dimension "${PER_DIM}")

  echo
  echo "████████ 模式：$([[ ${mode} == slow ]] && echo 慢思考 || echo 快思考)（run ${rid}）████████"

  step "① 估算范围与耗时"
  "${PY}" eval/run_eval.py --stage estimate "${COMMON[@]}" "${EXTRA_ARGS[@]}"
  if [[ ${ESTIMATE_ONLY} -eq 1 ]]; then continue; fi

  step "② 批量生成"
  GEN_TS=$(date +%s)
  "${PY}" eval/run_eval.py --stage generate "${COMMON[@]}" "${EXTRA_ARGS[@]}"
  echo "生成阶段耗时 $(( $(date +%s) - GEN_TS )) 秒"

  if [[ ${NO_JUDGE} -eq 0 ]]; then
    step "③ AI 判分（开放题诊断等价复核 + D09–D11 rubric）"
    JUDGE_TS=$(date +%s)
    "${PY}" eval/run_eval.py --stage judge "${COMMON[@]}" \
            --judge-workers "${JUDGE_WORKERS}" "${EXTRA_ARGS[@]}" || true
    if [[ ! -s "${run_dir}/answer_judgments.jsonl" && ! -s "${run_dir}/rubric_judgments.jsonl" ]]; then
      echo "  ⚠ AI 判分没有产出（可能没有可用密钥或网络不通），本轮只出规则判分结果"
    fi
    echo "判分阶段耗时 $(( $(date +%s) - JUDGE_TS )) 秒"
  fi

  step "④ 汇总与报告"
  "${PY}" eval/run_eval.py --stage score "${COMMON[@]}" "${EXTRA_ARGS[@]}"
  if [[ ! -f "${run_dir}/summary.json" ]]; then
    echo "✗ ${rid} 没有生成 summary.json，说明生成或判分阶段失败" >&2
    exit 1
  fi
done

if [[ ${ESTIMATE_ONLY} -eq 1 ]]; then exit 0; fi

step "⑤ 结果"
for rid in "${RUN_IDS[@]}"; do
  "${PY}" eval/show_report.py --run-dir "${PROJECT_DIR}/eval/runs/${rid}"
done
if [[ ${#RUN_IDS[@]} -ge 2 ]]; then
  "${PY}" eval/compare.py --run-ids "${RUN_IDS[@]}"
fi
echo "总耗时 $(( $(date +%s) - START_TS )) 秒"
echo "边跑边看进度：另开终端执行  bash ${PROJECT_DIR}/watch.sh ${RUN_IDS[0]}"
