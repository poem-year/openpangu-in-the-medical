#!/usr/bin/env bash
# 把 eval/runs/<run-id>/ 里的**报告类小文件**快照到 评测报告/<run-id>/，便于入库。
#
# 为什么不直接提交 eval/runs/：那里还有 predictions.jsonl / scores.jsonl / contexts.jsonl，
# 一轮就好几 MB 到几十 MB，且随时可重跑；报告类文件（summary.md / report.html /
# compare.md / skipped.jsonl）每轮只有几十 KB，值得留痕。
#
# 用法：
#   bash 评测报告/collect.sh                      # 收集全部有 summary.json 的轮次
#   bash 评测报告/collect.sh L3-slow-v3 L2-slow-v3 L0-slow-v3

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

OUT_ROOT="${OUT_ROOT:-评测报告}"
FILES=(summary.md summary.json report.html compare.md skipped.jsonl)

runs=("$@")
if [[ ${#runs[@]} -eq 0 ]]; then
  mapfile -t runs < <(for d in eval/runs/*/; do
    [[ -f "${d}summary.json" ]] && basename "$d"
  done)
fi

if [[ ${#runs[@]} -eq 0 ]]; then
  echo "没有找到任何完成的轮次（eval/runs/*/summary.json）" >&2
  exit 1
fi

for run in "${runs[@]}"; do
  src="eval/runs/${run}"
  dst="${OUT_ROOT}/${run}"
  if [[ ! -d "${src}" ]]; then
    echo "✗ 跳过 ${run}：${src} 不存在" >&2
    continue
  fi
  mkdir -p "${dst}"
  copied=0
  for f in "${FILES[@]}"; do
    if [[ -f "${src}/${f}" ]]; then
      cp "${src}/${f}" "${dst}/${f}"
      copied=$((copied + 1))
    fi
  done
  echo "✓ ${run}：快照 ${copied} 个文件 → ${dst}/"
done

echo
echo "提醒：新轮次记得在 ${OUT_ROOT}/README.md 的索引表里登记（说明协议与是否可用）。"
