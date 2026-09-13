#!/usr/bin/env bash
# 把当前项目提交并推送到你的 git 仓库
#
# 首次使用：先配一次远端（只需一次）
#   cd /data/openpangu
#   git remote add origin <你的仓库地址>
#   bash sync.sh "首次上传：评测方案与环境恢复脚本"
#
# 之后每次改完：
#   bash sync.sh "改了什么"

set -uo pipefail
cd "$(dirname "$0")" || exit 1

MSG="${1:-同步更新 $(date +%Y-%m-%d\ %H:%M)}"

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "✗ 还没有配置远端仓库，先执行：" >&2
  echo "   git remote add origin <你的仓库地址>" >&2
  exit 1
fi

echo "① 暂存改动"
git add -A
changed=$(git diff --cached --name-only | wc -l)
if [[ "${changed}" -eq 0 ]]; then
  echo "   没有新改动，直接推送已有提交"
else
  echo "   共 ${changed} 个文件"
  echo "② 提交：${MSG}"
  git commit -q -m "${MSG}" || { echo "✗ 提交失败" >&2; exit 1; }
fi

branch=$(git rev-parse --abbrev-ref HEAD)
echo "③ 推送到 origin/${branch}"
git push -u origin "${branch}" || {
  echo "✗ 推送失败。常见原因：" >&2
  echo "   - 远端仓库非空（先 git pull --rebase origin ${branch}）" >&2
  echo "   - 没配凭据（HTTPS 需要 token，SSH 需要公钥）" >&2
  exit 1
}

echo "✅ 完成：$(git log -1 --oneline)"
echo "   仓库地址：$(git remote get-url origin)"
