#!/usr/bin/env bash
# 在新服务器（同样的 910B2C + openEuler + CANN 9.0.0 容器）上一条命令恢复整套环境。
#
#   bash setup/restore.sh            # 缺什么装什么，最后跑冒烟验证
#   bash setup/restore.sh --check    # 只体检，不安装
#   bash setup/restore.sh --skip-cann # 已经装过 CANN 时跳过那一步
#
# 会做四件事（都可重复执行，已就绪的会跳过）：
#   1. 装完整 CANN 8.3.RC1 到 /data/Ascend（容器自带的 CANN 9.0.0 缺 910B 算子内核，跑不了）
#   2. git-lfs 拉权重到 /data/models/openPangu-R-7B-2512（约 17.6GB）
#   3. 建 .venv-pangu 并装 torch 2.7.1+cpu / torch-npu 2.7.1 / transformers 4.53.2
#   4. 冒烟：NPU 算子 + 模型加载 + 一次生成

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/openpangu}"
MODEL_DIR="${MODEL_DIR:-/data/models/openPangu-R-7B-2512}"
CANN_DIR="${CANN_DIR:-/data/Ascend}"
SYS_PY="${SYS_PY:-/usr/local/python3.11.15/bin/python3}"
VENV="${PROJECT_DIR}/.venv-pangu"
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
TORCH_WHEEL_URL="https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl"
CANN_TOOLKIT_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.3.RC1/Ascend-cann-toolkit_8.3.RC1_linux-x86_64.run"
CANN_KERNELS_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.3.RC1/Ascend-cann-kernels-910b_8.3.RC1_linux-x86_64.run"
MODEL_GIT="https://atomgit.com/ascend-tribe/openPangu-R-7B-2512.git"

CHECK_ONLY=0
SKIP_CANN=0
for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1 ;;
    --skip-cann) SKIP_CANN=1 ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  esac
done

ok()   { echo "  ✅ $*"; }
warn() { echo "  ⚠️  $*"; }
bad()  { echo "  ❌ $*"; }

need_cann() {
  # 判据：910B 算子内核目录存在且不是空壳（容器自带的 CANN 只有 ops_oam，几十 MB）
  local kdir="${CANN_DIR}/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b"
  [[ -d "${kdir}" ]] || return 0
  local n
  n=$(ls "${kdir}" 2>/dev/null | wc -l)
  [[ "${n}" -lt 50 ]]   # 完整内核有上千个算子，只有几十个说明是残缺安装
}
need_model()  { [[ ! -f "${MODEL_DIR}/config.json" || ! -f "${MODEL_DIR}/model-00001-of-00004.safetensors" ]]; }
need_venv()   { [[ ! -x "${VENV}/bin/python" ]]; }

echo "════════════════════════════════════════════════════════════════════"
echo " openPangu-R-7B-2512 环境恢复 ｜ 项目 ${PROJECT_DIR}"
echo "════════════════════════════════════════════════════════════════════"

# ---------- 0. 基础体检 ----------
echo; echo "① 基础环境"
[[ -x "${SYS_PY}" ]] && ok "Python ${SYS_PY}" || { bad "找不到 ${SYS_PY}"; exit 1; }
if command -v npu-smi >/dev/null 2>&1; then
  npu_line=$(npu-smi info 2>/dev/null | sed -n '7p')
  ok "NPU 可见：$(echo "${npu_line}" | tr -s ' ' | cut -c1-60)"
else
  bad "没有 npu-smi，确认这是昇腾容器"; exit 1
fi
[[ -d /usr/local/Ascend/driver ]] && ok "驱动目录存在" || warn "未见 /usr/local/Ascend/driver"
if need_cann; then
  warn "容器自带的 CANN 缺 910B 算子内核（NPU 上连 add 都会报 Parse dynamic kernel config fail）"
else
  ok "CANN 8.3.RC1 已就绪（${CANN_DIR}）"
fi
if need_model; then warn "模型权重尚未下载（${MODEL_DIR}）"; else ok "模型权重已就绪"; fi
if need_venv; then warn "Python 环境尚未创建（${VENV}）"; else ok "Python 环境已就绪"; fi

if [[ ${CHECK_ONLY} -eq 1 ]]; then
  echo; echo "体检结束（--check 模式，未做任何安装）"; exit 0
fi

# ---------- 1. CANN 8.3.RC1 ----------
echo; echo "② CANN 8.3.RC1"
if [[ ${SKIP_CANN} -eq 1 || ! need_cann ]]; then
  ok "跳过（已就绪）"
else
  mkdir -p "${CANN_DIR}"
  tmp_toolkit="/data/Ascend-cann-toolkit_8.3.RC1.run"
  tmp_kernels="/data/Ascend-cann-kernels-910b_8.3.RC1.run"
  [[ -s "${tmp_toolkit}" ]] || { echo "  下载 toolkit（2.4GB）..."; curl -sSL -o "${tmp_toolkit}" "${CANN_TOOLKIT_URL}" || exit 1; }
  [[ -s "${tmp_kernels}" ]] || { echo "  下载 kernels-910b（2.2GB）..."; curl -sSL -o "${tmp_kernels}" "${CANN_KERNELS_URL}" || exit 1; }
  echo "  安装 toolkit 到 ${CANN_DIR} ..."
  bash "${tmp_toolkit}" --quiet --install --install-path="${CANN_DIR}" > /tmp/cann_toolkit_install.log 2>&1 || { bad "toolkit 安装失败，见 /tmp/cann_toolkit_install.log"; exit 1; }
  echo "  安装 910B 算子内核 ..."
  bash "${tmp_kernels}" --quiet --install --install-path="${CANN_DIR}" > /tmp/cann_kernels_install.log 2>&1 || { bad "kernels 安装失败，见 /tmp/cann_kernels_install.log"; exit 1; }
  rm -f "${tmp_toolkit}" "${tmp_kernels}"
  ok "CANN 8.3.RC1 安装完成"
fi

# ---------- 2. 模型权重 ----------
echo; echo "③ 模型权重"
if ! need_model; then
  ok "跳过（已就绪）"
else
  command -v git-lfs >/dev/null 2>&1 || { bad "没有 git-lfs，先装 git-lfs"; exit 1; }
  mkdir -p "$(dirname "${MODEL_DIR}")"
  if [[ -d "${MODEL_DIR}/.git" ]]; then
    echo "  已有仓库，补拉 LFS 文件（约 17.6GB，可断点重试）..."
    git -C "${MODEL_DIR}" lfs pull || warn "lfs pull 未完成，重跑本脚本会继续"
  else
    echo "  克隆仓库（跳过 LFS，先把代码拉下来）..."
    GIT_LFS_SKIP_SMUDGE=1 git clone "${MODEL_GIT}" "${MODEL_DIR}" || exit 1
    echo "  拉权重（约 17.6GB）..."
    git -C "${MODEL_DIR}" config lfs.concurrenttransfers 4
    git -C "${MODEL_DIR}" lfs pull || warn "lfs pull 未完成，重跑本脚本会继续"
  fi
  need_model && bad "权重仍不完整，请重跑本脚本" || ok "权重就绪"
fi

# ---------- 3. Python 环境 ----------
echo; echo "④ Python 环境（${VENV}）"
if ! need_venv; then
  ok "跳过（已存在）"
else
  "${SYS_PY}" -m venv "${VENV}" || exit 1
  "${VENV}/bin/pip" install -q --upgrade pip >/dev/null 2>&1
  echo "  安装 torch 2.7.1+cpu ..."
  "${VENV}/bin/pip" install -q --index-url "${PIP_INDEX}" "${TORCH_WHEEL_URL}" || {
    bad "torch 安装失败（检查到 mirrors.aliyun.com 的网络）"; exit 1; }
  echo "  安装 torch-npu / transformers 等 ..."
  "${VENV}/bin/pip" install -q --index-url "${PIP_INDEX}" \
      torch-npu==2.7.1 transformers==4.53.2 sentencepiece einops \
      decorator attrs psutil scipy cffi protobuf absl-py wheel || { bad "依赖安装失败"; exit 1; }
  ok "Python 环境就绪"
fi

# ---------- 4. 冒烟 ----------
echo; echo "⑤ 冒烟验证"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/pangu_env.sh"
cd "${PROJECT_DIR}" || exit 1
"${VENV}/bin/python" - <<'PY' || { bad "NPU 算子没跑通"; exit 1; }
import torch, torch_npu
x = torch.ones(64, 64).npu()
print("  ✅ NPU 可用：", torch.npu.get_device_name(0), "｜ 算力自检", (x @ x).sum().item())
PY
"${VENV}/bin/python" scripts/smoke_pangu.py --max-new-tokens 32 --skip-fast 2>&1 | tail -3 || { bad "模型冒烟失败"; exit 1; }

echo
echo "════════════════════════════════════════════════════════════════════"
echo " 恢复完成。下一步："
echo "   bash run_eval.sh --quick --watch     # 冒烟跑一遍评测流程"
echo "   bash run_eval.sh --watch             # 正式跑（慢思考 + 快思考）"
echo "════════════════════════════════════════════════════════════════════"
