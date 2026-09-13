#!/bin/bash
# openPangu-R-7B-2512 运行环境变量（本机实测可用）
#
# 背景：容器自带 /usr/local/Ascend/cann-9.0.0 缺少 910B 算子内核（opp 下只有 ops_oam），
# NPU 上一个 add 都跑不起来。因此在 /data/Ascend 装了完整的 CANN 8.3.RC1（toolkit + kernels-910b）。
#
# 另外 CANN 8.3.RC1 安装脚本生成的 set_env.sh 第一行有个 bug（驱动路径里带空格，
# 会把 LD_LIBRARY_PATH 写坏，导致 torch_npu 找不到 libascend_hal.so），所以这里手工写全。
#
# 用法：source /data/openpangu/scripts/pangu_env.sh

export ASCEND_TOOLKIT_HOME=/data/Ascend/ascend-toolkit/latest
export ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}
export ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
export ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}
export TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit

export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64:${ASCEND_TOOLKIT_HOME}/tools/aml/lib64:${ASCEND_TOOLKIT_HOME}/tools/aml/lib64/plugin:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
export PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:${ASCEND_TOOLKIT_HOME}/tools/ccec_compiler/bin:$PATH

# 本容器只暴露一张卡。宿主机 npu-smi 里编号是 1，但容器内只有 /dev/davinci1，
# 运行时把它当逻辑 0；实测设成 0 可用，设成 1 会报 device id error。
export ASCEND_RT_VISIBLE_DEVICES=0
