# 环境恢复说明（换服务器用这份）

适用场景：**同样的模力方舟昇腾容器**（910B2C 单卡 / openEuler 24.03 / 容器自带 CANN 9.0.0 / Python 3.11.15）。
项目代码从 git 拉下来后，环境在容器里，**不在 git 里**，所以换机器只要跑一条命令重建：

```bash
cd /data/openpangu
bash setup/restore.sh          # 缺什么装什么，最后自动冒烟
bash setup/restore.sh --check  # 只体检，不安装
```

## 它会做什么

| 步骤 | 内容 | 耗时 |
| --- | --- | --- |
| ① 基础体检 | 检查 Python 3.11.15、NPU 可见、驱动目录；判断容器自带 CANN 是否可用 | 秒级 |
| ② CANN 8.3.RC1 | 从华为公开镜像下载 toolkit（2.4GB）+ 910B 算子内核（2.2GB），装到 `/data/Ascend` | 下载+安装约 10–15 分钟 |
| ③ 模型权重 | git-lfs 从 AtomGit 拉 `openPangu-R-7B-2512` 到 `/data/models`（17.6GB） | 20–60 分钟（看网速） |
| ④ Python 环境 | 建 `.venv-pangu`，装 torch 2.7.1+cpu / torch-npu 2.7.1 / transformers 4.53.2 等 | 3–5 分钟 |
| ⑤ 智能体环境 | 建 `.venv`，装 `requirements.txt`（LangChain 智能体 + A 线检索依赖，torch 走 CPU 轮子） | 3–5 分钟 |
| ⑥ 冒烟 | NPU 算力自检 + 模型加载 + 一次 32 token 生成 + 智能体离线测试全绿 | 1–2 分钟 |

每一步都**幂等**：已经就绪的会跳过，中断了重跑会接着来。

## 为什么不用容器自带的 CANN 9.0.0

该镜像里的 CANN 是残缺的：`opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910b/` 下只有 `ops_oam`，
缺 `ops_math` / `ops_nn` / `ops_cv` / `ops_transformer` / `ops_legacy`，`opp/static_kernel` 也不存在。
结果就是 NPU 上连一个 `add` 都跑不起来（报 `Parse dynamic kernel config fail`）。
而 CANN 9.0.0 的 kernels 包公开源上没有，8.2.RC1 / 8.3.RC1 才有，所以选择装 8.3.RC1。

另外 CANN 8.3.RC1 安装脚本生成的 `set_env.sh` 第一行有个空格 bug（会把 `LD_LIBRARY_PATH` 写坏，
导致 `torch_npu` 找不到 `libascend_hal.so`），所以环境变量统一走项目里的 `scripts/pangu_env.sh`。

## 磁盘与网络要求

| 项 | 要求 |
| --- | --- |
| `/data` 空间 | ≥ 40GB（CANN 12GB + 权重 17.6GB + 环境 2GB + 余量） |
| 能访问 | `ascend-repo.obs.cn-east-2.myhuaweicloud.com`（CANN）、`atomgit.com`（权重）、`pypi.tuna.tsinghua.edu.cn`、`mirrors.aliyun.com`（torch） |
| 大文件位置 | 一律放 `/data`，不要放 `/`（系统盘只有 30GB） |

## 常见问题

- **`libascend_hal.so: cannot open shared object file`**：没有 source `scripts/pangu_env.sh`，或者错误地 source 了 CANN 自带的 `set_env.sh`。
- **`device id error`**：`ASCEND_RT_VISIBLE_DEVICES` 必须是 `0`（容器只暴露一张卡，宿主机编号是 1，容器内逻辑编号是 0），`pangu_env.sh` 已设好。
- **权重下载中断**：直接重跑 `bash setup/restore.sh`，`git lfs pull` 会继续；也可以 `git -C /data/models/openPangu-R-7B-2512 lfs pull` 单独续传。
- **磁盘快满**：权重仓的 `.git/lfs/objects` 可以在拉完后清掉（省 17.6GB），工作区文件不受影响，代价是以后要重下。

## 关于 git 里有什么、没有什么

克隆下来就有的：全部代码（`agent/`、`tests/`、`eval/`、`scripts/`、`rag_med_project/`）、
文档、`正式测评集/` 的 13 个 jsonl + manifest + `_build/` 构建脚本、`setup/` 恢复脚本。

不在 git 里的（体积大或可重建）：

| 内容 | 原因 | 怎么恢复 |
| --- | --- | --- |
| `正式测评集/_raw/` | 439MB，其中单个文件 202MB（SLAKE/imgs.zip），超过 GitHub 100MB 上限；且 SLAKE 是已废弃的多模态来源 | `cd 正式测评集/_build && python3 download.py`（需要能访问 HuggingFace 镜像） |
| 模型权重 | 17.6GB | `bash setup/restore.sh` |
| CANN 8.3.RC1 | 12GB | `bash setup/restore.sh` |
| `.venv-pangu/` | 1.3GB | `bash setup/restore.sh` |
| `.venv/` | 约 1.7GB（含 CPU 版 torch 与句向量模型依赖） | `bash setup/restore.sh` |
| `kb/corpus_raw/` | A 线原始采集件 647MB（单个文件 563MB，超 GitHub 100MB 上限），且是第三方许可语料 | 见 `kb/corpus_raw/README.md`：`_scripts/download.py` 重下 → `_scripts/convert.py` 转换 |
| `kb/corpus/` | A 线入库语料 83MB（125 份 md），由原始件转换而来 | 上一步的 `convert.py` 生成（`sources.yaml` 例外，它在 git 里） |
| `kb/index/` | A 线向量索引 478MB，由语料生成 | `python -m kb.build`（切块在 `.venv`、嵌入可用 `.venv-pangu` + NPU，见 `kb/README.md` §3.2） |
| `eval/runs/` | 评测结果，随时可重跑 | `bash run_eval.sh --watch` |
