# openPangu 医疗诊断能力增强

2026 中国国际大学生创新大赛 · 华为企业命题。用 RAG + 智能体增强 openPangu 在医学诊断场景的回答：
**结论准确、每条可追溯、高风险问题守边界**。

项目一页纸见 [`项目综述.md`](项目综述.md)，开工前的工作约定见 [`AGENTS.md`](AGENTS.md)，
机器实况与文件地图见 [`环境说明.md`](环境说明.md)。

## 换一台同样的服务器怎么用

前提：模力方舟昇腾容器（910B2C 单卡 / openEuler 24.03 / 容器自带 CANN 9.0.0 / Python 3.11.15）。

```bash
git clone https://github.com/poem-year/openpangu-in-the-medical.git /data/openpangu
cd /data/openpangu
bash setup/restore.sh          # 自动补齐 CANN 8.3.RC1 + 权重 + Python 环境，最后冒烟
```

`restore.sh` 是幂等的：缺什么装什么，下载中断重跑会接着来。约 30–80 分钟（大头是 17.6GB 权重）。
想先看要做什么，跑 `bash setup/restore.sh --check`。

跑通了就能直接评测：

```bash
bash run_eval.sh --quick --watch      # 冒烟：每维度 2 题，2–3 分钟
bash run_eval.sh --watch              # 正式：慢思考 + 快思考两种模式，含 AI 判分与对比表
```

跑完看 [`L0基线评测报告.md`](L0基线评测报告.md)（基线总表 + 慢/快对比），或直接看
`eval/runs/L0-slow-b32/summary.md` 与 `report.html`。

要跟智能体对话，先起模型服务（OpenAI 兼容，约 15 秒加载 + 预热）：

```bash
bash scripts/serve_pangu.sh start      # 停止用 stop，看状态用 status
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 MODEL_NAME=openpangu-7b .venv/bin/python -m agent.cli
.venv/bin/python scripts/smoke_agent_server.py    # 端到端冒烟：两轮真实对话
```

细节见 [`模型部署说明.md`](模型部署说明.md) 第八节。

## 目录导航

| 路径 | 内容 |
| --- | --- |
| `agent/` | B 线：LangGraph 智能体、工具、双层审查、提示词 |
| `kb/` | A 线：语料接入 + BM25/向量混合检索 + 建库/标定 CLI，说明见 `kb/README.md` |
| `rag_med_project/` | A 线旧版（队友交付的 Chroma 实现，仅作对照，不参与运行） |
| `正式测评集/` | 评测题库（13 维度，332 题，构建脚本在 `_build/`） |
| `eval/` | 评测方案与脚本（生成 → 判分 → 报告，含实时看板），说明见 `eval/README.md` |
| `L0基线评测报告.md` | L0 基线结果（332 题 / 13 维度，慢思考与快思考对比，含指标口径） |
| `L2L3检索层评测报告.md` | 检索层（L2 纯向量 / L3 混合检索）结果、截断问题的诊断与修法 |
| `scripts/` | 模型侧：环境变量、推理入口、OpenAI 兼容服务与启停脚本 |
| `setup/` | **换服务器时用这个**：环境恢复脚本与说明 |
| `tests/` | 离线测试（228 项，不需要模型和网络） |
| `模型部署说明.md` | 本机部署细节、实测数据、已知坑 |
| `AGENTS.md` | 在这台机器上工作的约定（改代码前先读） |

## 什么不在 git 里

权重（17.6GB）、CANN（12GB）、Python 环境（1.3GB）、评测结果、原始数据集与知识库语料
都按体积排除，分别用 `setup/restore.sh`、`run_eval.sh`、`正式测评集/_build/download.py`、
`kb/corpus_raw/_scripts/download.py` + `convert.py` + `python -m kb.build` 重建，
完整对照表见 `setup/README.md`。

## 日常改动怎么同步

```bash
bash sync.sh "改了什么"
```
