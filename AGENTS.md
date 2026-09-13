# AGENTS.md · 本项目在这台服务器上的工作说明

> 改动本项目前先读完这份文件，再看 `交接文档.md` 和 `服务器环境说明.md`。

## 一、这是什么项目

「面向医疗诊断场景的 openPangu 能力增强方案」—— 2026 中国国际大学生创新大赛 · 华为企业命题。
用 RAG + 智能体增强 openPangu 在医学诊断场景的回答：**结论准确、每条可追溯、高风险问题守边界**。
项目一页纸见 `项目综述.md`。

## 二、目录导航

| 路径 | 内容 |
| --- | --- |
| `项目综述.md` | 项目一页纸（做什么 / 怎么做 / 谁做 / 怎么证明） |
| `交接文档.md` | **当前进度 + 下一步 + 已知坑（干活前必读）** |
| `服务器环境说明.md` | 这台机器的硬件、软件、目录、常用命令 |
| `智能体接口规范.md` | 三条线之间的接口契约（改接口要走其中 §6 变更流程） |
| `智能体设计方案.md` | B 线设计方案全文（含危险信号清单与红线话术） |
| `分工文档/` | A / B / C 三条线各自的任务书与验收标准 |
| `agent/` | B 线成果：LangGraph 智能体（入口 `agent/agent.py`，命令行 `agent/cli.py`） |
| `agent/prompts/` | 提示词 v1 与变更记录 `CHANGELOG.md` |
| `tests/` | 83 项离线测试（不需要模型、不需要联网） |
| `rag_med_project/` | A 线成果：检索模块 `retrieval.py` + Chroma 库 |
| `评测集/` | C 线成果：150 题诊断测评集（对应 L0–L5），详见其中 `README.md` |

## 三、三条线现在到哪了

- **A 知识库与检索**：检索代码已跑通；知识库只有测试文档，真实语料尚未入库。
- **B 智能体与提示词**：一期代码完成，83 项离线测试全绿；还没接真实模型。
- **C 模型服务与评测**：测评集 v2 已完成；模型服务未开始 —— **这是当前的主任务**。

## 四、工作约定

1. 改代码前先看 `交接文档.md` 的「下一步」和「已知坑」。
2. 动 `agent/` 或 `tests/` 之后必须 `python -m pytest -q` 全绿（当前 83 项），不绿不算完成。
3. 动 `agent/prompts/` 下任何文件：先更新 `agent/prompts/CHANGELOG.md`，再跑 `tests/test_scenarios_fake_model.py` 回归。
4. 改接口先改 `智能体接口规范.md`，按其中 §6 记录变更。
5. 与用户对话、写文档、写注释一律用中文；文档风格照 `项目综述.md`：短句、表格、少形容词。
6. 数据来源要注明：测评集出自 CMExam / CMB-Clin（Apache-2.0）与 IMCS-V2（MIT）。
7. DeepSeek API Key 在 `~/.codex/config.toml` 里，不要复制进项目文件、不要提交、不要在日志里打印。

## 五、这台机器

- 昇腾 910B2C 单卡，64GB 显存，`npu-smi info` 看状态
- openEuler 24.03 LTS-SP2，CANN 9.0.0，Python 3.11.15
- 工作目录 `/data/openpangu`，大文件一律放 `/data`（系统盘只有 30G）
- 跑测试：`cd /data/openpangu && python -m pytest -q`
- 本机 Codex 接的是 DeepSeek：`model = "deepseek-flash"`，跑 `codex` 即用

## 六、不要做的事

- 不要碰 `systemctl`、不要 `reboot`、不要把 `/` 写满
- 不要用 Docker / podman 起服务：容器里没有，也装不上
- 不要把模型权重放 `/root`：17.6GB，必须放 `/data`
- 不要自己另选部署版本组合：CANN 9.0.0 有明确配套版本，见 `交接文档.md`
- `git commit` 之前先问用户