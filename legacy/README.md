# legacy · 已被取代的脚本

这些脚本**还能用**，但不再是正式入口。新实验一律走 `eval/scripts/run.sh`——
只有它会写 `meta.json`、归档报告、追加登记表，并保证不覆盖历史结果。

| 脚本 | 原用途 | 现在用什么 |
| --- | --- | --- |
| `run_eval.sh` | 一键评测（L0 慢/快对比、后台 + 看板） | `bash eval/scripts/run.sh eval/configs/l0-baseline.yml`；想边跑边看用 `bash eval/scripts/watch.sh` |
| `watch.sh` | 看单个 run 的进度 | `bash eval/scripts/watch.sh [run-id 前缀]` |
| `watch_all.sh` | 看多个 run 的总览（NPU/磁盘） | `bash eval/scripts/watch.sh --all` |
| `stop_eval.sh` | 停正在跑的评测 | `bash eval/scripts/stop.sh`（`--all` 连模型服务一起停） |

保留它们的原因：旧文档、旧命令里还在引用；内部逻辑（引擎调用）与现在一致，
差别只在「谁来组织、结果放哪」。

**注意**：这些脚本写出来的 run 目录不会自动归档报告，需要时用
`bash 评测报告/collect.sh <run-id>` 补一份快照。
