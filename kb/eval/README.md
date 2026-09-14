# kb/eval · 检索评测集

这个目录放两样东西：

| 文件 | 是否入库 | 说明 |
| --- | --- | --- |
| `queries_candidates.jsonl` | 不入库（.gitignore） | `kb.eval_set` 生成的候选，未筛 |
| `queries.jsonl` | 入库 | 人工筛过、`kb.calibrate` 实际使用的定稿评测集 |

## 定稿格式

每行一条 JSON：

```json
{"id": "Q0001", "question": "高血压患者每天吃盐不能超过多少？", "type": "relevant", "expected_chunk_ids": ["kb::cardio::a1b2c3d4e5f6"], "reviewed": true}
{"id": "N0001", "question": "今天天气怎么样", "type": "irrelevant", "expected_doc_ids": [], "reviewed": true}
```

- `type`：`relevant` 参与 Recall@5；`irrelevant` 参与误召率统计，用来定阈值；
- 标准答案三种粒度，**按精细度自动选优先级**：`expected_chunk_ids`（最准）→
  `expected_sections` → `expected_doc_ids`（最粗）；
- 强烈建议用 `expected_chunk_ids`：一个 `doc_id` 通常是一整个科室（十几个病种），
  doc 级 gold 下"命中同科室任意一条"就算召回，测不出排序好坏。chunk_id 可以用
  `python -m kb.inspect "病名"` 查出来；
- `reviewed`：人工确认过为 true。

## 定稿流程

```bash
cd /data/openpangu

# 1. 从语料反向生成候选（走 DeepSeek，逐条可失败重跑）
.venv/bin/python -m kb.eval_set

# 2. 人工筛：删掉不合适的、核对 expected_doc_ids，另存为 kb/eval/queries.jsonl

# 3. 标定阈值：看 Recall@5 与误召率，挑一个写回 KB_SCORE_THRESHOLD
.venv/bin/python -m kb.calibrate --queries kb/eval/queries.jsonl
```

标定出来的阈值写进 `kb/config.py` 的默认值或环境变量 `KB_SCORE_THRESHOLD`，
并把这一次的结论记到 `kb/README.md` 或评测报告里，便于复现。
