# kb · 知识库与检索模块（A 线）

给智能体提供「带出处的资料片段」。检索实现、建库、索引、评测辅助都在本目录，
智能体侧只通过 `agent/retrieval.py` 的适配层调用。

## 一、对外接口

| 层 | 位置 | 契约 |
| --- | --- | --- |
| 智能体工具（模型可见） | `agent/retrieval.py::retrieve_evidence` | 工具名固定，返回带 chunk_id 的文本 |
| 检索适配层 | `agent/retrieval.py::retrieve` | `retrieve(query, k) -> list[Evidence]`（见《智能体接口规范.md》§2） |
| 检索实现 | `kb/search.py::retrieve` | `retrieve(query, k, *, strategy, threshold) -> list[RetrievedChunk]` |

`kb` 不依赖 `agent`（字段一一对应，但用自有 dataclass），所以评测脚本可以单独调它。

## 二、日常命令

```bash
cd /data/openpangu

# 1. 建库（增量：只重嵌新增/改过的语料；加 --rebuild 全量重建）
.venv/bin/python -m kb.build

# 2. 查一条，看 top-k、余弦、BM25 名次、融合分与来源分布
.venv/bin/python -m kb.inspect "高血压患者每天吃盐不能超过多少"
.venv/bin/python -m kb.inspect "高血压" -k 3 --strategy vector --threshold 0.0

# 3. 生成检索评测集候选（走 DeepSeek），人工筛过后存成 kb/eval/queries.jsonl
.venv/bin/python -m kb.eval_set

# 4. 标定阈值：扫一遍阈值网格，看 Recall@5 与无关问题误召率
.venv/bin/python -m kb.calibrate --queries kb/eval/queries.jsonl
```

评测（L2/L3）在项目根目录跑：

```bash
bash run_eval.sh --layer L2          # 纯向量检索（基线知识库）
bash run_eval.sh --layer L3          # 混合检索 + 标定阈值
```

`run_eval.py` 会自动先跑预检索阶段（用本目录的 `.venv`），把证据写进
`eval/runs/<run-id>/contexts.jsonl`，再交给 `.venv-pangu` 的生成阶段使用。

## 三、语料怎么放

1. 把 md / txt / pdf 放进 `kb/corpus/`（可以建子目录）；
2. 在 `kb/corpus/sources.yaml` 的 `files:` 下逐份登记 `file / doc_id / name / version /
   category / source_type`；
3. 跑 `python -m kb.build`。

### 3.1 从原始语料生成（`kb/corpus_raw/` → `kb/corpus/`）

`kb/corpus_raw/` 是采集层（4.5 GB 的 html/xml/jsonl/docx/pdf/zip），**不能直接入库**：
格式与结构都不满足上面的要求。转换脚本会自动做四件事（格式转换、内容清洗、
条款结构化、许可登记），产出 `kb/corpus/*.md` 与新的 `sources.yaml`：

```bash
cd /data/openpangu
.venv/bin/python kb/corpus_raw/_scripts/convert.py --count-chunks   # 转换 + 预估块数
```

筛选口径（`convert.py` 里的 `is_medical` / `_NOISE_TITLE` / `_TITLE_TOPIC`）：
标题必须有临床主题词，文件名/正文开头命中招聘、申请表、专业目录、产业政策等模式的一律剔除；
被剔掉的每一份都写进 `kb/corpus/conversion_report.json`（含原因），不静默跳过。
分级与许可清单见 `kb/corpus_raw/语料分级与许可.md`（对外发布前按它过滤）。

**收录范围**：`tier=1` 国家政策与地方规范（政府文件）、`tier=2` 开放获取文献（Europe PMC CC BY）、
`tier=3` 中文医学百科问答（shibing624/medical，Apache-2.0，按医学关键词筛选后去重）。
**不收录**测评题池（CMB / CMExam / MedQA / MedMCQA / MMLU / HealthBench / AgentClinic /
CmedqaRetrieval / MedCalc-Bench）——正式测评集就从这些池子抽题，入 KB 会造成评测泄漏；
问答与对话类语料（医患对话、DISC-Med-SFT）也不是可引用的事实来源，暂不入库。

### 3.2 两段式建库（CPU 切块 + NPU 嵌入）

本机 128 核 CPU 跑 BGE-M3 只有约 5～6 块/秒，8 万块要 4 小时以上；昇腾 910B 上
同一模型约 100 块/秒（受并发服务影响），差 20 倍。`.venv-pangu` 里没有
`langchain_text_splitters`（切块要用），所以分成两步跑：

```bash
# 1) 切块（agent 环境，秒级）
.venv/bin/python -m kb.build --dump-chunks /tmp/kb_chunks.jsonl

# 2) 嵌入落盘（NPU；必须先 source pangu 的环境变量）
source scripts/pangu_env.sh
KB_DEVICE=npu KB_EMBED_BATCH_SIZE=64 \
  .venv-pangu/bin/python -m kb.build --load-chunks /tmp/kb_chunks.jsonl --rebuild
```

NPU 后端在 `kb/embed.py::NpuBackend`（`KB_DEVICE=npu` 时自动启用）。它取 **[CLS] 位**
做句向量，与查询侧 `BgeM3Backend`（sentence-transformers，BGE-M3 的官方池化）口径一致——
实测同一句话两边余弦 0.99999；曾用 mean pooling 时只有 0.79，会让召回明显变差。
查询侧仍走 CPU，不依赖 NPU。

规则与约束：

- **出处只来自登记表与文档标题层级**，不从正文猜。`section` 是标题路径，
  如「心血管内科 / 高血压 / 诊断」。
- 没登记的文件、登记了却找不到的文件、解析为空的文件都会**报错并计入
  `build_report.json`**，不静默跳过。
- 只支持文本型 PDF；扫描件/图片型 PDF 解析为空会明确报错（本模块不做 OCR）。

## 四、切块与 chunk_id

- 有 Markdown 标题：按标题层级切，标题路径就是 `section`；
- 没有标题但有 `【条目名】`：按条目切，条目名就是 `section`；
- 都没有：按段落聚合到 500 字以内；
- 仍然超过 600 字的块，用 `RecursiveCharacterTextSplitter`（600/80）二次切分。
- `chunk_id = kb::<doc_id>::<内容哈希前12位>`：同一段文本永远同一 ID，改一个字符 ID 就变，
  这样「引用能指回原文」这件事是可验证的（审查层会用检索返回的 chunk_id 做防伪）。

## 五、检索流程与分数

```
查询 → 归一化嵌入 → 向量 top-N ┐
                              ├→ RRF 融合排序 → 余弦阈值过滤 → 同文档最多 N 条 → 取 k
       → 中文分词 BM25 top-N  ┘
```

- `strategy=hybrid`（默认）：向量 + BM25 融合，分数是 RRF 融合分（只保证同一查询内可比、降序）；
- `strategy=vector`：纯向量，分数就是余弦相似度（L2 基线用这个）；
- 阈值一律作用在**向量余弦**上（RRF 分数不可解释，只用于排序）；
- 无关问题若没有候选过阈值就返回 `[]`，工具层会如实说「未查到资料」——这是有意设计的，
  宁可说查不到，也不要拿不相干的片段当依据；
- 索引缺失或为空会抛 `KbIndexMissingError`，工具层降级为「检索服务暂不可用」，
  与「查不到资料」区分开，避免掩盖配置问题。

BM25 走中文**双字**分词（`zh-bigram-v2`，英文数字按整词）。刻意去掉了单字：
单字会把「病」「症」「炎」这种几乎每篇都有的字当特征，实测查「川崎病」时
BM25 第一名是「狂犬病」（只共享一个「病」字）。注意：语料太小时（几篇文档）
BM25 的 IDF 会退化成 0 或负数，那一支基本不起作用，此时以向量检索为主。

**延迟口径**：实测热态单次检索 P50 约 210ms、P95 约 300ms（查询嵌入占大头），
远低于接口规范要求的 2 秒。但**首次调用是冷启动**：要加载 BGE-M3（约 10–14 秒），
万级语料下第一次混合检索还要现场做 BM25 分词（秒级）。规范里的 P95 指热态稳态，
如果你要压首问延迟，可以在起服务后先空跑一次检索预热。

## 六、配置（环境变量）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `KB_CORPUS_DIR` | `kb/corpus` | 语料目录 |
| `KB_INDEX_DIR` | `kb/index` | 索引目录 |
| `KB_EMBED_MODEL` | `BAAI/bge-m3` | 嵌入模型（本机已缓存） |
| `KB_DEVICE` | `cpu` | 嵌入设备；建库时可设 `npu` 用昇腾加速（查询仍走 CPU） |
| `KB_TOP_K` | 5 | 默认返回条数 |
| `KB_SCORE_THRESHOLD` | 0.55 | 向量余弦阈值。2026-09-14 在 110 条定稿评测集上标定：0.55 是唯一同时满足「误召率 ≤10%」与高召回的点（详见 §八）；**换语料后必须重跑 `kb.calibrate`** |
| `KB_STRATEGY` | `hybrid` | `hybrid` / `vector` |
| `KB_RRF_K` | 60 | RRF 平滑常数 |
| `KB_DOC_QUOTA` | 2 | 同一文档最多贡献几条 |
| `KB_CANDIDATE_POOL` | 50 | 每路召回候选数（不小于 top_k） |
| `KB_EMBED_BATCH_SIZE` | 32 | 建库批大小 |

## 七、索引里有什么

当前索引（2026-09-13 重建）：**84,359 块 / 125 个文档 / 1024 维**，
`vectors.npy` 346MB + `chunks.jsonl` 132MB，全量重建约 20 分钟（NPU 嵌入）。

`kb/index/` 下：`vectors.npy`（L2 归一化后的向量）、`chunks.jsonl`（行号与向量对应）、
`bm25.json`（分词器版本）、`manifest.json`（语料 sha1 指纹 + 建库配置）、
`build_report.json`（每次建库的明细）。写入走「临时目录 → 整体替换」，
不会让 agent 读到写了一半的索引。

`chunks.jsonl` 每条记录带 `license / language / tier` 三个来源字段（来自 `sources.yaml`，
缺省为空串），对外发布前按 `license` 过滤即可，不必重新切块。

增量建库靠 `manifest.json` 里的文件指纹：文件没变就复用旧向量，变了只重嵌该文件，
文件删了就连带删掉它的向量（`removed_chunks`）。换嵌入模型会强制全量重建。

## 八、评测集与阈值

`kb/eval/queries.jsonl` 是定稿的检索评测集（110 条：相关 90 / 无关 20），每行一条：

```json
{"id": "Q0001", "question": "高血压每天吃盐上限是多少？", "type": "relevant", "expected_chunk_ids": ["kb::cardio::a1b2c3d4e5f6"]}
{"id": "N0001", "question": "今天天气怎么样", "type": "irrelevant"}
```

标准答案支持 `expected_chunk_ids` / `expected_sections` / `expected_doc_ids` 三种粒度，
按精细度自动选优先级；**建议用 chunk 级**——doc 级 gold 下一个科室（十几个病种）
命中任意一条就算召回，测不出排序好坏。chunk_id 用 `python -m kb.inspect "病名"` 查。

`kb.calibrate` 会用 `relevant` 题算 Recall@5、用 `irrelevant` 题算误召率，
建议在误召率不超过上限（默认 10%）的前提下取召回最高的阈值。

### 8.1 这一版评测集怎么来的

```bash
.venv/bin/python -m kb.eval_set --limit 45 --max-per-doc 1 --seed 20260914 --irrelevant 20
# 人工筛一遍候选，另存为 kb/eval/queries.jsonl（reviewed 全部标 true）
.venv/bin/python -m kb.calibrate --queries kb/eval/queries.jsonl
```

- 相关题由 DeepSeek 从**真实切块**反向写出来（问题写在资料之外、不许抄原句），
  gold 是那块切块自己的 `chunk_id`——不是 doc 级。doc 级不能当 gold：`baike-09`
  一卷就有 1753 块，命中任意一块都算召回，等于不测排序。
- 取样是**跨文档分层**的（固定种子 20260914，`--max-per-doc 1`）：直接取索引前
  40 条会全落在第 1 卷百科上，标出来的阈值只反映「百科 vs 百科」。
- 已知偏差：问题是从语料写出来的，用词天然贴近原文，所以这个分数**偏乐观**，
  它比的是「排序对不对」，不是「语料里有没有这个知识」。换成真人提问会更低。

### 8.2 标定结果（2026-09-14，84,359 块，k=5）

| 阈值 | Recall@5 | 误召率 | 平均返回条数 |
| --- | --- | --- | --- |
| 不过滤 | 0.611 | 1.000 | 5.00 |
| 0.50 | 0.611 | 0.200 | 4.19 |
| **0.55** | **0.600** | **0.050** | 3.85 |
| 0.60 | 0.556 | 0.000 | 3.07 |

**结论：默认阈值 0.55 保持不变**（误召率达标前提下召回最高）。0.45 及以下会把
80%~100% 的无关问题也召回，等于没有阈值。

按粒度与语种拆开看（阈值 0.55）：

| 口径 | 结果 |
| --- | --- |
| chunk 级 Recall@5 | 0.600（54/90） |
| **文档级 Recall@5** | **0.778（70/90）** |
| 误召率 | 0.050（1/20） |
| 中文卷（百科 / 政府文件） | 0.797（51/64） |
| **英文卷（Europe PMC CC BY）** | **0.115（3/26）** |

两点判断：

1. **chunk 级 0.60 是偏严的口径**。语料里有 48 卷百科，同一个病在多个卷里都写了，
   检索常把「同主题的另一卷」排在前面——文档级 0.778 与 chunk 级 0.60 的差
   （16 条）基本都是这种情况：查得到这个病，只是不是当初写题的那一块。
2. **真正的短板是跨语言。** 26 道题的 gold 落在英文文献上，只命中 3 道。中文问
   英文材料本来就是 BGE-M3 该做的事，排在后面主要是被 2845 万字的百科语料挤掉的
   （实测部分题 gold 余弦 0.41~0.53，低于阈值）。**要么给英文卷开跨语言检索通道
   （查询翻译成英文后单独检索再融合），要么承认英文卷只作背景、不进评测口径。**
   这条是下一步要决策的事，别当成已经解决的问题。
