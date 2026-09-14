# 医学诊断语料库（原始层）

> **2026-09-13 整理后现状**：本目录从 4.5 GB / 429 文件精简到 **647 MB**。
> 已删除「评估后决定不入库」的原始件（openFDA 药品说明书 1.4G、医患对话 1.4G、
> MedRAG 教材与文献 366M、CMB/CMExam 等题库 430M、Orphadata/MedlinePlus 155M 等），
> 原因是不进本次知识库（题库会造成评测泄漏、对话与药品说明书语言不符/体量过大）。
> 保留下来的只有本次真正用到的三类：政府文件、Europe PMC 全文、中文医学百科，
> 以及采集脚本与 `sources.json` 台账。需要重新采回被删部分：
> `python _scripts/download.py --category <类别>`（脚本仍在，台账里仍有 SHA256）。
> 分级与许可见 `语料分级与许可.md`。

> 采集日期：2026-09-13 ｜ 用途：A 线知识库的原始语料，供后续结构化对齐
> 状态：**只做采集与登记，不做清洗/切分/入库**；结构化对齐是下一步

## 一、这份语料从哪来

按测评集 13 个维度反推需要的能力，再去找对应的公开来源，分四类：

| 类别 | 对应测评维度 | 来源 |
| --- | --- | --- |
| 中文医学数据集 | D01 D02 D05 D06 | CMB、CMExam、PromptCBLUE(IMCS-V2)、中文医患对话、DISC-Med-SFT、shibing624/medical |
| 模型社区（魔搭） | D01 D02 D05 D12 | BAAI 行业语料医学分册、医学考试题库、医疗咨询 SFT、中医语料、CBLUE-KUAKE |
| 权威文件与指南 | D03 D04 D09 D10 | 中国政府网政策文件库、省级卫健委、中国疾控中心、Orphadata（罕见病） |
| 开放获取文献 | D03 D08 D13 | Europe PMC（只取 CC BY 全文） |
| 药品与患者语言 | D09 D11 D13 | openFDA 药品说明书、MedlinePlus 健康主题、中文维基百科医学条目 |
| 公式与量表 | D12 | MedCalc-Bench |

## 二、目录结构

```
corpus/
├── README.md            本文件
├── sources.json         来源登记表（每条：来源 / 许可 / URL / 本地路径 / 大小 / sha256）
├── _scripts/            采集脚本（可重复运行，已下载会跳过）
├── hf_datasets/         模型社区数据集（HuggingFace / hf-mirror）
│   └── modelscope/      魔搭（ModelScope）数据集
├── guidelines_cn/       中文政府文件与指南（gov.cn、省卫健委、疾控中心）
├── guidelines_intl/     国际开放指南与条目（Orphadata、MedlinePlus）
├── literature_oa/       开放获取文献全文（Europe PMC，CC BY）
├── drug_labels/         药品说明书（openFDA）
├── encyclopedia/        百科类语料（中文维基条目 + 数据集里的医疗百科）
└── formulas/            临床计算公式与量表（预留）
```

## 三、许可与使用边界（重要）

| 许可类型 | 语料 | 能否对外发布 |
| --- | --- | --- |
| Apache-2.0 / MIT | CMB、CMExam、PromptCBLUE、DISC-Med-SFT、HealthBench、PubMedQA 等 | 可以，注明出处 |
| CC-BY-4.0 | Meditron 指南库、Orphadata、部分欧洲文献 | 可以，必须署名 |
| CC-BY-SA-4.0 | MedCalc-Bench；中文维基条目为 CC BY-SA 4.0 | 可以，署名 + 相同方式共享 |
| 政府文件 | 中国政府网、省卫健委、疾控中心、openFDA（美国政府作品） | 可以，注明来源与文号 |
| 混合许可 | MedlinePlus | 逐条核对后再用 |
| **未声明许可** | MedQA、AgentClinic、CmedqaRetrieval、MedRAG 教材与 PubMed 摘要 | **仅内部研究，不得对外分发** |

> 对外提交作品前，务必用 `sources.json` 里的 `license` 字段过一遍，
> 把"仅内部研究"的语料排除在公开仓库之外。

## 四、怎么用（复现采集）

```bash
cd rag_med_project/knowledge_base/corpus
python _scripts/sources_hf.py           # 登记 HuggingFace 数据集
python _scripts/sources_modelscope.py   # 登记魔搭（ModelScope）数据集
python _scripts/sources_gov_cn.py       # 登记中国政府网文件
python _scripts/sources_province.py     # 登记省卫健委 / 疾控中心
python _scripts/sources_orphadata.py    # 登记 Orphadata 罕见病数据
python _scripts/sources_medlineplus.py  # 登记 MedlinePlus 健康主题
python _scripts/sources_openfda.py      # 登记 openFDA 药品说明书
python _scripts/sources_europepmc.py    # 登记 Europe PMC CC BY 全文
python _scripts/sources_wikipedia.py    # 采集中文维基医学条目（有速率限制）
python _scripts/download.py             # 按登记表下载（断点续跑）
python _scripts/extract_attachments.py  # 从政府正文里提取 PDF/Word 附件
python _scripts/filter_cn_docs.py --apply  # 剔除政府网检索混进来的非医学文件
python _scripts/validate.py             # 体检：缺失 / 空文件 / 疑似错误页
python _scripts/report.py               # 看总量统计
```

## 五、已知限制

1. **国家卫健委官网（nhc.gov.cn）反爬**：直接请求返回 412，本轮改用中国政府网文件库与省级卫健委站点替代。
2. **WHO IRIS 批量接口不可用**：检索 API 返回 403、旧域名不通；WHO 指南内容由 Meditron 指南库覆盖一部分，后续再补。
3. **中文维基有速率限制**：批量抓取被 429 限流（只成功少量条目）；百科层改由数据集里的医疗百科语料承担。
4. **政府网检索有噪声**：gov.cn 接口是宽松匹配，已用 `filter_cn_docs.py` 把非医学文件移到 `guidelines_cn/_excluded/`。
5. **未做结构化**：本目录只有原始文件与来源登记表；切片、字段对齐、chunk_id、入库属下一步工作。
6. **原始文件不入 git**：GB 级文件已加入 `.gitignore`，跨机器用本目录脚本重新采集。
7. **下一批可补**：台湾卫福部开放资料、香港卫生防护中心中文健康主题、国家卫健委官网（需绕反爬）、WHO 指南（IRIS 接口受限）。
