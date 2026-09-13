# C · 模型服务与评测

## 一、在方案里的位置

链路：用户提问 → 检索 → 生成 → 工具 → 安全校验 → 输出。

这部分做两件事。

一是把 openPangu 跑起来，给 A 和 B 一个统一的调用接口，他们不用管模型怎么启动的。

二是评测。出题、跑分、算指标，拿出来基础模型和增强方案的对比数据。系统能不能跑是第一件事决定的，能不能证明有效是第二件事决定的。

---

## 二、接口

模型接口，A 和 B 都调这个：

```
llm(消息列表, temperature=0.1, max_tokens=1024) → 文本
```

做成 OpenAI 兼容的 HTTP 接口最省事，三个人的代码都能直接接。temperature 和随机种子要固定，不然两次跑出来的分数对不上，实验就没意义了。

评测结果记在这张表里，每次实验加一行。这张表最后就是评估报告的原始材料。

| 编号 | 日期 | 层级 | 版本(RAG/提示词/模型) | 题量 | 准确率 | 引用可追溯率 | 幻觉率 | 安全合规率 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

---

## 三、技术路线

**模型服务**：把 openPangu 部署成一个能长期跑着的接口。按手头硬件挑框架就行：

- 昇腾环境用 MindIE，官方配套的
- 有英伟达卡就用 vLLM，文档全，社区也活跃
- 只想先把流程跑通，用 Ollama 最快

基础模型用 openPangu-7B。显存不够或者要做对比的时候换 openPangu-1B。

**评测**：用同一套题测不同阶段的系统，做前后对比。

| 层级 | 加了什么 |
| --- | --- |
| L0 | 裸模型直接问 |
| L1 | 加提示词 |
| L2 | 加 RAG |
| L3 | 换更好的检索 |
| L4 | 加工具调用 |
| L5 | 加安全规则，也就是最终方案 |

L0 得最先拿到。没有基线，后面的提升说不清楚。

指标看四个：

| 指标 | 怎么算 |
| --- | --- |
| 准确率 | 答到主要要点的题数 ÷ 总题数 |
| 引用可追溯率 | 有正确出处标注的结论数 ÷ 总结论数 |
| 幻觉率 | 编造来源或数据的回答数 ÷ 总回答数 |
| 安全合规率 | 红线题处理对的题数 ÷ 红线题总数 |

再找人打分，1 到 5 分，至少两个人分别打。

---

## 四、工具和参考资料

| 用途 | 工具 | 资料 |
| --- | --- | --- |
| 部署（英伟达卡） | vLLM | [文档](https://docs.vllm.ai/) |
| 部署（快速验证） | Ollama | [官网](https://ollama.com/) |
| 部署（CPU、本地） | llama.cpp | [仓库](https://github.com/ggml-org/llama.cpp) |
| 下载 openPangu | AtomGit AI 社区 | [链接](https://ai.gitcode.com/ascend-tribe) |
| 演示界面 | Streamlit | [官网](https://streamlit.io/) |
| 备选界面 | Gradio | [官网](https://www.gradio.app/) |
| RAG 效果评测 | RAGAS | [文档](https://docs.ragas.io/) |
| 显著性检验 | scipy | [官网](https://scipy.org/) |
| 下载模型和数据集 | HuggingFace、ModelScope | [HuggingFace](https://huggingface.co/datasets) · [ModelScope](https://modelscope.cn/datasets) |

---

## 五、评测数据来源

| 数据集 | 用途 |
| --- | --- |
| [C-MTEB/CmedqaRetrieval](https://huggingface.co/datasets/C-MTEB/CmedqaRetrieval) | 中文医学检索评测 |
| [cMedQA-V2.0](https://huggingface.co/datasets/wangrongsheng/cMedQA-V2.0) | 中文医学问答，可以挑一些当参考题 |
| [PromptCBLUE](https://huggingface.co/datasets/tchenglv/PromptCBLUE) | 中文医疗 NLP 多任务 |
| [medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa) | 英文医学选择题，翻译了能用 |

---

## 六、完成标准

- 同一个问题两次答案一样，参数是固定的
- L0 基线分最先拿到，L0 到 L5 六层都有数
- 每次实验记一行，版本号写清楚
- 演示流程提前走一遍，准备录屏兜底
