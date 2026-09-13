# B · 智能体与提示词

## 一、在方案里的位置

链路：用户提问 → 检索 → 生成 → 工具 → 安全校验 → 输出。

这部分管后半段。拿到 A 检索出的资料，组织成提示词交给模型，需要算数或查规则时调工具，高风险问题转走。

输入是用户问题加 A 返回的资料，输出是回答正文、引用出处和风险等级。

---

## 二、接口

`answer(问题, 历史对话)` 返回：

```json
{
  "answer": "根据资料，二甲双胍常见副作用包括……",
  "citations": [
    {"chunk_id": "metformin_leaflet_c42_001", "source": "盐酸二甲双胍片说明书", "section": "4.2 不良反应"}
  ],
  "risk_level": "normal",
  "used_tools": [],
  "refused": false
}
```

`citations` 里的 chunk_id 得是 A 返回的，不能自己编。`risk_level` 填 normal、caution 或 urgent。

---

## 三、技术路线

走 Agent，让模型按情况决定要不要查资料、要不要算一下、要不要转走，而不是一条道走到底。

要做三件事：

**提示词**：写清角色、任务、参考资料、输出格式、引用要求、安全规则，再附两三个例子。

**工具调用**：需要算数或查规则的时候调现成函数，别让模型心算。

**安全通道**：下面三类问题不给具体方案，改成提示就医。

| 情况 | 例子 | 怎么处理 |
| --- | --- | --- |
| 要求下诊断 | 我这是不是胃癌 | 不做判断，说清楚要做什么检查、挂哪个科 |
| 要求开药或给药量 | 我该吃多少毫克 | 不给方案，建议遵医嘱或咨询药师 |
| 急症描述 | 胸口剧痛还冒冷汗 | 直接建议打 120 或者去急诊 |

规则依据是《互联网诊疗管理办法（试行）》，还有说明书上的禁忌和注意事项。

---

## 四、工具和参考资料

| 用途 | 工具 | 资料 |
| --- | --- | --- |
| 智能体编排 | LangGraph | [文档](https://docs.langchain.com/oss/python/langgraph/overview) · [仓库](https://github.com/langchain-ai/langgraph) |
| 备选框架 | LangChain | [文档](https://python.langchain.com/) · [仓库](https://github.com/langchain-ai/langchain) |
| 提示词写法 | 提示工程指南（中文） | [promptingguide.ai/zh](https://www.promptingguide.ai/zh) |
| 演示界面 | Streamlit | [官网](https://streamlit.io/) |
| 备选界面 | Gradio | [官网](https://www.gradio.app/) |
| 下载问答样例 | HuggingFace、ModelScope | [HuggingFace](https://huggingface.co/datasets) · [ModelScope](https://modelscope.cn/datasets) |

工具挑两三个做就够，都是简单的计算和查表：

| 工具 | 输入输出 |
| --- | --- |
| BMI 计算器 | 身高体重，算出 BMI 和分级 |
| 单位换算 | 血糖、肌酐、血脂的 mg/dL 和 mmol/L 互转 |
| 常用量表 | PHQ-9、GAD-7、疼痛评分 |
| 用药提示 | 输入药名，返回说明书里的禁忌项 |

写成普通函数就行，出错返回提示，别让流程挂掉。

---

## 五、示例数据来源

| 数据集 | 用途 |
| --- | --- |
| [HuatuoGPT-sft-data-v1](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1) | 中文医学问答，挑几条做例子 |
| [huatuo_consultation_qa](https://huggingface.co/datasets/FreedomIntelligence/huatuo_consultation_qa) | 问诊对话，看看真人怎么提问 |
| [PromptCBLUE](https://huggingface.co/datasets/tchenglv/PromptCBLUE) | 中文医疗任务集，有问诊和诊断子任务 |

例子只用来定格式，不用让模型背内容。

---

## 六、完成标准

- 同一个问题连问两次，输出的结构一样
- 每条依据都能对上 A 返回的 chunk_id
- 三类红线问题都能拦住，不给具体诊疗方案
- A 返回空的时候老实说没找到资料，不编
- 提示词改了就存新版本号，旧版留着
