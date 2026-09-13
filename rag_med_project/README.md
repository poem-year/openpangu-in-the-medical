A 模块：RAG 检索模块
功能：医学知识库检索，输入问题，返回知识库召回片段 JSON，供 B 模块使用

项目结构
rag_med_project
├─ retrieval.py          # 主程序
├─ README.md
└─ knowledge_base
   ├─ docs               # 原始医学文档 (.txt)
   └─ chroma_db          # 向量数据库（自动生成）

依赖安装
pip install langchain langchain-community sentence-transformers chromadb

使用方式
1. 知识库更新：把 txt 文档放入 knowledge_base/docs，打开 retrieval.py，取消注释 build_knowledge_base()，运行一次；建库完成后重新注释。
2. 检索调用：
from retrieval import retrieve
result = retrieve(query="你的问题", k=2) 

返回 JSON 结构：
{
  "query": "输入的问题",
  "evidence": [
    {
      "chunk_id": 切片ID,
      "text": 召回文本,
      "source": 来源文件名,
      "section": 章节,
      "score": 相似度分数
    }
  ],
  "msg": success / 未检索到相关资料 / query不能为空
}

重要相似度说明
模型BGE-M3：score数值越小，代表相似度越高。
A模块内置阈值0.8，score≥0.8 的检索片段会直接过滤，不会返回给B模块。

msg状态含义：
- success：检索到符合要求的相关资料
- 未检索到相关资料：没有满足相似度阈值的内容
- query不能为空：输入为空字符串