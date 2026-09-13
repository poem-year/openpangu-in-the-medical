# -*- coding: utf-8 -*-
import os
import json
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# =====================配置区=====================
DOC_FOLDER = r"./knowledge_base/docs"
PERSIST_DIR = r"./knowledge_base/chroma_db"
EMBED_MODEL_NAME = "BAAI/bge-m3"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# 全局加载嵌入模型，避免每次检索重复加载
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": "cpu"}
)


def build_knowledge_base():
    """
    【离线建库函数，仅新增/修改文档时运行一次】
    读取docs下全部txt，文本切分，生成唯一chunk_id，存入Chroma持久化向量库
    """
    all_texts = []
    all_metadatas = []

    for fname in os.listdir(DOC_FOLDER):
        if not fname.endswith(".txt"):
            continue
        file_path = os.path.join(DOC_FOLDER, fname)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_text(content)

        for idx, chunk in enumerate(chunks):
            short_id = str(uuid4())[:8]
            cid = f"{fname.replace('.txt','')}_{short_id}"
            meta = {
                "chunk_id": cid,
                "source": fname,
                "section": "未划分章节"
            }
            all_texts.append(chunk)
            all_metadatas.append(meta)

    vector_store = Chroma.from_texts(
        texts=all_texts,
        metadatas=all_metadatas,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print(f"✅知识库构建完成，总切片数量：{len(all_texts)}")
    return vector_store


def load_vector_store():
    """加载本地持久化向量库"""
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    return db


def retrieve(query: str, k: int = 5):
    """
    A模块对外接口，供B模块导入调用
    :param query: 用户提问字符串
    :param k: 返回最多k条证据
    :return: dict，标准返回结构
    """
    # 兜底：空输入校验
    if not query or len(query.strip()) == 0:
        return {
            "query": query,
            "evidence": [],
            "msg": "query不能为空"
        }

    try:
        db = load_vector_store()
        docs_with_score = db.similarity_search_with_score(query, k=k)
    except Exception as e:
        return {
            "query": query,
            "evidence": [],
            "msg": f"数据库读取异常：{str(e)}"
        }

       # BGE-M3：score越小越相似，设定阈值0.8，高于等于0.8视为无关，直接丢弃
    SIMILAR_THRESHOLD = 0.8
    evidence = []
    for doc, score in docs_with_score:
        if score < SIMILAR_THRESHOLD:
            evidence.append({
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "text": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "section": doc.metadata.get("section", "未划分章节"),
                "score": round(float(score), 4)
            })


    if len(evidence) > 0:
        msg = "success"
    else:
        msg = "未检索到相关资料"

    return {
        "query": query,
        "evidence": evidence,
        "msg": msg
    }


if __name__ == "__main__":
    # ============【建库：新增文档才打开下面一行，用完立刻注释！】============
    # build_knowledge_base()

    # 测试用例1：正常医学问题
    res1 = retrieve("高血压的诊断标准是什么？", k=2)
    print("====测试1：正常提问====")
    print(json.dumps(res1, ensure_ascii=False, indent=2))

    # 测试用例2：知识库没有的无关问题
    res2 = retrieve("今天天气怎么样", k=2)
    print("\n====测试2：无相关资料====")
    print(json.dumps(res2, ensure_ascii=False, indent=2))

    # 测试用例3：空输入
    res3 = retrieve("", k=2)
    print("\n====测试3：空输入====")
    print(json.dumps(res3, ensure_ascii=False, indent=2))
