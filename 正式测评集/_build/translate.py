#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把英文题转成中文，结果按原文哈希缓存，重跑不再调接口。

接口信息从 ~/.codex/config.toml 的 [model_providers.deepseek] 读取，
密钥只在内存里用，不写进任何项目文件、不打进日志。
"""

import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = Path(__file__).resolve().parent / "translation_cache.json"
CONFIG_PATH = Path.home() / ".codex" / "config.toml"

SYSTEM_PROMPT = (
    "你是医学翻译。把输入 JSON 数组里的每一项翻译成简体中文。要求："
    "医学术语用中国大陆临床通用译法；药物名、器械名、检查名用通用中文名；"
    "数字、单位、化验值、英文缩写（如 CT、MRI、BMI、eGFR、ICU）保留原样；"
    "已经是中文的项原样返回；不要增删内容，不要加解释；"
    "只输出 JSON 数组，元素个数与输入一致，不要输出 markdown 代码块。"
)


def _load_provider():
    cfg = CONFIG_PATH.read_text(encoding="utf-8")
    block = re.search(r"\[model_providers\.deepseek\](.*?)(?=\n\[|\Z)", cfg, re.S)
    if not block:
        raise RuntimeError("config.toml 里没有 [model_providers.deepseek]")
    base = re.search(r'base_url\s*=\s*"([^"]+)"', block.group(1))
    token = re.search(r'experimental_bearer_token\s*=\s*"([^"]+)"', block.group(1))
    if not (base and token):
        raise RuntimeError("deepseek provider 缺少 base_url 或 token")
    return base.group(1).rstrip("/"), token.group(1)


class Translator:
    def __init__(self, model: str = "deepseek-chat", workers: int = 8):
        self.base, self.token = _load_provider()
        self.model = model
        self.workers = workers
        self.cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
        self.calls = 0

    # ---------------------------------------------------------------- 缓存

    @staticmethod
    def _key(text: str) -> str:
        return "zh:" + hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _cached(self, text: str):
        return self.cache.get(self._key(text))

    def save(self):
        CACHE_PATH.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=0, sort_keys=True),
            encoding="utf-8")

    # ---------------------------------------------------------------- 调用

    def _call(self, texts: list, retries: int = 4) -> list:
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(texts, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": 8192,
        }).encode("utf-8")
        payload = json.dumps(texts, ensure_ascii=False)
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(
                    self.base + "/chat/completions", data=body,
                    headers={"Content-Type": "application/json",
                             "Authorization": "Bearer " + self.token})
                with urllib.request.urlopen(req, timeout=300) as resp:
                    data = json.load(resp)
                self.calls += 1
                content = data["choices"][0]["message"]["content"].strip()
                content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.M).strip()
                out = json.loads(content)
                if not isinstance(out, list) or len(out) != len(texts):
                    raise ValueError(f"返回条数 {len(out) if isinstance(out, list) else '?'} != {len(texts)}")
                return [str(x) for x in out]
            except Exception as exc:  # noqa: BLE001
                if attempt == retries:
                    raise
                wait = 2 * attempt
                print(f"    重试 {attempt}/{retries - 1}：{type(exc).__name__} {str(exc)[:80]}")
                time.sleep(wait)
        return list(texts)

    # ---------------------------------------------------------------- 对外

    def to_zh_batch(self, texts: list) -> list:
        """按顺序翻译一批文本，命中缓存的直接返回。"""
        todo, out = [], [None] * len(texts)
        for i, t in enumerate(texts):
            hit = self._cached(t)
            if hit is not None:
                out[i] = hit
            else:
                todo.append((i, t))
        if not todo:
            return out

        # 按体积切块，避免单次请求过大
        chunks, cur, cur_len = [], [], 0
        for item in todo:
            ln = len(item[1])
            if cur and cur_len + ln > 4000:
                chunks.append(cur)
                cur, cur_len = [], 0
            cur.append(item)
            cur_len += ln
        if cur:
            chunks.append(cur)

        for chunk in chunks:
            translated = self._call([t for _, t in chunk])
            for (i, src), zh in zip(chunk, translated):
                self.cache[self._key(src)] = zh
                out[i] = zh
        return out

    def to_zh(self, text: str) -> str:
        return self.to_zh_batch([text])[0]
