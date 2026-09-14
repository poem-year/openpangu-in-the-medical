# -*- coding: utf-8 -*-
"""端到端冒烟：服务在跑的前提下，用 B 线智能体跑两轮真实对话。

跑法（用 .venv，不是 .venv-pangu）：
    bash scripts/serve_pangu.sh start
    .venv/bin/python scripts/smoke_agent_server.py
    .venv/bin/python scripts/smoke_agent_server.py --base-url http://127.0.0.1:8000/v1

两轮约 30~60 秒（模型单条解码约 37 tok/s）。任何一轮走成 `review_modified` 兜底
都算失败——那说明服务端的结构化输出没被 B 线接受。
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request


def check_health(base_url: str) -> bool:
    health = base_url.rstrip("/").removesuffix("/v1") + "/health"
    try:
        with urllib.request.urlopen(health, timeout=5) as response:
            return json.loads(response.read()).get("status") == "ok"
    except (urllib.error.URLError, OSError, ValueError):
        return False


def run_turn(question: str, history: list[dict]) -> tuple[dict, float]:
    from agent.agent import answer

    started = time.time()
    result = answer(question, history)
    return result, time.time() - started


def show(result, elapsed: float, title: str) -> None:
    print(f"\n===== {title}（{elapsed:.1f}s）=====")
    print(f"handled_as={result.handled_as}  risk_level={result.risk_level}  used_tools={result.used_tools}")
    print("reply:", result.reply.replace("\n", " ")[:220])
    for item in result.assessment.hypotheses:
        print(f"  - 假设 {item.name} [{item.likelihood}] 支持={item.supporting} 缺={item.missing}")
    print("  missing_info:", result.assessment.missing_info)
    print("  next_steps:", result.assessment.next_steps)
    print("  evidence:", [(e.source, e.ref[:24]) for e in result.assessment.evidence])


def main() -> int:
    parser = argparse.ArgumentParser(description="B 线智能体 × openPangu 服务端到端冒烟")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="openpangu-7b")
    args = parser.parse_args()

    if not check_health(args.base_url):
        print(f"✗ 服务没起来：{args.base_url}（先跑 bash scripts/serve_pangu.sh start）")
        return 2

    os.environ["OPENAI_BASE_URL"] = args.base_url
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "not-needed")
    os.environ["MODEL_NAME"] = args.model

    first = "咳嗽三天了，还有点低热"
    result1, elapsed1 = run_turn(first, [])
    show(result1, elapsed1, "第 1 轮")

    history = [{"role": "user", "content": first}, {"role": "assistant", "content": result1.reply}]
    second = "没有痰，也不胸闷，就是有点乏力"
    result2, elapsed2 = run_turn(second, history)
    show(result2, elapsed2, "第 2 轮")

    print(f"\n合计 {elapsed1 + elapsed2:.1f}s（单轮 {elapsed1:.1f}s / {elapsed2:.1f}s）")
    ok = all(r.handled_as != "review_modified" for r in (result1, result2))
    print("✅ 冒烟通过：两轮都产出了被 B 线接受的结构化输出" if ok
          else "❌ 冒烟失败：有轮次走了兜底，看 logs/pangu_server.log 与 logs/pangu_server/*.jsonl")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
