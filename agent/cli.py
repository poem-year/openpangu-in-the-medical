"""命令行聊天入口（开发期）。

用法：python -m agent.cli
命令：/exit 退出 | /new 新会话 | /debug 切换调试输出 | /json 查看上一条结构化结果 | /help 帮助
"""

from __future__ import annotations

import sys

from agent.agent import answer, build_agent, build_model
from agent.config import get_config
from agent.errors import AgentUnavailableError

BANNER = """医学诊断辅助智能体（开发版）
- 本助手提供初步评估与就医建议，不能替代医生面诊；紧急情况请立即拨打 120。
- 不需要提供姓名、身份证号等身份信息。
- 命令：/exit 退出 | /new 新会话 | /debug 切换调试 | /json 上一条结构化结果 | /help 帮助"""


def _setup_utf8() -> None:
    """Windows 控制台中文乱码处理（PYTHONIOENCODING 之外的兜底）。"""
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    _setup_utf8()
    cfg = get_config()
    if cfg.debug:
        import logging

        logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    model = build_model(cfg)
    agent = build_agent(config=cfg, model=model)

    history: list[dict] = []
    last_result = None
    print(BANNER)
    print(f"（当前模型：{cfg.model_name} @ {cfg.openai_base_url}）\n")

    while True:
        try:
            line = input("您> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return 0
        if not line:
            continue
        if line in ("/exit", "/quit", "退出"):
            print("再见。")
            return 0
        if line == "/new":
            history = []
            last_result = None
            print("（已开始新会话）")
            continue
        if line == "/debug":
            cfg.debug = not cfg.debug
            print(f"（调试模式：{'开' if cfg.debug else '关'}）")
            continue
        if line == "/json":
            if last_result is None:
                print("（暂无结果）")
            else:
                print(last_result.model_dump_json(indent=2))
            continue
        if line == "/help":
            print(BANNER)
            continue

        try:
            result = answer(line, history, config=cfg, model=model, agent=agent)
        except AgentUnavailableError as exc:
            print(f"[服务不可用] {exc}\n请检查模型服务（OPENAI_BASE_URL / MODEL_NAME）后重试。")
            continue
        except ValueError as exc:
            print(f"[输入有误] {exc}")
            continue

        print()
        print(result.reply)
        if cfg.debug:
            print()
            print(
                f"[risk_level={result.risk_level} | handled_as={result.handled_as} | "
                f"used_tools={result.used_tools}]"
            )
            print(result.assessment.model_dump_json(indent=2))
        print()

        history.append({"role": "user", "content": line})
        history.append({"role": "assistant", "content": result.reply})
        last_result = result


if __name__ == "__main__":
    raise SystemExit(main())
