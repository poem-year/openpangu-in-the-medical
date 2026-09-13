# -*- coding: utf-8 -*-
"""实时进度看板：在另一个终端里盯着评测跑到哪了。

用法：
    /data/openpangu/.venv-pangu/bin/python /data/openpangu/eval/watch.py --run-id L0-slow-b32
    # 或者指定目录：--run-dir /data/openpangu/eval/runs/L0-slow-b32

每 2 秒刷新一次：当前阶段、生成进度条、实时准确率、分维度进度、AI 判分进度、最近完成题目。
Ctrl-C 退出看板不影响评测本身。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from progress import read_progress, render_dashboard  # noqa: E402
from progress import render_compact  # noqa: E402


def render_multi(run_dirs: list[str], run_ids: list[str], interval: float, once: bool) -> None:
    """多 run 看板：先列每个 run 的状态，再展开当前活跃 run 的详细进度。"""
    import time

    while True:
        states = [(rid, d, read_progress(d)) for rid, d in zip(run_ids, run_dirs)]
        # 活跃 run：第一个还没跑完的；都跑完就显示最后一个
        active = None
        for rid, d, st in states:
            if st.get("stage") not in ("done", "failed"):
                active = (rid, d)
                break
        if active is None:
            active = (states[-1][0], states[-1][1])

        lines = ["═" * 84, " 一键评测（两种思考模式）实时进度", "═" * 84, ""]
        for rid, d, _ in states:
            lines.append(render_compact(d, rid))
        lines.append("")
        lines.append(f"───── 当前：{active[0]} ─────")
        lines.append(render_dashboard(active[1], active[0]))
        text = "\n".join(lines)

        if once:
            print(text)
            return
        sys.stdout.write("\033[2J\033[H" + text + "\n")
        sys.stdout.flush()
        if all(st.get("stage") in ("done", "failed") for _, _, st in states):
            break
        time.sleep(interval)
    print("\n两个模式都跑完了，报告见各自的 summary.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-ids", nargs="*", default=None, help="同时盯多个 run（两种思考模式）")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="只打印一次（便于脚本调用）")
    args = parser.parse_args()

    if args.run_ids:
        run_dirs = [os.path.join("/data/openpangu/eval/runs", rid) for rid in args.run_ids]
        render_multi(run_dirs, args.run_ids, args.interval, args.once)
        return

    run_dir = args.run_dir
    if run_dir is None:
        if not args.run_id:
            parser.error("需要 --run-id 或 --run-dir")
        run_dir = os.path.join("/data/openpangu/eval/runs", args.run_id)
    if not os.path.isdir(run_dir):
        print(f"找不到 run 目录：{run_dir}", file=sys.stderr)
        sys.exit(1)

    run_id = args.run_id or os.path.basename(run_dir.rstrip("/"))
    if args.once:
        print(render_dashboard(run_dir, run_id))
        return

    first = read_progress(run_dir)
    if first.get("stage") in ("done", "failed"):
        finished = time.strftime("%H:%M:%S", time.localtime(first.get("updated_at", time.time())))
        print(f"提示：run「{run_id}」早在 {finished} 就已结束，这不是刚跑完的评测。")
        print("      下面是它的最终结果；要跑新的评测请执行：bash /data/openpangu/run_eval.sh --watch")
        print()
        print(render_dashboard(run_dir, run_id))
        return

    try:
        while True:
            text = render_dashboard(run_dir, run_id)
            sys.stdout.write("\033[2J\033[H" + text + "\n")
            sys.stdout.flush()
            state = read_progress(run_dir)
            if state.get("stage") in ("done", "failed"):
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n（已退出看板，评测仍在后台继续）")
        return
    print("\n评测阶段已结束，报告见 summary.md")


if __name__ == "__main__":
    main()
