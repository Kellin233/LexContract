#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_repl.py
================================================================================
LexContract 合同证据链交互式 REPL 单会话脚本。

功能：
  1. 在单个进程内连续提问，共享同一个 Orchestrator
  2. 使用 session_id 限定当前合同文档作用域
  3. 输入 q/quit/exit 退出，Ctrl+C 优雅中断

Usage:
    python scripts/run_repl.py [--config path/to/config.yaml]
================================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.runner import initialize_modules, load_config, run_research, save_report, setup_logging


def print_help() -> None:
    print("""
可用命令:
  <任意问题>   执行深度研究
  save        保存上一条报告到文件
  help        显示此帮助
  q / quit / exit  退出 REPL
""")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LexContract 合同证据链交互式 REPL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--session_id", type=str, default=None, help="指定当前 session_id（默认自动生成）")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger("repl")

    config = load_config(args.config)

    # ------------------------------------------------------------------
    # Session 设置（不再读取或写入历史 session）
    # ------------------------------------------------------------------
    if args.session_id:
        session_id = args.session_id
        print(f"[REPL] 已指定 session: {session_id}")
    else:
        print("=" * 50)
        print("DeepResearch Agent 交互式 REPL")
        print("=" * 50)
        session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"\n[REPL] 当前 session: {session_id}")

    # ------------------------------------------------------------------
    # 初始化模块（只初始化一次，整个 REPL 生命周期复用）
    # ------------------------------------------------------------------
    print("[REPL] 正在初始化模块...")
    modules = initialize_modules(config, session_id=session_id)
    print(f"[REPL] 模块初始化完成，输入 'help' 查看命令，'q' 退出\n")

    last_report = None

    # ------------------------------------------------------------------
    # REPL 循环
    # ------------------------------------------------------------------
    while True:
        try:
            query = input(f"[{session_id}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[REPL] 收到中断信号，退出...")
            break

        if not query:
            continue

        cmd = query.lower()

        if cmd in ("q", "quit", "exit"):
            break
        elif cmd == "help":
            print_help()
            continue
        elif cmd == "save":
            if last_report:
                md_path, json_path = save_report(last_report, "repl_report", "outputs/reports")
                print(f"  报告已保存: {md_path} / {json_path}")
            else:
                print("  暂无报告可保存")
            continue

        # ------------------------------------------------------------------
        # 执行深度研究
        # ------------------------------------------------------------------
        print(f"[REPL] 正在研究: {query[:60]}...")
        start = time.time()
        try:
            report = asyncio.run(run_research(query, config, modules, session_id=session_id))
            elapsed = time.time() - start
            last_report = report

            print(f"\n  ✓ 报告完成 | {len(report.content)} 字 | 置信度 {report.confidence:.2f} | "
                  f"搜索 {report.num_searches} 轮 | 耗时 {elapsed:.1f}s")
            print("  输入 'save' 保存报告到文件\n")

        except Exception as e:
            logger.exception("研究执行失败")
            print(f"\n  ✗ 执行失败: {e}\n")

    # ------------------------------------------------------------------
    # 退出清理
    # ------------------------------------------------------------------
    print(f"\n[REPL] Session '{session_id}' 已结束；本次运行不会写入记忆数据库。")
    print("[REPL] 再见！")


if __name__ == "__main__":
    main()
