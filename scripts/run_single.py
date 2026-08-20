#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_single.py
================================================================================
LexTrace 合同证据链：单条问题运行脚本。

Usage:
    python scripts/run_single.py --query "供应商能否单方面终止合同？" --session S1 [--doc doc_a,doc_b] [--config path]
输出: outputs/reports/report_*.md（可读）+ report_*.json（结构化）
================================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

# 加入项目根目录，保证 `python scripts/run_single.py` 可直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.runner import initialize_modules, load_config, run_research, save_report, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LexTrace 合同证据链：单条问题运行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/run_single.py --query "供应商能否单方面终止合同？" --session S1
  python scripts/run_single.py --query "不可抗力情况下的责任如何分配" --session S1 --doc doc_a,doc_b
  python scripts/run_single.py --query "..." --config configs/custom.yaml
        """,
    )
    parser.add_argument("--query", type=str, required=True, help="用户问题（必填）")
    parser.add_argument("--session", type=str, default="", help="检索会话（工作区）ID，必填才能检索")
    parser.add_argument("--doc", type=str, default="", help="限定检索的文档 ID，逗号分隔；空=会话内全部")
    parser.add_argument("--config", type=str, default=None, help="自定义配置文件路径")
    parser.add_argument("--output_dir", type=str, default="outputs/reports", help="报告输出目录")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # 终端输出同时写入日志文件（无条件 tee，与原工程一致）
    import datetime as _dt
    os.makedirs(args.output_dir, exist_ok=True)
    log_filename = os.path.join(args.output_dir, f"run_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    class _Tee:
        """同时输出到终端和文件。"""
        def __init__(self, terminal, file):
            self.terminal, self.file = terminal, file
        def write(self, message):
            self.terminal.write(message)
            self.file.write(message)
            self.file.flush()
        def flush(self):
            self.terminal.flush()
            self.file.flush()
        def isatty(self):
            return True

    log_file = open(log_filename, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    print(f"[日志] 终端输出已同时保存到: {log_filename}")

    setup_logging(args.log_level)
    logger = logging.getLogger("main")

    doc_ids = [d.strip() for d in args.doc.split(",") if d.strip()] if args.doc else []

    try:
        config = load_config(args.config)
        logger.info(f"配置加载完成: {args.config or 'configs/default.yaml'}")
        modules = initialize_modules(config, session_id=args.session)

        import time as _time
        from src.core.runner import format_report_markdown

        _start = _time.time()
        report = asyncio.run(run_research(args.query, config, modules,
                                          session_id=args.session, doc_ids=doc_ids,
                                          output_dir=args.output_dir))
        elapsed = _time.time() - _start

        md_text = format_report_markdown(report, elapsed)
        md_path, json_path = save_report(report, args.query, args.output_dir)
        logger.info(f"报告已保存: {md_path} / {json_path}")

        print("\n" + "=" * 60)
        print("最终报告")
        print("=" * 60)
        print(md_text)
        print("=" * 60)
    except Exception:
        logger.exception("运行过程中发生错误")
        sys.exit(1)


if __name__ == "__main__":
    main()
