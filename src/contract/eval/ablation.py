#!/usr/bin/env python3
"""ContractNLI 编排消融入口。

同一批请求依次运行两条链路：
  direct: Planner → Searcher → Refiner
  reviewed_incremental: Planner → Searcher → Reviewer → 增量补查 → Refiner

本模块只负责组织两臂运行和生成对比文件，实际单臂执行复用 eval.main。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .main import main as eval_main

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

__all__ = ["main"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ContractNLI Multi-Agent 编排消融")
    p.add_argument(
        "--request-set",
        default=str(_PROJECT_ROOT / "configs/eval_sets/contractnli_15.json"),
        help="固定请求集合，默认按比例抽取的 ContractNLI 15 条（150 条的 10%）",
    )
    p.add_argument("--contractnli-jsonl", default=None)
    p.add_argument("--nli-session", default=None)
    p.add_argument("--nli-concurrency", type=int, default=None)
    p.add_argument("--searcher-max-rounds", type=int, default=None)
    p.add_argument("--searcher-max-searches-per-round", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--out", default="evaluation/runs/orchestration_ablation")
    p.add_argument("--ingest-nli", action="store_true")
    p.add_argument("--no-ingest", action="store_true")
    return p.parse_args(argv)


def _run_arm(args: argparse.Namespace, mode: str, arm_dir: Path) -> dict:
    arm_dir.mkdir(parents=True, exist_ok=True)
    cli = [
        "--mode", "contractnli",
        "--request-set", args.request_set,
        "--out", str(arm_dir),
        "--orchestration-mode", mode,
    ]
    for flag, value in (
        ("--contractnli-jsonl", args.contractnli_jsonl),
        ("--nli-session", args.nli_session),
        ("--config", args.config),
    ):
        if value:
            cli.extend([flag, str(value)])
    for flag, value in (
        ("--nli-concurrency", args.nli_concurrency),
        ("--searcher-max-rounds", args.searcher_max_rounds),
        ("--searcher-max-searches-per-round", args.searcher_max_searches_per_round),
        ("--seed", args.seed),
    ):
        if value is not None:
            cli.extend([flag, str(value)])
    if args.ingest_nli:
        cli.append("--ingest-nli")
    if args.no_ingest:
        cli.append("--no-ingest")

    rc = eval_main(cli)
    if rc != 0:
        raise RuntimeError(f"编排消融 arm={mode} 执行失败，返回码={rc}")
    summaries = sorted(arm_dir.glob("contractnli/*/summary.json"), key=lambda p: p.stat().st_mtime)
    if not summaries:
        raise FileNotFoundError(f"arm={mode} 未找到 summary.json")
    summary_path = summaries[-1]
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _flat_scalars(summary: dict) -> dict[str, float | int | None]:
    """展开两臂对比需要的标量指标，忽略嵌套分类字典。"""
    out: dict[str, float | int | None] = {}
    for key, value in (summary.get("metrics") or {}).items():
        if isinstance(value, (int, float)) or value is None:
            out[key] = value
        elif isinstance(value, dict):
            if key == "f1_per_class":
                prefix = "f1"
            elif key in {"drop_reasons", "stop_reason_counts"}:
                prefix = key
            else:
                continue
            for name, score in value.items():
                if isinstance(score, (int, float)):
                    out[f"{prefix}_{name}"] = score
    return out


def _write_comparison(root: Path, arms: dict[str, dict]) -> None:
    direct = _flat_scalars(arms["direct"])
    reviewed = _flat_scalars(arms["reviewed_incremental"])
    metrics = sorted(set(direct) | set(reviewed))
    delta = {}
    rows = []
    for name in metrics:
        d, r = direct.get(name), reviewed.get(name)
        change = r - d if isinstance(r, (int, float)) and isinstance(d, (int, float)) else None
        delta[name] = change
        rows.append({"metric": name, "direct": d, "reviewed_incremental": r, "delta_reviewed_minus_direct": change})

    comparison = {
        "mode": "contractnli_orchestration_ablation",
        "arms": arms,
        "delta_reviewed_minus_direct": delta,
        "output_dir": str(root),
    }
    (root / "summary.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    import csv

    with (root / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["metric"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    root = Path(args.out) / timestamp
    root.mkdir(parents=True, exist_ok=True)
    arms = {}
    for mode in ("direct", "reviewed_incremental"):
        print(f"\n=== [orchestration ablation] {mode} ===")
        arms[mode] = _run_arm(args, mode, root / mode)
    _write_comparison(root, arms)
    print(f"[orchestration ablation] 对比结果: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
