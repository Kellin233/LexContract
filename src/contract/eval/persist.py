"""评测持久化：records.jsonl（逐条全量过程）+ summary.json + metrics.csv + 断点续跑。

- 每条实例写一行 JSONL 并即时 flush，进程中断不丢已跑结果；
- 以稳定的 instance_id 记录已完成集合，下次运行跳过（断点续跑）。
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "make_run_dir",
    "append_record",
    "load_done_ids",
    "write_json",
    "write_csv",
    "now_iso",
    "parse_iso",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_iso(s: str) -> float:
    if not s:
        return time.time()
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return time.time()


def make_run_dir(out_root: str | Path, mode: str) -> Path:
    """创建 <out_root>/<mode>/<ts>/ 并返回（幂等）。"""
    run_dir = Path(out_root) / mode / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_record(records_path: str | Path, record: dict) -> None:
    """追加一条实例记录到 JSONL 并立即 flush。"""
    with open(records_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()


def load_done_ids(records_path: str | Path) -> set[str]:
    """读取已完成的 instance_id 集合（用于断点续跑）。"""
    path = Path(records_path)
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = rec.get("instance_id")
            if iid is not None:
                done.add(str(iid))
    return done


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def write_csv(path: str | Path, rows: list[dict]) -> None:
    """由 dict 列表写 CSV（列以首行键为准，缺失列填空）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{}]
    columns = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in columns})
