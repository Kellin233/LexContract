"""ContractNLI 评测执行器：复用 Planner（solve 通用方法）做前提-假设分类。

- 每条 (premise, hypothesis) 经 ContractNLIAdapter 包装为分类 prompt，
  调用 Planner（真正走 LLM 与稳健 JSON 解析），解析出标签；
- 标签不可解析 / LLM 调用失败 → 记为错误样例（pred_label=None）；
- 每跑完一条立即写入 records.jsonl，支持断点续跑 / 抽样复现。
"""
from __future__ import annotations

import random
import time
from pathlib import Path

from .adapter import ContractNLIAdapter
from .persist import append_record
from . import metrics as M
from .schemas import NliRecord

__all__ = ["run_contractnli", "summarize_contractnli_records"]


def run_contractnli(
    records: list[dict],
    *,
    planner,
    records_path: str | Path,
    limit: int | None = None,
    seed: int = 0,
    done_ids: set[str] | None = None,
) -> dict:
    """执行 ContractNLI 分类评测。返回 {evaluated, skipped_done, errors, samples} 统计。"""
    done = done_ids or set()
    if limit:
        rng = random.Random(seed)
        pending = [r for r in records if str(r.get("instance_id", "")) not in done]
        pending = rng.sample(pending, min(limit, len(pending)))
        selected = {str(r.get("instance_id", "")) for r in pending}
    else:
        selected = None

    evaluated = errors = skipped_done = 0
    for rec in records:
        iid = str(rec.get("instance_id", ""))
        if iid in done:
            skipped_done += 1
            continue
        if selected is not None and iid not in selected:
            continue

        premise = str(rec.get("premise", ""))
        hypothesis = str(rec.get("hypothesis", ""))
        gold = str(rec.get("label", ""))
        prompt = ContractNLIAdapter.build_prompt(premise, hypothesis)

        record = NliRecord(
            instance_id=iid,
            premise_id=str(rec.get("premise_id", "")),
            premise_preview=premise[:200],
            hypothesis=hypothesis,
            gold_label=gold,
            prompt=prompt,
        )
        t0 = time.time()
        try:
            data = planner.solve(prompt, system_prompt=ContractNLIAdapter.system_prompt())
            if data is None:
                record.pred_valid = False
                record.error = "no JSON object in response"
            else:
                label, reasoning = ContractNLIAdapter.parse_data(data)
                record.pred_label = label
                record.reasoning = reasoning
                record.pred_valid = label is not None
                if label is None:
                    record.error = "unparseable label"
                record.raw_response = str(data)[:400]  # dict 摘要
        except Exception as e:  # noqa: BLE001  # PlanParseError 等
            record.pred_valid = False
            record.error = f"{type(e).__name__}: {e}"

        record.elapsed_s = time.time() - t0
        record.correct = (record.pred_label == gold)
        if not record.pred_valid:
            errors += 1
        evaluated += 1
        append_record(records_path, record.model_dump(mode="json"))

    return {"evaluated": evaluated, "skipped_done": skipped_done, "errors": errors}


def summarize_contractnli_records(records_path: str | Path) -> dict:
    """从已持久化的 records.jsonl 计算 Accuracy / weighted F1 / per-class F1。"""
    import json as _json

    y_true: list[str] = []
    y_pred: list[str] = []
    n_total = n_errors = 0
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = _json.loads(line)
            n_total += 1
            if not rec.get("pred_valid"):
                n_errors += 1
                continue
            y_true.append(rec["gold_label"])
            y_pred.append(rec["pred_label"])
    acc_all = M.accuracy(y_true, y_pred) if y_true else 0.0
    return {
        "n_total": n_total,
        "n_errors": n_errors,
        "accuracy": acc_all,
        "f1_weighted": M.f1_weighted(y_true, y_pred) if y_true else 0.0,
        "f1_per_class": M.f1_per_class(y_true, y_pred),
        "class_counts_true": {c: y_true.count(c) for c in sorted(set(y_true))},
        "class_counts_pred": {c: y_pred.count(c) for c in sorted(set(y_pred))},
    }
