"""ContractNLI 评测执行器：复用 Planner 做前提-假设分类。

两种 mode（对齐 PAKTON 的两种口径）：
- indexed（默认，对齐 PAKTON 的 multi-agent 检索口径）：先把合同整库入库
  （ingest_contractnli_jsonl），每条 (premise, hypothesis) 通过 DocumentToolkit 在
  该合同的索引内按 hypothesis 检索出相关条款，把“条款原文+偏移”送进分类 prompt，
  再调用 Planner 归类。整份前提不进上下文，省 token 且更贴近真实用法。
- direct（等价 PAKTON 的 naive zero-shot baseline）：整段前提直接进 prompt。

每条实例：prompt → 模型输出 → 标签解析 → gold → 得分 → 遥测，即时写 records.jsonl，
以 instance_id 支持断点续跑 / 抽样复现。
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
    mode: str = "indexed",
    nli_session: str = "nli-contractnli",
    top_k: int = 8,
) -> dict:
    """执行 ContractNLI 分类评测。返回 {evaluated, skipped_done, errors} 统计。"""
    done = done_ids or set()
    if limit:
        rng = random.Random(seed)
        pending = [r for r in records if str(r.get("instance_id", "")) not in done]
        pending = rng.sample(pending, min(limit, len(pending)))
        selected = {str(r.get("instance_id", "")) for r in pending}
    else:
        selected = None

    toolkit = None
    if mode == "indexed":
        from ..tools import DocumentToolkit

        toolkit = DocumentToolkit(session_id=nli_session, doc_ids=None)

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

        record = NliRecord(
            instance_id=iid,
            premise_id=str(rec.get("premise_id", "")),
            subset=str(rec.get("subset", "")),
            premise_preview=premise[:200],
            hypothesis=hypothesis,
            gold_label=gold,
            mode=mode,
        )
        t0 = time.time()

        if mode == "indexed":
            doc_id = nli_doc_id_raw(record.premise_id)
            record.doc_id = doc_id
            toolkit.set_scope(nli_session, [doc_id])
            try:
                rows = toolkit.search(hypothesis, mode="hybrid", top_k=top_k)
            except Exception as e:  # noqa: BLE001
                rows = []
                record.error = f"{type(e).__name__}: {e}"
            record.retrieved_n = len(rows)
            record.retrieved_chunks = [
                {
                    "id": r.get("id", ""),
                    "text_preview": str(r.get("text", ""))[:120],
                    "span": list(r.get("charspan") or []),
                    "score": r.get("rrf_score") or r.get("bm25_score"),
                }
                for r in rows
            ]
            prompt = ContractNLIAdapter.build_retrieval_prompt(rows, hypothesis)
        else:
            prompt = ContractNLIAdapter.build_prompt(premise, hypothesis)

        record.prompt = prompt
        try:
            data = planner.solve(prompt, system_prompt=ContractNLIAdapter.system_prompt())
            if data is None:
                record.pred_valid = False
                record.error = record.error or "no JSON object in response"
            else:
                label, reasoning = ContractNLIAdapter.parse_data(data)
                record.pred_label = label
                record.reasoning = reasoning
                record.pred_valid = label is not None
                if label is None:
                    record.error = record.error or "unparseable label"
                record.raw_response = str(data)[:400]  # dict 摘要
        except Exception as e:  # noqa: BLE001  # PlanParseError 等
            record.pred_valid = False
            record.error = f"{type(e).__name__}: {e}"

        record.elapsed_s = time.time() - t0
        record.correct = (record.pred_label == gold)
        if mode == "indexed":
            record.telemetry = {"doc_found": _doc_exists(toolkit, nli_session, record.doc_id)}
        if not record.pred_valid:
            errors += 1
        evaluated += 1
        append_record(records_path, record.model_dump(mode="json"))

    return {"evaluated": evaluated, "skipped_done": skipped_done, "errors": errors}


def _doc_exists(toolkit, session_id: str, doc_id: str) -> bool:
    """检查某合同是否已在 session 中入库（只读 SELECT）。"""
    try:
        return bool(toolkit.get_document(doc_id))
    except Exception:  # noqa: BLE001
        return False


def nli_doc_id_raw(premise_id: str) -> str:
    """与 ingest_raw.nli_doc_id 保持一致的 doc_id 生成（避免循环导入）。"""
    return f"nli:{premise_id}"


def summarize_contractnli_records(records_path: str | Path) -> dict:
    """从已持久化的 records.jsonl 计算 Accuracy / weighted F1 / per-class F1。"""
    import json as _json

    y_true: list[str] = []
    y_pred: list[str] = []
    n_total = n_errors = 0
    n_indexed_retrieved_zero = 0
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = _json.loads(line)
            n_total += 1
            if rec.get("mode") == "indexed" and rec.get("retrieved_n", 0) == 0:
                n_indexed_retrieved_zero += 1
            if not rec.get("pred_valid"):
                n_errors += 1
                continue
            y_true.append(rec["gold_label"])
            y_pred.append(rec["pred_label"])
    acc_all = M.accuracy(y_true, y_pred) if y_true else 0.0
    return {
        "n_total": n_total,
        "n_errors": n_errors,
        "n_indexed_retrieved_zero": n_indexed_retrieved_zero,
        "accuracy": acc_all,
        "f1_weighted": M.f1_weighted(y_true, y_pred) if y_true else 0.0,
        "f1_per_class": M.f1_per_class(y_true, y_pred),
        "class_counts_true": {c: y_true.count(c) for c in sorted(set(y_true))},
        "class_counts_pred": {c: y_pred.count(c) for c in sorted(set(y_pred))},
    }
