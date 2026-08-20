"""LegalBenchRAG 评测执行器（仅 Searcher 链路，去掉确定性混合检索）。

每条 query 都跑完整 Searcher（多轮工具调用收集完整条款），指标全部基于
Searcher 找回的证据对 gold 的覆盖：
- 文档级（本系统扩展）：agent_doc_precision / agent_doc_recall ——
  Searcher 证据命中的相关文档占比（找齐/找准的文档层信号）；
- 字符级（对齐 gold 区间）：agent_span_precision / agent_span_recall / agent_span_f1 ——
  证据对 gold 字符区间的重叠（官方 PAKTON 口径的区间数学）。
每条 query 跑完立即写入 records.jsonl，支持断点续跑。
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from ..store import EvidenceStore
from .adapter import LegalBenchAdapter
from .persist import append_record
from . import metrics as M
from .schemas import LegalChunkHit, LegalQueryRecord

__all__ = ["run_legalbench", "summarize_legalbench_records"]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_legalbench(
    queries: list[dict],
    *,
    root: str | Path,
    benchmark: str,
    session_id: str,
    records_path: str | Path,
    make_searcher,
    doc_ids: list[str] | None = None,
) -> None:
    """执行一个 benchmark 的评测（仅 Searcher 链路），逐条写入 records.jsonl。

    make_searcher: 必填。可调用，返回一个已装配的 Searcher 实例——Searcher 是唯一检索链路。

    指标全部来自 Searcher 证据：
    - agent_doc_precision：证据文档中属于 gold 相关文档的比例（找准，文档层）；
    - agent_doc_recall：gold 相关文档被证据覆盖的比例（找齐，文档层）；
    - agent_span_{precision,recall,f1}：证据对 gold 字符区间的重叠（官方 PAKTON 区间口径）。
    """
    adapter = LegalBenchAdapter(root, benchmark, session_id, doc_ids)

    for rec in queries:
        iid = str(rec.get("instance_id", ""))
        gold_docs = list(rec.get("gold_docs", []))
        gold_spans = {fp: [list(x) for x in spans] for fp, spans in rec.get("gold_spans", {}).items()}
        query = str(rec.get("query", ""))

        record = LegalQueryRecord(
            instance_id=iid,
            benchmark=benchmark,
            query=query,
            gold_docs=gold_docs,
            gold_spans=gold_spans,
        )

        searcher = make_searcher()
        store = EvidenceStore()
        task = adapter.to_subtask(rec)
        ctx = adapter.make_context(store)
        record.prompt = adapter.task_prompt(rec)  # 留存 Searcher 模式的 prompt
        t_start = time.time()
        result = None
        try:
            result = asyncio.run(searcher.run(task, ctx))
        except Exception as e:  # noqa: BLE001
            record.searcher_error = f"{type(e).__name__}: {e}"

        record.elapsed_s = time.time() - t_start
        worker = getattr(result, "output", None)
        record.searcher_searched = bool(worker and worker.searched)
        record.raw_response = json_safe_trajectory(result, worker)
        if result is not None:
            record.searcher_trajectory = result.trajectory or []
            record.telemetry.update({
                "ran_agent": True,
                "status": str(result.status.value),
                "token_usage": result.token_usage,
                "n_trajectory": len(result.trajectory or []),
                "search_queries": list(getattr(worker, "search_queries", []) or []),
                "evidence_ids": [ev.evidence_id for ev in (getattr(worker, "evidences", []) or [])],
                "no_evidence_found": bool(getattr(worker, "no_evidence_found", False)),
                "drop_reasons": dict(getattr(worker, "drop_reasons", {}) or {}),
            })
        if worker is not None:
            hits = adapter.evidence_hits(worker.evidences)
            record.searcher_hits = [
                LegalChunkHit(rank=h.rank, doc_id=h.doc_id, file_path=h.file_path,
                              span=h.span, score=h.score)
                for h in hits
            ]
            ev_files = sorted({h.file_path for h in record.searcher_hits if h.file_path})
            gold_set = set(gold_docs)
            record.scores["agent_doc_precision"] = (
                len(set(ev_files) & gold_set) / len(ev_files) if ev_files else 0.0
            )
            record.scores["agent_doc_recall"] = (
                len(set(ev_files) & gold_set) / len(gold_set) if gold_set else 0.0
            )
            asp = M.span_precision_recall_f1(adapter.spans_by_file(hits), gold_spans)
            hit_stats = M.evidence_hit_rate(
                [(h.file_path, h.span) for h in record.searcher_hits], gold_spans
            )
            hit_count = int(hit_stats["hit_count"])
            returned_count = int(hit_stats["returned_count"])
            record.scores.update({
                "agent_span_precision": asp["precision"],
                "agent_span_recall": asp["recall"],
                "agent_span_f1": asp["f1"],
                "evidence_hit_rate": hit_count / returned_count if returned_count else 0.0,
            })
            record.telemetry.update({
                "evidence_returned_count": returned_count,
                "evidence_hit_count": hit_count,
                "materialize_failed_count": int(
                    getattr(worker, "materialize_failed_count", 0)
                    or (getattr(worker, "drop_reasons", {}) or {}).get("materialize-fail", 0)
                ),
                "verifier_rejected_count": int(getattr(worker, "verifier_rejected_count", 0) or 0),
                "verified_evidence_count": int(getattr(worker, "verified_evidence_count", 0) or 0),
                "candidate_count": int(getattr(worker, "candidate_count", 0) or 0),
                "search_tool_call_count": int(getattr(worker, "search_tool_call_count", 0) or 0),
            })
        else:
            record.searcher_error = record.searcher_error or "Searcher returned no result"

        append_record(records_path, record.model_dump(mode="json"))


def json_safe_trajectory(result, worker) -> str:
    """把 trajectory 折叠成可读文本（对话层：全量逐字，不截断）。

    结构化的完整轨迹另存于 LegalQueryRecord.searcher_trajectory。
    """
    if result is None or not getattr(result, "trajectory", None):
        return ""
    out: list[str] = []
    for step in result.trajectory:
        role = step.get("role")
        if role == "assistant":
            content = str(step.get("content", "")).replace("\n", " ")
            calls = step.get("tool_calls") or []
            if calls:
                parts = []
                for c in calls:
                    fn = c.get("function", {}) if isinstance(c, dict) else {}
                    parts.append(f"{fn.get('name')}({fn.get('arguments')})")
                out.append(f"assistant(calls={', '.join(parts)}) {content}".strip())
            else:
                out.append(f"assistant {content}".strip())
        elif role == "tool":
            name = step.get("name", "")
            res = str(step.get("result", "")).replace("\n", " ")
            out.append(f"tool[{name}] {res}")
    return "\n".join(out)


def summarize_legalbench_records(records_path: str | Path) -> dict:
    """从已持久化的 records.jsonl 计算一个 benchmark 的聚合计分（仅 Searcher 指标）。

    返回 {n_queries, n_agent, n_searcher_error, metrics(各指标在子任务内的均值)}。
    官方口径的"子任务等权 0.25"加权在 main.py 侧按 present benchmark 归一化后完成。
    """
    import json as _json

    from .persist import load_done_ids

    keys = ["agent_doc_precision", "agent_doc_recall",
            "agent_span_precision", "agent_span_recall", "agent_span_f1",
            "evidence_hit_rate"]
    stats: dict[str, list[float]] = {k: [] for k in keys}
    n_agent = 0
    n_searcher_error = 0
    evidence_returned = 0
    evidence_hit = 0
    candidate_count = 0
    verified_evidence = 0
    verifier_rejected = 0
    materialize_failed = 0
    search_tool_calls = 0
    drop_reasons_total: dict[str, int] = {}
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = _json.loads(line)
            scores = rec.get("scores", {})
            for key in stats:
                v = scores.get(key)
                if v is not None:
                    stats[key].append(float(v))
            if rec.get("telemetry", {}).get("ran_agent"):
                n_agent += 1
            tel = rec.get("telemetry") or {}
            evidence_returned += int(tel.get("evidence_returned_count", 0) or 0)
            evidence_hit += int(tel.get("evidence_hit_count", 0) or 0)
            candidate_count += int(tel.get("candidate_count", 0) or 0)
            verified_evidence += int(tel.get("verified_evidence_count", 0) or 0)
            verifier_rejected += int(tel.get("verifier_rejected_count", 0) or 0)
            materialize_failed += int(tel.get("materialize_failed_count", 0) or 0)
            search_tool_calls += int(tel.get("search_tool_call_count", 0) or 0)
            for reason, count in (tel.get("drop_reasons") or {}).items():
                try:
                    drop_reasons_total[str(reason)] = drop_reasons_total.get(str(reason), 0) + int(count or 0)
                except (TypeError, ValueError):
                    continue
            if rec.get("searcher_error"):
                n_searcher_error += 1
    n_queries = len(load_done_ids(records_path))
    return {
        "n_queries": n_queries,
        "n_agent": n_agent,
        "n_searcher_error": n_searcher_error,
        "metrics": {key: _mean(vals) for key, vals in stats.items() if vals},
        "evidence_returned_count": evidence_returned,
        "evidence_hit_count": evidence_hit,
        "evidence_hit_rate_micro": evidence_hit / evidence_returned if evidence_returned else 0.0,
        "candidate_count": candidate_count,
        "verified_evidence_count": verified_evidence,
        "verifier_rejected_count": verifier_rejected,
        "materialize_failed_count": materialize_failed,
        "search_tool_call_count": search_tool_calls,
        "drop_reasons": drop_reasons_total,
        "all": {key: vals for key, vals in stats.items()},
        "key_weights": {key: len(vals) for key, vals in stats.items()},
    }
