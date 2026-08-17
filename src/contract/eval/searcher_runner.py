"""LegalBenchRAG 评测执行器（双报：确定性混合排名 + LLM Searcher agent）。

- 确定性部分：全量 query（无 LLM），产出文档级 Recall@k / MRR，
  并附带 top-k 片段的字符区间 P/R/F1（作为参考）；
- Agent 部分：按 --agent-limit 抽样（seed 可复现）跑真实 Searcher，
  产出字符区间 P/R/F1（agent 证据对 gold 区间的覆盖率）；
- 每跑完一条 query 立即写入 records.jsonl，支持断点续跑。
"""
from __future__ import annotations

import asyncio
import random
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
    ks: list[int],
    records_path: str | Path,
    make_searcher=None,
    agent_limit: int | None = None,
    seed: int = 0,
    doc_ids: list[str] | None = None,
) -> None:
    """执行一个 benchmark 的评测，逐条写入 records.jsonl。返回 None（汇总交给 main）。

    make_searcher: 可调用，返回一个已装配的 Searcher 实例。
    agent_limit: 参与 agent 评测的最大 query 数（None=全量，0=不跑 agent）。
    """
    adapter = LegalBenchAdapter(root, benchmark, session_id, doc_ids)
    top_k = max(ks)

    # 参与 agent 评测的 query：在全部未完成 query 里按 seed 采样，保证可复现 + 续跑
    agent_ids = {str(q.get("instance_id", "")) for q in queries}
    if agent_limit:
        rng = random.Random(seed)
        agent_ids = set(rng.sample(sorted(agent_ids), min(agent_limit, len(agent_ids))))
    do_agent = make_searcher is not None

    for rec in queries:
        iid = str(rec.get("instance_id", ""))
        gold_docs = list(rec.get("gold_docs", []))
        gold_spans = {fp: [list(x) for x in spans] for fp, spans in rec.get("gold_spans", {}).items()}
        query = str(rec.get("query", ""))

        ranked = adapter.deterministic_rank(query, top_k=top_k)
        ranked_files = adapter.ranked_files(ranked)
        ranked_by_file = adapter.spans_by_file(ranked)

        record = LegalQueryRecord(
            instance_id=iid,
            benchmark=benchmark,
            query=query,
            gold_docs=gold_docs,
            gold_spans=gold_spans,
            ranked_hits=[
                LegalChunkHit(rank=h.rank, doc_id=h.doc_id, file_path=h.file_path,
                              span=h.span, score=h.score)
                for h in ranked
            ],
        )
        # 文档级：Recall@k + MRR
        for k in ks:
            record.scores[f"recall_at_{k}"] = M.recall_at_k(ranked_files, gold_docs, k)
        record.scores["mrr"] = M.mrr(ranked_files, gold_docs)
        # 字符区间：确定性 top-k 的覆盖率（参考口径）
        rsp = M.span_precision_recall_f1(ranked_by_file, gold_spans)
        record.scores.update({
            "span_precision": rsp["precision"],
            "span_recall": rsp["recall"],
            "span_f1": rsp["f1"],
        })

        # Agent 部分：真实 Searcher（对抽中的 query）
        if do_agent and iid in agent_ids:
            searcher = make_searcher()
            store = EvidenceStore()
            task = adapter.to_subtask(rec)
            ctx = adapter.make_context(store)
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
                record.telemetry.update({
                    "ran_agent": True,
                    "status": str(result.status.value),
                    "token_usage": result.token_usage,
                    "n_trajectory": len(result.trajectory or []),
                    "search_queries": list(getattr(worker, "search_queries", []) or []),
                    "evidence_ids": [ev.evidence_id for ev in (getattr(worker, "evidences", []) or [])],
                    "no_evidence_found": bool(getattr(worker, "no_evidence_found", False)),
                })
            if worker is not None:
                hits = adapter.evidence_hits(worker.evidences)
                record.searcher_hits = [
                    LegalChunkHit(rank=h.rank, doc_id=h.doc_id, file_path=h.file_path,
                                  span=h.span, score=h.score)
                    for h in hits
                ]
                asp = M.span_precision_recall_f1(adapter.spans_by_file(hits), gold_spans)
                record.scores.update({
                    "agent_span_precision": asp["precision"],
                    "agent_span_recall": asp["recall"],
                    "agent_span_f1": asp["f1"],
                })
            else:
                record.searcher_error = record.searcher_error or "Searcher returned no result"

        append_record(records_path, record.model_dump(mode="json"))


def json_safe_trajectory(result, worker) -> str:
    """把 trajectory 折叠成可读摘要（避免 records.jsonl 过大；原始输出仍可追溯）。"""
    if result is None or not getattr(result, "trajectory", None):
        return ""
    out: list[str] = []
    for step in result.trajectory:
        role = step.get("role")
        if role == "assistant":
            content = str(step.get("content", ""))[:300].replace("\n", " ")
            calls = step.get("tool_calls") or []
            if calls:
                names = [c.get("function", {}).get("name", "") for c in calls]
                out.append(f"assistant(calls={names}) {content}".strip())
            else:
                out.append(f"assistant {content}".strip())
        elif role == "tool":
            name = step.get("name", "")
            res = str(step.get("result", ""))[:200].replace("\n", " ")
            out.append(f"tool[{name}] {res}")
    return "\n".join(out)


def summarize_legalbench_records(records_path: str | Path, ks: list[int]) -> dict:
    """从已持久化的 records.jsonl 计算一个 benchmark 的聚合计分（宏平均口径）。"""
    import json as _json

    from .persist import load_done_ids

    ks = list(ks)
    stats = {f"recall_at_{k}": [] for k in ks}
    stats["mrr"] = []
    stats["span_precision"] = []
    stats["span_recall"] = []
    stats["span_f1"] = []
    stats["agent_span_precision"] = []
    stats["agent_span_recall"] = []
    stats["agent_span_f1"] = []
    n_agent = 0
    n_searcher_error = 0
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
            if rec.get("searcher_error"):
                n_searcher_error += 1
    return {
        "n_queries": len(load_done_ids(records_path)),
        "n_agent": n_agent,
        "n_searcher_error": n_searcher_error,
        "metrics": {key: _mean(vals) for key, vals in stats.items() if vals},
        "all": {key: vals for key, vals in stats.items()},
    }
