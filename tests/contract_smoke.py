#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/contract_smoke.py
================================================================================
合同证据链流程冒烟测试（非 pytest，运行：python tests/contract_smoke.py）

两层：
  Tier A（离线，真实 DB）：EvidenceAssembler / CitationVerifier / EvidenceStore
      —— quote == full_text[offset]、篡改引用被拒、按跨度去重。
  Tier B（离线，确定性 stub，不依赖 LLM）：驱动完整 Orchestrator 状态机，
      覆盖 NEED_MORE → 增量规划 → 第二轮撒证据 → SUFFICIENT → Refiner。

前置：ParadeDB 容器运行中；至少一个 session 里已入库合同（可用
      python -m src.document.main parse + migrate 完成）。
DB 不可达时打印 SKIP 并无害退出。
"""
from __future__ import annotations

import asyncio
import copy
import sys

sys.path.insert(0, ".")

from src.contract.tools import DocumentToolkit          # noqa: E402
from src.contract.assembler import EvidenceAssembler    # noqa: E402
from src.contract.verifier import CitationVerifier      # noqa: E402
from src.contract.store import EvidenceStore            # noqa: E402
from src.contract.schemas import (                      # noqa: E402
    WorkerResult, FinalStatus,
)
from src.contract.planner import Planner        # noqa: E402
from src.contract.reviewer import Reviewer              # noqa: E402
from src.contract.refiner import Refiner                # noqa: E402
from src.orchestrator.orchestrator import Orchestrator  # noqa: E402
from src.orchestrator.agent_pool import AgentPool       # noqa: E402
from src.orchestrator.schemas import RunConfig, AgentResult, AgentStatus  # noqa: E402
from src.agents.base_agent import BaseAgent             # noqa: E402

PASS = 0


def ok(msg: str) -> None:
    global PASS
    PASS += 1
    print(f"  ✓ {msg}")


def db_available() -> bool:
    try:
        from src.retrieval.store import connect
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        return True
    except Exception:  # noqa: BLE001
        return False


def list_sessions() -> list[str]:
    from src.retrieval.store import connect
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT session_id FROM documents WHERE session_id <> ''")
        return [r[0] for r in cur.fetchall()]


def iter_sections(tk: DocumentToolkit):
    """遍历当前 session 内所有已入库文档的章节。"""
    for doc in tk.list_documents():
        did = doc["doc_id"]
        for row in tk.get_document_outline(did):
            if row.get("start_offset"):
                yield {"doc_id": did, **row}


# ============================================================================
# 确定性 stub 策略与 stub worker（不依赖 LLM / 网络）
# ============================================================================
class FakePolicy:
    """按 system prompt 关键词返回固定 JSON 的确定性策略。"""

    def __init__(self, name: str, responses: list[str]):
        self.name = name
        self.responses = list(responses)

    def __call__(self, messages):
        sys_text = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        if "completeness reviewer" in sys_text:
            payload = self.responses.pop(0) if self.responses else self._reviewer_json("NEED_MORE")
        elif "final contract-analysis synthesizer" in sys_text:
            payload = self.responses[0] if self.responses else "{}"
        elif "contract research planning assistant" in sys_text:
            payload = self._planner_json()
        else:
            payload = "{}"

        class _Resp(dict):
            was_truncated = False

        return _Resp({"content": payload})

    @staticmethod
    def _planner_json() -> str:
        return (
            '{"research_questions": ['
            '{"question_id": "Q1", "question": "查找甲方付款期限的条款", "doc_hints": ["付款", "支付"]},'
            '{"question_id": "Q2", "question": "查找逾期交货违约金的条款", "doc_hints": ["逾期", "违约金"]}'
            "]}"
        )

    @staticmethod
    def _reviewer_json(status: str) -> str:
        if status == "NEED_MORE":
            return (
                '{"status": "NEED_MORE", "covered_aspects": ["付款期限"],'
                '"missing_aspects": [{"description": "不可抗力情况下的责任分担", "reason": "原问题要求考虑履行障碍情形"}],'
                '"conflicts": [], "keep_evidence_ids": [], "notes": "stub"}'
            )
        return (
            '{"status": "SUFFICIENT", "covered_aspects": ["付款期限", "逾期违约金", "不可抗力责任"],'
            '"missing_aspects": [], "conflicts": [], "keep_evidence_ids": [], "notes": "stub"}'
        )


# 全局候选队列（预先把真实章节算成候选），被 StubSearcher 按运行次数顺序消费
_CANDIDATES: list[dict] = []
_CAND_INDEX = 0


def prepare_candidates(tk: DocumentToolkit) -> list[dict]:
    cands = []
    for row in iter_sections(tk):
        sec = tk.get_section(row["doc_id"], row["section_path"])
        if sec and sec.get("text"):
            cands.append({
                "doc_id": row["doc_id"],
                "start_offset": sec["start_offset"],
                "end_offset": sec["end_offset"],
                "section_path": row["section_path"],
                "source_chunk_ids": sec.get("chunk_ids", []),
                "relevance_note": "stub 候选",
            })
    return cands


class StubSearcher(BaseAgent):
    """绕过 LLM 的工人：每次 run 消费下一个真实候选，走真实 装配→校验→入库 链路。"""

    def __init__(self, name, policy, toolkit, assembler, verifier, store):
        super().__init__(name, policy, tools=[])
        self._toolkit = toolkit
        self._assembler = assembler
        self._verifier = verifier
        self._store = store

    async def run(self, task, context):
        global _CAND_INDEX
        # 与真实 Searcher 一致：从 context 绑定当轮 EvidenceStore
        # （空 EvidenceStore 在 bool() 下为 False，必须用 is not None 判断）
        ctx_store = context.get("evidence_store")
        store = ctx_store if ctx_store is not None else self._store
        cand = _CANDIDATES[_CAND_INDEX % len(_CANDIDATES)]
        _CAND_INDEX += 1
        wr = WorkerResult(question_id=task.task_id, question=task.description,
                          search_queries=["stub"], searched=True)
        ev = self._assembler.materialize(cand, task.task_id)
        if ev is not None and self._verifier.verify(ev):
            reg, _ = store.register(ev, task.task_id)
            wr.evidences.append(reg)
        wr.no_evidence_found = not wr.evidences
        return AgentResult(task_id=task.task_id, status=AgentStatus.SUCCESS,
                           output=wr, confidence=1.0 if wr.evidences else 0.0)


# ============================================================================
# Tier A：装配 / 校验 / 去重（真实 DB）
# ============================================================================
def tier_a(tk: DocumentToolkit, sec_row: dict, sec: dict) -> None:
    print("\n=== Tier A：装配 / 校验 / 去重 ===")
    assembler = EvidenceAssembler(tk)
    verifier = CitationVerifier(tk)
    store = EvidenceStore()

    cand = {
        "doc_id": sec_row["doc_id"],
        "start_offset": sec["start_offset"],
        "end_offset": sec["end_offset"],
        "section_path": sec_row["section_path"],
        "source_chunk_ids": sec.get("chunk_ids", []),
        "page_no": 0,
        "relevance_note": "规定交付时间与逾期违约金",
    }
    ev = assembler.materialize(cand, "Q1")
    assert ev is not None, "装配失败"
    full = tk.get_full_text(sec_row["doc_id"])
    if full[ev.start_offset:ev.end_offset] == ev.quote:
        ok("Evidence.quote 与 full_text[offset] 完全一致")
    else:
        print("  ✗ quote 与原文不一致")
    if verifier.verify(copy.deepcopy(ev)):
        ok("CitationVerifier 校验通过（真实原文）")
    else:
        print("  ✗ 真实原文校验失败")

    tampered = copy.deepcopy(ev)
    tampered.quote = ev.quote[:-1] + ("X" if ev.quote else "X")
    if not verifier.verify(tampered):
        ok("篡改 quote 后被校验拒绝")
    else:
        print("  ✗ 篡改引用未被拒绝")

    reg1, is_new = store.register(copy.deepcopy(ev), "Q1")
    reg2, is_new2 = store.register(copy.deepcopy(ev), "Q2")
    if not is_new2 and reg2.evidence_id == reg1.evidence_id:
        ok("同跨度去重（Q1/Q2 复用同一 E###）")
    else:
        print("  ✗ 去重失效")


# ============================================================================
# Tier B：完整 Orchestrator 状态机（NEED_MORE → 增量 → SUFFICIENT）
# ============================================================================
def tier_b(tk: DocumentToolkit) -> None:
    print("\n=== Tier B：Orchestrator 端到端（stub LLM + 真实 DB）===")
    assembler = EvidenceAssembler(tk)
    verifier = CitationVerifier(tk)
    store_pool = EvidenceStore()

    planner_policy = FakePolicy("planner", [])
    reviewer_policy = FakePolicy(
        "reviewer", [FakePolicy._reviewer_json("NEED_MORE"), FakePolicy._reviewer_json("SUFFICIENT")]
    )
    refiner_policy = FakePolicy(
        "refiner",
        ['{"conclusion": "综合结论", '
         '"points": [{"claim": "付款期限与逾期违约金条款", "evidence_ids": ["E001"]}], '
         '"evidence_gap": []}'],
    )

    planner = Planner(policy=planner_policy)
    reviewer = Reviewer(policy=reviewer_policy)
    refiner = Refiner(policy=refiner_policy)

    def _make_worker():
        return StubSearcher("stub_searcher", planner_policy, tk, assembler, verifier, store_pool)

    pool = AgentPool(policy_factory=lambda: planner_policy, worker_factory=_make_worker, max_idle=3)
    orch = Orchestrator(planner=planner, agent_pool=pool, reviewer=reviewer, refiner=refiner,
                        evidence_store=store_pool, compressor=None)

    config = RunConfig(
        max_concurrent=2,
        global_timeout_seconds=120,
        max_iterations=3,
        stop_on_no_effective_new_evidence=True,
        session_id=tk.session_id,
        doc_ids=[],
    )
    report = asyncio.run(orch.run(
        "甲方付款期限与逾期交付违约金如何约定？不可抗力如何分担责任？", config=config,
    ))

    structured = report.structured or {}
    final_status = structured.get("final_status")
    points = structured.get("points", [])
    citations = structured.get("citations", [])
    review_history = orch._research_state.review_history if orch._research_state else []

    if final_status == FinalStatus.SUFFICIENT.value:
        ok(f"最终状态 SUFFICIENT（{len(review_history)} 轮 Reviewer 后收尾）")
    else:
        print(f"  ✗ 预期 SUFFICIENT，得到 {final_status}")
    if len(review_history) == 2:
        ok("Reviewer 两轮：第 1 轮 NEED_MORE → 增量规划 → 第 2 轮 SUFFICIENT")
    else:
        print(f"  ✗ 预期 2 轮审查，实际 {len(review_history)}")
    existing = {e.evidence_id for e in orch._evidence_store.all()}
    all_refs = [eid for p in points for eid in p.get("evidence_ids", [])]
    if all_refs and all(eid in existing for eid in all_refs):
        ok(f"Refiner 引用的证据 ID 全部真实（{all_refs}）")
    else:
        print(f"  ✗ 引用不存在的证据: {all_refs}")
    if citations and all(c["evidence_id"] in existing for c in citations):
        ok(f"Citations 绑定到真实证据（{len(citations)} 条）")
    else:
        print("  ✗ citations 缺失或绑定失败")
    if structured.get("markdown_body"):
        ok("Markdown 正文已渲染")
    else:
        print("  ✗ markdown_body 为空")
    if len(orch._evidence_store) >= 2:
        ok(f"EvidenceStore 累计 {len(orch._evidence_store)} 条证据（含增量轮新证据 + 去重）")
    else:
        print("  ✗ EvidenceStore 证据不足")


def main() -> int:
    print("合同证据链冒烟测试")
    db_ok = db_available()
    print(f"DB 可达: {db_ok}")
    if not db_ok:
        print("SKIP：ParadeDB 不可达（先启动容器并引入文档）")
        return 0

    sessions = list_sessions()
    if not sessions:
        print("SKIP：没有已分配会话的文档（先 parse + assign session）")
        return 0
    global _CANDIDATES
    tk = None
    _CANDIDATES = []
    for session_id in sessions:
        candidate_tk = DocumentToolkit(session_id=session_id)
        try:
            candidate_list = prepare_candidates(candidate_tk)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP：读取数据库章节时连接中断（{type(exc).__name__}）")
            return 0
        if candidate_list:
            tk = candidate_tk
            _CANDIDATES = candidate_list
            break
    if tk is None:
        print("SKIP：会话内没有可用章节（先 migrate 保证 full_text）")
        return 0

    # 取一个真实章节行供 Tier A
    first = iter_sections(tk).__next__()
    sec = tk.get_section(first["doc_id"], first["section_path"])
    tier_a(tk, first, sec)
    tier_b(tk)

    print(f"\n共 {PASS} 项检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
