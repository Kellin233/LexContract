"""EvidenceWorker：针对研究问题收集完整、可引用的原始证明条款。

边界（见方案）：只找证据，不输出任何结论。
- 可多次检索（同义扩展：终止/解除/单方解除/退出…）
- 通过 get_context/get_section/get_referenced_section 把碎片恢复成完整条款
- 最终输出结构化候选，交给 EvidenceAssembler + CitationVerifier 物化与校验
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from .assembler import EvidenceAssembler
from .schemas import WorkerResult
from .store import EvidenceStore
from .tools import DocumentToolkit
from .verifier import CitationVerifier
from ..agents.base_agent import BaseAgent
from ..orchestrator.schemas import SubTask, AgentResult, AgentStatus
from ..utils.tracing import trace_agent


__all__ = ["EvidenceWorker"]

SYSTEM_PROMPT = """\
You are a meticulous contract-evidence retrieval assistant. Your ONLY job is to locate and capture the ORIGINAL clauses of the contract documents relevant to the research question.

CRITICAL RULES:
1. NEVER interpret, judge, summarize, or reach conclusions about the contract. Only collect original text.
2. Use the retrieval tools to find clauses. Start with `search` (try synonyms if nothing useful: e.g. 终止/解除/退出/单方解除/提前终止).
3. When a hit is a fragment, use `get_context` to expand around it, or `get_section` to fetch the complete clause by section_path, or `get_document_outline` to locate where a clause is.
4. Follow internal cross-references: if a clause says "除第X条规定外" / "except as provided in Article X", call `get_referenced_section` to also capture that section.
5. Collect ALL clauses that bear on the research question — do not stop at the first hit.
6. For each captured clause, report its precise offsets (use the start_offset/end_offset shown by `get_section` / chunks) and the source chunk ids.

FINAL OUTPUT FORMAT (must be the last assistant message, JSON only — an array, may be empty):
[
  {{
    "doc_id": "...",
    "start_offset": 0,
    "end_offset": 0,
    "section_path": ["第十二条"],
    "source_chunk_ids": ["doc-xxx:3"],
    "page_no": 1,
    "relevance_note": "一句话说明该条款与本问题的关系（不作为引用）",
    "retrieval_score": 0.8
  }}
]

If after exhaustive searching nothing relevant is found, output [] (still JSON). Never fabricate document content. Do not write any prose outside the JSON array.
"""


def _extract_json_array(text: str) -> list | None:
    """从末尾 assistant 消息中稳健提取 JSON 数组；失败返回 None。"""
    raw = (text or "").strip()
    if not raw:
        return None
    # 去围栏
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = lines[1:] if lines and lines[0].startswith("```") else lines
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[(?:.|\n)*\]", raw)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            return None
    return None


class EvidenceWorker(BaseAgent):
    """证据检索 Worker：多轮 tool-calling，输出 WorkerResult（仅证据）。"""

    def __init__(
        self,
        name: str,
        policy,
        toolkit: DocumentToolkit,
        assembler: EvidenceAssembler,
        verifier: CitationVerifier,
        store: EvidenceStore,
        max_turns: int = 10,
    ) -> None:
        super().__init__(name, policy, tools=toolkit.get_tools())
        self.toolkit = toolkit
        self.assembler = assembler
        self.verifier = verifier
        self.store = store
        self.max_turns = max_turns
        self.tool_map: dict[str, Any] = {t.name: t for t in (self.tools or [])}

    @trace_agent(name="evidence_worker.run", tags=["contract", "worker"])
    async def run(self, task: SubTask, context: dict) -> AgentResult:
        question = task.description or context.get("question", "")
        question_id = task.task_id
        doc_hints = list(task.search_hints or context.get("doc_hints") or [])

        # 对象池可能跨运行复用本实例：每次运行按 context 重新绑定作用域与证据库
        self.toolkit.set_scope(
            str(context.get("session_id", "")),
            list(context["doc_ids"]) if context.get("doc_ids") else [],
        )
        # 注意：EvidenceStore 定义了 __len__，空库在 bool() 下为 False，
        # 因此这里必须用 is None 判断，不能用 `or` 兜底，否则会把空库误当未提供。
        ctx_store = context.get("evidence_store")
        store = ctx_store if ctx_store is not None else self.store

        task_desc = self._build_task_prompt(question, doc_hints, context)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_desc},
        ]

        if hasattr(self.policy, "set_tools"):
            self.policy.set_tools([t.get_openai_tool_schema() for t in self.tools])

        trajectory: list[dict] = []
        total_tokens = 0
        search_count = 0
        search_queries: list[str] = []
        candidates: list | None = None
        searched = False

        for turn in range(self.max_turns):
            # 若上一轮模型未调用工具且尚未产出可解析结果，强制其先检索
            if (
                turn > 0
                and messages
                and messages[-1].get("role") == "assistant"
                and not messages[-1].get("tool_calls")
                and candidates is None
            ):
                messages.append({
                    "role": "user",
                    "content": (
                        "You must call a retrieval tool now (search / get_context / get_section / "
                        "get_referenced_section) to gather evidence. Do not stop without evidence "
                        "or without a valid JSON array."
                    ),
                })

            try:
                response = await asyncio.to_thread(self.policy, messages)
            except RuntimeError as e:
                trajectory.append({"turn": turn, "error": str(e)})
                return AgentResult(task_id=question_id, status=AgentStatus.TIMEOUT,
                                   output=WorkerResult(question_id=question_id, question=question,
                                                       searched=searched, no_evidence_found=True),
                                   trajectory=trajectory, token_usage=total_tokens, confidence=0.0)

            content = response.get("content", "") or ""
            tool_calls = response.get("tool_calls", []) or []
            total_tokens += len(json.dumps(messages, ensure_ascii=False)) // 3

            trajectory.append({"turn": turn, "role": "assistant", "content": content,
                               "tool_calls": [dict(tc) for tc in tool_calls]})

            if tool_calls:
                results: list[dict] = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    result = await self._execute_tool(tool_name, args)
                    if tool_name == "search":
                        searched = True
                        search_count += 1
                        q = str(args.get("query", "")).strip()
                        if q and q not in search_queries:
                            search_queries.append(q)
                    trajectory.append({"turn": turn, "role": "tool",
                                       "tool_call_id": tc.get("id", ""), "name": tool_name,
                                       "result": result})
                    results.append({"tool_call_id": tc.get("id", ""), "name": tool_name, "result": result})

                assistant_msg = {"role": "assistant", "content": content}
                if response.get("reasoning_content"):
                    assistant_msg["reasoning_content"] = response["reasoning_content"]
                messages.append(assistant_msg)
                for r in results:
                    messages.append({
                        "role": "tool", "tool_call_id": r["tool_call_id"],
                        "content": json.dumps(r["result"], ensure_ascii=False, default=str),
                    })
                continue

            # 模型结束（无工具调用）：尝试解析最终候选
            parsed = _extract_json_array(content)
            if parsed is not None or search_count > 0:
                candidates = parsed if parsed is not None else []
                break
            # 无工具调用且无 JSON：本轮过，下一轮强制检索（循环首部处理）

        if candidates is None:
            candidates = []

        worker_result = self._assemble_worker_result(
            candidates, question_id, question, searched, search_queries, store
        )
        return AgentResult(
            task_id=question_id,
            status=AgentStatus.SUCCESS,
            output=worker_result,
            trajectory=trajectory,
            token_usage=total_tokens,
            confidence=1.0 if worker_result.evidences else 0.0,
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _assemble_worker_result(self, candidates: list, question_id: str, question: str,
                                searched: bool, search_queries: list[str],
                                store: "EvidenceStore | None" = None) -> WorkerResult:
        """候选 → 物化 → 校验 → 入证据库（去重）。"""
        store = store or self.store
        result = WorkerResult(
            question_id=question_id,
            question=question,
            search_queries=search_queries,
            searched=searched,
            no_evidence_found=False,
        )
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            ev = self.assembler.materialize(cand, question_id)
            if ev is None:
                continue
            if not self.verifier.verify(ev):
                continue  # 原文校验失败，丢弃
            registered, _ = store.register(ev, question_id)
            result.evidences.append(registered)
        result.no_evidence_found = result.searched and not result.evidences
        return result

    def _build_task_prompt(self, question: str, doc_hints: list[str], context: dict) -> str:
        lines = [
            "## Research question to investigate（只用它来决定找哪些条款，不要在此回答它）",
            question,
            "",
            "## Instructions",
            "1. Use the retrieval tools to find every clause relevant to this question.",
            "2. Capture COMPLETE clauses (continuous original text), not fragments.",
            "3. When done, reply with ONLY the JSON array of evidence candidates.",
        ]
        if doc_hints:
            lines += ["", f"Document/keyword hints: {', '.join(doc_hints)}"]
        if context.get("query"):
            lines += ["", f"Original user question (context): {context['query']}"]
        return "\n".join(lines)

    async def _execute_tool(self, tool_name: str, args: dict) -> dict:
        tool = self.tool_map.get(tool_name)
        if tool is None:
            return {"error": f"Tool '{tool_name}' not found"}
        try:
            return await tool.execute(**args)
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}
