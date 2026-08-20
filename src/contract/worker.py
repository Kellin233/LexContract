"""Searcher：针对研究问题收集完整、可引用的原始证明条款。

边界（见方案）：只找证据，不输出任何结论。
- 可多次检索（同义扩展：终止/解除/单方解除/退出…）
- 通过 get_section/get_chunk 把命中恢复成完整条款
- 最终输出结构化候选，交给 EvidenceAssembler + CitationVerifier 物化与校验
"""
from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from typing import Any

from .assembler import EvidenceAssembler
from .schemas import WorkerResult
from .store import EvidenceStore
from .tools import DocumentToolkit
from .verifier import CitationVerifier
from ..agents.base_agent import BaseAgent
from ..orchestrator.schemas import SubTask, AgentResult, AgentStatus
from ..utils.conversation_recorder import set_agent
from ..utils.tokens import estimate_messages_tokens, append_token_usage
from ..utils.tracing import trace_agent


__all__ = ["Searcher"]

# 总工具轮次上限：给"发检索词"、"展开原文"、"输出最终 JSON"都留轮次
MAX_TURNS = 5
# 发检索词的轮次上限（默认：最多 3 轮，每轮 1 个检索词 → 最多 3 个检索词；可由配置覆盖）
MAX_SEARCH_ROUNDS = 3
MAX_SEARCHES_PER_ROUND = 1

# 检索类工具：共享同一检索预算（总轮数 + 每轮检索调用数）。search/grep 之外的读取类工具不占预算。
_RETRIEVAL_TOOLS = frozenset({
    "search",
    "grep",
})


def build_system_prompt(max_rounds: int, max_searches_per_round: int) -> str:
    """Searcher 系统提示词；检索预算随 max_rounds / 每轮检索调用数 动态写入，避免提示词与配置不一致。"""
    rounds_txt = (f"at most {max_rounds} ROUNDS may issue retrieval calls"
                  if max_rounds != 1 else "only 1 ROUND may issue retrieval calls")
    per_round = (f"1 retrieval call" if max_searches_per_round == 1
                 else f"up to {max_searches_per_round} retrieval calls")
    return f"""\
You are a meticulous contract-evidence retrieval assistant. Your ONLY job is to locate and capture the ORIGINAL clauses of the contract documents relevant to the research question.

AVAILABLE TOOLS:
- list_documents: lists the documents in the current session (doc_id/title/source format), no full text.
- search(query, top_k, doc_ids): hybrid semantic search (vector + BM25 + rerank); returns short snippets with metadata and score. Use FIRST to locate clauses by MEANING or synonyms.
- grep(pattern, mode, top_k, case_sensitive, doc_id): exact literal or regex match over the original clause text; returns snippets. Use to CONFIRM exact wording or pin a precise article number.
- get_document_outline(doc_id): section paths + offsets, for navigating clauses by heading.
- get_section(doc_id, section_path): the COMPLETE continuous original text of a section, with offsets and chunk ids.
- get_chunk(chunk_id): the full original text of a single chunk.

CRITICAL RULES:
1. NEVER interpret, judge, summarize, or reach conclusions about the contract. Only collect original text.
2. A clause may be named in the question by a CANONICAL LABEL or standard term that does NOT appear verbatim in the contract (e.g. the "Regulatory Approvals" clause may be titled "Reasonable Best Efforts; Filings", the "Specific Performance" clause may be titled "Enforcement"). To locate a clause by its name, ALWAYS PREFER `search` with the issue's MEANING or synonyms (e.g. regulatory approvals / filings / consents / authorizations; specific performance / enforcement / equitable relief / irreparable harm); use `grep` (literal or regex) mainly to CONFIRM exact wording or to pin a precise article number. Try synonyms if nothing useful (e.g. termination / rescission / unilateral termination / early termination).
3. search/grep return only short snippets. They LOCATE evidence — they are NOT the evidence. To identify which chunks form a clause, call `get_section` (using the section_path shown in the hit) or `get_chunk` (using the chunk id); the system will aggregate the full clause automatically from the chunk ids you report.
4. When you have identified the target document(s), lock later search/grep calls by passing doc_ids / doc_id — otherwise hits mix passages from ALL documents in the corpus. CRITICAL: corpus doc_ids are long, opaque strings (e.g. "maud:VEREIT_Realty_Income_Corporation.pdf||VEREIT_Realty_Income_Corporation Amendment No.1.txt"); NEVER guess or reconstruct a doc_id from a title. Only pass a doc_id you have SEEN verbatim in a previous tool result. If a doc_id-locked search returns [], the id was wrong — re-search WITHOUT doc_ids using a party-name query (e.g. "VEREIT Realty Income specific performance") to surface the correct document.
5. Follow internal cross-references: if a clause says "except as provided in Article X" / "subject to Section X", use `grep` (e.g. pattern="Section X") to locate the referenced provision, then `get_section` to capture it.
6. Collect ALL clauses that bear on the research question — do not stop at the first hit.
7. You have a HARD BUDGET on retrieval calls (search / grep BOTH count): {rounds_txt}, with {per_round} per such round. Choose queries wisely (cover the key synonyms early; prefer semantic search for clause names); expansion via get_section/get_chunk/get_document_outline/list_documents and the final JSON reply do not count against the budget.
8. For each relevant clause, report the chunk ids that hit it, copied VERBATIM from tool results; if a clause spans multiple chunks, report all chunks it involves. The system will automatically complete the full clause at the finest section level — do NOT fill in any offsets, section paths, page numbers, or scores.

FINAL OUTPUT FORMAT (must be the last assistant message, JSON only — an array, may be empty):
[
  {{
    "source_chunk_ids": ["doc-xxx:3", "doc-xxx:4"],
    "relevance_note": "one sentence on why this clause is relevant to the question (not used as a citation)"
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


class Searcher(BaseAgent):
    """证据检索 Searcher：多轮 tool-calling，输出 WorkerResult（仅证据）。"""

    def __init__(
        self,
        name: str,
        policy,
        toolkit: DocumentToolkit,
        assembler: EvidenceAssembler,
        verifier: CitationVerifier,
        store: EvidenceStore,
        max_turns: int = MAX_TURNS,
        dedup_tool_results: bool = True,
        max_search_rounds: int = MAX_SEARCH_ROUNDS,
        max_searches_per_round: int = MAX_SEARCHES_PER_ROUND,
    ) -> None:
        super().__init__(name, policy, tools=toolkit.get_tools())
        self.toolkit = toolkit
        self.assembler = assembler
        self.verifier = verifier
        self.store = store
        self.max_turns = max_turns
        self.dedup_tool_results = dedup_tool_results
        self.max_search_rounds = max(1, int(max_search_rounds))
        self.max_searches_per_round = max(1, int(max_searches_per_round))
        self.system_prompt = build_system_prompt(self.max_search_rounds, self.max_searches_per_round)
        self.tool_map: dict[str, Any] = {t.name: t for t in (self.tools or [])}
        # 运行期已见集合（每次 run 重置；Searcher 由对象池复用，绝不能跨 run 残留）
        self._seen_chunk_ids: set[str] = set()
        self._seen_sections: set[tuple[str, tuple]] = set()
        self._deduped_count = 0

    @trace_agent(name="searcher.run", tags=["contract", "worker"])
    async def run(self, task: SubTask, context: dict) -> AgentResult:
        question = task.description or context.get("question", "")
        question_id = task.task_id
        set_agent(f"searcher/{question_id}")
        # 每次运行重置去重状态（对象池可能复用本实例）
        self._seen_chunk_ids.clear()
        self._seen_sections.clear()
        self._deduped_count = 0
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
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task_desc},
        ]

        if hasattr(self.policy, "set_tools"):
            self.policy.set_tools([t.get_openai_tool_schema() for t in self.tools])

        trajectory: list[dict] = []
        total_tokens = 0
        search_count = 0
        search_queries: list[str] = []
        search_rounds_used = 0
        candidates: list | None = None
        searched = False

        for turn in range(self.max_turns):
            finalize_turn = (turn == self.max_turns - 1)
            # 若上一轮模型未调用工具且尚未产出可解析结果，强制其先检索
            if (
                turn > 0
                and messages
                and messages[-1].get("role") == "assistant"
                and not messages[-1].get("tool_calls")
                and candidates is None
                and not finalize_turn
            ):
                messages.append({
                    "role": "user",
                    "content": (
                        "You must call a retrieval tool now (search / grep / get_section / get_chunk / "
                        "get_document_outline / list_documents) to gather evidence. "
                        "Do not stop without evidence or without a valid JSON array."
                    ),
                })

            # 最后一轮：不再接受工具调用，强制输出证据 JSON，避免"搜了很多却交不出候选"
            if finalize_turn and candidates is None:
                if hasattr(self.policy, "set_tools"):
                    try:
                        self.policy.set_tools([])
                    except Exception:  # noqa: BLE001
                        pass
                messages.append({
                    "role": "user",
                    "content": (
                        "Tool calls are now disabled. Reply with ONLY the JSON array of evidence "
                        "candidates (may be empty) based on what you have gathered."
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
            # token 口径统一（src/utils/tokens.py）；每轮对该消息数组估算，语义维持原样
            _delta_tokens = estimate_messages_tokens(messages)
            total_tokens += _delta_tokens
            append_token_usage(_delta_tokens)  # 计入本轮"所有 Agent"token 账本

            trajectory.append({"turn": turn, "role": "assistant", "content": content,
                               "tool_calls": [dict(tc) for tc in tool_calls]})

            # 收官轮不允许再执行工具（工具已被禁用），直接走解析分支
            if tool_calls and not finalize_turn:
                results: list[dict] = []
                turn_search_count = 0  # 单轮 search 次数上限控制
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    if tool_name in _RETRIEVAL_TOOLS:
                        # 检索轮数（发检索调用的轮）已用满：不再执行 search/grep（允许 get_section 等继续）
                        if search_rounds_used >= self.max_search_rounds:
                            _q = str(args.get("pattern") or args.get("query", ""))[:50]
                            note = {"error": f"Search-round budget exhausted ({self.max_search_rounds} rounds); this search was skipped "
                                             f"(you may still use get_section/get_chunk to expand existing hits): {_q}"}
                            trajectory.append({"turn": turn, "role": "tool",
                                               "tool_call_id": tc.get("id", ""), "name": tool_name,
                                               "result": note})
                            results.append({"tool_call_id": tc.get("id", ""), "name": tool_name, "result": note})
                            continue
                        # 单轮内检索调用数超限：不执行，回显提示
                        if turn_search_count >= self.max_searches_per_round:
                            _q = str(args.get("pattern") or args.get("query", ""))[:50]
                            note = {"error": f"At most {self.max_searches_per_round} retrieval calls per round; this call was skipped: "
                                             f"{_q}"}
                            trajectory.append({"turn": turn, "role": "tool",
                                               "tool_call_id": tc.get("id", ""), "name": tool_name,
                                               "result": note})
                            results.append({"tool_call_id": tc.get("id", ""), "name": tool_name, "result": note})
                            continue
                    result = await self._execute_tool(tool_name, args)
                    if self.dedup_tool_results:
                        result = self._dedup_tool_result(tool_name, result)
                    if tool_name in _RETRIEVAL_TOOLS:
                        turn_search_count += 1
                        searched = True
                        search_count += 1
                        q = str(args.get("pattern") or args.get("query", "")).strip()
                        if q and q not in search_queries:
                            search_queries.append(q)
                    trajectory.append({"turn": turn, "role": "tool",
                                       "tool_call_id": tc.get("id", ""), "name": tool_name,
                                       "result": result})
                    results.append({"tool_call_id": tc.get("id", ""), "name": tool_name, "result": result})

                # 只要本轮实际执行过 search，就计为一个"检索轮"
                if turn_search_count > 0:
                    search_rounds_used += 1

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
            if parsed is not None:
                candidates = parsed
                break
            if search_count > 0:
                # 已检索但未交出 JSON 候选：不丢弃检索成果，补一轮要求直接输出证据 JSON
                messages.append({
                    "role": "user",
                    "content": (
                        "You have already performed retrieval. Now reply with ONLY the JSON array of "
                        "evidence candidates (may be empty). No further tool calls are needed."
                    ),
                })
                continue
            # 无工具调用且无 JSON：本轮过，下一轮强制检索（循环首部处理）

        if candidates is None:
            candidates = []

        if self._deduped_count:
            print(f"[Searcher/{question_id}] dedup {self._deduped_count} repeated chunk/section texts "
                  f"(tool-result dedup on: {self.dedup_tool_results})")

        worker_result = self._assemble_worker_result(
            candidates, question_id, question, searched, search_queries, store,
            search_tool_call_count=search_count,
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
    def _dedup_tool_result(self, tool_name: str, result: Any) -> Any:
        """去掉本 Searcher 已注入过的完整原文（保留身份/偏移/得分骨架）。

        search/grep 只返回短 snippet，不参与全文去重；get_chunk 按切片去重，
        get_section 按章节去重。quote 由 EvidenceAssembler 从 DB 按偏移物化，
        与 tool message 的 text 无关，故不影响证据正确性。
        """
        if not self.dedup_tool_results:
            return result
        if tool_name == "get_chunk":
            return self._dedup_chunk_item(result)
        if tool_name == "get_section":
            return self._dedup_section_item(result)
        return result

    def _dedup_chunk_item(self, item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        cid = item.get("id")
        if not cid or not isinstance(item.get("text"), str):
            return item
        if cid in self._seen_chunk_ids:
            item["text"] = f"[already shown earlier — chunk {cid}; full text omitted]"
            self._deduped_count += 1
        else:
            self._seen_chunk_ids.add(cid)
        return item

    def _dedup_section_item(self, item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        path = item.get("section_path")
        if not isinstance(path, list) or not path or not isinstance(item.get("text"), str):
            return item
        key = (str(item.get("doc_id", "")), tuple(path))
        if key in self._seen_sections:
            item["text"] = f"[already shown earlier — section {path[-1]}; full text omitted]"
            self._deduped_count += 1
        else:
            self._seen_sections.add(key)
        return item

    def _assemble_worker_result(self, candidates: list, question_id: str, question: str,
                                searched: bool, search_queries: list[str],
                                store: "EvidenceStore | None" = None,
                                search_tool_call_count: int = 0) -> WorkerResult:
        """候选 → 物化 → 校验 → 入证据库（去重）。

        注意：EvidenceStore 定义了 __len__，空库在 bool() 下为 False，
        这里必须用 is None 判断；用 `or` 会把调用方传入的空库误当未提供，
        导致证据注册进 self.store 而非 orchestrator 的每次运行独立证据库。
        """
        store = store if store is not None else self.store
        result = WorkerResult(
            question_id=question_id,
            question=question,
            search_queries=search_queries,
            searched=searched,
            no_evidence_found=False,
            candidate_count=len(candidates),
            search_tool_call_count=search_tool_call_count,
        )
        drop_reasons: Counter = Counter()
        materialize_failed = 0
        verifier_rejected = 0
        verified_count = 0
        for cand in candidates:
            if not isinstance(cand, dict):
                drop_reasons["materialize-fail"] += 1
                materialize_failed += 1
                continue
            ev = self.assembler.materialize(cand, question_id)
            if ev is None:
                drop_reasons["materialize-fail"] += 1
                materialize_failed += 1
                continue
            if not self.verifier.verify(ev):
                drop_reasons[ev.verify_note or "verify-fail"] += 1
                verifier_rejected += 1
                continue  # 原文校验失败，丢弃
            verified_count += 1
            registered, _ = store.register(ev, question_id)
            result.evidences.append(registered)
        result.drop_reasons = dict(drop_reasons)
        result.materialize_failed_count = materialize_failed
        result.verifier_rejected_count = verifier_rejected
        result.verified_evidence_count = verified_count
        result.no_evidence_found = result.searched and not result.evidences
        return result

    def _build_task_prompt(self, question: str, doc_hints: list[str], context: dict) -> str:
        lines = [
            "## Research question to investigate (use it only to decide which clauses to look for; do not answer it here)",
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
