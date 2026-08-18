"""Reviewer：研究完整性审查（Research Completeness Review）。

只回答三个问题，不做任何结论/法律判断：
1. 当前证据是否足以覆盖原始问题？
2. 证据之间是否存在明显冲突？
3. 还缺哪些信息？

发现冲突只报告（可能是“一般规则+例外”或“初始期+续约期”），不裁判谁优先。
"""
from __future__ import annotations

from .jsonutil import extract_json_object
from .schemas import ReviewResult, ResearchState, ReviewStatus, EvidenceConflict, MissingAspect
from .store import EvidenceStore
from ..utils.tracing import trace_chain


__all__ = ["Reviewer"]

SYSTEM_PROMPT = """\
You are a contract-research completeness reviewer. You evaluate whether the EVIDENCE gathered so far is sufficient to answer a user's question about the contract documents.

You judge ONLY:
1. Coverage: does the evidence cover all aspects the question requires?
2. Conflict: do any two pieces of evidence plainly contradict each other (e.g. one says 30-day notice suffices, another says termination forbidden during initial term)?
3. Gaps: what information is still missing?

You must NOT:
- decide the final answer / reach legal conclusions
- judge which conflicting clause wins (it may be general-rule vs exception, or initial-term vs renewal-term)
- judge document validity or legal correctness

Output STRICT JSON only:
{{
  "status": "SUFFICIENT" or "NEED_MORE",
  "covered_aspects": ["aspects already covered"],
  "missing_aspects": [{{"description": "what is still missing", "reason": "why it is needed (tie back to the question)"}}],
  "conflicts": [{{"evidence_a_id": "E001", "evidence_b_id": "E002", "summary": "objective description of the conflict"}}],
  "keep_evidence_ids": ["evidence IDs to keep"],
  "notes": "additional notes"
}}
ONLY the JSON object, no prose."""


def _section_label(evidence) -> str:
    path = list(evidence.section_path or [])
    if not path:
        return ""
    # 用最末级（通常是编号+标题）作为章节标签
    return path[-1]


class Reviewer:
    def __init__(self, policy) -> None:
        self.policy = policy

    @trace_chain(name="contract_reviewer.review", tags=["contract", "reviewer"])
    def review(self, state: ResearchState, store: EvidenceStore,
               previous: ReviewResult | None = None) -> ReviewResult:
        prompt = self._build_prompt(state, store, previous)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.policy(messages)
        except RuntimeError:
            # 降级：LLM 失败视为 NEED_MORE（保守），保留后续重试由 orchestrator 控制
            return self._fallback(state, previous)
        content = response.get("content", "") or ""
        data = extract_json_object(content)

        result = ReviewResult(status=ReviewStatus.NEED_MORE)
        if data is None:
            return self._fallback(state, previous)

        try:
            status = str(data.get("status", "NEED_MORE")).upper()
            result.status = ReviewStatus.SUFFICIENT if status.startswith("SUFFICIENT") else ReviewStatus.NEED_MORE
            result.covered_aspects = [str(x) for x in (data.get("covered_aspects") or [])]
            for m in (data.get("missing_aspects") or []):
                if isinstance(m, dict):
                    result.missing_aspects.append(MissingAspect(
                        description=str(m.get("description", "")),
                        reason=str(m.get("reason", "")),
                    ))
            for c in (data.get("conflicts") or []):
                if isinstance(c, dict):
                    result.conflicts.append(EvidenceConflict(
                        evidence_a_id=str(c.get("evidence_a_id", "")),
                        evidence_b_id=str(c.get("evidence_b_id", "")),
                        summary=str(c.get("summary", "")),
                    ))
            result.keep_evidence_ids = [str(x) for x in (data.get("keep_evidence_ids") or [])]
            result.notes = str(data.get("notes", ""))
        except Exception:  # noqa: BLE001
            return self._fallback(state, previous)

        result.effective_new_evidence = self._has_effective_new_evidence(previous, result)
        return result

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _fallback(self, state: ResearchState, previous: ReviewResult | None) -> ReviewResult:
        """LLM 解析失败时的保守降级：沿用上一轮缺失项或视为 NEED_MORE。"""
        result = ReviewResult(status=ReviewStatus.NEED_MORE, notes="Reviewer LLM output could not be parsed; conservatively marked NEED_MORE.")
        if previous is not None:
            result.missing_aspects = list(previous.missing_aspects)
        result.effective_new_evidence = False
        return result

    @staticmethod
    def _has_effective_new_evidence(previous: ReviewResult | None, current: ReviewResult) -> bool:
        """上一轮提出的缺失要点是否在本轮被解决（据此决定是否提前停止）。"""
        if previous is None:
            return True
        prev_keys = {m.description for m in previous.missing_aspects}
        if not prev_keys:
            return True
        curr_keys = {m.description for m in current.missing_aspects}
        # 上一轮缺失的任一要点不再缺失 => 有有效新增
        return not prev_keys <= curr_keys

    def _build_prompt(self, state: ResearchState, store: EvidenceStore, previous: ReviewResult | None) -> str:
        lines = [
            f"## Original question\n{state.original_question}",
            "",
            "## Investigation questions",
        ]
        for q in state.questions:
            lines.append(f"- {q.question_id}: {q.question}")
        lines += ["", "## Evidence collected (EvidenceStore)"]
        all_evidence = store.all()
        if not all_evidence:
            lines.append("(no evidence yet)")
        for ev in all_evidence:
            label = _section_label(ev)
            snippet = ev.quote[:150].replace("\n", " ")
            lines.append(
                f"- {ev.evidence_id}: Document \"{ev.document_name}\" [{label}] page {ev.page_no} — "
                f"{snippet} ({ev.relevance_note[:80]})"
            )
        lines += ["", "## Previous review (context only)"]
        if previous is not None:
            lines.append(
                f"status={previous.status.value}, missing={[m.description for m in previous.missing_aspects]}, "
                f"conflicts={[c.evidence_a_id+' vs '+c.evidence_b_id for c in previous.conflicts]}"
            )
        else:
            lines.append("(none)")
        lines.append("")
        lines.append("Please evaluate only coverage / conflicts / gaps, and output strict JSON.")
        return "\n".join(lines)
