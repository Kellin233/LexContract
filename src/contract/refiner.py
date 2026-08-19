"""Refiner：整个系统中唯一承担综合结论生成的 Agent。

输入：原始问题 + 全部 verified evidence（全文）+ Reviewer 终态。
输出：结构化 RefinerResult（结论 + 分点 + 证据缺口），并渲染成 Markdown。

引用约束：正文只用 [E###]，由后处理系统把 E### 映射成《文档》第X条 页Y，
杜绝模型自编条款号。PARTIALLY_SUFFICIENT 时禁止幻想补全，须明说“文档中未找到足够条款确认”。
"""
from __future__ import annotations

from .jsonutil import extract_json_object
from .schemas import RefinerResult, RefinerPoint, Citation, FinalStatus, ResearchState
from .store import EvidenceStore
from ..utils.conversation_recorder import set_agent
from ..utils.tokens import estimate_messages_tokens, append_token_usage
from ..utils.tracing import trace_chain


__all__ = ["Refiner"]

SYSTEM_PROMPT = """\
You are the final contract-analysis synthesizer. You have the user's question and a set of VERIFIED original contract clauses (evidence). Produce the final answer.

RULES:
1. Cite evidence ONLY by its ID in brackets, e.g. [E001][E002]. NEVER invent article numbers like "Article 13.2" that are not backed by an evidence ID.
2. Every claim in `points` must be supported by at least one real evidence ID from the given list.
3. Reasoning across clauses is allowed and expected (main rule + exception, initial term vs renewal term, cross-references).
4. Select `supporting_evidence_ids`: the evidence that MOST supports your conclusion. Choose only what you actually relied upon — this may be ALL of them, or only a SUBSET. Do not just dump every piece of evidence; pick the most relevant/supportive ones.
5. If evidence is insufficient (status PARTIALLY_SUFFICIENT), explicitly state in `evidence_gap` what cannot be confirmed from this document set. Do NOT hallucinate to fill gaps.
6. `conclusion` is a 1-2 paragraph overall answer to the original question.
7. `points` are the supporting breakdown, each with the evidence IDs that back it.

Output STRICT JSON only:
{{
  "conclusion": "overall conclusion",
  "points": [{{"claim": "supporting point", "evidence_ids": ["E001", "E002"]}}],
  "supporting_evidence_ids": ["E001", "E003"],
  "evidence_gap": ["what could not be confirmed"]
}}
ONLY the JSON object, no prose."""


def _section_label(evidence) -> str:
    path = list(evidence.section_path or [])
    return path[-1] if path else ""


# Refiner 输入预算默认值（token）。超过只会告警/记录，本次不裁剪（裁剪策略后续接入）。
DEFAULT_REFINER_INPUT_TOKENS = 65536


class Refiner:
    def __init__(self, policy, input_token_budget: int | None = None,
                 system_prompt: str | None = None) -> None:
        self.policy = policy
        self.input_token_budget = (
            int(input_token_budget) if input_token_budget is not None else DEFAULT_REFINER_INPUT_TOKENS
        )
        # 按评测可切换系统提示词；不传时保持正式生产链路的默认提示词
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    @trace_chain(name="contract_refiner.refine", tags=["contract", "refiner"])
    def refine(self, state: ResearchState, store: EvidenceStore) -> RefinerResult:
        set_agent("refiner")
        final_status = state.final_status or FinalStatus.PARTIALLY_SUFFICIENT
        prompt = self._build_prompt(state, store, final_status)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        # Refiner 输入预算（token 口径）：超过只告警/记录，本次不裁剪
        _budget_note: str | None = None
        _budget_tokens = estimate_messages_tokens(messages)
        append_token_usage(_budget_tokens)  # 计入本轮全链路 token 账本
        if _budget_tokens > self.input_token_budget:
            _budget_note = (
                f"证据输入约 {_budget_tokens} tokens，超过 Refiner 预算 {self.input_token_budget} tokens；"
                f"本次未裁剪，仍全量喂入（裁剪策略暂未实现）。"
            )
            print(f"[Refiner][WARN] {_budget_note}")
        try:
            response = self.policy(messages)
        except RuntimeError:
            return self._degraded(state, store, final_status, "LLM 调用失败")
        content = response.get("content", "") or ""
        data = extract_json_object(content) or {}

        used_ids: list[str] = []
        for p in (data.get("points") or []):
            for eid in (p.get("evidence_ids") or []):
                if eid not in used_ids:
                    used_ids.append(eid)
        existing = {e.evidence_id for e in store.all()}
        # 过滤不存在的证据 ID，避免引用幻觉
        used_ids = [eid for eid in used_ids if eid in existing]

        # “最支持结论的证据”：优先取模型显式选中的子集；若缺失/为空，退化为分点引用并集
        supporting = [eid for eid in (data.get("supporting_evidence_ids") or []) if eid in existing]
        if not supporting:
            supporting = used_ids

        result = RefinerResult(
            conclusion=str(data.get("conclusion", "")),
            points=[
                RefinerPoint(
                    claim=str(p.get("claim", "")),
                    evidence_ids=[eid for eid in (p.get("evidence_ids") or []) if eid in existing],
                )
                for p in (data.get("points") or [])
            ],
            supporting_evidence_ids=supporting,
            evidence_gap=[str(g) for g in (data.get("evidence_gap") or [])],
            final_status=final_status,
        )
        # 依据仅落在“最支持结论的证据”子集上（可能是全部，也可能只是一部分）
        result.citations = self._build_citations(supporting, store)
        result.markdown_body = render_markdown(state.original_question, result)
        if _budget_note:
            result.evidence_gap.append(f"[预算告警] {_budget_note}")
        return result

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _build_citations(self, evidence_ids: list[str], store: EvidenceStore) -> list[Citation]:
        citations: list[Citation] = []
        for ev in store.get(evidence_ids):
            label = _section_label(ev)
            citations.append(Citation(
                evidence_id=ev.evidence_id,
                doc_title=ev.document_name,
                section_label=label,
                page_no=ev.page_no,
                quote=ev.quote[:200],
            ))
        return citations

    def _degraded(self, state: ResearchState, store: EvidenceStore,
                  final_status: FinalStatus, reason: str) -> RefinerResult:
        """LLM 失败时用全部证据拼一个最小报告，保证有输出。"""
        result = RefinerResult(
            conclusion=f"（Refiner 生成失败，原因：{reason}。以下仅列出支撑证据，未做综合结论。）",
            supporting_evidence_ids=store.all_ids(),
            evidence_gap=["当前未能生成综合结论，请结合下方证据自行判断。"],
            final_status=final_status,
        )
        result.citations = self._build_citations(store.all_ids(), store)
        result.markdown_body = render_markdown(state.original_question, result)
        return result

    def _build_prompt(self, state: ResearchState, store: EvidenceStore, final_status: FinalStatus) -> str:
        lines = [
            f"## Original question\n{state.original_question}",
            f"## Evidence status\n{final_status.value}"
            + (" (evidence judged sufficient)" if final_status == FinalStatus.SUFFICIENT else " (insufficient evidence / iteration limit reached; state the gaps honestly)"),
            "",
            "## Verified evidence (full text)",
        ]
        for ev in store.all():
            label = _section_label(ev) or "(no section)"
            lines.append(
                f"\n### {ev.evidence_id} Document \"{ev.document_name}\" [{label}] page {ev.page_no}\n"
                f"{ev.quote}"
            )
        lines.append("\nPlease produce the final answer based on the evidence above (strict JSON).")
        return "\n".join(lines)


def render_markdown(question: str, result: RefinerResult) -> str:
    """把 RefinerResult 渲染成可读 Markdown。"""
    status_text = "证据充分" if result.final_status == FinalStatus.SUFFICIENT else "有限证据（部分方面无法完全确认）"
    lines = [
        f"# 合同审查结论：{question}",
        "",
        f"> 证据状态：{status_text}",
        "",
        "## 结论",
        "",
        result.conclusion or "（无总结论）",
        "",
        "## 分点结论",
        "",
    ]
    if result.points:
        for i, p in enumerate(result.points, 1):
            refs = "".join(f"[{eid}]" for eid in p.evidence_ids)
            lines.append(f"{i}. {p.claim} {refs}")
    else:
        lines.append("（无分点）")
    lines += ["", "## 最支持结论的证据（可能为全部，也可能只列一部分）"]
    if result.citations:
        for c in result.citations:
            loc = f"{c.doc_title}"
            if c.section_label:
                loc += f"《{c.section_label}》"
            loc += f"，第 {c.page_no} 页" if c.page_no else ""
            lines.append(f"- **{c.evidence_id}** {loc}：{c.quote[:120]}")
    else:
        lines.append("（无引用）")
    lines += ["", "## 未能确认的缺口"]
    if result.evidence_gap:
        for g in result.evidence_gap:
            lines.append(f"- {g}")
    else:
        lines.append("（无）")
    lines.append("")
    return "\n".join(lines)
