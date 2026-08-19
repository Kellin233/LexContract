"""Planner：把原问题拆成“需要调查什么”（research questions），并支持增量补充。

原则（见方案）：
- 只生成“调查目标”，禁止生成中间答案。
- Initial：一次性覆盖主要调查角度。
- Incremental：只针对 Reviewer 上报的缺失要点补新问题，不重跑已完成问题。
"""
from __future__ import annotations

import json
import re
from typing import Any

from .schemas import ResearchQuestion, ResearchState, QuestionStatus
from ..utils.conversation_recorder import set_agent
from ..utils.tokens import append_token_usage, estimate_messages_tokens
from ..utils.tracing import trace_chain


__all__ = ["Planner", "PlanParseError"]

# 每轮规划最多生成的重点（调查问题）数
MAX_QUESTIONS_PER_CALL = 3


class PlanParseError(Exception):
    pass


INITIAL_PLAN_PROMPT = """\
You are a contract-clause investigation planner. Break the user's original question down into several "research points" (research questions) that need to be investigated.

## Original question
{question}

## Output format (JSON only, no extra text)
{{
  "research_questions": [
    {{
      "question_id": "Q1",
      "question": "Investigation goal (describe only what to look for in the contract, never include a conclusion)",
      "doc_hints": ["keyword1", "keyword2"]
    }}
  ]
}}

## Rules
1. Each point describes only "what to investigate"; e.g. "find the notice-period clause for the receiving party's unilateral early termination" — never write conclusions like "the receiving party may / may not terminate early".
2. A single question often requires combining several clauses to answer, so points should cover all relevant angles (both sides of an issue, exceptions, cross-references, post-termination obligations, etc.).
3. Generate at most 3 points per call; each point must be directly relevant to the original question, prioritizing the most important angles.
4. Do not generate duplicate points.
5. doc_hints should use terms commonly found in the contract text (e.g. "termination", "unilateral termination", "early termination", "force majeure", "breach of contract", "notice period").""" 

INCREMENTAL_PLAN_PROMPT = """\
You are a contract-clause investigation planner. Earlier rounds have not yet covered all points; please supplement with new investigation questions.

## Original question
{question}

## Completed investigation questions (do not repeat)
{completed_questions}

## Evidence collected so far (summary)
{evidence_summary}

## Missing points reported by the Reviewer
{missing_aspects}

## Conflicts found (background only; do not use to cover missing points)
{conflicts}

## Output format (JSON only, no extra text)
{{
  "research_questions": [
    {{
      "question_id": "Q{n}",
      "question": "Investigation goal targeting the missing point (no conclusion)",
      "doc_hints": ["keyword"]
    }}
  ]
}}

## Rules
1. Generate questions only for the missing points; do not repeat existing questions.
2. Each question still describes only "what to investigate", without conclusions.
3. question_id numbering continues sequentially from Q{n}.
4. Generate at most 3 new investigation questions per call; do not pad with irrelevant questions."""


def _extract_json_object(text: str) -> dict | None:
    """稳健提取 JSON 对象（去围栏/去噪/外层花括号）。"""
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = lines[1:] if lines and lines[0].startswith("```") else lines
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group(0).strip()
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _deserialize_questions(questions: list[Any]) -> list[ResearchQuestion]:
    out: list[ResearchQuestion] = []
    for i, item in enumerate(questions, 1):
        if not isinstance(item, dict):
            continue
        out.append(ResearchQuestion(
            question_id=item.get("question_id") or f"Q{i}",
            question=str(item.get("question", "")).strip(),
            doc_hints=[str(h) for h in (item.get("doc_hints") or [])],
            status=QuestionStatus.PENDING,
        ))
    return [q for q in out if q.question]


class Planner:
    def __init__(self, policy) -> None:
        self.policy = policy

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------
    @trace_chain(name="planner.initial_plan", tags=["contract", "planner"])
    def initial_plan(self, question: str) -> list[ResearchQuestion]:
        """首次规划：拆解调查要点。"""
        set_agent("planner/initial_plan")
        prompt = INITIAL_PLAN_PROMPT.format(question=question)
        questions = self._call(prompt)
        if questions:
            return questions
        # 降级：直接把原问题当作唯一调查目标
        return [ResearchQuestion(question_id="Q1", question=question)]

    @trace_chain(name="planner.incremental_plan", tags=["contract", "planner"])
    def incremental_plan(self, state: ResearchState) -> list[ResearchQuestion]:
        """增量规划：只补缺失要点。"""
        set_agent("planner/incremental_plan")
        next_seq = len(state.questions) + 1
        completed = "\n".join(
            f"- Q{q.question_id}: {q.question}"
            for q in state.questions
            if q.question_id in state.completed_question_ids or q.question_id in state.active_question_ids
        ) or "(none)"
        evidence_summary = self._evidence_summary(state)
        missing = "\n".join(
            f"- {m.description} (reason: {m.reason})"
            for m in state.missing_aspects
        ) or "(none)"
        conflicts = "\n".join(
            f"- {c.summary} (E{c.evidence_a_id} vs E{c.evidence_b_id})"
            for c in state.conflicts
        ) or "(none)"

        prompt = INCREMENTAL_PLAN_PROMPT.format(
            question=state.original_question,
            completed_questions=completed,
            evidence_summary=evidence_summary,
            missing_aspects=missing,
            conflicts=conflicts,
            n=next_seq,
        )
        questions = self._call(prompt, start_seq=next_seq)
        if questions:
            return questions
        # 降级：把缺失要点直接转成调查问题（保证有进展）；也受每轮 3 个上限约束
        fallback = []
        for i, m in enumerate(state.missing_aspects[:MAX_QUESTIONS_PER_CALL], start=next_seq):
            fallback.append(ResearchQuestion(
                question_id=f"Q{i}",
                question=f"Find clauses relevant to: {m.description}",
                doc_hints=[m.description[:20]],
            ))
        return fallback

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _evidence_summary(self, state: ResearchState) -> str:
        lines = []
        for qid in state.completed_question_ids:
            ids = state.evidence_by_question.get(qid, [])
            lines.append(f"- {qid}: {len(ids)} evidence items")
        return "\n".join(lines) or "(no evidence yet)"

    def _call(self, prompt: str, start_seq: int = 1) -> list[ResearchQuestion]:
        messages = [
            {"role": "system", "content": "You are a contract research planning assistant. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ]
        append_token_usage(estimate_messages_tokens(messages))  # 计入本轮全链路 token 账本
        try:
            response = self.policy(messages)
        except RuntimeError as e:
            raise PlanParseError(f"LLM call failed during contract planning: {e}") from e
        content = response.get("content", "") or ""
        data = _extract_json_object(content)
        if data is None:
            return []
        questions = _deserialize_questions(data.get("research_questions", []))
        # 统一从 start_seq 起连续编号，屏蔽 LLM 自编号的跳号/重复
        for i, q in enumerate(questions, start=start_seq):
            q.question_id = f"Q{i}"
        # 每轮最多 MAX_QUESTIONS_PER_CALL 个，超出部分直接丢弃
        return questions[:MAX_QUESTIONS_PER_CALL]
