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
from ..utils.tracing import trace_chain


__all__ = ["Planner", "PlanParseError"]


class PlanParseError(Exception):
    pass


INITIAL_PLAN_PROMPT = """\
你是一名合同条款调查规划员。请把用户的原始问题拆解为若干个“需要调查的要点”（research questions）。

## 原始问题
{question}

## 输出格式（只输出 JSON，无多余文字）
{{
  "research_questions": [
    {{
      "question_id": "Q1",
      "question": "调查目标（只描述需要去合同里找什么，绝不包含结论）",
      "doc_hints": ["关键词1", "关键词2"]
    }}
  ]
}}

## 规则
1. 每个要点只描述“需要调查什么”，例如“查找乙方主动提前终止的通知期限条款”，不要写“乙方可以提前终止/不可以提前终止”这类结论。
2. 一个问题往往要同时结合多个章节才能回答，因此要点应覆盖所有相关角度（正反情形、例外、交叉引用、后续义务等）。
3. 生成 3-8 个要点；要点必须与原始问题直接相关。
4. 不要生成重复要点。
5. doc_hints 使用合同原文常见的术语（如“终止”“解除”“单方解除”“提前终止”“不可抗力”“违约责任”“通知期限”）。""" 

INCREMENTAL_PLAN_PROMPT = """\
你是一名合同条款调查规划员。前面几轮调查尚未覆盖全部要点，请你补充新的调查问题。

## 原始问题
{question}

## 已完成的调查问题（不要再重复）
{completed_questions}

## 已收集的证据概要
{evidence_summary}

## Reviewer 指出的缺失要点
{missing_aspects}

## 发现的冲突（仅作背景，不要用于覆盖缺失点）
{conflicts}

## 输出格式（只输出 JSON，无多余文字）
{{
  "research_questions": [
    {{
      "question_id": "Q{n}",
      "question": "针对缺失要点的调查目标（不含结论）",
      "doc_hints": ["关键词"]
    }}
  ]
}}

## 规则
1. 只为缺失要点生成问题；与已有问题重复的不要生成。
2. 每个问题仍只描述“需要调查什么”，不包含结论。
3. question_id 从 Q{n} 开始连续编号。
4. 通常 1-4 个即可，不要为了凑数生成无关问题。"""


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
        prompt = INITIAL_PLAN_PROMPT.format(question=question)
        questions = self._call(prompt)
        if questions:
            return questions
        # 降级：直接把原问题当作唯一调查目标
        return [ResearchQuestion(question_id="Q1", question=question)]

    @trace_chain(name="planner.solve", tags=["contract", "planner", "nli"])
    def solve(self, prompt: str, system_prompt: str | None = None) -> dict | None:
        """通用执行：调用 LLM 并稳健解析 JSON（供 ContractNLI 分类等评测复用）。

        - LLM 调用失败：抛 PlanParseError（由调用方决定如何记为错误样例）；
        - 输出无法解析出 JSON 对象：返回 None（调用方记为标签解析失败）。
        """
        messages = [
            {"role": "system", "content": system_prompt or "You are a legal text classification assistant. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.policy(messages)
        except RuntimeError as e:
            raise PlanParseError(f"LLM call failed: {e}") from e
        content = response.get("content", "") or ""
        return _extract_json_object(content)

    @trace_chain(name="planner.incremental_plan", tags=["contract", "planner"])
    def incremental_plan(self, state: ResearchState) -> list[ResearchQuestion]:
        """增量规划：只补缺失要点。"""
        next_seq = len(state.questions) + 1
        completed = "\n".join(
            f"- Q{q.question_id}: {q.question}"
            for q in state.questions
            if q.question_id in state.completed_question_ids or q.question_id in state.active_question_ids
        ) or "（无）"
        evidence_summary = self._evidence_summary(state)
        missing = "\n".join(
            f"- {m.description}（原因：{m.reason}）"
            for m in state.missing_aspects
        ) or "（无）"
        conflicts = "\n".join(
            f"- {c.summary}（E{c.evidence_a_id} vs E{c.evidence_b_id}）"
            for c in state.conflicts
        ) or "（无）"

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
        # 降级：把每个缺失要点直接转成调查问题（保证有进展）
        fallback = []
        for i, m in enumerate(state.missing_aspects, start=next_seq):
            fallback.append(ResearchQuestion(
                question_id=f"Q{i}",
                question=f"查找与“{m.description}”相关的条款",
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
            lines.append(f"- {qid}: {len(ids)} 条证据")
        return "\n".join(lines) or "（尚无证据）"

    def _call(self, prompt: str, start_seq: int = 1) -> list[ResearchQuestion]:
        messages = [
            {"role": "system", "content": "You are a contract research planning assistant. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ]
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
        return questions
