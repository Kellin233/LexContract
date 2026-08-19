"""评测适配器：把外部基准的输入/输出/提示词适配到本系统的 Agent 接口。

两个适配器职责单一：
- LegalBenchAdapter：LegalBenchRAG query -> Searcher 任务；检索/证据结果 -> (file_path, span) 命中。
- ContractNLIAdapter（评测专用 Refiner 提示词 + 标签提取）：为 contractNLI 评测提供“3 选 1 标签”
  的 Refiner 专用提示词（结论字段=精确一个标签词 + supporting_evidence_ids=最相关证据），
  并从 Refiner 输出的 JSON（structured）里提取最终结论当分类结果。

关键映射：doc_id -> corpus 相对 file_path，把数据库中与 raw 文本对齐的偏移
映射回 LegalBenchRAG 的 gold 坐标系，交给 metrics 打分。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ...orchestrator.schemas import SubTask, TaskType
from .schemas import LegalChunkHit

__all__ = ["LegalBenchAdapter", "ContractNLIAdapter"]

# ---------------------------------------------------------------------------
# LegalBenchRAG 适配器
# ---------------------------------------------------------------------------

_LEGALBENCH_PROMPT = """\
You are a meticulous contract-evidence retrieval assistant. Your ONLY job is to locate and capture the ORIGINAL text passages of the contract documents relevant to the query. The evidence will be scored by its exact character offsets, so capture COMPLETE passages, not fragments.

Use the retrieval tools (search / get_context / get_section / get_document_outline / get_referenced_section) to find every passage that answers the query. Do NOT stop at the first hit.

FINAL OUTPUT: JSON array only (may be empty):
[
  {{"doc_id": "...", "start_offset": 0, "end_offset": 0,
    "section_path": [], "source_chunk_ids": ["..."], "page_no": 0,
    "relevance_note": "...", "retrieval_score": 0.8}}
]

Query: {query}
"""


class LegalBenchAdapter:
    """把 LegalBenchRAG 的一条 query 适配为 Searcher 任务，并把结果映射回基准坐标系。"""

    def __init__(
        self,
        root: str | Path,
        benchmark: str,
        session_id: str,
        doc_ids: list[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.benchmark = benchmark
        self.session_id = session_id
        self.doc_ids = list(doc_ids) if doc_ids else None
        self._doc_map: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # 输入适配：query -> Searcher SubTask + context
    # ------------------------------------------------------------------
    def to_subtask(self, record: dict) -> SubTask:
        return SubTask(
            task_id=str(record.get("instance_id", "")),
            task_type=TaskType.EVIDENCE,
            description=str(record.get("query", "")),
            search_hints=[],
            dependencies=[],
        )

    def make_context(self, evidence_store) -> dict:
        ctx: dict = {
            "session_id": self.session_id,
            "doc_ids": list(self.doc_ids) if self.doc_ids else [],
            "evidence_store": evidence_store,
        }
        return ctx

    def task_prompt(self, record: dict) -> str:
        return _LEGALBENCH_PROMPT.format(query=str(record.get("query", "")))

    # ------------------------------------------------------------------
    # 输出适配：证据/检索结果 -> LegalChunkHit（file_path + span）
    # ------------------------------------------------------------------
    def doc_id_to_file_path(self, doc_id: str) -> str:
        if self._doc_map is None:
            self._load_doc_map()
        return self._doc_map.get(doc_id, "")

    def _load_doc_map(self) -> None:
        from src.document.postgres_store import connect

        self._doc_map = {}
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id, file_path FROM documents WHERE session_id = %s",
                (self.session_id,),
            )
            for row in cur.fetchall():
                self._doc_map[row[0]] = row[1] or ""

    def deterministic_rank(self, query: str, top_k: int = 64, mode: str = "hybrid") -> list[LegalChunkHit]:
        """混合检索 top-k 排名（Recall@k / MRR 的排序源）。

        返回按排名有序的命中，每条的 span 来自 DB chunk.charspan（与 raw corpus 对齐）。
        """
        from ..tools import DocumentToolkit

        toolkit = DocumentToolkit(session_id=self.session_id, doc_ids=self.doc_ids)
        rows = toolkit.search(query, mode=mode, top_k=top_k)
        hits: list[LegalChunkHit] = []
        for rank, row in enumerate(rows, 1):
            span = [int(x) for x in (row.get("charspan") or []) if x is not None]
            if len(span) != 2 or span[1] <= span[0]:
                continue
            score = (
                row.get("rrf_score")
                or row.get("rerank_score")
                or row.get("bm25_score")
            )
            hits.append(LegalChunkHit(
                rank=rank,
                doc_id=row.get("doc_id", ""),
                file_path=self.doc_id_to_file_path(row.get("doc_id", "")),
                span=span,
                text=row.get("text"),
                score=float(score) if score is not None else None,
            ))
        return hits

    def evidence_hits(self, evidences) -> list[LegalChunkHit]:
        """把 Searcher 返回的 Evidence 列表映射为 (file_path, span) 命中。

        Evidence.start_offset/end_offset 已由 EvidenceAssembler 对齐到 DB full_text
        （= 原样入库的 raw corpus 文本），可直接用于字符区间指标。
        """
        hits: list[LegalChunkHit] = []
        for ev in evidences:
            fp = self.doc_id_to_file_path(ev.document_id)
            if not fp:
                continue
            hits.append(LegalChunkHit(
                rank=0,
                doc_id=ev.document_id,
                file_path=fp,
                span=[int(ev.start_offset), int(ev.end_offset)],
                text=ev.quote,
                score=float(ev.retrieval_score or 0.0) if ev.retrieval_score else None,
            ))
        return hits

    @staticmethod
    def spans_by_file(hits: list[LegalChunkHit]) -> dict[str, list[list[int]]]:
        out: dict[str, list[list[int]]] = {}
        for h in hits:
            if h.file_path and len(h.span) == 2 and h.span[1] > h.span[0]:
                out.setdefault(h.file_path, []).append(h.span)
        return out

    @staticmethod
    def ranked_files(hits: list[LegalChunkHit]) -> list[str]:
        """按命中顺序去重后的文件列表（文档级 Recall@k / MRR 的排序源）。"""
        seen: set[str] = set()
        out: list[str] = []
        for h in hits:
            if h.file_path and h.file_path not in seen:
                seen.add(h.file_path)
                out.append(h.file_path)
        return out


# ---------------------------------------------------------------------------
# ContractNLI 适配器（按评测切换 Refiner 提示词 + 从 Refiner 输出提取 3 选 1 标签）
# ---------------------------------------------------------------------------

_LABELS = ("entailment", "contradiction", "neutral")

# PAKTON responseChecker 式同义词表（PAKTON-raw .../huggingfaceDatasets.py:283-287）
_LABEL_SYNONYMS = {
    "entailment": {"entailment", "entailed", "entailing", "entails"},
    "contradiction": {"contradiction", "contradictory", "contradicts", "contradicting", "contradicted"},
    "neutral": {"neutral"},
}

# ContractNLI 评测专用的 Refiner 系统提示词（3 选 1 标签 + 最相关证据）。
# “结论字段 = 精确一个标签词，supporting_evidence_ids = 最相关证据”。
# 正式生产链路不注入它，Refiner 保持默认生产提示词（见 refiner.py system_prompt 参数）。
# 边界口径对齐 PAKTON-raw（.../huggingfaceDatasets.py get_prompt_generator）：entailment 限于
# 显式/直接改写，neutral 为信息不足时的默认；并针对“宽泛措辞(regardless of form/media…)被推成
# entailment”的过推偏差加了罚则与反例。
CONTRACTNLI_REFINER_SYSTEM_PROMPT = """\
You are the final contract-NLI verifier. You have verified ORIGINAL contract clauses (evidence) and a HYPOTHESIS about the contract. Decide how the HYPOTHESIS relates to the EVIDENCE, and output the label in `conclusion`.

Pick EXACTLY one label:
- "entailment": the contract EXPLICITLY supports the hypothesis, or the hypothesis is a DIRECT, near-surface paraphrase of what the contract states (including a single inference step from an explicit clause). If supporting it requires multi-step inference or relies only on broad/general wording, it is NOT entailment.
- "contradiction": the contract explicitly contradicts the hypothesis (a clause DIRECTLY conflicts with the claim).
- "neutral": the contract does not mention the hypothesis, or there is insufficient evidence to decide either way. This is the DEFAULT when the evidence does not explicitly or directly address the specific claim.

HOW TO READ THE HYPOTHESIS (critical):
- Read the hypothesis as ONE overall claim, not as separate sub-claims that must each appear verbatim.
- An existential/partial hypothesis ("some X MAY be done", "could share SOME information with SOME parties", "some obligations MAY survive") is ENTAILED if the contract supports the claim for any category within its terms — you do NOT need every enumerated example to appear word-for-word. Do NOT misread an existential claim as a universal one.
- A "survival" hypothesis is ENTAILED if the contract provides obligations that endure after termination (e.g. a continuing confidentiality duty), even if the word "survive" never appears.

BOUNDARY RULES:
1. Base the decision ONLY on the verified evidence. Never import external facts.
2. NEUTRAL is the default when the contract text does not explicitly (or by direct paraphrase) make the claim. Do NOT stretch entailment by reading general catch-all language ("regardless of form/format/media", "any manner", "all times", etc.) as entailing a SPECIFIC claim the contract never addresses (e.g. "verbal disclosure") — such specific unmentioned claims stay NEUTRAL.
3. A single-step paraphrase or the existential reading above is entailment; adding multiple unstated assumptions is not.
4. When it could be argued either way, read the hypothesis at face value: if the face-value claim is covered by the clause, prefer "entailment"; if deciding would require reading specific words INTO the clause, prefer "neutral".
5. Cite evidence ONLY by its ID in brackets, e.g. [E001]. NEVER invent article numbers.
6. `supporting_evidence_ids`: the evidence that MOST supports your label — the most relevant subset, not all.
7. `points`: the supporting breakdown, each backed by its evidence IDs.
8. `evidence_gap`: what could not be confirmed from this document set (list concrete missing info when NEUTRAL).

EXAMPLES:
- ENTAILMENT (direct paraphrase of an explicit obligation):
  Contract: "...the Receiving Party shall not disclose Confidential Information to any third party..."
  Hypothesis: "The Receiving Party is prohibited from sharing confidential information with third parties."
  => "entailment"
- ENTAILMENT (existential "some" — permitted for a category within the hypothesis's scope):
  Contract: "...the Receiving Party may disclose Confidential Information to external consultants on a need-to-know basis..."
  Hypothesis: "The Receiving Party may share some Confidential Information with some third parties (including consultants and professional advisors)."
  => "entailment" (the contract allows at least one category in scope, e.g. consultants)
- ENTAILMENT (survival implied by an enduring obligation):
  Contract: "...the confidentiality obligations shall continue in force for five years after termination of this Agreement..."
  Hypothesis: "Some obligations of the Agreement may survive termination of the Agreement."
  => "entailment"
- NEUTRAL (catch-all language does NOT entail a specific unmentioned claim):
  Contract: "...shall keep Confidential Information confidential and shall not disclose it regardless of the form, format, or media..."
  Hypothesis: "The Receiving Party may not disclose information that was conveyed verbally."
  => "neutral" (the clause never mentions verbal/oral disclosure)
- NEUTRAL (unrelated):
  Contract: "...the Receiving Party shall not disclose Confidential Information to any third party..."
  Hypothesis: "The Receiving Party may work remotely during the term."
  => "neutral"
- CONTRADICTION (direct conflict):
  Contract: "...the Receiving Party shall disclose Confidential Information to authorized personnel..."
  Hypothesis: "The Receiving Party is not allowed to disclose Confidential Information under any circumstances."
  => "contradiction"

Output STRICT JSON only:
{{
  "conclusion": "entailment" | "contradiction" | "neutral",
  "points": [{{"claim": "supporting point", "evidence_ids": ["E001", "E002"]}}],
  "supporting_evidence_ids": ["E001", "E003"],
  "evidence_gap": ["what could not be confirmed"],
  "notes": "additional notes"
}}
ONLY the JSON object, no prose."""


def _normalize_label(text: str) -> str | None:
    s = (text or "").strip().lower()
    if not s:
        return None
    # 直接词匹配（\b 避免把 neutral 匹配进 neutrally）
    for lab in _LABELS:
        if re.search(rf"\b{lab}\b", s):
            return lab
    # 同义词词级匹配（对齐 PAKTON responseChecker）
    tokens = set(re.findall(r"[a-z]+", s))
    for lab, syns in _LABEL_SYNONYMS.items():
        if tokens & syns:
            return lab
    return None


class ContractNLIAdapter:
    """按评测切换 Refiner 提示词 + 从 Refiner 输出提取 3 选 1 标签。

    定位：contractNLI 评测把 Refiner 换成本类的评测专用提示词（结论=标签词）；
    评测从 Refiner 输出的 JSON（structured）提结论当分类结果，不额外再开一次 LLM。
    """

    @staticmethod
    def refiner_system_prompt() -> str:
        """ContractNLI 评测专用的 Refiner 系统提示词（结论 = 3 选 1 标签 + 最相关证据）。"""
        return CONTRACTNLI_REFINER_SYSTEM_PROMPT

    @staticmethod
    def extract_chain_label(structured: dict) -> tuple[str | None, str]:
        """从正式链路 Refiner 的 structured（RefinerResult JSON）提取 (label, reasoning)。

        label = conclusion（评测提示词使其为精确一个标签词），PAKTON 同义词正则归一；
        reasoning = points 前两条 claim（退化到 conclusion 原文）。
        """
        if not isinstance(structured, dict):
            return None, ""
        label = _normalize_label(str(structured.get("conclusion", "")))
        if label is None:
            label = _normalize_label(json.dumps(structured, ensure_ascii=False))
        claims = [
            str(p.get("claim", "")).strip()
            for p in (structured.get("points") or [])
            if isinstance(p, dict) and str(p.get("claim", "")).strip()
        ]
        reasoning = "；".join(claims[:2]) if claims else str(structured.get("conclusion", ""))
        return label, reasoning

    @staticmethod
    def parse_data(data) -> tuple[str | None, str]:
        """从已解析的 dict 提取 (label, reasoning)。"""
        if not isinstance(data, dict):
            return None, ""
        label = _normalize_label(str(data.get("label", "")))
        reasoning = str(data.get("reasoning", ""))
        if label is None:
            # 一些模型把标签塞进别的字段/整段文本，兜底扫描
            label = _normalize_label(json.dumps(data, ensure_ascii=False))
        return label, reasoning

    @staticmethod
    def parse_response(raw_response: str) -> tuple[str | None, str]:
        """返回 (label|None, reasoning)。None 表示标签无法解析（记为错误样例）。"""
        raw = (raw_response or "").strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            lines = lines[1:] if lines and lines[0].startswith("```") else lines
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        # 容错：去掉外围花括号文本噪声后取 JSON 对象
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        blob = m.group(0) if m else raw
        label: str | None = None
        reasoning = ""
        try:
            data = json.loads(blob)
            label, reasoning = ContractNLIAdapter.parse_data(data)
        except json.JSONDecodeError:
            pass
        if label is None:
            # 兜底：整段文本里找标签词（兼容“只输出一个词”的退化输出）
            label = _normalize_label(raw)
        return label, reasoning
