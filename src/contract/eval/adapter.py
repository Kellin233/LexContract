"""评测适配器：把外部基准的输入/输出/提示词适配到本系统的 Agent 接口。

两个适配器职责单一：
- LegalBenchAdapter：LegalBenchRAG query -> Searcher 任务；检索/证据结果 -> (file_path, span) 命中。
- ContractNLIAdapter：(premise, hypothesis) -> 分类 prompt；原始输出 -> 标签。

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
# ContractNLI 适配器
# ---------------------------------------------------------------------------

_LABELS = ("entailment", "contradiction", "neutral")

CONTRACTNLI_PROMPT = """\
You are a legal expert. Judge whether the HYPOTHESIS is entailed by, contradicted by, or neutral with respect to the CONTRACT PREMISE.

- "entailment": the premise necessarily implies the hypothesis.
- "contradiction": the premise necessarily contradicts the hypothesis.
- "neutral": neither entailment nor contradiction can be determined.

Output STRICT JSON only:
{{"label": "entailment" | "contradiction" | "neutral", "reasoning": "one sentence"}}

## CONTRACT PREMISE
{premise}

## HYPOTHESIS
{hypothesis}
"""


def _normalize_label(text: str) -> str | None:
    s = (text or "").strip().lower()
    for lab in _LABELS:
        if s == lab or s.startswith(lab) or lab in s:
            return lab
    return None


class ContractNLIAdapter:
    """把 (premise, hypothesis) 适配为分类 prompt，并把输出解析为标签。"""

    @staticmethod
    def build_prompt(premise: str, hypothesis: str) -> str:
        return CONTRACTNLI_PROMPT.format(premise=premise, hypothesis=hypothesis)

    @staticmethod
    def system_prompt() -> str:
        return "You are a legal text classification assistant. Output valid JSON only."

    @staticmethod
    def parse_data(data) -> tuple[str | None, str]:
        """从已解析的 dict（如 Planner.solve 的返回值）提取 (label, reasoning)。"""
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
