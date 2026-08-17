"""EvidenceAssembler：把 Worker 命中的碎片 Chunk 恢复成可引用的完整原始条款。

核心约束（见方案）：
- “整理” = 从原始文档中扩展、拼接、截取连续原文，绝不让 LLM 改写。
- quote 一律从 DB full_text[start:end] 截取；偏移来自候选（LLM 上报 get_section/search 结果），
  非法时回退为 source_chunk_ids 的并集区间。
"""
from __future__ import annotations

from typing import Optional

from .schemas import Evidence
from .tools import DocumentToolkit


class EvidenceAssembler:
    """将结构化证据候选物化为 Evidence（quote = 原始连续文本）。"""

    def __init__(self, toolkit: DocumentToolkit):
        self.toolkit = toolkit

    def materialize(self, cand: dict, question_id: str) -> Optional[Evidence]:
        """把一个候选 dict 物化为 Evidence；无法定位原文时返回 None。

        cand 至少需要 doc_id；start_offset/end_offset 可来自 get_section 的返回，
        source_chunk_ids 用于回退与校验。
        """
        doc_id = str(cand.get("doc_id", "")).strip()
        if not doc_id:
            return None

        full = self.toolkit.get_full_text(doc_id)
        if not full:
            return None

        start, end = self._resolve_offsets(cand, full)
        if start is None or end is None:
            return None

        chunk_ids = [str(c) for c in (cand.get("source_chunk_ids") or []) if str(c).strip()]
        if not chunk_ids:
            # 无切片记录则重新由区间反查（尽力而为）
            chunk_ids = self._chunks_in_span(doc_id, start, end)

        page_no = int(cand.get("page_no") or 0)
        if page_no <= 0:
            page_no = self.toolkit.get_page_for_span(doc_id, start, end)

        doc = self.toolkit.get_document(doc_id) or {}

        return Evidence(
            question_id=question_id,
            document_id=doc_id,
            document_name=doc.get("title", ""),
            section_path=list(cand.get("section_path") or []),
            page_no=page_no,
            source_chunk_ids=chunk_ids,
            start_offset=start,
            end_offset=end,
            quote=full[start:end],
            retrieval_score=float(cand.get("retrieval_score") or 0.0),
            relevance_note=str(cand.get("relevance_note") or "")[:500],
        )

    def _resolve_offsets(self, cand: dict, full: str) -> tuple[int | None, int | None]:
        """按优先级确定原文区间：[显式偏移] → [切片并集] → [回退无]。"""
        s = cand.get("start_offset")
        e = cand.get("end_offset")
        try:
            s, e = int(s), int(e)
        except (TypeError, ValueError):
            s, e = None, None
        if (
            s is not None and e is not None
            and 0 <= s <= e <= len(full)
            and full[s:e].strip()
        ):
            return s, e

        # 回退：把 source_chunk_ids 的 charspan 并起来
        spans: list[tuple[int, int]] = []
        for cid in (cand.get("source_chunk_ids") or []):
            chunk = self.toolkit.get_chunk(str(cid))
            if not chunk:
                continue
            cs = chunk.get("charspan") or []
            if len(cs) == 2:
                try:
                    spans.append((int(cs[0]), int(cs[1])))
                except (TypeError, ValueError):
                    continue
        if not spans:
            return None, None
        lo = min(a for a, _ in spans)
        hi = max(b for _, b in spans)
        if 0 <= lo <= hi <= len(full) and full[lo:hi].strip():
            return lo, hi
        return None, None

    def _chunks_in_span(self, doc_id: str, start: int, end: int) -> list[str]:
        """区间反查切片 ID（排序稳定）。"""
        from src.retrieval.store import connect

        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM chunks
                WHERE doc_id = %s AND NOT (charspan[2] <= %s OR charspan[1] >= %s)
                ORDER BY charspan[1], chunk_index
                """,
                (doc_id, start, end),
            )
            return [r[0] for r in cur.fetchall()]
