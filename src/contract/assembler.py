"""EvidenceAssembler：把 Worker 上报的命中切片聚合为可引用的完整原始条款。

证据单元 = 最末级 section。候选只携带 source_chunk_ids + relevance_note，
不携带任何偏移 / 章节路径 / 页码 / 分数：
1. 所有命中切片必须属于同一文档、同一 section_path（含全空）；
2. 优先整章聚合：get_section 成功且文本不超过 MAX_EVIDENCE_SECTION_TOKENS 时，
   区间 = 整章区间、证据切片 = 整章切片；
3. 整章超限或 section_path 为空时回退为命中切片并集，回退前做零空洞连续性校验；
4. quote 一律从 DB full_text[start:end] 截取，模型无权改写。
"""
from __future__ import annotations

from typing import Optional

from src.retrieval import config as _retrieval_config

from .schemas import Evidence
from .tools import DocumentToolkit
from ..utils.tokens import estimate_tokens


class EvidenceAssembler:
    """将结构化证据候选物化为 Evidence（quote = 原始连续文本）。"""

    def __init__(self, toolkit: DocumentToolkit):
        self.toolkit = toolkit

    def materialize(self, cand: dict, question_id: str) -> Optional[Evidence]:
        """把一个候选（source_chunk_ids + relevance_note）物化为 Evidence；无法定位时返回 None。"""
        raw_chunk_ids = cand.get("source_chunk_ids")
        if not isinstance(raw_chunk_ids, list):
            return None
        chunk_ids = list(dict.fromkeys(
            str(c).strip() for c in raw_chunk_ids if str(c).strip()
        ))
        if not chunk_ids:
            return None

        # 逐片取回；任一不存在即拒绝
        chunks = []
        for cid in chunk_ids:
            chunk = self.toolkit.get_chunk(cid)
            if not chunk:
                return None
            chunks.append(chunk)

        # 文档一致性
        doc_id = str(chunks[0].get("doc_id", "")).strip()
        if not doc_id or any(str(c.get("doc_id", "")).strip() != doc_id for c in chunks[1:]):
            return None

        full = self.toolkit.get_full_text(doc_id)
        if not full:
            return None

        # 候选切片的区间必须各自合法，不允许倒置或越界区间被并集掩盖。
        if any(self._chunk_span(c, len(full)) is None for c in chunks):
            return None

        # 章节一致性：所有切片 section_path 必须完全相等（含全空）
        paths = [list(c.get("section_path") or []) for c in chunks]
        if any(p != paths[0] for p in paths[1:]):
            return None
        section_path = list(paths[0])

        # 整章聚合优先
        if section_path:
            sec = self.toolkit.get_section(doc_id, section_path)
            if sec is not None:
                validated = self._validate_section(sec, doc_id, section_path, full)
                if validated is None:
                    return None
                start, end, section_text, section_chunk_ids = validated
            else:
                section_text, section_chunk_ids = "", []
            if sec is not None and estimate_tokens(section_text) <= _retrieval_config.MAX_EVIDENCE_SECTION_TOKENS:
                evidence_chunk_ids = section_chunk_ids
            else:
                fb = self._fallback_span(chunks)
                if fb is None:
                    return None
                start, end = fb
                evidence_chunk_ids = list(chunk_ids)
        else:
            fb = self._fallback_span(chunks)
            if fb is None:
                return None
            start, end = fb
            evidence_chunk_ids = list(chunk_ids)

        if not (0 <= start <= end <= len(full)) or not full[start:end].strip():
            return None

        page_no = self.toolkit.get_page_for_span(doc_id, start, end)
        doc = self.toolkit.get_document(doc_id) or {}

        return Evidence(
            question_id=question_id,
            document_id=doc_id,
            document_name=doc.get("title", ""),
            section_path=section_path,
            page_no=page_no,
            source_chunk_ids=evidence_chunk_ids,
            start_offset=start,
            end_offset=end,
            quote=full[start:end],
            retrieval_score=0.0,
            relevance_note=str(cand.get("relevance_note") or "")[:500],
        )

    @staticmethod
    def _chunk_span(chunk: dict, full_length: int) -> Optional[tuple[int, int]]:
        cs = chunk.get("charspan") or []
        if len(cs) != 2:
            return None
        try:
            lo, hi = int(cs[0]), int(cs[1])
        except (TypeError, ValueError):
            return None
        if not (0 <= lo < hi <= full_length):
            return None
        return lo, hi

    @staticmethod
    def _validate_section(
        sec: dict,
        doc_id: str,
        section_path: list[str],
        full: str,
    ) -> Optional[tuple[int, int, str, list[str]]]:
        """校验 get_section 的内部返回，防止错文档/错章节数据进入 Evidence。"""
        if str(sec.get("doc_id") or "").strip() != doc_id:
            return None
        if list(sec.get("section_path") or []) != section_path:
            return None
        chunk_ids = list(dict.fromkeys(
            str(x).strip() for x in (sec.get("chunk_ids") or []) if str(x).strip()
        ))
        if not chunk_ids:
            return None
        try:
            start = int(sec.get("start_offset"))
            end = int(sec.get("end_offset"))
        except (TypeError, ValueError):
            return None
        if not (0 <= start < end <= len(full)):
            return None
        text = sec.get("text")
        if not isinstance(text, str) or text != full[start:end]:
            return None
        return start, end, text, chunk_ids

    def _fallback_span(self, chunks: list[dict]) -> Optional[tuple[int, int]]:
        """回退路径：命中切片按 charspan[0] 排序，零空洞连续性校验后取并集区间。"""
        spans: list[tuple[int, int]] = []
        for c in chunks:
            cs = c.get("charspan") or []
            if len(cs) != 2:
                return None
            try:
                lo, hi = int(cs[0]), int(cs[1])
            except (TypeError, ValueError):
                return None
            if lo < 0 or hi <= lo:
                return None
            spans.append((lo, hi))
        spans.sort()
        for (prev_lo, prev_hi), (lo, hi) in zip(spans, spans[1:]):
            if prev_hi < lo:  # 空洞
                return None
        return spans[0][0], spans[-1][1]
