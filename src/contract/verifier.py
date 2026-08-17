"""CitationVerifier：证据原文完整性校验器（纯程序化，不依赖 LLM）。

校验点：
1. quote 是否与 DB full_text[start:end] 完全一致；
2. 区间是否落在合法字符范围内；
3. source_chunk_ids 是否都属于该文档，且其 charspan 恰好覆盖 [start, end]（无空洞）。
只回答“证据确实来自原始文档”，不做任何事实判断。
"""
from __future__ import annotations

from .schemas import Evidence
from .tools import DocumentToolkit


class CitationVerifier:
    def __init__(self, toolkit: DocumentToolkit):
        self.toolkit = toolkit

    def verify(self, evidence: Evidence) -> bool:
        """校验通过返回 True，并把 evidence.verified 置 True；否则 False。"""
        ok = self._check(evidence)
        evidence.verified = ok
        return ok

    def _check(self, evidence: Evidence) -> bool:
        full = self.toolkit.get_full_text(evidence.document_id)
        if not full:
            return False
        s, e = evidence.start_offset, evidence.end_offset
        if not (0 <= s <= e <= len(full)):
            return False
        # 1) 原文精确一致
        if full[s:e] != evidence.quote:
            return False

        # 2) 切片归属与区间覆盖（source_chunk_ids 拼出的并集必须无空洞盖住 [s, e]）
        spans: list[tuple[int, int]] = []
        for cid in evidence.source_chunk_ids:
            chunk = self.toolkit.get_chunk(str(cid))
            if not chunk:
                return False
            if chunk.get("doc_id") != evidence.document_id:
                return False
            cs = chunk.get("charspan") or []
            if len(cs) == 2:
                try:
                    spans.append((int(cs[0]), int(cs[1])))
                except (TypeError, ValueError):
                    pass
        if not spans:
            return False

        spans.sort()
        pos = s
        for a, b in spans:
            if b <= pos:
                continue
            if a > pos:
                return False  # 出现空洞
            pos = max(pos, b)
            if pos >= e:
                return True
        # 允许约 1 个字符的容差（换行/字符跨度边界）
        return e - pos <= 1
