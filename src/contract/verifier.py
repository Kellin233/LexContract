"""CitationVerifier：证据原文完整性校验器（纯程序化，不依赖 LLM）。

校验点：
1. quote 是否与 DB full_text[start:end] 完全一致；
2. 区间是否落在合法字符范围内；
3. source_chunk_ids 是否都属于该文档/章节，且 chunk 正文与 charspan 对齐、并集覆盖 [start, end]。
只回答“证据确实来自原始文档”，不做任何事实判断。

第 3 点用“覆盖率”判定而不是旧版的“内部零空洞”：chunk 切分偶尔在边界留 1 字符
小缝，不该因此丢弃整条定位正确的证据。要求并集覆盖 [s,e] ≥98% 且两端落在并集内
（±1 字符容差），同时保持对“编造的 chunk id / 别文档的 chunk”零容忍。
校验失败时在 evidence.verify_note 写机器可读原因，便于在评测记录里统计丢弃分布。
"""
from __future__ import annotations

from .schemas import Evidence
from .tools import DocumentToolkit

# 引用切片并集对 [s,e] 的最小覆盖率；低于它视为真实空洞
_COVERAGE_MIN = 0.98
# 证据端点允许不在引用切片并集内的字符容差（吸收边界舍入差异）
_ENDPOINT_TOL = 1


class CitationVerifier:
    def __init__(self, toolkit: DocumentToolkit):
        self.toolkit = toolkit

    def verify(self, evidence: Evidence) -> bool:
        """校验通过返回 True 并清空 verify_note；否则 False 并把原因写入 verify_note。"""
        ok, note = self._check(evidence)
        evidence.verified = ok
        evidence.verify_note = "" if ok else note
        return ok

    def _check(self, evidence: Evidence) -> tuple[bool, str]:
        full = self.toolkit.get_full_text(evidence.document_id)
        if not full:
            return False, "no-full-text"
        s, e = evidence.start_offset, evidence.end_offset
        try:
            s, e = int(s), int(e)
        except (TypeError, ValueError):
            return False, "offsets-not-int"
        if not (0 <= s <= e <= len(full)):
            return False, f"offsets-out-of-range [{s},{e}] len={len(full)}"
        # 1) 原文精确一致
        if full[s:e] != evidence.quote:
            return False, "quote-mismatch"

        # 2) 切片归属与区间覆盖：先裁到 [s,e]，再并集求覆盖率
        clipped: list[tuple[int, int]] = []
        expected_path = list(evidence.section_path or [])
        for cid in evidence.source_chunk_ids:
            chunk = self.toolkit.get_chunk(str(cid))
            if not chunk:
                return False, f"missing-chunk {cid}"
            if chunk.get("doc_id") != evidence.document_id:
                return False, f"off-doc-chunk {cid}"
            if list(chunk.get("section_path") or []) != expected_path:
                return False, "off-section-chunk"
            cs = chunk.get("charspan") or []
            if len(cs) != 2:
                return False, "invalid-chunk-span"
            try:
                raw_lo, raw_hi = int(cs[0]), int(cs[1])
            except (TypeError, ValueError):
                return False, "invalid-chunk-span"
            if not (0 <= raw_lo < raw_hi <= len(full)):
                return False, "invalid-chunk-span"
            if not isinstance(chunk.get("text"), str) or full[raw_lo:raw_hi] != chunk["text"]:
                return False, "chunk-text-mismatch"
            lo, hi = max(raw_lo, s), min(raw_hi, e)
            if hi > lo:
                clipped.append((lo, hi))
        if not clipped:
            return False, "chunks-no-overlap"

        clipped.sort()
        covered = 0
        cur_lo, cur_hi = clipped[0]
        for lo, hi in clipped[1:]:
            if lo <= cur_hi:
                cur_hi = max(cur_hi, hi)
            else:
                covered += cur_hi - cur_lo
                cur_lo, cur_hi = lo, hi
        covered += cur_hi - cur_lo

        total = e - s
        ratio = covered / total if total > 0 else 0.0
        if ratio + 1e-9 < _COVERAGE_MIN:
            return False, f"low-coverage {ratio:.3f} gap={total - covered}"

        # 端点容差：两端必须落在引用切片并集内（±1 字符），杜绝“只盖中间”的伪覆盖
        if not any(a - _ENDPOINT_TOL <= s <= b + _ENDPOINT_TOL for a, b in clipped):
            return False, f"start-outside s={s}"
        if not any(a - _ENDPOINT_TOL <= e <= b + _ENDPOINT_TOL for a, b in clipped):
            return False, f"end-outside e={e}"
        return True, ""
