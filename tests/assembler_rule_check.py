"""Hermetic 验证：EvidenceAssembler 的最末级 section 自动聚合物化（不依赖 DB / LLM）。

候选只携带 source_chunk_ids + relevance_note，物化规则：
1. 缺 ids / 切片不存在 / 切片跨文档 / 切片跨 section → None；
2. 小 section（estimate_tokens <= MAX_EVIDENCE_SECTION_TOKENS）→ 整章聚合，
   区间与 chunk_ids 等于 get_section 返回值；
3. 大 section → 回退命中切片并集（连续性通过）；
4. 回退路径出现区间空洞 → None；
5. Evidence.retrieval_score 恒 0.0，section_path / page_no / quote 正确。

用法: python tests/assembler_rule_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.contract.assembler import EvidenceAssembler
from src.retrieval import config as retrieval_config


# 约 52000 字符、约 10000 词，保证大 section 超 token 上限
FULL = ("lorem ipsum dolor sit amet " * 2000)


def _mk_chunk(cid: str, doc_id: str, start: int, end: int, section_path: list[str]) -> dict:
    return {
        "id": cid,
        "text": FULL[start:end],
        "doc_id": doc_id,
        "doc_title": "Doc",
        "section_path": list(section_path),
        "page_no": start // 500 + 1,
        "charspan": [start, end],
        "source_format": "txt",
    }


class FakeToolkit:
    """只实现 EvidenceAssembler 用到的 toolkit 方法，全部为纯内存伪实现。"""

    def __init__(self, chunks: list[dict], sections: dict[tuple, dict]):
        self.chunks = {c["id"]: c for c in chunks}
        self.sections = sections

    def get_chunk(self, chunk_id: str) -> dict | None:
        return self.chunks.get(chunk_id)

    def get_full_text(self, doc_id: str) -> str:
        return FULL if doc_id == "doc1" else ""

    def get_document(self, doc_id: str) -> dict:
        return {"doc_id": doc_id, "title": "Doc", "source_format": "txt"}

    def get_section(self, doc_id: str, section_path: list[str]) -> dict | None:
        return self.sections.get(tuple(section_path))

    def get_page_for_span(self, doc_id: str, start: int, end: int) -> int:
        return start // 500 + 1


# 章节 S1（小，整章可聚合）：覆盖 [0, 2000)
# 章节 S2（大，超 token 上限）：覆盖 [2000, 52000)
S1 = ["Art 1"]
S2 = ["Art 2"]
CHUNKS = [
    _mk_chunk("c0", "doc1", 0, 500, S1),
    _mk_chunk("c1", "doc1", 500, 1000, S1),
    _mk_chunk("c2", "doc1", 1000, 1500, S1),
    _mk_chunk("c3", "doc1", 1500, 2000, S1),
    _mk_chunk("c4", "doc1", 2000, 3000, S2),
    _mk_chunk("c5", "doc1", 3000, 4000, S2),
    _mk_chunk("c6", "doc1", 4000, 5000, S2),
    _mk_chunk("c7", "doc1", 5000, len(FULL), S2),
    _mk_chunk("cX", "doc2", 0, 500, S1),
]
SECTIONS = {
    tuple(S1): {
        "doc_id": "doc1", "section_path": list(S1),
        "start_offset": 0, "end_offset": 2000,
        "text": FULL[0:2000], "chunk_ids": ["c0", "c1", "c2", "c3"],
    },
    tuple(S2): {
        "doc_id": "doc1", "section_path": list(S2),
        "start_offset": 2000, "end_offset": len(FULL),
        "text": FULL[2000:], "chunk_ids": ["c4", "c5", "c6", "c7"],
    },
}


def main() -> int:
    failed: list[str] = []
    tk = FakeToolkit(CHUNKS, SECTIONS)
    a = EvidenceAssembler(tk)

    # 预检：S1 应是小 section、S2 应是大 section
    from src.utils.tokens import estimate_tokens

    if estimate_tokens(FULL[0:2000]) > retrieval_config.MAX_EVIDENCE_SECTION_TOKENS:
        failed.append("测试数据失效：S1 应小于 token 上限")
    if estimate_tokens(FULL[2000:]) <= retrieval_config.MAX_EVIDENCE_SECTION_TOKENS:
        failed.append("测试数据失效：S2 应大于 token 上限")

    # 1. 缺 ids / 全空白 → None
    for bad in ({}, [], ["  "], None):
        if a.materialize({"source_chunk_ids": bad}, "q") is not None:
            failed.append("缺 ids 应返回 None")

    # 2. 切片不存在 → None
    if a.materialize({"source_chunk_ids": ["c0", "missing"]}, "q") is not None:
        failed.append("含不存在切片应返回 None")

    # 3. 跨文档 → None
    if a.materialize({"source_chunk_ids": ["c0", "cX"]}, "q") is not None:
        failed.append("跨文档切片应返回 None")

    # 4. 跨 section → None
    if a.materialize({"source_chunk_ids": ["c0", "c4"]}, "q") is not None:
        failed.append("跨 section 切片应返回 None")

    # 5. 小 section 整章聚合：候选只报 c0/c1，物化为整章 [0,2000) + 全部切片
    ev = a.materialize({"source_chunk_ids": ["c0", "c1"], "relevance_note": "note"}, "q")
    if ev is None:
        failed.append("小 section 聚合不应返回 None")
    else:
        if (ev.start_offset, ev.end_offset) != (0, 2000):
            failed.append(f"小 section 区间应 [0,2000)，实际 [{ev.start_offset},{ev.end_offset}]")
        if ev.source_chunk_ids != ["c0", "c1", "c2", "c3"]:
            failed.append(f"小 section 切片应为整章，实际 {ev.source_chunk_ids}")
        if ev.section_path != S1:
            failed.append(f"section_path 应为 {S1}，实际 {ev.section_path}")
        if ev.quote != FULL[0:2000]:
            failed.append("quote 应为整章连续原文")
        if ev.retrieval_score != 0.0:
            failed.append(f"retrieval_score 应恒 0.0，实际 {ev.retrieval_score}")
        if ev.page_no != 1:
            failed.append(f"page_no 应为 1，实际 {ev.page_no}")

    # 6. 大 section 回退：候选 c4/c5 连续 → 切片并集 [2000,4000)，切片保持候选原值
    ev = a.materialize({"source_chunk_ids": ["c4", "c5"]}, "q")
    if ev is None:
        failed.append("大 section 回退不应返回 None")
    else:
        if (ev.start_offset, ev.end_offset) != (2000, 4000):
            failed.append(f"大 section 回退区间应 [2000,4000)，实际 [{ev.start_offset},{ev.end_offset}]")
        if ev.source_chunk_ids != ["c4", "c5"]:
            failed.append(f"回退切片应保持候选原值，实际 {ev.source_chunk_ids}")
        if ev.quote != FULL[2000:4000]:
            failed.append("回退 quote 应为切片并集原文")
        if ev.section_path != S2:
            failed.append(f"回退 section_path 应为 {S2}，实际 {ev.section_path}")
        if ev.retrieval_score != 0.0:
            failed.append("回退 retrieval_score 应恒 0.0")

    # 7. 回退空洞拒绝：c4 [2000,3000) 与 c6 [4000,5000) 之间缺 c5
    if a.materialize({"source_chunk_ids": ["c4", "c6"]}, "q") is not None:
        failed.append("回退路径区间空洞应返回 None")

    # 8. relevance_note 截断 500
    ev = a.materialize({"source_chunk_ids": ["c0"], "relevance_note": "x" * 600}, "q")
    if ev is not None and len(ev.relevance_note) != 500:
        failed.append(f"relevance_note 应截断到 500，实际 {len(ev.relevance_note)}")

    if failed:
        print("FAIL")
        for f in failed:
            print(f"  - {f}")
        return 1

    print("PASS")
    print("  - 缺 ids / 切片缺失 / 跨文档 / 跨 section → None ✓")
    print("  - 小 section 整章聚合（区间 + 整章切片 + 完整 quote）✓")
    print("  - 大 section 回退切片并集（保持候选原值）✓")
    print("  - 回退空洞拒绝 ✓   retrieval_score 恒 0.0 ✓   relevance_note 截断 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
