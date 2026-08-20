"""结构感知切分器。

策略：优先按自然语义结构切（标题为边界）；若某块内容过长，逐级下切，最终以句子为边界。
相邻切片在同一章节内可带少量重叠（CHUNK_OVERLAP_TOKENS）以保持跨片语义连续；
跨章节（标题边界）不重叠，避免污染章节归属。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .models import Chunk, ChunkMetadata, ParsedBlock
from .text_utils import estimate_tokens, truncate_sentence_spans

_TOKEN_CAP = config.CHUNK_TOKEN_CAP
_MIN_TOKENS = config.CHUNK_MIN_TOKENS
_OVERLAP_TOKENS = config.CHUNK_OVERLAP_TOKENS


@dataclass
class _Part:
    """缓冲区内的一小段内容及其全局字符偏移（供 charspan 溯源）。"""

    text: str
    start: int
    end: int
    page_no: int = 0
    bbox: list[float] = field(default_factory=list)
    label: str = "paragraph"


@dataclass
class _Buffer:
    """正在累积的切片内容。"""

    parts: list[_Part] = field(default_factory=list)
    page_no: int = 0
    bbox: list[float] = field(default_factory=list)
    section_path: list[str] = field(default_factory=list)
    label: str = "paragraph"
    has_fresh_content: bool = False

    @property
    def start_offset(self) -> int:
        return self.parts[0].start if self.parts else 0

    @property
    def end_offset(self) -> int:
        return self.parts[-1].end if self.parts else 0

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.join_text())

    def join_text(self) -> str:
        if not self.parts:
            return ""
        out = [self.parts[0].text]
        for previous, part in zip(self.parts, self.parts[1:]):
            gap = part.start - previous.end
            if gap == 0:
                separator = ""
            elif gap == 1:
                separator = "\n"
            else:
                raise ValueError(
                    f"non-contiguous chunk parts: previous_end={previous.end}, start={part.start}"
                )
            out.append(separator)
            out.append(part.text)
        return "".join(out)

    def is_empty(self) -> bool:
        return not self.parts

    def add_content(self, blk: ParsedBlock) -> None:
        if not blk.text:
            return
        if len(blk.offset) != 2:
            raise ValueError("content block must have a [start, end] offset")
        start, end = int(blk.offset[0]), int(blk.offset[1])
        if start < 0 or end < start or end - start != len(blk.text):
            raise ValueError(
                f"content block text/offset mismatch: span=[{start},{end}], text_len={len(blk.text)}"
            )
        if not self.parts:
            self.page_no = blk.page_no
            self.bbox = blk.bbox
            self.label = blk.label
        self.parts.append(
            _Part(
                text=blk.text,
                start=start,
                end=end,
                page_no=blk.page_no,
                bbox=blk.bbox,
                label=blk.label,
            )
        )
        self.has_fresh_content = True

    def take_tail(self, max_tokens: int) -> list[_Part]:
        """从尾部取出合计不超过 max_tokens 的内容（供重叠继承），原缓冲区不变。"""
        if max_tokens <= 0:
            return []
        kept: list[_Part] = []
        total = 0
        for part in reversed(self.parts):
            pt = estimate_tokens(part.text)
            if total + pt > max_tokens:
                break
            kept.append(part)
            total += pt
        kept.reverse()
        return kept


def _finalize(buf: _Buffer, doc_id: str, doc_title: str, source_format: str, index: int) -> Chunk | None:
    text = buf.join_text()
    if not text.strip():
        return None
    chapter = buf.section_path[0] if buf.section_path else ""
    section = buf.section_path[-1] if buf.section_path else ""
    meta = ChunkMetadata(
        doc_id=doc_id,
        doc_title=doc_title,
        section_path=list(buf.section_path),
        chapter=chapter,
        section=section,
        page_no=buf.page_no,
        bbox=list(buf.bbox),
        charspan=[buf.start_offset, buf.end_offset],
        label=buf.label,
        source_format=source_format,
    )
    return Chunk(id=f"{doc_id}:{index}", text=text, metadata=meta)


def chunk_blocks(
    blocks: list[ParsedBlock],
    doc_id: str,
    doc_title: str,
    source_format: str,
) -> list[Chunk]:
    """从归一化结构块生成切片。"""
    section_stack: list[str] = []
    chunks: list[Chunk] = []
    index = 0
    buf = _Buffer()
    # 记录当前章节栈，作为新增块的切片背景
    current_path: list[str] = []

    def flush(overlap: bool = False) -> None:
        nonlocal buf, index
        had_fresh_content = buf.has_fresh_content
        if not buf.is_empty() and had_fresh_content:
            c = _finalize(buf, doc_id, doc_title, source_format, index)
            if c:
                chunks.append(c)
                index += 1
        # 只有本轮加入过新内容才继承 overlap；否则会生成只含重叠尾部的重复切片。
        carry = buf.take_tail(_OVERLAP_TOKENS) if overlap and had_fresh_content else []
        new_buf = _Buffer(section_path=list(current_path))
        if carry:
            new_buf.parts = carry
            new_buf.page_no = carry[0].page_no
            new_buf.bbox = list(carry[0].bbox)
            new_buf.label = carry[0].label
        buf = new_buf

    for blk in blocks:
        if blk.kind == "heading":
            # 跨章节边界：不重叠
            flush(overlap=False)
            # 维护章节栈：同级或更高级标题则弹出，当前标题入栈
            while section_stack and len(section_stack) >= blk.level:
                section_stack.pop()
            section_stack.append(blk.text.strip())
            current_path = list(section_stack)
            continue

        # content
        buf.section_path = list(current_path)
        if blk.text.strip():
            # 若单个内容块过长：先尝试追加，超限则落袋并按句子切分
            if not buf.is_empty() and buf.tokens + estimate_tokens(blk.text) > _TOKEN_CAP:
                flush(overlap=True)
            if estimate_tokens(blk.text) > _TOKEN_CAP:
                # 逐级下切至句子边界（无标点长串由字符区间兜底）
                for rel_start, rel_end in truncate_sentence_spans(blk.text, _TOKEN_CAP):
                    sent = blk.text[rel_start:rel_end]
                    if not buf.is_empty() and buf.tokens + estimate_tokens(sent) > _TOKEN_CAP:
                        flush(overlap=True)
                    buf.add_content(
                        ParsedBlock(
                            kind="content",
                            label=blk.label,
                            text=sent,
                            page_no=blk.page_no,
                            bbox=blk.bbox,
                            offset=[blk.offset[0] + rel_start, blk.offset[0] + rel_end],
                        )
                    )
                flush(overlap=True)
            else:
                buf.add_content(blk)
                if buf.tokens >= _TOKEN_CAP:
                    flush(overlap=True)
    flush(overlap=False)
    return chunks
