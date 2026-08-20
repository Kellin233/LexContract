"""轻量文本工具：token 估算与句子切分。"""
from __future__ import annotations

import re

# 中文/英文通用句子切分：按句号、问号、感叹号、分号、换行切分。
# 保留结尾标点和其后空白，使切分区间能连续覆盖原文。
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])\s*")

# token 估算口径统一在 src/utils/tokens.py，这里转发保持下游（chunker/评测）兼容
from ..utils.tokens import estimate_tokens  # noqa: E402


def search_text(title: str, section_path: list[str], body: str, *, max_path_chars: int = 400) -> str:
    """构造仅供 embedding 用的可检索文本。

    把文档标题与章节路径并进正文，让向量带上"所属章节"的语义，提升检索召回；
    不改变存储/展示用的原始正文（body）。标题与章节过长时按倒序压缩到 max_path_chars。
    """
    path = " > ".join(part for part in (section_path or []) if part)
    parts: list[str] = []
    if title and title.strip():
        parts.append(title.strip())
    if path:
        parts.append(path)
    prefix = " | ".join(parts)
    if max_path_chars and len(prefix) > max_path_chars:
        prefix = prefix[: max_path_chars]
    return f"{prefix}\n{body}" if prefix else body


def split_sentences(text: str) -> list[str]:
    """按句子边界切分，保留原文中的空格、换行和标点。"""
    return [text[start:end] for start, end in sentence_spans(text)]


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """返回句子在原文中的相对区间，区间连续覆盖全文。"""
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENT_SPLIT_RE.finditer(text):
        end = match.end()
        if end > start:
            spans.append((start, end))
            start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _hard_split_spans(text: str, start: int, end: int, max_tokens: int) -> list[tuple[int, int]]:
    """对仍超 token 上限的区间按字符做有界切分兜底。

    用于无标点、无法按句子下切的超长串：逐字符扩展前缀直到估算 token 越过
    上限，保证每片有界且至少 1 个字符（避免死循环）。
    """
    pieces: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        low, high = cursor + 1, end
        best = cursor + 1
        while low <= high:
            mid = (low + high) // 2
            if estimate_tokens(text[cursor:mid]) <= max_tokens:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        pieces.append((cursor, best))
        cursor = best
    return pieces


def truncate_sentence_spans(text: str, max_tokens: int) -> list[tuple[int, int]]:
    """把超长文本按句子切分，返回原文相对区间。

    用于“结构优先后仍过长时，按句子为边界二次切分”。
    切分结果保留原文中的所有字符，对无标点长串再按字符下切。
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    sentences = sentence_spans(text)
    if not sentences:
        return []

    groups: list[tuple[int, int]] = []
    group_start = sentences[0][0]
    for sent_start, sent_end in sentences:
        if estimate_tokens(text[group_start:sent_end]) <= max_tokens:
            continue
        if sent_start > group_start:
            groups.append((group_start, sent_start))
            group_start = sent_start
        if estimate_tokens(text[group_start:sent_end]) > max_tokens:
            groups.extend(_hard_split_spans(text, group_start, sent_end, max_tokens))
            group_start = sent_end

    if group_start < len(text):
        if estimate_tokens(text[group_start:]) <= max_tokens:
            groups.append((group_start, len(text)))
        else:
            groups.extend(_hard_split_spans(text, group_start, len(text), max_tokens))
    return groups
