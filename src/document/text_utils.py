"""轻量文本工具：token 估算与句子切分。"""
from __future__ import annotations

import re

# 中文/英文通用句子切分：按句号、问号、感叹号、分号、换行切分。
# 保留结尾标点；用负向断言避免把小数点/缩写误切。
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])\s*")

# 简易 token 估算：中文字符按 0.6、其余按空白切分。
_ZH_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（不引入 tiktoken 依赖）。"""
    zh = len(_ZH_RE.findall(text))
    rest = _ZH_RE.sub(" ", text)
    non_zh = len(rest.split())
    return int(zh * 0.6 + non_zh)


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
    """按句子边界切分，返回非空句子列表。"""
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text)]
    return [p for p in parts if p]


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """对仍超 token 上限的文本按字符做有界切分兜底。

    用于无标点、无法按句子下切的超长串：逐字符扩展前缀直到估算 token 越过
    上限，保证每片有界且至少 1 个字符（避免死循环）。
    """
    pieces: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = start + 1
        while end < n and estimate_tokens(text[start : end + 1]) <= max_tokens:
            end += 1
        pieces.append(text[start:end])
        start = end
    return pieces


def truncate_sentences(text: str, max_tokens: int) -> list[str]:
    """把超长文本按句子切分，并在 token 上限内尽量合并句子成组。

    用于“结构优先后仍过长时，按句子为边界二次切分”。
    对切分后仍超限的分组（无标点长串等）再做有界字符兜底，保证每片较小。
    """
    sentences = split_sentences(text)
    groups: list[str] = []
    current: list[str] = []
    current_tok = 0
    for sent in sentences:
        st = estimate_tokens(sent)
        if current and current_tok + st > max_tokens:
            groups.append("".join(current))
            current, current_tok = [], 0
        current.append(sent)
        current_tok += st
    if current:
        groups.append("".join(current))
    # 硬上限兜底：任何分组仍超限则按字符进一步下切
    bounded: list[str] = []
    for group in groups:
        if estimate_tokens(group) <= max_tokens:
            bounded.append(group)
        else:
            bounded.extend(_hard_split(group, max_tokens))
    return bounded
