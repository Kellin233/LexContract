"""分词器：与 BM25 索引对齐的中文/多语分词。

入库时 `chunks.search_tokens` 与查询时都用同样的分词逻辑，保证 pg_search
索引与查询 token 一致。逻辑：NFKC 归一化 → jieba 分词 → 过滤空白/纯标点。
"""
from __future__ import annotations

import re
import unicodedata

import jieba

# 全角→半角后仍为纯标点/符号/数字的词（无检索意义）过滤
_IGNORE_RE = re.compile(r"^[\W_0-9]+$")

# 首次调用时初始化 jieba（可注入自定义词典目录）
_INITIALIZED = False


def _ensure_jieba() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    # 不启用 HMM（法律文书多为规范用词，精确模式更稳）
    jieba.initialize()
    _INITIALIZED = True


def _normalize(text: str) -> str:
    # NFKC 归一化：全角→半角、兼容性字符合并
    return unicodedata.normalize("NFKC", text or "")


def tokenize(text: str) -> list[str]:
    """对文本（索引或查询）分词，返回有意义 token 列表。"""
    _ensure_jieba()
    norm = _normalize(text)
    tokens: list[str] = []
    for tok in jieba.cut(norm):
        tok = tok.strip()
        if not tok:
            continue
        if _IGNORE_RE.match(tok):
            continue
        tokens.append(tok)
    return tokens


def search_tokens(text: str) -> str:
    """返回空格分隔的 token 串，用于写入 chunks.search_tokens。"""
    return " ".join(tokenize(text))


def build_bm25_query(query: str) -> str:
    """将自然语言查询转为 pg_search 能识别的 `search_tokens:(t1 t2)` 形式。

    空 token（纯标点等）返回空串，调用方据此返回空结果。
    """
    tokens = tokenize(query or "")
    if not tokens:
        return ""
    return f"search_tokens:({' '.join(tokens)})"
