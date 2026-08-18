"""分词器：与 BM25 索引对齐的中英混排分词。

工业界标准做法：通用 Unicode 词切分（UAX#29 风格，数字天然保留）打底，
叠加少量合同领域规则（条款号 / 附件文件号 / 金额），中文段走 jieba 词典切词。

入库时 `chunks.search_tokens` 与查询时都用同样的分词逻辑，保证 pg_search
索引与查询 token 一致。逻辑：NFKC 归一化 → 小写 → 按中英片段分流 →
（中文 jieba ｜ 英文 Unicode 切分 + 合同规则）→ 合并 → 过滤纯标点/符号。
数字一律保留（条款号、时限、金额都是检索词，不做任何丢弃）。
"""
from __future__ import annotations

import re
import unicodedata

import jieba

# 纯标点/符号串（无检索意义）过滤；不含数字——数字/金额/编号都是检索 term
_IGNORE_RE = re.compile(r"^[\W_]+$")

# CJK 汉字连续段（含扩展区），用于把文本切成"中文段 / 非中文段"；
# 带捕获括号，re.split 才能保留被切掉的中文段
_CJK_RUN_RE = re.compile(r"([\u3400-\u4dbf\u4e00-\u9fff]+)")

# 合同特殊 token（非中文段优先匹配，整段保留，不词干、不过滤）。
# 注意两点：
#  - 顺序：更具体的规则（金额/百分比/千分位）放在裸数字前面，避免被 `\d+` 吞掉前缀；
#  - 只保留 pg_search 索引能原样保住的形态（数字/点号条款号/百分比/金额）。
#    括号款号（(b)/(2)）、连字符附件号（ex-10.1）会被索引侧 tokenizer 削平，
#    因此不合并它们——括号字母走通用 `\w+` 拆成独立 token，查询语法才安全。
#   金额/百分比：$5,000,000、$1.2m、5,000,000、10%、8.5%
#   条款/编号：8.2、12.3.4、4.2A、3.14
_SPECIAL_PAT = (
    r"[$€£][\d,]+(?:\.\d+)?(?:[kmb])?"
    r"|\d+(?:,\d{3})+(?:\.\d+)?"
    r"|\d+(?:\.\d+)?%"
    r"|\d+(?:\.\d+)*(?:[a-z])?"
)

# 通用英文 token：Unicode 字母/数字连续段（数字天然保留）
_WORD_PAT = r"\w+"

_TOKENS_RE = re.compile(f"(?P<special>{_SPECIAL_PAT})|(?P<word>{_WORD_PAT})")

_INITIALIZED = False


def _ensure_loaded() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    # 不启用 HMM（法律文书多为规范用词，精确模式更稳）
    jieba.initialize()
    _INITIALIZED = True


def _normalize(text: str) -> str:
    # NFKC 归一化（全角→半角、兼容性字符合并）+ 小写（英文大小写归并）
    return unicodedata.normalize("NFKC", text or "").lower()


def _tokenize_non_cjk(seg: str) -> list[str]:
    """非中文片段：通用 Unicode 词切分 + 合同特殊规则，数字全部保留。"""
    return [m.group(0) for m in _TOKENS_RE.finditer(seg)]


def tokenize(text: str) -> list[str]:
    """对文本（索引或查询）分词，返回有意义 token 列表。"""
    _ensure_loaded()
    norm = _normalize(text)
    tokens: list[str] = []
    for seg in _CJK_RUN_RE.split(norm):
        if not seg:
            continue
        if _CJK_RUN_RE.fullmatch(seg):
            # 中文段：jieba 词典切词（数字由切词结果自然保留）
            seg_tokens = jieba.cut(seg)
        else:
            seg_tokens = _tokenize_non_cjk(seg)
        for tok in seg_tokens:
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
