"""确定性评测指标（纯函数，无副作用、无 LLM）。

- 文档级检索指标：Recall@k / Precision@k / MRR（对 LegalBenchRAG 的 top-k 排名）。
- 字符区间指标：span Precision / Recall / F1（对 LLM Searcher 的证据覆盖），
  口径为“逐文件区间合并后按字符重叠长度”计算（对齐 PAKTON 的区间数学）。
- 分类指标：Accuracy / weighted F1 / per-class F1（对 ContractNLI）。
"""
from __future__ import annotations

from typing import Iterable, Sequence

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "union_spans",
    "span_overlap_len",
    "span_precision_recall_f1",
    "accuracy",
    "f1_weighted",
    "f1_per_class",
    "default_ks",
]

# LegalBenchRAG 论文口径的 top-k 集合
DEFAULT_KS = [1, 2, 4, 8, 16, 32, 64]


def default_ks() -> list[int]:
    return list(DEFAULT_KS)


# ---------------------------------------------------------------------------
# 文档级指标：ranked 是有序的文档标识列表（如 file_path），gold 是相关文档集合
# ---------------------------------------------------------------------------

def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """top-k 命中文档与 gold 相关文档的交集大小 / gold 大小。gold 为空视为 0。"""
    if k <= 0:
        return 0.0
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    hit = set(ranked[:k]) & gold_set
    return len(hit) / len(gold_set)


def precision_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    gold_set = set(gold)
    if k <= 0 or not ranked:
        return 0.0
    return len(set(ranked[:k]) & gold_set) / k


def mrr(ranked: Sequence[str], gold: Sequence[str]) -> float:
    """第一个相关文档的倒数排名；无相关命中返回 0。"""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    for i, doc in enumerate(ranked):
        if doc in gold_set:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# 字符区间指标：对象是 file -> [[start, end], ...]
# ---------------------------------------------------------------------------

def union_spans(spans: Iterable[Sequence[int]]) -> list[tuple[int, int]]:
    """合并重叠/相邻区间，返回有序且互不重叠的区间列表。"""
    ordered = sorted((int(s), int(e)) for s, e in spans if e > s)
    merged: list[tuple[int, int]] = []
    for s, e in ordered:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def span_overlap_len(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> int:
    """两组区间（已合并）的交集字符长度。"""
    ua, ub = union_spans(a), union_spans(b)
    i = j = 0
    total = 0
    while i < len(ua) and j < len(ub):
        s, e = max(ua[i][0], ub[j][0]), min(ua[i][1], ub[j][1])
        if e > s:
            total += e - s
        if ua[i][1] <= ub[j][1]:
            i += 1
        else:
            j += 1
    return total


def _union_len(spans: Sequence[Sequence[int]]) -> int:
    return sum(e - s for s, e in union_spans(spans))


def span_precision_recall_f1(
    retrieved_by_file: dict[str, Sequence[Sequence[int]]],
    gold_by_file: dict[str, Sequence[Sequence[int]]],
) -> dict[str, float]:
    """跨文件聚合的字符重叠指标：P/R/F1（全局加总口径）。"""
    retrieved_chars = sum(_union_len(spans) for spans in retrieved_by_file.values())
    gold_chars = sum(_union_len(spans) for spans in gold_by_file.values())
    overlap = 0
    all_files = set(retrieved_by_file) | set(gold_by_file)
    for fp in all_files:
        overlap += span_overlap_len(
            retrieved_by_file.get(fp, []),
            gold_by_file.get(fp, []),
        )
    precision = overlap / retrieved_chars if retrieved_chars else 0.0
    recall = overlap / gold_chars if gold_chars else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# 分类指标：标签为字符串（entailment/contradiction/neutral 或任意）
# ---------------------------------------------------------------------------

def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def f1_weighted(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """weighted F1（按各类支撑数加权），与 sklearn.metrics.f1_score(average='weighted') 一致。"""
    from sklearn.metrics import f1_score
    if not y_true:
        return 0.0
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def f1_per_class(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    """class: F1，逐类单独计算（precision/recall 的宏均值分母为该类出现次数）。"""
    classes = sorted(set(y_true) | set(y_pred))
    out: dict[str, float] = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        out[c] = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return out
