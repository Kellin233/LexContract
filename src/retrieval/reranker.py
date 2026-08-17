"""BGE cross-encoder 重排：对检索候选按 (query, chunk.text) 打分并取 top-k。"""
from __future__ import annotations

from functools import lru_cache

from . import config
from .models import RetrievedChunk


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL_NAME, max_length=config.RERANK_MAX_LENGTH)


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int | None = None) -> list[RetrievedChunk]:
    """对候选切片重排，返回按重排分降序截断后的列表（就地更新 rerank_score）。"""
    if not chunks:
        return []
    top_k = top_k if top_k is not None else config.RERANK_TOP_K
    pairs = [(query, c.text) for c in chunks]
    model = _model()
    batch = config.RERANK_BATCH_SIZE
    scores: list[float] = []
    for i in range(0, len(pairs), batch):
        part = pairs[i : i + batch]
        # CrossEncoder 返回 (logit, label)；取 logit 即相关分
        scores.extend(model.predict(part))
    for c, s in zip(chunks, scores):
        c.rerank_score = float(s)
    ranked = sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)
    return ranked[:top_k]
