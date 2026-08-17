"""本地多语 embedding 生成（sentence-transformers）。

默认模型 BAAI/bge-m3（1024 维，中英文皆适用），懒加载避免拖慢无关 CLI 命令。
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from . import config


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    # 优先本地缓存加载（离线可用）；失败（无缓存且需联网下载）再回退常规加载
    try:
        return SentenceTransformer(
            config.EMBED_MODEL_NAME, device=config.EMBED_DEVICE, local_files_only=True
        )
    except Exception:
        return SentenceTransformer(config.EMBED_MODEL_NAME, device=config.EMBED_DEVICE)


@lru_cache(maxsize=1)
def dim() -> int:
    """模型实际输出维度（作为 schema 依据）。"""
    try:
        return _model().get_sentence_embedding_dimension()
    except Exception:
        return config.EMBED_DIM


def embed_texts(texts: list[str]) -> list[list[float]]:
    """对一批文本生成向量，返回归一化浮点列表。"""
    if not texts:
        return []
    model = _model()
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return [v.tolist() for v in vecs]


def embed_text(text: str) -> Optional[list[float]]:
    if not text.strip():
        return None
    out = embed_texts([text])
    return out[0] if out else None
