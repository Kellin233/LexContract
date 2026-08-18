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


def embed_texts(texts: list[str], *, batch_size: int = 8) -> list[list[float]]:
    """对一批文本生成向量，返回归一化浮点列表。

    batch_size 默认 8：sentence-transformers 会把整批 padding 到该批最长序列，
    批过大（如 32 × 500 token）会让单次前向序列膨胀，在 WSL2 8GB 显存上曾触发
    OOM（dxg 驱动 -12），并连带把 CUDA 上下文带崩导致整进程卡死；取偏小的批以
    换取稳定。
    """
    if not texts:
        return []
    model = _model()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=int(batch_size),
    )
    return [v.tolist() for v in vecs]


def embed_text(text: str) -> Optional[list[float]]:
    if not text.strip():
        return None
    out = embed_texts([text])
    return out[0] if out else None
