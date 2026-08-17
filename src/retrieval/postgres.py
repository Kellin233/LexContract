"""PostgreSQL-backed vector, BM25 (pg_search) and hybrid (加权 RRF) 检索。

对齐 PAKTON-develop 的 PostgresRetriever 设计，落到 LexContract 原生 psycopg + pgvector 栈。
查询向量化复用 document.embedder（BAAI/bge-m3）。
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Callable

from . import config
from .models import RetrievedChunk
from .tokenizer import build_bm25_query
from .store import connect


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.9g}" for v in values) + "]"


def _default_query_encoder() -> Callable[[str], list[float]]:
    # 复用 document 模块的 bge-m3 embedder（同一模型/维度，懒加载）
    from src.document.embedder import embed_text

    def encode(query: str) -> list[float]:
        vec = embed_text(query)
        if not vec:
            raise ValueError("query embedding is empty")
        return vec

    return encode


# 命中切片需从库中带出的定位/元数据列（不变量+可选得分列）
_CHUNK_COLS = (
    "c.id, c.text, c.doc_id, d.title, d.session_id, "
    "c.page_no, c.section_path, c.charspan, d.source_format"
)


class PostgresRetriever:
    """三种公开检索模式（vector/bm25/hybrid）共用的数据库适配器。"""

    def __init__(self, embed_query: Callable[[str], list[float]] | None = None):
        self.embed_query = embed_query or _default_query_encoder()
        self._embedding_lock = threading.Lock()

    # --- 会话作用域：拒绝无作用域查询 ---
    @staticmethod
    def _scope(session_id: str) -> str:
        if not session_id or not str(session_id).strip():
            raise ValueError("session_id is required; refusing an unscoped search")
        return str(session_id).strip()

    @staticmethod
    def _to_chunks(rows, mode: str) -> list[RetrievedChunk]:
        out = []
        for row in rows:
            (cid, text, doc_id, title, session_id, page_no,
             section_path, charspan, source_format, *score_col) = row
            score = float(score_col[0]) if score_col else None
            if mode == "vector":
                scores = {"vectordb_similarity_score": 1.0 - score if score is not None else None}
            elif mode == "bm25":
                scores = {"bm25_score": score}
            else:  # hybrid 由 vector/bm25 合并而来，此处不直接取分
                scores = {}
            out.append(
                RetrievedChunk.from_row(
                    id=cid, text=text, doc_id=doc_id, title=title or "",
                    session_id=session_id or "", page_no=page_no or 0,
                    section_path=list(section_path or []),
                    charspan=list(charspan or []),
                    source_format=source_format or "",
                    retriever=mode, scores=scores,
                )
            )
        return out

    # --- 向量检索 ---
    def vector(self, query: str, *, session_id: str, limit: int = 64,
               doc_ids: list[str] | None = None) -> list[RetrievedChunk]:
        with self._embedding_lock:
            vec = self.embed_query(query)
        scoped = self._scope(session_id)
        # 占位符顺序：vec, session, [doc_ids], limit
        params: list = [_vector_literal(vec), scoped]
        doc_filter = ""
        if doc_ids is not None:
            if not doc_ids:
                doc_filter = " AND false"
            else:
                doc_filter = " AND c.doc_id = ANY(%s)"
                params.append(list(doc_ids))
        params.append(limit)
        sql = f"""
            SELECT {_CHUNK_COLS},
                   c.embedding <=> %s::vector AS distance
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE d.session_id = %s{doc_filter}
            ORDER BY distance ASC, c.chunk_index ASC
            LIMIT %s
        """
        rows = self._execute(sql, params)
        return self._to_chunks(rows, "vector")

    # --- BM25 检索（pg_search）---
    def bm25(self, query: str, *, session_id: str, limit: int = 64,
             doc_ids: list[str] | None = None) -> list[RetrievedChunk]:
        safe_query = build_bm25_query(query)
        if not safe_query:
            return []
        scoped = self._scope(session_id)
        # 占位符顺序：session, [doc_ids], query, limit
        params: list = [scoped]
        doc_filter = ""
        if doc_ids is not None:
            if not doc_ids:
                doc_filter = " AND false"
            else:
                doc_filter = " AND c.doc_id = ANY(%s)"
                params.append(list(doc_ids))
        params.append(safe_query)
        params.append(limit)
        sql = f"""
            SELECT {_CHUNK_COLS},
                   paradedb.score(c.id) AS score
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE d.session_id = %s{doc_filter}
              AND c.search_tokens @@@ %s
            ORDER BY score DESC, c.chunk_index ASC
            LIMIT %s
        """
        rows = self._execute(sql, params)
        return self._to_chunks(rows, "bm25")

    # --- 混合检索：加权 Reciprocal Rank Fusion ---
    def hybrid(self, query: str, *, session_id: str, limit: int | None = None,
               candidate_k: int | None = None, rrf_k: int | None = None,
               vector_weight: float | None = None, bm25_weight: float | None = None,
               doc_ids: list[str] | None = None) -> list[RetrievedChunk]:
        limit = limit or config.TOP_K
        candidate_k = candidate_k or max(limit * 4, config.CANDIDATE_K)
        rrf_k = rrf_k if rrf_k is not None else config.RRF_K
        vector_weight = config.VECTOR_WEIGHT if vector_weight is None else vector_weight
        bm25_weight = config.BM25_WEIGHT if bm25_weight is None else bm25_weight

        scoped = self._scope(session_id)
        vector_rows = self.vector(query, session_id=scoped, limit=candidate_k, doc_ids=doc_ids)
        lexical_rows = self.bm25(query, session_id=scoped, limit=candidate_k, doc_ids=doc_ids)

        merged: dict[str, RetrievedChunk] = {}
        scores: defaultdict[str, float] = defaultdict(float)
        for rank, row in enumerate(vector_rows, 1):
            merged.setdefault(row.id, row)
            scores[row.id] += vector_weight / (rrf_k + rank)
        for rank, row in enumerate(lexical_rows, 1):
            current = merged.setdefault(row.id, row)
            current.bm25_score = row.bm25_score
            scores[row.id] += bm25_weight / (rrf_k + rank)

        ordered = sorted(merged.values(), key=lambda r: scores[r.id], reverse=True)[:limit]
        for row in ordered:
            row.retriever = "hybrid"
            row.rrf_score = scores[row.id]
        return ordered

    # --- 统一入口 ---
    def retrieve(self, query: str, *, mode: str = "hybrid", session_id: str,
                 limit: int | None = None, candidate_k: int | None = None,
                 doc_ids: list[str] | None = None) -> list[RetrievedChunk]:
        scoped = self._scope(session_id)
        if mode == "vector":
            return self.vector(query, session_id=scoped, limit=limit or config.TOP_K, doc_ids=doc_ids)
        if mode == "bm25":
            return self.bm25(query, session_id=scoped, limit=limit or config.TOP_K, doc_ids=doc_ids)
        if mode == "hybrid":
            return self.hybrid(query, session_id=scoped, limit=limit, candidate_k=candidate_k, doc_ids=doc_ids)
        raise ValueError("mode must be one of: vector, bm25, hybrid")

    def _execute(self, sql: str, params: list):
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()
