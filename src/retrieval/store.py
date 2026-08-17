"""PostgreSQL 连接与检索模块的管理操作：建表/会话分派/search_tokens 回填。"""
from __future__ import annotations

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from . import config
from .tokenizer import search_tokens


def dsn() -> str:
    return (
        f"host={config.PG_HOST} port={config.PG_PORT} "
        f"dbname={config.PG_DB} user={config.PG_USER} password={config.PG_PASSWORD}"
    )


def connect() -> psycopg.Connection:
    conn = psycopg.connect(dsn(), autocommit=False)
    register_vector(conn)
    return conn


def init_db(conn: psycopg.Connection) -> None:
    """执行检索相关的建表/加列/建扩展脚本（幂等）。"""
    schema_path = Path(__file__).resolve().parent / "schema.sql"
    with conn.cursor() as cur:
        cur.execute(schema_path.read_text(encoding="utf-8"))
    conn.commit()


def assign_session(conn: psycopg.Connection, doc_id: str, session_id: str) -> bool:
    """把一个文档分派到指定会话（工作区）。返回是否成功（文档存在）。"""
    session_id = (session_id or "").strip()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE documents SET session_id = %s WHERE doc_id = %s",
            (session_id, doc_id),
        )
        conn.commit()
        return cur.rowcount > 0


def unassign_session(conn: psycopg.Connection, doc_id: str) -> bool:
    """取消文档的会话归属（置为 ''，即不参与检索）。"""
    with conn.cursor() as cur:
        cur.execute("UPDATE documents SET session_id = '' WHERE doc_id = %s", (doc_id,))
        conn.commit()
        return cur.rowcount > 0


def list_sessions(conn: psycopg.Connection) -> list[dict]:
    """列出各会话的文档归属情况（空会话过滤，'' 归为 unassigned）。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(NULLIF(session_id, ''), '(unassigned)') AS session_id,
                   count(*) AS doc_count
            FROM documents
            GROUP BY session_id
            ORDER BY session_id
            """
        )
        return [{"session_id": r[0], "doc_count": r[1]} for r in cur.fetchall()]


def backfill_search_tokens(conn: psycopg.Connection, doc_ids: list[str] | None = None) -> int:
    """为缺失/指定的切片回填 search_tokens（增量，非破坏性）。

    只更新 search_tokens 为空串的切片；如传 doc_ids 则限定范围。
    返回更新的切片数。
    """
    sql = "SELECT id, text FROM chunks WHERE search_tokens = ''"
    params: tuple = ()
    if doc_ids:
        placeholders = ", ".join("%s" for _ in doc_ids)
        sql += f" AND doc_id IN ({placeholders})"
        params = tuple(doc_ids)
    with conn.cursor() as cur:
        rows = cur.execute(sql, params).fetchall()
        updated = 0
        for chunk_id, text in rows:
            tokens = search_tokens(text)
            if not tokens:
                continue
            cur.execute(
                "UPDATE chunks SET search_tokens = %s WHERE id = %s",
                (tokens, chunk_id),
            )
            updated += 1
        conn.commit()
        return updated
