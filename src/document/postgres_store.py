"""PostgreSQL + pgvector 存储：连接、建表、事务 upsert 文档及切片（含向量）。"""
from __future__ import annotations

import json
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from . import config


def dsn() -> str:
    return (
        f"host={config.PG_HOST} port={config.PG_PORT} "
        f"dbname={config.PG_DB} user={config.PG_USER} password={config.PG_PASSWORD}"
    )


def connect():
    conn = psycopg.connect(dsn(), autocommit=False)
    register_vector(conn)
    return conn


def init_db(conn) -> None:
    """执行建表脚本（含 vector 扩展与索引）。"""
    schema_path = Path(__file__).resolve().parent / "schema.sql"
    with conn.cursor() as cur:
        cur.execute(schema_path.read_text(encoding="utf-8"))
    conn.commit()


def upsert_document(conn, doc) -> None:
    """事务内写入文档与全部切片（含 embedding）。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (doc_id, file_path, title, source_format, full_text)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (doc_id)
            DO UPDATE SET file_path=EXCLUDED.file_path, title=EXCLUDED.title,
                          source_format=EXCLUDED.source_format,
                          full_text=EXCLUDED.full_text
            """,
            (doc.doc_id, doc.file_path, doc.title, doc.source_format, doc.full_text),
        )
        # 先删旧切片，再整体重插，保证幂等
        cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc.doc_id,))
        for i, chunk in enumerate(doc.chunks):
            m = chunk.metadata
            cur.execute(
                """
                INSERT INTO chunks
                    (id, doc_id, chunk_index, text, metadata, section_path, page_no, charspan, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    chunk.id,
                    doc.doc_id,
                    i,
                    chunk.text,
                    json.dumps(m.model_dump(), ensure_ascii=False),
                    m.section_path,
                    m.page_no,
                    m.charspan,
                    chunk.embedding,
                ),
            )
    conn.commit()
