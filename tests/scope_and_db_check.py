#!/usr/bin/env python3
"""DocumentToolkit 作用域与文档导入数据库边界的离线回归检查。"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.contract.tools import DocumentToolkit
from src.document import main as document_main
import src.retrieval.store as retrieval_store


class FakeCursor:
    def __init__(self, conn: "FakeConnection"):
        self.conn = conn
        self.rows: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql: str, params=()):
        text = " ".join(sql.split()).lower()
        params = tuple(params)
        if "select 1 from documents" in text:
            self.rows = [(1,)] if params == ("doc-a", "S1") else []
        elif "select full_text from documents" in text:
            self.rows = [("alpha",)]
        elif "select c.id, c.text, c.section_path" in text:
            self.rows = [("doc-a:0", "alpha", ["Section 1"], 1, [0, 5])]
        elif "select c.id, c.text, c.doc_id" in text:
            self.rows = [("doc-a:0", "alpha", "doc-a", "A", ["Section 1"], 1, [0, 5], "txt")]
        elif "select c.page_no" in text:
            self.rows = [(1,)]
        else:
            self.rows = []
        return self

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class FakeConnection:
    def __init__(self):
        self.rollback_called = False
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True
        return False

    def cursor(self):
        return FakeCursor(self)

    def rollback(self):
        self.rollback_called = True

    def close(self):
        self.closed = True


def _assert_raises(fn, label: str) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"{label}: expected ValueError")


def check_scope() -> None:
    original_connect = retrieval_store.connect
    retrieval_store.connect = FakeConnection
    try:
        unscoped = DocumentToolkit()
        _assert_raises(lambda: unscoped.get_full_text("doc-a"), "empty session")

        scoped = DocumentToolkit(session_id="S1")
        _assert_raises(lambda: scoped.get_full_text("doc-b"), "cross-session document")
        assert scoped.get_full_text("doc-a") == "alpha"
        assert scoped.get_chunk("doc-a:0")["doc_id"] == "doc-a"
        assert scoped.get_section("doc-a", ["Section 1"])["text"] == "alpha"

        whitelisted = DocumentToolkit(session_id="S1", doc_ids=["doc-a"])
        _assert_raises(lambda: whitelisted.get_full_text("doc-b"), "doc_ids whitelist")
        _assert_raises(lambda: whitelisted.get_chunk("doc-b:0"), "chunk whitelist")
    finally:
        retrieval_store.connect = original_connect


def check_db_boundaries() -> None:
    original_connect = retrieval_store.connect
    original_init_db = retrieval_store.init_db
    original_backfill = retrieval_store.backfill_search_tokens
    try:
        failed_schema_conn = FakeConnection()
        retrieval_store.connect = lambda: failed_schema_conn
        retrieval_store.init_db = lambda _conn: (_ for _ in ()).throw(RuntimeError("pg_search missing"))
        assert document_main._ensure_retrieval_schema() is False
        assert failed_schema_conn.rollback_called and failed_schema_conn.closed

        failed_backfill_conn = FakeConnection()
        retrieval_store.connect = lambda: failed_backfill_conn
        retrieval_store.backfill_search_tokens = lambda _conn, _ids: (_ for _ in ()).throw(RuntimeError("backfill failed"))
        assert document_main._backfill_search_tokens(["doc-a"]) == 0
        assert failed_backfill_conn.rollback_called and failed_backfill_conn.closed

        source = inspect.getsource(document_main.cmd_parse)
        assert "with connect() as db_conn" in source
        assert "db_conn = connect()" not in source
    finally:
        retrieval_store.connect = original_connect
        retrieval_store.init_db = original_init_db
        retrieval_store.backfill_search_tokens = original_backfill


def main() -> int:
    check_scope()
    check_db_boundaries()
    print("PASS: scope and database boundary checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
