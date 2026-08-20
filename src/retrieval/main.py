"""LexTrace retrieval 模块 CLI。

用法：
    python3 -m src.retrieval.main init-db                       # 加列；有扩展时建 BM25 索引
    python3 -m src.retrieval.main assign <doc_id> --session S1  # 文档分派到会话
    python3 -m src.retrieval.main unassign <doc_id>             # 取消分派
    python3 -m src.retrieval.main sessions                      # 列出会话
    python3 -m src.retrieval.main backfill [--doc-id ...]       # 回填 search_tokens
    python3 -m src.retrieval.main query "<问题>" --session S1 [--mode hybrid|vector|bm25] \
        [--top-k 10] [--doc-id ...] [--no-rerank] [--candidate-k 100]
"""
from __future__ import annotations

import argparse
import sys

from . import config
from .postgres import PostgresRetriever


def cmd_init_db(args) -> int:
    from .store import connect, init_db

    with connect() as conn:
        bm25_ok = init_db(conn)
    if bm25_ok is False:
        print("数据库基础结构初始化完成；未检测到 pg_search，BM25 已降级，仅向量/全文检索可用。")
    else:
        print("数据库结构初始化完成（session_id + search_tokens + BM25 索引）。")
    return 0


def cmd_assign(args) -> int:
    from .store import assign_session, connect

    with connect() as conn:
        ok = assign_session(conn, args.doc_id, args.session)
    print(f"文档 {args.doc_id} -> 会话 {args.session or '(取消归属)'}: {'成功' if ok else '未找到'}")
    return 0 if ok else 1


def cmd_unassign(args) -> int:
    from .store import connect, unassign_session

    with connect() as conn:
        ok = unassign_session(conn, args.doc_id)
    print(f"取消文档 {args.doc_id} 的会话归属: {'成功' if ok else '未找到'}")
    return 0 if ok else 1


def cmd_sessions(args) -> int:
    from .store import connect, list_sessions

    with connect() as conn:
        sessions = list_sessions(conn)
    if not sessions:
        print("（无文档/会话）")
        return 0
    width = max((len(s["session_id"]) for s in sessions), default=4)
    print(f"{'session_id':<{width}}  doc_count")
    for s in sessions:
        print(f"{s['session_id']:<{width}}  {s['doc_count']}")
    return 0


def cmd_backfill(args) -> int:
    from .store import backfill_search_tokens, connect

    with connect() as conn:
        n = backfill_search_tokens(conn, args.doc_id)
    print(f"回填 search_tokens 完成，共更新 {n} 个切片。")
    return 0


def _score_label(c) -> str:
    if c.rerank_score is not None:
        return f"rerank={c.rerank_score:.4f}"
    if c.rrf_score is not None:
        return f"rrf={c.rrf_score:.4f}"
    if c.vectordb_similarity_score is not None:
        return f"sim={c.vectordb_similarity_score:.4f}"
    if c.bm25_score is not None:
        return f"bm25={c.bm25_score:.4f}"
    return ""


def cmd_query(args) -> int:
    retriever = PostgresRetriever()
    try:
        chunks = retriever.retrieve(
            args.query, mode=args.mode, session_id=args.session,
            limit=args.top_k, candidate_k=args.candidate_k,
            doc_ids=args.doc_id or None,
        )
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    rerank_enabled = not args.no_rerank
    if rerank_enabled and chunks:
        from .reranker import rerank

        chunks = rerank(args.query, chunks, top_k=args.top_k)

    print(f"模式={args.mode} 命中 {len(chunks)} 条"
          + ("（已重排）" if rerank_enabled else ""))
    for i, c in enumerate(chunks, 1):
        loc = " > ".join(c.section_path) or c.doc_title or c.doc_id
        print(f"\n[{i}] {c.id}  [{_score_label(c)}]")
        print(f"    文档: {c.doc_title}（{c.doc_id}） 页码:P{c.page_no}  章节:{loc}")
        preview = c.text.replace("\n", " ").strip()
        print(f"    {preview[:160]}{'...' if len(preview) > 160 else ''}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="retrieval", description="LexTrace 检索服务（BM25/向量/混合）")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init-db", help="初始化 session_id/search_tokens 列；有 pg_search 时建立 BM25 索引")
    sp.set_defaults(func=cmd_init_db)

    sp = sub.add_parser("assign", help="文档分派到会话")
    sp.add_argument("doc_id")
    sp.add_argument("--session", default="", help="目标会话 ID（默认空，即取消归属）")
    sp.set_defaults(func=cmd_assign)

    sp = sub.add_parser("unassign", help="取消文档会话归属")
    sp.add_argument("doc_id")
    sp.set_defaults(func=cmd_unassign)

    sp = sub.add_parser("sessions", help="列出会话")
    sp.set_defaults(func=cmd_sessions)

    sp = sub.add_parser("backfill", help="为缺失 search_tokens 的切片回填分词")
    sp.add_argument("--doc-id", nargs="*", help="限定文档 ID（默认全部）")
    sp.set_defaults(func=cmd_backfill)

    sp = sub.add_parser("query", help="检索查询")
    sp.add_argument("query")
    sp.add_argument("--session", required=True, help="会话 ID（必填，保证作用域）")
    sp.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    sp.add_argument("--top-k", type=int, default=config.TOP_K)
    sp.add_argument("--candidate-k", type=int, default=None, help="混合检索候选池大小")
    sp.add_argument("--doc-id", nargs="*", help="限定文档（默认会话内全部）")
    sp.add_argument("--no-rerank", action="store_true", help="关闭 BGE 重排")
    sp.set_defaults(func=cmd_query)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
