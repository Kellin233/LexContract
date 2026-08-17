"""LexContract document 模块 CLI。

用法：
    python main.py init-db                         # 初始化 PostgreSQL(pgvector) 表结构
    python main.py parse <文件或目录> [--out-dir ...] [--no-db] [--no-embed]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config
from .parser import parse_file
from .chunker import chunk_blocks
from .json_exporter import export_document


def _iter_input_files(path: Path):
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in config.SUPPORTED_EXTS:
                yield p
    else:
        if path.suffix.lower() not in config.SUPPORTED_EXTS:
            print(f"[warn] 不支持的类型，跳过: {path}", file=sys.stderr)
            return
        yield path


def cmd_init_db(args) -> int:
    from .postgres_store import connect, init_db

    with connect() as conn:
        init_db(conn)
    print("数据库结构初始化完成。")
    return 0


def _ensure_retrieval_schema(conn) -> bool:
    """尽力应用检索模块所需的 schema（session_id / search_tokens / pg_search 索引）。

    依赖 ParadeDB 的 pg_search 扩展；普通 PostgreSQL 上会失败，此时仅保留向量检索可用。
    """
    try:
        from src.retrieval.store import init_db as retrieval_init_db
        retrieval_init_db(conn)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 检索 schema 未就绪（需 ParadeDB 的 pg_search 扩展），跳过 BM25 相关回填: {e}", file=sys.stderr)
        return False


def _backfill_search_tokens(conn, doc_ids: list[str] | None) -> int:
    """为切片回填 BM25 search_tokens（幂等，仅处理缺失行）。返回回填数。"""
    try:
        from src.retrieval.store import backfill_search_tokens
        n = backfill_search_tokens(conn, doc_ids)
        if n:
            print(f"  -> 已回填 BM25 search_tokens: {n} 个切片")
        return n
    except Exception as e:  # noqa: BLE001
        print(f"[warn] search_tokens 回填失败（跳过，仅向量检索可用）: {e}", file=sys.stderr)
        return 0


def _backfill_full_text_from_json(conn) -> int:
    """对已入库但缺失 full_text 的文档，从 JSON 导出文件恢复全文。返回恢复数。"""
    import json as _json

    out_dir = config.OUTPUT_DIR
    if not out_dir.is_dir():
        return 0
    restored = 0
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id FROM documents WHERE full_text = ''")
        missing = [r[0] for r in cur.fetchall()]
        for doc_id in missing:
            fp = out_dir / f"{doc_id}.json"
            if not fp.is_file():
                continue
            try:
                data = _json.loads(fp.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            full_text = data.get("full_text", "")
            if not full_text:
                continue
            cur.execute(
                "UPDATE documents SET full_text = %s WHERE doc_id = %s",
                (full_text, doc_id),
            )
            restored += 1
    conn.commit()
    return restored


def cmd_migrate(_args) -> int:
    """把既有库迁移到证据链阶段：加 full_text 列、恢复全文、回填 BM25 tokens。"""
    from .postgres_store import connect, init_db

    with connect() as conn:
        try:
            init_db(conn)  # 幂等：建表 + full_text 列
        except Exception as e:  # noqa: BLE001
            print(f"[error] 初始化 document schema 失败: {e}", file=sys.stderr)
            return 1
        retrieval_ok = _ensure_retrieval_schema(conn)
        restored = _backfill_full_text_from_json(conn)
    print(f"数据库迁移完成：full_text 恢复 {restored} 篇。")

    with connect() as conn:
        if retrieval_ok:
            n = _backfill_search_tokens(conn, None)
            print(f"search_tokens 回填完成：{n} 个切片。")
        else:
            print("[warn] 检索扩展未就绪，search_tokens 回填跳过。")
    return 0


def cmd_parse(args) -> int:
    from .embedder import embed_texts

    if not args.no_db:
        from .postgres_store import connect, upsert_document
        db_conn = connect()
    else:
        db_conn = None

    files = list(_iter_input_files(Path(args.input)))
    if not files:
        print("未找到可解析的文件。", file=sys.stderr)
        return 1

    for i, fp in enumerate(files, 1):
        print(f"[{i}/{len(files)}] 解析 {fp.name} ...")
        try:
            doc = parse_file(fp)
            # 结构优先切片
            doc.chunks = chunk_blocks(
                doc.blocks, doc.doc_id, doc.title, doc.source_format
            )
        except Exception as e:  # noqa: BLE001
            print(f"[error] 解析失败 {fp}: {e}", file=sys.stderr)
            continue

        # 生成 embedding 并挂到切片
        if not args.no_embed:
            from .text_utils import search_text

            embed_inputs = [
                search_text(doc.title, c.metadata.section_path, c.text)
                for c in doc.chunks
            ]
            vecs = embed_texts(embed_inputs)
            for c, v in zip(doc.chunks, vecs):
                c.embedding = v

        # 导出 JSON
        out = export_document(doc, Path(args.out_dir) if args.out_dir else None)
        print(f"  -> JSON: {out}（{len(doc.chunks)} 个切片）")

        # 入库
        if db_conn is not None:
            try:
                upsert_document(db_conn, doc)
                print(f"  -> 已写入 PostgreSQL: {doc.doc_id}")
                # 确保 BM25 依赖的 search_tokens 列存在并回填（幂等、尽力而为）
                _ensure_retrieval_schema(db_conn)
                _backfill_search_tokens(db_conn, [doc.doc_id])
            except Exception as e:  # noqa: BLE001
                print(f"[error] 入库失败 {doc.doc_id}: {e}", file=sys.stderr)

    if db_conn is not None:
        db_conn.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="document", description="文档解析 + 向量入库")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init-db", help="初始化 PostgreSQL 表结构")
    sp.set_defaults(func=cmd_init_db)

    sp = sub.add_parser("migrate", help="迁移既有库到证据链阶段（加 full_text、恢复全文、回填 BM25 tokens）")
    sp.set_defaults(func=cmd_migrate)

    sp = sub.add_parser("parse", help="解析文档并（可选）写入数据库")
    sp.add_argument("input", help="文件或目录路径")
    sp.add_argument("--out-dir", help="JSON 输出目录（默认见配置 OUTPUT_DIR）")
    sp.add_argument("--no-db", action="store_true", help="不写入 PostgreSQL")
    sp.add_argument("--no-embed", action="store_true", help="不生成 embedding")
    sp.set_defaults(func=cmd_parse)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
