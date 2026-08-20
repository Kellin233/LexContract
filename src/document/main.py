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


def _ensure_retrieval_schema() -> bool:
    """尽力应用检索模块所需的 schema（session_id / search_tokens / pg_search 索引）。

    依赖 ParadeDB 的 pg_search 扩展；普通 PostgreSQL 上会失败，此时仅保留向量检索可用。
    """
    from src.retrieval.store import connect, init_db as retrieval_init_db
    conn = None
    try:
        conn = connect()
        result = retrieval_init_db(conn)
        if result is False:
            print(
                "[warn] 检索基础字段已就绪，但 pg_search 不可用，BM25 功能降级；全文/grep 与向量检索仍可用。",
                file=sys.stderr,
            )
        return result is not False
    except Exception as e:  # noqa: BLE001
        if conn is not None:
            conn.rollback()
        print(f"[warn] 检索 schema 未就绪（需 ParadeDB 的 pg_search 扩展），跳过 BM25 相关回填: {e}", file=sys.stderr)
        return False
    finally:
        if conn is not None:
            conn.close()


def _backfill_search_tokens(doc_ids: list[str] | None) -> int:
    """为切片回填 BM25 search_tokens（幂等，仅处理缺失行）。返回回填数。"""
    from src.retrieval.store import backfill_search_tokens, connect
    conn = None
    try:
        conn = connect()
        n = backfill_search_tokens(conn, doc_ids)
        if n:
            print(f"  -> 已回填 BM25 search_tokens: {n} 个切片")
        return n
    except Exception as e:  # noqa: BLE001
        if conn is not None:
            conn.rollback()
        print(f"[warn] search_tokens 回填失败（跳过，仅向量检索可用）: {e}", file=sys.stderr)
        return 0
    finally:
        if conn is not None:
            conn.close()


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


def _repair_legacy_charspans(conn) -> int:
    """修复可证明是单个前导换行错位的旧切片区间。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE chunks c
            SET charspan = ARRAY[c.charspan[1] + 1, c.charspan[2]],
                metadata = jsonb_set(
                    COALESCE(c.metadata, '{}'::jsonb),
                    '{charspan}',
                    to_jsonb(ARRAY[c.charspan[1] + 1, c.charspan[2]]::int[]),
                    true
                )
            FROM documents d
            WHERE d.doc_id = c.doc_id
              AND array_length(c.charspan, 1) = 2
              AND c.charspan[1] >= 0
              AND c.charspan[2] > c.charspan[1]
              AND c.charspan[2] <= char_length(d.full_text)
              AND c.charspan[2] - c.charspan[1] = char_length(c.text) + 1
              AND substring(d.full_text FROM c.charspan[1] + 1 FOR 1) = E'\\n'
              AND substring(d.full_text FROM c.charspan[1] + 2 FOR char_length(c.text)) = c.text
            RETURNING c.id
            """
        )
        repaired = len(cur.fetchall())
    conn.commit()
    return repaired


def _find_unresolved_chunk_alignment(conn, sample_limit: int = 20) -> tuple[int, list[str]]:
    """扫描切片正文与全文区间，返回异常总数及有限样本 ID。"""
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id, full_text FROM documents")
        full_texts = {row[0]: row[1] or "" for row in cur.fetchall()}
        cur.execute("SELECT id, doc_id, text, charspan FROM chunks ORDER BY id")
        unresolved = 0
        samples: list[str] = []
        for chunk_id, doc_id, text, charspan in cur.fetchall():
            full = full_texts.get(doc_id, "")
            try:
                start, end = int(charspan[0]), int(charspan[1])
            except (TypeError, ValueError, IndexError):
                start, end = -1, -1
            aligned = (
                0 <= start < end <= len(full)
                and isinstance(text, str)
                and full[start:end] == text
            )
            if not aligned:
                unresolved += 1
                if len(samples) < sample_limit:
                    samples.append(str(chunk_id))
    return unresolved, samples


def cmd_migrate(_args) -> int:
    """把既有库迁移到证据链阶段，并安全修复旧切片区间。"""
    from .postgres_store import connect, init_db

    try:
        with connect() as conn:
            init_db(conn)  # 幂等：建表 + full_text 列
    except Exception as e:  # noqa: BLE001
        print(f"[error] 初始化 document schema 失败: {e}", file=sys.stderr)
        return 1

    retrieval_ok = _ensure_retrieval_schema()
    with connect() as conn:
        restored = _backfill_full_text_from_json(conn)
        repaired = _repair_legacy_charspans(conn)
        unresolved, samples = _find_unresolved_chunk_alignment(conn)
    print(f"数据库迁移完成：full_text 恢复 {restored} 篇，安全修复区间 {repaired} 个。")
    if unresolved:
        print(
            f"[warn] 仍有 {unresolved} 个切片的正文与 charspan 不一致，请重新解析原始文件。示例: {', '.join(samples)}",
            file=sys.stderr,
        )

    if retrieval_ok:
        n = _backfill_search_tokens(None)
        print(f"search_tokens 回填完成：{n} 个切片。")
    else:
        print("[warn] 检索扩展未就绪，search_tokens 回填跳过。")
    return 0


def cmd_parse(args) -> int:
    from .embedder import embed_texts

    db_enabled = not args.no_db
    retrieval_ok = _ensure_retrieval_schema() if db_enabled else False
    if db_enabled:
        from .postgres_store import connect, upsert_document

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
        if db_enabled:
            db_conn = None
            try:
                db_conn = connect()
                upsert_document(db_conn, doc)
                print(f"  -> 已写入 PostgreSQL: {doc.doc_id}")
                if retrieval_ok:
                    _backfill_search_tokens([doc.doc_id])
            except Exception as e:  # noqa: BLE001
                if db_conn is not None:
                    db_conn.rollback()
                print(f"[error] 入库失败 {doc.doc_id}: {e}", file=sys.stderr)
            finally:
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
