"""LegalBenchRAG 语料“原样入库”（偏移对齐的关键步骤）。

LegalBenchRAG 的 gold span 是 corpus 原始 txt 的字符偏移。现有 `_parse_text` 会 strip
空白/去空行，导致 full_text 与原文偏移不一致，因此这里独立实现“原样入库”：
- full_text = 与 loaders.corpus_text 完全一致的读法（文本模式，统一换行）逐字保留；
- chunk 的 charspan 对齐 raw 文本；documents.file_path = corpus 相对路径（如 contractnli/x.txt），
  供评测把 doc_id / 证据区间映射回 file_path + span。
"""
from __future__ import annotations

import time
from pathlib import Path

from src.document.models import Chunk, ChunkMetadata, ParsedDocument


def split_text_spans(text: str, max_chars: int = 600, min_chars: int = 100) -> list[list[int]]:
    """把原文切成对齐 raw 偏移的片段 [start, end]。在段落/句号边界处截断。"""
    spans: list[list[int]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            seg = text[start:end]
            cut = seg.rfind("\n")
            if cut > min_chars:
                end = start + cut
            else:
                m = -1
                for i in range(len(seg) - 1, -1, -1):
                    if seg[i] in ".!?。！？":
                        m = i
                        break
                if m > min_chars:
                    end = start + m + 1
        if end <= start:
            end = start + 1
        spans.append([start, end])
        start = end
    return spans


def _chunk_document(doc_id: str, title: str, file_path: str, full_text: str,
                    source_format: str = "txt", max_chars: int = 600) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, (s, e) in enumerate(split_text_spans(full_text, max_chars=max_chars)):
        text = full_text[s:e].strip()
        if not text:
            continue
        chunks.append(Chunk(
            id=f"{doc_id}:{i}",
            text=text,
            metadata=ChunkMetadata(
                doc_id=doc_id,
                doc_title=title,
                section_path=[],
                page_no=0,
                bbox=[],
                charspan=[s, e],
                label="paragraph",
                source_format=source_format,
            ),
        ))
    return chunks


def ingest_corpus_dir(
    root: Path,
    benchmark: str,
    session_id: str,
    *,
    embed: bool = True,
    max_chars: int = 600,
    doc_subset: str | None = None,
    max_files: int | None = None,
) -> dict:
    """把 corpus/<benchmark>/*.txt 原样入库并分配到会话。

    root: LegalBenchRAG 根目录（含 corpus/）。benchmark: contractnli|cuad|maud|privacy_qa。
    返回统计 {docs, chunks, elapsed_s, skipped}。
    """
    from src.document.embedder import embed_texts
    from src.document.postgres_store import connect, init_db, upsert_document
    from src.retrieval.store import backfill_search_tokens

    corpus_dir = Path(root) / "corpus" / benchmark
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus 目录不存在: {corpus_dir}")

    files = sorted(p for p in corpus_dir.iterdir() if p.suffix.lower() == ".txt")
    if doc_subset:
        files = [p for p in files if doc_subset in p.name]
    if max_files:
        files = files[:max_files]
    if not files:
        return {"docs": 0, "chunks": 0, "elapsed_s": 0.0, "skipped": []}

    from src.contract.eval.loaders import corpus_text

    conn = connect()
    try:
        init_db(conn)  # 幂等：确保 full_text 列存在
    except Exception as e:  # noqa: BLE001
        print(f"[ingest] init_db 警告: {e}")
        conn.rollback()

    docs = 0
    chunks_total = 0
    skipped: list[str] = []
    doc_ids: list[str] = []

    for fp in files:
        rel = f"{benchmark}/{fp.name}"
        try:
            content = corpus_text(root, rel)
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{rel}: {e}")
            continue
        if not content.strip():
            skipped.append(f"{rel}: empty")
            continue
        doc_id = f"{benchmark}:{fp.name}"[:200]
        title = fp.stem[:200]
        doc = ParsedDocument(
            doc_id=doc_id,
            file_path=rel,
            title=title,
            source_format="txt",
            full_text=content,
            structure=[],
            blocks=[],
            chunks=[],
        )
        doc.chunks = _chunk_document(doc_id, title, rel, content, max_chars=max_chars)
        if not doc.chunks:
            skipped.append(f"{rel}: no chunks")
            continue

        # embedding（可选，大语料慢时可用 bm25-only）
        if embed:
            try:
                vecs = embed_texts([c.text for c in doc.chunks])
                for c, v in zip(doc.chunks, vecs):
                    c.embedding = v
            except Exception as e:  # noqa: BLE001
                print(f"[ingest] {rel} embedding 失败，该文档降级为 bm25-only: {e}")

        try:
            upsert_document(conn, doc)
        except Exception as e:  # noqa: BLE001
            conn.rollback()
            skipped.append(f"{rel}: upsert -> {e}")
            continue
        docs += 1
        chunks_total += len(doc.chunks)
        doc_ids.append(doc_id)
        print(f"[ingest] {rel}: {len(doc.chunks)} chunks")

    # 分配会话 + 回填 BM25 tokens
    if doc_ids:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET session_id = %s WHERE doc_id = ANY(%s)",
                (session_id, doc_ids),
            )
            conn.commit()
        try:
            n = backfill_search_tokens(conn, doc_ids)
            print(f"[ingest] search_tokens 回填 {n} 个切片")
        except Exception as e:  # noqa: BLE001
            print(f"[ingest] backfill 警告: {e}")
            conn.rollback()

    conn.close()
    return {"docs": docs, "chunks": chunks_total, "elapsed_s": 0.0, "skipped": skipped}


def ingest_benchmark_cli(root: str, benchmark: str, session_id: str, embed: bool = True,
                         max_files: int | None = None) -> None:
    """CLI 入口：入库单个 benchmark。"""
    from src.contract.eval import loaders as L

    root_path = L.find_legalbench_root(root)
    t0 = time.time()
    stats = ingest_corpus_dir(root_path, benchmark, session_id, embed=embed, max_files=max_files)
    stats["elapsed_s"] = time.time() - t0
    print(f"[ingest] 完成 benchmark={benchmark} session={session_id}: {stats}")


if __name__ == "__main__":
    import argparse
    import sys

    _ROOT = Path(__file__).resolve().parent.parent.parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    p = argparse.ArgumentParser(description="LegalBenchRAG corpus 原样入库 PG")
    p.add_argument("benchmark", help="contractnli|cuad|maud|privacy_qa")
    p.add_argument("--session", default=None, help="会话 ID（默认 lb-<benchmark>）")
    p.add_argument("--root", default=None, help="LegalBenchRAG 根目录")
    p.add_argument("--no-embed", action="store_true", help="不生成 embedding（降级 BM25-only）")
    p.add_argument("--max-files", type=int, default=None, help="只入库前 N 个文件（调试用）")
    args = p.parse_args()
    ingest_benchmark_cli(
        root=args.root,
        benchmark=args.benchmark,
        session_id=args.session or f"lb-{args.benchmark}",
        embed=not args.no_embed,
        max_files=args.max_files,
    )
