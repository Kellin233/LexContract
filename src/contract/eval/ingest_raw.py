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


def ingest_contractnli_jsonl(
    jsonl_path: str | Path,
    session_id: str,
    *,
    embed: bool = True,
    max_chars: int = 600,
    max_contracts: int | None = None,
    subsets: list[str] | None = None,
) -> dict:
    """把 ContractNLI 的 distinct 合同（前提）整库入库（供检索式分类评测）。

    - 每个前提=一份文档：doc_id = "nli:{idx}"（idx=jsonl 行号，与实例 premise_id 对齐）、
      file_path = "contractnli/{idx}.txt"（占位，无真实路径）、full_text=前提原文；
    - chunk 用 split_text_spans 对齐 raw 偏移，回填 search_tokens；
    - 可选 bge-m3 embedding（embed=False 时退化为 BM25-only）。

    注意：默认**不**执行，需显式调用/CLI（评测里也只在 --ingest-nli 时入库）。
    """
    from src.document.embedder import embed_texts
    from src.document.postgres_store import connect, init_db, upsert_document
    from src.retrieval.store import backfill_search_tokens
    from src.contract.eval import loaders as L

    premises = L.load_contractnli_premises(Path(jsonl_path), subsets=subsets)
    if max_contracts:
        premises = premises[:max_contracts]
    if not premises:
        return {"docs": 0, "chunks": 0, "elapsed_s": 0.0, "skipped": []}

    conn = connect()
    try:
        init_db(conn)
    except Exception as e:  # noqa: BLE001
        print(f"[ingest] init_db 警告: {e}")
        conn.rollback()

    docs = chunks_total = 0
    skipped: list[str] = []
    doc_ids: list[str] = []
    for p in premises:
        idx = p["idx"]
        doc_id = nli_doc_id(idx)
        full_text = p["premise"]
        file_path = f"contractnli/{idx}.txt"
        doc = ParsedDocument(
            doc_id=doc_id,
            file_path=file_path,
            title=f"contractnli/{idx}",
            source_format="txt",
            full_text=full_text,
            structure=[],
            blocks=[],
            chunks=[],
        )
        doc.chunks = _chunk_document(doc_id, doc.title, file_path, full_text, max_chars=max_chars)
        if not doc.chunks:
            skipped.append(f"{idx}: no chunks")
            continue
        if embed:
            try:
                vecs = embed_texts([c.text for c in doc.chunks])
                for c, v in zip(doc.chunks, vecs):
                    c.embedding = v
            except Exception as e:  # noqa: BLE001
                print(f"[ingest] nli:{idx} embedding 失败，降级 bm25-only: {e}")
        try:
            upsert_document(conn, doc)
        except Exception as e:  # noqa: BLE001
            conn.rollback()
            skipped.append(f"{idx}: upsert -> {e}")
            continue
        docs += 1
        chunks_total += len(doc.chunks)
        doc_ids.append(doc_id)
        print(f"[ingest] nli:{idx} subset={p['subset']} premise_len={p['premise_len']} chunks={len(doc.chunks)}")

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


def nli_doc_id(idx: str) -> str:
    """ContractNLI 前提→doc_id 的稳定映射（与实例 premise_id 对齐）。"""
    return f"nli:{idx}"


def nli_doc_ids(idx_list: list[str]) -> list[str]:
    return [nli_doc_id(i) for i in idx_list]


def ingest_benchmark_cli(root: str, benchmark: str, session_id: str, embed: bool = True,
                         max_files: int | None = None) -> None:
    """CLI 入口：入库单个 LegalBenchRAG benchmark。"""
    from src.contract.eval import loaders as L

    root_path = L.find_legalbench_root(root)
    t0 = time.time()
    stats = ingest_corpus_dir(root_path, benchmark, session_id, embed=embed, max_files=max_files)
    stats["elapsed_s"] = time.time() - t0
    print(f"[ingest] 完成 benchmark={benchmark} session={session_id}: {stats}")


def ingest_nli_cli(jsonl_path: str | None, session_id: str, embed: bool = True,
                   max_contracts: int | None = None) -> None:
    """CLI 入口：入库 ContractNLI distinct 合同。"""
    from src.contract.eval import loaders as L

    path = L.find_contractnli_jsonl(jsonl_path)
    t0 = time.time()
    stats = ingest_contractnli_jsonl(path, session_id, embed=embed, max_contracts=max_contracts)
    stats["elapsed_s"] = time.time() - t0
    print(f"[ingest] 完成 ContractNLI 合同入库 session={session_id}: {stats}")


if __name__ == "__main__":
    import argparse
    import sys

    _ROOT = Path(__file__).resolve().parent.parent.parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    p = argparse.ArgumentParser(
        description="语料原样入库 PG：LegalBenchRAG benchmark 或 ContractNLI 合同"
    )
    p.add_argument("dataset", choices=["contractnli", "cuad", "maud", "privacy_qa", "nli"],
                   help="LegalBenchRAG benchmark 名；nli=ContractNLI 合同")
    p.add_argument("--session", default=None, help="会话 ID（默认 lb-<benchmark> 或 nli-contractnli）")
    p.add_argument("--root", default=None, help="LegalBenchRAG 根目录")
    p.add_argument("--contractnli-jsonl", default=None, help="ContractNLI jsonl/zip（dataset=nli 时）")
    p.add_argument("--no-embed", action="store_true", help="不生成 embedding（降级 BM25-only）")
    p.add_argument("--max-files", type=int, default=None, help="LegalBenchRAG：只入库前 N 个文件")
    p.add_argument("--max-contracts", type=int, default=None, help="nli：只入库前 N 份合同（调试用）")
    args = p.parse_args()
    if args.dataset == "nli":
        ingest_nli_cli(args.contractnli_jsonl, args.session or "nli-contractnli",
                       embed=not args.no_embed, max_contracts=args.max_contracts)
    else:
        ingest_benchmark_cli(args.root, args.dataset, args.session or f"lb-{args.dataset}",
                             embed=not args.no_embed, max_files=args.max_files)
