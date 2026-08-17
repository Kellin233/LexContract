"""LegalBenchRAG 语料“原样入库”（偏移对齐的关键步骤）。

LegalBenchRAG 的 gold span 是 corpus 原始 txt 的字符偏移。现有 `_parse_text` 会 strip
空白/去空行，导致 full_text 与原文偏移不一致，因此这里独立实现“原样入库”：
- full_text = 与 loaders.corpus_text 完全一致的读法（文本模式，统一换行）逐字保留；
- chunk 的 charspan 对齐 raw 文本；documents.file_path = corpus 相对路径（如 contractnli/x.txt），
  供评测把 doc_id / 证据区间映射回 file_path + span。
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from src.document.models import Chunk, ChunkMetadata, ParsedDocument


def _heading_level(line: str) -> int | None:
    """标题层级启发式：复用 document 解析器判定 + 英文法律条款前缀 + 全大写短行。

    返回层级（数字编号=段数；罗马/英文前缀/全大写=浅层），非标题返回 None。
    只做判定，不做任何文本重排——保证 raw 偏移不变。
    """
    from src.document.parser import _heuristic_heading_level

    t = line.strip()
    if not t:
        return None
    lvl = _heuristic_heading_level(line)
    if lvl is not None:
        return lvl
    if len(t) > 80:
        return None
    # 英文法律条款前缀：ARTICLE I / Section 4.2 / EXHIBIT A / SCHEDULE 1 ...
    if re.match(
        r"^\s*(?:article|section|clause|exhibit|schedule|appendix|annex|attachment|part|title|item|recital)\b",
        t, re.IGNORECASE,
    ):
        return 1
    # 全大写短行（如 "CERTAIN DEFINITIONS"、"SECURITY AGREEMENT"）
    letters = [c for c in t if c.isalpha()]
    if (
        letters
        and len(t) <= 60
        and sum(c.isupper() for c in letters) / len(letters) >= 0.8
        and not t.rstrip().endswith(".")
    ):
        return 1
    return None


def _split_lines_with_offsets(text: str) -> list[tuple[int, int, str]]:
    """按 \n 切行并给出每行的全局字符偏移 [start, end)。"""
    lines: list[tuple[int, int, str]] = []
    pos, n = 0, len(text)
    while pos < n:
        nl = text.find("\n", pos)
        if nl == -1:
            lines.append((pos, n, text[pos:]))
            break
        lines.append((pos, nl, text[pos:nl]))
        pos = nl + 1
    return lines


def token_budget_chunks(
    text: str,
    *,
    max_tokens: int = 600,
    min_tokens: int = 50,
    overlap_tokens: int = 50,
    detect_headings: bool = True,
) -> list[dict]:
    """在 raw 偏移上按 token 预算 + 标题边界切分（对齐正常链路切片策略）。

    与结构切片同理：标题作为章节边界（跨章节不重叠），同章节内容累积到 max_tokens 后
    落袋并带上 overlap_tokens 的重叠；全部在原始文本偏移上进行，不重排文本。
    返回 [{start, end, section_path: [标题...], text}]。
    """
    from src.document.text_utils import estimate_tokens

    lines = _split_lines_with_offsets(text)
    chunks: list[dict] = []
    section_stack: list[str] = []
    cur: list[tuple[int, int, str]] = []

    def cur_tokens() -> int:
        return sum(estimate_tokens(c) for _, _, c in cur)

    def flush(overlap: bool = True) -> None:
        nonlocal cur
        if cur:
            s, e = cur[0][0], cur[-1][1]
            chunks.append({"start": s, "end": e,
                           "section_path": list(section_stack), "text": text[s:e]})
        carry: list[tuple[int, int, str]] = []
        if overlap and cur:
            total = 0
            for item in reversed(cur):
                t = estimate_tokens(item[2])
                if total + t > overlap_tokens:
                    break
                carry.append(item)
                total += t
            carry.reverse()
        cur = carry

    for s, e, content in lines:
        stripped = content.strip()
        if not stripped:
            continue
        if detect_headings:
            lvl = _heading_level(content)
            if lvl is not None:
                flush(overlap=False)
                while section_stack and len(section_stack) >= lvl:
                    section_stack.pop()
                section_stack.append(stripped[:120])
                continue
        t = estimate_tokens(content)
        if cur and cur_tokens() + t > max_tokens:
            flush(overlap=True)
        cur.append((s, e, content))

    flush(overlap=False)
    return chunks


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


def _chunk_document(
    doc_id: str, title: str, file_path: str, full_text: str,
    source_format: str = "txt",
    max_chars: int = 600, max_tokens: int | None = None,
    min_tokens: int = 50, overlap_tokens: int = 50,
    detect_headings: bool = True,
) -> list[Chunk]:
    """按 token 预算（默认，对齐正常链路）或 600 字符定长（max_tokens=None）切分。

    无论哪种都在 raw 偏移上切，charspan 直接对齐语料原文。
    """
    if max_tokens:
        segs = token_budget_chunks(
            full_text, max_tokens=max_tokens, min_tokens=min_tokens,
            overlap_tokens=overlap_tokens, detect_headings=detect_headings,
        )
    else:
        segs = [
            {"start": s, "end": e, "section_path": [], "text": full_text[s:e]}
            for s, e in split_text_spans(full_text, max_chars=max_chars)
        ]
    chunks: list[Chunk] = []
    for i, seg in enumerate(segs):
        text = seg["text"]
        # 关键：存 full_text[start:end] 的精确切片（不 strip），保证
        # chunk.text == full_text[charspan]，否则 CitationVerifier / span 对齐会失真
        if not text.strip():
            continue
        chunks.append(Chunk(
            id=f"{doc_id}:{i}",
            text=text,
            metadata=ChunkMetadata(
                doc_id=doc_id,
                doc_title=title,
                section_path=list(seg.get("section_path", [])),
                page_no=0,
                bbox=[],
                charspan=[seg["start"], seg["end"]],
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
    max_tokens: int | None = None,
    min_tokens: int = 50,
    overlap_tokens: int = 50,
    detect_headings: bool = True,
    doc_subset: str | None = None,
    max_files: int | None = None,
    file_list: list[str] | None = None,
) -> dict:
    """把 corpus/<benchmark>/*.txt 原样入库并分配到会话。

    root: LegalBenchRAG 根目录（含 corpus/）。benchmark: contractnli|cuad|maud|privacy_qa。
    file_list: 只入库这些 corpus 相对路径（如 ["cuad/xxx.txt"]）——用于 --ingest-only-referenced
    调试；默认全量入库（PAKTON 默认 SORT_BY_DOCUMENT=False 即整库入库）。
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
    if file_list:
        wanted = {f"{benchmark}/{p.name}" for p in files} & set(file_list)
        files = [p for p in files if f"{benchmark}/{p.name}" in wanted]
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
        doc.chunks = _chunk_document(doc_id, title, rel, content, max_chars=max_chars,
                                     max_tokens=max_tokens, min_tokens=min_tokens,
                                     overlap_tokens=overlap_tokens, detect_headings=detect_headings)
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
    max_tokens: int | None = None,
    min_tokens: int = 50,
    overlap_tokens: int = 50,
    detect_headings: bool = True,
    max_contracts: int | None = None,
    subsets: list[str] | None = None,
    idx_subset: list[str] | None = None,
) -> dict:
    """把 ContractNLI 的 distinct 合同（前提）整库入库（供检索式分类评测）。

    - 每个前提=一份文档：doc_id = "nli:{idx}"（idx=jsonl 行号，与实例 premise_id 对齐）、
      file_path = "contractnli/{idx}.txt"（占位，无真实路径）、full_text=前提原文；
    - chunk 默认按 token 预算 + 标题边界（对齐正常链路）；max_tokens=None 退回 600 字符定长；
      始终在前提原文偏移上切，回填 search_tokens；
    - 可选 bge-m3 embedding（embed=False 时退化为 BM25-only）。

    idx_subset: 只入库这些 premise_id（对齐 PAKTON：只入被采样实例引用的合同）。
    注意：默认**不**执行，需显式调用/CLI（评测里也只在 --ingest-nli 时入库）。
    """
    from src.document.embedder import embed_texts
    from src.document.postgres_store import connect, init_db, upsert_document
    from src.retrieval.store import backfill_search_tokens
    from src.contract.eval import loaders as L

    premises = L.load_contractnli_premises(Path(jsonl_path), subsets=subsets)
    if idx_subset:
        wanted = set(idx_subset)
        premises = [p for p in premises if p["idx"] in wanted]
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
        doc.chunks = _chunk_document(doc_id, doc.title, file_path, full_text, max_chars=max_chars,
                                     max_tokens=max_tokens, min_tokens=min_tokens,
                                     overlap_tokens=overlap_tokens, detect_headings=detect_headings)
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
                         max_files: int | None = None, max_tokens: int | None = None,
                         min_tokens: int = 50, overlap_tokens: int = 50) -> None:
    """CLI 入口：入库单个 LegalBenchRAG benchmark。"""
    from src.contract.eval import loaders as L

    root_path = L.find_legalbench_root(root)
    t0 = time.time()
    stats = ingest_corpus_dir(root_path, benchmark, session_id, embed=embed, max_files=max_files,
                              max_tokens=max_tokens, min_tokens=min_tokens,
                              overlap_tokens=overlap_tokens)
    stats["elapsed_s"] = time.time() - t0
    print(f"[ingest] 完成 benchmark={benchmark} session={session_id}: {stats}")


def ingest_nli_cli(jsonl_path: str | None, session_id: str, embed: bool = True,
                   max_contracts: int | None = None, max_tokens: int | None = None,
                   min_tokens: int = 50, overlap_tokens: int = 50) -> None:
    """CLI 入口：入库 ContractNLI distinct 合同。"""
    from src.contract.eval import loaders as L

    path = L.find_contractnli_jsonl(jsonl_path)
    t0 = time.time()
    stats = ingest_contractnli_jsonl(path, session_id, embed=embed, max_contracts=max_contracts,
                                     max_tokens=max_tokens, min_tokens=min_tokens,
                                     overlap_tokens=overlap_tokens)
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
    p.add_argument("--chunk-max-tokens", type=int, default=None,
                   help="切片 token 预算（默认按配置/None=600 字符定长）")
    p.add_argument("--chunk-min-tokens", type=int, default=50, help="切片最小 token（默认 50）")
    p.add_argument("--chunk-overlap-tokens", type=int, default=50, help="同章节相邻切片重叠 token（默认 50）")
    p.add_argument("--max-files", type=int, default=None, help="LegalBenchRAG：只入库前 N 个文件")
    p.add_argument("--max-contracts", type=int, default=None, help="nli：只入库前 N 份合同（调试用）")
    args = p.parse_args()
    chunk_kwargs = {"max_tokens": args.chunk_max_tokens,
                    "min_tokens": args.chunk_min_tokens,
                    "overlap_tokens": args.chunk_overlap_tokens}
    if args.dataset == "nli":
        ingest_nli_cli(args.contractnli_jsonl, args.session or "nli-contractnli",
                       embed=not args.no_embed, max_contracts=args.max_contracts, **chunk_kwargs)
    else:
        ingest_benchmark_cli(args.root, args.dataset, args.session or f"lb-{args.dataset}",
                             embed=not args.no_embed, max_files=args.max_files, **chunk_kwargs)
