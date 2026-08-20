"""证据物化链路回归检查（不写数据库、不依赖 LLM）。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.contract.assembler import EvidenceAssembler
from src.contract.schemas import Evidence
from src.contract.store import EvidenceStore
from src.contract.tools import DocumentToolkit
from src.contract.verifier import CitationVerifier
from src.contract.worker import Searcher
from src.document import chunker
from src.document.chunker import chunk_blocks
from src.document.models import ParsedBlock
from src.document.parser import _assemble_full_text
from src.retrieval import store as retrieval_store


class MemoryToolkit:
    def __init__(self, full_text: str, rows: list[dict]):
        self.full_text = full_text
        self.rows = {row["id"]: row for row in rows}

    def get_chunk(self, chunk_id: str) -> dict | None:
        return self.rows.get(chunk_id)

    def get_full_text(self, _doc_id: str) -> str:
        return self.full_text

    def get_document(self, doc_id: str) -> dict:
        return {"doc_id": doc_id, "title": "Doc"}

    def get_page_for_span(self, _doc_id: str, start: int, _end: int) -> int:
        return start // 100 + 1

    def get_section(self, _doc_id: str, _section_path: list[str]) -> dict | None:
        return None


def _rows_from_chunks(full_text: str, chunks) -> list[dict]:
    rows = []
    for chunk in chunks:
        span = list(chunk.metadata.charspan)
        rows.append({
            "id": chunk.id,
            "text": chunk.text,
            "doc_id": "doc1",
            "doc_title": "Doc",
            "section_path": list(chunk.metadata.section_path),
            "page_no": 1,
            "charspan": span,
            "source_format": "txt",
        })
        assert chunk.text == full_text[span[0]:span[1]]
    return rows


def check_chunk_spans_and_late_materialization() -> None:
    old_cap, old_overlap = chunker._TOKEN_CAP, chunker._OVERLAP_TOKENS
    try:
        chunker._TOKEN_CAP = 50
        chunker._OVERLAP_TOKENS = 10
        text = " ".join(
            f"Sentence {i:04d} contains unique clause wording marker-{i}."
            for i in range(180)
        )
        blocks = [ParsedBlock(kind="content", text=text, page_no=1, offset=[])]
        full_text = _assemble_full_text(blocks)
        chunks = chunk_blocks(blocks, "doc1", "Doc", "txt")
        assert len(chunks) > 3
        spans = [tuple(c.metadata.charspan) for c in chunks]
        assert len(spans) == len(set(spans)), "发现只包含 overlap 的重复尾部切片"
        rows = _rows_from_chunks(full_text, chunks)

        toolkit = MemoryToolkit(full_text, rows)
        assembler = EvidenceAssembler(toolkit)
        verifier = CitationVerifier(toolkit)
        last = chunks[-1]
        evidence = assembler.materialize({"source_chunk_ids": [last.id]}, "Q1")
        assert evidence is not None
        assert evidence.quote == last.text
        assert (evidence.start_offset, evidence.end_offset) == tuple(last.metadata.charspan)
        assert verifier.verify(evidence)
    finally:
        chunker._TOKEN_CAP, chunker._OVERLAP_TOKENS = old_cap, old_overlap


def check_parser_block_alignment() -> None:
    blocks = [
        ParsedBlock(kind="heading", level=1, text="Article 1"),
        ParsedBlock(kind="content", text="Payment is due."),
    ]
    full_text = _assemble_full_text(blocks)
    chunks = chunk_blocks(blocks, "doc1", "Doc", "txt")
    assert chunks
    for chunk in chunks:
        start, end = chunk.metadata.charspan
        assert chunk.text == full_text[start:end]


def check_verifier_defenses() -> None:
    full = "alpha beta gamma"
    good = {
        "id": "doc1:0", "text": "alpha", "doc_id": "doc1",
        "section_path": ["S"], "charspan": [0, 5],
    }
    wrong_text = {**good, "id": "doc1:1", "text": "wrong"}
    wrong_section = {**good, "id": "doc1:2", "section_path": ["Other"]}
    toolkit = MemoryToolkit(full, [good, wrong_text, wrong_section])
    verifier = CitationVerifier(toolkit)

    mismatch = Evidence(
        document_id="doc1", section_path=["S"], source_chunk_ids=["doc1:1"],
        start_offset=0, end_offset=5, quote="alpha",
    )
    assert not verifier.verify(mismatch)
    assert mismatch.verify_note == "chunk-text-mismatch"

    off_section = Evidence(
        document_id="doc1", section_path=["S"], source_chunk_ids=["doc1:2"],
        start_offset=0, end_offset=5, quote="alpha",
    )
    assert not verifier.verify(off_section)
    assert off_section.verify_note == "off-section-chunk"

    second = {"id": "doc1:3", "text": " beta", "doc_id": "doc1",
              "section_path": ["S"], "charspan": [5, 10]}
    toolkit.rows[second["id"]] = second
    multi = Evidence(
        document_id="doc1", section_path=["S"], source_chunk_ids=["doc1:0", "doc1:3"],
        start_offset=0, end_offset=10, quote=full[:10],
    )
    assert verifier.verify(multi)


class _WorkerToolkit:
    def get_tools(self):
        return []


class _WorkerAssembler:
    def materialize(self, candidate, question_id):
        if candidate.get("kind") == "raise":
            raise ValueError("invalid chunk id")
        start = int(candidate["start"])
        return Evidence(
            question_id=question_id, document_id="doc1", start_offset=start,
            end_offset=start + 5, quote=f"text{start}",
        )


class _WorkerVerifier:
    def verify(self, evidence):
        if evidence.start_offset == 10:
            raise RuntimeError("database lookup failed")
        return True


def check_worker_exception_isolation() -> None:
    searcher = Searcher(
        name="searcher", policy=lambda _messages: {"content": "[]"},
        toolkit=_WorkerToolkit(), assembler=_WorkerAssembler(),
        verifier=_WorkerVerifier(), store=EvidenceStore(),
    )
    result = searcher._assemble_worker_result(
        [{"start": 0}, {"kind": "raise"}, {"start": 5}, {"start": 10}],
        "Q1", "q", True, [], EvidenceStore(),
    )
    assert len(result.evidences) == 2
    assert result.materialize_failed_count == 1
    assert result.verifier_rejected_count == 1
    assert result.drop_reasons == {
        "materialize-exception": 1,
        "verify-exception": 1,
    }


class _PageCursor:
    def __init__(self):
        self.sql = ""
        self.params = ()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql, params=()):
        self.sql = sql
        self.params = tuple(params)
        return self

    def fetchone(self):
        return (7,)


class _PageConnection:
    def __init__(self, cursor):
        self.cursor_obj = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return self.cursor_obj


def check_cross_chunk_page_lookup() -> None:
    cursor = _PageCursor()
    original_connect = retrieval_store.connect
    retrieval_store.connect = lambda: _PageConnection(cursor)
    try:
        toolkit = object.__new__(DocumentToolkit)
        toolkit.session_id = "S1"
        toolkit._doc_ids = []
        toolkit._assert_scope = lambda _doc_id: None
        assert toolkit.get_page_for_span("doc1", 10, 30) == 7
    finally:
        retrieval_store.connect = original_connect
    compact_sql = " ".join(cursor.sql.split()).lower()
    assert "c.charspan[2] > %s" in compact_sql
    assert "c.charspan[1] < %s" in compact_sql
    assert cursor.params == ("doc1", "S1", 10, 30)


def main() -> int:
    checks = [
        check_chunk_spans_and_late_materialization,
        check_parser_block_alignment,
        check_verifier_defenses,
        check_worker_exception_isolation,
        check_cross_chunk_page_lookup,
    ]
    for check in checks:
        check()
    print(f"PASS ({len(checks)} evidence materialization checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
