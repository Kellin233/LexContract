"""Hermetic 验证：Searcher 工具结果去重（不依赖 DB / LLM / embedding 模型）。

6 工具收敛后：search/grep 只返回 snippet、不参与全文去重；完整原文只由
get_chunk / get_section 承载并按切片/章节去重。验证：
1. search 同义词重叠时 snippet 不做全文 stub；
2. get_chunk 同一切片第二次返回时正文被替换为短标记；
3. get_section 同章节第二次引用时正文被替换为短标记；
4. 去重开 vs 关：dedup 计数正确、total_tokens 更小、首见正文仍然完整。

用法: python tests/tool_result_dedup_check.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.contract.tools import DocumentTool
from src.contract.worker import Searcher
from src.orchestrator.schemas import SubTask, TaskType


def _chunk_stub(cid: str) -> str:
    return f"[already shown earlier — chunk {cid}; full text omitted]"


def _section_stub(label: str) -> str:
    return f"[already shown earlier — section {label}; full text omitted]"


def _full(cid: str) -> str:
    return (f"TEXT_{cid} " * 25).strip()


def _chunk_snippet(cid: str) -> dict:
    return {
        "id": cid,
        "snippet": f"SNIP_{cid}",
        "doc_id": cid.split(":")[0],
        "doc_title": "doc1",
        "session_id": "S1",
        "section_path": ["Art 1"],
        "page_no": 1,
        "charspan": [0, 100],
        "source_format": "pdf",
        "rrf_score": 0.5,
        "rerank_score": None,
    }


def _chunk_full(cid: str) -> dict:
    return {
        "id": cid,
        "text": _full(cid),
        "doc_id": cid.split(":")[0],
        "doc_title": "doc1",
        "section_path": ["Art 1"],
        "page_no": 1,
        "charspan": [0, 100],
        "source_format": "pdf",
    }


def _fake_search(query: str, top_k: int = 10, doc_ids: list[str] | None = None) -> list[dict]:
    if query == "termination":
        return [_chunk_snippet("d1:0"), _chunk_snippet("d1:1"), _chunk_snippet("d1:2")]
    if query == "terminate":  # 同义词 → top-3 与上一轮高度重叠，但 snippet 不参与去重
        return [_chunk_snippet("d1:1"), _chunk_snippet("d1:2"), _chunk_snippet("d1:3")]
    return []


def _fake_grep(pattern: str, mode: str = "literal", top_k: int = 10,
               case_sensitive: bool = False, doc_id: str | None = None) -> list[dict]:
    return []


def _fake_get_chunk(chunk_id: str) -> dict:
    return _chunk_full(chunk_id)


def _section_full() -> str:
    return (f"SECTION " * 30).strip()


def _fake_get_section(doc_id: str, section_path: list[str]) -> dict:
    return {
        "doc_id": doc_id,
        "section_path": list(section_path or []),
        "start_offset": 0, "end_offset": 20,
        "text": _section_full(),
        "chunk_ids": ["d1:0", "d1:1", "d1:2"],
    }


def _fake_get_outline(doc_id: str) -> list[dict]:
    return [{"doc_id": doc_id, "section_path": ["Art 1"], "start_offset": 0, "end_offset": 20}]


def _fake_list_documents() -> list[dict]:
    return [{"doc_id": "d1", "title": "doc1", "source_format": "pdf"}]


class FakeToolkit:
    """只实现 Searcher 用到的 set_scope / get_tools，handler 全部为纯函数伪实现。"""

    def set_scope(self, session_id: str, doc_ids: list[str] | None = None) -> None:
        pass

    def get_tools(self) -> list[DocumentTool]:
        return [
            DocumentTool("list_documents", "list", {
                "type": "object", "properties": {}, "required": [],
            }, _fake_list_documents),
            DocumentTool("search", "search", {
                "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
            }, _fake_search),
            DocumentTool("grep", "grep", {
                "type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"],
            }, _fake_grep),
            DocumentTool("get_chunk", "get_chunk", {
                "type": "object", "properties": {"chunk_id": {"type": "string"}}, "required": ["chunk_id"],
            }, _fake_get_chunk),
            DocumentTool("get_section", "get_section", {
                "type": "object", "properties": {"doc_id": {"type": "string"}, "section_path": {"type": "array"}},
                "required": ["doc_id", "section_path"],
            }, _fake_get_section),
            DocumentTool("get_document_outline", "get_document_outline", {
                "type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"],
            }, _fake_get_outline),
        ]


class FakePolicy:
    """按脚本返回预置响应（每次调用弹出一条）。"""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)

    def set_tools(self, *args, **kwargs) -> None:
        pass

    def __call__(self, messages: list) -> dict:
        if not self._responses:
            raise RuntimeError("FakePolicy 响应耗尽")
        return self._responses.pop(0)


def _tool_call(uid: str, name: str, arguments: dict) -> dict:
    return {"id": uid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}


def _responses() -> list[dict]:
    return [
        {"content": "", "tool_calls": [_tool_call("t0", "search", {"query": "termination"})]},
        {"content": "", "tool_calls": [_tool_call("t1", "search", {"query": "terminate"})]},
        {"content": "", "tool_calls": [_tool_call("t2", "get_chunk", {"chunk_id": "d1:1"})]},
        {"content": "", "tool_calls": [_tool_call("t3", "get_chunk", {"chunk_id": "d1:1"})]},
        {"content": "", "tool_calls": [_tool_call("t4", "get_section", {"doc_id": "d1", "section_path": ["Art 1"]})]},
        {"content": "", "tool_calls": [_tool_call("t5", "get_section", {"doc_id": "d1", "section_path": ["Art 1"]})]},
        {"content": json.dumps([{
            "source_chunk_ids": ["d1:0"],
            "relevance_note": "termination clause",
        }], ensure_ascii=False), "tool_calls": []},
    ]


async def _run(dedup: bool):
    searcher = Searcher(
        name="searcher",
        policy=FakePolicy(_responses()),
        toolkit=FakeToolkit(),
        assembler=SimpleNamespace(materialize=lambda cand, qid: {"ev": cand}),
        verifier=SimpleNamespace(verify=lambda ev: True),
        store=SimpleNamespace(register=lambda ev, qid: (ev, True)),
        max_turns=8,
        dedup_tool_results=dedup,
    )
    subtask = SubTask(
        task_id="q-1",
        task_type=TaskType.EVIDENCE,
        description="找所有与解除/终止相关的条款",
        dependencies=[],
        timeout_seconds=300,
        priority=1,
        expected_type="evidence",
        search_hints=[],
    )
    result = await searcher.run(subtask, {"doc_ids": [], "question": "解除条款"})
    return searcher, result


def _tool_results(result) -> list[dict]:
    return [e for e in result.trajectory if e.get("role") == "tool"]


def main() -> int:
    failed: list[str] = []

    on_searcher, result_on = asyncio.run(_run(dedup=True))
    tool = _tool_results(result_on)
    assert len(tool) == 6, f"期望 6 条 tool 记录，实际 {len(tool)}"

    # 第 1/2 次 search：snippet 不参与全文去重，两次都原样返回
    r0 = tool[0]["result"]
    assert [x["snippet"] for x in r0] == ["SNIP_d1:0", "SNIP_d1:1", "SNIP_d1:2"], "首次 search 应返回 snippet"
    r1 = tool[1]["result"]
    assert [x["snippet"] for x in r1] == ["SNIP_d1:1", "SNIP_d1:2", "SNIP_d1:3"], \
        "search 同义词重叠时 snippet 不应被 stub"
    # get_chunk 首次 → 全文保留；第二次同一切片 → stub
    assert tool[2]["result"]["text"] == _full("d1:1"), "首次 get_chunk 应保留全文"
    assert tool[3]["result"]["text"] == _chunk_stub("d1:1"), "重复 get_chunk 应去重"
    # get_section 首次 → 全文；同章节二次 → stub
    assert tool[4]["result"]["text"] == _section_full(), "首次 get_section 应保留全文"
    assert tool[5]["result"]["text"] == _section_stub("Art 1"), "同章节二次 get_section 应去重"

    if on_searcher._deduped_count != 2:  # noqa: SLF001  (get_chunk 1 + get_section 1)
        failed.append(f"deduped_count 期望 2，实际 {on_searcher._deduped_count}")  # noqa: SLF001

    # 对照：关闭去重 → 全文全部保留、deduped_count=0、token 更高
    off_searcher, result_off = asyncio.run(_run(dedup=False))
    tool_off = _tool_results(result_off)
    if tool_off[3]["result"]["text"] != _full("d1:1"):
        failed.append("关闭去重时重复 get_chunk 不应被改")
    if tool_off[5]["result"]["text"] != _section_full():
        failed.append("关闭去重时重复 get_section 不应被改")
    if off_searcher._deduped_count != 0:  # noqa: SLF001
        failed.append(f"关闭去重时 deduped_count 应 0，实际 {off_searcher._deduped_count}")  # noqa: SLF001
    if not (result_on.token_usage < result_off.token_usage):
        failed.append(f"去重后 token 应更小: on={result_on.token_usage} off={result_off.token_usage}")

    if failed:
        print("FAIL")
        for f in failed:
            print(f"  - {f}")
        return 1

    print(f"PASS  (dedup on: {result_on.token_usage} tokens / status={result_on.status.value}; "
          f"dedup off: {result_off.token_usage} tokens)")
    print("  - search/grep snippet 不触发全文去重 ✓")
    print("  - get_chunk 重复返回被 stub ✓   get_section 同章节二次引用被 stub ✓")
    print("  - deduped_count=2 ✓   token 下降 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
