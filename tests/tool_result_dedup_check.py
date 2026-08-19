"""Hermetic 验证：Searcher 工具结果去重（不依赖 DB / LLM / embedding 模型）。

伪 toolkit 返回重叠切片 / 章节，伪 policy 按脚本返回 tool_calls，验证：
1. 重复切片（search 同义词重叠、get_context 带回底片）正文被替换为短标记；
2. 同章节被第二次引用（get_referenced_section 解析回同一章节）时正文去重；
3. 去重开 vs 关：dedup 计数正确、total_tokens 更小、首见正文仍然完整。

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
    # 正文刻意做得足够长（模拟真实几百 token 的切片），短标记必须显著更省
    return (f"TEXT_{cid} " * 25).strip()


def _chunk(cid: str) -> dict:
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


def _fake_search(query: str, mode: str = "hybrid", top_k: int = 20) -> list[dict]:
    if query == "termination":
        return [_chunk("d1:0"), _chunk("d1:1"), _chunk("d1:2")]
    if query == "terminate":  # 同义词 → top-3 与上一轮高度重叠
        return [_chunk("d1:1"), _chunk("d1:2"), _chunk("d1:3")]
    return []


def _fake_get_context(chunk_id: str, before: int = 2, after: int = 2) -> list[dict]:
    if chunk_id == "d1:1":
        # 底片 d1:1 已由 search 返回过，理应被去重
        return [_chunk("d1:1"), _chunk("d1:2")]
    return []


def _section_full() -> str:
    return (f"SECTION " * 30).strip()


def _fake_get_section(doc_id: str, section_path: list[str] | None = None) -> dict:
    return {
        "doc_id": doc_id,
        "section_path": list(section_path or []),
        "start_offset": 0, "end_offset": 20,
        "text": _section_full(),
        "chunk_ids": ["d1:0", "d1:1", "d1:2"],
    }


def _fake_get_referenced_section(doc_id: str, ref: str) -> dict:
    return _fake_get_section(doc_id, ["Art 1"])  # 与之前的 get_section 同章节


def _fake_get_chunk(chunk_id: str) -> dict:
    return _chunk(chunk_id)


def _fake_get_outline(doc_id: str) -> list[dict]:
    return [{"doc_id": doc_id, "section_path": ["Art 1"], "start_offset": 0, "end_offset": 20}]


class FakeToolkit:
    """只实现 Searcher 用到的 set_scope / get_tools，handler 全部为纯函数伪实现。"""

    def set_scope(self, session_id: str, doc_ids: list[str] | None = None) -> None:
        pass

    def get_tools(self) -> list[DocumentTool]:
        return [
            DocumentTool("search", "search", {
                "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
            }, _fake_search),
            DocumentTool("get_chunk", "get_chunk", {
                "type": "object", "properties": {"chunk_id": {"type": "string"}}, "required": ["chunk_id"],
            }, _fake_get_chunk),
            DocumentTool("get_context", "get_context", {
                "type": "object", "properties": {"chunk_id": {"type": "string"}}, "required": ["chunk_id"],
            }, _fake_get_context),
            DocumentTool("get_section", "get_section", {
                "type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"],
            }, _fake_get_section),
            DocumentTool("get_document_outline", "get_document_outline", {
                "type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"],
            }, _fake_get_outline),
            DocumentTool("get_referenced_section", "get_referenced_section", {
                "type": "object", "properties": {"doc_id": {"type": "string"}, "ref": {"type": "string"}},
                "required": ["doc_id", "ref"],
            }, _fake_get_referenced_section),
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
        {"content": "", "tool_calls": [_tool_call("t2", "get_context", {"chunk_id": "d1:1"})]},
        {"content": "", "tool_calls": [_tool_call("t3", "get_section", {"doc_id": "d1", "section_path": ["Art 1"]})]},
        {"content": "", "tool_calls": [_tool_call("t4", "get_referenced_section", {"doc_id": "d1", "ref": "Art 1"})]},
        {"content": json.dumps([{
            "doc_id": "d1", "start_offset": 0, "end_offset": 20,
            "section_path": ["Art 1"], "source_chunk_ids": ["d1:0"],
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
    assert len(tool) == 5, f"期望 5 条 tool 记录，实际 {len(tool)}"

    # 第 1 次 search：三片全部首见，正文完整
    r0 = tool[0]["result"]
    assert [x["text"] for x in r0] == [_full("d1:0"), _full("d1:1"), _full("d1:2")], "首次 search 应保留全文"
    # 第 2 次 search：d1:1 / d1:2 已见 → 短标记；d1:3 新的 → 全文
    r1 = tool[1]["result"]
    assert [x["text"] for x in r1] == [
        _chunk_stub("d1:1"),
        _chunk_stub("d1:2"),
        _full("d1:3"),
    ], f"二次 search 应去重 d1:1/d1:2: {[x['text'] for x in r1]}"
    # get_context 带回底片 d1:1/d1:2 → 全去重
    r2 = tool[2]["result"]
    assert [_chunk_stub("d1:1"), _chunk_stub("d1:2")] == [x["text"] for x in r2], \
        f"get_context 底片应去重: {[x['text'] for x in r2]}"
    # get_section 首次 → 全文保留
    assert tool[3]["result"]["text"] == _section_full(), "首次 get_section 应保留全文"
    # get_referenced_section 解析回同一章节 → 去重
    assert tool[4]["result"]["text"] == _section_stub("Art 1"), \
        f"同章节二次引用应去重: {tool[4]['result']['text']}"

    dedup_ok = result_on.token_usage, result_on.status.value
    if not (on_searcher._deduped_count == 5):  # noqa: SLF001
        failed.append(f"deduped_count 期望 5，实际 {on_searcher._deduped_count}")  # noqa: SLF001

    # 对照：关闭去重 → 全文全部保留、deduped_count=0、token 更高
    off_searcher, result_off = asyncio.run(_run(dedup=False))
    tool_off = _tool_results(result_off)
    if [x["text"] for x in tool_off[1]["result"]] != [_full("d1:1"), _full("d1:2"), _full("d1:3")]:
        failed.append("关闭去重时二次 search 不应被改")
    if off_searcher._deduped_count != 0:  # noqa: SLF001
        failed.append(f"关闭去重时 deduped_count 应 0，实际 {off_searcher._deduped_count}")  # noqa: SLF001
    if not (result_on.token_usage < result_off.token_usage):
        failed.append(f"去重后 token 应更小: on={result_on.token_usage} off={result_off.token_usage}")

    if failed:
        print("FAIL")
        for f in failed:
            print(f"  - {f}")
        return 1

    print(f"PASS  (dedup on: {dedup_ok[0]} tokens / status={dedup_ok[1]}; "
          f"dedup off: {result_off.token_usage} tokens)")
    print("  - 二次 search 去重 d1:1/d1:2 ✓   get_context 底片去重 ✓")
    print("  - 同章节二次引用去重 ✓   deduped_count=5 ✓   token 下降 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
