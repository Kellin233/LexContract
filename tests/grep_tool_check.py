"""Hermetic 验证：6 工具收敛后的检索预算与 snippet 化（不依赖 DB / LLM / embedding）。

验证：
1. search/grep 输出含 snippet、不含 text；snippet 长度受 SNIPPET_CHARS 控制；
2. search 与 grep 共享同一检索预算（总轮数 + 每轮检索调用数），grep 单独算一轮；
3. 同轮第二个检索调用被 per-round 拦截；跨轮累计超 round 上限被拦截；
4. grep 的 pattern 记入 search_queries；被拦截的调用不计入；
5. DocumentToolkit.get_tools() 只注册 6 个工具，提示词随预算参数化。

用法: python tests/grep_tool_check.py
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

from src.contract.tools import DocumentTool, DocumentToolkit, _around_snippet, _head_snippet
from src.contract.worker import Searcher, build_system_prompt
from src.orchestrator.schemas import SubTask, TaskType


def _chunk(cid: str) -> dict:
    return {
        "id": cid,
        "snippet": f"SNIP_{cid}",
        "doc_id": cid.split(":")[0],
        "doc_title": "d1",
        "session_id": "S1",
        "section_path": ["Art 1"],
        "page_no": 1,
        "charspan": [0, 100],
        "source_format": "pdf",
        "rrf_score": 0.5,
        "rerank_score": None,
    }


def _fake_search(query: str, top_k: int = 10, doc_ids: list[str] | None = None) -> list[dict]:
    return [_chunk("d1:0"), _chunk("d1:1")]


def _fake_grep(pattern: str, mode: str = "literal", top_k: int = 10,
               case_sensitive: bool = False, doc_id: str | None = None) -> list[dict]:
    if pattern == "terminat":
        return [_chunk("d1:1"), _chunk("d1:2")]
    if pattern == "zzz":
        return [_chunk("d1:4")]
    return []


def _fake_list_documents() -> list[dict]:
    return [{"doc_id": "d1", "title": "d1", "source_format": "pdf"}]


def _fake_get_chunk(chunk_id: str) -> dict | None:
    return None


def _fake_get_section(doc_id: str, section_path: list[str]) -> dict | None:
    return None


def _fake_get_outline(doc_id: str) -> list[dict]:
    return []


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
                "type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"],
            }, _fake_get_section),
            DocumentTool("get_document_outline", "get_document_outline", {
                "type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"],
            }, _fake_get_outline),
        ]


class FakePolicy:
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
        # 轮1：search（首见两片）
        {"content": "", "tool_calls": [_tool_call("t0", "search", {"query": "termination"})]},
        # 轮2：grep 独立算一轮
        {"content": "", "tool_calls": [_tool_call("t1", "grep", {"pattern": "terminat"})]},
        # 轮3：同轮两个检索调用 → search 执行，grep 被 per-round 拦截
        {"content": "", "tool_calls": [
            _tool_call("t2", "search", {"query": "terminate"}),
            _tool_call("t3", "grep", {"pattern": "excluded"}),
        ]},
        # 轮4：grep 新切片（round=3 仍可用）
        {"content": "", "tool_calls": [_tool_call("t4", "grep", {"pattern": "zzz"})]},
        # 轮5：round 上限已满 → 被 round-budget 拦截
        {"content": "", "tool_calls": [_tool_call("t5", "grep", {"pattern": "blocked"})]},
        # 最终：输出证据 JSON
        {"content": json.dumps([{
            "source_chunk_ids": ["d1:0"],
            "relevance_note": "termination clause",
        }], ensure_ascii=False), "tool_calls": []},
    ]


async def _run(max_rounds: int = 4, max_per_round: int = 1):
    searcher = Searcher(
        name="searcher",
        policy=FakePolicy(_responses()),
        toolkit=FakeToolkit(),
        assembler=SimpleNamespace(materialize=lambda cand, qid: {"ev": cand}),
        verifier=SimpleNamespace(verify=lambda ev: True),
        store=SimpleNamespace(register=lambda ev, qid: (ev, True)),
        max_turns=8,
        dedup_tool_results=True,
        max_search_rounds=max_rounds,
        max_searches_per_round=max_per_round,
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

    # snippet 函数：长度受 SNIPPET_CHARS 控制
    long_text = "x" * 500
    if len(_head_snippet(long_text)) != 200:
        failed.append(f"_head_snippet 默认应 200 字符，实际 {len(_head_snippet(long_text))}")
    if len(_head_snippet(long_text, 50)) != 50:
        failed.append("_head_snippet 应支持显式 limit")
    around = _around_snippet("abcXYZdef", 3, 6, 50)
    if "XYZ" not in around or len(around) > 50:
        failed.append(f"_around_snippet 应以命中为中心且不超 limit: {around!r}")

    # 工具注册面：恰好 6 个
    real_tk = DocumentToolkit()
    names = sorted(t.name for t in real_tk.get_tools())
    expected = ["get_chunk", "get_document_outline", "get_section", "grep", "list_documents", "search"]
    if names != expected:
        failed.append(f"工具注册应为 {expected}，实际 {names}")

    searcher, result = asyncio.run(_run())
    tool = _tool_results(result)

    # 轮1 search：两片 snippet（不含 text）
    r0 = tool[0]["result"]
    assert all("text" not in x for x in r0), "search 不应返回 text"
    assert [x["snippet"] for x in r0] == ["SNIP_d1:0", "SNIP_d1:1"], "search 应返回 snippet"
    # 轮2 grep：pattern 计入 search_queries，输出为 snippet
    r1 = tool[1]["result"]
    assert all("text" not in x for x in r1), "grep 不应返回 text"
    assert [x["snippet"] for x in r1] == ["SNIP_d1:1", "SNIP_d1:2"], "grep 应返回 snippet"
    # 轮3：search 执行；同轮 grep 被 per-round 拦截
    r2 = tool[2]["result"]
    assert [x["snippet"] for x in r2] == ["SNIP_d1:0", "SNIP_d1:1"], "轮3 search 应返回 snippet"
    assert tool[3]["result"].get("error", "").startswith("At most 1 retrieval call"), \
        f"同轮第二个检索调用应被 per-round 拦截: {tool[3]['result']}"
    # 轮4 grep：新命中
    assert [x["snippet"] for x in tool[4]["result"]] == ["SNIP_d1:4"], "轮4 grep 应返回新切片 snippet"
    # 轮5：round 上限耗尽 → 拦截
    assert "Search-round budget exhausted" in str(tool[5]["result"]), f"超轮数应被拦截: {tool[5]['result']}"

    if searcher._deduped_count != 0:  # noqa: SLF001  (snippet 不参与全文去重)
        failed.append(f"snippet 不应触发全文去重，实际 {searcher._deduped_count}")  # noqa: SLF001

    sq = result.output.search_queries
    for expect in ("termination", "terminat", "terminate", "zzz"):
        if expect not in sq:
            failed.append(f"search_queries 缺 {expect!r}，实际 {sq}")
    if any(x in sq for x in ("excluded", "blocked")):
        failed.append(f"被拦截的检索不应记入 search_queries: {sq}")
    if not result.output.searched:
        failed.append("searched 应为 True")

    # 提示词：列出 search/grep + 预算参数化
    p_31 = build_system_prompt(3, 1)
    for tok in ("search", "grep", "get_section", "get_chunk", "get_document_outline", "list_documents",
                "at most 3 ROUNDS may issue retrieval calls", "1 retrieval call per such round"):
        if tok not in p_31:
            failed.append(f"build_system_prompt(3,1) 缺 {tok!r}")
    p_13 = build_system_prompt(1, 3)
    for tok in ("only 1 ROUND may issue retrieval calls", "up to 3 retrieval calls"):
        if tok not in p_13:
            failed.append(f"build_system_prompt(1,3) 缺 {tok!r}")

    if failed:
        print("FAIL")
        for f in failed:
            print(f"  - {f}")
        return 1

    print(f"PASS  (tokens={result.token_usage}, status={result.status.value}; "
          f"search_queries={sq}; tools={names})")
    print("  - search/grep 只返回 snippet、不含 text ✓   snippet 长度受 SNIPPET_CHARS 控制 ✓")
    print("  - grep 独立算一轮 ✓   同轮第二个检索被 per-round 拦截 ✓   超 round 上限被拦截 ✓")
    print("  - 被拦截调用不入 search_queries ✓   工具注册恰为 6 个 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
