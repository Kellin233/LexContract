"""Hermetic 验证：新检索工具（search_* 拆分 + grep）的预算与去重（不依赖 DB / LLM / embedding）。

search 的三种模式拆成 search_vector / search_bm25 / search_hybrid 三个独立工具，另加 grep
（字面/正则精确匹配）；四个检索工具共享同一检索预算（总轮数 + 每轮检索调用数）。本脚本验证：

1. 同一切片无论先经 search_* 还是 grep 首次注入原文唯一（共用 _seen_chunk_ids 去重集合）；
2. 同一轮里第二个检索调用被"每轮超限"note 拦截（per-round cap）；
3. 跨轮累计超过 round 上限被拦截；grep 单独算一个检索轮；
4. grep 的 pattern 记入 search_queries / searched；
5. build_system_prompt 列出四个检索工具且随预算参数化。

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

from src.contract.tools import DocumentTool
from src.contract.worker import Searcher, build_system_prompt
from src.orchestrator.schemas import SubTask, TaskType


def _full(cid: str) -> str:
    return (f"TEXT_{cid} " * 25).strip()


def _chunk_stub(cid: str) -> str:
    return f"[already shown earlier — chunk {cid}; full text omitted]"


def _chunk(cid: str) -> dict:
    return {
        "id": cid,
        "text": _full(cid),
        "doc_id": cid.split(":")[0],
        "doc_title": "d1",
        "section_path": ["Art 1"],
        "page_no": 1,
        "charspan": [0, 100],
        "source_format": "pdf",
    }


def _fake_search_bm25(query: str, top_k: int = 20) -> list[dict]:
    return [_chunk("d1:0"), _chunk("d1:1")]


def _fake_search_vector(query: str, top_k: int = 20) -> list[dict]:
    return [_chunk("d1:2"), _chunk("d1:3")]


def _fake_search_hybrid(query: str, top_k: int = 20) -> list[dict]:
    return []


def _fake_grep(pattern: str, mode: str = "literal", top_k: int = 20,
               case_sensitive: bool = False) -> list[dict]:
    # 与 search_bm25 有重叠切片 d1:1，用来验证跨工具共用去重集合
    if pattern == "terminat":
        return [_chunk("d1:1"), _chunk("d1:2")]
    if pattern == "zzz":
        return [_chunk("d1:4")]
    return []


class FakeToolkit:
    def set_scope(self, session_id: str, doc_ids: list[str] | None = None) -> None:
        pass

    def get_tools(self) -> list[DocumentTool]:
        return [
            DocumentTool("search_bm25", "bm25", {
                "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
            }, _fake_search_bm25),
            DocumentTool("search_vector", "vector", {
                "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
            }, _fake_search_vector),
            DocumentTool("search_hybrid", "hybrid", {
                "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
            }, _fake_search_hybrid),
            DocumentTool("grep", "grep", {
                "type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"],
            }, _fake_grep),
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
        # 轮1：search_bm25（首见两片）
        {"content": "", "tool_calls": [_tool_call("t0", "search_bm25", {"query": "termination"})]},
        # 轮2：grep（d1:1 已见→stub；d1:2 新→全文）——验证跨工具去重 + grep 独立算一轮
        {"content": "", "tool_calls": [_tool_call("t1", "grep", {"pattern": "terminat"})]},
        # 轮3：同轮两个检索调用 → search_vector 执行，grep 被 per-round 拦截
        {"content": "", "tool_calls": [
            _tool_call("t2", "search_vector", {"query": "terminate"}),
            _tool_call("t3", "grep", {"pattern": "excluded"}),
        ]},
        # 轮4：grep 新切片（round=3 仍可用）
        {"content": "", "tool_calls": [_tool_call("t4", "grep", {"pattern": "zzz"})]},
        # 轮5：round 上限已满（4 轮）→ 被 round-budget 拦截
        {"content": "", "tool_calls": [_tool_call("t5", "grep", {"pattern": "blocked"})]},
        # 最终：输出证据 JSON
        {"content": json.dumps([{
            "doc_id": "d1", "start_offset": 0, "end_offset": 20,
            "section_path": ["Art 1"], "source_chunk_ids": ["d1:0"],
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
    searcher, result = asyncio.run(_run())
    tool = _tool_results(result)

    # 轮1 search_bm25：两片首见全文
    r0 = tool[0]["result"]
    assert [x["text"] for x in r0] == [_full("d1:0"), _full("d1:1")], f"轮1 search_bm25 应保留全文: {[x['text'] for x in r0]}"
    # 轮2 grep：d1:1 与 search_bm25 共享去重集合 → stub；d1:2 新 → 全文
    r1 = tool[1]["result"]
    assert [x["text"] for x in r1] == [_chunk_stub("d1:1"), _full("d1:2")], \
        f"grep 应复用 search 已见集合: {[x['text'] for x in r1]}"
    # 轮3：search_vector 执行（d1:2 已见→stub，d1:3 新）；同轮 grep 被 per-round 拦截
    r2 = tool[2]["result"]
    assert [x["text"] for x in r2] == [_chunk_stub("d1:2"), _full("d1:3")], \
        f"轮3 search_vector 应去重 d1:2: {[x['text'] for x in r2]}"
    assert hasattr(tool[3]["result"], "get") and tool[3]["result"].get("error", "").startswith("At most 1 retrieval call"), \
        f"同轮第二个检索调用应被 per-round 拦截: {tool[3]['result']}"
    # 轮4 grep：新切片全文
    assert [x["text"] for x in tool[4]["result"]] == [_full("d1:4")], f"轮4 grep 新切片应全文: {tool[4]['result']}"
    # 轮5：round 上限耗尽 → 拦截
    assert "Search-round budget exhausted" in str(tool[5]["result"]), f"超轮数应被 round 拦截: {tool[5]['result']}"

    if searcher._deduped_count != 2:  # noqa: SLF001  (重复：d1:1、d1:2 各 1 次)
        failed.append(f"deduped_count 期望 2，实际 {searcher._deduped_count}")  # noqa: SLF001
    # grep 的 pattern 记入 search_queries，且 tracked 到拦截前的合法调用
    sq = result.output.search_queries
    for expect in ("termination", "terminat", "terminate", "zzz"):
        if expect not in sq:
            failed.append(f"search_queries 缺 {expect!r}，实际 {sq}")
    if any(x in sq for x in ("excluded", "blocked")):  # 被拦截的调用不应计入
        failed.append(f"被拦截的检索不应记入 search_queries: {sq}")
    if not result.output.searched:
        failed.append("searched 应为 True")

    # 提示词：列出四检索工具 + 预算参数化
    p_31 = build_system_prompt(3, 1)
    for tok in ("search_vector", "search_bm25", "search_hybrid", "grep",
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
          f"search_queries={sq}; deduped={searcher._deduped_count})")  # noqa: SLF001
    print("  - 跨工具(±grep)去重集合共享 ✓   grep 独立算一轮 ✓")
    print("  - 同轮第二个检索被 per-round 拦截 ✓   超 round 上限被拦截 ✓   被拦截调用不入 search_queries ✓")
    print("  - 提示词列出四检索工具且随预算参数化 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
