"""ConversationRecorder：把每次 LLM 调用（请求消息 + 响应）以"对话层"逐行写入 jsonl。

设计要点:
  - 独立组件：Agent 层只负责打标签（set_agent），Policy 层在 __call__ 里统一上报；
  - 用 contextvars 传递 run 会话与 agent 标签——asyncio.to_thread 会继承调用方的 context，
    因此 VLLMPolicy.__call__ 在 worker 线程里也能读到当前会话/标签；
  - "对话层"清洗：完整记录 system/user/assistant 消息原文与 assistant 的 tool_calls 意图
    （模型想让工具做什么，属对话层）；工具返回的大 JSON 只留摘要
    （role=tool 消息 -> name + 字符数 + 前 N 字），不做状态快照；
  - 逐行 flush，崩溃不丢；由 configs/default.yaml 的 conversation.enabled 控制，默认关闭。
"""
from __future__ import annotations

import contextvars
import json
import os
import threading
from datetime import datetime
from typing import Any

__all__ = ["ConversationRecorder", "recorder", "set_agent", "agent_label", "run_id_of"]

# 当前运行会话（run_id）与当前 Agent 标签；None/"" 表示未激活
_run_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("conv_run", default=None)
_agent_var: contextvars.ContextVar[str] = contextvars.ContextVar("conv_agent", default="")

# 工具返回结果的摘要长度（对话层：不落盘完整工具返回）
TOOL_SUMMARY_CHARS = 200


def set_agent(label: str) -> None:
    """记录当前 agent 的标签（如 searcher/Q1、planner/initial_plan）。"""
    _agent_var.set(label or "")


def agent_label() -> str:
    return _agent_var.get() or ""


def run_id_of() -> str | None:
    return _run_var.get()


# ---------------------------------------------------------------------------
# 对话层清洗
# ---------------------------------------------------------------------------
def clean_messages(messages: list) -> list[dict]:
    """把消息数组清洗成对话层记录：tool 消息只留摘要，assistant 保留 tool_calls 意图原文。"""
    out: list[dict] = []
    if not messages:
        return out
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "user"))
        if role == "tool":
            body = m.get("content")
            if not isinstance(body, str):
                try:
                    body = json.dumps(body, ensure_ascii=False, default=str)
                except Exception:  # noqa: BLE001
                    body = str(body)
            out.append({
                "role": "tool",
                "name": str(m.get("name", "")),
                "tool_call_id": str(m.get("tool_call_id", "")),
                "result_chars": len(body),
                "result_summary": body[:TOOL_SUMMARY_CHARS],
            })
            continue
        rec: dict[str, Any] = {"role": role}
        if "content" in m:
            rec["content"] = str(m.get("content", ""))
        if role == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                tcs.append({
                    "id": str(tc.get("id", "")),
                    "name": str(fn.get("name", "")),
                    "arguments": str(fn.get("arguments", "")),
                })
            rec["tool_calls"] = tcs
        if m.get("reasoning_content"):
            rec["reasoning_content"] = str(m["reasoning_content"])
        out.append(rec)
    return out


def clean_response(response) -> dict:
    """把 policy 返回的响应（OpenAICompatibleDict）清洗成对话层记录。"""
    resp: dict[str, Any] = {}
    if response is None:
        return resp
    if hasattr(response, "get"):
        resp["content"] = str(response.get("content", ""))
        tcs = response.get("tool_calls") or []
        resp["tool_calls"] = [
            {
                "id": str(tc.get("id", "")) if isinstance(tc, dict) else "",
                "name": str((tc.get("function") or {}).get("name", "")) if isinstance(tc, dict) else "",
                "arguments": str((tc.get("function") or {}).get("arguments", "")) if isinstance(tc, dict) else "",
            }
            for tc in tcs
        ]
        if response.get("reasoning_content"):
            resp["reasoning_content"] = str(response["reasoning_content"])
    else:
        resp["content"] = str(response)
    return resp


def _now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


class ConversationRecorder:
    """本地对话留档器（逐行 JSONL，追加写 + flush）。

    会话由 run 级代码开启（set_session），Agent 标签由各 Agent 通过 set_agent 设置，
    Policy 层调用 record_llm_call 落盘。未开启会话时所有记录方法都是 no-op。
    """

    def __init__(self) -> None:
        self.enabled: bool = False
        self._lock = threading.Lock()
        self._sessions: dict[str, dict] = {}  # run_id -> {"path","fh","turns":{agent:int}}

    # ------------------------------------------------------------------
    # 会话控制（由运行层调用）
    # ------------------------------------------------------------------
    def set_session(self, run_id: str, path: str) -> None:
        """开启一个会话：准备追加写句柄，并把 run_id 绑到当前 context。"""
        if not run_id or not path:
            return
        with self._lock:
            if run_id not in self._sessions:
                parent = os.path.dirname(path) or "."
                if parent:
                    os.makedirs(parent, exist_ok=True)
                self._sessions[run_id] = {
                    "path": path,
                    "fh": open(path, "a", encoding="utf-8"),
                    "turns": {},
                }
        _run_var.set(run_id)

    def clear_session(self, run_id: str | None = None) -> None:
        """关闭会话（默认关闭当前 context 的会话）。"""
        rid = run_id or _run_var.get()
        sess = None
        if rid:
            with self._lock:
                sess = self._sessions.pop(rid, None)
        if sess and sess.get("fh"):
            try:
                sess["fh"].close()
            except Exception:  # noqa: BLE001
                pass
        if rid:
            _run_var.set(None)

    # ------------------------------------------------------------------
    # 记录
    # ------------------------------------------------------------------
    def record(self, kind: str, **fields) -> None:
        """写任意类型的一行（run_start / run_end 等）。"""
        if not self.enabled:
            return
        rid = _run_var.get()
        if not rid:
            return
        line = {"kind": kind, "run_id": rid, "ts": _now(), "agent": agent_label() or None, **fields}
        self._write(rid, line)

    def record_llm_call(
        self,
        messages: list,
        response,
        *,
        usage: dict | None = None,
        elapsed_ms: int | None = None,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        """记录一次 LLM 调用（请求消息 + 响应），对话层深度。"""
        if not self.enabled:
            return
        rid = _run_var.get()
        if not rid:
            return
        label = agent_label()
        with self._lock:
            sess = self._sessions.get(rid)
            if sess is None:
                return
            turn = sess["turns"].get(label, 0) + 1
            sess["turns"][label] = turn
        line = {
            "kind": "llm_call",
            "run_id": rid,
            "ts": _now(),
            "agent": label or None,
            "turn": turn,
            "usage": usage or {},
            "elapsed_ms": elapsed_ms,
            "status": status,
            "error": error,
            "messages": clean_messages(messages),
            "response": clean_response(response),
        }
        self._write(rid, line)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _write(self, rid: str, line: dict) -> None:
        with self._lock:
            sess = self._sessions.get(rid)
            if sess is None:
                return
            try:
                sess["fh"].write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
                sess["fh"].flush()
            except Exception:  # noqa: BLE001  留档失败不能影响主流程
                pass


recorder = ConversationRecorder()
