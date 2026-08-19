"""token 口径统一出处：消息/文本的启发式 token 估算。

全项目"上下文窗口 / 消息长度 / 预算"的计算统一走以下函数（不再用字符数）。
启发式（不引入 tiktoken）：中文字符按 0.6 token/字、其余按空白分词计 1 token。

另外提供一个"运行期 token 账本"（contextvars 承压）：Orchestrator 每次 run 开一个新账本，
Planner / Searcher / Reviewer / Refiner 每次发 LLM 请求时把估算的入参 token 追加进去，用于评测
统计"整条链路所有 Agent"的总用量。用 ContextVar 而非实例属性：评测并发实例各自独立 context，
而 Planner/Reviewer/Refiner 会在实例间共享，实例级计数器会互相踩。
"""
from __future__ import annotations

import re
from contextvars import ContextVar

__all__ = [
    "estimate_tokens",
    "estimate_messages_tokens",
    "enter_token_ledger",
    "exit_token_ledger",
    "append_token_usage",
]

# 简易 token 估算：中文字符按 0.6、其余按空白切分
_ZH_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

# 每条消息的固定开销（OpenAI 类 tokenizer 对每条消息约 4 token 起步）
MESSAGE_OVERHEAD_TOKENS = 4

# ---------------------------------------------------------------------------
# 运行期 token 账本（contextvars）
# ---------------------------------------------------------------------------
_token_ledger: ContextVar[list[int] | None] = ContextVar("token_ledger", default=None)


def enter_token_ledger() -> list[int]:
    """开启一次运行的 token 账本（Orchestrator.run 入口调用），返回账本列表。"""
    ledger: list[int] = []
    _token_ledger.set(ledger)
    return ledger


def exit_token_ledger() -> None:
    """关闭当前账本（Orchestrator.run 收尾调用），避免跨 run 泄漏。"""
    _token_ledger.set(None)


def append_token_usage(tokens: int) -> None:
    """把一次 LLM 调用的入参 token 估算记进账本；无账本（run 之外）时静默跳过。"""
    ledger = _token_ledger.get()
    if ledger is not None:
        ledger.append(int(tokens))



def estimate_tokens(text: str) -> int:
    """粗略估算一段文本的 token 数（不引入 tiktoken 依赖）。"""
    text = str(text)
    zh = len(_ZH_RE.findall(text))
    rest = _ZH_RE.sub(" ", text)
    non_zh = len(rest.split())
    return int(zh * 0.6 + non_zh)


def estimate_messages_tokens(messages: list) -> int:
    """估算 OpenAI 格式消息数组的 token 数（统一口径）。

    逐条累计：content、assistant 的 tool_calls 的 arguments/name、tool 消息的
    tool_call_id/name，并加每条消息的固定开销。
    """
    total = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        total += MESSAGE_OVERHEAD_TOKENS
        content = m.get("content")
        if content:
            total += estimate_tokens(str(content))
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                total += estimate_tokens(str(fn.get("arguments", "")))
                total += estimate_tokens(str(fn.get("name", "")))
        if m.get("tool_call_id"):
            total += estimate_tokens(str(m["tool_call_id"]))
        if m.get("name"):
            total += estimate_tokens(str(m["name"]))
    return total
