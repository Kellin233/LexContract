"""
LexTrace — 核心数据结构定义

所有跨模块传递的数据结构集中定义于此，保证类型一致性和可维护性。
使用 Python 3.10+ 的 | 联合类型语法。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


__all__ = [
    "OrchestratorState",
    "TaskType",
    "AgentStatus",
    "SubTask",
    "AgentResult",
    "ResearchReport",
    "RunConfig",
    "OrchestrationMode",
]


# ============================================================================
# 枚举定义
# ============================================================================

class OrchestratorState(Enum):
    """M1 编排层状态机。

    合同证据链流:   IDLE → PLANNING → DISPATCHING → COLLECTING → REVIEWING
                      → (SUFFICIENT) REFINING
                      → (NEED_MORE) INCREMENTAL_PLANNING → DISPATCHING
                      → (超限/无有效新增) REFINING(PARTIALLY_SUFFICIENT)
    REPLANNING 为旧流程保留，合同流不再使用。
    """
    IDLE = "idle"
    PLANNING = "planning"
    DISPATCHING = "dispatching"
    COLLECTING = "collecting"
    SYNTHESIZING = "synthesizing"
    REPLANNING = "replanning"
    REVIEWING = "reviewing"
    INCREMENTAL_PLANNING = "incremental_planning"
    REFINING = "refining"
    DONE = "done"
    FAILED = "failed"


class TaskType(Enum):
    """Sub-task 的任务类型，决定由哪类 Agent 执行。"""
    SEARCH = "search"
    ANALYZE = "analyze"
    VERIFY = "verify"
    EVIDENCE = "evidence"


class AgentStatus(Enum):
    """单个 Sub-task 的执行结果状态。"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class OrchestrationMode(str, Enum):
    """合同证据链的编排模式。"""

    DIRECT = "direct"
    REVIEWED_INCREMENTAL = "reviewed_incremental"


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class SubTask:
    """规划器生成的原子任务单元。

    Attributes:
        task_id: 全局唯一标识，用于 DAG 依赖引用。
        task_type: 任务类型，决定调度到哪个 Agent。
        description: 自然语言描述，传给 Agent 的指令。
        dependencies: 依赖的 task_id 列表，这些任务完成后才能执行本任务。
        context_keys: 需要从共享 Memory 中读取的上下文键名。
        timeout_seconds: 单任务超时阈值（秒）。
        priority: 优先级，数值越小优先级越高。
        expected_type: 期望结果类型，辅助 Agent 调整输出格式。
        search_hints: 搜索类任务的额外关键词提示。
    """
    task_id: str
    task_type: TaskType
    description: str
    dependencies: list[str] = field(default_factory=list)
    context_keys: list[str] = field(default_factory=list)
    timeout_seconds: int = 300
    priority: int = 1
    expected_type: str = "factual"  # factual | analytical | comparative | temporal
    search_hints: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """Agent 执行 SubTask 后的结果。

    Attributes:
        task_id: 对应 SubTask 的 task_id。
        status: 执行状态（成功/失败/超时）。
        output: 实际输出内容，类型由任务决定（str | dict | list）。
        trajectory: 多轮交互轨迹，用于日志和后续分析。
        token_usage: 本次任务消耗的 token 数。
        confidence: 结果置信度 [0.0, 1.0]。
    """
    task_id: str
    status: AgentStatus
    output: Any = None
    trajectory: list[dict] = field(default_factory=list)
    token_usage: int = 0
    confidence: float = 0.0


@dataclass
class ResearchReport:
    """最终交付给用户的研究报告。

    Attributes:
        query: 原始研究问题。
        content: 报告正文（Markdown 格式）。
        sources: 引用的信息源列表，每条包含 url/title/snippet。
        confidence: 整体置信度。
        num_searches: 实际执行的搜索/分析轮数。
        num_replan: 重规划次数。
        final_score: 最终综合评分（由外部评测模块写入）。
        structured: 合同证据链流程的结构化结果（RefinerResult 的 JSON 字典）。
        telemetry: 本次编排、检索、校验和引用审计的结构化遥测。
    """
    query: str
    content: str
    sources: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    num_searches: int = 0
    num_replan: int = 0
    final_score: float = 0.0
    structured: dict = field(default_factory=dict)
    telemetry: dict = field(default_factory=dict)


@dataclass
class RunConfig:
    """单次运行的全局配置。

    Attributes:
        max_concurrent: 最大并发 Sub-agent 数。
        global_timeout_seconds: 全局硬超时（秒）。
        max_replan_rounds: 最大重规划轮数（deep-research 流）。
        max_sub_questions: 单次规划最多子问题数。
        enable_evolution: 是否启用自我进化（预留 M6 接口）。
        session_id: 合同证据链流的检索会话作用域。
        doc_ids: 合同证据链流限定检索的文档；空=全部。
        max_iterations: 合同证据链流最大研究轮数（Planner→Worker→Reviewer 一圈为 1）。
        enable_evidence_verification: 是否启用 CitationVerifier 原文校验。
        stop_on_no_effective_new_evidence: 无有效新增证据时是否提前停止。
        orchestration_mode: direct 或 reviewed_incremental。
    """
    max_concurrent: int = 5
    global_timeout_seconds: int = 600
    max_replan_rounds: int = 3
    max_sub_questions: int = 8
    enable_evolution: bool = False
    session_id: str = ""
    doc_ids: list[str] = field(default_factory=list)
    max_iterations: int = 3
    enable_evidence_verification: bool = True
    stop_on_no_effective_new_evidence: bool = True
    orchestration_mode: str = OrchestrationMode.REVIEWED_INCREMENTAL.value
