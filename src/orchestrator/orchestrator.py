"""
LexContract — 合同证据链编排器（改造自 deep-research M1 Orchestrator）

状态机流程：
  IDLE → PLANNING(Planner.initial_plan)
       → DISPATCHING(Searcher 并行，沿用 DAG 分层 + Semaphore + 超时)
       → COLLECTING(EvidenceAssembly → CitationVerifier → EvidenceStore → 写入 ResearchState)
            → REVIEWING(Reviewer：覆盖度/冲突/缺口)
            ├─ SUFFICIENT → REFINING(Refiner → Final Answer)
            ├─ NEED_MORE 且 iteration < max_iterations 且 有有效新增
            │      → INCREMENTAL_PLANNING → DISPATCHING
            └─ 否则（达上限 / 无有效新增）→ REFINING（状态记为 PARTIALLY_SUFFICIENT）
       → DONE / FAILED

direct 消融模式在 COLLECTING 后直接进入 REFINING，跳过 Reviewer 和增量补查。

旧 deep-research 的 SYNTHESIZING / REPLANNING 状态不在合同流中使用。
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from .schemas import (
    OrchestratorState,
    SubTask,
    AgentResult,
    AgentStatus,
    ResearchReport,
    RunConfig,
    TaskType,
    OrchestrationMode,
)
from .agent_pool import AgentPool
from ..planner.dag import DAG
from ..contract.schemas import (
    ResearchState,
    WorkerResult,
    FinalStatus,
    ReviewStatus,
    ReviewResult,
    QuestionStatus,
)
from ..contract.store import EvidenceStore
from ..contract.planner import Planner
from ..contract.reviewer import Reviewer
from ..contract.refiner import Refiner
from ..utils.tokens import enter_token_ledger, exit_token_ledger
from ..utils.tracing import trace_chain

SharedMemoryStore = Any  # M4（可选保留，仅落最终报告）


__all__ = ["Orchestrator"]


class Orchestrator:
    """合同证据链编排器。

    Attributes:
        planner: Planner（initial + incremental）。
        agent_pool: 对象池，EVIDENCE 任务返回 Searcher。
        reviewer / refiner: 完整性与结论模块。
        evidence_store: 运行期证据库（每次 run 重置）。
        compressor: 可选，仅用于压缩规划/审查历史，绝不压缩 Evidence。
        memory_store: 可选 M4，仅用于持久化最终报告。
    """

    def __init__(
        self,
        planner: Planner,
        agent_pool: AgentPool,
        reviewer: Reviewer | None = None,
        refiner: Refiner | None = None,
        evidence_store: EvidenceStore | None = None,
        compressor: Any | None = None,
        memory_store: Any | None = None,
    ) -> None:
        self.planner = planner
        self.agent_pool = agent_pool
        self.reviewer = reviewer
        self.refiner = refiner
        self.compressor = compressor
        self.memory_store = memory_store

        # 运行期状态
        self._runtime: dict[str, Any] = {}
        self._results: list[AgentResult] = []
        self._dag: DAG | None = None
        self._task_map: dict[str, SubTask] = {}
        self._current_state = OrchestratorState.IDLE
        self._query: str = ""
        self._config: RunConfig = RunConfig()
        self._start_time: float = 0.0
        # EvidenceStore 实现了 __len__，空库在 bool() 下为 False；必须按 None 判断，
        # 否则调用方传入的空库会被静默替换。
        self._evidence_store = evidence_store if evidence_store is not None else EvidenceStore()
        self._research_state: ResearchState | None = None
        self._num_searches = 0
        self._planning_rounds = 0
        self._searcher_count = 0
        self._search_tool_call_count = 0
        self._searcher_token_usage_total = 0
        self._reviewer_calls = 0
        self._reviewer_sufficient = False
        self._candidate_count_total = 0
        self._materialize_failed_count = 0
        self._verifier_rejected_count = 0
        self._verified_evidence_count = 0
        self._drop_reasons: dict[str, int] = {}
        self._stop_reason = ""
        # 本次运行所有 Agent（Planner/Searcher/Reviewer/Refiner）的 LLM token 账本（run 时开启）
        self._token_ledger: list[int] = []

        self._state_handlers: dict[OrchestratorState, Callable[[], asyncio.Future[OrchestratorState]]] = {
            OrchestratorState.IDLE: self._on_idle,
            OrchestratorState.PLANNING: self._do_planning,
            OrchestratorState.DISPATCHING: self._do_dispatching,
            OrchestratorState.COLLECTING: self._do_collecting,
            OrchestratorState.REVIEWING: self._do_reviewing,
            OrchestratorState.INCREMENTAL_PLANNING: self._do_incremental_planning,
            OrchestratorState.REFINING: self._do_refining,
            OrchestratorState.DONE: self._on_done,
            OrchestratorState.FAILED: self._on_failed,
        }

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------
    @trace_chain(name="orchestrator.run", tags=["contract", "orchestrator"])
    async def run(self, query: str, config: RunConfig | None = None) -> ResearchReport:
        self._query = query
        self._config = config or RunConfig()
        self._start_time = time.monotonic()
        self._runtime.clear()
        self._results.clear()
        self._dag = None
        self._task_map.clear()
        self._current_state = OrchestratorState.IDLE
        self._evidence_store = EvidenceStore()  # 每次运行独立证据库
        self._research_state = None
        self._num_searches = 0
        self._planning_rounds = 0
        self._searcher_count = 0
        self._search_tool_call_count = 0
        self._searcher_token_usage_total = 0
        self._reviewer_calls = 0
        self._reviewer_sufficient = False
        self._candidate_count_total = 0
        self._materialize_failed_count = 0
        self._verifier_rejected_count = 0
        self._verified_evidence_count = 0
        self._drop_reasons = {}
        self._stop_reason = ""

        self._token_ledger = enter_token_ledger()
        try:
            while self._current_state not in (OrchestratorState.DONE, OrchestratorState.FAILED):
                if self._is_global_timeout():
                    if self._current_state in (
                        OrchestratorState.DISPATCHING,
                        OrchestratorState.COLLECTING,
                        OrchestratorState.REVIEWING,
                        OrchestratorState.INCREMENTAL_PLANNING,
                    ):
                        print("[Timeout] 全局超时，强制进入 Refiner（使用已有证据）")
                        self._stop_reason = "global_timeout"
                        if self._research_state is not None:
                            self._research_state.final_status = FinalStatus.PARTIALLY_SUFFICIENT
                        self._current_state = OrchestratorState.REFINING
                    else:
                        self._current_state = OrchestratorState.FAILED
                    continue

                handler = self._state_handlers.get(self._current_state)
                if handler is None:
                    raise RuntimeError(f"Unknown state: {self._current_state}")
                next_state = await handler()
                self._current_state = next_state
                print(f"[Orchestrator] State transition: {self._current_state.value}")
        finally:
            exit_token_ledger()

        if self._current_state == OrchestratorState.DONE:
            report = self._runtime.get("final_report")
            if report is None:
                report = ResearchReport(query=query, content="Report generation failed unexpectedly.")
            report.num_searches = self._num_searches
            report.num_replan = max(0, (self._research_state.iteration if self._research_state else 1) - 1)
            report.telemetry = self.last_run_telemetry(report)
            if self.memory_store is not None:
                try:
                    self._store_final_to_memory(report)
                except Exception as e:  # noqa: BLE001
                    print(f"[M4] Failed to store final report: {e}")
            return report

        failed_report = ResearchReport(query=query, content="Research failed due to persistent errors or global timeout.")
        failed_report.telemetry = self.last_run_telemetry(failed_report)
        return failed_report

    # ------------------------------------------------------------------
    # 状态机处理器
    # ------------------------------------------------------------------
    async def _on_idle(self) -> OrchestratorState:
        return OrchestratorState.PLANNING

    async def _do_planning(self) -> OrchestratorState:
        """初始规划：拆解研究问题（不生成任何结论）。"""
        self._planning_rounds += 1
        try:
            questions = self.planner.initial_plan(self._query)
        except Exception as e:  # noqa: BLE001
            print(f"[Planning] Failed: {e}")
            self._stop_reason = "planning_failed"
            return OrchestratorState.FAILED

        if not questions:
            print("[Planning] Planner 未返回任何调查问题")
            self._stop_reason = "planning_empty"
            return OrchestratorState.FAILED

        self._research_state = ResearchState(
            original_question=self._query,
            iteration=1,
            session_id=self._config.session_id,
            doc_ids=list(self._config.doc_ids),
            questions=questions,
            active_question_ids=[q.question_id for q in questions],
        )
        self._dag, self._task_map = self._build_dag_from_questions(questions)
        print(f"[Planning] ✓ 初始调查要点 {len(questions)} 个: {[q.question_id for q in questions]}")
        for q in questions:
            print(f"[Planning]   {q.question_id}: {q.question}")
        return OrchestratorState.DISPATCHING

    async def _do_dispatching(self) -> OrchestratorState:
        """DAG 分层 + Semaphore 并发执行 Searcher。"""
        if self._dag is None or len(self._dag) == 0:
            return OrchestratorState.COLLECTING

        semaphore = asyncio.Semaphore(self._config.max_concurrent)
        parallel_groups = self._dag.get_parallel_groups()
        all_results: list[AgentResult] = []

        for layer_idx, group in enumerate(parallel_groups):
            print(f"[Dispatch] ▶ Layer {layer_idx + 1}/{len(parallel_groups)}: {group} (并行执行)")
            self._searcher_count += len(group)

            async def _run_one(task_id: str) -> AgentResult:
                async with semaphore:
                    subtask = self._task_map.get(task_id)
                    if subtask is None:
                        return AgentResult(task_id=task_id, status=AgentStatus.FAILED,
                                           output=f"SubTask '{task_id}' not found")
                    context = self._build_task_context(subtask)
                    agent = await self.agent_pool.get_agent(subtask.task_type)
                    try:
                        result = await asyncio.wait_for(
                            agent.run(subtask, context),
                            timeout=subtask.timeout_seconds,
                        )
                    except asyncio.TimeoutError:
                        result = AgentResult(task_id=task_id, status=AgentStatus.TIMEOUT,
                                             output=f"Task timed out after {subtask.timeout_seconds}s")
                    except Exception as e:  # noqa: BLE001
                        result = AgentResult(task_id=task_id, status=AgentStatus.FAILED,
                                             output=f"Exception: {type(e).__name__}: {e}")
                    finally:
                        await self.agent_pool.release_agent(agent)
                    return result

            coros = [_run_one(tid) for tid in group]
            layer_results = await asyncio.gather(*coros, return_exceptions=True)
            for lr in layer_results:
                if isinstance(lr, Exception):
                    all_results.append(AgentResult(task_id="unknown", status=AgentStatus.FAILED,
                                                   output=f"Dispatch exception: {lr}"))
                else:
                    all_results.append(lr)

        self._results = all_results
        return OrchestratorState.COLLECTING

    async def _do_collecting(self) -> OrchestratorState:
        """收集 WorkerResult → 更新 ResearchState（证据装配已完成于 Worker 内部，此处记账）。"""
        state = self._require_state()
        for r in self._results:
            self._searcher_token_usage_total += int(getattr(r, "token_usage", 0) or 0)
            if r.status != AgentStatus.SUCCESS:
                print(f"  [worker-fail] {r.task_id}: {r.status.value} -> {str(r.output)[:300]}")
                continue
            wr = r.output
            if not isinstance(wr, WorkerResult):
                continue
            qid = wr.question_id
            if qid in state.active_question_ids:
                state.active_question_ids.remove(qid)
            if qid not in state.completed_question_ids:
                state.completed_question_ids.append(qid)
            q = state.get_question(qid)
            if q is not None:
                q.status = QuestionStatus.COMPLETED if wr.evidences else QuestionStatus.FAILED
            if wr.evidences:
                state.evidence_by_question[qid] = [e.evidence_id for e in wr.evidences]
            for e in wr.evidences:
                if e.evidence_id not in state.evidence_ids:
                    state.evidence_ids.append(e.evidence_id)
            self._num_searches += len(wr.search_queries)
            self._search_tool_call_count += int(getattr(wr, "search_tool_call_count", 0) or 0)
            self._candidate_count_total += int(getattr(wr, "candidate_count", 0) or 0)
            self._materialize_failed_count += int(getattr(wr, "materialize_failed_count", 0) or 0)
            self._verifier_rejected_count += int(getattr(wr, "verifier_rejected_count", 0) or 0)
            self._verified_evidence_count += int(getattr(wr, "verified_evidence_count", 0) or 0)
            for reason, count in (getattr(wr, "drop_reasons", {}) or {}).items():
                try:
                    self._drop_reasons[str(reason)] = self._drop_reasons.get(str(reason), 0) + int(count or 0)
                except (TypeError, ValueError):
                    # 旧/自定义 WorkerResult 可能带有非数值原因计数，忽略该条但不影响主链路。
                    continue

        if self._config.orchestration_mode == OrchestrationMode.DIRECT.value:
            state.final_status = FinalStatus.UNREVIEWED
            self._stop_reason = "direct_after_search"
            return OrchestratorState.REFINING

        success = sum(1 for r in self._results if r.status == AgentStatus.SUCCESS)
        print(f"[Collect] 本轮子任务完成: {success}/{len(self._results)}；现有证据 {len(self._evidence_store)} 条")
        return OrchestratorState.REVIEWING

    async def _do_reviewing(self) -> OrchestratorState:
        """Reviewer：判断证据是否覆盖问题 / 冲突 / 缺口 → 决定继续或收尾。"""
        state = self._require_state()
        if self.reviewer is None:
            state.final_status = FinalStatus.PARTIALLY_SUFFICIENT
            self._stop_reason = "reviewer_unavailable"
            return OrchestratorState.REFINING

        self._reviewer_calls += 1
        previous = state.review_history[-1] if state.review_history else None
        review = self.reviewer.review(state, self._evidence_store, previous)
        state.review_history.append(review)
        state.missing_aspects = review.missing_aspects
        # 累计冲突（按无序对去重）
        for c in review.conflicts:
            key = tuple(sorted([c.evidence_a_id, c.evidence_b_id]))
            if not any(tuple(sorted([x.evidence_a_id, x.evidence_b_id])) == key for x in state.conflicts):
                state.conflicts.append(c)

        print(f"[Review] iteration={state.iteration}, status={review.status.value}, "
              f"missing={len(review.missing_aspects)}, conflicts={len(state.conflicts)}, "
              f"effective_new={review.effective_new_evidence}")

        if review.status == ReviewStatus.SUFFICIENT:
            state.final_status = FinalStatus.SUFFICIENT
            self._reviewer_sufficient = True
            self._stop_reason = "reviewer_sufficient"
            return OrchestratorState.REFINING

        # NEED_MORE：达轮次上限或无有效新增 → 部分足够，收尾
        if state.iteration >= self._config.max_iterations:
            state.final_status = FinalStatus.PARTIALLY_SUFFICIENT
            self._stop_reason = "max_iterations"
            return OrchestratorState.REFINING
        if self._config.stop_on_no_effective_new_evidence and not review.effective_new_evidence:
            state.final_status = FinalStatus.PARTIALLY_SUFFICIENT
            self._stop_reason = "no_effective_new_evidence"
            return OrchestratorState.REFINING

        return OrchestratorState.INCREMENTAL_PLANNING

    async def _do_incremental_planning(self) -> OrchestratorState:
        """增量规划：只补缺失要点，然后继续派发 Worker。"""
        state = self._require_state()
        self._planning_rounds += 1
        try:
            new_questions = self.planner.incremental_plan(state)
        except Exception as e:  # noqa: BLE001
            print(f"[Incremental] Failed: {e}")
            state.final_status = FinalStatus.PARTIALLY_SUFFICIENT
            self._stop_reason = "incremental_plan_failed"
            return OrchestratorState.REFINING

        state.iteration += 1
        if not new_questions:
            state.final_status = FinalStatus.PARTIALLY_SUFFICIENT
            self._stop_reason = "incremental_plan_empty"
            return OrchestratorState.REFINING
        for q in new_questions:
            state.questions.append(q)
            state.active_question_ids.append(q.question_id)
        self._dag, self._task_map = self._build_dag_from_questions(new_questions)
        print(f"[Incremental] 新增调查要点 {len(new_questions)} 个 (iteration={state.iteration}): "
              f"{[q.question_id for q in new_questions]}")
        return OrchestratorState.DISPATCHING

    async def _do_refining(self) -> OrchestratorState:
        """Refiner：唯一生成结论的 Agent，产出结构化结果并渲染 Markdown。"""
        state = self._require_state()
        if self.refiner is None:
            self._stop_reason = self._stop_reason or "refiner_unavailable"
            self._runtime["final_report"] = ResearchReport(
                query=self._query,
                content="Refiner 未配置。",
                confidence=0.0,
            )
            return OrchestratorState.DONE

        try:
            refiner_result = self.refiner.refine(state, self._evidence_store)
        except Exception as e:  # noqa: BLE001
            print(f"[Refiner] Failed: {e}")
            self._stop_reason = self._stop_reason or "refiner_failed"
            refiner_result = None

        if refiner_result is None:
            self._runtime["final_report"] = ResearchReport(
                query=self._query,
                content="Refiner 生成失败。",
                confidence=0.0,
            )
            return OrchestratorState.DONE

        confidence = 1.0 if state.final_status == FinalStatus.SUFFICIENT else 0.6
        report = ResearchReport(
            query=self._query,
            content=refiner_result.markdown_body,
            structured=refiner_result.model_dump(mode="json"),
            confidence=confidence,
            num_replan=max(0, state.iteration - 1),
        )
        self._runtime["final_report"] = report
        print(f"[Refiner] ✓ 结论已生成（{state.final_status.value}），{len(refiner_result.citations)} 条引用")
        return OrchestratorState.DONE

    async def _on_done(self) -> OrchestratorState:
        return OrchestratorState.DONE

    async def _on_failed(self) -> OrchestratorState:
        return OrchestratorState.FAILED

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _require_state(self) -> ResearchState:
        if self._research_state is None:
            raise RuntimeError("ResearchState not initialized (run planning first)")
        return self._research_state

    def _build_dag_from_questions(self, questions) -> tuple[DAG, dict[str, SubTask]]:
        dag = DAG()
        task_map: dict[str, SubTask] = {}
        for idx, q in enumerate(questions):
            dag.add_node(q.question_id)
            task_map[q.question_id] = SubTask(
                task_id=q.question_id,
                task_type=TaskType.EVIDENCE,
                description=q.question,
                dependencies=[],  # 调查要点之间彼此独立，同层并行
                timeout_seconds=300,
                priority=1,
                expected_type="evidence",
                search_hints=list(q.doc_hints),
            )
        dag.topological_sort()  # 触发无环校验
        return dag, task_map

    def _build_task_context(self, subtask: SubTask) -> dict:
        return {
            "query": self._query,
            "question_id": subtask.task_id,
            "session_id": self._config.session_id,
            "doc_ids": list(self._config.doc_ids),
            "evidence_store": self._evidence_store,
        }

    # ------------------------------------------------------------------
    # 评测遥测（只读，不改变状态机行为）
    # ------------------------------------------------------------------
    def last_run_token_usage(self) -> int:
        """本轮 Searcher 的估算 token 总用量（estimate_messages_tokens 口径）。"""
        return self._searcher_token_usage_total

    def last_run_total_token_usage(self) -> int:
        """本轮所有 Agent（Planner + 并行 Searcher + Reviewer + Refiner）的估算 token 总用量。"""
        return sum(int(t) for t in self._token_ledger)

    def last_run_evidence(self) -> "EvidenceStore":
        """本轮运行期证据库（评测把最相关证据原文交给标签适配器时取用）。"""
        return self._evidence_store

    def last_run_telemetry(self, report=None) -> dict:
        """返回本轮编排、检索、校验与停止原因遥测。"""
        structured = getattr(report, "structured", {}) if report is not None else {}
        citation_audit = dict(structured.get("citation_audit") or {}) if isinstance(structured, dict) else {}
        telemetry = {
            "orchestration_mode": self._config.orchestration_mode,
            "planning_rounds": self._planning_rounds,
            "searcher_count": self._searcher_count,
            "search_tool_call_count": self._search_tool_call_count,
            "search_query_count": self._num_searches,
            "searcher_token_usage": self._searcher_token_usage_total,
            "reviewer_calls": self._reviewer_calls,
            "reviewer_sufficient": self._reviewer_sufficient,
            "stop_reason": self._stop_reason or "unknown",
            "early_stop_triggered": self._stop_reason == "no_effective_new_evidence",
            "max_iterations_reached": self._stop_reason == "max_iterations",
            "candidate_count": self._candidate_count_total,
            "materialize_failed_count": self._materialize_failed_count,
            "verifier_rejected_count": self._verifier_rejected_count,
            "verified_evidence_count": self._verified_evidence_count,
            "drop_reasons": dict(self._drop_reasons),
            "n_evidence": len(self._evidence_store),
            "citation_audit": citation_audit,
        }
        # 同时保留嵌套审计对象和扁平字段，便于报告消费者直接读取指标；旧消费者只会看到新增字段。
        telemetry.update(citation_audit)
        return telemetry

    def _is_global_timeout(self) -> bool:
        return time.monotonic() - self._start_time > self._config.global_timeout_seconds

    def _store_final_to_memory(self, report: ResearchReport) -> None:
        from src.memory.long_term import MemoryEntry

        entry = MemoryEntry(
            entry_id=f"final_report:{int(time.time())}",
            claim=str(report.content)[:800],
            source="orchestrator",
            confidence=report.confidence,
            agent_id="orchestrator",
            timestamp=time.time(),
            evidence_type="primary",
            embedding=[],
            topic=self._query[:50],
        )
        self.memory_store.put(entry)
