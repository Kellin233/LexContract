"""证据审计、命中率和编排遥测的离线回归测试。"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.contract.eval.metrics import evidence_hit_rate
from src.contract.eval.schemas import NliRecord
from src.contract.eval import planner_runner
from src.contract.refiner import Refiner
from src.contract.schemas import (
    Evidence,
    FinalStatus,
    RefinerResult,
    ResearchQuestion,
    ResearchState,
    ReviewResult,
    ReviewStatus,
    WorkerResult,
)
from src.contract.store import EvidenceStore
from src.contract.worker import Searcher
from src.contract.eval.planner_runner import summarize_contractnli_records
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.schemas import AgentResult, AgentStatus, RunConfig, OrchestratorState


class _Policy:
    def __init__(self, content: str):
        self.content = content

    def __call__(self, _messages):
        return {"content": self.content}


def _state() -> ResearchState:
    return ResearchState(original_question="q", final_status=FinalStatus.SUFFICIENT)


def _registered_evidence(store: EvidenceStore, verified: bool = True) -> Evidence:
    ev = Evidence(
        document_id="doc-1",
        document_name="Doc",
        start_offset=0,
        end_offset=5,
        quote="alpha",
        verified=verified,
    )
    return store.register(ev, "Q1")[0]


def test_refiner_audits_unique_raw_ids_before_filtering():
    store = EvidenceStore()
    ev = _registered_evidence(store)
    refiner = Refiner(_Policy(
        '{"conclusion":"ok",'
        f'"points":[{{"claim":"p","evidence_ids":["{ev.evidence_id}","E999"]}}],'
        f'"supporting_evidence_ids":["{ev.evidence_id}","E999"]}}'
    ))

    result = refiner.refine(_state(), store)
    audit = result.citation_audit
    assert audit["total_citation_count"] == 2
    assert audit["existing_evidence_id_count"] == 1
    assert audit["missing_evidence_id_count"] == 1
    assert audit["source_text_match_count"] == 1
    assert audit["citation_validity_rate"] == 0.5
    assert result.supporting_evidence_ids == [ev.evidence_id]


def test_refiner_zero_citation_rate_is_zero():
    result = Refiner(_Policy('{"conclusion":"no citation","points":[]}')).refine(_state(), EvidenceStore())
    assert result.citation_audit["total_citation_count"] == 0
    assert result.citation_audit["citation_validity_rate"] == 0.0


def test_refiner_audit_excludes_tampered_source_text():
    store = EvidenceStore()
    ev = _registered_evidence(store, verified=False)
    result = Refiner(_Policy(
        f'{{"conclusion":"ok","points":[{{"claim":"p","evidence_ids":["{ev.evidence_id}"]}}]}}'
    )).refine(_state(), store)
    assert result.citation_audit["total_citation_count"] == 1
    assert result.citation_audit["existing_evidence_id_count"] == 1
    assert result.citation_audit["source_text_match_count"] == 0
    assert result.citation_audit["citation_validity_rate"] == 0.0


def test_evidence_hit_rate_requires_positive_overlap():
    gold = {"doc": [[10, 20]]}
    stats = evidence_hit_rate([
        ("doc", [10, 10]),      # 空区间，不计返回证据
        ("doc", [0, 10]),       # 仅相邻边界，不命中
        ("doc", [19, 30]),      # 正长度重叠，命中
        ("other", [10, 20]),    # 文档不同，不命中
    ], gold)
    assert stats == {"hit_count": 1, "returned_count": 3, "rate": 1 / 3}
    assert evidence_hit_rate([], gold)["rate"] == 0.0


class _Assembler:
    def materialize(self, candidate, question_id):
        if candidate.get("kind") == "materialize-fail":
            return None
        return Evidence(
            question_id=question_id,
            document_id="doc-1",
            start_offset=0,
            end_offset=5,
            quote="reject" if candidate.get("kind") == "verify-fail" else "alpha",
        )


class _Verifier:
    def verify(self, evidence):
        return evidence.quote != "reject"


class _Toolkit:
    def get_tools(self):
        return []


def test_searcher_validation_counters():
    searcher = Searcher(
        name="searcher",
        policy=lambda _messages: {"content": "[]"},
        toolkit=_Toolkit(),
        assembler=_Assembler(),
        verifier=_Verifier(),
        store=EvidenceStore(),
    )
    result = searcher._assemble_worker_result(
        [{"kind": "ok"}, {"kind": "verify-fail"}, {"kind": "materialize-fail"}],
        "Q1", "q", True, ["q"], EvidenceStore(), search_tool_call_count=2,
    )
    assert result.candidate_count == 3
    assert result.materialize_failed_count == 1
    assert result.verifier_rejected_count == 1
    assert result.verified_evidence_count == 1
    assert result.search_tool_call_count == 2


def test_orchestrator_accumulates_searcher_telemetry_and_direct_mode():
    orch = Orchestrator(None, None, reviewer=None, refiner=None, evidence_store=EvidenceStore())
    orch._config = RunConfig(orchestration_mode="direct")
    orch._research_state = ResearchState(original_question="q", iteration=1)
    orch._results = [AgentResult(
        task_id="Q1", status=AgentStatus.SUCCESS, token_usage=11,
        output=WorkerResult(
            question_id="Q1", question="q", searched=True,
            search_tool_call_count=2, candidate_count=5, verifier_rejected_count=3,
            materialize_failed_count=1, verified_evidence_count=1,
            drop_reasons={"quote-mismatch": 2, "materialize-fail": 1},
        ),
    )]
    next_state = asyncio.run(orch._do_collecting())
    assert next_state == OrchestratorState.REFINING
    assert orch._research_state.final_status == FinalStatus.UNREVIEWED
    assert orch._stop_reason == "direct_after_search"

    orch._results = [AgentResult(
        task_id="Q2", status=AgentStatus.SUCCESS, token_usage=7,
        output=WorkerResult(question_id="Q2", question="q", search_tool_call_count=4),
    )]
    orch._config.orchestration_mode = "reviewed_incremental"
    asyncio.run(orch._do_collecting())
    assert orch.last_run_token_usage() == 18
    assert orch._search_tool_call_count == 6
    assert orch._verifier_rejected_count == 3
    assert orch._candidate_count_total == 5
    assert orch._verified_evidence_count == 1
    assert orch._drop_reasons == {"quote-mismatch": 2, "materialize-fail": 1}


def test_old_contractnli_records_still_summarize():
    old_record = {
        "instance_id": "old-1",
        "gold_label": "entailment",
        "pred_label": "entailment",
        "pred_valid": True,
        "telemetry": {},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", encoding="utf-8") as f:
        f.write(json.dumps(old_record) + "\n")
        f.flush()
        summary = summarize_contractnli_records(f.name)
    assert summary["accuracy"] == 1.0
    assert summary["citation_validity_rate"] == 0.0
    assert summary["reviewer_sufficient_rate"] is None


def test_contractnli_persists_each_completed_task(monkeypatch, tmp_path):
    """并发任务完成后立即写入，不能等所有 Future 都结束。"""
    slow_started = threading.Event()
    fast_allowed = threading.Event()
    slow_done = threading.Event()
    persisted_ids = []
    persisted_before_slow_done = []

    def fake_build_base(_modules, _config):
        return {}

    def fake_run_one(rec, *_args, **_kwargs):
        instance_id = str(rec["instance_id"])
        if instance_id == "slow":
            slow_started.set()
            assert fast_allowed.wait(timeout=1)
            time.sleep(0.05)
            slow_done.set()
        else:
            assert slow_started.wait(timeout=1)
            fast_allowed.set()
        return NliRecord(
            instance_id=instance_id,
            hypothesis="h",
            gold_label="entailment",
            pred_label="entailment",
            pred_valid=True,
        )

    def fake_append(_path, record):
        instance_id = str(record["instance_id"])
        persisted_ids.append(instance_id)
        if instance_id == "fast":
            persisted_before_slow_done.append(not slow_done.is_set())

    monkeypatch.setattr(planner_runner, "_build_nli_base", fake_build_base)
    monkeypatch.setattr(planner_runner, "_run_single_instance", fake_run_one)
    monkeypatch.setattr(planner_runner, "append_record", fake_append)

    stats = planner_runner.run_contractnli_fullchain(
        [{"instance_id": "slow"}, {"instance_id": "fast"}],
        modules={},
        config={"contract": {}},
        records_path=tmp_path / "records.jsonl",
        concurrency=2,
    )

    assert stats == {"evaluated": 2, "skipped_done": 0, "errors": 0}
    assert persisted_ids == ["fast", "slow"]
    assert persisted_before_slow_done == [True]


class _Reviewer:
    def __init__(self, result):
        self.result = result

    def review(self, _state, _store, _previous=None):
        return self.result


class _Planner:
    def incremental_plan(self, _state):
        return []


class _PathPlanner:
    def initial_plan(self, _query):
        return [ResearchQuestion(question_id="Q1", question="first")]

    def incremental_plan(self, _state):
        return [ResearchQuestion(question_id="Q2", question="second")]


class _PathPool:
    async def get_agent(self, _task_type):
        return _PathSearcher()

    async def release_agent(self, _agent):
        return None


class _PathSearcher:
    async def run(self, subtask, _context):
        return AgentResult(
            task_id=subtask.task_id,
            status=AgentStatus.SUCCESS,
            output=WorkerResult(
                question_id=subtask.task_id,
                question=subtask.description,
                searched=True,
                candidate_count=1,
                search_tool_call_count=1,
            ),
        )


class _PathReviewer:
    def __init__(self):
        self.calls = 0

    def review(self, _state, _store, _previous=None):
        self.calls += 1
        if self.calls == 1:
            return ReviewResult(status=ReviewStatus.NEED_MORE, effective_new_evidence=True)
        return ReviewResult(status=ReviewStatus.SUFFICIENT)


class _PathRefiner:
    def refine(self, state, _store):
        return RefinerResult(conclusion="ok", final_status=state.final_status, markdown_body="ok")


def test_orchestrator_runs_both_orchestration_paths():
    async def _run(mode):
        reviewer = _PathReviewer()
        orchestrator = Orchestrator(
            _PathPlanner(), _PathPool(), reviewer=reviewer, refiner=_PathRefiner(),
            evidence_store=EvidenceStore(),
        )
        report = await orchestrator.run(
            "q", RunConfig(max_iterations=2, orchestration_mode=mode)
        )
        return report, reviewer

    direct, direct_reviewer = asyncio.run(_run("direct"))
    assert direct.structured["final_status"] == FinalStatus.UNREVIEWED.value
    assert direct_reviewer.calls == 0
    assert direct.telemetry["stop_reason"] == "direct_after_search"

    reviewed, reviewed_reviewer = asyncio.run(_run("reviewed_incremental"))
    assert reviewed.structured["final_status"] == FinalStatus.SUFFICIENT.value
    assert reviewed_reviewer.calls == 2
    assert reviewed.telemetry["planning_rounds"] == 2
    assert reviewed.telemetry["searcher_count"] == 2
    assert reviewed.telemetry["search_tool_call_count"] == 2
    assert reviewed.telemetry["stop_reason"] == "reviewer_sufficient"


def test_orchestrator_stop_reasons():
    sufficient = Orchestrator(None, None, reviewer=_Reviewer(ReviewResult(status=ReviewStatus.SUFFICIENT)), refiner=None)
    sufficient._config = RunConfig(max_iterations=3)
    sufficient._research_state = ResearchState(original_question="q", iteration=1)
    asyncio.run(sufficient._do_reviewing())
    assert sufficient._stop_reason == "reviewer_sufficient"

    no_new = Orchestrator(None, None, reviewer=_Reviewer(
        ReviewResult(status=ReviewStatus.NEED_MORE, effective_new_evidence=False)
    ), refiner=None)
    no_new._config = RunConfig(max_iterations=3, stop_on_no_effective_new_evidence=True)
    no_new._research_state = ResearchState(original_question="q", iteration=1)
    asyncio.run(no_new._do_reviewing())
    assert no_new._stop_reason == "no_effective_new_evidence"

    maxed = Orchestrator(None, None, reviewer=_Reviewer(
        ReviewResult(status=ReviewStatus.NEED_MORE, effective_new_evidence=True)
    ), refiner=None)
    maxed._config = RunConfig(max_iterations=1)
    maxed._research_state = ResearchState(original_question="q", iteration=1)
    asyncio.run(maxed._do_reviewing())
    assert maxed._stop_reason == "max_iterations"

    empty_plan = Orchestrator(_Planner(), None, reviewer=None, refiner=None)
    empty_plan._config = RunConfig(max_iterations=3)
    empty_plan._research_state = ResearchState(original_question="q", iteration=1)
    asyncio.run(empty_plan._do_incremental_planning())
    assert empty_plan._stop_reason == "incremental_plan_empty"


if __name__ == "__main__":
    _tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for _test in _tests:
        _test()
    print(f"PASS ({len(_tests)} offline telemetry tests)")
