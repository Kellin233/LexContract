"""ContractNLI 评测执行器：每条实例跑一遍正式完整链路，Refiner 按评测切换专用提示词出 3 选 1 标签。

每实例流程：
  1. 把 hypothesis 当研究问题，作用域锁到该合同（RunConfig.session_id/doc_ids=[nli:<premise_id>]）；
  2. 复用 src.core.runner.run_research 跑正式完整链路（Planner 拆要点 → Searcher 多轮证据 →
     Reviewer → Refiner），其中 Refiner 注入 ContractNLI 评测专用提示词（结论 = 3 选 1 标签）；
  3. 从 report.structured（Refiner 输出 JSON）用 ContractNLIAdapter.extract_chain_label 提标签；
  4. 写 records.jsonl（NliRecord，mode="fullchain"），以 instance_id 断点续跑 / 抽样复现。

并发：实例之间并行（ThreadPoolExecutor，concurrency 可调，默认 2）。并发要求“每条实例独立
toolkit / agent_pool / orchestrator”（toolkit 的会话+文档作用域是共享可变状态，跨实例会互相踩），
只有 policy / planner / reviewer / NLI-refiner 共享。

正式生产链路不注入该提示词，Refiner 保持默认生产提示词（refiner.py system_prompt 参数）。
"""
from __future__ import annotations

import asyncio
import copy
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .adapter import ContractNLIAdapter
from .persist import append_record
from .schemas import NliRecord

__all__ = ["run_contractnli_fullchain", "summarize_contractnli_records"]


def run_contractnli_fullchain(
    records: list[dict],
    *,
    modules: dict,
    config: dict,
    records_path: str | Path,
    limit: int | None = None,
    seed: int = 0,
    done_ids: set[str] | None = None,
    nli_session: str = "nli-contractnli",
    output_dir: str = "outputs/reports",
    concurrency: int = 2,
    searcher_max_rounds: int | None = None,
    searcher_max_searches_per_round: int | None = None,
    orchestration_mode: str = "reviewed_incremental",
) -> dict:
    """执行 ContractNLI 评测（实例间并发），支持 direct / reviewed_incremental 编排模式。"""
    done = done_ids or set()
    if limit:
        rng = random.Random(seed)
        pending = [r for r in records if str(r.get("instance_id", "")) not in done]
        pending = rng.sample(pending, min(limit, len(pending)))
        selected = {str(r.get("instance_id", "")) for r in pending}
    else:
        selected = None

    jobs = [
        rec for rec in records
        if str(rec.get("instance_id", "")) not in done
        and (selected is None or str(rec.get("instance_id", "")) in selected)
    ]
    skipped = len(records) - len(jobs)

    _c = config.get("contract", {})
    max_rounds = int(searcher_max_rounds if searcher_max_rounds is not None else _c.get("searcher_max_search_rounds", 3))
    max_per_round = int(searcher_max_searches_per_round
                        if searcher_max_searches_per_round is not None else _c.get("searcher_max_searches_per_round", 1))
    dedup = bool(_c.get("searcher_dedup_tool_results", True))

    run_config = copy.deepcopy(config)
    if not isinstance(run_config.get("contract"), dict):
        run_config["contract"] = {}
    run_config["contract"]["orchestration_mode"] = orchestration_mode
    base = _build_nli_base(modules, run_config)  # 共享 policy/planner/reviewer + NLI-refiner

    def _run_one(rec: dict) -> NliRecord:
        return _run_single_instance(rec, base, run_config, nli_session=nli_session, output_dir=output_dir,
                                    max_rounds=max_rounds, max_per_round=max_per_round, dedup=dedup)

    evaluated = 0
    errors = 0

    def _persist(record: NliRecord) -> None:
        """任务完成后立即落盘，支持中断后的断点续跑。"""
        nonlocal evaluated, errors
        if not record.pred_valid:
            errors += 1
        append_record(records_path, record.model_dump(mode="json"))
        evaluated += 1

    if int(concurrency) > 1:
        with ThreadPoolExecutor(max_workers=int(concurrency)) as ex:
            futures = [ex.submit(_run_one, job) for job in jobs]
            for future in as_completed(futures):
                _persist(future.result())
    else:
        for job in jobs:
            _persist(_run_one(job))

    return {"evaluated": evaluated, "skipped_done": skipped, "errors": errors}


def _build_nli_base(modules: dict, config: dict) -> dict:
    """一份把 Refiner 换成分区专用提示词的共享 base（policy/planner/reviewer/NLI-refiner）。

    注意：不包含 toolkit/agent_pool/orchestrator（每条实例按自己的会话/文档作用域独立构建，
    避免共享 toolkit.set_scope 在并发实例间互相覆盖文档检索范围）。
    """
    from src.contract.refiner import Refiner

    base = dict(modules)
    base["refiner"] = Refiner(
        policy=modules.get("summarizer_policy") or modules.get("default_policy"),
        input_token_budget=config.get("contract", {}).get("refiner_input_token_budget", 65536),
        system_prompt=ContractNLIAdapter.refiner_system_prompt(),
    )
    return base


def _build_instance_modules(base: dict, config: dict, nli_session: str,
                            max_rounds: int, max_per_round: int, dedup: bool) -> dict:
    """按单条实例构建独立 modules：新 toolkit / evidence_store / agent_pool / orchestrator。

    隔离原因：DocumentToolkit 的 session_id/doc_ids 是共享可变状态，并发实例 doc_ids 不同，
    必须每实例一套，否则一个实例的检索会跑到另一实例的合同作用域里。
    """
    from src.contract.tools import DocumentToolkit
    from src.contract.assembler import EvidenceAssembler
    from src.contract.verifier import CitationVerifier
    from src.contract.store import EvidenceStore
    from src.contract.worker import Searcher
    from src.orchestrator.agent_pool import AgentPool
    from src.orchestrator.orchestrator import Orchestrator

    tk = DocumentToolkit(session_id=nli_session, doc_ids=None)
    estore = EvidenceStore()
    solver = base.get("solver_policy") or base.get("default_policy")

    def make_searcher():
        return Searcher(
            name="searcher",
            policy=solver,
            toolkit=tk,
            assembler=EvidenceAssembler(tk),
            verifier=CitationVerifier(tk),
            store=estore,
            dedup_tool_results=dedup,
            max_search_rounds=max_rounds,
            max_searches_per_round=max_per_round,
        )

    pool = AgentPool(policy_factory=lambda: solver, worker_factory=make_searcher, max_idle=3)

    m = dict(base)
    m["toolkit"] = tk
    m["evidence_store"] = estore
    m["agent_pool"] = pool
    m["orchestrator"] = Orchestrator(
        planner=base.get("planner"),
        agent_pool=pool,
        reviewer=base.get("reviewer"),
        refiner=base.get("refiner"),
        evidence_store=estore,
        compressor=None,
    )
    return m


def _run_single_instance(rec: dict, base: dict, config: dict, *, nli_session: str, output_dir: str,
                         max_rounds: int, max_per_round: int, dedup: bool) -> NliRecord:
    """跑一条实例的完整链路 → 提出标签 → 返回 NliRecord（不落盘；由调用方统一按序写）。"""
    from src.core.runner import run_research

    iid = str(rec.get("instance_id", ""))
    premise_id = str(rec.get("premise_id", ""))
    hypothesis = str(rec.get("hypothesis", ""))
    gold = str(rec.get("label", ""))
    doc_id = nli_doc_id_raw(premise_id)

    record = NliRecord(
        instance_id=iid,
        premise_id=premise_id,
        subset=str(rec.get("subset", "")),
        premise_preview=str(rec.get("premise", ""))[:200],
        hypothesis=hypothesis,
        gold_label=gold,
        mode="fullchain",
        doc_id=doc_id,
    )
    t0 = time.time()
    try:
        if not _doc_exists(nli_session, doc_id):
            record.error = "doc not found in session"
            record.pred_valid = False
        else:
            instance = _build_instance_modules(base, config, nli_session, max_rounds, max_per_round, dedup)
            report = asyncio.run(run_research(
                hypothesis, config, instance,
                session_id=nli_session, doc_ids=[doc_id], output_dir=output_dir,
            ))
            _apply_chain_label(report, record, instance["orchestrator"])
    except Exception as e:  # noqa: BLE001
        record.pred_valid = False
        record.error = record.error or f"{type(e).__name__}: {e}"

    record.elapsed_s = time.time() - t0
    record.correct = (record.pred_label == gold)
    record.telemetry.setdefault("doc_found", "doc not found" not in (record.error or ""))
    record.telemetry.setdefault("run_completed", record.pred_valid or record.error is None)
    return record


def _apply_chain_label(report, record: NliRecord, orchestrator) -> None:
    """从正式链路 Refiner 输出的 structured 提取 3 选 1 标签，回填 record（无第二次 LLM 调用）。"""
    if report is None:
        record.pred_valid = False
        record.error = record.error or "research report is None"
        return
    structured = getattr(report, "structured", None) or {}

    label, reasoning = ContractNLIAdapter.extract_chain_label(structured)
    record.pred_label = label
    record.pred_valid = label is not None
    record.reasoning = reasoning
    record.raw_response = json.dumps(structured, ensure_ascii=False)  # Refiner 输出逐字留档
    if label is None:
        record.error = record.error or "unparseable label"

    evidence_store = orchestrator.last_run_evidence() if orchestrator is not None else None
    supporting_ids = list(structured.get("supporting_evidence_ids") or [])
    orchestration_telemetry = (
        orchestrator.last_run_telemetry(report)
        if orchestrator is not None and hasattr(orchestrator, "last_run_telemetry")
        else {}
    )
    citation_audit = dict(structured.get("citation_audit") or {})
    record.telemetry = {
        "refiner_mode": "contractnli_nli_prompt",
        "searcher_token_usage": orchestrator.last_run_token_usage() if orchestrator is not None else 0,
        "total_token_usage": orchestrator.last_run_total_token_usage() if orchestrator is not None else 0,
        "num_searches": int(getattr(report, "num_searches", 0) or 0),
        "num_replan": int(getattr(report, "num_replan", 0) or 0),
        "final_status": structured.get("final_status"),
        "n_evidence": len(evidence_store.all()) if evidence_store is not None else 0,
        "n_supporting": len(supporting_ids),
        "chain_conclusion_preview": str(structured.get("conclusion", ""))[:200],
        "run_completed": True,
    }
    record.telemetry.update(orchestration_telemetry)
    record.telemetry.update(citation_audit)


def _doc_exists(session_id: str, doc_id: str) -> bool:
    """检查某合同是否已在目标会话中入库（只读 SELECT，按 session 过滤）。"""
    from src.retrieval.store import connect

    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM documents WHERE doc_id = %s AND session_id = %s",
                (doc_id, session_id),
            )
            return cur.fetchone() is not None
    except Exception:  # noqa: BLE001
        return False


def nli_doc_id_raw(premise_id: str) -> str:
    """与 ingest_raw.nli_doc_id 保持一致的 doc_id 生成（避免循环导入）。"""
    return f"nli:{premise_id}"


def summarize_contractnli_records(records_path: str | Path) -> dict:
    """从已持久化的 records.jsonl 计算 Accuracy / weighted F1 / per-class F1。"""
    import json as _json

    y_true: list[str] = []
    y_pred: list[str] = []
    n_total = n_errors = 0
    sum_searcher_tokens = 0
    sum_total_tokens = 0
    citation_total = citation_existing = citation_missing = citation_source_match = 0
    candidate_total = verified_evidence = verifier_rejected = materialize_failed = 0
    drop_reasons_total: dict[str, int] = {}
    planning_rounds = searcher_count = search_tool_calls = 0
    completed_runs = 0
    reviewer_calls_total = reviewer_runs = reviewer_sufficient_runs = early_stop_runs = max_iteration_runs = 0
    stop_reason_counts: dict[str, int] = {}
    elapsed_total = 0.0
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = _json.loads(line)
            n_total += 1
            tel = rec.get("telemetry") or {}
            sum_searcher_tokens += int(tel.get("searcher_token_usage", 0) or 0)
            sum_total_tokens += int(tel.get("total_token_usage", 0) or 0)
            citation_total += int(tel.get("total_citation_count", 0) or 0)
            citation_existing += int(tel.get("existing_evidence_id_count", 0) or 0)
            citation_missing += int(tel.get("missing_evidence_id_count", 0) or 0)
            citation_source_match += int(tel.get("source_text_match_count", 0) or 0)
            candidate_total += int(tel.get("candidate_count", 0) or 0)
            verified_evidence += int(tel.get("verified_evidence_count", 0) or 0)
            verifier_rejected += int(tel.get("verifier_rejected_count", 0) or 0)
            materialize_failed += int(tel.get("materialize_failed_count", 0) or 0)
            for reason, count in (tel.get("drop_reasons") or {}).items():
                try:
                    drop_reasons_total[str(reason)] = drop_reasons_total.get(str(reason), 0) + int(count or 0)
                except (TypeError, ValueError):
                    continue
            planning_rounds += int(tel.get("planning_rounds", 0) or 0)
            searcher_count += int(tel.get("searcher_count", 0) or 0)
            search_tool_calls += int(tel.get("search_tool_call_count", 0) or 0)
            elapsed_total += float(rec.get("elapsed_s", 0.0) or 0.0)
            run_completed = bool(tel.get("run_completed", rec.get("pred_valid", False)))
            if run_completed:
                completed_runs += 1
            reviewer_calls = int(tel.get("reviewer_calls", 0) or 0)
            reviewer_calls_total += reviewer_calls
            if reviewer_calls > 0:
                reviewer_runs += 1
                reviewer_sufficient_runs += int(bool(tel.get("reviewer_sufficient", False)))
            stop_reason = str(tel.get("stop_reason", "") or "")
            if stop_reason:
                stop_reason_counts[stop_reason] = stop_reason_counts.get(stop_reason, 0) + 1
            early_stop_runs += int(stop_reason == "no_effective_new_evidence")
            max_iteration_runs += int(stop_reason == "max_iterations")
            if not rec.get("pred_valid"):
                n_errors += 1
                continue
            y_true.append(rec["gold_label"])
            y_pred.append(rec["pred_label"])
    from . import metrics as M

    acc_all = M.accuracy(y_true, y_pred) if y_true else 0.0
    avg_den = completed_runs or n_total or 1
    return {
        "n_total": n_total,
        "n_errors": n_errors,
        "accuracy": acc_all,
        "f1_weighted": M.f1_weighted(y_true, y_pred) if y_true else 0.0,
        "f1_per_class": M.f1_per_class(y_true, y_pred),
        "class_counts_true": {c: y_true.count(c) for c in sorted(set(y_true))},
        "class_counts_pred": {c: y_pred.count(c) for c in sorted(set(y_pred))},
        "searcher_token_usage": sum_searcher_tokens,
        "total_token_usage": sum_total_tokens,
        "citation_total_count": citation_total,
        "existing_evidence_id_count": citation_existing,
        "missing_evidence_id_count": citation_missing,
        "source_text_match_count": citation_source_match,
        "citation_validity_rate": citation_source_match / citation_total if citation_total else 0.0,
        "candidate_count": candidate_total,
        "verified_evidence_count": verified_evidence,
        "verifier_rejected_count": verifier_rejected,
        "materialize_failed_count": materialize_failed,
        "drop_reasons": drop_reasons_total,
        "planning_rounds_avg": planning_rounds / avg_den,
        "searcher_count_avg": searcher_count / avg_den,
        "search_tool_call_count_avg": search_tool_calls / avg_den,
        "reviewer_sufficient_rate": (
            reviewer_sufficient_runs / reviewer_runs if reviewer_runs else None
        ),
        "reviewer_calls_total": reviewer_calls_total,
        "reviewed_run_count": reviewer_runs,
        "reviewer_sufficient_count": reviewer_sufficient_runs,
        "stop_reason_counts": stop_reason_counts,
        "early_stop_rate": early_stop_runs / avg_den,
        "max_iteration_rate": max_iteration_runs / avg_den,
        "elapsed_s_avg": elapsed_total / avg_den,
    }
