#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评测 CLI：LegalBenchRAG（RAG 能力）与 ContractNLI（端到端分类能力）。

用法:
  python -m src.contract.eval.main --mode legalbenchrag [--only cuad,maud ...]
  python -m src.contract.eval.main --mode contractnli [--limit 100] [--subset contractnli_b]

两个评测都把“每条输入 → prompt → 模型输出 → gold → 得分 → 过程痕迹”全量落盘到
<out>/<mode>/<时间戳>/，并以稳定 instance_id 支持断点续跑（已完成的自动跳过）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.contract.eval import loaders as L
from src.contract.eval import metrics as M
from src.contract.eval.adapter import LegalBenchAdapter  # noqa: F401  (API 公开)
from src.contract.eval.persist import (
    load_done_ids,
    make_run_dir,
    now_iso,
    parse_iso,
    write_csv,
    write_json,
)
from src.contract.eval.planner_runner import run_contractnli_fullchain, summarize_contractnli_records
from src.contract.eval.searcher_runner import run_legalbench, summarize_legalbench_records

__all__ = ["main"]


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LexTrace 评测 CLI")
    p.add_argument("--mode", required=True, choices=["legalbenchrag", "contractnli"],
                   help="评测模式")
    p.add_argument("--legalbench-root", default=None,
                   help="LegalBenchRAG 根目录（默认自动探测）")
    p.add_argument("--contractnli-jsonl", default=None,
                   help="ContractNLI jsonl（或含它的 zip，默认自动探测）")
    p.add_argument("--only", default=None,
                   help="LegalBenchRAG 只跑这些 benchmark，逗号分隔")
    p.add_argument("--subset", default=None,
                   help="ContractNLI 只跑某个 subset（如 contractnli_b / contractnli_v3）")
    p.add_argument("--limit", type=int, default=None,
                   help="ContractNLI 抽样数（None=配置默认/全量）")
    p.add_argument("--nli-session", default=None,
                   help="ContractNLI 合同索引会话（默认 nli-contractnli）")
    p.add_argument("--nli-concurrency", type=int, default=None,
                   help="ContractNLI 实例并行度（默认取 eval.nli_concurrency，默认 2）")
    p.add_argument("--searcher-max-rounds", type=int, default=None,
                   help="Searcher 最多检索轮数（覆写 contract.searcher_max_search_rounds；A/B 对照用）")
    p.add_argument("--searcher-max-searches-per-round", type=int, default=None,
                   help="Searcher 每轮最多检索词数（覆写 contract.searcher_max_searches_per_round；A/B 对照用）")
    p.add_argument("--orchestration-mode", choices=["direct", "reviewed_incremental"],
                   default=None, help="ContractNLI 编排模式：direct 或 reviewed_incremental")
    p.add_argument("--ingest-nli", action="store_true",
                   help="ContractNLI 会话为空时允许自动入库（默认禁止）")
    p.add_argument("--request-set", default=None,
                   help="预先生成的请求/入库集合 JSON（sampling 模块产物）；只跑其中列出的请求")
    p.add_argument("--ingest-only-referenced", action="store_true",
                   help="LegalBenchRAG 入库仅限 request-set 引用的文档（调试用）；"
                        "默认入库每基准全量语料以对齐 PAKTON")
    p.add_argument("--seed", type=int, default=None, help="抽样种子")
    p.add_argument("--session", default=None,
                   help="覆盖所有 benchmark 的数据库会话（单 benchmark 调试用）")
    p.add_argument("--out", default=None, help="输出根目录（默认取配置 out_dir）")
    p.add_argument("--config", default=None, help="配置文件路径")
    p.add_argument("--no-ingest", action="store_true",
                   help="corpus 未入库时不自动入库，直接报错")
    return p.parse_args(argv)


def _load_eval_config(args: argparse.Namespace) -> dict:
    from src.core.runner import load_config

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        cfg = {}
    eval_cfg = dict(cfg.get("eval", {}) or {})
    eval_cfg.setdefault("out_dir", "evaluation/runs")
    eval_cfg.setdefault("seed", 0)
    eval_cfg.setdefault("embedding_corpus", True)
    eval_cfg.setdefault("nli_concurrency", 2)
    eval_cfg.setdefault("searcher_max_search_rounds", 3)
    eval_cfg.setdefault("searcher_max_searches_per_round", 1)
    eval_cfg.setdefault("sessions", {})
    return eval_cfg


def _init_modules() -> dict:
    from src.core.runner import initialize_modules, load_config

    return initialize_modules(load_config(None), session_id="eval")


def _load_request_set(args: argparse.Namespace, mode: str) -> dict | None:
    """读取 pre-generated 请求/入库集合（仅当文件存在且 mode 匹配）。"""
    if not args.request_set:
        return None
    import json as _json

    try:
        data = _json.loads(Path(args.request_set).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[eval] 请求集合读取失败 {args.request_set}: {e}")
        raise SystemExit(1) from e
    if data.get("mode") != mode:
        print(f"[eval] 请求集合 mode={data.get('mode')} 与本评测 mode={mode} 不匹配")
        raise SystemExit(1)
    return data


def _seed(args: argparse.Namespace, eval_cfg: dict) -> int:
    return args.seed if args.seed is not None else int(eval_cfg["seed"])


# ---------------------------------------------------------------------------
# LegalBenchRAG
# ---------------------------------------------------------------------------
def _session_doc_count(session_id: str) -> int:
    from src.document.postgres_store import connect

    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents WHERE session_id = %s", (session_id,))
        return (cur.fetchone() or (0,))[0]


def _make_searcher_factory(session_id: str, solver_policy) -> object:
    from src.contract.tools import DocumentToolkit
    from src.contract.assembler import EvidenceAssembler
    from src.contract.verifier import CitationVerifier
    from src.contract.store import EvidenceStore
    from src.contract.worker import Searcher

    toolkit = DocumentToolkit(session_id=session_id, doc_ids=None)

    def factory():
        return Searcher(
            name="searcher",
            policy=solver_policy,
            toolkit=toolkit,
            assembler=EvidenceAssembler(toolkit),
            verifier=CitationVerifier(toolkit),
            store=EvidenceStore(),
        )

    return factory


def run_legalbenchrag(args: argparse.Namespace, eval_cfg: dict) -> int:
    from src.contract.eval.ingest_raw import ingest_corpus_dir

    modules = _init_modules()
    solver_policy = modules.get("solver_policy") or modules.get("default_policy")

    root = L.find_legalbench_root(args.legalbench_root)
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
    else:
        names = L.legal_benchmark_names(root)
    if not names:
        print("[legalbenchrag] 没有可评测的 benchmark（检查 --only / 数据目录）")
        return 1

    # Searcher 是唯一检索链路（确定性混合检索已移除），无需 agent 抽样 / ks / cutoffs
    sessions_cfg = dict(eval_cfg.get("sessions", {}) or {})
    chunk_cfg = {
        "max_tokens": eval_cfg.get("ingest_max_tokens"),
        "min_tokens": int(eval_cfg.get("ingest_min_tokens", 50)),
        "overlap_tokens": int(eval_cfg.get("ingest_overlap_tokens", 50)),
    }
    out_dir = Path(args.out or eval_cfg["out_dir"])
    run_dir = make_run_dir(out_dir, "legalbenchrag")

    # 可选的预生成请求集合（sampling 模块产物）：只跑其中列出的请求 / 只入被引用文档
    request_set = _load_request_set(args, "legalbenchrag")
    request_queries: dict[str, set[str]] = {}
    request_docs: dict[str, list[str]] = {}
    if request_set:
        for b, info in (request_set.get("samples", {}) or {}).items():
            request_queries[b] = set(info.get("queries", []))
            request_docs[b] = list(info.get("doc_ids", []))

    per_benchmark: dict = {}
    evidence_returned_total = 0
    evidence_hit_total = 0
    candidate_total = 0
    verified_evidence_total = 0
    verifier_rejected_total = 0
    materialize_failed_total = 0
    search_tool_calls_total = 0
    drop_reasons_total: dict[str, int] = {}
    n_instances = n_errors = n_resumed = 0

    for name in names:
        session = args.session or sessions_cfg.get(name, f"lb-{name}")
        print(f"\n=== [legalbenchrag] benchmark={name} session={session} ===")
        file_list = request_docs.get(name) if (request_set and args.ingest_only_referenced) else None
        if file_list:
            print(f"  [ingest] 仅入库被采样请求引用的 {len(file_list)} 个文档（调试用；默认全量入库以对齐 PAKTON）")
        if _session_doc_count(session) == 0:
            if args.no_ingest:
                print(f"  [错误] session {session} 无文档且禁用了自动入库；"
                      f"请先运行 ingest_corpus_dir 或去掉 --no-ingest")
                per_benchmark[name] = {"error": "no docs in session"}
                continue
            print(f"  [ingest] 会话 {session} 为空，原样入库 corpus/{name} ...")
            ingest_corpus_dir(root, name, session, embed=bool(eval_cfg.get("embedding_corpus", True)),
                              file_list=file_list, **chunk_cfg)
        print(f"  [ingest] 会话 {session} 文档数 = {_session_doc_count(session)}")

        records_path = run_dir / f"{name}.jsonl"
        done = load_done_ids(records_path)
        queries = L.load_legalbench_queries(root, name)
        if name in request_queries:
            queries = [q for q in queries if str(q.get("instance_id", "")) in request_queries[name]]
        pending = [q for q in queries if str(q.get("instance_id", "")) not in done]
        if not pending and not records_path.exists():
            # 请求集过滤后该 benchmark 无实例（如 10% 抽取向下取整为 0）：
            # 不生成记录也不参与整体汇总，避免 summarize 读不存在的文件
            print(f"  [queries] 请求集过滤后 {name} 无实例，跳过")
            continue
        n_resumed += len(queries) - len(pending)
        print(f"  [queries] 总 {len(queries)}，已完成 {len(queries) - len(pending)}，本次将处理 {len(pending)}"
              + "（全部走 Searcher 链路）")

        factory = _make_searcher_factory(session, solver_policy)
        run_legalbench(
            pending,
            root=root,
            benchmark=name,
            session_id=session,
            records_path=records_path,
            make_searcher=factory,
        )
        summ = summarize_legalbench_records(records_path)
        benchmark_metrics = dict(summ["metrics"])
        benchmark_metrics["evidence_hit_rate"] = summ.get("evidence_hit_rate_micro", 0.0)
        per_benchmark[name] = {
            "n_queries": summ["n_queries"],
            "n_agent": summ["n_agent"],
            "n_errors": summ["n_searcher_error"],
            "metrics": benchmark_metrics,
            "telemetry": {
                "evidence_returned_count": summ.get("evidence_returned_count", 0),
                "evidence_hit_count": summ.get("evidence_hit_count", 0),
                "candidate_count": summ.get("candidate_count", 0),
                "verified_evidence_count": summ.get("verified_evidence_count", 0),
                "verifier_rejected_count": summ.get("verifier_rejected_count", 0),
                "materialize_failed_count": summ.get("materialize_failed_count", 0),
                "search_tool_call_count": summ.get("search_tool_call_count", 0),
                "drop_reasons": summ.get("drop_reasons", {}),
            },
        }
        n_instances += summ["n_queries"]
        n_errors += summ["n_searcher_error"]
        evidence_returned_total += int(summ.get("evidence_returned_count", 0) or 0)
        evidence_hit_total += int(summ.get("evidence_hit_count", 0) or 0)
        candidate_total += int(summ.get("candidate_count", 0) or 0)
        verified_evidence_total += int(summ.get("verified_evidence_count", 0) or 0)
        verifier_rejected_total += int(summ.get("verifier_rejected_count", 0) or 0)
        materialize_failed_total += int(summ.get("materialize_failed_count", 0) or 0)
        search_tool_calls_total += int(summ.get("search_tool_call_count", 0) or 0)
        for reason, count in (summ.get("drop_reasons") or {}).items():
            try:
                drop_reasons_total[str(reason)] = drop_reasons_total.get(str(reason), 0) + int(count or 0)
            except (TypeError, ValueError):
                continue
        m = summ["metrics"]
        # 仅 Searcher 链路指标（确定性混合检索已移除）：
        # 文档覆盖率（找齐/找准）+ 证据对 gold 区间的字符重叠
        print(f"  [doc] agent_doc_precision={m.get('agent_doc_precision', 0):.3f}  "
              f"agent_doc_recall={m.get('agent_doc_recall', 0):.3f}")
        print(f"  [span] agent_span: P={m.get('agent_span_precision', 0):.3f}  "
              f"R={m.get('agent_span_recall', 0):.3f}  F1={m.get('agent_span_f1', 0):.3f}")

    # 整体聚合：官方口径 = Σ_bench (0.25 × 子任务内均值)，只对本次实际运行的
    # benchmark 归一化（全量 4 个时严格等于官方 0.25 等权）
    present = [b for b in per_benchmark if isinstance(per_benchmark[b], dict)
               and "metrics" in per_benchmark[b]]
    w_total = 0.25 * len(present)
    overall_metrics: dict = {}
    if present:
        keys = set()
        for b in present:
            keys |= set(per_benchmark[b]["metrics"])
        for key in keys:
            vals = [per_benchmark[b]["metrics"].get(key) for b in present]
            vals = [v for v in vals if v is not None]
            if vals:
                overall_metrics[key] = sum(0.25 * v for v in vals) / w_total
    if present:
        overall_metrics["evidence_hit_rate"] = (
            evidence_hit_total / evidence_returned_total if evidence_returned_total else 0.0
        )

    _write_legalbench_outputs(run_dir, eval_cfg, per_benchmark,
                              overall_metrics, n_instances, n_errors, n_resumed, args,
                              telemetry={
                                  "evidence_returned_count": evidence_returned_total,
                                  "evidence_hit_count": evidence_hit_total,
                                  "evidence_hit_rate": overall_metrics.get("evidence_hit_rate", 0.0),
                                  "candidate_count": candidate_total,
                                  "verified_evidence_count": verified_evidence_total,
                                  "verifier_rejected_count": verifier_rejected_total,
                                  "materialize_failed_count": materialize_failed_total,
                                  "search_tool_call_count": search_tool_calls_total,
                                  "drop_reasons": drop_reasons_total,
                              })
    print(f"\n[legalbenchrag] 完成，输出目录: {run_dir}")
    return 0


def _write_legalbench_outputs(run_dir: Path, eval_cfg: dict,
                              per_benchmark: dict,
                              overall_metrics: dict,
                              n_instances: int, n_errors: int, n_resumed: int,
                              args: argparse.Namespace,
                              telemetry: dict | None = None) -> None:
    started_at = now_iso()
    summary = {
        "mode": "legalbenchrag",
        "started_at": started_at,
        "finished_at": now_iso(),
        "config": {
            "legalbench_root": str(args.legalbench_root),
            "benchmarks": list(per_benchmark),
            "request_set": args.request_set,
        },
        "口径说明": {
            "agent_doc_precision/recall": "Searcher 证据文档覆盖率（本系统扩展）：证据命中的 gold 相关文档占比 / gold 相关文档被覆盖比例（找准/找齐，文档层）",
            "agent_span_*": "Searcher 证据（恢复的完整条款）与 gold 字符区间的重叠 P/R/F1（官方 PAKTON 区间口径）",
            "链路说明": "LegalBenchRAG 仅跑完整 Searcher 链路（确定性混合检索已移除），每条 query 都经 Searcher 检索",
            "整体聚合": "常规指标 overall = Σ_bench(0.25 × 子任务内均值)，对本次实际运行的 benchmark 归一化；evidence_hit_rate 使用所有返回证据的总命中数 / 总返回数",
        },
        "metrics": overall_metrics,
        "telemetry": telemetry or {},
        "per_benchmark": per_benchmark,
        "n_instances": n_instances,
        "n_errors": n_errors,
        "n_resumed_skipped": n_resumed,
        "output_dir": str(run_dir),
    }
    write_json(run_dir / "summary.json", summary)

    # metrics.csv：宽表（每 benchmark 一行 + overall 一行），仅 Searcher 链路指标
    metric_keys = ["agent_doc_precision", "agent_doc_recall",
                   "agent_span_precision", "agent_span_recall", "agent_span_f1",
                   "evidence_hit_rate"]
    rows: list[dict] = []
    for name, info in per_benchmark.items():
        row = {"benchmark": name, "n": info.get("n_queries", 0),
               "n_agent": info.get("n_agent", 0), "n_errors": info.get("n_errors", 0)}
        for k in metric_keys:
            row[k] = round(info.get("metrics", {}).get(k, 0), 4)
        rows.append(row)
    if overall_metrics:
        orow = {"benchmark": "overall", "n": n_instances, "n_agent": "",
                "n_errors": n_errors}
        for k in metric_keys:
            orow[k] = round(overall_metrics.get(k, 0), 4)
        rows.append(orow)
    write_csv(run_dir / "metrics.csv", rows)


# ---------------------------------------------------------------------------
# ContractNLI
# ---------------------------------------------------------------------------
def run_contractnli_cli(args: argparse.Namespace, eval_cfg: dict) -> int:
    from src.core.runner import load_config

    modules = _init_modules()
    config = load_config(args.config)
    path = L.find_contractnli_jsonl(args.contractnli_jsonl)

    # 可选的预生成请求集合（sampling 模块产物）：只跑列出的实例 / 只入被引用的合同
    request_set = _load_request_set(args, "contractnli")
    request_ids: set[str] = set()
    idx_subset: list[str] | None = None
    if request_set:
        request_ids = {inst["instance_id"] for inst in request_set.get("instances", [])}
        idx_subset = list(request_set.get("contracts", []))

    records = L.load_contractnli_records(path, subset=args.subset)
    if request_ids:
        records = [r for r in records if str(r.get("instance_id", "")) in request_ids]
    if not records:
        print(f"[contractnli] 没有可评测的实例（subset={args.subset!r}）")
        return 1

    nli_session = args.nli_session or eval_cfg.get("nli_session", "nli-contractnli")

    # 完整链路要求每条实例的合同已在会话中（检索范围 = nli:<premise_id>）
    n_docs = _session_doc_count(nli_session)
    if n_docs == 0:
        if args.ingest_nli:
            print(f"[contractnli] 会话 {nli_session} 为空，自动入库 "
                  + (f"{len(idx_subset)} 份被采样实例引用的合同" if idx_subset else "distinct 合同")
                  + " ...")
            from src.contract.eval.ingest_raw import ingest_contractnli_jsonl
            ingest_contractnli_jsonl(path, nli_session, embed=bool(eval_cfg.get("embedding_corpus", True)),
                                     idx_subset=idx_subset,
                                     max_tokens=eval_cfg.get("ingest_max_tokens"),
                                     min_tokens=int(eval_cfg.get("ingest_min_tokens", 50)),
                                     overlap_tokens=int(eval_cfg.get("ingest_overlap_tokens", 50)))
        else:
            print(f"[contractnli] 会话 {nli_session} 无合同入库（完整链路模式需要该合同在场）。")
            print("  请先手动入库（不会自动执行）:  python -m src.contract.eval.ingest_raw nli --contractnli-jsonl <jsonl/zip> --session " + nli_session)
            print("  或者加 --ingest-nli 让本次自动入库。")
            return 1
    print(f"[contractnli] 会话 {nli_session} 合同数 = {_session_doc_count(nli_session)}（完整链路，Refiner 按评测切换 3 选 1 提示词）")

    limit = args.limit if args.limit is not None else eval_cfg.get("contractnli_limit")
    seed = _seed(args, eval_cfg)
    out_dir = Path(args.out or eval_cfg["out_dir"])
    run_dir = make_run_dir(out_dir, "contractnli")
    records_path = run_dir / "records.jsonl"
    done = load_done_ids(records_path)

    print(f"[contractnli] 总 {len(records)} 条实例（subset={args.subset or 'all'}，完整链路 + 3 选 1 Refiner 提示词），"
          f"已完成 {len(done)}，本次处理上限 {limit or '全部'}")

    # 检索预算 / 实例并行度（CLI 优先，其次 config 的 contract.* 键，最后默认）
    _contract_cfg = config.get("contract", {}) or {}
    max_rounds = args.searcher_max_rounds if args.searcher_max_rounds is not None \
        else int(_contract_cfg.get("searcher_max_search_rounds", 3))
    max_per_round = args.searcher_max_searches_per_round if args.searcher_max_searches_per_round is not None \
        else int(_contract_cfg.get("searcher_max_searches_per_round", 1))
    orchestration_mode = args.orchestration_mode or str(
        _contract_cfg.get("orchestration_mode", "reviewed_incremental")
    )
    concurrency = args.nli_concurrency if args.nli_concurrency is not None \
        else int(eval_cfg.get("nli_concurrency", 2))
    print(f"[contractnli] Searcher 检索预算 = {max_rounds} 轮 × 每轮 {max_per_round} 问；"
          f"实例并行度 = {concurrency}；编排模式 = {orchestration_mode}")

    stats = run_contractnli_fullchain(records, modules=modules, config=config,
                                      records_path=records_path,
                                      limit=limit, seed=seed, done_ids=done,
                                      nli_session=nli_session, output_dir=str(run_dir),
                                      concurrency=concurrency,
                                      searcher_max_rounds=max_rounds,
                                      searcher_max_searches_per_round=max_per_round,
                                      orchestration_mode=orchestration_mode)
    summ = summarize_contractnli_records(records_path)

    summary = {
        "mode": "contractnli",
        "started_at": now_iso(),
        "finished_at": now_iso(),
        "config": {"limit": limit, "seed": seed, "subset": args.subset,
                   "nli_session": nli_session,
                   "searcher_max_search_rounds": max_rounds,
                   "searcher_max_searches_per_round": max_per_round,
                   "orchestration_mode": orchestration_mode,
                   "concurrency": concurrency,
                   "contractnli_jsonl": str(path)},
        "metrics": {k: v for k, v in summ.items() if k not in ("class_counts_true", "class_counts_pred")},
        "telemetry": {
            "citation_total_count": summ.get("citation_total_count", 0),
            "existing_evidence_id_count": summ.get("existing_evidence_id_count", 0),
            "missing_evidence_id_count": summ.get("missing_evidence_id_count", 0),
            "source_text_match_count": summ.get("source_text_match_count", 0),
            "citation_validity_rate": summ.get("citation_validity_rate", 0.0),
            "candidate_count": summ.get("candidate_count", 0),
            "verified_evidence_count": summ.get("verified_evidence_count", 0),
            "materialize_failed_count": summ.get("materialize_failed_count", 0),
            "verifier_rejected_count": summ.get("verifier_rejected_count", 0),
            "drop_reasons": summ.get("drop_reasons", {}),
            "planning_rounds_avg": summ.get("planning_rounds_avg", 0.0),
            "searcher_count_avg": summ.get("searcher_count_avg", 0.0),
            "search_tool_call_count_avg": summ.get("search_tool_call_count_avg", 0.0),
            "reviewer_calls_total": summ.get("reviewer_calls_total", 0),
            "reviewed_run_count": summ.get("reviewed_run_count", 0),
            "reviewer_sufficient_count": summ.get("reviewer_sufficient_count", 0),
            "reviewer_sufficient_rate": summ.get("reviewer_sufficient_rate"),
            "early_stop_rate": summ.get("early_stop_rate", 0.0),
            "max_iteration_rate": summ.get("max_iteration_rate", 0.0),
            "stop_reason_counts": summ.get("stop_reason_counts", {}),
            "searcher_token_usage": summ.get("searcher_token_usage", 0),
            "total_token_usage": summ.get("total_token_usage", 0),
            "elapsed_s_avg": summ.get("elapsed_s_avg", 0.0),
        },
        "class_counts_true": summ.get("class_counts_true", {}),
        "class_counts_pred": summ.get("class_counts_pred", {}),
        "n_instances": summ["n_total"],
        "n_errors": summ["n_errors"],
        "n_resumed_skipped": stats["skipped_done"],
        "output_dir": str(run_dir),
    }
    write_json(run_dir / "summary.json", summary)
    rows = [{"metric": "accuracy", "value": round(summ["accuracy"], 4)},
            {"metric": "f1_weighted", "value": round(summ["f1_weighted"], 4)}]
    for cls, f1 in sorted(summ.get("f1_per_class", {}).items()):
        rows.append({"metric": f"f1_{cls}", "value": round(f1, 4)})
    extra_metrics = [
        "citation_validity_rate", "citation_total_count", "existing_evidence_id_count",
        "missing_evidence_id_count", "source_text_match_count", "verifier_rejected_count",
        "candidate_count", "verified_evidence_count", "materialize_failed_count",
        "planning_rounds_avg", "searcher_count_avg",
        "search_tool_call_count_avg", "reviewer_sufficient_rate", "early_stop_rate",
        "max_iteration_rate", "reviewer_calls_total", "reviewed_run_count",
        "reviewer_sufficient_count", "elapsed_s_avg", "searcher_token_usage", "total_token_usage",
    ]
    for name in extra_metrics:
        value = summ.get(name)
        rows.append({"metric": name, "value": "" if value is None else round(value, 4) if isinstance(value, float) else value})
    for reason, count in sorted((summ.get("drop_reasons") or {}).items()):
        rows.append({"metric": f"drop_reason_{reason}", "value": count})
    for reason, count in sorted((summ.get("stop_reason_counts") or {}).items()):
        rows.append({"metric": f"stop_reason_{reason}", "value": count})
    write_csv(run_dir / "metrics.csv", rows)

    print(f"[contractnli] 本次新增 {stats['evaluated']} 条（累计 {summ['n_total']}，错误 {summ['n_errors']}）")
    print(f"[contractnli] accuracy={summ['accuracy']:.4f}  f1_weighted={summ['f1_weighted']:.4f}")
    print(f"[contractnli] token（估算）: Searcher={summ.get('searcher_token_usage', 0)}  全Agent={summ.get('total_token_usage', 0)}")
    print(f"[contractnli] 输出目录: {run_dir}")
    return 0


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    eval_cfg = _load_eval_config(args)
    t0 = time.time()
    if args.mode == "legalbenchrag":
        rc = run_legalbenchrag(args, eval_cfg)
    else:
        rc = run_contractnli_cli(args, eval_cfg)
    print(f"[eval] 总耗时 {time.time() - t0:.1f}s")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
