#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评测 CLI：LegalBenchRAG（RAG 能力）与 ContractNLI（端到端分类能力）。

用法:
  python -m src.contract.eval.main --mode legalbenchrag [--only cuad,maud ...] [--agent-limit 20]
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
from src.contract.eval.planner_runner import run_contractnli, summarize_contractnli_records
from src.contract.eval.searcher_runner import run_legalbench, summarize_legalbench_records

__all__ = ["main"]


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LexContract 评测 CLI")
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
    p.add_argument("--k", default=None, help="Recall@k 的 k 列表，逗号分隔")
    p.add_argument("--agent-limit", type=int, default=None,
                   help="每个 benchmark 用真实 LLM Searcher 的 query 数（0=只跑确定性，None=配置默认）")
    p.add_argument("--limit", type=int, default=None,
                   help="ContractNLI 抽样数（None=配置默认/全量）")
    p.add_argument("--nli-mode", default=None, choices=["indexed", "direct"],
                   help="ContractNLI 评测方式：indexed=整库入库+检索式（默认，对齐 PAKTON）；"
                        "direct=整段前提直喂（naive baseline）")
    p.add_argument("--nli-session", default=None,
                   help="ContractNLI 合同索引会话（默认 nli-contractnli）")
    p.add_argument("--nli-top-k", type=int, default=None,
                   help="ContractNLI 检索式下每条的检索条款数（默认 8）")
    p.add_argument("--ingest-nli", action="store_true",
                   help="ContractNLI 会话为空时允许自动入库 distinct 合同（默认禁止）")
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
    eval_cfg.setdefault("k", M.default_ks())
    eval_cfg.setdefault("agent_limit", 20)
    eval_cfg.setdefault("seed", 0)
    eval_cfg.setdefault("embedding_corpus", True)
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


def _resolve_ks(args: argparse.Namespace, eval_cfg: dict) -> list[int]:
    if args.k:
        return [int(x.strip()) for x in args.k.split(",") if x.strip()]
    return list(eval_cfg["k"])


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

    ks = _resolve_ks(args, eval_cfg)
    seed = _seed(args, eval_cfg)
    agent_limit = args.agent_limit if args.agent_limit is not None else eval_cfg.get("agent_limit")
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
    overall: dict[str, list[float]] = {}
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
              + (f"，agent 抽样 {agent_limit}" if agent_limit else "")
              + ("（确定性部分全量）" if pending else ""))

        factory = _make_searcher_factory(session, solver_policy)
        run_legalbench(
            pending,
            root=root,
            benchmark=name,
            session_id=session,
            ks=ks,
            records_path=records_path,
            make_searcher=factory if (agent_limit is None or agent_limit > 0) else None,
            agent_limit=agent_limit,
            seed=seed,
        )
        summ = summarize_legalbench_records(records_path, ks)
        per_benchmark[name] = {
            "n_queries": summ["n_queries"],
            "n_agent": summ["n_agent"],
            "n_errors": summ["n_searcher_error"],
            "metrics": summ["metrics"],
        }
        n_instances += summ["n_queries"]
        n_errors += summ["n_searcher_error"]
        m = summ["metrics"]
        print(f"  [recall@k] " + "  ".join(f"R@{k}={m.get(f'recall_at_{k}', 0):.3f}" for k in ks))
        print(f"  [mrr] {m.get('mrr', 0):.3f}   [span_f1] {m.get('span_f1', 0):.3f}"
              + (f"   [agent_span_f1] {m.get('agent_span_f1', 0):.3f}" if "agent_span_f1" in m else ""))
        # 累加整体（等权宏平均）
        for key, val in m.items():
            overall.setdefault(key, []).append(val)

    # 整体 = 各 benchmark 均值（无 agent 数据的维度跳过）
    overall_metrics = {key: sum(vals) / len(vals) for key, vals in overall.items() if vals}
    for m in per_benchmark.values():
        if "error" in m:
            overall_metrics = {}

    _write_legalbench_outputs(run_dir, eval_cfg, ks, per_benchmark, overall_metrics,
                              n_instances, n_errors, n_resumed, args)
    print(f"\n[legalbenchrag] 完成，输出目录: {run_dir}")
    return 0


def _write_legalbench_outputs(run_dir: Path, eval_cfg: dict, ks: list[int],
                              per_benchmark: dict, overall_metrics: dict,
                              n_instances: int, n_errors: int, n_resumed: int,
                              args: argparse.Namespace) -> None:
    started_at = now_iso()
    summary = {
        "mode": "legalbenchrag",
        "started_at": started_at,
        "finished_at": now_iso(),
        "config": {
            "k": ks,
            "agent_limit": args.agent_limit,
            "seed": _seed(args, eval_cfg),
            "legalbench_root": str(args.legalbench_root),
            "benchmarks": list(per_benchmark),
        },
        "metrics": overall_metrics,
        "per_benchmark": per_benchmark,
        "n_instances": n_instances,
        "n_errors": n_errors,
        "n_resumed_skipped": n_resumed,
        "output_dir": str(run_dir),
    }
    write_json(run_dir / "summary.json", summary)

    # metrics.csv：宽表（每 benchmark 一行 + overall 一行）
    metric_keys = [f"recall_at_{k}" for k in ks] + \
        ["mrr", "span_precision", "span_recall", "span_f1",
         "agent_span_precision", "agent_span_recall", "agent_span_f1"]
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
    modules = _init_modules()
    planner_policy = modules.get("planner_policy") or modules.get("default_policy")
    from src.contract.planner import Planner

    planner = Planner(policy=planner_policy)
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

    mode = args.nli_mode or eval_cfg.get("contractnli_mode", "indexed")
    nli_session = args.nli_session or eval_cfg.get("nli_session", "nli-contractnli")
    top_k = args.nli_top_k if args.nli_top_k is not None else int(eval_cfg.get("contractnli_top_k", 8))

    if mode == "indexed":
        n_docs = _session_doc_count(nli_session)
        if n_docs == 0:
            if args.ingest_nli:
                print(f"[contractnli] 会话 {nli_session} 为空，自动入库 "
                      + (f"{len(idx_subset)} 份被采样实例引用的合同（PAKTON 对齐）" if idx_subset else "distinct 合同")
                      + " ...")
                from src.contract.eval.ingest_raw import ingest_contractnli_jsonl
                ingest_contractnli_jsonl(path, nli_session, embed=bool(eval_cfg.get("embedding_corpus", True)),
                                         idx_subset=idx_subset,
                                         max_tokens=eval_cfg.get("ingest_max_tokens"),
                                         min_tokens=int(eval_cfg.get("ingest_min_tokens", 50)),
                                         overlap_tokens=int(eval_cfg.get("ingest_overlap_tokens", 50)))
            else:
                print(f"[contractnli] 会话 {nli_session} 无合同入库（indexed 模式需要全库）。")
                print("  请先手动入库（不会自动执行）:  python -m src.contract.eval.ingest_raw nli --contractnli-jsonl <jsonl/zip> --session " + nli_session)
                print("  或者加 --ingest-nli 让本次自动入库。")
                return 1
        print(f"[contractnli] 会话 {nli_session} 合同数 = {_session_doc_count(nli_session)}（indexed：整库入库+检索式）")

    limit = args.limit if args.limit is not None else eval_cfg.get("contractnli_limit")
    seed = _seed(args, eval_cfg)
    out_dir = Path(args.out or eval_cfg["out_dir"])
    run_dir = make_run_dir(out_dir, "contractnli")
    records_path = run_dir / "records.jsonl"
    done = load_done_ids(records_path)

    print(f"[contractnli] 总 {len(records)} 条实例（subset={args.subset or 'all'}，mode={mode}），"
          f"已完成 {len(done)}，本次处理上限 {limit or '全部'}")
    stats = run_contractnli(records, planner=planner, records_path=records_path,
                            limit=limit, seed=seed, done_ids=done,
                            mode=mode, nli_session=nli_session, top_k=top_k)
    summ = summarize_contractnli_records(records_path)

    summary = {
        "mode": "contractnli",
        "started_at": now_iso(),
        "finished_at": now_iso(),
        "config": {"limit": limit, "seed": seed, "subset": args.subset,
                   "nli_mode": mode, "nli_session": nli_session, "nli_top_k": top_k,
                   "contractnli_jsonl": str(path)},
        "metrics": {k: v for k, v in summ.items() if k not in ("class_counts_true", "class_counts_pred")},
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
    write_csv(run_dir / "metrics.csv", rows)

    print(f"[contractnli] 本次新增 {stats['evaluated']} 条（累计 {summ['n_total']}，错误 {summ['n_errors']}"
          + (f"，检索空 {summ['n_indexed_retrieved_zero']}" if mode == "indexed" else "") + "）")
    print(f"[contractnli] accuracy={summ['accuracy']:.4f}  f1_weighted={summ['f1_weighted']:.4f}")
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
