"""请求/入库集合采样（LegalBenchRAG 对齐 PAKTON 的"请求数封顶 194 + 等权聚合"口径）。

- LegalBenchRAG：把 total 条请求按 4 个 benchmark 的"原任务比例"（各基准 query 数占
  全体比例）分配；若 weight="equal" 则按 PAKTON 的等权 0.25 分配（每条基准均分）。
  doc_ids 仅记录"被采到请求引用的文档"（供 --ingest-only-referenced 调试）；
  默认入库仍为每基准全量语料（PAKTON 默认 SORT_BY_DOCUMENT=False 即整库入库）。
- ContractNLI：按"原标签比例"（entailment/contradiction/neutral 在原始子集中的分布）
  分层抽样 total 条；入库合同 = 被采到实例对应的 distinct 合同（nli:{premise_id}），
  与 PAKTON"每条先索引该合同再问"一致。

全部用稳定 seed 可复现；输出可直接交给 main.py 的 --request-set 消费。
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from . import loaders as L

__all__ = ["legalbench_request_set", "contractnli_request_set", "write_request_set"]


def _alloc(total: int, weights: list[float]) -> list[int]:
    """把 total 按 weights 做最大余数分配，返回整数分配（和为 total）。"""
    if not weights or total <= 0:
        return [0] * len(weights)
    wsum = sum(weights)
    raw = [total * w / wsum for w in weights]
    floored = [int(x) for x in raw]
    rem = total - sum(floored)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floored[i], reverse=True)
    for i in order[:rem]:
        floored[i] += 1
    return floored


def legalbench_request_set(
    root: str | Path,
    total: int = 100,
    seed: int = 0,
    only: list[str] | None = None,
    weight: str = "count",
) -> dict:
    """生成 LegalBenchRAG 请求与入库集合。

    返回 {mode, total, seed, weight, samples:{benchmark:{count, queries:[instance_id],
    doc_ids:[corpus 相对 file_path]}}}。
    """
    names = only or L.legal_benchmark_names(Path(root))
    qs_by_name = {n: L.load_legalbench_queries(Path(root), n) for n in names}
    counts = {n: len(qs_by_name[n]) for n in names}

    if weight == "equal":
        alloc = _alloc(total, [1.0] * len(names))
    else:  # 原任务比例
        alloc = _alloc(total, [counts[n] for n in names])

    samples: dict = {}
    for n, nq in zip(names, alloc):
        ids_all = [q["instance_id"] for q in qs_by_name[n]]
        rng = random.Random(f"legalbench:{n}:{seed}")
        chosen = rng.sample(ids_all, min(nq, len(ids_all))) if nq > 0 else []
        chosen_docs: set[str] = set()
        for q in qs_by_name[n]:
            if q["instance_id"] in chosen:
                chosen_docs.update(q["gold_docs"])
        samples[n] = {
            "count": len(chosen),
            "queries": chosen,
            "doc_ids": sorted(chosen_docs),
        }
    return {
        "mode": "legalbenchrag",
        "total": sum(v["count"] for v in samples.values()),
        "seed": seed,
        "weight": weight,
        "samples": samples,
    }


def contractnli_request_set(
    jsonl_path: str | Path,
    total: int = 150,
    seed: int = 0,
    subset: str | None = "test",
) -> dict:
    """生成 ContractNLI 请求与入库集合（按原标签比例分层抽样）。

    返回 {mode, subset, total, seed, label_counts, label_proportions_orig,
    instances:[{instance_id, premise_id, label}], contracts:[premise_id...]}。
    """
    path = L.find_contractnli_jsonl(str(jsonl_path)) if jsonl_path else L.find_contractnli_jsonl()
    recs = L.load_contractnli_records(path, subset=subset)
    if not recs:
        raise ValueError(f"ContractNLI 无实例（subset={subset!r}）")

    dist = Counter(r["label"] for r in recs)
    labels = sorted(dist)
    alloc = _alloc(total, [dist[l] for l in labels])

    chosen: list[dict] = []
    for lab, n in zip(labels, alloc):
        pool = [r for r in recs if r["label"] == lab]
        rng = random.Random(f"nli:{lab}:{seed}")
        picked = rng.sample(pool, min(n, len(pool)))
        chosen.extend({"instance_id": r["instance_id"], "premise_id": r["premise_id"],
                       "label": r["label"]} for r in picked)
    contracts = sorted({c["premise_id"] for c in chosen})
    return {
        "mode": "contractnli",
        "subset": subset,
        "total": len(chosen),
        "seed": seed,
        "label_counts": {l: dist[l] for l in labels},
        "label_counts_sampled": {l: sum(1 for c in chosen if c["label"] == l) for l in labels},
        "label_proportions_orig": {l: round(dist[l] / len(recs), 4) for l in labels},
        "instances": chosen,
        "contracts": contracts,
    }


def write_request_set(obj: dict, out_path: str | Path) -> Path:
    """落盘请求集合 JSON。"""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
