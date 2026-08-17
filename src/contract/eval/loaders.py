"""评测数据加载器：从本机文件读取 ContractNLI 与 LegalBenchRAG 数据。

数据来源（本机已具备，无需联网）：
- ContractNLI: PAKTON-develop/.../data/cache/contract_nli_long.zip
  -> contract_nli_long.jsonl: {"premise", "hypothesises/labels":[{hypothesis,label}], "subset"}
- LegalBenchRAG: <root>/benchmarks/{contractnli,cuad,maud,privacy_qa}.json
                 <root>/corpus/{...}/*.txt
"""
from __future__ import annotations

import json
from pathlib import Path

# ---------- 默认路径探测（本机已知位置） ----------
_CONTRACTNLI_CANDIDATES = [
    "/root/LLM projects/PAKTON-develop/Experiments and Evaluation/Quantitative/data/cache/contract_nli_long.zip",
    "/root/LLM projects/PAKTON-raw/data/cache/contract_nli_long.zip",  # 兜底
]

_LEGALBENCH_CANDIDATES = [
    "/root/LLM projects/LexTrace-main/evaluation/data/legalbenchrag",
    "/root/LLM projects/PolicyPilot/evaluation/data/legalbenchrag",
]

LEGALBENCH_NAMES = ["contractnli", "cuad", "maud", "privacy_qa"]


def find_contractnli_jsonl(explicit: str | None = None) -> Path:
    """返回 ContractNLI jsonl 的读取入口。

    - explicit 若为 .jsonl 文件，直接返回；
    - 若为 .zip（含 contract_nli_long.jsonl），返回该 zip（由调用方按 zip 读取）。
    """
    if explicit:
        p = Path(explicit)
        p = p / "contract_nli_long.jsonl" if p.is_dir() else p
        return p
    for cand in _CONTRACTNLI_CANDIDATES:
        p = Path(cand)
        if p.is_file():
            return p
    raise FileNotFoundError(
        "未找到 ContractNLI 数据；请用 --contractnli-jsonl 指定 .jsonl 或包含它的 .zip"
    )


def _read_jsonl_lines(path: Path) -> list[str]:
    if path.suffix == ".zip":
        import zipfile

        with zipfile.ZipFile(path) as z:
            member = next(n for n in z.namelist() if n.endswith(".jsonl"))
            return z.read(member).decode("utf-8").splitlines()
    return path.read_text(encoding="utf-8").splitlines()


def load_contractnli_records(path: Path, subset: str | None = None) -> list[dict]:
    """展开 jsonl 为逐条分类实例。

    每条返回:
      {premise_id, premise, hypothesis, label, subset}
    premise 为整份合同文本；label ∈ {entailment, contradiction, neutral}。
    """
    out: list[dict] = []
    for idx, line in enumerate(_read_jsonl_lines(path)):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        premise = rec.get("premise", "")
        sub = rec.get("subset", "")
        if subset and sub != subset:
            continue
        for i, h in enumerate(rec.get("hypothesises/labels", [])):
            out.append({
                "instance_id": f"contractnli-{sub}-{idx}-{i}",
                "premise_id": f"{idx}",
                "premise": premise,
                "hypothesis": h.get("hypothesis", ""),
                "label": h.get("label", ""),
                "subset": sub,
            })
    return out


def load_contractnli_premises(path: Path, subsets: list[str] | None = None) -> list[dict]:
    """按“前提（合同）”去重读取，供整库入库用。

    每条返回 {idx, subset, premise, n_hypotheses, premise_len}；idx 即 jsonl 行号
    （与 load_contractnli_records 的 premise_id 对齐，实例 doc_id 稳定）。
    """
    out: list[dict] = []
    for idx, line in enumerate(_read_jsonl_lines(path)):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        sub = rec.get("subset", "")
        if subsets and sub not in subsets:
            continue
        premise = rec.get("premise", "")
        if not premise:
            continue
        out.append({
            "idx": f"{idx}",
            "subset": sub,
            "premise": premise,
            "premise_len": len(premise),
            "n_hypotheses": len(rec.get("hypothesises/labels", [])),
        })
    return out


# ---------- LegalBenchRAG ----------
def find_legalbench_root(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    for cand in _LEGALBENCH_CANDIDATES:
        p = Path(cand)
        if (p / "benchmarks").is_dir():
            return p
    raise FileNotFoundError("未找到 LegalBenchRAG 数据；请用 --legalbench-root 指定")


def legal_benchmark_names(root: Path) -> list[str]:
    return sorted(
        p.stem for p in (root / "benchmarks").glob("*.json")
        if p.stem in LEGALBENCH_NAMES
    )


def load_legalbench_queries(root: Path, name: str) -> list[dict]:
    """读取一个 benchmark，返回 query 列表。

    每项: {benchmark, query, gold_docs:[file_path...], gold_spans:{file_path:[[s,e],...]}}
    """
    path = root / "benchmarks" / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict] = []
    for i, t in enumerate(data.get("tests", [])):
        snippets = t.get("snippets", [])
        gold_spans: dict[str, list[list[int]]] = {}
        gold_docs: list[str] = []
        for sn in snippets:
            fp = sn.get("file_path", "")
            sp = sn.get("span", [])
            if not fp or len(sp) != 2:
                continue
            if fp not in gold_spans:
                gold_spans[fp] = []
            gold_spans[fp].append([int(sp[0]), int(sp[1])])
            if fp not in gold_docs:
                gold_docs.append(fp)
        out.append({
            "benchmark": name,
            "instance_id": f"{name}-{i}",
            "query": t.get("query", ""),
            "gold_docs": gold_docs,
            "gold_spans": gold_spans,
        })
    return out


def corpus_text(root: Path, file_path: str) -> str:
    """按 legalbenchrag 相同方式读取语料原文（文本模式 → 统一换行），保证偏移一致。"""
    p = root / "corpus" / file_path
    return p.read_text(encoding="utf-8", errors="replace")


def corpus_slice(root: Path, file_path: str, span: list[int]) -> str:
    text = corpus_text(root, file_path)
    s, e = int(span[0]), int(span[1])
    return text[s:e]


def verify_benchmark_alignment(root: Path, name: str, n: int = 5) -> dict:
    """抽样校验 gold span 切片文本与 benchmark 自带 answer 是否一致。

    benchmark json 里 snippets 若带 `answer` 字段，用它逐项比对（span -> corpus 切片）；
    无 answer 字段时退化为“切片是否存在且非空”的非破坏性检查。
    返回 {checked, ok, mismatched, samples:[{file_path, span, slice_head, answer_head, ok}]}
    """
    path = root / "benchmarks" / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    checked = mismatched = 0
    samples: list[dict] = []
    for t in data.get("tests", []):
        for sn in t.get("snippets", []):
            fp = sn.get("file_path", "")
            sp = sn.get("span", [])
            if not fp or len(sp) != 2:
                continue
            if checked >= n:
                break
            try:
                sliced = corpus_slice(root, fp, sp)
            except Exception as e:  # noqa: BLE001
                sliced, err = "", str(e)
            ans = sn.get("answer")
            ok = ans is None or ans == sliced or ans.replace("\n", " ") in sliced or sliced in ans
            if ok is False:
                mismatched += 1
            checked += 1
            samples.append({
                "file_path": fp, "span": sp,
                "slice_head": sliced[:80], "answer_head": (ans or "")[:80], "ok": ok,
            })
        if checked >= n:
            break
    return {"checked": checked, "ok": checked - mismatched, "mismatched": mismatched, "samples": samples}
