"""评测记录的数据结构（用于持久化与汇总）。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class LegalChunkHit(BaseModel):
    """一次检索命中的一条片段（统一表示：确定性 top-k 与 LLM Searcher 证据共用）。"""

    rank: int = Field(default=0, description="排名（从 1 开始；LLM Searcher 证据可无排名）")
    doc_id: str = ""
    file_path: str = ""  # corpus 相对路径（对齐 gold 的 file_path）
    span: list[int] = Field(default_factory=list)  # [start, end] 字符偏移（对齐 raw corpus）
    text: Optional[str] = None
    score: Optional[float] = None


class LegalQueryRecord(BaseModel):
    """LegalBenchRAG 单条 query 的完整评测记录。"""

    instance_id: str
    benchmark: str
    query: str
    gold_docs: list[str] = Field(default_factory=list, description="相关文档 file_path 集合")
    gold_spans: dict[str, list[list[int]]] = Field(default_factory=dict, description="file_path -> [[s,e],...]")
    # 确定性混合检索的 top-k 排名（Recall@k / MRR 依据）
    ranked_hits: list[LegalChunkHit] = Field(default_factory=list)
    # LLM Searcher 实际召回的证据（span 覆盖指标依据）
    searcher_hits: list[LegalChunkHit] = Field(default_factory=list)
    # 单条得分：recall_at_{k} / mrr / span_precision / span_recall / span_f1 /
    #           agent_span_precision / agent_span_recall / agent_span_f1 / evidence_hit_rate
    scores: dict[str, float] = Field(default_factory=dict)
    searcher_searched: bool = False
    searcher_error: Optional[str] = None
    prompt: str = ""  # agent 模式下的 prompt（用于留存）
    raw_response: str = ""  # agent 原始输出
    searcher_trajectory: list = Field(default_factory=list)  # agent 结构化完整轨迹（含 tool 调用）
    telemetry: dict = Field(default_factory=dict)
    elapsed_s: float = 0.0


class NliRecord(BaseModel):
    """ContractNLI 单条 (premise, hypothesis) 的分类评测记录。"""

    instance_id: str
    premise_id: str = ""  # 合同在 jsonl 中的行索引
    subset: str = ""  # train/dev/test
    premise_preview: str = ""  # 仅存前 N 字符，避免超大 JSON
    hypothesis: str
    gold_label: str
    pred_label: Optional[str] = None  # entailment/contradiction/neutral 或 None(解析失败)
    pred_valid: bool = True  # True=解析出合法标签；False=记录为错误样例(-1)
    mode: str = "fullchain"  # fullchain=完整链路(Planner→Searcher→Reviewer→Refiner)+3选1 Refiner 提示词
    doc_id: str = ""  # 对应合同 doc_id（fullchain 下为 nli:<premise_id>）
    retrieved_n: int = 0  # legacy indexed 模式字段（fullchain 用 telemetry.n_evidence）
    retrieved_chunks: list = Field(default_factory=list)  # [{id, text_preview, span, score}]
    prompt: str = ""
    raw_response: str = ""
    reasoning: str = ""  # 可选
    correct: Optional[bool] = None
    telemetry: dict = Field(default_factory=dict)
    elapsed_s: float = 0.0
    error: Optional[str] = None


class EvalSummary(BaseModel):
    """一次运行的整体汇总（写入 summary.json）。"""

    mode: str  # legalbenchrag | contractnli
    started_at: str = ""
    finished_at: str = ""
    elapsed_s: float = 0.0
    config: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict, description="聚合指标")
    telemetry: dict = Field(default_factory=dict, description="编排、检索、证据校验和引用审计汇总")
    per_benchmark: dict = Field(default_factory=dict, description="legalbenchrag 按 benchmark 分组的指标")
    n_instances: int = 0
    n_errors: int = 0
    n_resumed_skipped: int = 0
    output_dir: str = ""
