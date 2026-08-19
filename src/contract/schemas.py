"""合同证据链研究系统：领域数据结构（Evidence / ResearchQuestion / WorkerResult / ReviewResult / RefinerResult / ResearchState）。

设计原则（见方案）：
- Worker 只产出 Evidence（原始连续原文），不产出任何结论。
- Reviewer 只判断“证据是否覆盖问题 / 是否冲突 / 还缺什么”，不裁判。
- Refiner 是唯一生成结论的 Agent，结论必须绑定 [E###] 证据。
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class QuestionStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ReviewStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    NEED_MORE = "NEED_MORE"


class FinalStatus(str, Enum):
    """最终研究状态：SUFFICIENT=证据足够；PARTIALLY_SUFFICIENT=达上限/无有效新增，仍进 Refiner。"""
    SUFFICIENT = "SUFFICIENT"
    PARTIALLY_SUFFICIENT = "PARTIALLY_SUFFICIENT"


class Evidence(BaseModel):
    """一条可引用的原始证据。

    quote 必须是原始文档中的连续原文（由 EvidenceAssembler 从 DB full_text 截取），
    不允许 LLM 改写。relevance_note 只是 Worker 的帮助说明，不能作为引用依据。
    """

    evidence_id: str = Field(default="", description="证据 ID，如 E001（由 EvidenceStore 注册时分配）")
    question_id: str = Field(default="", description="产生该证据的研究问题 ID")
    document_id: str = Field(default="")
    document_name: str = Field(default="")
    section_path: list[str] = Field(default_factory=list, description="章节层级路径")
    page_no: int = Field(default=0)
    source_chunk_ids: list[str] = Field(default_factory=list, description="组装该证据用到的切片 ID")
    start_offset: int = Field(default=0, description="在文档全文中的起始字符偏移")
    end_offset: int = Field(default=0, description="在文档全文中的结束字符偏移")
    quote: str = Field(default="", description="原始文档连续原文")
    retrieval_score: float = Field(default=0.0)
    relevance_note: str = Field(default="", description="Worker 对相关性的解释（辅助理解，不作引用）")
    verified: bool = Field(default=False, description="是否通过 CitationVerifier 校验")
    verify_note: str = Field(default="", description="校验失败原因的机器可读说明（成功后为空串；供评测记录统计丢弃分布）")


class ResearchQuestion(BaseModel):
    """研究问题：只描述“需要调查什么”，绝不包含结论。"""

    question_id: str
    question: str = Field(description="调查目标（不含结论）")
    doc_hints: list[str] = Field(default_factory=list, description="候选文档/关键词提示")
    status: QuestionStatus = QuestionStatus.PENDING


class WorkerResult(BaseModel):
    """Searcher 的最终输出：只携带证据，不携带结论。"""

    question_id: str
    question: str
    evidences: list[Evidence] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    searched: bool = False
    no_evidence_found: bool = False
    drop_reasons: dict[str, int] = Field(
        default_factory=dict,
        description="候选转证据时的丢弃原因计数（materialize-fail 或 verify_note 类名），供评测观测",
    )


class MissingAspect(BaseModel):
    description: str = Field(description="还缺什么信息")
    reason: str = Field(description="为什么需要（挂回原问题）")


class EvidenceConflict(BaseModel):
    evidence_a_id: str
    evidence_b_id: str
    summary: str = Field(default="", description="冲突的客观描述，不包含“谁优先”之类的裁判")


class ReviewResult(BaseModel):
    """Reviewer 输出：只评估研究完整性，不生成结论。"""

    status: ReviewStatus
    covered_aspects: list[str] = Field(default_factory=list)
    missing_aspects: list[MissingAspect] = Field(default_factory=list)
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
    keep_evidence_ids: list[str] = Field(default_factory=list)
    notes: str = Field(default="", description="Reviewer 补充说明")
    effective_new_evidence: bool = Field(
        default=True,
        description="本轮的证据是否实际覆盖了上一轮提出的缺失要点（决定是否提前停止）",
    )


class Citation(BaseModel):
    """Refiner 结果中绑定到证据的引用（由证据元数据生成，禁止模型自编条款号）。"""

    evidence_id: str
    doc_title: str
    section_label: str = Field(description="章节标签，如 “第12.3条”")
    page_no: int
    quote: str = Field(description="被引用的原文片段")


class RefinerPoint(BaseModel):
    claim: str = Field(description="分点结论")
    evidence_ids: list[str] = Field(default_factory=list, description="支撑该分点的证据 ID，必须真实存在")


class RefinerResult(BaseModel):
    """Refiner 一次性综合推理的最终结果（结构化）。"""

    conclusion: str = Field(default="", description="总结论（一两段）")
    points: list[RefinerPoint] = Field(default_factory=list, description="分点结论")
    # 最支持最终结论的证据（可能等于全部，也可能只是其中一部分）
    # —— Refiner 负责“筛选最相关证据”，引用/依据仅落到这组证据上。
    supporting_evidence_ids: list[str] = Field(default_factory=list,
                                               description="最支持最终结论的证据 ID（可能为全部或部分，须真实存在）")
    evidence_gap: list[str] = Field(default_factory=list, description="当前文档中未能确认的缺口")
    citations: list[Citation] = Field(default_factory=list)
    final_status: FinalStatus = FinalStatus.PARTIALLY_SUFFICIENT
    notes: str = Field(default="")
    # 保留行文原文（含 [E###] 引用占位），便于人读与调试
    markdown_body: str = Field(default="")


class ResearchState(BaseModel):
    """当前研究进行到哪一步（不保存 LLM 中间思考）。"""

    original_question: str
    iteration: int = 0
    session_id: str = ""
    doc_ids: list[str] = Field(default_factory=list, description="限定检索的文档；空=全部")
    questions: list[ResearchQuestion] = Field(default_factory=list)
    completed_question_ids: list[str] = Field(default_factory=list)
    active_question_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list, description="全局注册顺序的 Evidence ID 列表")
    evidence_by_question: dict[str, list[str]] = Field(default_factory=dict)
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
    missing_aspects: list[MissingAspect] = Field(default_factory=list)
    review_history: list[ReviewResult] = Field(default_factory=list)
    final_status: Optional[FinalStatus] = None
    effective_new_evidence: bool = True

    def get_question(self, question_id: str) -> Optional[ResearchQuestion]:
        for q in self.questions:
            if q.question_id == question_id:
                return q
        return None
