"""检索模块的数据模型。"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """单条检索命中的切片（面向后续多轮分析/结论生成模块消费）。"""

    id: str = Field(description="切片 ID（{doc_id}:{index}）")
    text: str = Field(description="切片文本")
    doc_id: str = Field(default="", description="所属文档 ID")
    doc_title: str = Field(default="", description="文档标题")
    session_id: str = Field(default="", description="所属会话/工作区")
    # 定位信息（与 document.ChunkMetadata 对齐，便于溯源原文）
    page_no: int = Field(default=0)
    section_path: list[str] = Field(default_factory=list, description="章节层级路径（标题文本列表）")
    charspan: list[int] = Field(default_factory=list, description="全局字符偏移 [start,end]")
    source_format: str = Field(default="")

    retriever: str = Field(default="hybrid", description="vector / bm25 / hybrid")
    # 各类得分（按 mode 填充，重排后含 rerank_score）
    bm25_score: Optional[float] = Field(default=None)
    vectordb_similarity_score: Optional[float] = Field(default=None)
    rrf_score: Optional[float] = Field(default=None)
    rerank_score: Optional[float] = Field(default=None)

    @classmethod
    def from_row(
        cls,
        *,
        id: str,
        text: str,
        doc_id: str,
        title: str,
        session_id: str,
        page_no: int,
        section_path: list[str],
        charspan: list[int],
        source_format: str,
        retriever: str,
        scores: dict[str, Any],
    ) -> "RetrievedChunk":
        return cls(
            id=id,
            text=text,
            doc_id=doc_id,
            doc_title=title,
            session_id=session_id,
            page_no=page_no,
            section_path=section_path or [],
            charspan=charspan or [],
            source_format=source_format,
            retriever=retriever,
            bm25_score=scores.get("bm25_score"),
            vectordb_similarity_score=scores.get("vectordb_similarity_score"),
            rrf_score=scores.get("rrf_score"),
            rerank_score=scores.get("rerank_score"),
        )
