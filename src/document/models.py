"""文档解析与向量入库模块的数据模型。"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """单个切片的定位与来源元数据。

    对应需求：所属文档id、页码、页内坐标、章节层级路径、字符偏移量、文档标题、chapter/section。
    """

    doc_id: str = Field(description="所属文档 ID")
    doc_title: str = Field(default="", description="文档标题")
    # 章节层级路径：从顶级章节到当前章节的所有标题文本（含编号）
    section_path: list[str] = Field(default_factory=list, description="章节层级路径（标题文本列表）")
    # 便捷字段：最高一级章节（如“第3条”）与当前小节（如“3.2”）
    chapter: str = Field(default="", description="顶级章节标题")
    section: str = Field(default="", description="当前小节标题")

    page_no: int = Field(description="切片起始页码（1 基）")
    # 页内坐标边界框 (x0, y0, x1, y1)，单位随 Docling（点数），坐标原点为页面左上角
    bbox: list[float] = Field(default_factory=list, description="起始页内 bbox [x0,y0,x1,y1]")
    # 字符偏移：切片文本在拼接后文档全文中的全局起止位置
    charspan: list[int] = Field(default_factory=list, description="全局字符偏移 [start, end]")

    label: str = Field(default="paragraph", description="内容类型：paragraph/list_item/table/title/section_header")
    source_format: str = Field(default="", description="来源格式 txt/pdf/docx")

    # 说明：embedding 不入 metadata，单独挂在 Chunk 上；bbox 为列表便于 JSON 序列化。


class Chunk(BaseModel):
    id: str = Field(description="唯一切片 ID，形如 {doc_id}:{序号}")
    text: str = Field(description="切片文本")
    metadata: ChunkMetadata
    # 向量不参与 JSON 导出，仅入库时使用；解析阶段可能为空
    embedding: Optional[list[float]] = Field(default=None, exclude=True)


class ParsedBlock(BaseModel):
    """解析器输出的归一化结构块：标题或内容，供切分器消费。"""

    kind: str = Field(description="heading / content")
    label: str = Field(default="paragraph", description="paragraph/list_item/table/title/section_header")
    level: int = Field(default=0, description="标题层级（content 为 0）")
    text: str = Field(default="")
    page_no: int = Field(default=0)
    bbox: list[float] = Field(default_factory=list)
    # 块文本在 ParsedDocument.full_text 中的全局起止偏移
    offset: list[int] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    """单个文档的完整解析结果。"""

    doc_id: str = Field(description="文档 ID（通常是文件名去扩展名 + 短哈希）")
    file_path: str = Field(description="原始文件路径")
    title: str = Field(default="", description="文档标题")
    source_format: str = Field(default="", description="txt/pdf/docx")
    full_text: str = Field(default="", description="按 Docling 阅读顺序拼接的全文，作为字符偏移基准")
    # 结构信息（标题层级树，便于调试与后续模块引用）
    structure: list[dict[str, Any]] = Field(default_factory=list)
    # 归一化结构块（切片输入），不落盘 JSON
    blocks: list[ParsedBlock] = Field(default_factory=list, exclude=True)
    chunks: list[Chunk] = Field(default_factory=list)
