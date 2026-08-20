"""Docling 结构化解析：把 TXT/PDF/DOCX 归一化为结构块列表。

- PDF：Docling 提供 label（title/section_header/paragraph…）、标题层级、页码与页内 bbox。
- DOCX：提供 label 与标题层级，但无边距分页（无页码/坐标）。
- TXT：Docling 不识别结构，退化为按空白行分段的纯文本，需用启发式识别章节标题。

输出 ParsedBlock（含全局字符偏移）与 ParsedDocument（含拼接全文与结构树）。
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import TableItem

from . import config
from .models import ParsedBlock, ParsedDocument

# 结构层级的空值
_NO_LEVEL = 0

# ---------- 标题启发式（用于 TXT 或 Docling 未标出标题的场景） ----------
_CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百千万0-9０-９]+[条章节款]")
# 阿拉伯多级编号：1. / 1.2 / 1.2.3 （层级 = 段数）
_ARABIC_RE = re.compile(r"^\d+(?:\.\d+)*(?:[、.)\s]|$)")
# 标准罗马数字（大写）+ 分隔符：I. / IV. / XII.
_ROMAN_RE = re.compile(
    r"^(?=[IVXLCDM]{1,8}\b)"
    r"(?:M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))"
    r"(?=[.):、])"
)
# 英文字母编号：A. / a)
_ALPHA_RE = re.compile(r"^[A-Za-z](?=[.):、])")
# 括号条目：（一）(二) (1) (A)
_PAREN_RE = re.compile(r"^[（(](?:\d{1,3}|[A-Za-z]|[一二三四五六七八九十百]{1,4})[）)]")
# 中文顿号条目：一、 二、
_CJK_LIST_RE = re.compile(r"^[一二三四五六七八九十]{1,4}[、]")


def _heuristic_heading_level(text: str) -> int | None:
    """估算标题层级；非标题返回 None。"""
    t = text.strip()
    if len(t) > 60:
        return None
    # 以句号/问号/感叹号/分号结尾的更像句子，不作为标题
    if t[-1:] in "。！？；.!?;":
        return None
    if _CHAPTER_RE.match(t):
        return 1
    m = _ARABIC_RE.match(t)
    if m:
        return m.group(0).strip(" 、.)\t").count(".") + 1
    if _ROMAN_RE.match(t) or _ALPHA_RE.match(t):
        return 2
    if _PAREN_RE.match(t) or _CJK_LIST_RE.match(t):
        return 3
    return None


# ---------- 全局转换器 ----------
_converter: DocumentConverter | None = None


def _build_converter() -> DocumentConverter:
    """按配置构造转换器：数字版 PDF 默认关闭 OCR 以提速。"""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption

    if not config.PDF_DO_OCR:
        opts = PdfPipelineOptions()
        opts.do_ocr = False
        opts.do_table_structure = config.PDF_DO_TABLE_STRUCTURE
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
            }
        )
    return DocumentConverter()


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = _build_converter()
    return _converter


# ---------- 非常轻量的单例，避免重复 init ----------
_conversion_cache: dict[Path, ParsedDocument] = {}


def _item_to_block(item, label: str, text: str) -> ParsedBlock:
    """从 Docling 条目提取页码/坐标/文本，组装成块。"""
    page_no = 0
    bbox: list[float] = []
    if item.prov:
        p = item.prov[0]
        page_no = p.page_no
        if p.bbox is not None:
            bbox = [float(v) for v in p.bbox.as_tuple()]
    return ParsedBlock(kind="content", label=label, level=_NO_LEVEL, text=text,
                       page_no=page_no, bbox=bbox)


def _extract_blocks(doc) -> list[ParsedBlock]:
    """按阅读顺序提取标题块与内容块。"""
    blocks: list[ParsedBlock] = []
    for item, _level in doc.iterate_items():
        if isinstance(item, TableItem):
            text = item.export_to_markdown()
            blocks.append(_item_to_block(item, "table", text))
            continue
        label = getattr(item, "label", None)
        label_val = label.value if label is not None else "text"
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue

        is_docling_heading = label_val in ("title", "section_header")
        if is_docling_heading:
            lvl = getattr(item, "level", None) or 1
            kind = "heading"
            blocks.append(
                ParsedBlock(kind=kind, label=label_val, level=int(lvl), text=text,
                            page_no=0, bbox=[], offset=[])
            )
            continue

        # 内容块：优先 docling 标签，否则启发式标题
        hl = _heuristic_heading_level(text) if label_val in ("text", "paragraph") else None
        if hl is not None:
            lbl = "section_header"
            blocks.append(
                ParsedBlock(kind="heading", label=lbl, level=hl, text=text,
                            page_no=0, bbox=[], offset=[])
            )
            continue

        block = _item_to_block(item, label_val if label_val != "text" else "paragraph", text)
        blocks.append(block)
    return blocks


def _assemble_full_text(blocks: list[ParsedBlock]) -> str:
    """拼接全文并回填每个块的全局字符偏移。"""
    parts: list[str] = []
    offset = 0
    for blk in blocks:
        if not blk.text:
            blk.offset = [offset, offset]
            continue
        # 用换行分隔块，保证偏移可回溯
        sep = "\n" if parts else ""
        text = blk.text if not sep else "\n" + blk.text
        # 块区间只覆盖块正文；块间分隔换行仍保留在 full_text 中。
        start = offset + len(sep)
        offset += len(text)
        blk.offset = [start, offset]
        parts.append(text)
    return "".join(parts)


def _doc_title(doc, path: Path, blocks: list[ParsedBlock]) -> str:
    # 文档首个短内容行若不构成句子，通常即标题（多为 TXT 场景）
    if blocks and blocks[0].kind == "content":
        first = blocks[0].text.strip()
        if first and len(first) <= 30 and first[-1:] not in "。！？；.!?;":
            return first
    for blk in blocks:
        if blk.kind == "heading":
            return blk.text
    name = getattr(doc, "name", None)
    if name:
        return str(name)
    return path.stem


def _parse_text(path: Path) -> ParsedDocument:
    """TXT：本地读取并做启发式章节标记，不依赖 Docling（其对 TXT 无结构输出）。

    逐行判定标题（层次由编号标记启发式给出）与内容，交由切片器组装。
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    blocks: list[ParsedBlock] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lvl = _heuristic_heading_level(stripped)
        if lvl is not None:
            blocks.append(
                ParsedBlock(kind="heading", label="section_header",
                            level=lvl, text=stripped, offset=[])
            )
        else:
            blocks.append(
                ParsedBlock(kind="content", label="paragraph",
                            text=stripped, offset=[])
            )
    full_text = _assemble_full_text(blocks)
    title = _doc_title(None, path, blocks)
    content_hash = hashlib.md5((full_text or text).encode("utf-8", "ignore")).hexdigest()[:8]
    doc_id = f"{path.stem}-{content_hash}"
    structure = [{"level": b.level, "text": b.text} for b in blocks if b.kind == "heading"]
    return ParsedDocument(
        doc_id=doc_id,
        file_path=str(path),
        title=title,
        source_format="txt",
        full_text=full_text,
        structure=structure,
        blocks=blocks,
        chunks=[],
    )


def parse_file(path: Path | str) -> ParsedDocument:
    """解析单个文件，返回 ParsedDocument（含结构块，尚未切片/向量化）。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    source_format = path.suffix.lower().lstrip(".")
    # TXT 直接本地解析，绕开 Docling 对纯文本的无结构路径
    if source_format == "txt":
        return _parse_text(path)

    doc = _get_converter().convert(path).document
    blocks = _extract_blocks(doc)
    full_text = _assemble_full_text(blocks)

    title = _doc_title(doc, path, blocks)

    # 文档 ID：文件名去扩展名 + 内容短哈希（稳定、可追溯）
    content_hash = hashlib.md5((full_text or path.read_bytes()).encode("utf-8", "ignore")).hexdigest()[:8]
    doc_id = f"{path.stem}-{content_hash}"

    structure = [
        {"level": b.level, "text": b.text}
        for b in blocks if b.kind == "heading"
    ]

    return ParsedDocument(
        doc_id=doc_id,
        file_path=str(path),
        title=title,
        source_format=source_format,
        full_text=full_text,
        structure=structure,
        blocks=blocks,
        # chunks 由 chunker 生成，此处留空
        chunks=[],
    )
