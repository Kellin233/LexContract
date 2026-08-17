"""把解析结果导出为每文档一个 JSON 文件。"""
from __future__ import annotations

import json
from pathlib import Path

from . import config
from .models import ParsedDocument


def export_document(doc: ParsedDocument, out_dir: Path | None = None) -> Path:
    out_dir = (out_dir or config.OUTPUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # embedding 不落盘（exclude=True），保持体积可控
    payload = doc.model_dump(mode="json")
    out_path = out_dir / f"{doc.doc_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path
