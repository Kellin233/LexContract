"""共用 JSON 稳健解析工具（去围栏/去噪/最外层对象或数组提取）。"""
from __future__ import annotations

import json
import re
from typing import Any


def _clean_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = lines[1:] if lines and lines[0].startswith("```") else lines
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return raw


def extract_json_object(text: str) -> dict | None:
    raw = _clean_fence(text)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group(0).strip()
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def extract_json_array(text: str) -> list | None:
    raw = _clean_fence(text)
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[(?:.|\n)*\]", raw)
    if m:
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, list) else None
        except json.JSONDecodeError:
            return None
    return None
