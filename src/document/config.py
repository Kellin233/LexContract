"""LexTrace document 模块配置：从环境变量/.env 读取。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent  # src/document -> src -> LexTrace 根

# 加载当前模块目录或项目根目录下的 .env；.env.local 优先级更高
load_dotenv(BASE_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BASE_DIR / ".env.local", override=True)
load_dotenv(PROJECT_ROOT / ".env.local", override=True)

# --- PostgreSQL ---
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB = os.getenv("PG_DB", "lextrace")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")

# --- 输出目录（JSON） ---
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "output")).resolve()

# --- 切片参数 ---
CHUNK_TOKEN_CAP = int(os.getenv("CHUNK_TOKEN_CAP", "600"))
CHUNK_MIN_TOKENS = int(os.getenv("CHUNK_MIN_TOKENS", "20"))
# 相邻切片重叠部分（token）；仅在同一章节内生效，跨章节不重叠
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))

# --- embedding ---
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")

# --- Docling PDF 选项 ---
# 数字版 PDF 默认关闭 OCR（快且避免卡顿）；扫描件需置 1 开启
PDF_DO_OCR = os.getenv("PDF_DO_OCR", "0") == "1"
PDF_DO_TABLE_STRUCTURE = os.getenv("PDF_DO_TABLE_STRUCTURE", "1") == "1"

SUPPORTED_EXTS = {".txt", ".pdf", ".docx", ".doc"}
