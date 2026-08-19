"""LexContract retrieval 模块配置：从环境变量/.env 读取（复用 document 的 PG/EMBED 配置键）。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# 加载当前模块目录或项目根目录下的 .env
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent.parent / ".env")  # src/retrieval -> src -> LexContract 根

# --- PostgreSQL（与 document 模块一致）---
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB = os.getenv("PG_DB", "lexcontract")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")

# --- embedding（复用 document 模块模型）---
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")

# --- 检索默认参数 ---
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "100"))
TOP_K = int(os.getenv("TOP_K", "10"))
# search/grep 返回给 Agent 的片段长度（字符数），仅后端可配，不对 Agent 暴露参数
SNIPPET_CHARS = int(os.getenv("SNIPPET_CHARS", "200"))
# 证据物化：整章（最末级 section）文本超过该 token 数时回退为命中切片并集
MAX_EVIDENCE_SECTION_TOKENS = int(os.getenv("MAX_EVIDENCE_SECTION_TOKENS", "3000"))

# 加权 Reciprocal Rank Fusion
RRF_K = int(os.getenv("RRF_K", "60"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.5"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.5"))

# --- 重排（BGE cross-encoder）---
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "10"))
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "32"))
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "512"))
