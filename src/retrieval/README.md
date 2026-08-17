# LexContract / retrieval —— 检索服务模块（BM25 + 向量 + 混合 + 重排）

`src/retrieval/` 是 LexContract 项目的一个独立子模块，基于 `document/` 模块已入库的切片与向量，提供面向合同交叉检索的**检索服务**：

- **BM25（稀疏/关键词）检索**：基于数据库内 **pg_search (ParadeDB)** 扩展，索引 `chunks.search_tokens`（jieba 分词后的空格 token 串）。
- **向量（稠密）检索**：基于 **pgvector** 的 `chunks.embedding`（BAAI/bge-m3, 1024 维, HNSW cosine 索引）。
- **混合检索**：BM25 + 向量候选，按 **加权 Reciprocal Rank Fusion (RRF)** 融合（`weight/(k+rank)`，默认 k=60，权重 0.5/0.5）。
- **重排（可选）**：BGE cross-encoder（`BAAI/bge-reranker-v2-m3`）对候选按 `(query, chunk)` 打分取 top-k。
- **会话/工作区作用域**：检索严格限定在某个 `session`，拒绝无作用域查询。

设计参考 PAKTON-develop 的 `EvidenceAgent/retrievers/postgres.py`。

## 目录结构

```
src/retrieval/
├── requirements.txt      # 依赖（新增 jieba；其余复用 document）
├── .env.example          # 配置模板
├── config.py             # 读取环境变量/.env
├── models.py             # RetrievedChunk 结果模型
├── tokenizer.py          # jieba + NFKC 分词（索引/查询对齐）
├── store.py              # 连接、建表、会话分派、search_tokens 回填
├── postgres.py           # PostgresRetriever：vector / bm25 / hybrid / retrieve
├── reranker.py           # BGE cross-encoder 重排
├── schema.sql            # pg_search 扩展 + 加列 + BM25 索引（幂等）
├── main.py               # CLI
└── README.md
```

## 前置条件

1. **PostgreSQL + pgvector + pg_search**：BM25 依赖 **pg_search (ParadeDB)** 扩展。标准 `pgvector/pgvector` 镜像**不含**该扩展，请使用 ParadeDB 版镜像（内建 pgvector + pg_search），例如本机本地验证用的：
   ```bash
   docker run -d --name lexcontract-paradedb \
     -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=lexcontract \
     -p 5434:5432 paradedb/paradedb:0.25.2-pg16
   ```
   然后在 `.env` 把 `PG_PORT` 指向对应端口（与 document 模块共用同一数据库）。注意：直接从普通镜像拷贝 pg_search 的 `.so` 常因 glibc 版本不匹配而失败，建议直接用 ParadeDB 镜像。
2. **模型下载**：查询向量化复用 document 的 `BAAI/bge-m3`；重排用 `BAAI/bge-reranker-v2-m3`（首次运行下载）。直连不可达时设置 `HF_ENDPOINT=https://hf-mirror.com` 下载；下载完成后可用 `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1` 离线加载。

## 安装与配置

```bash
pip install -r src/retrieval/requirements.txt
cp src/retrieval/.env.example src/retrieval/.env   # 按需修改
```

## 使用（在项目根目录运行）

```bash
# 0) 先入库：document 模块（见其 README）
python3 -m src.document.main init-db
python3 -m src.document.main parse 合同.pdf

# 1) 初始化检索结构（pg_search 扩展 + session_id / search_tokens 列 + BM25 索引）
python3 -m src.retrieval.main init-db

# 2) 回填 search_tokens（document 入库未写该列，须先回填）
python3 -m src.retrieval.main backfill

# 3) 把文档分派到会话（工作区）
python3 -m src.retrieval.main assign <doc_id> --session S1

# 4) 查询
python3 -m src.retrieval.main query "甲方逾期付款的违约金如何计算" \
    --session S1 --mode hybrid --top-k 10
```

### CLI 子命令

| 命令 | 说明 |
|------|------|
| `init-db` | 建 pg_search 扩展、加 `session_id`/`search_tokens` 列、建 BM25 索引（幂等） |
| `assign <doc_id> --session <sid>` | 将文档分派到会话（`--session ''` 或省略=取消归属） |
| `unassign <doc_id>` | 取消文档的会话归属 |
| `sessions` | 列出各会话的文档数 |
| `backfill [--doc-id ...]` | 为 `search_tokens=''` 的切片回填 jieba 分词 |
| `query "<问题>" --session <sid> [选项]` | 检索查询 |

`query` 选项：
- `--mode hybrid|vector|bm25`（默认 `hybrid`）
- `--top-k N`（默认 `TOP_K=10`）
- `--candidate-k N`（混合检索候选池大小）
- `--doc-id ...`（限定文档，默认会话内全部）
- `--no-rerank`（默认开启 BGE 重排，可关闭）

## 数据库变更（相对 document）

`retrieval/schema.sql` 在 `documents`/`chunks` 上做**增量、幂等**变更：

- `documents` 新增 `session_id TEXT NOT NULL DEFAULT ''`（会话作用域，`''`=未分配）。
- `chunks` 新增 `search_tokens TEXT NOT NULL DEFAULT ''`（Jieba 空格分词串）。
- 新增 pg_search **BM25 索引** `USING bm25 (id, search_tokens) WITH (key_field='id')`。
- 向量检索沿用 document 已建的 HNSW(cosine) 索引，无破坏性改动。

## 结果模型（RetrievedChunk）

每次命中返回：`id`、`text`、`doc_id`、`doc_title`、`session_id`、`page_no`、`section_path`（章节路径）、`charspan`（全文偏移）、`source_format`，加模式相关得分（`vectordb_similarity_score` / `bm25_score` / `rrf_score`，重排后含 `rerank_score`）。该结构供后续**多轮分析/结论生成**模块作为可溯源的证据上下文使用。

## 说明与限制（本阶段）

- **不做 HTTP/服务化**：当前仅库 + CLI，供后续模块以 Python 直接调用 `PostgresRetriever`。
- **search_tokens 靠 backfill 补齐**：`document/parse` 入库时未写该列，需在入库后运行 `backfill`；后续可让 document 入库时顺带写入以省去该步。
- **多轮分析 / 结论生成**：属于后续独立模块，本模块只负责“依据证据检索”。
