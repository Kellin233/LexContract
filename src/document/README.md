# LexContract / document —— 文档解析 + 向量入库模块

`src/document/` 是 LexContract 项目的一个独立子模块，负责合同/法律文档的**结构化解析、切片、向量化与入库**，为后续的检索、多轮分析、结论生成等模块提供统一的数据基础。

支持输入：**TXT / PDF / DOCX**（`.doc` 亦有兜底处理）。

## 职责链路

```
输入文档 ──parse──▶ Docling 结构化解析
                 （标题/章节层级、段落、列表、表格）
        ──chunk──▶ 切片（结构优先；过长逐级下切；句子为边界）
        ──embed──▶ 本地多语模型生成向量（默认 BAAI/bge-m3, 1024 维）
        ──store──▶ 每文档 JSON + 写入 PostgreSQL(pgvector)
```

## 目录结构

```
src/document/
├── requirements.txt      # 本模块依赖
├── .env.example          # 配置模板
├── config.py             # 读取环境变量/.env
├── models.py             # Chunk / ChunkMetadata / ParsedBlock / ParsedDocument
├── parser.py             # Docling 结构化解析 + TXT 本地标记解析（标题层级、页码、页内 bbox、全局字符偏移）
├── chunker.py            # 结构感知切片器（同章节 overlap + 无标点有界兜底）
├── text_utils.py         # token 估算、句子切分、embedding 可检索文本
├── embedder.py           # sentence-transformers 本地多语向量化
├── postgres_store.py     # pgvector 连接、建表、事务入库
├── json_exporter.py      # 导出每文档 JSON
├── schema.sql            # documents / chunks 表结构
└── main.py               # CLI：init-db / parse
```

## 安装

```bash
cd src/document
pip install -r requirements.txt
```

> 说明：依赖体积较大（含 Docling 与 sentence-transformers/torch）。首次运行 PDF 解析或 embedding 需下载相应模型。

配置：`cp .env.example .env` 并按需修改（数据库连接、切片 token 上限、embedding 模型等）。

## 前置条件

1. **PostgreSQL + pgvector**：需要一个已安装 `pgvector` 扩展的 PostgreSQL 实例。例如本机用 docker 起一个：
   ```bash
   docker run -d --name lexcontract-pg \
     -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=lexcontract \
     -p 5432:5432 pgvector/pgvector:pg16
   ```
   如使用其他宿主机端口，必须在项目 `.env` 中显式设置统一的 `PG_PORT`。
2. **模型下载**：embedding 默认模型 `BAAI/bge-m3` 首次运行会从 HuggingFace 下载（约 2GB）；PDF 布局/表格模型同理。若 HuggingFace 直连不可达，设置镜像端点：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

## 使用（在项目根目录运行）

```bash
# 1) 初始化数据库表结构（含 vector 扩展与 HNSW 索引）
python3 -m src.document.main init-db

# 2) 解析单个文件或目录：写 JSON + 向量入库
python3 -m src.document.main parse 合同.pdf
python3 -m src.document.main parse ./src/document/samples --out-dir ./src/document/output

# 可选参数
#   --no-db      仅解析导出 JSON，不写数据库
#   --no-embed   不生成 embedding
```

## 每个切片的元数据（`metadata`）

| 字段 | 含义 |
|------|------|
| `doc_id` | 所属文档 ID（文件名 + 内容短哈希，稳定可追踪） |
| `doc_title` | 文档标题 |
| `section_path` | 章节层级路径（标题文本列表，如 `["第二条 交付与付款","2.1 交付时间"]`） |
| `chapter` / `section` | 便捷字段：顶级章节 / 当前小节 |
| `page_no` | 起始页码（1 基）；无分页的来源（DOCX/TXT）为 0 |
| `bbox` | 页内坐标 `[x0,y0,x1,y1]`（Docling 输出单位，原点左上角） |
| `charspan` | 切片在拼接全文中的全局字符偏移 `[start,end]` |
| `label` | 内容类型：paragraph/list_item/table/… |
| `source_format` | txt / pdf / docx |

## 数据库表

- `documents(doc_id PK, file_path, title, source_format, created_at)`
- `chunks(id PK, doc_id FK, chunk_index, text, metadata JSONB, section_path TEXT[], page_no, charspan INT[], embedding vector(1024))`

切片向量存于 `chunks.embedding`，并建有 HNSW（cosine）索引供后续检索模块使用。

## 说明与限制（本阶段）

- **数字版 PDF 默认关闭 OCR**（`PDF_DO_OCR=0`）以提速；扫描件请在 `.env` 设为 `1`。
- DOCX/TXT 无分页信息，因此 `page_no=0`、`bbox` 为空；PDF 才带页码与坐标。
- JSON 导出不含 embedding 与内部结构块（保持体积可控），数据库存全量。
- 本阶段不含向量检索、多轮分析、结论生成——它们属于后续独立模块。
