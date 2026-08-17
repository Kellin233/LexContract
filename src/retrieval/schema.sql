-- LexContract / retrieval 模块：在 document 模块已有结构上做增量演进
-- 需数据库已安装 pg_search (ParadeDB) 扩展；vector 扩展与 HNSW 索引已由 document 模块建好。
-- 语句全部幂等（IF NOT EXISTS），可安全重复执行。

CREATE EXTENSION IF NOT EXISTS pg_search;

-- 工作区/会话作用域：一个文档归属一个会话，'' 表示未分配（不参与检索）
ALTER TABLE documents ADD COLUMN IF NOT EXISTS session_id TEXT NOT NULL DEFAULT '';

-- BM25 检索字段：入库/回填时用 jieba 分词后的空格分隔 token 串
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_tokens TEXT NOT NULL DEFAULT '';

-- pg_search BM25 索引（key_field 为 chunks 主键）
CREATE INDEX IF NOT EXISTS idx_chunks_bm25
    ON chunks USING bm25 (id, search_tokens) WITH (key_field = 'id');

-- 索引开关（供 init-db 后确认）
