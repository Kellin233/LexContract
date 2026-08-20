-- LexTrace / document 模块：PostgreSQL + pgvector 表结构
-- 先启用扩展（需数据库已安装 pgvector 插件）
CREATE EXTENSION IF NOT EXISTS vector;

-- embedding 维度默认 1024（BAAI/bge-m3）；如换模型请同步修改 vector(<dim>) 并重建列
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    file_path    TEXT NOT NULL,
    title        TEXT NOT NULL DEFAULT '',
    source_format TEXT NOT NULL DEFAULT '',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL DEFAULT 0,
    text         TEXT NOT NULL,
    -- 位置/来源元数据（JSON）
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    section_path TEXT[] NOT NULL DEFAULT '{}',
    page_no      INT NOT NULL DEFAULT 0,
    charspan     INT[] NOT NULL DEFAULT '{}',
    embedding    vector(1024),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 后续检索阶段使用的索引（本阶段可建可不建）
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);

-- 命名迁移（幂等）：旧列 heading_path 统一为 section_path。
-- 仅当旧列存在且新列不存在时执行，已迁移/全新库不受影响。
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'chunks' AND column_name = 'heading_path')
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name = 'chunks' AND column_name = 'section_path') THEN
        ALTER TABLE chunks RENAME COLUMN heading_path TO section_path;
    END IF;
END $$;

-- 证据链阶段新增列（幂等）：documents.full_text 保存拼接全文，
-- 供 CitationVerifier 的 quote == 原文[start:end] 精确校验与条款级工具拼装原文。
ALTER TABLE documents ADD COLUMN IF NOT EXISTS full_text TEXT NOT NULL DEFAULT '';
