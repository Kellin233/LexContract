"""评测包：LegalBenchRAG（RAG 能力） + ContractNLI（端到端分类能力）。

评测先决条件：
- 测试代码/数据（zip/jsonl/benchmark json + corpus）位于本机（PAKTON-develop / LexTrace-main 等），
  由 loaders.py 探测默认路径，也可显式传入。
- LegalBenchRAG 语料以“原样文本”入库（见 ingest_raw.py），保证 gold span 与全文偏移精确对齐。
"""
