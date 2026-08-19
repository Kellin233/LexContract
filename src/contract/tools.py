"""DocumentToolkit：Searcher 的检索工具面。

包装 src/retrieval 的 PostgresRetriever（hybrid search / grep / get_chunk），
以及条款级工具（get_section / get_document_outline），
供 Worker 通过 snippet 定位后恢复完整原文条款。

所有方法返回 JSON 可序列化的 dict/list，方便直接作为 tool message 内容。
"""
from __future__ import annotations

import asyncio
import re
import threading
from typing import Any

from src.retrieval import config as _retrieval_config


def _head_snippet(text: str, limit: int | None = None) -> str:
    """取文本开头不超过 limit（默认 SNIPPET_CHARS）字符的片段（search 用）。"""
    limit = _retrieval_config.SNIPPET_CHARS if limit is None else limit
    return text[:max(0, limit)]


def _around_snippet(text: str, start: int, end: int, limit: int | None = None) -> str:
    """以命中区间 [start, end) 为中心截取长度不超过 limit 的窗口（grep 用）。"""
    limit = _retrieval_config.SNIPPET_CHARS if limit is None else limit
    if limit <= 0:
        return ""
    if end - start >= limit:
        return text[start : start + limit]
    half = (limit - (end - start)) // 2
    lo = max(0, start - half)
    return text[lo : lo + limit]


def _split_chunk_id(chunk_id: str) -> tuple[str, int]:
    """把切片 ID（{doc_id}:{index}）还原为 (doc_id, index)。

    doc_id 本身可能含冒号（如 "maud:VEREIT_...txt"），必须从右侧切分。
    """
    doc_id, _, idx = chunk_id.rpartition(":")
    return doc_id, int(idx)


def _search_row_to_item(r) -> dict:
    """把 RetrievedChunk 裁剪为 search 工具输出：snippet + 元数据 + rrf/rerank 得分。"""
    data = r.model_dump(mode="json")
    return {
        "id": data["id"],
        "doc_id": data["doc_id"],
        "doc_title": data["doc_title"],
        "session_id": data["session_id"],
        "page_no": data["page_no"],
        "section_path": data["section_path"],
        "charspan": data["charspan"],
        "source_format": data["source_format"],
        "snippet": _head_snippet(data["text"]),
        "rrf_score": data.get("rrf_score"),
        "rerank_score": data.get("rerank_score"),
    }


class DocumentToolkit:
    """对检索与条款级查询的统一访问对象（session 作用域 + 可选 doc 过滤）。"""

    def __init__(self, session_id: str = "", doc_ids: list[str] | None = None):
        from src.retrieval.postgres import PostgresRetriever

        self.retriever = PostgresRetriever()
        self.session_id = (session_id or "").strip()
        self._doc_ids = list(doc_ids) if doc_ids else []
        self._lock = threading.Lock()  # 保护 embedding 并发（bge-m3 首次加载）

    def set_scope(self, session_id: str, doc_ids: list[str] | None = None) -> None:
        """按运行期绑定会话与文档过滤（供对象池复用 Agent 时重置作用域）。"""
        self.session_id = (session_id or "").strip()
        self._doc_ids = list(doc_ids) if doc_ids else []

    def _effective_doc_ids(self, doc_ids: list[str] | None = None) -> list[str] | None:
        """合并调用方文档过滤与运行期白名单，防止调用方绕过 ``_doc_ids``。"""
        requested = list(doc_ids) if doc_ids else []
        if self._doc_ids:
            allowed = set(self._doc_ids)
            invalid = [doc_id for doc_id in requested if doc_id not in allowed]
            if invalid:
                raise ValueError(
                    f"doc_ids outside the configured scope: {', '.join(invalid)}"
                )
            return list(self._doc_ids) if not requested else requested
        return requested or None

    def _assert_scope(self, doc_id: str) -> None:
        """确认文档属于当前 session 且满足运行期文档白名单。"""
        doc_id = (doc_id or "").strip()
        if not self.session_id:
            raise ValueError("refusing an unscoped document lookup (session_id is empty)")
        if not doc_id:
            raise ValueError("doc_id is required for a scoped document lookup")
        if self._doc_ids and doc_id not in self._doc_ids:
            raise ValueError(f"document {doc_id!r} is outside the configured doc_ids scope")

        from src.retrieval.store import connect

        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM documents WHERE doc_id = %s AND session_id = %s",
                (doc_id, self.session_id),
            )
            if cur.fetchone() is None:
                raise ValueError(
                    f"document {doc_id!r} is not available in session {self.session_id!r}"
                )

    # ------------------------------------------------------------------
    # 基础能力
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 10,
               doc_ids: list[str] | None = None) -> list[dict]:
        """混合检索（向量 + BM25 RRF + 重排），返回候选切片（snippet + 元数据 + 得分）。

        doc_ids 非空时把检索收敛到指定文档集合；空列表/None 表示整个会话语料
        （底层对空列表会生成 AND false，这里统一转 None）。
        snippet 取切片头部窗口，长度由后端配置 SNIPPET_CHARS 控制，不对 Agent 暴露。
        """
        limit = max(1, min(top_k, 20))
        scoped = self._effective_doc_ids(doc_ids)
        with self._lock:
            rows = self.retriever.retrieve(
                query,
                mode="hybrid",
                session_id=self.session_id,
                limit=limit,
                doc_ids=scoped,
            )
        return [_search_row_to_item(r) for r in rows]

    def grep(self, pattern: str, mode: str = "literal", top_k: int = 10,
             case_sensitive: bool = False, doc_id: str | None = None) -> list[dict]:
        """在切片原文里做精确字符匹配（literal 字面 / regex 正则），返回命中切片。

        与 search 系列不同：grep 只按原文是否包含 pattern 过滤，不做打分排序，
        按文档内顺序返回，并附 match_count（该切片内命中次数）与 snippet
        （首个命中位置前后各约 SNIPPET_CHARS/2 字符窗口）。

        doc_id 非空时把匹配收敛到单篇文档（多文档语料里先定位文档再精确确认条款）。
        """
        if not self.session_id:
            raise ValueError("refusing an unscoped grep search (session_id is empty)")
        if not pattern:
            return []
        from src.retrieval.store import connect

        params: list = [self.session_id]
        scope_sql = ""
        if doc_id:
            self._assert_scope(doc_id)
            scope_sql = " AND c.doc_id = %s"
            params.append(doc_id)
        elif self._doc_ids:
            scope_sql = " AND c.doc_id = ANY(%s)"
            params.append(list(self._doc_ids))

        if mode == "regex":
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                rx = re.compile(pattern, flags)
            except re.error as e:
                raise ValueError(f"invalid grep regex {pattern!r}: {e}") from e
            operator = "~" if case_sensitive else "~*"  # POSIX regex（PostgreSQL ~）
            match_sql = f"c.text {operator} %s"
            params.append(pattern)
        else:  # literal：转义 LIKE 通配符后，把包裹好的 '%...%' 当单参数传
            esc = pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            operator = "LIKE" if case_sensitive else "ILIKE"
            match_sql = f"c.text {operator} %s"
            params.append(f"%{esc}%")

        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT c.id, c.text, c.doc_id, d.title, c.section_path, c.page_no,
                       c.charspan, d.source_format, c.chunk_index
                FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                WHERE d.session_id = %s{scope_sql} AND {match_sql}
                ORDER BY c.doc_id, c.chunk_index
                LIMIT %s
                """,
                (*params, max(top_k, 1)),
            )
            rows = cur.fetchall()

        out = []
        for r in rows:
            text = r[1]
            if mode == "regex":
                m = rx.search(text)
                n = len(rx.findall(text))
                span = (m.start(), m.end()) if m else None
            else:  # 与 SQL 过滤口径一致：case_sensitive=False 时计数也不区分大小写
                if case_sensitive:
                    i = text.find(pattern)
                    n = text.count(pattern)
                else:
                    low = text.lower()
                    p = pattern.lower()
                    i = low.find(p)
                    n = low.count(p)
                span = (i, i + len(pattern)) if i >= 0 else None
            snippet = _around_snippet(text, span[0], span[1]) if span else _head_snippet(text)
            out.append({
                "id": r[0], "snippet": snippet, "doc_id": r[2], "doc_title": r[3],
                "section_path": list(r[4] or []), "page_no": r[5] or 0,
                "charspan": list(r[6] or []), "source_format": r[7] or "",
                "match_count": n, "mode": mode,
            })
        return out

    def get_chunk(self, chunk_id: str) -> dict | None:
        """按切片 ID 取切片原文与元数据。"""
        doc_id, _, chunk_index = str(chunk_id).rpartition(":")
        if not doc_id or not chunk_index.isdigit():
            raise ValueError(f"invalid chunk_id: {chunk_id!r}")
        self._assert_scope(doc_id)
        from src.retrieval.store import connect

        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.text, c.doc_id, d.title, c.section_path, c.page_no, c.charspan, d.source_format
                FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.id = %s AND d.session_id = %s
                """,
                (chunk_id, self.session_id),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "text": row[1], "doc_id": row[2], "doc_title": row[3],
            "section_path": list(row[4] or []), "page_no": row[5] or 0,
            "charspan": list(row[6] or []), "source_format": row[7] or "",
        }

    # ------------------------------------------------------------------
    # 条款级能力
    # ------------------------------------------------------------------
    def get_full_text(self, doc_id: str) -> str:
        self._assert_scope(doc_id)
        from src.retrieval.store import connect

        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT d.full_text FROM documents d WHERE d.doc_id = %s AND d.session_id = %s",
                (doc_id, self.session_id),
            )
            row = cur.fetchone()
        return row[0] if row and row[0] else ""

    def get_document(self, doc_id: str) -> dict | None:
        """文档基本信息（标题/来源格式）。"""
        self._assert_scope(doc_id)
        from src.retrieval.store import connect

        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT d.doc_id, d.title, d.source_format FROM documents d "
                "WHERE d.doc_id = %s AND d.session_id = %s",
                (doc_id, self.session_id),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {"doc_id": row[0], "title": row[1] or "", "source_format": row[2] or ""}

    def list_documents(self) -> list[dict]:
        """返回当前会话内全部文档的元数据（doc_id/title/source_format），不含正文。"""
        if not self.session_id:
            raise ValueError("refusing an unscoped list_documents (session_id is empty)")
        from src.retrieval.store import connect

        with connect() as conn, conn.cursor() as cur:
            doc_filter = ""
            params: list[Any] = [self.session_id]
            if self._doc_ids:
                doc_filter = " AND d.doc_id = ANY(%s)"
                params.append(list(self._doc_ids))
            cur.execute(
                f"""
                SELECT d.doc_id, d.title, d.source_format
                FROM documents d
                WHERE d.session_id = %s{doc_filter}
                ORDER BY d.doc_id
                """,
                params,
            )
            return [
                {"doc_id": r[0], "title": r[1] or "", "source_format": r[2] or ""}
                for r in cur.fetchall()
            ]

    def get_page_for_span(self, doc_id: str, start: int, end: int) -> int:
        """返回覆盖 [start, end] 区间切片的起始页码（找不到则 0）。"""
        self._assert_scope(doc_id)
        from src.retrieval.store import connect

        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.page_no FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.doc_id = %s AND d.session_id = %s
                  AND c.charspan[1] <= %s AND c.charspan[2] >= %s
                ORDER BY charspan[1] LIMIT 1
                """,
                (doc_id, self.session_id, start, end),
            )
            row = cur.fetchone()
        return row[0] if row and row[0] else 0

    def _section_chunks(self, doc_id: str, section_path: list[str]) -> list[dict]:
        """取某章节下所有切片（精确匹配；无则前缀匹配），按起始偏移排序。"""
        self._assert_scope(doc_id)
        from src.retrieval.store import connect

        with connect() as conn, conn.cursor() as cur:
            # 精确匹配
            cur.execute(
                """
                SELECT c.id, c.text, c.section_path, c.page_no, c.charspan
                FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.doc_id = %s AND d.session_id = %s AND c.section_path = %s
                ORDER BY (c.charspan[1] IS NOT NULL) DESC, c.charspan[1], c.id
                """,
                (doc_id, self.session_id, list(section_path)),
            )
            rows = cur.fetchall()
            if not rows and section_path:
                # 前缀匹配：给定路径是某切片章节路径的前缀（section_path[1:n] = prefix）
                prefix = list(section_path)
                cur.execute(
                    """
                    SELECT c.id, c.text, c.section_path, c.page_no, c.charspan
                    FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                    WHERE c.doc_id = %s AND d.session_id = %s
                      AND c.section_path[1:%s] = %s
                    ORDER BY (c.charspan[1] IS NOT NULL) DESC, c.charspan[1], c.id
                    """,
                    (doc_id, self.session_id, len(prefix), prefix),
                )
                rows = cur.fetchall()
        return [
            {
                "id": r[0], "text": r[1], "section_path": list(r[2] or []),
                "page_no": r[3] or 0, "charspan": list(r[4] or []),
            }
            for r in rows
        ]

    def get_section(self, doc_id: str, section_path: list[str]) -> dict | None:
        """返回某章节的完整连续原文（从 full_text 截取 [min_start, max_end]）。

        保证 quote 属于原始连续文本；无 full_text 时降级为切片文本拼接。
        """
        chunks = self._section_chunks(doc_id, list(section_path or []))
        if not chunks:
            return None
        starts = [c["charspan"][0] for c in chunks if len(c["charspan"]) == 2 and c["charspan"][0] is not None]
        ends = [c["charspan"][1] for c in chunks if len(c["charspan"]) == 2 and c["charspan"][1] is not None]
        full_text = self.get_full_text(doc_id)
        if starts and ends and full_text:
            s, e = min(starts), max(ends)
            if 0 <= s <= e <= len(full_text):
                return {
                    "doc_id": doc_id,
                    "section_path": chunks[0]["section_path"],
                    "start_offset": s, "end_offset": e,
                    "text": full_text[s:e],
                    "chunk_ids": [c["id"] for c in chunks],
                }
        return {
            "doc_id": doc_id,
            "section_path": chunks[0]["section_path"],
            "start_offset": 0, "end_offset": 0,
            "text": "\n".join(c["text"] for c in chunks),
            "chunk_ids": [c["id"] for c in chunks],
        }

    def get_document_outline(self, doc_id: str) -> list[dict]:
        """文档目录：各章节路径 + 起始/结束偏移（按阅读顺序）。"""
        self._assert_scope(doc_id)
        from src.retrieval.store import connect

        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.section_path,
                       MIN(c.charspan[1]) AS start_char,
                       MAX(c.charspan[2]) AS end_char
                FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.doc_id = %s AND d.session_id = %s
                  AND array_length(c.section_path, 1) > 0
                GROUP BY c.section_path
                ORDER BY start_char NULLS LAST
                """,
                (doc_id, self.session_id),
            )
            rows = cur.fetchall()
        return [
            {
                "doc_id": doc_id,
                "section_path": list(r[0] or []),
                "start_offset": r[1] if r[1] is not None else 0,
                "end_offset": r[2] if r[2] is not None else 0,
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # 工具适配面（供 Searcher 使用）
    # ------------------------------------------------------------------
    def get_tools(self) -> list["DocumentTool"]:
        """构建 OpenAI function-calling 工具对象列表。"""
        return [
            DocumentTool(
                name="list_documents",
                description=(
                    "Lists the documents available in the current session "
                    "(doc_id, title, source format), without full text. "
                    "Call this first to discover which documents exist; doc_ids must be copied verbatim."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                handler=self.list_documents,
            ),
            DocumentTool(
                name="search",
                description=(
                    "Hybrid semantic search (vector + BM25, reranked): returns short snippets with "
                    "metadata and score. Use for meaning-based clause search; a clause may be named by "
                    "a canonical label that does not appear verbatim in the contract, so search with "
                    "synonyms and meanings. Pass doc_ids to lock the search to specific documents."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The semantic search query (synonyms are allowed)"},
                        "top_k": {"type": "integer", "description": "Number of candidates to return (max 20)"},
                        "doc_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional list of doc_ids to lock the search to specific documents (copy verbatim from tool results); omit to search the whole corpus"},
                    },
                    "required": ["query"],
                },
                handler=self.search,
            ),
            DocumentTool(
                name="get_chunk",
                description=(
                    "Reads the full original text and metadata of a single chunk by its ID. "
                    "Use to expand a located hit into full text."
                ),
                parameters={
                    "type": "object",
                    "properties": {"chunk_id": {"type": "string", "description": "Chunk ID, e.g. doc1:3"}},
                    "required": ["chunk_id"],
                },
                handler=self.get_chunk,
            ),
            DocumentTool(
                name="get_section",
                description=(
                    "Returns the complete continuous original text of a section "
                    "(e.g. ['Article 12', '12.3']) with offsets and chunk ids. "
                    "Use to capture the FULL clause after locating it via search/grep."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string", "description": "Document ID (copy verbatim from search/grep/list_documents results)"},
                        "section_path": {"type": "array", "items": {"type": "string"}, "description": "Section path (list of heading texts)"},
                    },
                    "required": ["doc_id", "section_path"],
                },
                handler=self.get_section,
            ),
            DocumentTool(
                name="get_document_outline",
                description=(
                    "Returns the document outline (section paths and offsets), to navigate clauses by "
                    "heading before reading a full section."
                ),
                parameters={
                    "type": "object",
                    "properties": {"doc_id": {"type": "string"}},
                    "required": ["doc_id"],
                },
                handler=self.get_document_outline,
            ),
            DocumentTool(
                name="grep",
                description=(
                    "Exact match search over the ORIGINAL clause text: returns only chunks that literally "
                    "contain a phrase or a regular expression. Use to confirm exact wording or locate "
                    "a provision by precise terms / an article number when semantic search is too loose "
                    "(e.g. pattern='exceptions', or regex pattern='第[0-9]+条'). "
                    "mode='literal' (default) matches the plain substring; mode='regex' compiles the pattern "
                    "with Python re (supports constructs like lookahead). Returns a short snippet around the "
                    "first hit plus match_count per chunk."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Literal text or regular expression to find"},
                        "mode": {"type": "string", "enum": ["literal", "regex"], "description": "literal substring (default) or regex"},
                        "case_sensitive": {"type": "boolean", "description": "Case-sensitive matching (default false)"},
                        "top_k": {"type": "integer", "description": "Max number of matching chunks to return (max 20)"},
                        "doc_id": {"type": "string", "description": "Optional: lock the grep to one document (copy verbatim from tool results); omit to search the whole corpus"},
                    },
                    "required": ["pattern"],
                },
                handler=self.grep,
            ),
        ]


class DocumentTool:
    """与 src/tools/* 一致的轻量工具协议：name/description/get_openai_tool_schema/execute。"""

    def __init__(self, name: str, description: str, parameters: dict[str, Any], handler):
        self.name = name
        self.description = description
        self.parameters = parameters
        self._handler = handler

    def get_openai_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute(self, **kwargs) -> Any:
        return await asyncio.to_thread(self._handler, **kwargs)
