"""DocumentToolkit：Searcher 的检索工具面。

包装 src/retrieval 的 PostgresRetriever（search/get_chunk/get_context），
并新增条款级工具（get_section / get_document_outline / get_referenced_section），
供 Worker 把碎片 Chunk 恢复成完整原文条款，以及跟随“除第X条外”这类交叉引用。

所有方法返回 JSON 可序列化的 dict/list，方便直接作为 tool message 内容。
"""
from __future__ import annotations

import asyncio
import re
import threading
from typing import Any


# ---------- 引用解析启发式 ----------
# “第14条” / “第 14.3 条” / “14.3” / “Article 14” / “Section 4.2”
_CH_ART_RE = re.compile(r"第\s*([0-9０-９]+|[一二三四五六七八九十百千万]+)\s*条\s*(?:[\.．]?\s*(\d+(?:\.\d+)*))?")
_CH_SEC_RE = re.compile(r"^第\s*([0-9０-９]+|[一二三四五六七八九十百千万]+)\s*(?:章|节)")
_EN_RE = re.compile(r"(?:Article|Section|Clause)\s+(\d+(?:\.\d+)*)", re.IGNORECASE)
_ARABIC_RE = re.compile(r"^(\d+(?:\.\d+)*)")

_CJK_DIGITS = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cjk_to_int(s: str) -> int | None:
    total = 0
    section = 0
    digit = 0
    for ch in s:
        if ch in "零〇":
            digit = 0
        elif ch in _CJK_DIGITS:
            digit = _CJK_DIGITS[ch]
        elif ch == "十":
            section = (section or 1) * 10 if digit == 0 else (section + digit) * 10
            digit = 0
        elif ch == "百":
            section += (digit or 1) * 100
            digit = 0
        elif ch == "千":
            section += (digit or 1) * 1000
            digit = 0
    total = section + digit
    return total if total else None


def _normalize_ref_number(ref: str) -> str | None:
    """把“第14条 / 第 十四 条 / Article 14 / 4.2”归一化为数字串（最长前缀主编号 + 子编号）。"""
    ref = ref.strip()
    m = _CH_ART_RE.search(ref)
    if m:
        if m.group(1).isdigit() or set(m.group(1)) <= set("０１２３４５６７８９"):
            main = str(int(m.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789"))))
        else:
            main = str(_cjk_to_int(m.group(1)) or 0)
        sub = m.group(2) or ""
        return f"{main}.{sub}" if sub else main
    m = _EN_RE.search(ref)
    if m:
        return m.group(1)
    pure = ref.replace("第", "").replace("条", "").replace(" ", "").strip()
    m = _ARABIC_RE.match(pure)
    if m:
        return m.group(1)
    return None


def _label_number(label: str) -> str | None:
    """提取章节标签（如 “12.3 违约责任” / “第14条 不可抗力”）的前导编号串。"""
    m = _CH_ART_RE.search(label)
    if m:
        if m.group(1).isdigit() or set(m.group(1)) <= set("０１２３４５６７８９"):
            main = str(int(m.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789"))))
        else:
            main = str(_cjk_to_int(m.group(1)) or 0)
        sub = m.group(2) or ""
        return f"{main}.{sub}" if sub else main
    m = _ARABIC_RE.match(label.strip())
    if m:
        return m.group(1)
    return None


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

    # ------------------------------------------------------------------
    # 基础能力
    # ------------------------------------------------------------------
    def search(self, query: str, mode: str = "hybrid", top_k: int = 20,
               doc_id: str | None = None) -> list[dict]:
        """混合/向量/BM25 检索，返回候选 Chunk（含元数据与得分）。

        doc_id 非空时把检索收敛到单篇文档（多文档语料里先定位文档再查条款）。
        """
        with self._lock:
            rows = self.retriever.retrieve(
                query,
                mode=mode,
                session_id=self.session_id,
                limit=max(top_k, 1),
                doc_ids=[doc_id] if doc_id else (self._doc_ids or None),
            )
        return [r.model_dump(mode="json") for r in rows]

    def search_vector(self, query: str, top_k: int = 20, doc_id: str | None = None) -> list[dict]:
        """向量语义检索（bge-m3 相似度），擅长整句含义 / 同义表述。"""
        return self.search(query, mode="vector", top_k=top_k, doc_id=doc_id)

    def search_bm25(self, query: str, top_k: int = 20, doc_id: str | None = None) -> list[dict]:
        """BM25 关键词排名检索，擅长命名词面关键词。"""
        return self.search(query, mode="bm25", top_k=top_k, doc_id=doc_id)

    def search_hybrid(self, query: str, top_k: int = 20, doc_id: str | None = None) -> list[dict]:
        """混合检索（向量 + BM25 RRF 融合），不确定用哪种时的稳妥默认。"""
        return self.search(query, mode="hybrid", top_k=top_k, doc_id=doc_id)

    def grep(self, pattern: str, mode: str = "literal", top_k: int = 20,
             case_sensitive: bool = False, doc_id: str | None = None) -> list[dict]:
        """在切片原文里做精确字符匹配（literal 字面 / regex POSIX 正则），返回命中切片。

        与 search 系列不同：grep 只按原文是否包含 pattern 过滤，不做打分排序，
        按文档内顺序返回，并附 match_count（该切片内命中次数）。

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
                n = len(rx.findall(text))
            else:  # 与 SQL 过滤口径一致：case_sensitive=False 时计数也不区分大小写
                n = text.count(pattern) if case_sensitive else text.lower().count(pattern.lower())
            out.append({
                "id": r[0], "text": text, "doc_id": r[2], "doc_title": r[3],
                "section_path": list(r[4] or []), "page_no": r[5] or 0,
                "charspan": list(r[6] or []), "source_format": r[7] or "",
                "match_count": n, "mode": mode,
            })
        return out

    def get_chunk(self, chunk_id: str) -> dict | None:
        """按切片 ID 取切片原文与元数据。"""
        from src.retrieval.store import connect

        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.text, c.doc_id, d.title, c.section_path, c.page_no, c.charspan, d.source_format
                FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.id = %s
                """,
                (chunk_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "text": row[1], "doc_id": row[2], "doc_title": row[3],
            "section_path": list(row[4] or []), "page_no": row[5] or 0,
            "charspan": list(row[6] or []), "source_format": row[7] or "",
        }

    def get_context(self, chunk_id: str, before: int = 2, after: int = 2) -> list[dict]:
        """同文档内按 chunk_index 取前后邻居切片（保持阅读顺序）。"""
        from src.retrieval.store import connect

        base = self.get_chunk(chunk_id)
        if base is None:
            return []
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_index FROM chunks WHERE id = %s", (chunk_id,)
            )
            row = cur.fetchone()
            if row is None:
                return []
            idx = row[0]
            lo, hi = max(0, idx - max(before, 0)), idx + max(after, 0)
            cur.execute(
                """
                SELECT c.id, c.text, c.doc_id, d.title, c.section_path, c.page_no, c.charspan, d.source_format
                FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.doc_id = %s AND c.chunk_index BETWEEN %s AND %s
                ORDER BY c.chunk_index
                """,
                (base["doc_id"], lo, hi),
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0], "text": r[1], "doc_id": r[2], "doc_title": r[3],
                "section_path": list(r[4] or []), "page_no": r[5] or 0,
                "charspan": list(r[6] or []), "source_format": r[7] or "",
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # 条款级能力
    # ------------------------------------------------------------------
    def get_full_text(self, doc_id: str) -> str:
        from src.retrieval.store import connect

        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT full_text FROM documents WHERE doc_id = %s", (doc_id,))
            row = cur.fetchone()
        return row[0] if row and row[0] else ""

    def get_document(self, doc_id: str) -> dict | None:
        """文档基本信息（标题/来源格式）。"""
        from src.retrieval.store import connect

        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id, title, source_format FROM documents WHERE doc_id = %s",
                (doc_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {"doc_id": row[0], "title": row[1] or "", "source_format": row[2] or ""}

    def get_page_for_span(self, doc_id: str, start: int, end: int) -> int:
        """返回覆盖 [start, end] 区间切片的起始页码（找不到则 0）。"""
        from src.retrieval.store import connect

        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT page_no FROM chunks
                WHERE doc_id = %s AND charspan[1] <= %s AND charspan[2] >= %s
                ORDER BY charspan[1] LIMIT 1
                """,
                (doc_id, start, end),
            )
            row = cur.fetchone()
        return row[0] if row and row[0] else 0

    def _section_chunks(self, doc_id: str, section_path: list[str]) -> list[dict]:
        """取某章节下所有切片（精确匹配；无则前缀匹配），按起始偏移排序。"""
        from src.retrieval.store import connect

        with connect() as conn, conn.cursor() as cur:
            # 精确匹配
            cur.execute(
                """
                SELECT id, text, section_path, page_no, charspan
                FROM chunks WHERE doc_id = %s AND section_path = %s
                ORDER BY (charspan[1] IS NOT NULL) DESC, charspan[1], id
                """,
                (doc_id, list(section_path)),
            )
            rows = cur.fetchall()
            if not rows and section_path:
                # 前缀匹配：给定路径是某切片章节路径的前缀（section_path[1:n] = prefix）
                prefix = list(section_path)
                cur.execute(
                    """
                    SELECT id, text, section_path, page_no, charspan
                    FROM chunks WHERE doc_id = %s AND section_path[1:%s] = %s
                    ORDER BY (charspan[1] IS NOT NULL) DESC, charspan[1], id
                    """,
                    (doc_id, len(prefix), prefix),
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
        from src.retrieval.store import connect

        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT section_path,
                       MIN(charspan[1]) AS start_char,
                       MAX(charspan[2]) AS end_char
                FROM chunks
                WHERE doc_id = %s AND array_length(section_path, 1) > 0
                GROUP BY section_path
                ORDER BY start_char NULLS LAST
                """,
                (doc_id,),
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

    def get_referenced_section(self, doc_id: str, ref: str) -> dict | None:
        """跟随条款间引用（“除第14条外”/“Article 4.2”）返回对应章节原文。"""
        ref_num = _normalize_ref_number(ref)
        if ref_num is None:
            return None
        outline = self.get_document_outline(doc_id)
        # 精确：章节标签前导编号 == 引用编号
        for o in outline:
            label = o["section_path"][-1] if o["section_path"] else ""
            num = _label_number(label)
            if num == ref_num:
                return self.get_section(doc_id, o["section_path"])
        # 宽松：引用主编号是某章节编号（或其父编号）的前缀
        for o in outline:
            label = o["section_path"][-1] if o["section_path"] else ""
            num = _label_number(label)
            if num and (num.startswith(ref_num) or ref_num.startswith(num)):
                return self.get_section(doc_id, o["section_path"])
        return None

    # ------------------------------------------------------------------
    # 工具适配面（供 Searcher 使用）
    # ------------------------------------------------------------------
    def get_tools(self) -> list["DocumentTool"]:
        """构建 OpenAI function-calling 工具对象列表。"""
        return [
            DocumentTool(
                name="search_vector",
                description=(
                    "Semantic search (vector): returns passages similar in MEANING, good for whole-clause "
                    "meaning and paraphrases. Use when you need semantically relevant clauses."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query (synonyms are allowed)"},
                        "top_k": {"type": "integer", "description": "Number of candidates to return"},
                        "doc_id": {"type": "string", "description": "Lock the search to one document (its doc_id appears in search results). Use after you have identified the target document in a multi-document corpus."},
                    },
                    "required": ["query"],
                },
                handler=self.search_vector,
            ),
            DocumentTool(
                name="search_bm25",
                description=(
                    "Keyword search (BM25): returns passages ranked by keyword hits. Use when the question "
                    "names specific terms and you want passages near those exact tokens."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query (keywords)"},
                        "top_k": {"type": "integer", "description": "Number of candidates to return"},
                        "doc_id": {"type": "string", "description": "Lock the search to one document (its doc_id appears in search results). Use after you have identified the target document in a multi-document corpus."},
                    },
                    "required": ["query"],
                },
                handler=self.search_bm25,
            ),
            DocumentTool(
                name="search_hybrid",
                description=(
                    "Hybrid search (vector + BM25, fused): balanced default when unsure which mode fits. "
                    "Complements search_vector / search_bm25."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "top_k": {"type": "integer", "description": "Number of candidates to return"},
                        "doc_id": {"type": "string", "description": "Lock the search to one document (its doc_id appears in search results). Use after you have identified the target document in a multi-document corpus."},
                    },
                    "required": ["query"],
                },
                handler=self.search_hybrid,
            ),
            DocumentTool(
                name="get_chunk",
                description="Reads the original text and metadata of a single chunk by its ID.",
                parameters={
                    "type": "object",
                    "properties": {"chunk_id": {"type": "string", "description": "Chunk ID, e.g. doc1:3"}},
                    "required": ["chunk_id"],
                },
                handler=self.get_chunk,
            ),
            DocumentTool(
                name="get_context",
                description="Reads the chunks adjacent to a given chunk, to expand a truncated passage into a complete clause.",
                parameters={
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string"},
                        "before": {"type": "integer", "description": "How many chunks to include before"},
                        "after": {"type": "integer", "description": "How many chunks to include after"},
                    },
                    "required": ["chunk_id"],
                },
                handler=self.get_context,
            ),
            DocumentTool(
                name="get_section",
                description="Returns the complete continuous original text of a section (e.g. ['Article 12', '12.3']).",
                parameters={
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string", "description": "Document ID (shown in search results)"},
                        "section_path": {"type": "array", "items": {"type": "string"}, "description": "Section path (list of heading texts)"},
                    },
                    "required": ["doc_id", "section_path"],
                },
                handler=self.get_section,
            ),
            DocumentTool(
                name="get_document_outline",
                description="Returns the document outline (section paths and offsets), to locate where a clause is.",
                parameters={
                    "type": "object",
                    "properties": {"doc_id": {"type": "string"}},
                    "required": ["doc_id"],
                },
                handler=self.get_document_outline,
            ),
            DocumentTool(
                name="get_referenced_section",
                description="Follows a cross-reference between clauses and returns the full original text of the referenced section "
                            '(e.g. pass "Article 14" for "except as provided in Article 14").',
                parameters={
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "ref": {"type": "string", "description": 'Reference text, e.g. "Article 14" / "Article 4.2"'},
                    },
                    "required": ["doc_id", "ref"],
                },
                handler=self.get_referenced_section,
            ),
            DocumentTool(
                name="grep",
                description=(
                    "Exact match search over the ORIGINAL clause text: returns only chunks that literally "
                    "contain a phrase or a POSIX regular expression. Use to confirm exact wording or locate "
                    "a provision by precise terms / an article number when semantic search is too loose "
                    "(e.g. pattern='exceptions', or regex pattern='第[0-9]+条'). "
                    "mode='literal' (default) matches the plain substring; mode='regex' compiles the pattern "
                    "as a regular expression (PostgreSQL ARE syntax; Python re is used only to count matches, "
                    "so Python-only constructs like lookahead may still fail in the SQL query)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Literal text or POSIX regular expression to find"},
                        "mode": {"type": "string", "enum": ["literal", "regex"], "description": "literal substring (default) or regex"},
                        "case_sensitive": {"type": "boolean", "description": "Case-sensitive matching (default false)"},
                        "top_k": {"type": "integer", "description": "Max number of matching chunks to return"},
                        "doc_id": {"type": "string", "description": "Lock the grep to one document (its doc_id appears in search results). Use after you have identified the target document."},
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
