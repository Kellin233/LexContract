<div align="center">

# 📜 LexContract — 合同证据链研究系统

*从合同文档到"可验证"的审查结论，全链路自动化*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Async](https://img.shields.io/badge/Async-asyncio-orange.svg)](https://docs.python.org/3/library/asyncio.html)

</div>

---

## 📖 这是什么

LexContract 是一个面向合同 / 法律文档的证据链式研究系统：针对一个问题，多个研究 Agent 只负责从合同库里收集"可引用的原始条款证据"，由唯一的结论 Agent 基于证据生成带引用、可回查、不编造的审查报告。

核心流程不是"检索 → 中间总结 → 再总结 → 最终答案"，而是：

```
问题
 ↓
Planner 拆解"需要调查什么"（只生成调查要点，禁止中间结论）
 ↓
多个 Searcher 并行收集证据（只找完整可引用的原始条款，不输出结论）
 ↓
EvidenceAssembler 把命中切片恢复成连续原文（quote 一律取自 DB full_text，LLM 不得改写）
 ↓
CitationVerifier 纯程序化校验（quote == full_text[start:end]，不依赖 LLM）
 ↓
EvidenceStore 去重入库（按 document_id + start/end 偏移），只按 ID 传递
 ↓
Reviewer 只判三件事：证据是否覆盖问题 / 是否明显冲突 / 还缺什么（不裁判）
 ↓
┌─ SUFFICIENT ───────────────────────────→ Refiner 生成结论（唯一结论 Agent，JSON + Markdown）
│
└─ NEED_MORE 且 未达轮数上限 且有有效新增
        ↓ 增量 Planner（只补缺失要点）→ 重新派发 Searcher → Reviewer
（达上限 / 无有效新增 → PARTIALLY_SUFFICIENT，仍进 Refiner 并如实说明缺口）
```

贯穿全代码的硬约束：

1. **证据 = 原文，永不改写、永不压缩丢失**：`Evidence.quote` 是从数据库 `documents.full_text[start:end]` 截取的**连续原文**，禁止模型改写；
2. **引用只许用证据 ID**：结论正文只用 `[E001]` 标注，由后处理把 E### 映射成《文档》章节 + 页码，杜绝模型自编条款号；
3. **分层职责**：Searcher 只找证据、Reviewer 只评完整性、Refiner 是唯一生成结论的 Agent；中间层一律不产出推理性结论，降低错误在多轮 Agent 间传播的风险；
4. **原文随时可取回**：证据带 `evidence_id + section_path + charspan`，全文在库里，凭 ID 即回查。

---

## 🧠 系统设计

**Agent 编排**

流程由 `src/orchestrator/orchestrator.py` 中的有限状态机驱动：

`IDLE → PLANNING → DISPATCHING → COLLECTING → REVIEWING →（INCREMENTAL_PLANNING → DISPATCHING…）→ REFINING → DONE / FAILED`

- **Planner**（`src/contract/planner.py`）：`initial_plan` 把原问题拆成最多 3 个调查要点（只定义"查什么"，不含任何中间结论）；`incremental_plan` 只针对 Reviewer 上报的缺口补新要点。整个流程最多 3 轮（`contract.max_iterations`）。
- **Searcher**（`src/contract/worker.py`）：每个调查要点对应一个 Searcher 子任务，多轮工具循环收集证据，最终只输出 `WorkerResult`（仅证据）。Searcher 实例由 `AgentPool` 对象池管理：延迟创建、空闲复用、超时 / 异常降级重建、被上下文截断的 policy 直接丢弃。
- **并行派发**：调查要点之间互相独立，按 DAG 分层用 `asyncio.Semaphore` 限流并发（默认 `contract.max_concurrent=3`），单子任务超时 300 秒；任一环节触发全局超时（默认 900 秒）会强制进入 Refiner，用已有证据收尾。
- **Reviewer**（`src/contract/reviewer.py`）：只评估三件事——证据是否覆盖问题、是否存在明显冲突、还缺什么；`effective_new_evidence` 为假或达到轮次上限即收尾。
- **Refiner**（`src/contract/refiner.py`）：唯一的结论 Agent，输出结构化 `RefinerResult`（`conclusion / points / supporting_evidence_ids / citations / evidence_gap / final_status`）并渲染 Markdown；证据不足时以 `PARTIALLY_SUFFICIENT` 状态如实说明缺口，不编造。
- **并发隔离**：评测等场景下每条实例拥有独立的 toolkit / agent_pool / orchestrator（共享 policy / planner / reviewer / refiner），避免 `toolkit.set_scope` 在并发实例间互相覆盖文档作用域。

**上下文管理**

系统**不做语义级文本压缩**（证据必须保留原文），改用分层手段控制上下文：

- **统一 token 口径**（`src/utils/tokens.py`）：`estimate_tokens` 估算文本 token（中文按 0.6 token/字、其余按空白分词 1 token），消息级 `estimate_messages_tokens` 另计每条消息 4 token 固定开销；全项目上下文 / 预算计算统一走该口径。
- **窗口兜底**（`src/models/vllm_policy.py`）：`VLLMPolicy.max_context_tokens` 默认 128K，可被 `model.context_window_tokens` 按后端覆盖（deepseek 128000 / mimo 32000）。超阈值时**丢旧轮次**而非截断内容——保留 system 与最近交互，且不拆开 `assistant(tool_calls)` 与紧随的 `tool` 消息；极端情况才对最新一条做内容级截断并打 `[CONTENT_TRUNCATED]` 标记。被截断的 policy 会被对象池丢弃、不再复用。
- **按需构造输入**：Planner / Reviewer 只喂精简信息（问题、要点、证据 ID 列表），Refiner 是唯一全量读取证据原文的环节，输入预算默认 65536 token（`contract.refiner_input_token_budget`），超预算只告警并记入报告的 `evidence_gap`，暂不裁剪。
- **检索预算**：`search / grep` 两个检索类工具共用同一预算（默认 3 轮 × 每轮 1 个检索调用），预算直接写进 Searcher 系统提示词，改配置即改提示词；`get_*` 展馆工具与最终 JSON 不计入。
- **工具结果去重**：同一 Searcher 多轮内，同一切片 / 同一章节的完整原文只注入一次，重复项正文替换为短标记（保留 id / 偏移 / 得分骨架），可开关做 A/B（`contract.searcher_dedup_tool_results`）。
- **轮间只传 E###**：`EvidenceStore` 按跨度去重后只传证据 ID，要读正文时按 ID 从库里取出，避免全量字节在 Agent 间流转。
- **全链路 token 账本**：用 `contextvars` 为每次运行开启账本，Planner / Searcher / Reviewer / Refiner 每次 LLM 调用的入参估算 token 都计入；评测并发实例各自独立 context，互不干扰。
- **对话留档（可选）**：`conversation.enabled: true` 时，`ConversationRecorder` 把每次 LLM 调用的对话层（system / user / assistant 原文 + tool_calls 意图）逐行写入报告同目录的 jsonl；工具返回的大 JSON 只留摘要，证据全文不入留档文件。

**RAG 设计**

- **数据底座**：PostgreSQL + pgvector（向量检索）；安装 ParadeDB `pg_search` 时启用 BM25，普通 PostgreSQL 自动降级为向量/全文检索。连接信息统一由 `.env`/`.env.local` 的 `PG_*` 配置，两个模块只使用同一个 `PG_PORT`（默认 5432）。
- **入库**（`src/document/`）：docling 解析 txt / pdf / docx → 结构感知切块（标题为边界优先、超长逐级下切到句子、章节内相邻片带 50 token 重叠、跨章节不重叠，常规入库默认 600 token/片，评测语料入库为 500）→ bge-m3 向量化 → 写入 `documents / chunks` 两表；`full_text` 逐字保留原文，`charspan` 为全文全局字符偏移，保证证据可精确回查。
- **三种检索模式**（`src/retrieval/postgres.py`）：`vector`（cosine 相似度）/ `bm25`（pg_search `@@@`）/ `hybrid`（加权 RRF 融合，默认权重 0.5 / 0.5、k=60、候选上限 100）。
- **作用域强制**：`session_id` 必填，无作用域直接拒绝查询；可附加 `doc_ids` 过滤，或在检索时用 `doc_id` 把查询锁定到单篇文档（多文档语料先定位文档再查条款）。
- **工具化检索**（`src/contract/tools.py`）：Searcher 侧暴露 6 个工具——`list_documents`（会话文档元数据）、`search`（hybrid 语义检索，默认融合向量 + BM25 + 重排）、`grep`（字面 / 正则精确匹配），以及展馆工具 `get_chunk / get_section / get_document_outline`（切片与条款级完整原文读取）。`search / grep` 只返回可配置长度的 snippet（`SNIPPET_CHARS`，默认 200 字符），完整原文一律由 `get_chunk / get_section` 获取。
- **重排**：`retrieval` CLI 查询工具支持 BGE cross-encoder 重排（`BAAI/bge-reranker-v2-m3`，默认开启，`--no-rerank` 关闭）；主证据链检索当前直接使用 RRF 融合结果。

**证据处理**

- **结构**（`src/contract/schemas.py`）：`Evidence` 携带 `evidence_id / question_id / document_id / section_path / page_no / source_chunk_ids / start_offset / end_offset / quote / verified` 等字段；`quote` 必须是 DB 原文的连续切片。
- **装配**：Searcher 候选只报 `source_chunk_ids + relevance_note`，`EvidenceAssembler` 按最末级 section 自动聚合完整条款（整章超 `MAX_EVIDENCE_SECTION_TOKENS` 时回退命中切片并集，带零空洞连续性校验），quote 一律从 `documents.full_text[start:end]` 截取，模型无权改写。
- **校验**：`CitationVerifier` 纯程序化逐字符比对 `quote == full_text[start:end]`，并要求切片并集对证据区间覆盖率 ≥98%、两端落在并集内（±1 字符容差），失败即丢弃并计入 `drop_reasons`，不依赖 LLM 判断。（"零空洞"连续性校验在装配器的整章超限回退路径上。）
- **去重与注册**：`EvidenceStore` 按 `(document_id, start, end)` 去重，为每条证据分配 `E###` 运行期 ID。
- **引用**：Refiner 正文只用 `[E###]` 占位，后处理由证据元数据生成 `citations`（《文档》章节 + 页码），杜绝模型自编条款号；`supporting_evidence_ids` 只落最支撑最终结论的证据子集。
- **缺口**：无法确认的项如实写入 `evidence_gap`；`PARTIALLY_SUFFICIENT` 状态仍生成结论，但结论中明确标注缺口。

---

## 🧱 模块结构

| 模块 | 职责 |
|------|------|
| `src/contract/` | 证据链领域核心：Planner / Searcher / Reviewer / Refiner / EvidenceStore / DocumentToolkit / 数据结构 |
| `src/orchestrator/` | 状态机编排器 + AgentPool 对象池（DAG 分层并发、Semaphore、超时） |
| `src/document/` | 文档解析（txt / pdf / docx）、结构感知切块、bge-m3 向量化、入库 |
| `src/retrieval/` | PostgreSQL 检索：vector / BM25 / hybrid（加权 RRF）、会话作用域、CLI 可选重排 |
| `src/models/` | 多后端 LLM 路由（DeepSeek / MiMo / vLLM / OpenAI），上下文窗口与采样参数管理 |
| `src/contract/eval/` | 评测子系统（LegalBenchRAG / ContractNLI，见"评测"） |
| `src/utils/` | `.env` 加载、token 口径、对话留档、LangSmith 追踪 |

> 主链路只依赖上表中的模块。当前合同流程不包含对抗审查阶段；旧 deep-research 的 `adversarial / evolution / compressor / tools`、通用 Researcher / Summarizer、旧 Planner / Judge / 消融模块均已移除；`src/agents/` 仅保留合同 Searcher 依赖的基础 Agent 协议。`src/memory/` 仅用于把最终报告写入 SQLite 存档，不参与检索。

---

## 🚀 快速开始

**环境准备**

```bash
pip install -r requirements.txt
cp .env.template .env
# 需要配置的：DEEPSEEK_API_KEY（主流程）、MIMO_API_KEY（可选）
# PG_*（PostgreSQL）与 EMBED_*（向量模型）可直接在 .env.template 基础上配置；
# 两个模块共用同一个 PG_PORT，ParadeDB 映射到其他端口时只改这一处；
# .env.local 优先级高于 .env 且被 .gitignore 忽略，适合放个人配置
```

**数据底座（PostgreSQL）**

```bash
# 初始化检索 schema（session_id / search_tokens；有 pg_search 时启用 BM25 索引）
python -m src.retrieval.main init-db

# 解析并把合同入库（含全文 + 向量；有 pg_search 时回填 BM25 tokens）
python -m src.document.main parse <合同文件或目录>

# 把文档分派到会话（检索强制要求 session 作用域）
python -m src.retrieval.main assign <doc_id> --session S1
python -m src.retrieval.main sessions        # 查看会话
```

**跑一条合同问题**

```bash
python scripts/run_single.py --query "供应商能否单方面终止合同？" --session S1
python scripts/run_single.py --query "乙方逾期交付货物，需要承担什么责任？" --session S1 --doc doc_a,doc_b
```

输出 `outputs/reports/report_*.md`（可读报告）+ `report_*.json`（结构化 RefinerResult）+ 同目录 `run_*.log`。

**冒烟测试**

```bash
python tests/contract_smoke.py
```

两层：Tier A（离线，真实 DB）验证 `EvidenceAssembler / CitationVerifier / EvidenceStore` —— quote 与原文一致、篡改引用被拒、按跨度去重；Tier B（确定性 stub LLM，不依赖网络）驱动完整 Orchestrator 状态机。DB 不可达时打印 SKIP 正常退出。

---

## 🎛️ 配置与核心参数

全局配置集中在 `configs/default.yaml`（主流程唯一加载的配置）：

| 段 | 关键项 | 说明 |
|----|--------|------|
| `model` | `backend` / `backend_mapping` | 默认后端与各模块后端分工（如 solver/planner/summarizer → deepseek，judge → mimo） |
| `model.backend_sampling` | `modules.*` | 模块级采样参数（temperature / max_tokens） |
| `model.context_window_tokens` | `{deepseek: 128000, mimo: 32000}` | 每后端上下文窗口（token），"丢旧轮次"截断兜底阈值 |
| `contract` | `max_iterations` / `max_concurrent` / `global_timeout_seconds` / `stop_on_no_effective_new_evidence` / `enable_evidence_verification` / `refiner_input_token_budget` / `searcher_dedup_tool_results` / `searcher_max_search_rounds` / `searcher_max_searches_per_round` | 证据链流预算与开关（默认：最多 3 轮、并发 3、全局超时 900s、Refiner 预算 65536 tokens、检索 3 轮 × 1 问、去重开） |
| `conversation` | `enabled` | 对话留档开关（默认 false） |
| `eval` | `cutoffs` / `nli_session` / `nli_concurrency` / `sessions` | 评测参数（见"评测"） |

---

## 📊 评测

评测子系统在 `src/contract/eval/`，每次运行把"输入 → prompt → 原始输出 → gold → 得分 → 遥测"全量持久化到 `evaluation/runs/<mode>/<时间戳>/`，支持断点续跑（按稳定 `instance_id` 跳过已完成实例）。

**评测设计**

- **LegalBenchRAG**（`contractnli / cuad / maud / privacy_qa`）：每条 query 跑完整 LLM Searcher（多轮检索收集完整条款），指标为文档层 `agent_doc_precision / agent_doc_recall`（证据命中的 gold 相关文档占比 / 覆盖比例）＋ 字符层 `agent_span_precision / agent_span_recall / agent_span_f1`（证据对 gold 字符区间的重叠，官方区间口径）。
- **ContractNLI**（端到端分类）：每条实例把 hypothesis 当研究问题、作用域锁到该合同，跑一遍正式完整链路（Planner → Searcher → Reviewer → Refiner），评测给 Refiner 切换 3 选 1 标签专用提示词（`entailment / contradiction / neutral`），从 `conclusion` 字段提取分类结果；正式生产链路不注入该提示词。输出 Accuracy / weighted F1 / per-class F1。实例间可并发（默认 2）。

**证据与编排遥测**

- 引用审计按最终答案中去重后的原始 Evidence ID 统计：`citation_total_count`、`existing_evidence_id_count`、`missing_evidence_id_count`、`source_text_match_count`，其中 `citation_validity_rate = source_text_match_count / citation_total_count`，无引用时为 0。
- Searcher 记录候选数、物化失败数、CitationVerifier 拦截数与通过数及具体 `drop_reasons`，并跨所有研究轮次累计；完整链路另记录平均规划轮数、Searcher 数、`search/grep` Tool Call 数、Reviewer `SUFFICIENT` 比例、Early Stop（仅 `no_effective_new_evidence`）比例和达到 `max_iterations` 比例。停止原因还包括 `reviewer_sufficient`、`direct_after_search`、`incremental_plan_empty`。
- LegalBenchRAG 的 `evidence_hit_rate`：返回的完整条款与同文档 gold 字符区间有正长度重叠即命中，整体为所有任务的命中证据总数 / 返回证据总数；其他 LegalBenchRAG 指标仍按 benchmark 等权聚合。

```bash
# 语料入库（LegalBenchRAG 会话为空时评测会自动处理；ContractNLI 需手动入库或加 --ingest-nli）
python -m src.contract.eval.ingest_raw contractnli --root <LegalBenchRAG根目录>   # cuad / maud / privacy_qa
python -m src.contract.eval.ingest_raw nli --contractnli-jsonl <jsonl或zip> --session nli-contractnli

# 跑评测
python -m src.contract.eval.main --mode legalbenchrag
python -m src.contract.eval.main --mode legalbenchrag --only privacy_qa
python -m src.contract.eval.main --mode contractnli --limit 15
python -m src.contract.eval.main --mode contractnli --limit 15 --nli-concurrency 2 \
  --searcher-max-rounds 1 --searcher-max-searches-per-round 3   # 检索预算 A/B 对照腿

# 小批量冒烟：预生成请求集（configs/eval_sets/smoke_*.json）只跑子集
python -m src.contract.eval.main --mode legalbenchrag --request-set configs/eval_sets/smoke_legalbenchrag_3.json
python -m src.contract.eval.main --mode contractnli --request-set configs/eval_sets/smoke_contractnli_5.json

# ContractNLI 编排消融（默认使用固定的 10% 请求集 configs/eval_sets/contractnli_15.json）
python -m src.contract.eval.ablation
# ContractNLI 150 条与 LegalBenchRAG 100 条正常评测
python -m src.contract.eval.main --mode contractnli --request-set configs/eval_sets/contractnli_150.json
python -m src.contract.eval.main --mode legalbenchrag --request-set configs/eval_sets/legalbenchrag_100.json
```

**评测表现（2026-08 实测）**

**ContractNLI（完整链路，150 条，3 轮 × 1 问，并发 2）**

| 指标 | 结果 |
|------|------|
| Accuracy | 0.853 |
| Weighted F1 | 0.852 |
| Per-class F1 | entailment 0.894 / neutral 0.853 / contradiction 0.667 |
| 错误样例 | 0 / 150 |
| 全链路估算 token | 约 1038 万（其中 Searcher 约 405 万） |

**LegalBenchRAG（Searcher 链路，100 条 query，4 个 benchmark）**

| 指标 | 整体 | contractnli(14) | cuad(59) | maud(24) | privacy_qa(3) |
|------|------|------|------|------|------|
| agent_doc_precision / recall | 0.852 | 0.643 | 0.932 | 0.833 | 1.000 |
| agent_span_precision | 0.107 | 0.131 | 0.082 | 0.139 | 0.078 |
| agent_span_recall | 0.703 | 0.536 | 0.742 | 0.535 | 0.999 |
| agent_span_f1 | 0.162 | 0.187 | 0.126 | 0.193 | 0.143 |

**检索预算 A/B（同一 5 条任务，并发 2）**：`3 轮 × 1 问` 与 `1 轮 × 3 问` 预测完全一致（Acc 0.80 / F1 0.72）；`1 轮 × 3 问` 墙钟更快（223s vs 308s，-27%）但 Searcher token 更多（145.6k vs 100.7k，+45%）。默认取 `3 轮 × 1 问` 以省 token。

**解读**：文档级命中率显著高于字符级精度——Searcher 找对文档 / 条款的能力强（cuad 0.93、privacy_qa 1.00），而 span 精度低是设计使然：证据恢复的是**完整条款**，比 gold 标注的精确子区间更宽（召回高、精度低）。这是"证据 = 完整原文、禁止截断改写"这一硬约束的直接体现。

---

## 🗺️ Roadmap

- [x] 合同证据链主流程 + 断点式评测体系
- [x] 多后端 LLM 路由（DeepSeek / MiMo / vLLM / OpenAI）
- [x] 规划 / 检索配额与上下文预算治理
- [ ] **证据板（Evidence Board）**：把进入 Agent 上下文的证据统一为干净的 JSON（`evidence_id / doc / section / page / quote` 等少数字段，去掉调试打分字段），Reviewer / Refiner 读同一块板，超预算时按 `supporting_evidence_ids` 分层显示并凭 ID 回查
- [ ] 跨 run 记忆复用（基于已核实条款结论，接 PostgreSQL，统一 bge-m3 向量）
- [ ] Web UI

---

## 🤝 贡献 & License

欢迎提交 Issue 与 PR。

[MIT](LICENSE) © LexContract Contributors
