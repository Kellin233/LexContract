<div align="center">

# 📜 LexContract — 合同证据链研究系统

### *从合同文档到"可验证"的审查结论，全链路自动化*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Async](https://img.shields.io/badge/Async-asyncio-orange.svg)](https://docs.python.org/3/library/asyncio.html)

</div>

---

## 📖 这是什么

LexContract 由一套通用 multi-agent 深度研究框架（原 DeepResearch Agent）**原地改造**而来，聚焦**合同 / 法律文档的证据链式研究**：针对一个问题，多个研究 Agent 只负责从合同库里收集"可引用的原始条款证据"，由唯一的结论 Agent 基于证据生成带引用、可回查、不编造的审查报告。

核心不是"检索 → 中间总结 → 再总结 → 最终答案"，而是：

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

### 设计原则（贯穿全代码的硬约束）

1. **证据 = 原文，永不改写、永不压缩丢失**：`Evidence.quote` 是从数据库 `documents.full_text[start:end]` 截取的**连续原文**，禁止模型改写；
2. **引用只许用证据 ID**：结论正文只用 `[E001]` 标注，由后处理把 E### 映射成《文档》章节 + 页码，杜绝模型自编条款号；
3. **分层职责**：Searcher 只找证据、Reviewer 只评完整性、Refiner 是唯一生成结论的 Agent；中间层一律不产出推理性结论，降低错误在多轮 Agent 间传播的风险；
4. **需要的原文随时可取回**：证据带 `evidence_id + section_path + charspan`，全文在库里，凭 ID 即回查。

---

## 🧱 模块结构（`src/`）

### 现行主链路

| 模块 | 职责 |
|------|------|
| `src/contract/schemas.py` | 领域模型：`Evidence / ResearchQuestion / WorkerResult / ReviewResult / RefinerResult / ResearchState` |
| `src/contract/planner.py` | Planner：`initial_plan` 拆调查要点 + `incremental_plan` 只补缺失；每轮最多 3 个 |
| `src/contract/worker.py` | `Searcher(BaseAgent)`：多轮工具循环收集证据，输出 `WorkerResult`（仅证据） |
| `src/contract/tools.py` | `DocumentToolkit`：检索工具 `search_vector / search_bm25 / search_hybrid`（语义/关键词/融合三档拆成独立工具）＋ `grep`（字面/正则精确匹配），展馆工具 `get_chunk / get_context / get_section / get_document_outline / get_referenced_section`（条款级检索 + 交叉引用跟随）；四个检索工具共用检索预算 |
| `src/contract/assembler.py` | `EvidenceAssembler`：候选 → 从 full_text 截取连续原文的 `Evidence` |
| `src/contract/verifier.py` | `CitationVerifier`：quote 与 DB 原文逐字符比对 + 切片覆盖无空洞 |
| `src/contract/store.py` | `EvidenceStore`：运行期内存证据库，按 `(doc_id, start, end)` 去重，分配 `E###` |
| `src/contract/reviewer.py` | Reviewer：覆盖度 / 冲突 / 缺口审查；`effective_new_evidence` 决定是否提前收尾 |
| `src/contract/refiner.py` | Refiner：唯一结论 Agent；`supporting_evidence_ids` 只落最支撑结论的证据子集 |
| `src/contract/jsonutil.py` | 稳健 JSON 提取（对象 / 数组，去围栏 / 去噪） |
| `src/orchestrator/` | 状态机编排器（PLANNING → DISPATCHING → COLLECTING → REVIEWING → INCREMENTAL_PLANNING → REFINING）+ `AgentPool` 对象池（DAG 分层并发 + Semaphore + 超时） |
| `src/models/` | 多后端 LLM 路由（DeepSeek / MiMo / vLLM / OpenAI），模块级采样参数管理 |
| `src/utils/` | `.env` 加载（`env_config.py`）、LangSmith 追踪（`tracing.py`） |

### 数据底座（复用）

| 模块 | 职责 |
|------|------|
| `src/document/` | 文档解析（txt / pdf / docx）→ 结构感知切块 → bge-m3 向量化 → 入库；`full_text` 列保存原文 |
| `src/retrieval/` | `PostgresRetriever`：vector（pgvector）/ BM25（ParadeDB `pg_search`）/ hybrid（加权 RRF）；`session_id` 作用域强制 |
| `src/contract/eval/` | 评测子系统（见下文"评测"） |

### 遗留模块（改造前 deep-research 时代，当前合同链路**不再使用**）

- `src/adversarial/`（Red/Blue 对抗）、`src/evolution/`（GRPO 自进化）、`src/compressor/`（M3 上下文压缩）
- `src/tools/`（web_search / arxiv / browser / calculator 等 web 研究工具）
- `src/agents/researcher.py` / `summarizer.py`（仅 `base_agent.py` 作为 Searcher 基类在用）
- `src/memory/`：`SharedMemoryStore` 当前只用于把**最终报告**写入 SQLite（`data/memory.db`），不参与检索；`short_term.py` 未使用
- 顶层 `evaluation/` 及 `scripts/run_ablation.py / run_benchmark.py / run_evolution.py / run_judge.py / run_all_experiments.py / run_repl.py` 面向旧 web 流程，已不适配
- `configs/agents/`、`configs/evolution/`、`configs/tools/` 系列 YAML、`pyproject.toml` 里的包名/入口均为旧工程残留

> 读代码时别被以上旧模块误导——主流程只依赖 `src/contract/` + `src/orchestrator/` + `src/document/` + `src/retrieval/` + `src/models/`。

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 依赖（Python 3.10+）
pip install -r requirements.txt

# 配置 .env（连接信息：API Key / Base URL / Model；复制模板后填入）
cp .env.template .env
# 需要配置的：DEEPSEEK_API_KEY（主流程）、MIMO_API_KEY（评分/对抗，可选）、PG_*、EMBED_MODEL_NAME
```

### 2. 数据底座（PostgreSQL）

合同向量库依赖 **PostgreSQL + pgvector + ParadeDB `pg_search` 扩展**（BM25）。连接配置在 `.env`（默认 `localhost:5433/lexcontract`，可用 `PG_*` 覆盖）。

```bash
# 初始化检索 schema（pg_search 扩展 / session_id / search_tokens / BM25 索引）
python -m src.retrieval.main init-db

# 解析并把合同入库（含全文 + 向量 + BM25 tokens）
python -m src.document.main parse <合同文件或目录>

# 旧库迁移（加 full_text 列 + 恢复全文 + 回填 BM25 tokens）
python -m src.document.main migrate

# 把文档分派到会话（检索强制要求 session 作用域）
python -m src.retrieval.main assign <doc_id> --session S1
python -m src.retrieval.main sessions        # 查看会话
```

### 3. 跑一条合同问题

```bash
python scripts/run_single.py --query "乙方逾期交付货物，需要承担什么责任？" --session S1
python scripts/run_single.py --query "供应商能否单方面终止合同？" --session S1 --doc doc_a,doc_b
```

输出 `outputs/reports/report_*.md`（可读报告）+ `report_*.json`（结构化 RefinerResult：`conclusion / points[.evidence_ids] / supporting_evidence_ids / citations / evidence_gap / final_status`）+ 同目录 `run_*.log`（全程 tee 日志）。

### 4. 冒烟测试

```bash
python tests/contract_smoke.py
```

两层：**Tier A**（离线，真实 DB）验证 `EvidenceAssembler / CitationVerifier / EvidenceStore` —— quote 与原文一致、篡改引用被拒、按跨度去重；**Tier B**（确定性 stub LLM，不依赖网络）驱动完整 Orchestrator 状态机（NEED_MORE → 增量规划 → 第二轮 → SUFFICIENT → Refiner）。DB 不可达时打印 SKIP 正常退出。

---

## 🎛️ 配置与核心参数

全局配置集中在 `configs/default.yaml`（唯一被主流程加载的配置）：

| 段 | 关键项 | 说明 |
|----|--------|------|
| `model` | `backend` / `backend_mapping` | 默认后端与各模块后端分工（如 solver/planner/summarizer → deepseek，judge → mimo） |
| `model.backend_sampling` | `modules.*` | 模块级采样参数（temperature / max_tokens）的唯一出处 |
| `model.context_window_tokens` | `{deepseek: 128000, mimo: 32000}` | 每后端上下文窗口（token），作为"丢旧轮次"截断兜底阈值 |
| `contract` | `max_iterations` / `max_concurrent` / `global_timeout_seconds` / `stop_on_no_effective_new_evidence` / `enable_evidence_verification` / `refiner_input_token_budget` / `searcher_dedup_tool_results` / `searcher_max_search_rounds` / `searcher_max_searches_per_round` | 证据链流预算与开关（refiner 预算默认 65536 tokens；searcher 工具结果去重默认 true；检索预算默认 3 轮 × 每轮 1 问） |
| `conversation` | `enabled` | 对话留档开关（默认 false，见下） |
| `eval` | `k` / `cutoffs` / `nli_session` / `nli_concurrency` / `sessions` | 评测参数（见下文） |

### 对话留档（每个 Agent 的 LLM 请求/响应）

默认关闭。开启后（`conversation.enabled: true`），主研究链会把每个 Agent 每次 LLM 调用的**对话**逐行写进报告同目录的 `agent_conversations_<ts>_<query>.jsonl`：

- `kind: llm_call` 行：`agent`（`planner/initial_plan` / `planner/incremental_plan` / `reviewer` / `refiner` / `searcher/{Q##}`）+ `turn` + 完整 `messages`（system/user/assistant 原文）+ `response`（assistant 原始输出、tool_calls 意图）+ 真实 `usage`（API 上报 token）+ `elapsed_ms` + `status`；
- `kind: run_start / run_end` 行：query、session、终态、搜索轮数，让文件自描述；
- 属于"对话层"深度：工具返回的大 JSON **不落盘**，`role=tool` 的消息只留摘要（`result_chars` + 前 200 字）；证据全文不入此文件（在 DB，按 `E###` 引用）。

实现：独立 `ConversationRecorder`（`src/utils/conversation_recorder.py`），Agent 层打标签、`VLLMPolicy.__call__` 统一上报，用 `contextvars` 跨 `asyncio.to_thread` 传递，不阻塞主流程。评测链路（`evaluation/runs/.../records.jsonl`）同步去截断：ContractNLI 的 `raw_response` 存逐字原始输出；LegalBenchRAG 的 Searcher 补 `prompt` 字段并新增完整 `searcher_trajectory`。

**token 口径**：运行期用 `contextvars` 开一个"全链路 token 账本"（`src/utils/tokens.py` 的 `enter/exit/append_token_usage`）。Orchestrator 每次 run 开账本，Planner / 并行 Searcher / Reviewer / Refiner 每次发 LLM 请求都把入参消息的估算 token（`estimate_messages_tokens`）记进去——用 ContextVar 而非实例属性，评测并发实例各自独立 context、共享的 Planner/Reviewer/Refiner 不会互相踩。ContractNLI 单条 telemetry 里 `searcher_token_usage` = 仅 Searcher，`total_token_usage` = 全部 Agent 之和。

### 上下文预算（"压缩"的现行做法）

现行链路**不做语义级文本压缩**（证据必须保留原文），改用以下手段把上下文控制住：

- **输入窗口（token 口径）**：`VLLMPolicy.max_context_tokens` 默认 128K，按后端在 `model.context_window_tokens` 覆盖；仅超此阈值才触发"丢旧轮次"的滑动窗口兜底（`src/models/vllm_policy.py`）。全项目上下文/预算统一用 `src/utils/tokens.py` 的 `estimate_tokens` 估算（中文按 0.6、其余按分词）——不再用字符数；
- **规划配额**：Planner 每次规划最多 **3 个**调查要点、整个流程最多 **3 轮**（`MAX_QUESTIONS_PER_CALL=3`，`max_iterations=3`）；
- **检索配额（可配，检索类调用共用）**：`contract.searcher_max_search_rounds`（默认 3）与 `contract.searcher_max_searches_per_round`（默认 1）→ 默认最多 **3 个检索轮 × 每轮 1 个检索调用**（≈最多 3 次检索）。检索类调用 = 四个检索工具共用同一预算：`search_vector / search_bm25 / search_hybrid`（语义/关键词/融合三档，拆自原 `search.mode`）＋ `grep`（字面 / POSIX 正则精确匹配，锁定确切措辞或条款号）；展馆工具 `get_*` 与最终 JSON 不计入。`src/contract/worker.py::build_system_prompt` 把预算写进 Searcher 系统提示词，预算改配置即改提示词。并设"收官轮"强制模型输出证据 JSON，避免搜很多却交不出候选。预算可按腿用 CLI `--searcher-max-rounds / --searcher-max-searches-per-round` 覆盖做 A/B。**实测（同一 5 条任务，并发 2）**：`3轮×1问` vs `1轮×3问` 预测完全一致（Acc 0.80 / F1 0.72），无质量差异；`1轮×3问` 更快（223s vs 308s，-27% 墙钟）但 Searcher token 更多（145.6k vs 100.7k，+45%）——少轮次省时间、多检索词烧 token；
- **Searcher 工具结果去重**：`contract.searcher_dedup_tool_results`（默认 `true`）。同一 Searcher 多轮内，同一切片/同一章节的完整原文只注入一次——同义词检索重叠、`get_context` 带回已见底片、`get_referenced_section` 二次落到同一章节，重复项正文替换为短标记（保留 id/偏移/得分骨架），可开可关做 A/B（`src/contract/worker.py::_dedup_tool_result`）；
- **按需构造上下文**：Planner/Reviewer 只喂精简信息，Refiner 才是唯一全量读证据正文的环节；
- **Refiner 输入预算**：`contract.refiner_input_token_budget`（默认 65536 tokens）；超预算只告警并记入报告的 `notes`/`evidence_gap`（"— 证据输入超 Refiner 预算 N tokens…"），**暂不裁剪**证据子集（裁剪策略待做），证据仍逐字全量喂入；
- **用 Evidence ID 而非全量字节在两轮之间传递**：`EvidenceStore` 按跨度去重后只传 `E###`，要读正文时按 ID 取出。

> 历史教训：曾经存在 `store = store or self.store` 的写法，`EvidenceStore` 的 `__len__` 使空库在 `bool()` 下为 False，导致本次运行的证据库被当作"未提供"而静默丢弃，Searcher 找到的证据始终进不了 Reviewer/Refiner，最终 Refiner 在零证据下编造数字（0.5%、15 日等与原文不符的内容）。已改为 `is None` 判断并加注释（`src/contract/worker.py:_assemble_worker_result`），并补了端到端验证。

---

## 📊 评测（`src/contract/eval/`）

针对合同流程的回归评测：

- **LegalBenchRAG**（`contractnli / cuad / maud / privacy_qa` 4 个 benchmark）：每条 query 跑完整 LLM Searcher（多轮检索收集完整条款），指标为文档层 `agent_doc_precision / agent_doc_recall`（证据命中的 gold 相关文档占比 / 覆盖比例）＋ 字符层 `agent_span_precision / agent_span_recall / agent_span_f1`（证据对 gold 区间的字符重叠，官方 PAKTON 口径）；确定性混合检索已移除；
- **ContractNLI**（端到端分类）：每条实例**跑一遍正式完整链路**（`hypothesis` 当研究问题、作用域锁到该合同 `nli:<premise_id>`，`Planner→Searcher→Reviewer→Refiner`），并**按评测给 Refiner 切换专用提示词**——Refiner 直接输出 3 选 1 标签（`entailment / contradiction / neutral`）＋ 最相关证据（`supporting_evidence_ids`），评测从 Refiner 输出的 JSON（`conclusion` 字段，PAKTON 同义词正则归一）提取标签当分类结果。**正式生产链路不注入该提示词，Refiner 保持默认生产提示词**。输出 `Accuracy / weighted F1 / per-class F1`。旧 `indexed/direct` 检索式链路已移除；**实例间可并发**（`eval.nli_concurrency` / CLI `--nli-concurrency`，默认 2），每条实例独立 toolkit/agent_pool/orchestrator（共享 policy/planner/reviewer/Refiner），避免共享 `toolkit.set_scope` 在并发下互相覆盖文档作用域；
- **偏移对齐是前提**：`ingest_raw.py` 把语料"原样入 PG"（`full_text` 逐字保留原文，`charspan` 即语料原文坐标），gold span 才能精确比对；
- **全量持久化 + 断点续跑**：每条 `input → prompt → 原始输出 → gold → 得分 → 遥测` 即时写入 `records.jsonl`，`summary.json` + `metrics.csv` 汇总，以稳定 `instance_id` 跳过已完成实例。

```bash
# 语料入库（LegalBenchRAG 会话为空时评测会自动处理，也可手动）
python -m src.contract.eval.ingest_raw contractnli        # cuad / maud / privacy_qa
# ContractNLI 合同入库（完整链路需要合同已在会话中；不会自动入库）
python -m src.contract.eval.ingest_raw nli

# 跑评测
python -m src.contract.eval.main --mode legalbenchrag
python -m src.contract.eval.main --mode legalbenchrag --only privacy_qa
python -m src.contract.eval.main --mode contractnli --limit 15     # 完整链路 + 3 选 1 Refiner 提示词
python -m src.contract.eval.main --mode contractnli --limit 15 --nli-session nli-10
# 实例并发（默认 2）+ 检索预算 A/B 覆盖
python -m src.contract.eval.main --mode contractnli --limit 5 --nli-concurrency 2 \
  --searcher-max-rounds 1 --searcher-max-searches-per-round 3   # 对照腿：1 轮 × 3 问
```

输出在 `configs/default.yaml` 的 `eval:` 段可配（top-k、抽样、会话、输出目录）。

---

## 🗺️ Roadmap（当前方向 🔥）

- [x] 合同证据链主流程 + 断点式评测体系
- [x] 多后端 LLM 路由（DeepSeek / MiMo / vLLM / OpenAI）
- [x] 规划 / 检索配额与上下文预算治理（3 要点/轮；3 检索轮 × 1 检索词默认，可配；窗口 128K 按后端覆盖）
- [x] 修复"证据被静默丢弃导致 Refiner 编造"的根因 bug
- [ ] **证据板（Evidence Board）**：把进入 Agent 上下文的证据统一为干净的 JSON（`evidence_id / doc / section / page / quote` 等少数字段，去掉调试打分字段），Reviewer/Refiner 读同一块板，超预算时按 `supporting_evidence_ids` 分层显示并凭 ID 回查
- [ ] 跨 run 记忆复用（基于已核实条款结论，接 PostgreSQL，统一 bge-m3 向量）
- [ ] Web UI

---

## 🤝 贡献 & License

欢迎提交 Issue 与 PR。

[MIT](LICENSE) © LexContract Contributors
