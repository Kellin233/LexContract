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
| `src/contract/tools.py` | `DocumentToolkit`：`search / get_chunk / get_context / get_section / get_document_outline / get_referenced_section`（条款级检索 + 交叉引用跟随） |
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
| `contract` | `max_iterations` / `max_concurrent` / `global_timeout_seconds` / `stop_on_no_effective_new_evidence` / `enable_evidence_verification` | 证据链流预算与开关 |
| `eval` | `k` / `agent_limit` / `contractnli_mode` / `sessions` | 评测参数（见下文） |

### 上下文预算（"压缩"的现行做法）

现行链路**不做语义级文本压缩**（证据必须保留原文），改用以下手段把上下文控制住：

- **输入窗口**：`VLLMPolicy.MAX_CONTEXT_CHARS = 150_000`（≈50K content tokens），仅超此阈值才触发"丢旧轮次"的滑动窗口兜底（`src/models/vllm_policy.py`）；
- **规划配额**：Planner 每次规划最多 **3 个**调查要点、整个流程最多 **3 轮**（`MAX_QUESTIONS_PER_CALL=3`，`max_iterations=3`）；
- **检索配额**：Searcher 最多 **3 个检索轮** × 每轮最多 **3 个检索词**（≈最多 9 次搜索；`MAX_SEARCH_ROUNDS=3`、`MAX_SEARCHES_PER_ROUND=3`），并设"收官轮"强制模型输出证据 JSON，避免搜很多却交不出候选（`src/contract/worker.py`）；
- **按需构造上下文**：Planner/Reviewer 只喂精简信息，Refiner 才是唯一全量读证据正文的环节；
- **用 Evidence ID 而非全量字节在两轮之间传递**：`EvidenceStore` 按跨度去重后只传 `E###`，要读正文时按 ID 取出。

> 历史教训：曾经存在 `store = store or self.store` 的写法，`EvidenceStore` 的 `__len__` 使空库在 `bool()` 下为 False，导致本次运行的证据库被当作"未提供"而静默丢弃，Searcher 找到的证据始终进不了 Reviewer/Refiner，最终 Refiner 在零证据下编造数字（0.5%、15 日等与原文不符的内容）。已改为 `is None` 判断并加注释（`src/contract/worker.py:_assemble_worker_result`），并补了端到端验证。

---

## 📊 评测（`src/contract/eval/`）

针对合同流程的回归评测：

- **LegalBenchRAG**（`contractnli / cuad / maud / privacy_qa` 4 个 benchmark）：**确定性混合检索**全量跑（文档级 `Recall@k(k=1..64)` + MRR）＋ **LLM Searcher agent** 按 `--agent-limit` 抽样跑（字符区间 `Precision/Recall/F1`）双报告；
- **ContractNLI**（端到端分类）：默认 **`indexed` 模式 = 合同整库入库 + 按假设检索出相关条款再分类**（对齐 PAKTON 的"文档内检索"口径）；`--nli-mode direct` 保留"整段前提直喂"的 naive baseline。输出 `Accuracy / weighted F1 / per-class F1`；
- **偏移对齐是前提**：`ingest_raw.py` 把语料"原样入 PG"（`full_text` 逐字保留原文，`charspan` 即语料原文坐标），gold span 才能精确比对；
- **全量持久化 + 断点续跑**：每条 `input → prompt → 原始输出 → gold → 得分 → 遥测` 即时写入 `records.jsonl`，`summary.json` + `metrics.csv` 汇总，以稳定 `instance_id` 跳过已完成实例。

```bash
# 语料入库（LegalBenchRAG 会话为空时评测会自动处理，也可手动）
python -m src.contract.eval.ingest_raw contractnli        # cuad / maud / privacy_qa
# ContractNLI 合同入库（indexed 模式前必须执行；不会自动入库）
python -m src.contract.eval.ingest_raw nli

# 跑评测
python -m src.contract.eval.main --mode legalbenchrag
python -m src.contract.eval.main --mode legalbenchrag --only privacy_qa --agent-limit 20
python -m src.contract.eval.main --mode contractnli --limit 100
python -m src.contract.eval.main --mode contractnli --nli-mode direct --limit 10
```

输出在 `configs/default.yaml` 的 `eval:` 段可配（top-k、抽样、会话、输出目录）。

---

## 🗺️ Roadmap（当前方向 🔥）

- [x] 合同证据链主流程 + 断点式评测体系
- [x] 多后端 LLM 路由（DeepSeek / MiMo / vLLM / OpenAI）
- [x] 规划 / 检索配额与上下文预算治理（3 轮 × 3 要点；3 检索轮 × 3 检索词；窗口 50K）
- [x] 修复"证据被静默丢弃导致 Refiner 编造"的根因 bug
- [ ] **证据板（Evidence Board）**：把进入 Agent 上下文的证据统一为干净的 JSON（`evidence_id / doc / section / page / quote` 等少数字段，去掉调试打分字段），Reviewer/Refiner 读同一块板，超预算时按 `supporting_evidence_ids` 分层显示并凭 ID 回查
- [ ] 跨 run 记忆复用（基于已核实条款结论，接 PostgreSQL，统一 bge-m3 向量）
- [ ] Web UI

---

## 🤝 贡献 & License

欢迎提交 Issue 与 PR。

[MIT](LICENSE) © LexContract Contributors
