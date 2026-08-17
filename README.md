<div align="center">

# 🚀 DeepResearch Agent

### *从复杂 Query 到结构化深度研究报告，全链路自动化*

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Async](https://img.shields.io/badge/Async-asyncio-orange.svg)](https://docs.python.org/3/library/asyncio.html)

</div>

---

## 📖 项目背景

大语言模型在单一问答场景表现优异，但在**复杂深度研究任务**中面临三个核心挑战：

1. 🔥 **信息爆炸与上下文遗忘** —— 长文本检索后关键信息淹没在噪声中，模型难以聚焦
2. 👻 **幻觉与事实漂移** —— 多轮推理过程中，模型倾向于"编造"未经验证的事实
3. 📊 **缺乏系统性评估** —— 现有评测多以单轮 QA 为主，缺少对"深度研究报告"这一输出形态的端到端评价体系

本项目从零构建了一套**面向深度研究任务的 Agent 系统**，覆盖规划、执行、记忆、对抗、进化、评测全链路。

---

## 🎯 实验动机

> **"如果一个 Agent 只能回答简单问题，那它和搜索引擎有什么区别？"**

我们的动机是：**让 AI 真正具备"深度研究"的能力**——不只是检索信息，而是像人类研究员一样：
- 🧩 **拆解复杂问题** → 将模糊的研究目标分解为可执行的子任务
- 🔍 **多源信息整合** → 从网页、论文、数据库等多渠道收集证据
- ⚖️ **批判性审视** → 主动发现并修正报告中的错误和偏见
- 📝 **结构化输出** → 生成带引用、有逻辑、可验证的研究报告

---

## 💡 解决方法

### 六大模块协同工作

| 模块 | 职责 | 核心技术 |
|------|------|---------|
| 🎛️ **M1 Orchestrator** | 多智能体编排与调度 | 自研 asyncio + DAG 执行引擎，9 状态状态机 |
| 🗺️ **M2 Planner** | 复杂问题拆解 | JSON DAG 动态规划，支持执行中 replan |
| 🗜️ **M3 Compressor** | 长上下文压缩 | Embedding 语义三级过滤 + TextRank 关键句提取 |
| 🧠 **M4 Memory Store** | 跨 Agent 共享记忆 | SQLite + numpy 向量索引，去重/矛盾检测/LRU 淘汰 |
| ⚔️ **M5 Adversarial Loop** | 对抗降噪 | Red-Blue 循环攻击-修复，内置收敛与震荡检测 |
| 🧬 **M6 Evolution Engine** | 在线自进化 | GRPO 强化学习 + 符号规则学习（预留接口） |

### 数据流全景

```raw
用户 Query
    ↓
🗺️ Planner 拆解为 DAG 子任务图
    ↓
🎛️ Orchestrator 按拓扑排序并发调度
    ↓
🤖 Worker Agents 调用 🔍 搜索 / 📄 论文 / 🌐 网页 工具
    ↓
🧠 Memory Store 写入中间结果（去重 + 矛盾检测）
    ↓
🗜️ Compressor 压缩长上下文（L1→L2→L3）
    ↓
⚔️ Red Agent 攻击 → Blue Agent 修复 → 评分引擎评估
    ↓
📝 Summarizer 合成最终 Markdown 报告
    ↓
📤 输出带元信息的结构化研究报告
```

---

## ✨ 项目精彩之处

### 🏗️ 1. 自研编排引擎，不依赖 LangGraph/AutoGen

> 为什么不用现成的框架？因为深度研究任务需要**完全可控的调度逻辑**。

- 基于 `asyncio` + `Semaphore` 实现 **DAG 拓扑并发执行**
- **9 状态状态机**：IDLE → PLANNING → DISPATCHING → COLLECTING → SYNTHESIZING → ADVERSARIAL → DONE
- **三级降级策略**：单任务超时标记继续 → >50% 失败触发 replan → 全局超时强制合成

### ⚔️ 2. Red-Blue 对抗降噪 —— 主动抑制幻觉

> 灵感来自 GAN 的对抗训练思想，但应用于**文本质量优化**。

- **Red Agent** 从 5 个维度攻击报告：事实性、逻辑一致性、引用质量、覆盖面、时效性
- **Blue Agent** 执行 4 种修复操作：ADD / DELETE / MODIFY / VERIFY
- **收敛控制**：评分达标（≥8.0）/ 变化收敛（Δ < 0.3）/ 轮数上限（3 轮）三选一终止
- **震荡检测**：已修复问题重新出现 → 判定震荡 → 优雅终止

### 🗜️ 3. 语义级上下文压缩 —— 不是简单截断

> 关键词匹配会丢失语义，简单截断会丢失关键信息。**我们用 Embedding 做语义压缩**。

- **L1 粗过滤**：cosine similarity < 0.6 丢弃，> 0.95 完整保留
- **L2 细筛选**：TextRank + Query-Biased 提取关键句
- **L3 精保留**：高度相关内容保留原文，避免摘要失真

### 🧠 4. 跨 Agent 共享记忆 —— 会"反思"的系统

- 写前自动**去重**（cosine > 0.92）
- **矛盾检测**：启发式反义词 + 语义对立识别
- 三种**矛盾消解策略**：Majority Vote / Source Weight / LLM Judge
- **Session 隔离**：不同用户/会话的记忆物理隔离

### 🔌 5. 多后端 LLM 路由 —— 零源码切换模型

```yaml
# configs/default.yaml
model:
  backend_mapping:
    solver: "deepseek"      # 强推理
    planner: "deepseek"     # 结构化输出
    red_agent: "mimo"       # 稳定、低成本
    blue_agent: "mimo"
    judge: "mimo"
    compressor: "mimo"
```

- 支持 DeepSeek / MiMo 2.5 Pro / vLLM / OpenAI **热切换**
- 模块级采样参数集中管理，**避免配置漂移**
- `.env` 驱动，**零源码修改**接入新后端

### 📊 6. 完整的深度研究评测体系

> 不做"跑几个例子看看"的评测，做**可复现、可量化、有统计显著性**的评测。

| 评测层级 | 方法 | 特点 |
|---------|------|------|
| 📏 **规则指标** | 事实准确率 / 幻觉率 / 引用覆盖率 / 逻辑一致性 | 免费、可复现、零 API 成本 |
| 📚 **公共数据集** | HotpotQA 多跳 QA 深度研究变体 | 传统 EM/F1 + 新增语义覆盖度 |
| 🏗️ **自建评测集** | ResearchBench 35 题 × 11 领域 | 含 expected_topics + ground_truth |
| 👨‍⚖️ **LLM-as-Judge** | MiMo 5 维度 0-10 分深度评分 | 定性+定量互补 |
| 🥊 **Head-to-Head** | Agent vs 单轮 LLM 直接对比 | pairwise 更可靠 |
| 📈 **统计显著性** | Bootstrap 95% CI + Cohen's d + t-test | 拒绝"随机波动" |

---

## 🚀 快速开始

### 环境准备

```bash
# 1. 克隆项目
git clone https://github.com/qiqihezh/deepresearch-agent.git
cd deep_research_agent

# 2. 创建 uv 虚拟环境并激活
uv venv .venv
source .venv/bin/activate

# 3. 安装核心依赖
pip install -r requirements.txt

# 4. 配置 API Key（复制模板后填入）
cp .env.example .env
# 编辑 .env：填入 DEEPSEEK_API_KEY、BOCHA_API_KEY 等
```

### 三种运行方式

**🎯 单条 Query（单次深度研究）**
```bash
python scripts/run_single.py \
    --query "2024-2025年大模型Agent技术趋势与落地案例研究" \
    --config configs/default.yaml
```

**💬 交互式 REPL（支持 Session 继承与连续追问）**
```bash
python scripts/run_repl.py
# 交互命令: ls / sessions / save / q
```

**🔬 批量实验（全量评测体系，overnight 可跑完）**
```bash
python scripts/run_all_experiments.py \
    --report_file outputs/reports1/report_xxx.md \
    --report_query "你的研究问题"
```

> 批量实验默认配置：模块消融 5×12 题 + 轮数消融 4×12 题 + 标准评测 35 题 + 领域对比 3×5 题 + Agent vs LLM 3 题 + Judge 1 次 = **165 次独立研究运行**

---

## 📁 仓库结构

```raw
deep_research_agent/
├── 📁 configs/                    # YAML 配置中心
│   ├── default.yaml               # 全局默认配置
│   ├── agents/                    # Agent 行为配置
│   ├── interaction_config/        # 交互层配置
│   └── tool_config/               # 工具层配置
│
├── 📁 src/                        # 核心源码（~5000 行）
│   ├── 📁 core/                   # 核心运行层
│   │   ├── runner.py              # 初始化模块 + 执行完整研究流程
│   │   ├── judge.py               # MiMo Judge 统一接口
│   │   └── ablation.py            # 消融实验通用框架
│   │
│   ├── 📁 orchestrator/           # 🎛️ M1: 多智能体编排器
│   ├── 📁 planner/                # 🗺️ M2: 自适应规划器
│   ├── 📁 compressor/             # 🗜️ M3: 上下文压缩器
│   ├── 📁 memory/                 # 🧠 M4: 共享记忆存储
│   ├── 📁 adversarial/            # ⚔️ M5: 对抗降噪循环
│   ├── 📁 evolution/              # 🧬 M6: 自进化引擎
│   ├── 📁 agents/                 # 🤖 Agent 实现
│   ├── 📁 models/                 # 🔌 模型路由层
│   ├── 📁 tools/                  # 🛠️ 工具层
│   └── 📁 utils/                  # 🧰 工具函数
│
├── 📁 evaluation/                 # 评测体系（~2000 行）
│   ├── benchmarks/                # 评测集（ResearchBench / HotpotQA）
│   ├── metrics/                   # 指标（规则 / Judge / 统计 / 综合）
│   └── analyze_ablation.py        # 消融实验结果分析
│
├── 📁 scripts/                    # 可执行脚本
│   ├── run_single.py              # 🎯 单条 query CLI
│   ├── run_repl.py                # 💬 交互式 REPL
│   ├── run_all_experiments.py     # 🔬 一键批量实验
│   ├── run_ablation.py            # 消融实验独立入口
│   ├── run_benchmark.py           # 🥊 Agent vs LLM
│   ├── run_eval.py                # 标准评测入口
│   ├── run_judge.py               # 👨‍⚖️ Judge 深度评分
│   └── validate_env.py            # 环境配置检查
│
├── 📁 verl/                       # veRL 训练框架（GRPO）
├── requirements.txt               # 依赖清单（分级安装）
└── README.md                      # 📖 本文件
```

---

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| 🐍 语言 | Python 3.11 |
| ⚡ 异步框架 | asyncio |
| 🧠 LLM 后端 | DeepSeek API / MiMo 2.5 Pro / vLLM / OpenAI |
| 🔢 嵌入模型 | sentence-transformers (`all-MiniLM-L6-v2`) |
| 💾 持久化 | SQLite + numpy 向量索引 |
| 🎓 训练框架 | veRL (GRPO) |
| 🔭 可观测性 | LangSmith |
| 📦 虚拟环境 | uv |

---

## ⚖️ 合同证据链模式（LexContract，默认主流程）

本项目已原地改造为 **合同/法律文档证据链研究系统**，替代原 web 深度研究流程。

核心不是“检索 → 中间总结 → 再总结 → 最终答案”，而是：

```raw
问题
 ↓
Planner 拆解“需要调查什么”（研究问题，禁止中间结论）
 ↓
多个 Searcher 并行收集证据（只找完整可引用的原始条款，不写结论）
 ↓
EvidenceAssembler 把碎片 Chunk 恢复成连续原文（quote 一律取自 DB full_text）
 ↓
CitationVerifier 纯程序化校验（quote == full_text[start:end]，无 LLM 判断）
 ↓
EvidenceStore 去重（按 document_id + start_offset + end_offset），只传 Evidence ID
 ↓
Reviewer 只判三件事：证据是否覆盖问题 / 是否明显冲突 / 还缺什么
 ↓
┌─ SUFFICIENT ────────────────────→ Refiner（唯一结论 Agent，输出 JSON + Markdown）
│
└─ NEED_MORE 且 iteration<3 且有有效新增
        ↓ 增量 Planner（只补缺失要点）→ 重新派发 Worker → Reviewer
（达 3 轮 / 无有效新增 → PARTIALLY_SUFFICIENT，仍进 Refiner 并如实说明缺口）
```

### 新增模块（`src/contract/`）

| 文件 | 职责 |
|------|------|
| `schemas.py` | Evidence / ResearchQuestion / WorkerResult / ReviewResult / RefinerResult / ResearchState |
| `planner.py` | Planner：initial_plan（拆调查要点）+ incremental_plan（只补缺失） |
| `worker.py` | Searcher：多轮检索循环，输出 WorkerResult（仅证据，绝不输出结论） |
| `tools.py` | DocumentToolkit：`search` / `get_chunk` / `get_context` / `get_section` / `get_document_outline` / `get_referenced_section` |
| `assembler.py` | EvidenceAssembler：碎片 Chunk → 用于引用的连续原文 Evidence |
| `verifier.py` | CitationVerifier：quote 与 DB 原文精确比对 + 切片覆盖无空洞 |
| `store.py` | EvidenceStore：按跨度去重，只按 ID 传递/读取 |
| `reviewer.py` | Reviewer：研究完整性审查（覆盖度/冲突/缺口），不裁判 |
| `refiner.py` | Refiner：唯一结论 Agent，引用约束 `[E###]`，JSON + Markdown 双输出 |
| `jsonutil.py` | 稳健 JSON 提取（对象/数组，去围栏/去噪） |

### 复用的既有底座

- `src/document/`：Docling 结构解析 + 结构感知切块 + bge-m3 向量化 → PostgreSQL/pgvector（`Chunk/ChunkMetadata` 已含 `section_path/page_no/charspan`）
- `src/retrieval/`：`PostgresRetriever`（vector / BM25(pg_search) / hybrid 加权 RRF），session 作用域隔离
- `src/orchestrator/`：DAG 分层并发 + Semaphore + 单任务超时保持在原有状态机内（新增 `REVIEWING / INCREMENTAL_PLANNING / REFINING` 状态，`TaskType.EVIDENCE`）
- `src/planner/dag.py`、`src/models/`（多后端 LLM 路由）、`src/planner/budget_tracker.py`

### 数据库要求（相对原 document 模块的增量）

- `documents` 表新增 `full_text TEXT` 列（CitationVerifier 精确校验与条款拼接依赖）
- 依赖 **ParadeDB**（`pg_search`）+ pgvector；BM25 依赖 `chunks.search_tokens`（jieba 回填）
- 迁移旧库：`python -m src.document.main migrate`（加列 + 从 JSON 恢复全文 + 回填 tokens；`parse` 已自动回填）

### 使用

```bash
# 1) 解析入库（含全文 + 向量 + BM25 tokens）
python -m src.document.main parse <合同文件或目录>

# 2) 把文档分派到会话（检索必须作用域）
python -m src.retrieval.main assign <doc_id> --session S1

# 3) 提问（输出 .md + .json 双报告）
python scripts/run_single.py --query "供应商能否单方面终止合同？" --session S1
python scripts/run_single.py --query "不可抗力如何影响交货义务" --session S1 --doc doc_a,doc_b
```

输出示例：`outputs/reports/report_*.md`（可读报告）+ `report_*.json`（结构化 RefinerResult：`conclusion / points / citations / evidence_gap / final_status`）。

### 冒烟测试

```bash
python tests/contract_smoke.py
```

两层：Tier A 用真实 DB 验证 `quote == full_text[offset]`、篡改引用被拒、跨度去重；Tier B 用确定性 stub LLM 驱动完整 Orchestrator（NEED_MORE → 增量规划 → 第二轮 → SUFFICIENT → Refiner），不依赖网络。

### 与原工程的关系

- 原地改造：`scripts/run_single.py` 已改为合同入口（`--session` 必填）；`evaluation/`、`run_ablation.py`、`run_benchmark.py` 等面向旧 web 流程的脚本已不再适配（标注弃用）。原 `ResearcherAgent` 的 web 工具、M5 Red/Blue 对抗、M6 进化在合同流程中不再使用（文件保留便于回溯）。
- 设计要点：Chunk 只服务检索效率，Evidence 才服务引用与推理；中间层不产出推理性结论，把主要推理压力集中到最终 Refiner，降低错误在多轮 Agent 间传播的风险。

---

## 🗺️ Roadmap

- [x] 自研编排引擎（asyncio + DAG）
- [x] Red-Blue 对抗降噪
- [x] 语义级上下文压缩
- [x] 跨 Agent 共享记忆
- [x] 多后端 LLM 路由
- [x] 完整评测体系（规则 + Judge + 统计显著性）
- [x] REPL 交互式会话
- [ ] 实验结果填充（进行中 🔥）
- [ ] Web UI（Gradio/Streamlit）
- [ ] 用户反馈闭环
- [ ] 多模态支持（图像/表格）

---

## 🤝 贡献

欢迎提交 Issue 和 PR！无论是 bug 修复、功能增强还是文档改进，我们都非常感谢。

---

## 📄 License

[MIT](LICENSE) © 2025 DeepResearch Agent Contributors
