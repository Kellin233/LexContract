#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/core/runner.py
================================================================================
LexContract 合同证据链执行入口（改造自 DeepResearch Agent runner）。

对外接口:
    - load_config(config_path) -> dict
    - initialize_modules(config, session_id) -> dict
    - run_research(query, config, modules, session_id, doc_ids) -> ResearchReport
    - save_report(report, query, output_dir) -> (md_path, json_path)
================================================================================
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# 将项目根目录加入 sys.path，确保 src 包可导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
def setup_logging(log_level: str = "INFO") -> None:
    """配置全局日志格式与级别。"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------
def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 模块初始化
# ---------------------------------------------------------------------------
def initialize_modules(config: dict, session_id: str = "") -> dict[str, Any]:
    """根据配置初始化合同证据链的所有核心模块。"""
    logger = logging.getLogger("runner")
    logger.info("正在初始化合同证据链模块...")

    modules: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 多后端 LLM 初始化（从 .env + configs/default.yaml 读取配置）
    # ------------------------------------------------------------------
    from src.models.model_router import ModelRouter

    model_cfg = config.get("model", {})
    default_backend = model_cfg.get("backend", "deepseek")
    backend_mapping = model_cfg.get("backend_mapping", {})
    backend_sampling = model_cfg.get("backend_sampling", {})

    def _get_sampling_kwargs(module_name: str, backend_name: str) -> dict:
        kwargs = {}
        if backend_name in backend_sampling:
            kwargs.update(backend_sampling[backend_name])
        kwargs.update(backend_sampling.get("modules", {}).get(module_name, {}))
        return kwargs

    default_kwargs = _get_sampling_kwargs("default", default_backend)
    default_policy = ModelRouter.create_backend(default_backend, **default_kwargs)
    modules["default_policy"] = default_policy
    logger.info(f"[LLM] 默认后端已加载: {default_backend}")

    for module_name, backend_name in backend_mapping.items():
        try:
            kwargs = _get_sampling_kwargs(module_name, backend_name)
            modules[f"{module_name}_policy"] = ModelRouter.create_backend(backend_name, **kwargs)
            logger.info(f"[LLM] {module_name} → 后端={backend_name}")
        except Exception as e:  # noqa: BLE001
            # 未配置/不可用的后端（如 judge→mimo 未配 key）回退到默认后端，避免整条链路不可用
            logger.warning(f"[LLM] {module_name} 后端 {backend_name} 不可用，回退默认后端: {e}")
            modules[f"{module_name}_policy"] = default_policy

    # ------------------------------------------------------------------
    # 数据检索与装配（document + retrieval 底座）
    # ------------------------------------------------------------------
    from src.contract.tools import DocumentToolkit
    from src.contract.assembler import EvidenceAssembler
    from src.contract.verifier import CitationVerifier
    from src.contract.store import EvidenceStore
    from src.contract.worker import Searcher

    toolkit = DocumentToolkit(session_id=session_id)
    assembler = EvidenceAssembler(toolkit)
    verifier = CitationVerifier(toolkit)
    modules["toolkit"] = toolkit
    modules["evidence_store"] = EvidenceStore()

    def _make_searcher():
        return Searcher(
            name="searcher",
            policy=modules.get("solver_policy", default_policy),
            toolkit=toolkit,
            assembler=assembler,
            verifier=verifier,
            store=modules["evidence_store"],  # 运行期由 context 覆盖为每轮独立 EvidenceStore
        )

    # ------------------------------------------------------------------
    # 规划 / 审查 / 结论
    # ------------------------------------------------------------------
    from src.contract.planner import Planner
    from src.contract.reviewer import Reviewer
    from src.contract.refiner import Refiner
    from src.orchestrator.agent_pool import AgentPool
    from src.orchestrator.orchestrator import Orchestrator

    planner = Planner(policy=modules.get("planner_policy", default_policy))
    reviewer = Reviewer(policy=modules.get("judge_policy", default_policy))
    refiner = Refiner(policy=modules.get("summarizer_policy", default_policy))
    modules["planner"] = planner
    modules["reviewer"] = reviewer
    modules["refiner"] = refiner

    agent_pool = AgentPool(
        policy_factory=lambda: modules.get("solver_policy", default_policy),
        worker_factory=_make_searcher,
        max_idle=3,
    )
    modules["agent_pool"] = agent_pool

    # 可选 M4 持久化（仅落最终报告）
    from src.memory.memory_store import SharedMemoryStore
    memory_cfg = config.get("memory", {})
    memory_store = SharedMemoryStore(
        db_path=memory_cfg.get("db_path", "data/memory.db"),
        session_id=session_id,
    )
    modules["memory_store"] = memory_store

    orchestrator = Orchestrator(
        planner=planner,
        agent_pool=agent_pool,
        reviewer=reviewer,
        refiner=refiner,
        evidence_store=modules["evidence_store"],
        compressor=None,
        memory_store=memory_store,
    )
    modules["orchestrator"] = orchestrator
    logger.info("[M1] 合同证据链 Orchestrator 模块已初始化")
    return modules


# ---------------------------------------------------------------------------
# 研究流程主函数
# ---------------------------------------------------------------------------
async def run_research(
    query: str,
    config: dict,
    modules: dict[str, Any],
    session_id: str = "",
    doc_ids: list[str] | None = None,
):
    """执行合同证据链流程，返回 ResearchReport。"""
    import asyncio

    logger = logging.getLogger("runner")
    logger.info(f"开始合同研究，查询: {query[:80]}...")

    from src.orchestrator.schemas import RunConfig

    contract_cfg = config.get("contract", {})
    orch_cfg = config.get("orchestrator", {})
    run_cfg = RunConfig(
        max_concurrent=contract_cfg.get("max_concurrent", orch_cfg.get("max_concurrent", 5)),
        global_timeout_seconds=contract_cfg.get("global_timeout_seconds", orch_cfg.get("global_timeout_seconds", 600)),
        max_iterations=contract_cfg.get("max_iterations", 3),
        stop_on_no_effective_new_evidence=contract_cfg.get("stop_on_no_effective_new_evidence", True),
        enable_evidence_verification=contract_cfg.get("enable_evidence_verification", True),
        session_id=session_id or "",
        doc_ids=list(doc_ids or []),
    )

    orchestrator = modules["orchestrator"]
    report = await orchestrator.run(query, config=run_cfg)
    logger.info(
        f"[Orchestrator] 报告生成完成 | 状态={report.structured.get('final_status') if report.structured else '?'} | "
        f"置信度={report.confidence:.2f} | 搜索={report.num_searches} | 增量轮数={report.num_replan}"
    )
    return report


# ---------------------------------------------------------------------------
# 输出格式化与保存
# ---------------------------------------------------------------------------
def format_report_markdown(report, elapsed: float) -> str:
    """把 ResearchReport 组装成最终 Markdown（正文 + 元信息）。"""
    content = report.content if isinstance(report.content, str) else str(report.content)
    structured = report.structured or {}
    status = structured.get("final_status", "?")
    lines = [
        content,
        "---",
        "",
        "## 元信息",
        "",
        f"- **证据状态**: {status}",
        f"- **置信度**: {report.confidence:.2f}",
        f"- **搜索轮数**: {report.num_searches}",
        f"- **增量规划轮数**: {report.num_replan}",
        f"- **总耗时**: {elapsed:.2f} 秒",
        "",
    ]
    return "\n".join(lines)


def _output_stem(query: str, timestamp: str) -> str:
    safe_query = "".join(c if c.isalnum() or c in "_-" else "_" for c in query[:20])
    return f"report_{timestamp}_{safe_query}"


def save_report(report, query: str, output_dir: str = "outputs/reports") -> tuple[str, str]:
    """写 .md（可读报告）与 .json（结构化结果），返回 (md_path, json_path)。"""
    if not isinstance(report, str) and hasattr(report, "structured"):
        structured = report.structured or {}
    elif isinstance(report, str):
        structured = {}
    else:
        structured = getattr(report, "structured", {}) or {}

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _output_stem(query, timestamp)

    md_path = os.path.join(output_dir, f"{stem}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report.content if not isinstance(report, str) else report)

    json_path = os.path.join(output_dir, f"{stem}.json")
    payload = {
        "query": query,
        "final_status": structured.get("final_status", None),
        "confidence": getattr(report, "confidence", 0.0),
        "num_searches": getattr(report, "num_searches", 0),
        "num_replan": getattr(report, "num_replan", 0),
        "result": structured,
    }
    import json as _json

    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False, indent=2)

    return md_path, json_path
