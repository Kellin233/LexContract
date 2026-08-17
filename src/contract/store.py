"""EvidenceStore：运行期内存证据库，按 (document_id, start_offset, end_offset) 去重。

Worker A/B 同时命中同一条款时复用同一 E###；外部（Planner/Reviewer/Refiner）
只传 evidence_id，真正需要阅读正文时再按需取出。
"""
from __future__ import annotations

from .schemas import Evidence


class EvidenceStore:
    def __init__(self) -> None:
        self._by_id: dict[str, Evidence] = {}
        self._by_key: dict[tuple[str, int, int], str] = {}
        self._counter = 0

    def register(self, evidence: Evidence, question_id: str) -> tuple[Evidence, bool]:
        """登记证据；命中去重键则复用已有 Evidence。返回 (evidence, is_new)。"""
        key = (evidence.document_id, evidence.start_offset, evidence.end_offset)
        existing_id = self._by_key.get(key)
        if existing_id is not None:
            return self._by_id[existing_id], False
        self._counter += 1
        evidence.evidence_id = f"E{self._counter:03d}"
        evidence.question_id = question_id
        self._by_id[evidence.evidence_id] = evidence
        self._by_key[key] = evidence.evidence_id
        return evidence, True

    def get(self, evidence_ids: list[str]) -> list[Evidence]:
        return [self._by_id[e] for e in evidence_ids if e in self._by_id]

    def get_by_id(self, evidence_id: str) -> Evidence | None:
        return self._by_id.get(evidence_id)

    def all_ids(self) -> list[str]:
        # 按注册顺序返回
        return [e.evidence_id for e in sorted(self._by_id.values(), key=lambda x: int(x.evidence_id[1:]))]

    def all(self) -> list[Evidence]:
        return [self._by_id[e] for e in self.all_ids()]

    def __len__(self) -> int:
        return len(self._by_id)
