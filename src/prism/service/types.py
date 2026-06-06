"""Service 层返回值类型（frozen dataclass）。

API 层负责将这些 dataclass 转为 dict（JSON 序列化友好），
CLI 层负责格式化为人类可读输出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ─── Fact CRUD ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FactResult:
    """add / edit 的返回值。"""

    fact_id: int
    is_new: bool
    entities: tuple[str, ...]
    category: str


@dataclass(frozen=True, slots=True)
class RemoveResult:
    """remove 的返回值。"""

    fact_id: int
    archived: bool


@dataclass(frozen=True, slots=True)
class FeedbackResult:
    """helpful / unhelpful 的返回值。"""

    fact_id: int
    applied: bool
    new_trust_score: float | None
    new_helpful_count: int | None


@dataclass(frozen=True, slots=True)
class ArchiveResult:
    """archive 的返回值。"""

    fact_id: int
    archived: bool
    reason: str


@dataclass(frozen=True, slots=True)
class RestoreResult:
    """restore 的返回值。"""

    fact_id: int
    restored: bool
    category: str | None


@dataclass(frozen=True, slots=True)
class FactDetail:
    """show 的返回值 — 单条 fact 完整字段。"""

    fact_id: int
    content: str
    category: str
    status: str
    trust_score: float
    helpful_count: int
    retrieval_count: int
    mirror_source: str | None
    mirror_target: str | None
    supersedes_id: int | None
    enrichment_status: str
    archived_at: str | None
    archive_reason: str | None
    created_at: str
    entities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FactSummary:
    """list 中每行的摘要。合并了 API 和 CLI 的字段集。"""

    fact_id: int
    content: str
    category: str
    status: str
    trust_score: float
    helpful_count: int
    mirror_source: str | None
    created_at: str
    archived_at: str | None
    archive_reason: str | None


@dataclass(frozen=True, slots=True)
class ListResult:
    """list 的返回值。"""

    facts: tuple[FactSummary, ...]
    count: int
    truncated: bool
    filter: dict[str, Any]


# ─── Search ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SearchHit:
    """search 结构化返回中的单条命中。"""

    fact_id: int
    content: str
    score: float
    path_scores: dict[str, float]


@dataclass(frozen=True, slots=True)
class CoOccurrence:
    """related 返回中的单条共现实体。"""

    entity: str
    co_occurrence: int


@dataclass(frozen=True, slots=True)
class ContradictionItem:
    """contradict 返回中的单条矛盾候选。"""

    contradiction_id: int
    fact_a: int
    fact_b: int
    score: float
    detected_at: str
    content_a: str
    content_b: str
    category_a: str | None
    category_b: str | None


# ─── Admin / Enrichment ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EnrichmentStatusEntry:
    """enrichment_diagnose 中 status_distribution 的单行。"""

    enrichment_status: str
    count: int
    null_vector: int


@dataclass(frozen=True, slots=True)
class QueueItem:
    """enrichment_diagnose 中 queue_items 的单行。"""

    fact_id: int
    attempts: int
    last_error: str | None
    enqueued_at: str | None
    last_attempt_at: str | None


@dataclass(frozen=True, slots=True)
class MissingVectorEntry:
    """enrichment_diagnose 中 missing_vectors 的单行。"""

    fact_id: int
    content: str


@dataclass(frozen=True, slots=True)
class EnrichmentDiagnosis:
    """enrichment_diagnose 的返回值。"""

    queue_count: int
    status_distribution: tuple[EnrichmentStatusEntry, ...]
    queue_items: tuple[QueueItem, ...]
    missing_vectors: tuple[MissingVectorEntry, ...]
    missing_vector_count: int


@dataclass(frozen=True, slots=True)
class MirrorMdStats:
    """mirror_memory_md 的返回值。"""

    total: int
    added: int
    skipped_duplicate: int
    failed: int
    archived: int = 0


@dataclass(frozen=True, slots=True)
class EnrichmentFixResult:
    """enrichment_fix 的返回值。"""

    dry_run: bool
    semantic_available: bool
    embedded: int = 0
    embed_failed: int = 0
    pending_cleared: int = 0
    queue_cleared: int = 0
    vstore_rebuilt: bool = False
    vstore_vectors: int = 0
    # dry_run 专用
    to_embed: int = 0
    pending_to_clear: int = 0
    queue_to_clear: int = 0
