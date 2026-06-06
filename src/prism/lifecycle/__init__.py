"""Prism 生命周期管理：归档 / 信任衰减 / 矛盾检测 / 物理清除。

子模块：

- :mod:`prism.lifecycle.archive` — 单条归档 + TTL cron + 90d 物理清除
- :mod:`prism.lifecycle.trust_decay` — 信任衰减 + helpful/unhelpful
- :mod:`prism.lifecycle.contradict` — 矛盾检测增量版
"""

from __future__ import annotations

from .archive import (
    ARCHIVE_REASON_BUILTIN_REMOVED,
    ARCHIVE_REASON_CONTRADICTED,
    ARCHIVE_REASON_GHOST,
    ARCHIVE_REASON_LOW_TRUST,
    ARCHIVE_REASON_MANUAL,
    ARCHIVE_REASON_REPLACED,
    ARCHIVE_REASON_TTL,
    ArchiveResult,
    PurgeResult,
    archive_by_ttl,
    archive_fact,
    purge_old_archived,
)
from .contradict import (
    DEFAULT_CONTRADICT_THRESHOLD,
    DEFAULT_CORPUS_LIMIT,
    DEFAULT_ENTITY_OVERLAP_MIN,
    DEFAULT_MIN_ENTITIES,
    DEFAULT_SEMANTIC_SIM_MAX,
    ContradictDetector,
    ContradictResult,
    detect_contradiction,
    list_contradictions,
)
from .trust_decay import (
    DEFAULT_HELPFUL_DELTA,
    DEFAULT_LOW_TRUST_THRESHOLD,
    DEFAULT_UNHELPFUL_DELTA,
    DecayResult,
    FeedbackResult,
    FeedbackSignal,
    apply_feedback,
    apply_trust_decay,
)

__all__ = [
    "ARCHIVE_REASON_BUILTIN_REMOVED",
    "ARCHIVE_REASON_CONTRADICTED",
    "ARCHIVE_REASON_GHOST",
    "ARCHIVE_REASON_LOW_TRUST",
    "ARCHIVE_REASON_MANUAL",
    "ARCHIVE_REASON_REPLACED",
    "ARCHIVE_REASON_TTL",
    "DEFAULT_CONTRADICT_THRESHOLD",
    "DEFAULT_CORPUS_LIMIT",
    "DEFAULT_ENTITY_OVERLAP_MIN",
    "DEFAULT_HELPFUL_DELTA",
    "DEFAULT_LOW_TRUST_THRESHOLD",
    "DEFAULT_MIN_ENTITIES",
    "DEFAULT_SEMANTIC_SIM_MAX",
    "DEFAULT_UNHELPFUL_DELTA",
    "ArchiveResult",
    "ContradictDetector",
    "ContradictResult",
    "DecayResult",
    "FeedbackResult",
    "FeedbackSignal",
    "PurgeResult",
    "apply_feedback",
    "apply_trust_decay",
    "archive_by_ttl",
    "archive_fact",
    "detect_contradiction",
    "list_contradictions",
    "purge_old_archived",
]
