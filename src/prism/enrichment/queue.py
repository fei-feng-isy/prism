"""enrichment_queue 表的封装 + crash-safe 取/标记语义（at-least-once delivery）。

本模块现在是 :class:`prism.db.EnrichmentQueueRepository` 的薄包装，
内部 SQL 操作全部委托到 Repository。保留 :class:`QueueItem` 和
:class:`EnrichmentQueue` 的公开 API 以保持向后兼容。

单 worker 假设（SQLite 无 FOR UPDATE SKIP LOCKED）。
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

from prism.db import EnrichmentQueueRepository

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QueueItem:
    """从队列取出的待处理 fact。`attempts` 是自增后的新值（即「这是第几次尝试」）。"""

    fact_id: int
    content: str
    attempts: int
    last_error: str | None


class EnrichmentQueue:
    """SQLite-backed 异步富化任务队列（薄包装）。

    内部委托到 :class:`EnrichmentQueueRepository`。
    """

    DEFAULT_MAX_ATTEMPTS = 3

    def __init__(
        self,
        db: sqlite3.Connection,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self._repo = EnrichmentQueueRepository(db, max_attempts=max_attempts)

    @property
    def max_attempts(self) -> int:
        return self._repo.max_attempts

    def enqueue(self, fact_id: int) -> bool:
        """加入队列；返回 True 表示新增，False 表示已存在。"""
        return self._repo.enqueue(fact_id)

    def pop_next(self) -> QueueItem | None:
        """取最旧的待处理 fact，原子 attempts++。空队列返回 None。

        仅返回 facts.status='active' 的 fact；已 archived/deleted 的 fact
        即使在队列中也会被忽略（设计上不应出现，但作为防御）。
        """
        row = self._repo.pop_next()
        if row is None:
            return None
        return QueueItem(
            fact_id=row["fact_id"],
            content=row["content"],
            attempts=row["attempts"],
            last_error=row["last_error"],
        )

    def mark_done(self, fact_id: int) -> None:
        """标记 fact 富化成功：从队列删除 + facts.enrichment_status='done'。"""
        self._repo.mark_done(fact_id)

    def mark_failed(self, fact_id: int, error: str) -> None:
        """记一次失败。达到 max_attempts 后从队列移除 + facts.enrichment_status='failed'。"""
        self._repo.mark_failed(fact_id, error)

    def pending_count(self) -> int:
        return self._repo.pending_count()

    def stats(self) -> dict[str, int]:
        """返回 {pending, done, failed} 三段计数。done/failed 来自 facts 表。"""
        return self._repo.stats()
