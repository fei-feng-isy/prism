"""enrichment_queue 表的封装 + crash-safe 取/标记语义（at-least-once delivery）。

单 worker 假设（SQLite 无 FOR UPDATE SKIP LOCKED）。
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QueueItem:
    """从队列取出的待处理 fact。`attempts` 是自增后的新值（即「这是第几次尝试」）。"""

    fact_id: int
    content: str
    attempts: int
    last_error: str | None


class EnrichmentQueue:
    """SQLite-backed 异步富化任务队列。

    单 worker 假设；构造时 `max_attempts < 1` raise ValueError。
    """

    DEFAULT_MAX_ATTEMPTS = 3
    _MAX_ERROR_LEN = 1024

    def __init__(
        self,
        db: sqlite3.Connection,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(
                f"max_attempts must be >= 1, got {max_attempts}"
            )
        self._db = db
        self._max_attempts = max_attempts

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    def enqueue(self, fact_id: int) -> bool:
        """加入队列；返回 True 表示新增，False 表示已存在。"""
        with self._db:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO enrichment_queue (fact_id) VALUES (?)",
                (fact_id,),
            )
            return cur.rowcount > 0

    def pop_next(self) -> QueueItem | None:
        """取最旧的待处理 fact，原子 `attempts++`。空队列返回 None。

        仅返回 `facts.status='active'` 的 fact；已 archived/deleted 的 fact
        即使在队列中也会被忽略（设计上不应出现，但作为防御）。
        """
        with self._db:
            row = self._db.execute(
                """
                SELECT q.fact_id, f.content, q.attempts, q.last_error
                FROM enrichment_queue q
                JOIN facts f ON f.fact_id = q.fact_id
                WHERE f.status = 'active'
                ORDER BY q.enqueued_at ASC, q.fact_id ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            fact_id, content, attempts, last_error = row
            new_attempts = int(attempts) + 1
            self._db.execute(
                "UPDATE enrichment_queue SET attempts = ?, "
                "last_attempt_at = CURRENT_TIMESTAMP WHERE fact_id = ?",
                (new_attempts, fact_id),
            )
            return QueueItem(
                fact_id=int(fact_id),
                content=str(content),
                attempts=new_attempts,
                last_error=last_error,
            )

    def mark_done(self, fact_id: int) -> None:
        """标记 fact 富化成功：从队列删除 + facts.enrichment_status='done'。"""
        with self._db:
            self._db.execute(
                "DELETE FROM enrichment_queue WHERE fact_id = ?",
                (fact_id,),
            )
            self._db.execute(
                "UPDATE facts SET enrichment_status = 'done', "
                "updated_at = CURRENT_TIMESTAMP WHERE fact_id = ?",
                (fact_id,),
            )

    def mark_failed(self, fact_id: int, error: str) -> None:
        """记一次失败。达到 max_attempts 后从队列移除 + facts.enrichment_status='failed'。"""
        with self._db:
            row = self._db.execute(
                "SELECT attempts FROM enrichment_queue WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if row is None:
                # 并发删除 / 从未入队，static fail tolerant
                return
            attempts = int(row[0])
            truncated = error[: self._MAX_ERROR_LEN]
            if attempts >= self._max_attempts:
                self._db.execute(
                    "DELETE FROM enrichment_queue WHERE fact_id = ?",
                    (fact_id,),
                )
                self._db.execute(
                    "UPDATE facts SET enrichment_status = 'failed', "
                    "updated_at = CURRENT_TIMESTAMP WHERE fact_id = ?",
                    (fact_id,),
                )
                log.warning(
                    "enrichment fact_id=%s 达到 max_attempts=%s 放弃；error=%s",
                    fact_id,
                    self._max_attempts,
                    truncated,
                )
            else:
                self._db.execute(
                    "UPDATE enrichment_queue SET last_error = ?, "
                    "last_attempt_at = CURRENT_TIMESTAMP WHERE fact_id = ?",
                    (truncated, fact_id),
                )

    def pending_count(self) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) FROM enrichment_queue"
        ).fetchone()
        return int(row[0])

    def stats(self) -> dict[str, int]:
        """返回 {pending, done, failed} 三段计数。done/failed 来自 facts 表。"""
        pending = self.pending_count()
        done = int(
            self._db.execute(
                "SELECT COUNT(*) FROM facts WHERE enrichment_status = 'done'"
            ).fetchone()[0]
        )
        failed = int(
            self._db.execute(
                "SELECT COUNT(*) FROM facts WHERE enrichment_status = 'failed'"
            ).fetchone()[0]
        )
        return {"pending": pending, "done": done, "failed": failed}
