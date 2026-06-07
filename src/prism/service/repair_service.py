"""向量修复与 enrichment 队列清理业务逻辑（Service 层）。"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from prism.vstore.factory import count_distinct_embedding_models
from prism.vstore.local_numpy import LocalNumpyVectorStore

from .types import (
    EnrichmentDiagnosis,
    EnrichmentFixResult,
    EnrichmentStatusEntry,
    MissingVectorEntry,
    QueueItem,
)

if TYPE_CHECKING:
    from prism.semantic import SemanticBackend

log = logging.getLogger(__name__)

__all__ = ["RepairService", "rebuild_vstore"]

_CONTENT_PREVIEW_LEN: Final[int] = 120
_QUEUE_ITEMS_CAP: Final[int] = 100
_MISSING_VEC_CAP: Final[int] = 100


class RepairService:
    """向量修复与 enrichment 队列清理。

    Args:
        db: 已初始化 schema 的 SQLite 连接
        semantic: :class:`SemanticBackend` 实例
    """

    def __init__(self, db: sqlite3.Connection, semantic: SemanticBackend) -> None:
        self._db = db
        self._semantic = semantic

    def enrichment_diagnose(self) -> EnrichmentDiagnosis:
        queue_count = self._count("SELECT COUNT(*) FROM enrichment_queue")

        status_rows = self._db.execute(
            "SELECT enrichment_status, COUNT(*) AS cnt, "
            "SUM(CASE WHEN semantic_vector IS NULL THEN 1 ELSE 0 END) AS null_vec "
            "FROM facts GROUP BY enrichment_status"
        ).fetchall()
        status_distribution = tuple(
            EnrichmentStatusEntry(
                enrichment_status=str(r["enrichment_status"]),
                count=int(r["cnt"]),
                null_vector=int(r["null_vec"]),
            )
            for r in status_rows
        )

        q_rows = self._db.execute(
            "SELECT fact_id, attempts, last_error, enqueued_at, last_attempt_at "
            "FROM enrichment_queue ORDER BY enqueued_at ASC LIMIT ?",
            (_QUEUE_ITEMS_CAP,),
        ).fetchall()
        queue_items = tuple(
            QueueItem(
                fact_id=int(r["fact_id"]),
                attempts=int(r["attempts"]),
                last_error=r["last_error"],
                enqueued_at=str(r["enqueued_at"]) if r["enqueued_at"] else None,
                last_attempt_at=str(r["last_attempt_at"]) if r["last_attempt_at"] else None,
            )
            for r in q_rows
        )

        missing_total = self._count(
            "SELECT COUNT(*) FROM facts WHERE semantic_vector IS NULL AND status = 'active'"
        )
        mv_rows = self._db.execute(
            "SELECT fact_id, content FROM facts "
            "WHERE semantic_vector IS NULL AND status = 'active' "
            "ORDER BY fact_id ASC LIMIT ?",
            (_MISSING_VEC_CAP,),
        ).fetchall()
        missing_vectors = tuple(
            MissingVectorEntry(
                fact_id=int(r["fact_id"]),
                content=str(r["content"])[:_CONTENT_PREVIEW_LEN],
            )
            for r in mv_rows
        )

        return EnrichmentDiagnosis(
            queue_count=queue_count,
            status_distribution=status_distribution,
            queue_items=queue_items,
            missing_vectors=missing_vectors,
            missing_vector_count=missing_total,
        )

    def enrichment_fix(self, *, dry_run: bool = False) -> EnrichmentFixResult:
        sem = self._semantic
        sem_available = sem.is_available()

        missing = self._db.execute(
            "SELECT fact_id, content FROM facts "
            "WHERE semantic_vector IS NULL AND status = 'active' "
            "ORDER BY fact_id ASC"
        ).fetchall()

        pending_count = self._count(
            "SELECT COUNT(*) FROM facts WHERE enrichment_status = 'pending'"
        )
        queue_count = self._count("SELECT COUNT(*) FROM enrichment_queue")

        if dry_run:
            return EnrichmentFixResult(
                dry_run=True,
                semantic_available=sem_available,
                to_embed=len(missing),
                pending_to_clear=pending_count,
                queue_to_clear=queue_count,
            )

        embedded = 0
        embed_failed = 0
        if missing and sem_available:
            batch_size = 64
            model_name = sem.name
            for start in range(0, len(missing), batch_size):
                chunk = missing[start : start + batch_size]
                texts = [str(r["content"]) for r in chunk]
                try:
                    mat = sem.encode_batch(texts)
                except Exception as e:
                    embed_failed += len(chunk)
                    log.warning(
                        "enrichment_fix encode_batch 失败 [%d, %d): %s",
                        start, start + len(chunk), e,
                    )
                    continue
                for row, vec in zip(chunk, mat, strict=True):
                    try:
                        self._db.execute(
                            "UPDATE facts SET semantic_vector = ?, embedding_model = ? "
                            "WHERE fact_id = ?",
                            (
                                vec.astype(np.float32).tobytes(),
                                model_name,
                                int(row["fact_id"]),
                            ),
                        )
                        embedded += 1
                    except Exception as e:
                        embed_failed += 1
                        log.warning(
                            "enrichment_fix UPDATE fact_id=%s 失败: %s",
                            row["fact_id"], e,
                        )

        self._db.execute("BEGIN")
        try:
            cleared = self._db.execute(
                "UPDATE facts SET enrichment_status = 'done', "
                "updated_at = CURRENT_TIMESTAMP "
                "WHERE enrichment_status = 'pending'"
            ).rowcount
            q_deleted = self._db.execute("DELETE FROM enrichment_queue").rowcount
            self._db.execute("COMMIT")
        except Exception:
            self._db.execute("ROLLBACK")
            raise

        db_path_str = self._db_path()
        vstore_path: Path | None = None
        if db_path_str != ":memory:":
            vstore_path = Path(db_path_str).with_suffix(".vstore.npz")

        vstore_rebuilt = rebuild_vstore(
            self._db,
            vstore_path=vstore_path,
            dim=sem.dim,
            new_model=sem.name,
        )

        vstore_vectors = 0
        if vstore_rebuilt and vstore_path is not None:
            try:
                data = np.load(str(vstore_path))
                vstore_vectors = int(data["fact_ids"].shape[0])
            except Exception:
                pass

        return EnrichmentFixResult(
            dry_run=False,
            semantic_available=sem_available,
            embedded=embedded,
            embed_failed=embed_failed,
            pending_cleared=cleared,
            queue_cleared=q_deleted,
            vstore_rebuilt=vstore_rebuilt,
            vstore_vectors=vstore_vectors,
        )

    def _db_path(self) -> str:
        try:
            for row in self._db.execute("PRAGMA database_list").fetchall():
                if row["name"] == "main":
                    return str(row["file"]) or ":memory:"
        except sqlite3.Error as e:
            log.warning("读取 db_path 失败：%s", e)
        return ":memory:"

    def _count(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        row = self._db.execute(sql, params).fetchone()
        return int(row[0]) if row else 0


def rebuild_vstore(
    db: sqlite3.Connection,
    *,
    vstore_path: Path | None,
    dim: int,
    new_model: str,
) -> bool:
    rows = db.execute(
        "SELECT fact_id, semantic_vector FROM facts "
        "WHERE status='active' AND embedding_model = ? "
        "AND semantic_vector IS NOT NULL",
        (new_model,),
    ).fetchall()

    if not rows:
        return False

    vectors: list[tuple[int, np.ndarray]] = []
    for r in rows:
        blob = r["semantic_vector"]
        v = np.frombuffer(blob, dtype=np.float32)
        if v.shape[0] != dim:
            log.warning(
                "fact_id=%s vector dim=%d 与目标 %d 不符；跳过",
                r["fact_id"], v.shape[0], dim,
            )
            continue
        vectors.append((int(r["fact_id"]), v.copy()))

    if not vectors:
        return False

    vstore = LocalNumpyVectorStore(
        dim=dim,
        path=str(vstore_path) if vstore_path is not None else None,
        load=False,
        initial_capacity=max(64, len(vectors)),
    )
    for fact_id, vec in vectors:
        vstore.add(fact_id, vec)
    vstore.persist()
    return True
