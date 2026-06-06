"""事实 CRUD 业务逻辑（Service 层）。

统一 ``PrismRemember`` (add/remove/helpful/unhelpful) +
``PrismAdmin`` (list/archive/restore) + ``cli/memory.py`` (list/show/edit/remove)
的业务逻辑，返回 frozen dataclass。

API 层和 CLI 层各自负责 dict/markdown 格式化。
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Mapping
from typing import Any, Final

from ..lifecycle import FeedbackSignal, apply_feedback
from ..mirror import MIRROR_SOURCE_BUILTIN, MIRROR_SOURCE_USER, PrismMirror
from .types import (
    ArchiveResult,
    FactDetail,
    FactResult,
    FactSummary,
    FeedbackResult,
    ListResult,
    RemoveResult,
    RestoreResult,
)

log = logging.getLogger(__name__)

__all__ = ["FactService"]

_LIST_DEFAULT_LIMIT: Final[int] = 50
_LIST_HARD_CAP: Final[int] = 500
_LIST_STATUS_VALUES: Final[frozenset[str]] = frozenset({"active", "archived"})
_ARCHIVE_REASON_MANUAL: Final[str] = "manual"


class FactService:
    """事实 CRUD — Service 层唯一业务入口。

    Args:
        db: 已初始化 schema 的 SQLite 连接
        mirror: 已构造好的 :class:`PrismMirror`（持有 conn / bank / vstore 等写路径）
    """

    def __init__(self, db: sqlite3.Connection, mirror: PrismMirror) -> None:
        self._db = db
        self._mirror = mirror

    # ─── 写入 ────────────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        category: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        *,
        source: str = MIRROR_SOURCE_USER,
    ) -> FactResult:
        merged: dict[str, Any] = dict(metadata or {})
        if category is not None:
            merged["category"] = category

        result = self._mirror.mirror_add(
            content,
            metadata=merged,
            source=source,
            target=None,
        )
        if result is None:
            raise ValueError("content 不能为空")

        return FactResult(
            fact_id=result.fact_id,
            is_new=result.is_new,
            entities=tuple(result.entities),
            category=result.category,
        )

    def edit(
        self,
        fact_id: int,
        content: str,
        category: str | None = None,
    ) -> FactResult:
        row = self._db.execute(
            "SELECT content FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        if row is None:
            raise LookupError(f"fact_id={fact_id} 不存在")

        metadata: dict[str, Any] = {"supersedes_id": fact_id}
        if category is not None:
            metadata["category"] = category

        result = self._mirror.mirror_replace(
            content=content,
            metadata=metadata,
            source=MIRROR_SOURCE_BUILTIN,
            target="memory",
        )
        if result is None:
            raise LookupError(
                f"mirror_replace 未生效（fact_id={fact_id} 可能已被并发删除）"
            )

        return FactResult(
            fact_id=result.fact_id,
            is_new=result.is_new,
            entities=tuple(result.entities),
            category=result.category,
        )

    def remove(self, fact_id: int, reason: str = "manual") -> RemoveResult:
        if not isinstance(fact_id, int) or fact_id <= 0:
            raise ValueError(f"fact_id 必须是正整数：{fact_id!r}")
        result = self._mirror.mirror_remove(fact_id=fact_id, reason=reason)
        if result is None:
            return RemoveResult(fact_id=fact_id, archived=False)
        return RemoveResult(fact_id=result.fact_id, archived=True)

    def helpful(self, fact_id: int) -> FeedbackResult:
        return self._apply_feedback(fact_id, "helpful")

    def unhelpful(self, fact_id: int) -> FeedbackResult:
        return self._apply_feedback(fact_id, "unhelpful")

    def _apply_feedback(
        self, fact_id: int, signal: FeedbackSignal
    ) -> FeedbackResult:
        result = apply_feedback(self._db, fact_id, signal)
        if result is None:
            return FeedbackResult(
                fact_id=fact_id,
                applied=False,
                new_trust_score=None,
                new_helpful_count=None,
            )
        return FeedbackResult(
            fact_id=result.fact_id,
            applied=True,
            new_trust_score=result.new_trust_score,
            new_helpful_count=result.new_helpful_count,
        )

    # ─── 查询 ────────────────────────────────────────────────────────────

    def list(
        self,
        *,
        category: str | None = None,
        status: str | None = "active",
        mirror_source: str | None = None,
        limit: int = _LIST_DEFAULT_LIMIT,
        offset: int = 0,
    ) -> ListResult:
        if not isinstance(limit, int) or limit < 1:
            raise ValueError(f"limit 必须是正整数：{limit!r}")
        if status is not None and status not in _LIST_STATUS_VALUES:
            raise ValueError(
                f"status 取值非法 {status!r}（合法：{sorted(_LIST_STATUS_VALUES)} 或 None）"
            )
        effective_limit = min(limit, _LIST_HARD_CAP)

        clauses: list[str] = []
        params: list[Any] = []
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if mirror_source is not None:
            clauses.append("mirror_source = ?")
            params.append(mirror_source)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        sql = (
            "SELECT fact_id, content, category, status, trust_score, "
            "helpful_count, mirror_source, created_at, archived_at, archive_reason "
            f"FROM facts{where} ORDER BY created_at DESC, fact_id DESC "
            "LIMIT ? OFFSET ?"
        )
        rows = self._db.execute(
            sql, (*params, effective_limit + 1, offset)
        ).fetchall()
        truncated = len(rows) > effective_limit
        rows = rows[:effective_limit]

        facts = tuple(
            FactSummary(
                fact_id=int(r["fact_id"]),
                content=str(r["content"]),
                category=str(r["category"]),
                status=str(r["status"]),
                trust_score=float(r["trust_score"]),
                helpful_count=int(r["helpful_count"]),
                mirror_source=r["mirror_source"],
                created_at=str(r["created_at"]),
                archived_at=(
                    str(r["archived_at"]) if r["archived_at"] is not None else None
                ),
                archive_reason=(
                    str(r["archive_reason"])
                    if r["archive_reason"] is not None
                    else None
                ),
            )
            for r in rows
        )

        return ListResult(
            facts=facts,
            count=len(facts),
            truncated=truncated,
            filter={
                "category": category,
                "status": status,
                "mirror_source": mirror_source,
                "limit": effective_limit,
                "offset": offset,
            },
        )

    def show(self, fact_id: int) -> FactDetail | None:
        row = self._db.execute(
            "SELECT fact_id, content, category, status, mirror_source, mirror_target, "
            "supersedes_id, trust_score, helpful_count, retrieval_count, "
            "archived_at, archive_reason, enrichment_status, created_at "
            "FROM facts WHERE fact_id = ?",
            (fact_id,),
        ).fetchone()
        if row is None:
            return None

        entity_rows = self._db.execute(
            "SELECT e.name FROM fact_entities fe "
            "JOIN entities e ON e.entity_id = fe.entity_id "
            "WHERE fe.fact_id = ? ORDER BY e.name",
            (fact_id,),
        ).fetchall()
        entities = tuple(str(r["name"]) for r in entity_rows)

        return FactDetail(
            fact_id=int(row["fact_id"]),
            content=str(row["content"] or ""),
            category=str(row["category"] or ""),
            status=str(row["status"] or ""),
            trust_score=float(row["trust_score"]),
            helpful_count=int(row["helpful_count"]),
            retrieval_count=int(row["retrieval_count"]),
            mirror_source=row["mirror_source"],
            mirror_target=row["mirror_target"],
            supersedes_id=row["supersedes_id"],
            enrichment_status=str(row["enrichment_status"] or ""),
            archived_at=row["archived_at"],
            archive_reason=row["archive_reason"],
            created_at=str(row["created_at"]),
            entities=entities,
        )

    # ─── 归档管理 ────────────────────────────────────────────────────────

    def archive(
        self, fact_id: int, reason: str = _ARCHIVE_REASON_MANUAL
    ) -> ArchiveResult:
        if not isinstance(fact_id, int) or fact_id <= 0:
            raise ValueError(f"fact_id 必须是正整数：{fact_id!r}")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"reason 必须是非空字符串：{reason!r}")

        result = self._mirror.mirror_remove(fact_id=fact_id, reason=reason)
        archived = result is not None
        if not archived:
            row = self._db.execute(
                "SELECT status FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is not None and str(row["status"]) == "archived":
                archived = True
        return ArchiveResult(fact_id=fact_id, archived=archived, reason=reason)

    def restore(self, fact_id: int) -> RestoreResult:
        if not isinstance(fact_id, int) or fact_id <= 0:
            raise ValueError(f"fact_id 必须是正整数：{fact_id!r}")

        row = self._db.execute(
            "SELECT status, category, hrr_vector, semantic_vector "
            "FROM facts WHERE fact_id = ?",
            (fact_id,),
        ).fetchone()
        if row is None:
            return RestoreResult(fact_id=fact_id, restored=False, category=None)

        category = str(row["category"]) if row["category"] is not None else None

        if str(row["status"]) == "active":
            return RestoreResult(fact_id=fact_id, restored=False, category=category)

        with self._db:
            self._db.execute(
                "UPDATE facts SET status = 'active', "
                "archived_at = NULL, archive_reason = NULL "
                "WHERE fact_id = ?",
                (fact_id,),
            )

        if category is not None:
            self._restore_vectors(
                fact_id, category, row["hrr_vector"], row["semantic_vector"]
            )

        return RestoreResult(fact_id=fact_id, restored=True, category=category)

    def _restore_vectors(
        self,
        fact_id: int,
        category: str,
        hrr_blob: bytes | None,
        semantic_blob: bytes | None,
    ) -> None:
        self._mirror.restore_vectors(fact_id, category, hrr_blob, semantic_blob)
