"""检索业务逻辑（Service 层）。

统一 ``PrismRecall`` 的 search / probe / reason / related / contradict，
返回 frozen dataclass。
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING, Any, Final

from ..lifecycle import list_contradictions
from .types import ContradictionItem, CoOccurrence, SearchHit

if TYPE_CHECKING:
    from ..retriever import RetrievalPipeline, SmartPrefetch

log = logging.getLogger(__name__)

__all__ = ["SearchService"]

_DEFAULT_SEARCH_LIMIT: Final[int] = 5
_DEFAULT_PROBE_LIMIT: Final[int] = 10
_DEFAULT_REASON_LIMIT: Final[int] = 10
_DEFAULT_RELATED_LIMIT: Final[int] = 10
_DEFAULT_CONTRADICT_LIMIT: Final[int] = 10


class SearchService:
    """检索 — Service 层唯一业务入口。

    Args:
        db: 已初始化 schema 的 SQLite 连接
        pipeline: :class:`RetrievalPipeline`（三路融合）
        prefetch: :class:`SmartPrefetch`（LLM 注入 markdown）
    """

    def __init__(
        self,
        db: sqlite3.Connection,
        pipeline: RetrievalPipeline,
        prefetch: SmartPrefetch,
    ) -> None:
        self._db = db
        self._pipeline = pipeline
        self._prefetch = prefetch

    # ─── search ──────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        limit: int = _DEFAULT_SEARCH_LIMIT,
        min_trust: float | None = None,
    ) -> list[SearchHit]:
        results = self._pipeline.recall(query, k=limit, category=category)
        if min_trust is not None:
            filtered_ids = self._filter_by_min_trust(
                [r.fact_id for r in results], min_trust
            )
            results = [r for r in results if r.fact_id in filtered_ids]
        return [
            SearchHit(
                fact_id=r.fact_id,
                content=r.content,
                score=r.score,
                path_scores=dict(r.path_scores),
            )
            for r in results
        ]

    def search_markdown(self, query: str) -> str:
        return self._prefetch.prefetch(query)

    # ─── probe ───────────────────────────────────────────────────────────

    def probe(
        self,
        entity: str,
        *,
        category: str | None = None,
        limit: int = _DEFAULT_PROBE_LIMIT,
    ) -> list[dict[str, Any]]:
        if not entity.strip():
            return []
        if limit < 1:
            return []

        params: list[Any] = [entity.strip()]
        sql = (
            "SELECT f.fact_id, f.content, f.category, f.trust_score "
            "FROM facts f "
            "JOIN fact_entities fe ON fe.fact_id = f.fact_id "
            "JOIN entities e ON e.entity_id = fe.entity_id "
            "WHERE e.name = ? AND f.status = 'active'"
        )
        if category is not None:
            sql += " AND f.category = ?"
            params.append(category)
        sql += " ORDER BY f.fact_id DESC LIMIT ?"
        params.append(int(limit))

        rows = self._db.execute(sql, params).fetchall()
        return [
            {
                "fact_id": int(row["fact_id"]),
                "content": str(row["content"]),
                "category": str(row["category"]) if row["category"] else None,
                "trust_score": float(row["trust_score"]),
            }
            for row in rows
        ]

    # ─── reason ──────────────────────────────────────────────────────────

    def reason(
        self,
        entities: list[str],
        *,
        category: str | None = None,
        limit: int = _DEFAULT_REASON_LIMIT,
    ) -> list[dict[str, Any]]:
        names = [e.strip() for e in entities if e and e.strip()]
        unique_names = sorted(set(names))
        if not unique_names or limit < 1:
            return []

        placeholders = ",".join("?" for _ in unique_names)
        params: list[Any] = list(unique_names)
        sql = (
            "SELECT f.fact_id, f.content, f.category, f.trust_score "
            "FROM facts f "
            "JOIN fact_entities fe ON fe.fact_id = f.fact_id "
            "JOIN entities e ON e.entity_id = fe.entity_id "
            f"WHERE e.name IN ({placeholders}) AND f.status = 'active'"
        )
        if category is not None:
            sql += " AND f.category = ?"
            params.append(category)
        sql += (
            " GROUP BY f.fact_id "
            "HAVING COUNT(DISTINCT e.entity_id) = ? "
            "ORDER BY f.fact_id DESC LIMIT ?"
        )
        params.append(len(unique_names))
        params.append(int(limit))

        rows = self._db.execute(sql, params).fetchall()
        return [
            {
                "fact_id": int(row["fact_id"]),
                "content": str(row["content"]),
                "category": str(row["category"]) if row["category"] else None,
                "trust_score": float(row["trust_score"]),
            }
            for row in rows
        ]

    # ─── related ─────────────────────────────────────────────────────────

    def related(
        self,
        entity: str,
        *,
        category: str | None = None,
        limit: int = _DEFAULT_RELATED_LIMIT,
    ) -> list[CoOccurrence]:
        if not entity.strip():
            return []
        if limit < 1:
            return []

        anchor = entity.strip()
        params: list[Any] = [anchor]
        sql = (
            "SELECT e2.name AS name, COUNT(*) AS co "
            "FROM fact_entities fe1 "
            "JOIN entities e1 ON e1.entity_id = fe1.entity_id "
            "JOIN fact_entities fe2 ON fe2.fact_id = fe1.fact_id "
            "JOIN entities e2 ON e2.entity_id = fe2.entity_id "
            "JOIN facts f ON f.fact_id = fe1.fact_id "
            "WHERE e1.name = ? AND e2.name != e1.name AND f.status = 'active'"
        )
        if category is not None:
            sql += " AND f.category = ?"
            params.append(category)
        sql += (
            " GROUP BY e2.name "
            "ORDER BY co DESC, e2.name ASC LIMIT ?"
        )
        params.append(int(limit))

        rows = self._db.execute(sql, params).fetchall()
        return [
            CoOccurrence(entity=str(row["name"]), co_occurrence=int(row["co"]))
            for row in rows
        ]

    # ─── contradict ──────────────────────────────────────────────────────

    def contradict(
        self,
        *,
        category: str | None = None,
        threshold: float | None = None,
        limit: int = _DEFAULT_CONTRADICT_LIMIT,
    ) -> list[ContradictionItem]:
        raw = list_contradictions(
            self._db,
            category=category,
            threshold=threshold,
            limit=limit,
        )
        return [
            ContradictionItem(
                contradiction_id=int(r["contradiction_id"]),
                fact_a=int(r["fact_a"]),
                fact_b=int(r["fact_b"]),
                score=float(r["score"]),
                detected_at=str(r["detected_at"]),
                content_a=str(r["content_a"]),
                content_b=str(r["content_b"]),
                category_a=r.get("category_a"),
                category_b=r.get("category_b"),
            )
            for r in raw
        ]

    # ─── internal helpers ────────────────────────────────────────────────

    def _filter_by_min_trust(
        self, fact_ids: list[int], min_trust: float
    ) -> set[int]:
        if not fact_ids:
            return set()
        placeholders = ",".join("?" for _ in fact_ids)
        sql = (
            f"SELECT fact_id FROM facts "
            f"WHERE fact_id IN ({placeholders}) AND trust_score >= ?"
        )
        params: list[Any] = [*fact_ids, float(min_trust)]
        rows = self._db.execute(sql, params).fetchall()
        return {int(row["fact_id"]) for row in rows}
