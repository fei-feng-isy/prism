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

    def search_markdown(self, query: str, *, limit: int | None = None) -> str:
        return self._prefetch.prefetch(query, limit=limit)

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

        from prism.db import EntitiesRepository
        rows = EntitiesRepository(self._db).get_fact_ids_by_entity(entity, category, limit)
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

        from prism.db import EntitiesRepository
        rows = EntitiesRepository(self._db).get_fact_ids_by_entities(unique_names, category, limit)
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

        from prism.db import EntitiesRepository
        rows = EntitiesRepository(self._db).get_co_occurring_entities(entity, category, limit)
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
        from prism.db import FactsRepository
        return FactsRepository(self._db).filter_ids_by_min_trust(fact_ids, min_trust)
