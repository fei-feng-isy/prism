"""``prism_admin`` 运维 API。

薄壳：action dispatch + dataclass→dict 转换。
业务逻辑委托 :class:`~prism.service.AdminService` 和 :class:`~prism.service.FactService`。
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Final

from ..service.admin_service import AdminService
from ..service.fact_service import FactService

log = logging.getLogger(__name__)

__all__ = ["PrismAdmin"]

_DEFERRED_ACTIONS: Final[frozenset[str]] = frozenset()
_LIST_DEFAULT_LIMIT: Final[int] = 50


class PrismAdmin:
    """``prism_admin`` 工具入口（薄壳）。

    Args:
        admin_service: 已构造好的 :class:`AdminService`
        fact_service: 已构造好的 :class:`FactService`
    """

    def __init__(
        self,
        admin_service: AdminService,
        fact_service: FactService,
    ) -> None:
        self._admin = admin_service
        self._fact = fact_service

    def __call__(self, action: str, **kwargs: Any) -> Any:
        if action == "stats":
            return self.stats(**kwargs)
        if action == "list":
            return self.list(**kwargs)
        if action == "archive":
            return self.archive(**kwargs)
        if action == "restore":
            return self.restore(**kwargs)
        if action == "enrichment_diagnose":
            return self.enrichment_diagnose(**kwargs)
        if action == "enrichment_fix":
            return self.enrichment_fix(**kwargs)
        if action in _DEFERRED_ACTIONS:
            raise NotImplementedError(
                f"prism_admin(action={action!r}) 暂未实现"
            )
        raise ValueError(f"未知 action：{action!r}")

    def stats(self, category: str | None = None) -> dict[str, Any]:
        return self._admin.stats(category=category)

    def list(
        self,
        category: str | None = None,
        status: str | None = "active",
        limit: int = _LIST_DEFAULT_LIMIT,
    ) -> dict[str, Any]:
        result = self._fact.list(category=category, status=status, limit=limit)
        return {
            "facts": [
                {
                    "fact_id": f.fact_id,
                    "content": f.content,
                    "category": f.category,
                    "status": f.status,
                    "trust_score": f.trust_score,
                    "helpful_count": f.helpful_count,
                    "created_at": f.created_at,
                    "archived_at": f.archived_at,
                    "archive_reason": f.archive_reason,
                }
                for f in result.facts
            ],
            "count": result.count,
            "truncated": result.truncated,
            "filter": {
                "category": category,
                "status": status,
                "limit": result.filter["limit"],
            },
        }

    def archive(
        self, fact_id: int, reason: str = "manual"
    ) -> dict[str, Any]:
        r = self._fact.archive(fact_id, reason=reason)
        return asdict(r)

    def restore(self, fact_id: int) -> dict[str, Any]:
        r = self._fact.restore(fact_id)
        return asdict(r)

    def enrichment_diagnose(self) -> dict[str, Any]:
        r = self._admin.enrichment_diagnose()
        return {
            "queue_count": r.queue_count,
            "status_distribution": [asdict(s) for s in r.status_distribution],
            "queue_items": [asdict(q) for q in r.queue_items],
            "missing_vectors": [asdict(m) for m in r.missing_vectors],
            "missing_vector_count": r.missing_vector_count,
        }

    def enrichment_fix(self, *, dry_run: bool = False) -> dict[str, Any]:
        r = self._admin.enrichment_fix(dry_run=dry_run)
        return asdict(r)
