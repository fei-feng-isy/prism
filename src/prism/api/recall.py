"""``prism_recall`` 用户检索 API。

薄壳：action dispatch + dataclass→dict/str 转换。
业务逻辑委托 :class:`~prism.service.SearchService`。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Final

from ..service.search_service import SearchService

__all__ = ["PrismRecall"]

_DEFERRED_ACTIONS: Final[frozenset[str]] = frozenset()


class PrismRecall:
    """``prism_recall`` 工具入口（薄壳）。

    Args:
        search_service: 已构造好的 :class:`SearchService`
    """

    def __init__(self, search_service: SearchService) -> None:
        self._svc = search_service

    def __call__(self, action: str, **kwargs: Any) -> Any:
        if action == "search":
            return self.search(**kwargs)
        if action == "probe":
            return self.probe(**kwargs)
        if action == "reason":
            return self.reason(**kwargs)
        if action == "related":
            return self.related(**kwargs)
        if action == "contradict":
            return self.contradict(**kwargs)
        if action in _DEFERRED_ACTIONS:
            raise NotImplementedError(
                f"prism_recall(action={action!r}) 暂未实现"
            )
        raise ValueError(f"未知 action：{action!r}")

    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        limit: int = 5,
        min_trust: float | None = None,
        as_markdown: bool = True,
    ) -> str | list[dict[str, Any]]:
        if as_markdown:
            if category is not None or min_trust is not None:
                raise ValueError(
                    "as_markdown=True 时不支持 category/min_trust 过滤；"
                    "请用 as_markdown=False 走结构化路径"
                )
            return self._svc.search_markdown(query)

        hits = self._svc.search(
            query, category=category, limit=limit, min_trust=min_trust
        )
        return [asdict(h) for h in hits]

    def probe(
        self,
        entity: str,
        *,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return self._svc.probe(entity, category=category, limit=limit)

    def reason(
        self,
        entities: list[str],
        *,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return self._svc.reason(entities, category=category, limit=limit)

    def related(
        self,
        entity: str,
        *,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        hits = self._svc.related(entity, category=category, limit=limit)
        return [asdict(h) for h in hits]

    def contradict(
        self,
        *,
        category: str | None = None,
        threshold: float | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        hits = self._svc.contradict(
            category=category, threshold=threshold, limit=limit
        )
        return [asdict(h) for h in hits]
