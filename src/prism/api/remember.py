"""``prism_remember`` 用户显式写入 API。

薄壳：action dispatch + 参数校验 + dataclass→dict 转换。
业务逻辑委托 :class:`~prism.service.FactService`。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Final

from ..service.fact_service import FactService

__all__ = ["PrismRemember"]

_DEFERRED_ACTIONS: Final[frozenset[str]] = frozenset({"update"})


class PrismRemember:
    """``prism_remember`` 工具入口（薄壳）。

    Args:
        fact_service: 已构造好的 :class:`FactService`
    """

    def __init__(self, fact_service: FactService) -> None:
        self._svc = fact_service

    def __call__(self, action: str, **kwargs: Any) -> dict[str, Any]:
        if action == "add":
            return self.add(**kwargs)
        if action == "remove":
            return self.remove(**kwargs)
        if action == "helpful":
            return self.helpful(**kwargs)
        if action == "unhelpful":
            return self.unhelpful(**kwargs)
        if action in _DEFERRED_ACTIONS:
            raise NotImplementedError(f"prism_remember(action={action!r}) 暂未实现")
        raise ValueError(f"未知 action：{action!r}")

    def add(
        self,
        content: str,
        category: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        r = self._svc.add(content, category=category, metadata=metadata)
        return {
            "fact_id": r.fact_id,
            "is_new": r.is_new,
            "entities": list(r.entities),
            "category": r.category,
        }

    def remove(self, fact_id: int) -> dict[str, Any]:
        if not isinstance(fact_id, int) or fact_id <= 0:
            raise ValueError(f"fact_id 必须是正整数：{fact_id!r}")
        r = self._svc.remove(fact_id)
        return {"fact_id": r.fact_id, "archived": r.archived}

    def helpful(self, fact_id: int) -> dict[str, Any]:
        r = self._svc.helpful(fact_id)
        return asdict(r)

    def unhelpful(self, fact_id: int) -> dict[str, Any]:
        r = self._svc.unhelpful(fact_id)
        return asdict(r)
