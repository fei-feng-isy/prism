"""智能预取。

``SmartPrefetch`` 把 :class:`RetrievalPipeline.recall` 结果格式化为 LLM 注入
markdown，提供 warm-up（模型预加载）和冷启动短路。

结果按 entity-hit（结构化命中）和 semantic-only 分两组渲染。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from prism.semantic import SemanticUnavailable

if TYPE_CHECKING:
    from .fusion import RecallResult, RetrievalPipeline

log = logging.getLogger(__name__)

__all__ = ["SmartPrefetch"]


# entity 路径归一化分 ≥ 此阈值视为「结构化命中」（标 🎯 而非 🔍）
# 0.0 不合理（min-max 归一化后全等会 = 1.0 全 entity-hit）；
# 0.5 是经验值：单个共享 entity 在 2-3 entity fact 上 Jaccard ≈ 0.3-0.5
_ENTITY_HIT_THRESHOLD: Final[float] = 0.5

# warm-up 用的探针字符串。空串某些 backend 会 reject，给个最短合法中文
_WARMUP_PROBE: Final[str] = "预热"

# 注入 markdown 的固定 header
_HEADER: Final[str] = "## Prism 记忆召回（自动预取）"
_ENTITY_TAG: Final[str] = "🎯 实体命中："
_SEMANTIC_TAG: Final[str] = "🔍 语义相关："


@dataclass(frozen=True, slots=True)
class _FormattedGroup:
    """格式化时按命中类型分组的中间结构（仅本模块使用）。"""

    tag: str
    results: list[RecallResult]


class SmartPrefetch:
    """把 :class:`RetrievalPipeline.recall` 结果格式化为 LLM 注入用 markdown。

    Args:
        pipeline: 已配置好的 RetrievalPipeline 实例
        max_results: 注入总条数上限（默认 5）
        entity_hit_threshold: 视为 entity-hit 的归一化 entity 分阈值
            （默认 ``_ENTITY_HIT_THRESHOLD`` = 0.5）

    Note:
        构造**不**触发任何 IO；warm-up 需调用方显式调 :meth:`warmup`
        （通常在 provider initialize 钩子）。
    """

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        *,
        max_results: int = 5,
        entity_hit_threshold: float = _ENTITY_HIT_THRESHOLD,
    ) -> None:
        if max_results < 1:
            raise ValueError(f"max_results 必须 ≥ 1：got {max_results}")
        if not 0.0 <= entity_hit_threshold <= 1.0:
            raise ValueError(
                f"entity_hit_threshold 必须 ∈ [0, 1]：got {entity_hit_threshold}"
            )
        self._pipeline = pipeline
        self._max_results = max_results
        self._entity_threshold = entity_hit_threshold
        self._warmed: bool = False

    @property
    def warmed(self) -> bool:
        return self._warmed

    @warmed.setter
    def warmed(self, value: bool) -> None:
        self._warmed = value

    # ─── 公共入口 ────────────────────────────────────────────────────────

    def prefetch(self, query: str) -> str:
        """按 query 走融合 recall 并格式化为 markdown 注入片段。

        Args:
            query: 用户查询文本（空串 / 仅空白 → 返回空串）

        Returns:
            ``""`` 表示无可注入（query 空 / 冷启动 / 三路均无候选）；
            否则返回完整 markdown 片段（含 header + 分组）
        """
        if not query.strip():
            return ""

        # 冷启动短路：vstore 空且 DB 无 fact → 无可能召回
        if self._is_cold_start():
            return ""

        results = self._pipeline.recall(query, k=self._max_results)
        if not results:
            return ""

        return self._format(results)

    def warmup(self) -> bool:
        """触发一次空 encode 让 sentence-transformers 模型 lazy load。

        失败（degraded backend / 加载异常）静默返 False — warm-up 是优化而非
        正确性要求，prefetch 仍能在 degraded 模式下走 fts+entity。

        Returns:
            ``True`` 表示 encode 成功（含 degraded backend 返 False 的情况）。
            幂等：多次调用只触发一次实际 encode。
        """
        if self._warmed:
            return True
        sem = self._pipeline.semantic
        if not sem.is_available():
            # degraded backend 不需要 warm-up（encode 永不被调）
            self._warmed = True
            return False
        try:
            sem.encode(_WARMUP_PROBE)
        except SemanticUnavailable as e:
            log.warning("smart_prefetch warmup encode 失败（degraded fallback）：%s", e)
            self._warmed = True  # 不重试 — degraded 路径
            return False
        except Exception as e:
            log.warning("smart_prefetch warmup encode 异常：%s", e)
            self._warmed = True  # 同上不重试
            return False
        self._warmed = True
        return True

    # ─── 内部 helpers ────────────────────────────────────────────────────

    def _is_cold_start(self) -> bool:
        """vstore 空 + DB 无 active fact → True。

        两个条件都查：仅看 vstore 空会误判（degraded backend 永不写 vstore
        但 fts/entity 路径仍能召回）。
        """
        if len(self._pipeline.vstore) > 0:
            return False
        row = self._pipeline.db.execute(
            "SELECT 1 FROM facts WHERE status = 'active' LIMIT 1"
        ).fetchone()
        return row is None

    def _format(self, results: list[RecallResult]) -> str:
        """按 entity-hit 与 semantic-only 分两组，渲染 markdown。"""
        entity_hits: list[RecallResult] = []
        semantic_hits: list[RecallResult] = []
        for r in results:
            if r.path_scores.get("entity", 0.0) >= self._entity_threshold:
                entity_hits.append(r)
            else:
                semantic_hits.append(r)

        lines = [_HEADER, ""]
        if entity_hits:
            lines.append(_ENTITY_TAG)
            lines.extend(self._render_items(entity_hits))
            if semantic_hits:
                lines.append("")  # 两组之间空行
        if semantic_hits:
            lines.append(_SEMANTIC_TAG)
            lines.extend(self._render_items(semantic_hits))
        return "\n".join(lines)

    @staticmethod
    def _render_items(items: list[RecallResult]) -> list[str]:
        """``  • [score] content`` 单行渲染；score 保留两位小数。"""
        return [f"  • [{r.score:.2f}] {r.content}" for r in items]
