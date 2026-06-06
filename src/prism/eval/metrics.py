"""检索质量指标。

所有函数都是纯函数：以 ``actual_ids`` (按相关度降序的检索结果) 和
``expected_ids`` (ground truth 集合) 作为输入，不耦合任何后端。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "AggregateMetrics",
    "PerQueryMetrics",
    "aggregate",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k 必须是 >= 1 的整数，实际：{k!r}")


def precision_at_k(
    actual: Sequence[int],
    expected: Iterable[int],
    k: int,
) -> float:
    """P@k = |actual[:k] ∩ expected| / k

    重复出现的 id 只计一次（按 set 去重）；分母固定为 k。
    空 expected 视为未定义 → 0.0。
    """
    _validate_k(k)
    exp = set(expected)
    if not exp:
        return 0.0
    top = list(actual)[:k]
    hits = len(set(top) & exp)
    return hits / k


def recall_at_k(
    actual: Sequence[int],
    expected: Iterable[int],
    k: int,
) -> float:
    """R@k = |actual[:k] ∩ expected| / |expected|"""
    _validate_k(k)
    exp = set(expected)
    if not exp:
        return 0.0
    top = list(actual)[:k]
    hits = len(set(top) & exp)
    return hits / len(exp)


def reciprocal_rank(
    actual: Sequence[int],
    expected: Iterable[int],
) -> float:
    """单条查询的 1/rank（首个相关项的排名倒数）；无命中返回 0。"""
    exp = set(expected)
    if not exp:
        return 0.0
    for idx, fact_id in enumerate(actual, start=1):
        if fact_id in exp:
            return 1.0 / idx
    return 0.0


def mean_reciprocal_rank(per_query_rr: Iterable[float]) -> float:
    """MRR = average(reciprocal_rank(q) for q in queries)"""
    values = list(per_query_rr)
    if not values:
        return 0.0
    return sum(values) / len(values)


@dataclass(frozen=True, slots=True)
class PerQueryMetrics:
    """单条 query 的指标快照（评估 runner 输出）。"""

    query: str
    k: int
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    actual_ids: tuple[int, ...]
    expected_ids: tuple[int, ...]
    must_include_satisfied: bool
    must_exclude_satisfied: bool


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """跨 query 汇总指标。"""

    n_queries: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mrr: float
    empty_rate: float          # 返回 0 结果的 query 占比
    must_include_pass_rate: float
    must_exclude_pass_rate: float


def aggregate(per_query: Iterable[PerQueryMetrics]) -> AggregateMetrics:
    """对一组单 query 指标做平均聚合。空输入返回全 0。"""
    items = list(per_query)
    n = len(items)
    if n == 0:
        return AggregateMetrics(
            n_queries=0,
            mean_precision_at_k=0.0,
            mean_recall_at_k=0.0,
            mrr=0.0,
            empty_rate=0.0,
            must_include_pass_rate=0.0,
            must_exclude_pass_rate=0.0,
        )
    return AggregateMetrics(
        n_queries=n,
        mean_precision_at_k=sum(m.precision_at_k for m in items) / n,
        mean_recall_at_k=sum(m.recall_at_k for m in items) / n,
        mrr=sum(m.reciprocal_rank for m in items) / n,
        empty_rate=sum(1 for m in items if not m.actual_ids) / n,
        must_include_pass_rate=sum(1 for m in items if m.must_include_satisfied) / n,
        must_exclude_pass_rate=sum(1 for m in items if m.must_exclude_satisfied) / n,
    )
