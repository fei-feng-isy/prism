"""评估 runner — 后端无关。

给一个 ``retrieve(query, k, facts) -> Sequence[int]`` 的可调用，对评估集中
每条 query 计算 P@k / R@k / RR / must_include / must_exclude。
测试用的 baseline 是 substring matching，可换成真正的 ``prism_recall`` 调用。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .loader import EvalCase, EvalFact
from .metrics import (
    AggregateMetrics,
    PerQueryMetrics,
    aggregate,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "EvalReport",
    "RetrieveFn",
    "evaluate_case",
    "evaluate_cases",
    "naive_substring_retriever",
]


class RetrieveFn(Protocol):
    """检索函数签名 — 给 query 和当前 case 的 facts，返回按相关度降序的 fact 下标。"""

    def __call__(
        self,
        query: str,
        k: int,
        facts: Sequence[EvalFact],
    ) -> Sequence[int]: ...


def naive_substring_retriever(
    query: str,
    k: int,
    facts: Sequence[EvalFact],
) -> list[int]:
    """Baseline 检索：query 拆词后做 substring 计数排序。

    设计意图：让评估集"能跑出非零分数"以校验 runner / 指标管线正常；
    不追求 recall 质量。语义层上线后由 ``prism_recall`` 替换。
    """
    q = query.strip()
    # 中英文都做：英文按空白拆词；中文用 2-gram。两者并入候选 token。
    tokens: list[str] = []
    for w in q.split():
        if any("一" <= ch <= "鿿" for ch in w):
            # 中文段落 → 2-gram
            for i in range(len(w) - 1):
                tokens.append(w[i : i + 2])
        else:
            tokens.append(w.lower())
    if not tokens:
        return []
    scored: list[tuple[int, int]] = []
    for idx, fact in enumerate(facts):
        content = fact.content
        lower = content.lower()
        hits = 0
        for tok in tokens:
            if tok in content or tok in lower:
                hits += 1
        if hits > 0:
            scored.append((hits, idx))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [idx for _hits, idx in scored[:k]]


def evaluate_case(
    case: EvalCase,
    retrieve: RetrieveFn,
) -> list[PerQueryMetrics]:
    """对单条用例的所有 query 跑评估。"""
    results: list[PerQueryMetrics] = []
    for q in case.queries:
        actual_seq = list(retrieve(q.query, q.k, case.setup_facts))
        actual_top = actual_seq[: q.k]
        expected_set = set(q.expected_ids)
        results.append(
            PerQueryMetrics(
                query=q.query,
                k=q.k,
                precision_at_k=precision_at_k(actual_top, expected_set, q.k),
                recall_at_k=recall_at_k(actual_top, expected_set, q.k),
                reciprocal_rank=reciprocal_rank(actual_seq, expected_set),
                actual_ids=tuple(actual_top),
                expected_ids=tuple(q.expected_ids),
                must_include_satisfied=set(q.must_include).issubset(set(actual_top)),
                must_exclude_satisfied=not (set(q.must_exclude) & set(actual_top)),
            )
        )
    return results


class EvalReport:
    """跨 case 的汇总报告，保留 per-query 明细。"""

    __slots__ = ("cases", "per_query", "summary")

    def __init__(
        self,
        cases: list[EvalCase],
        per_query: list[PerQueryMetrics],
    ) -> None:
        self.cases = cases
        self.per_query = per_query
        self.summary: AggregateMetrics = aggregate(per_query)


def evaluate_cases(
    cases: Sequence[EvalCase],
    retrieve: RetrieveFn,
) -> EvalReport:
    """跑完所有 cases，返回 :class:`EvalReport`。"""
    all_q: list[PerQueryMetrics] = []
    for c in cases:
        all_q.extend(evaluate_case(c, retrieve))
    return EvalReport(list(cases), all_q)
