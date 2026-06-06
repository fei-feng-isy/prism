"""Prism 评估模块。

公开 API：
    - :func:`load_cases` / :class:`EvalCase` — jsonl 加载与 schema
    - 指标函数（:func:`precision_at_k` / :func:`recall_at_k` / :func:`reciprocal_rank`）
    - :func:`evaluate_cases` — runner（后端无关）
    - :func:`naive_substring_retriever` — baseline substring retriever
"""

from __future__ import annotations

from .loader import (
    EvalCase,
    EvalFact,
    EvalQuery,
    EvalSetError,
    load_case_from_dict,
    load_cases,
)
from .metrics import (
    AggregateMetrics,
    PerQueryMetrics,
    aggregate,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .runner import (
    EvalReport,
    RetrieveFn,
    evaluate_case,
    evaluate_cases,
    naive_substring_retriever,
)

__all__ = [
    "AggregateMetrics",
    "EvalCase",
    "EvalFact",
    "EvalQuery",
    "EvalReport",
    "EvalSetError",
    "PerQueryMetrics",
    "RetrieveFn",
    "aggregate",
    "evaluate_case",
    "evaluate_cases",
    "load_case_from_dict",
    "load_cases",
    "mean_reciprocal_rank",
    "naive_substring_retriever",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
