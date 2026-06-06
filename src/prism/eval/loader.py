"""评估集 jsonl loader。

文件格式：每行一条独立 JSON 对象（jsonl），空行 / 全空白行被跳过。
每条用例包含若干 ``setup_facts``（按列表顺序为 fact_id：0、1、2…）以及若干
``queries``；``expected_ids`` / ``must_include`` / ``must_exclude`` 均为
``setup_facts`` 列表的下标，便于评估器在不依赖 DB 状态下做 ranking 比对。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "EvalCase",
    "EvalFact",
    "EvalQuery",
    "EvalSetError",
    "load_case_from_dict",
    "load_cases",
]


class EvalSetError(ValueError):
    """评估集格式错误（schema 校验失败）。"""


@dataclass(frozen=True, slots=True)
class EvalFact:
    content: str
    category: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvalQuery:
    query: str
    expected_ids: tuple[int, ...]
    k: int
    must_include: tuple[int, ...] = ()
    must_exclude: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    setup_facts: tuple[EvalFact, ...]
    queries: tuple[EvalQuery, ...]
    tags: tuple[str, ...] = field(default_factory=tuple)


# ─── 解析 ──────────────────────────────────────────────────────────────────


def _coerce_tuple_of_int(name: str, raw: Any, *, n_facts: int) -> tuple[int, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise EvalSetError(f"{name} 必须是 int 列表，实际：{type(raw).__name__}")
    out: list[int] = []
    for x in raw:
        if isinstance(x, bool) or not isinstance(x, int):
            raise EvalSetError(f"{name} 元素必须是 int，实际：{x!r}")
        if x < 0 or x >= n_facts:
            raise EvalSetError(
                f"{name} 下标越界（{x}），setup_facts 长度为 {n_facts}"
            )
        out.append(x)
    return tuple(out)


def _coerce_tuple_of_str(name: str, raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise EvalSetError(f"{name} 必须是字符串列表，实际：{type(raw).__name__}")
    for x in raw:
        if not isinstance(x, str):
            raise EvalSetError(f"{name} 元素必须是 str，实际：{x!r}")
    return tuple(raw)


def _parse_fact(idx: int, obj: Any) -> EvalFact:
    if not isinstance(obj, dict):
        raise EvalSetError(f"setup_facts[{idx}] 必须是对象")
    content = obj.get("content")
    if not isinstance(content, str) or not content.strip():
        raise EvalSetError(f"setup_facts[{idx}].content 必须是非空字符串")
    category = obj.get("category")
    if category is not None and not isinstance(category, str):
        raise EvalSetError(f"setup_facts[{idx}].category 必须是字符串或省略")
    tags = _coerce_tuple_of_str(f"setup_facts[{idx}].tags", obj.get("tags"))
    return EvalFact(content=content, category=category, tags=tags)


def _parse_query(idx: int, obj: Any, *, n_facts: int) -> EvalQuery:
    if not isinstance(obj, dict):
        raise EvalSetError(f"queries[{idx}] 必须是对象")
    query = obj.get("query")
    if not isinstance(query, str) or not query.strip():
        raise EvalSetError(f"queries[{idx}].query 必须是非空字符串")
    expected = _coerce_tuple_of_int(
        f"queries[{idx}].expected_ids", obj.get("expected_ids"), n_facts=n_facts
    )
    if not expected:
        raise EvalSetError(f"queries[{idx}].expected_ids 不能为空")
    k_raw = obj.get("k")
    if isinstance(k_raw, bool) or not isinstance(k_raw, int) or k_raw < 1:
        raise EvalSetError(f"queries[{idx}].k 必须是 >= 1 的整数")
    must_include = _coerce_tuple_of_int(
        f"queries[{idx}].must_include", obj.get("must_include"), n_facts=n_facts
    )
    must_exclude = _coerce_tuple_of_int(
        f"queries[{idx}].must_exclude", obj.get("must_exclude"), n_facts=n_facts
    )
    overlap = set(must_include) & set(must_exclude)
    if overlap:
        raise EvalSetError(
            f"queries[{idx}].must_include 与 must_exclude 不能交叠：{sorted(overlap)}"
        )
    return EvalQuery(
        query=query,
        expected_ids=expected,
        k=k_raw,
        must_include=must_include,
        must_exclude=must_exclude,
    )


def load_case_from_dict(obj: dict[str, Any]) -> EvalCase:
    """单条用例校验 + 解析；评估集 fixtures 共用同一份 schema。"""
    if not isinstance(obj, dict):
        raise EvalSetError(f"用例必须是对象，实际：{type(obj).__name__}")
    case_id = obj.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise EvalSetError("id 必须是非空字符串")
    raw_facts = obj.get("setup_facts")
    if not isinstance(raw_facts, list) or not raw_facts:
        raise EvalSetError(f"{case_id}: setup_facts 必须是非空列表")
    facts = tuple(_parse_fact(i, f) for i, f in enumerate(raw_facts))
    raw_queries = obj.get("queries")
    if not isinstance(raw_queries, list) or not raw_queries:
        raise EvalSetError(f"{case_id}: queries 必须是非空列表")
    queries = tuple(
        _parse_query(i, q, n_facts=len(facts)) for i, q in enumerate(raw_queries)
    )
    tags = _coerce_tuple_of_str(f"{case_id}: tags", obj.get("tags"))
    return EvalCase(id=case_id, setup_facts=facts, queries=queries, tags=tags)


def load_cases(path: str | Path) -> list[EvalCase]:
    """从 jsonl 文件加载所有用例；跳过空行，对每行单独报错。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"评估集文件不存在：{p}")
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    with p.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise EvalSetError(
                    f"{p}:{lineno} JSON 解析失败：{exc.msg}"
                ) from exc
            case = load_case_from_dict(obj)
            if case.id in seen_ids:
                raise EvalSetError(f"{p}:{lineno} 重复 id：{case.id}")
            seen_ids.add(case.id)
            cases.append(case)
    if not cases:
        raise EvalSetError(f"{p} 不包含任何用例")
    return cases
