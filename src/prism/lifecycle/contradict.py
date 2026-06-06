"""矛盾检测增量版。

判定公式：``score = entity_overlap * (1 - semantic_sim)``，当
jaccard 实体重叠 >= 0.5 且 cosine 相似度 < 0.4 时视为矛盾候选。

:class:`ContradictDetector` 维护 ``recent_changed_ids``，每次 cron 做
``changed x corpus`` 增量对比。:func:`list_contradictions` 直查
``contradiction_log``。同一对幂等（``(min,max)`` 归一化 + 去重）。
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np

if TYPE_CHECKING:
    from ..vstore import VectorStore

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CONTRADICT_THRESHOLD",
    "DEFAULT_CORPUS_LIMIT",
    "DEFAULT_ENTITY_OVERLAP_MIN",
    "DEFAULT_MIN_ENTITIES",
    "DEFAULT_SEMANTIC_SIM_MAX",
    "ContradictDetector",
    "ContradictResult",
    "detect_contradiction",
    "list_contradictions",
]


# ─── 阈值默认 ────────────────────────────────────────────────────────────

DEFAULT_MIN_ENTITIES: Final[int] = 3
"""``min(len(entities_a), len(entities_b)) < N`` 直接跳过。"""

DEFAULT_ENTITY_OVERLAP_MIN: Final[float] = 0.5
"""jaccard 实体重叠 ≥ 此值才考虑矛盾。"""

DEFAULT_SEMANTIC_SIM_MAX: Final[float] = 0.4
"""cos 相似度 < 此值才考虑矛盾。"""

DEFAULT_CONTRADICT_THRESHOLD: Final[float] = 0.3
"""``score = overlap * (1 - sim)`` ≥ 此值才写 ``contradiction_log``。

按公式上界 1.0 × (1-0) = 1.0 + 默认阈值 0.5×0.6=0.3
（overlap=0.5 + sim=0.4 边缘）。
"""

DEFAULT_CORPUS_LIMIT: Final[int] = 2000
"""``check()`` 单次扫描 active fact 上限。"""


# ─── 数据结构 ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ContradictResult:
    """单次 :meth:`ContradictDetector.check` 的结果。

    Attributes:
        scanned_changed: 本次扫描的 changed fact 数
        scanned_corpus: 本次扫描的语料 active fact 数
        pairs_examined: 实际 ``detect_contradiction`` 调用次数（去除 self / 缺向量）
        contradictions_logged: 写入 ``contradiction_log`` 的新行数
        logged_pairs: 写入的 ``(a, b, score)`` 三元组（按 a<b 归一化，升序）
    """

    scanned_changed: int
    scanned_corpus: int
    pairs_examined: int
    contradictions_logged: int
    logged_pairs: tuple[tuple[int, int, float], ...]


@dataclass(slots=True)
class _FactRow:
    fact_id: int
    entities: frozenset[str]


# ─── 公式 ────────────────────────────────────────────────────────────────


def detect_contradiction(
    entities_a: Iterable[str],
    entities_b: Iterable[str],
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    *,
    min_entities: int = DEFAULT_MIN_ENTITIES,
    overlap_min: float = DEFAULT_ENTITY_OVERLAP_MIN,
    sim_max: float = DEFAULT_SEMANTIC_SIM_MAX,
) -> float:
    """返回矛盾分 ``score``。

    ``score = entity_overlap * (1 - semantic_sim)`` 当且仅当 overlap ≥
    ``overlap_min`` 且 sim < ``sim_max`` 且双方实体数 ≥ ``min_entities``；
    否则返 0.0。

    Args:
        entities_a / entities_b: 各自的实体名集合（外部去重）
        vec_a / vec_b: 已归一化的向量（``vstore.fetch`` 返回的语义向量）
        min_entities: ``min(|a|, |b|) < min_entities`` → 跳过
        overlap_min: jaccard 重叠下限
        sim_max: cos 相似上限

    Returns:
        ``[0.0, 1.0]`` 区间内的分数；0.0 表示不构成矛盾候选。
    """
    set_a = frozenset(entities_a)
    set_b = frozenset(entities_b)
    if min(len(set_a), len(set_b)) < min_entities:
        return 0.0

    union = set_a | set_b
    if not union:
        return 0.0
    overlap = len(set_a & set_b) / len(union)
    if overlap < overlap_min:
        return 0.0

    sim = float(np.dot(vec_a, vec_b))
    if sim >= sim_max:
        return 0.0

    return float(overlap * (1.0 - sim))


# ─── 增量检测器 ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class ContradictDetector:
    """增量矛盾检测 cron。

    Args:
        conn: 已 ``init_schema`` 的 SQLite 连接
        vstore: 向量存储；``vstore.fetch(all_ids)`` 一次性批量取向量
        threshold: ``score ≥ threshold`` 才写日志（默认 :data:`DEFAULT_CONTRADICT_THRESHOLD`）
        corpus_limit: 单次扫描的 active fact 上限（默认 :data:`DEFAULT_CORPUS_LIMIT`）
        min_entities / overlap_min / sim_max: 透传到 :func:`detect_contradiction`

    Usage:
        >>> detector = ContradictDetector(conn, vstore)
        >>> detector.notify_changed(fact_id)  # mirror_add / mirror_replace 触发
        >>> result = detector.check()         # cron 触发；返回 ContradictResult
    """

    conn: sqlite3.Connection
    vstore: VectorStore
    threshold: float = DEFAULT_CONTRADICT_THRESHOLD
    corpus_limit: int = DEFAULT_CORPUS_LIMIT
    min_entities: int = DEFAULT_MIN_ENTITIES
    overlap_min: float = DEFAULT_ENTITY_OVERLAP_MIN
    sim_max: float = DEFAULT_SEMANTIC_SIM_MAX
    recent_changed_ids: set[int] = field(default_factory=set)

    def notify_changed(self, fact_id: int) -> None:
        """登记新增 / 更新的 fact_id；下一次 :meth:`check` 时纳入对比。"""
        if fact_id > 0:
            self.recent_changed_ids.add(int(fact_id))

    def check(self) -> ContradictResult:
        """运行一轮增量检测；清空 ``recent_changed_ids``。

        语义：``changed × all_active``，避免 N×N 全扫；首次或全量校验可手动
        把所有 active id 加入 ``recent_changed_ids``。
        """
        if not self.recent_changed_ids:
            return ContradictResult(
                scanned_changed=0,
                scanned_corpus=0,
                pairs_examined=0,
                contradictions_logged=0,
                logged_pairs=(),
            )

        changed_ids = sorted(self.recent_changed_ids)
        self.recent_changed_ids.clear()

        changed = self._load_facts(changed_ids)
        corpus = self._load_active_corpus(self.corpus_limit)

        all_ids: set[int] = {f.fact_id for f in changed} | {
            f.fact_id for f in corpus
        }
        if not all_ids:
            return ContradictResult(
                scanned_changed=len(changed),
                scanned_corpus=len(corpus),
                pairs_examined=0,
                contradictions_logged=0,
                logged_pairs=(),
            )

        try:
            vectors = self.vstore.fetch(list(all_ids))
        except Exception:
            log.exception("vstore.fetch 失败；本轮 contradict 跳过")
            return ContradictResult(
                scanned_changed=len(changed),
                scanned_corpus=len(corpus),
                pairs_examined=0,
                contradictions_logged=0,
                logged_pairs=(),
            )

        pairs_examined = 0
        logged: list[tuple[int, int, float]] = []
        existing_pairs = self._load_existing_open_pairs()

        for new_fact in changed:
            new_vec = vectors.get(new_fact.fact_id)
            if new_vec is None:
                continue
            for existing in corpus:
                if existing.fact_id == new_fact.fact_id:
                    continue
                ex_vec = vectors.get(existing.fact_id)
                if ex_vec is None:
                    continue
                pairs_examined += 1
                score = detect_contradiction(
                    new_fact.entities,
                    existing.entities,
                    new_vec,
                    ex_vec,
                    min_entities=self.min_entities,
                    overlap_min=self.overlap_min,
                    sim_max=self.sim_max,
                )
                if score < self.threshold:
                    continue
                a, b = _normalize_pair(new_fact.fact_id, existing.fact_id)
                if (a, b) in existing_pairs:
                    continue
                self._log_contradiction(a, b, score)
                existing_pairs.add((a, b))
                logged.append((a, b, score))

        return ContradictResult(
            scanned_changed=len(changed),
            scanned_corpus=len(corpus),
            pairs_examined=pairs_examined,
            contradictions_logged=len(logged),
            logged_pairs=tuple(sorted(logged, key=lambda t: (t[0], t[1]))),
        )

    # ─── 内部 ─────────────────────────────────────────────────────────────

    def _load_facts(self, fact_ids: list[int]) -> list[_FactRow]:
        if not fact_ids:
            return []
        placeholders = ",".join("?" for _ in fact_ids)
        rows = self.conn.execute(
            f"SELECT f.fact_id, e.name AS entity FROM facts f "
            f"LEFT JOIN fact_entities fe ON fe.fact_id = f.fact_id "
            f"LEFT JOIN entities e ON e.entity_id = fe.entity_id "
            f"WHERE f.fact_id IN ({placeholders}) AND f.status = 'active'",
            fact_ids,
        ).fetchall()
        return _group_entities(rows)

    def _load_active_corpus(self, limit: int) -> list[_FactRow]:
        rows = self.conn.execute(
            "SELECT f.fact_id, e.name AS entity FROM facts f "
            "LEFT JOIN fact_entities fe ON fe.fact_id = f.fact_id "
            "LEFT JOIN entities e ON e.entity_id = fe.entity_id "
            "WHERE f.fact_id IN ("
            "  SELECT fact_id FROM facts WHERE status = 'active' "
            "  ORDER BY fact_id DESC LIMIT ?"
            ")",
            (int(limit),),
        ).fetchall()
        return _group_entities(rows)

    def _load_existing_open_pairs(self) -> set[tuple[int, int]]:
        rows = self.conn.execute(
            "SELECT fact_a, fact_b FROM contradiction_log WHERE resolved = 0"
        ).fetchall()
        return {
            _normalize_pair(int(r["fact_a"]), int(r["fact_b"])) for r in rows
        }

    def _log_contradiction(self, fact_a: int, fact_b: int, score: float) -> None:
        self.conn.execute(
            "INSERT INTO contradiction_log (fact_a, fact_b, score) VALUES (?, ?, ?)",
            (fact_a, fact_b, float(score)),
        )


# ─── 直查（PrismRecall.contradict 路径）─────────────────────────────────


def list_contradictions(
    conn: sqlite3.Connection,
    *,
    category: str | None = None,
    threshold: float | None = None,
    limit: int = 10,
    resolved: int = 0,
) -> list[dict[str, Any]]:
    """列出 ``contradiction_log`` 中的矛盾候选。

    Args:
        conn: SQLite 连接
        category: 限定双方 fact 的 category（任一方匹配即纳入）；None = 不限
        threshold: 过滤 ``score >= threshold``；None = 不限
        limit: 最多返回条数
        resolved: 0=open（默认）/ 1=resolved / 2=false_positive

    Returns:
        list[dict] 按 ``score DESC, detected_at DESC`` 排序：
        ``{contradiction_id, fact_a, fact_b, score, detected_at,
        content_a, content_b, category_a, category_b}``
    """
    if limit < 1:
        return []

    params: list[Any] = [int(resolved)]
    sql = (
        "SELECT cl.id AS cid, cl.fact_a, cl.fact_b, cl.score, cl.detected_at, "
        "       fa.content AS content_a, fb.content AS content_b, "
        "       fa.category AS category_a, fb.category AS category_b "
        "FROM contradiction_log cl "
        "JOIN facts fa ON fa.fact_id = cl.fact_a "
        "JOIN facts fb ON fb.fact_id = cl.fact_b "
        "WHERE cl.resolved = ?"
    )
    if threshold is not None:
        sql += " AND cl.score >= ?"
        params.append(float(threshold))
    if category is not None:
        sql += " AND (fa.category = ? OR fb.category = ?)"
        params.extend([category, category])
    sql += " ORDER BY cl.score DESC, cl.detected_at DESC LIMIT ?"
    params.append(int(limit))

    rows = conn.execute(sql, params).fetchall()
    return [
        {
            "contradiction_id": int(r["cid"]),
            "fact_a": int(r["fact_a"]),
            "fact_b": int(r["fact_b"]),
            "score": float(r["score"]),
            "detected_at": str(r["detected_at"]),
            "content_a": str(r["content_a"]),
            "content_b": str(r["content_b"]),
            "category_a": str(r["category_a"]) if r["category_a"] else None,
            "category_b": str(r["category_b"]) if r["category_b"] else None,
        }
        for r in rows
    ]


# ─── 工具 ────────────────────────────────────────────────────────────────


def _normalize_pair(a: int, b: int) -> tuple[int, int]:
    """``(min, max)`` 归一化：让 (a, b) 与 (b, a) 视为同一对。"""
    return (a, b) if a < b else (b, a)


def _group_entities(rows: Iterable[sqlite3.Row]) -> list[_FactRow]:
    """LEFT JOIN 行 → list[_FactRow]，按 fact_id 分组（无实体的 fact 也保留）。"""
    grouped: dict[int, set[str]] = {}
    for row in rows:
        fid = int(row["fact_id"])
        ent = row["entity"]
        bucket = grouped.setdefault(fid, set())
        if ent is not None:
            bucket.add(str(ent))
    return [
        _FactRow(fact_id=fid, entities=frozenset(ents))
        for fid, ents in sorted(grouped.items())
    ]
