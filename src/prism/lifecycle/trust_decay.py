"""信任衰减 cron + helpful/unhelpful 反馈。

衰减公式：``trust_score = max(min_trust_floor, trust_score * decay_per_day ^ days)``。
采用全局周期衰减（``prism_stats`` 跟踪上次 run），衰减后低于阈值自动归档。

反馈：helpful +0.05 / unhelpful -0.10（不对称，有反馈优于无反馈）。
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final, Literal

from ..config import CategoryDecay, LifecycleConfig
from .archive import (
    ARCHIVE_REASON_LOW_TRUST,
    archive_fact,
)

if TYPE_CHECKING:
    from ..enrichment import EnrichmentQueue
    from ..hrr import IncrementalBank, RebuildDebouncer

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_HELPFUL_DELTA",
    "DEFAULT_LOW_TRUST_THRESHOLD",
    "DEFAULT_UNHELPFUL_DELTA",
    "STATS_KEY_LAST_DECAY",
    "DecayResult",
    "FeedbackResult",
    "FeedbackSignal",
    "apply_feedback",
    "apply_trust_decay",
]


# ─── 默认常量 ────────────────────────────────────────────────────────────

DEFAULT_LOW_TRUST_THRESHOLD: Final[float] = 0.1
"""trust_score 低于此值触发自动归档（``archive_reason='low_trust'``）。"""

DEFAULT_HELPFUL_DELTA: Final[float] = 0.05
DEFAULT_UNHELPFUL_DELTA: Final[float] = 0.10
"""helpful 加 0.05，unhelpful 减 0.10 — social signals。"""

TRUST_SCORE_MAX: Final[float] = 1.0
TRUST_SCORE_MIN: Final[float] = 0.0

STATS_KEY_LAST_DECAY: Final[str] = "last_trust_decay_at"
"""``prism_stats`` 表中跟踪上次 decay run 的键名。"""


FeedbackSignal = Literal["helpful", "unhelpful"]


@dataclass(frozen=True, slots=True)
class DecayResult:
    """:func:`apply_trust_decay` 的结果。

    Attributes:
        scanned: 本次扫描的 active fact 数
        decayed: 实际更新 trust_score 的 fact 数（decay_per_day=1.0 且未达 floor
            的 fact 不计）
        archived: 衰减后 trust_score < threshold 而归档的 fact 数
        archived_fact_ids: 归档的 fact_id 列表（升序）
        days_since_last_run: 本次衰减覆盖的天数
        cutoff: 本次 run 的时间戳，写入 ``prism_stats``
    """

    scanned: int
    decayed: int
    archived: int
    archived_fact_ids: tuple[int, ...]
    days_since_last_run: int
    cutoff: datetime


@dataclass(frozen=True, slots=True)
class FeedbackResult:
    """:func:`apply_feedback` 的结果。"""

    fact_id: int
    signal: FeedbackSignal
    delta: float
    new_trust_score: float
    new_helpful_count: int


# ─── 衰减 cron ────────────────────────────────────────────────────────────


def apply_trust_decay(
    conn: sqlite3.Connection,
    *,
    lifecycle: LifecycleConfig | None = None,
    now: datetime | None = None,
    low_trust_threshold: float = DEFAULT_LOW_TRUST_THRESHOLD,
    bank: IncrementalBank | None = None,
    debouncer: RebuildDebouncer | None = None,
    enrichment_queue: EnrichmentQueue | None = None,
) -> DecayResult:
    """对所有 active fact 按 category 应用 trust 衰减；低信任 fact 自动归档。

    Args:
        conn: 已 ``init_schema`` 的 SQLite 连接
        lifecycle: :class:`~prism.config.LifecycleConfig` 实例。None 时用默认值
        now: 测试可注入；None 时用当前 UTC 时间
        low_trust_threshold: 衰减后低于此值触发归档（默认 0.1）
        bank / debouncer / enrichment_queue: 透传到 :func:`~prism.lifecycle.archive.archive_fact`

    Returns:
        :class:`DecayResult`，含本次扫描 / 衰减 / 归档计数。
    """
    cfg = lifecycle if lifecycle is not None else LifecycleConfig()
    actual_now = now if now is not None else _utcnow_naive()

    days = _days_since_last_decay(conn, actual_now)

    rows = conn.execute(
        "SELECT fact_id, category, trust_score FROM facts WHERE status = 'active'"
    ).fetchall()

    scanned = len(rows)
    decayed = 0
    archived_ids: list[int] = []

    for row in rows:
        fid = int(row["fact_id"])
        category = str(row["category"]) if row["category"] is not None else "general"
        current = float(row["trust_score"]) if row["trust_score"] is not None else 0.5

        decay_cfg = _resolve_decay(cfg, category)
        new_trust = _compute_new_trust(current, decay_cfg, days)

        if new_trust != current:
            conn.execute(
                "UPDATE facts SET trust_score = ? WHERE fact_id = ?",
                (new_trust, fid),
            )
            decayed += 1

        if new_trust < low_trust_threshold:
            result = archive_fact(
                conn,
                fid,
                reason=ARCHIVE_REASON_LOW_TRUST,
                bank=bank,
                debouncer=debouncer,
                enrichment_queue=enrichment_queue,
                now=actual_now,
            )
            if result is not None and result.archived:
                archived_ids.append(fid)

    _write_last_decay(conn, actual_now)

    return DecayResult(
        scanned=scanned,
        decayed=decayed,
        archived=len(archived_ids),
        archived_fact_ids=tuple(archived_ids),
        days_since_last_run=days,
        cutoff=actual_now,
    )


# ─── helpful / unhelpful 反馈 ────────────────────────────────────────────


def apply_feedback(
    conn: sqlite3.Connection,
    fact_id: int,
    signal: FeedbackSignal,
    *,
    helpful_delta: float = DEFAULT_HELPFUL_DELTA,
    unhelpful_delta: float = DEFAULT_UNHELPFUL_DELTA,
) -> FeedbackResult | None:
    """对单条 fact 应用 helpful / unhelpful 反馈。

    - **helpful**: ``trust_score += helpful_delta``（上界 1.0），``helpful_count += 1``
    - **unhelpful**: ``trust_score -= unhelpful_delta``（下界 0.0），``helpful_count -= 1``

    ``helpful_count`` 复用为净反馈计数（``+helpful - unhelpful``）。

    Args:
        conn: SQLite 连接
        fact_id: 目标 fact
        signal: 'helpful' 或 'unhelpful'
        helpful_delta / unhelpful_delta: 可调节增量

    Returns:
        ``FeedbackResult`` 成功；``None`` 表示 fact_id 不存在或 status != 'active'。

    Raises:
        ValueError: signal 不在 {'helpful', 'unhelpful'}
    """
    if signal not in ("helpful", "unhelpful"):
        raise ValueError(f"signal 必须是 'helpful' 或 'unhelpful'，实际：{signal!r}")

    row = conn.execute(
        "SELECT trust_score, helpful_count, status FROM facts WHERE fact_id = ?",
        (fact_id,),
    ).fetchone()
    if row is None:
        log.warning("apply_feedback: fact_id=%s 不存在", fact_id)
        return None

    if str(row["status"]) != "active":
        log.debug(
            "apply_feedback: fact_id=%s status=%s，跳过", fact_id, row["status"]
        )
        return None

    current = float(row["trust_score"]) if row["trust_score"] is not None else 0.5
    helpful_count = int(row["helpful_count"]) if row["helpful_count"] is not None else 0

    if signal == "helpful":
        delta = helpful_delta
        new_trust = min(TRUST_SCORE_MAX, current + delta)
        new_count = helpful_count + 1
    else:
        delta = -unhelpful_delta
        new_trust = max(TRUST_SCORE_MIN, current + delta)
        new_count = helpful_count - 1

    with _txn(conn):
        conn.execute(
            "UPDATE facts SET trust_score = ?, helpful_count = ? WHERE fact_id = ?",
            (new_trust, new_count, fact_id),
        )

    return FeedbackResult(
        fact_id=fact_id,
        signal=signal,
        delta=delta,
        new_trust_score=new_trust,
        new_helpful_count=new_count,
    )


# ─── 内部工具 ────────────────────────────────────────────────────────────


def _resolve_decay(cfg: LifecycleConfig, category: str) -> CategoryDecay:
    """按 category 查 decay 配置，未配置回退 general，仍未配回退默认。"""
    if category in cfg.decay_by_category:
        return cfg.decay_by_category[category]
    if "general" in cfg.decay_by_category:
        return cfg.decay_by_category["general"]
    return CategoryDecay()


def _compute_new_trust(
    current: float, decay_cfg: CategoryDecay, days: int
) -> float:
    """``new = max(min_floor, current * decay^days)``，已应用 [0, 1] 边界。"""
    if days <= 0:
        # 同日内多次 run 不重复衰减
        return _clamp(current)
    factor = decay_cfg.decay_per_day ** days
    raw = current * factor
    floored = max(decay_cfg.min_trust_floor, raw)
    return _clamp(floored)


def _clamp(v: float) -> float:
    return max(TRUST_SCORE_MIN, min(TRUST_SCORE_MAX, v))


def _days_since_last_decay(conn: sqlite3.Connection, now: datetime) -> int:
    """从 ``prism_stats`` 读上次 run 时间；不存在 → 1（首次按 1 天衰减）。"""
    row = conn.execute(
        "SELECT value FROM prism_stats WHERE key = ?", (STATS_KEY_LAST_DECAY,)
    ).fetchone()
    if row is None:
        return 1
    try:
        last = datetime.fromisoformat(str(row["value"]))
        if last.tzinfo is not None:
            last = last.replace(tzinfo=None)
    except (ValueError, TypeError):
        log.warning(
            "prism_stats[%s] 不是合法 ISO 时间字符串：%r，按 1 天衰减",
            STATS_KEY_LAST_DECAY, row["value"],
        )
        return 1
    delta_days = (now - last).days
    return max(0, delta_days)


def _write_last_decay(conn: sqlite3.Connection, now: datetime) -> None:
    """把本次 run 时间写入 ``prism_stats``（UPSERT）。"""
    iso = now.replace(microsecond=0).isoformat(sep=" ")
    conn.execute(
        "INSERT INTO prism_stats (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
        (STATS_KEY_LAST_DECAY, iso),
    )


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@contextmanager
def _txn(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
