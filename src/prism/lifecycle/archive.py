"""fact 归档（archive）核心。

提供显式 / 计划任务触发的归档入口：DB 标记 archived + bank.remove +
enrichment_queue.mark_done。物理清除通过 :func:`purge_old_archived` 完成。

所有操作幂等。
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from ..enrichment import EnrichmentQueue
    from ..hrr import IncrementalBank, RebuildDebouncer
    from ..vstore import VectorStore

log = logging.getLogger(__name__)

__all__ = [
    "ARCHIVE_REASON_BUILTIN_REMOVED",
    "ARCHIVE_REASON_CONTRADICTED",
    "ARCHIVE_REASON_GHOST",
    "ARCHIVE_REASON_LOW_TRUST",
    "ARCHIVE_REASON_MANUAL",
    "ARCHIVE_REASON_REPLACED",
    "ARCHIVE_REASON_TTL",
    "ArchiveResult",
    "PurgeResult",
    "archive_by_ttl",
    "archive_fact",
    "purge_old_archived",
]


# ─── archive_reason 取值 ─────────────────

ARCHIVE_REASON_REPLACED: Final[str] = "replaced"
ARCHIVE_REASON_MANUAL: Final[str] = "manual"
ARCHIVE_REASON_BUILTIN_REMOVED: Final[str] = "builtin_removed"
ARCHIVE_REASON_GHOST: Final[str] = "ghost"
ARCHIVE_REASON_TTL: Final[str] = "ttl_expired"
ARCHIVE_REASON_LOW_TRUST: Final[str] = "low_trust"
ARCHIVE_REASON_CONTRADICTED: Final[str] = "contradicted"


@dataclass(frozen=True, slots=True)
class ArchiveResult:
    """单条归档操作的结果。

    Attributes:
        fact_id: 操作目标
        archived: ``True`` 本次真正改写了 status；``False`` 表示已是 archived 的
            no-op 路径
        reason: 写入的 ``archive_reason`` 值（no-op 时是入参 reason，DB 中保留原值）
        category: fact 的 category（用于日志 / 调用方决策）
    """

    fact_id: int
    archived: bool
    reason: str
    category: str


@dataclass(frozen=True, slots=True)
class PurgeResult:
    """:func:`purge_old_archived` 的结果。

    Attributes:
        deleted_count: 实际从 ``facts`` 表 DELETE 的行数
        deleted_fact_ids: 已删除的 fact_id 列表（按 ID 升序）
        cutoff: 用于判定的截止时间戳
    """

    deleted_count: int
    deleted_fact_ids: tuple[int, ...]
    cutoff: datetime


# ─── 单条归档 ─────────────────────────────────────────────────────────────


def archive_fact(
    conn: sqlite3.Connection,
    fact_id: int,
    *,
    reason: str = ARCHIVE_REASON_MANUAL,
    bank: IncrementalBank | None = None,
    debouncer: RebuildDebouncer | None = None,
    enrichment_queue: EnrichmentQueue | None = None,
    now: datetime | None = None,
) -> ArchiveResult | None:
    """归档单条 fact。

    与 :meth:`prism.mirror.PrismMirror.mirror_remove` 共享同一份「DB UPDATE +
    队列 mark_done + bank.remove + debouncer.schedule」语义；区别是本函数
    走「显式按 fact_id 归档」路径，不解析 metadata / content 定位，更适合
    admin API / TTL cron / 低信任 cron / 矛盾检测调用。

    Args:
        conn: 已 ``init_schema`` 的 SQLite 连接
        fact_id: 目标 fact 主键
        reason: ``archive_reason`` 列写入值。默认 ``'manual'``。常用：
            :data:`ARCHIVE_REASON_TTL` / :data:`ARCHIVE_REASON_LOW_TRUST` /
            :data:`ARCHIVE_REASON_CONTRADICTED`
        bank: 注入后调 ``bank.remove(category, hrr_vector)``。dim 不匹配或
            category 不存在时仅 WARN，不阻塞 DB UPDATE
        debouncer: 注入后在 bank.remove 之后调 ``schedule()``，触发 50ms 窗口
            合并 calibrate
        enrichment_queue: 注入后调 ``mark_done(fact_id)`` 清队列行
        now: 测试可注入；None 时用 ``CURRENT_TIMESTAMP``

    Returns:
        - ``ArchiveResult(archived=True, ...)`` 成功归档
        - ``ArchiveResult(archived=False, ...)`` fact 已是 archived（no-op，
          不动 ``archive_reason``）
        - ``None`` fact_id 不存在
    """
    row = conn.execute(
        "SELECT content, category, status, hrr_vector "
        "FROM facts WHERE fact_id = ?",
        (fact_id,),
    ).fetchone()
    if row is None:
        log.warning("archive_fact: fact_id=%s 不存在", fact_id)
        return None

    category = str(row["category"]) if row["category"] is not None else "general"
    existing_status = str(row["status"])

    if existing_status != "active":
        log.debug(
            "archive_fact: fact_id=%s 已 %s，no-op", fact_id, existing_status
        )
        return ArchiveResult(
            fact_id=fact_id,
            archived=False,
            reason=reason,
            category=category,
        )

    with _txn(conn):
        if now is None:
            conn.execute(
                "UPDATE facts SET status = 'archived', "
                "archived_at = CURRENT_TIMESTAMP, archive_reason = ? "
                "WHERE fact_id = ? AND status = 'active'",
                (reason, fact_id),
            )
        else:
            conn.execute(
                "UPDATE facts SET status = 'archived', "
                "archived_at = ?, archive_reason = ? "
                "WHERE fact_id = ? AND status = 'active'",
                (_to_iso(now), reason, fact_id),
            )

    # 队列清理
    if enrichment_queue is not None:
        try:
            enrichment_queue.mark_done(fact_id)
        except Exception as e:
            log.warning(
                "archive_fact: queue.mark_done 失败 fact_id=%s: %s", fact_id, e
            )

    # bank.remove + 去抖 calibrate
    if bank is not None and row["hrr_vector"] is not None:
        try:
            vec = np.frombuffer(row["hrr_vector"], dtype=np.float64)
            if vec.shape == (bank.dim,):
                bank.remove(category, vec)
            else:
                log.warning(
                    "archive_fact: hrr_vector shape 异常 fact_id=%s shape=%s",
                    fact_id, vec.shape,
                )
        except Exception as e:
            log.warning(
                "archive_fact: bank.remove 失败 fact_id=%s category=%s: %s",
                fact_id, category, e,
            )

    if debouncer is not None:
        try:
            debouncer.schedule()
        except Exception as e:
            log.warning(
                "archive_fact: debouncer.schedule 失败 fact_id=%s: %s",
                fact_id, e,
            )

    return ArchiveResult(
        fact_id=fact_id,
        archived=True,
        reason=reason,
        category=category,
    )


# ─── TTL cron ─────────────────────────────────────────────────────────────


def archive_by_ttl(
    conn: sqlite3.Connection,
    *,
    now: datetime | None = None,
    bank: IncrementalBank | None = None,
    debouncer: RebuildDebouncer | None = None,
    enrichment_queue: EnrichmentQueue | None = None,
    batch_size: int = 500,
) -> list[int]:
    """扫描 ``ttl_days > 0`` 且过期的 active fact，全部归档。

    过期判定：``julianday(now) - julianday(created_at) > ttl_days``。
    ``ttl_days = 0`` 视为不过期。

    Args:
        conn: 已 ``init_schema`` 的 SQLite 连接
        now: 测试可注入；None 时用当前 UTC 时间
        bank / debouncer / enrichment_queue: 透传到 :func:`archive_fact`
        batch_size: 单批扫描上限（防一次拉全表）

    Returns:
        本次归档的 fact_id 列表（按扫描顺序，即 fact_id 升序）。空列表表示
        无过期 fact。
    """
    actual_now = now if now is not None else _utcnow_naive()
    now_str = _to_iso(actual_now)

    rows = conn.execute(
        "SELECT fact_id FROM facts "
        "WHERE status = 'active' AND ttl_days > 0 "
        "AND julianday(?) - julianday(created_at) > ttl_days "
        "ORDER BY fact_id LIMIT ?",
        (now_str, batch_size),
    ).fetchall()

    archived: list[int] = []
    for row in rows:
        fid = int(row["fact_id"])
        result = archive_fact(
            conn,
            fid,
            reason=ARCHIVE_REASON_TTL,
            bank=bank,
            debouncer=debouncer,
            enrichment_queue=enrichment_queue,
            now=actual_now,
        )
        if result is not None and result.archived:
            archived.append(fid)
    return archived


# ─── 90d 物理清除 cron ────────────────────────────────────────


def purge_old_archived(
    conn: sqlite3.Connection,
    *,
    retention_days: int = 90,
    now: datetime | None = None,
    vstore: VectorStore | None = None,
    batch_size: int = 500,
) -> PurgeResult:
    """物理删除超过 ``retention_days`` 仍处于 archived 状态的 fact。

    事务内解开 supersedes_id / loser_fact_id FK 后 DELETE，CASCADE 级联清理关联表。
    vstore 清理在事务外逐条容错执行。

    Args:
        conn: 已 ``init_schema`` 的 SQLite 连接
        retention_days: 保留窗口天数，默认 90（来自 ``LifecycleConfig.archive_after_days``）
        now: 测试可注入；None 时用当前 UTC 时间
        vstore: 注入后逐条调 ``vstore.remove(fact_id)`` 清向量存储
        batch_size: 单批 SELECT/DELETE 上限

    Returns:
        :class:`PurgeResult`，含已删除条数与 fact_id 列表。
    """
    actual_now = now if now is not None else _utcnow_naive()
    cutoff = _to_iso(actual_now)

    rows = conn.execute(
        "SELECT fact_id FROM facts "
        "WHERE status = 'archived' AND archived_at IS NOT NULL "
        "AND julianday(?) - julianday(archived_at) > ? "
        "ORDER BY fact_id LIMIT ?",
        (cutoff, int(retention_days), batch_size),
    ).fetchall()

    if not rows:
        return PurgeResult(
            deleted_count=0,
            deleted_fact_ids=tuple(),
            cutoff=actual_now,
        )

    fact_ids = tuple(int(r["fact_id"]) for r in rows)
    placeholders = ",".join("?" * len(fact_ids))

    with _txn(conn):
        # 解开 supersedes_id 反向引用（FK 非级联）
        conn.execute(
            f"UPDATE facts SET supersedes_id = NULL "
            f"WHERE supersedes_id IN ({placeholders})",
            fact_ids,
        )
        # 解开 contradiction_log.loser_fact_id（FK 非级联）
        conn.execute(
            f"UPDATE contradiction_log SET loser_fact_id = NULL "
            f"WHERE loser_fact_id IN ({placeholders})",
            fact_ids,
        )
        # 物理删除（FK CASCADE 自动级联 fact_entities / enrichment_queue / contradiction_log.fact_a/b）
        conn.execute(
            f"DELETE FROM facts WHERE fact_id IN ({placeholders})",
            fact_ids,
        )

    # vstore 清理（事务外，逐条容错 — 失败不影响 DB 一致性）
    if vstore is not None:
        for fid in fact_ids:
            try:
                vstore.remove(fid)
            except Exception as e:
                log.warning(
                    "purge_old_archived: vstore.remove 失败 fact_id=%s: %s",
                    fid, e,
                )

    return PurgeResult(
        deleted_count=len(fact_ids),
        deleted_fact_ids=fact_ids,
        cutoff=actual_now,
    )


# ─── 工具 ────────────────────────────────────────────────────────────────


def _utcnow_naive() -> datetime:
    """返回 naive UTC datetime（与 SQLite ``CURRENT_TIMESTAMP`` 同语义）。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _to_iso(dt: datetime) -> str:
    """转 SQLite 友好 ``YYYY-MM-DD HH:MM:SS`` ISO 字符串。"""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def _txn(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """``isolation_level=None`` 连接专用的轻量事务：成功 COMMIT / 异常 ROLLBACK。"""
    conn.execute("BEGIN")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
