"""fact 归档（archive）核心。

提供显式 / 计划任务触发的归档入口：DB 标记 archived + bank.remove +
enrichment_queue.mark_done。物理清除通过 :func:`purge_old_archived` 完成。

所有操作幂等。

v2: SQL 操作委托到 Repository 层。
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

from ..db import FactsRepository, ContradictionRepository

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
    facts = FactsRepository(conn)
    row = facts.get_fact_by_id(fact_id)
    if row is None:
        log.warning("archive_fact: fact_id=%s 不存在", fact_id)
        return None

    category = str(row.get("category", "general"))
    existing_status = str(row.get("status", "active"))

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
        facts.archive_fact(fact_id, reason, now=_to_iso(now) if now is not None else None)

    # 队列清理
    if enrichment_queue is not None:
        try:
            enrichment_queue.mark_done(fact_id)
        except Exception as e:
            log.warning(
                "archive_fact: queue.mark_done 失败 fact_id=%s: %s", fact_id, e
            )

    # bank.remove + 去抖 calibrate
    hrr_blob = row.get("hrr_vector")
    if bank is not None and hrr_blob is not None:
        try:
            vec = np.frombuffer(hrr_blob, dtype=np.float64)
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


# ─── TTL 批量归档 ──────────────────────────────────────────────────────────


def archive_by_ttl(
    conn: sqlite3.Connection,
    *,
    now: datetime | None = None,
    bank: IncrementalBank | None = None,
    debouncer: RebuildDebouncer | None = None,
    enrichment_queue: EnrichmentQueue | None = None,
    batch_size: int = 500,
) -> list[int]:
    """扫描并归档所有 TTL 已到期的 active fact。

    仅处理 ``ttl_days > 0`` 的 fact；``created_at + ttl_days days <= now``
    时触发归档（``archive_reason='ttl_expired'``）。

    Returns:
        本次归档的 fact_id 列表（按 fact_id 升序）。空列表表示无过期 fact。
    """
    facts = FactsRepository(conn)
    actual_now = now if now is not None else _utcnow_naive()
    now_str = _to_iso(actual_now)

    ttl_ids = facts.get_expired_ttl_ids(now_str, batch_size=batch_size)
    if not ttl_ids:
        return []

    archived: list[int] = []
    for fid in ttl_ids:
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
            log.debug("TTL archived fact_id=%s", fid)
    return archived


# ─── 物理清除 ───────────────────────────────────────────────────────────────


def purge_old_archived(
    conn: sqlite3.Connection,
    *,
    now: datetime | None = None,
    retention_days: int = 90,
    vstore: VectorStore | None = None,
    batch_size: int = 500,
) -> PurgeResult:
    """物理删除超过保留期的 archived fact。

    保留期 = ``archived_at + retention_days days <= now``。
    利用 FK CASCADE 自动清理关联行；删除前先解除非级联 FK 引用。

    Returns:
        :class:`PurgeResult`，含本次删除的 fact 数及 ID 列表。
    """
    facts = FactsRepository(conn)
    ctl = ContradictionRepository(conn)
    actual_now = now if now is not None else _utcnow_naive()
    cutoff = _to_iso(actual_now)

    fact_ids = facts.get_purge_candidates(cutoff, retention_days, batch_size)
    if not fact_ids:
        return PurgeResult(
            deleted_count=0,
            deleted_fact_ids=(),
            cutoff=actual_now,
        )

    with _txn(conn):
        # 解开 supersedes_id 反向引用（FK 非级联）
        facts.unlink_supersedes(fact_ids)
        # 解开 contradiction_log.loser_fact_id（FK 非级联）
        ctl.unlink_loser_facts(fact_ids)
        # 物理删除（FK CASCADE 自动级联 fact_entities / enrichment_queue / contradiction_log.fact_a/b）
        facts.delete_facts(fact_ids)

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

    log.info("purge 完成：删除 %d 条 archived fact（retention=%d 天，cutoff=%s）",
             len(fact_ids), retention_days, cutoff)

    return PurgeResult(
        deleted_count=len(fact_ids),
        deleted_fact_ids=tuple(fact_ids),
        cutoff=actual_now,
    )


# ─── 工具 ────────────────────────────────────────────────────────────────────


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _to_iso(dt: datetime) -> str:
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def _txn(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """轻量事务上下文：成功 COMMIT / 异常 ROLLBACK。"""
    conn.execute("BEGIN")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
