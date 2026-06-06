"""`prism migrate --from holographic` CLI 与核心函数。

把 Holographic ``memory_store.db`` 全量迁到 Prism DB：逐条 add、按 content
去重、保留原 trust_score / created_at / helpful_count / retrieval_count（UPDATE
后置），不迁移 Holographic entities 表（用 Prism Stage 1 重抽）。
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from prism.cli.interface import single
from prism.config import (
    DEFAULT_PROFILE,
    DEFAULT_USER_ID,
    default_config,
    discover_config_path,
    load_config,
    resolve_db_path_for_user,
)
from prism.mcp.wire import RuntimeOptions, build_runtime

if TYPE_CHECKING:
    from prism.mcp.wire import PrismRuntime

log = logging.getLogger(__name__)

__all__ = [
    "MigrationStats",
    "main",
    "migrate_from_holographic",
]


@dataclass(frozen=True, slots=True)
class MigrationStats:
    """单次迁移的统计快照。"""

    total: int
    """源库 facts 总行数。"""
    added: int
    """新写入 Prism 的 fact 数。"""
    skipped_duplicate: int
    """content 已在 Prism 中存在，跳过的数量。"""
    failed: int
    """写入失败（抛异常）的数量；详情进 log.warning。"""


# ─── 核心迁移 ─────────────────────────────────────────────────────────────


def _iter_holographic_facts(src_db: Path) -> Iterator[sqlite3.Row]:
    """流式读 Holographic facts 表 — 不一次 load 全表，支持 100k+ 规模。

    返回字段：``fact_id, content, category, tags, trust_score,
    retrieval_count, helpful_count, created_at``
    """
    if not src_db.exists():
        raise FileNotFoundError(f"Holographic 源库不存在：{src_db}")
    conn = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        # 按 created_at ASC 顺序，保证迁移后 fact_id 顺序大致与时间一致
        yield from conn.execute(
            "SELECT fact_id, content, category, "
            "COALESCE(tags, '') AS tags, "
            "COALESCE(trust_score, 0.5) AS trust_score, "
            "COALESCE(retrieval_count, 0) AS retrieval_count, "
            "COALESCE(helpful_count, 0) AS helpful_count, "
            "created_at "
            "FROM facts ORDER BY created_at ASC, fact_id ASC"
        )
    finally:
        conn.close()


def migrate_from_holographic(
    src_db: Path,
    runtime: PrismRuntime,
) -> MigrationStats:
    """把 Holographic ``memory_store.db`` 全量迁到已构造好的 Prism runtime。

    Args:
        src_db: Holographic ``memory_store.db`` 绝对路径
        runtime: 已 ``build_runtime()`` 好的 Prism — 调用方负责 shutdown

    Returns:
        :class:`MigrationStats` — 调用方据此决定是否落 admin stats 或 alert

    Note:
        - content 为空 / NULL 行直接跳过（计入 failed）
        - 已存在 content（mirror_add ``is_new=False``）计入 skipped_duplicate
        - 单行失败不中止整批 — 失败计入 failed + log.warning，继续下一条
    """
    total = 0
    added = 0
    skipped = 0
    failed = 0

    for row in _iter_holographic_facts(src_db):
        total += 1
        content = (row["content"] or "").strip()
        if not content:
            failed += 1
            log.warning(
                "迁移跳过：源 fact_id=%s content 为空", row["fact_id"]
            )
            continue

        try:
            result = runtime.remember.add(
                content=content,
                category=row["category"] or "general",
            )
        except Exception as e:
            failed += 1
            log.warning(
                "迁移失败：源 fact_id=%s 异常=%s", row["fact_id"], e
            )
            continue

        if not result["is_new"]:
            skipped += 1
            continue

        # 保留 trust / created_at / counts — 一次 UPDATE 批改 4 字段
        try:
            runtime.db.execute(
                "UPDATE facts SET trust_score = ?, created_at = ?, "
                "helpful_count = ?, retrieval_count = ? WHERE fact_id = ?",
                (
                    float(row["trust_score"]),
                    row["created_at"],
                    int(row["helpful_count"]),
                    int(row["retrieval_count"]),
                    result["fact_id"],
                ),
            )
            added += 1
        except sqlite3.Error as e:
            failed += 1
            log.warning(
                "迁移后置 UPDATE 失败：fact_id=%s 异常=%s",
                result["fact_id"],
                e,
            )

    log.info(
        "迁移完成：total=%d added=%d skipped_duplicate=%d failed=%d",
        total,
        added,
        skipped,
        failed,
    )
    return MigrationStats(
        total=total, added=added, skipped_duplicate=skipped, failed=failed
    )


# ─── CLI argparse 入口 ────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m prism migrate",
        description="把 Holographic memory_store.db 迁移到 Prism DB。",
    )
    p.add_argument(
        "--from",
        dest="source",
        choices=["holographic"],
        required=True,
        help="源类型；目前仅支持 holographic",
    )
    p.add_argument(
        "--src",
        type=Path,
        required=True,
        help="源 DB 路径（Holographic memory_store.db）",
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="目标 DB 路径；未指定时按 cfg.db.path_template + user_id/profile 解析",
    )
    p.add_argument(
        "--user-id",
        default=DEFAULT_USER_ID,
        help=f"目标 user_id（默认 {DEFAULT_USER_ID!r}）— 仅 --dst 未指定时生效",
    )
    p.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help=f"目标 profile（默认 {DEFAULT_PROFILE!r}）— 仅 --dst 未指定时生效",
    )
    p.add_argument(
        "--data-home",
        default=None,
        help="数据根目录覆盖（默认 ~/.prism）— 仅 --dst 未指定时生效",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="可选 prism YAML 配置；未指定走 default_config()",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """``python -m prism.cli.migrate ...`` 入口；返 0 成功 / 非 0 失败。"""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    src: Path = args.src
    if not src.exists():
        print(f"错误：源 DB 不存在 {src}", file=sys.stderr)
        return 2

    cfg_path = args.config or discover_config_path()
    cfg = load_config(cfg_path) if cfg_path else default_config()

    dst: Path
    if args.dst is not None:
        dst = args.dst
    else:
        dst = resolve_db_path_for_user(
            cfg.db, user_id=args.user_id, profile=args.profile,
            data_home=args.data_home,
        )

    print(f"迁移：{src} → {dst}", file=sys.stderr)

    runtime = build_runtime(
        RuntimeOptions(
            db_path_override=str(dst),
            # 迁移期不需要异步 LLM 富化 + 不需要预热 prefetch（加速冷启动）
            start_worker=False,
            warmup_prefetch=False,
        )
    )
    try:
        stats = migrate_from_holographic(src, runtime)
    finally:
        runtime.shutdown()

    print(
        f"完成：total={stats.total} added={stats.added} "
        f"skipped_duplicate={stats.skipped_duplicate} failed={stats.failed}",
        file=sys.stderr,
    )
    return 0 if stats.failed == 0 else 1


MANIFEST = single(
    "migrate",
    lambda argv: main(argv),
    "migrate             Holographic → Prism 数据迁移",
)

if __name__ == "__main__":
    raise SystemExit(main())