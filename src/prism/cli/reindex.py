"""`prism reindex --model <name>` CLI + 核心函数。

升级 embedding 模型时全量重编码 active facts 并重建 ``.vstore.npz``。
幂等：已升级的 fact 自动跳过，中断后重跑只补剩余行。
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from collections.abc import Callable
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
from prism.db import bootstrap
from prism.service.repair_service import _rebuild_vstore

if TYPE_CHECKING:
    from prism.semantic import SemanticBackend

log = logging.getLogger(__name__)

__all__ = [
    "ReindexStats",
    "main",
    "reindex",
]


@dataclass(frozen=True, slots=True)
class ReindexStats:
    """单次 reindex 的统计快照。"""

    total_active: int
    """active 状态 fact 总数。"""
    already_on_target: int
    """已经是目标 model 的 fact 数（跳过，幂等关键）。"""
    reencoded: int
    """本次新写入 / 升级的 fact 数。"""
    failed: int
    """encode / UPDATE 失败的 fact 数（详情进 log.warning）。"""
    vstore_rebuilt: bool
    """``.vstore.npz`` 是否被重建（dry-run 或 0 fact 时为 False）。"""


# ─── 核心函数 ─────────────────────────────────────────────────────────────


def _select_targets(
    db: sqlite3.Connection,
    new_model: str,
) -> list[tuple[int, str]]:
    """选出需要重编码的 active facts —
    ``embedding_model != new_model`` 或 NULL（首次写入失败补救）。"""
    rows = db.execute(
        "SELECT fact_id, content FROM facts "
        "WHERE status = 'active' "
        "AND (embedding_model IS NULL OR embedding_model != ?) "
        "ORDER BY fact_id ASC",
        (new_model,),
    ).fetchall()
    return [(int(r["fact_id"]), str(r["content"])) for r in rows]


def reindex(
    db: sqlite3.Connection,
    backend: SemanticBackend,
    *,
    vstore_path: Path | None,
    batch_size: int = 64,
    dry_run: bool = False,
) -> ReindexStats:
    """全量重编码 + 重建 vstore。调用方负责 db / backend 生命周期。

    Args:
        db: 打开的 Prism DB 连接（已 ``init_schema``）
        backend: 满足 :class:`SemanticBackend` 协议的目标编码器；其 ``name``
            字段会被写入 ``facts.embedding_model``
        vstore_path: ``.vstore.npz`` 落盘路径；None = 纯内存（测试场景）
        batch_size: 单次 ``encode_batch`` 的 fact 数；过大会撑爆 GPU 显存，
            过小会失去批处理收益。CPU bge-small 实测 64 为 sweet spot
        dry_run: True 时仅扫描 + 报告，不写 DB / 不重建 vstore

    Returns:
        :class:`ReindexStats`；调用方据此决定 exit code。
    """
    new_model = backend.name
    target_dim = backend.dim

    # 1) 总览 + 目标行（双查：active 总数 + 待升级列表）
    total_active = int(
        db.execute(
            "SELECT COUNT(*) FROM facts WHERE status='active'"
        ).fetchone()[0]
    )
    targets = _select_targets(db, new_model)
    already_on_target = total_active - len(targets)

    if dry_run:
        log.info(
            "dry-run：active=%d 已是 %s 的=%d 待重编=%d",
            total_active, new_model, already_on_target, len(targets),
        )
        return ReindexStats(
            total_active=total_active,
            already_on_target=already_on_target,
            reencoded=0,
            failed=0,
            vstore_rebuilt=False,
        )

    if not targets:
        log.info("无 fact 需要重编码（全部已在 %s）", new_model)
        return ReindexStats(
            total_active=total_active,
            already_on_target=already_on_target,
            reencoded=0,
            failed=0,
            vstore_rebuilt=False,
        )

    # 2) 批量 encode + UPDATE
    reencoded = 0
    failed = 0
    for start in range(0, len(targets), batch_size):
        chunk = targets[start : start + batch_size]
        texts = [text for _, text in chunk]

        try:
            mat = backend.encode_batch(texts)
        except Exception as e:
            failed += len(chunk)
            log.warning(
                "encode_batch 失败 [%d, %d)：%s", start, start + len(chunk), e
            )
            continue

        if mat.shape != (len(chunk), target_dim):
            failed += len(chunk)
            log.warning(
                "backend 返回 shape=%s 与期望 (%d, %d) 不符；跳过此批",
                mat.shape, len(chunk), target_dim,
            )
            continue

        # 一行一 UPDATE — sqlite 在 isolation_level=None 下自动语句级原子，
        # 单行失败不污染整批；批级事务的代价（rollback 整批）不值
        for (fact_id, _), vec in zip(chunk, mat, strict=True):
            try:
                db.execute(
                    "UPDATE facts SET semantic_vector = ?, embedding_model = ? "
                    "WHERE fact_id = ?",
                    (vec.astype("float32").tobytes(), new_model, fact_id),
                )
                reencoded += 1
            except sqlite3.Error as e:
                failed += 1
                log.warning("UPDATE fact_id=%s 失败：%s", fact_id, e)

    # 3) 重建 vstore：从最新 facts.semantic_vector 全量重建
    vstore_rebuilt = _rebuild_vstore(
        db, vstore_path=vstore_path, dim=target_dim, new_model=new_model
    )

    log.info(
        "reindex 完成：active=%d 待升=%d 实升=%d failed=%d vstore_rebuilt=%s",
        total_active, len(targets), reencoded, failed, vstore_rebuilt,
    )
    return ReindexStats(
        total_active=total_active,
        already_on_target=already_on_target,
        reencoded=reencoded,
        failed=failed,
        vstore_rebuilt=vstore_rebuilt,
    )


# ─── CLI argparse 入口 ────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m prism reindex",
        description="全量重编码 facts.semantic_vector 到指定 embedding 模型。",
    )
    p.add_argument(
        "--model",
        required=True,
        help="目标 embedding 模型名（HuggingFace ID 或本地路径，例如 "
        "'BAAI/bge-base-zh-v1.5'）",
    )
    p.add_argument(
        "--dim",
        type=int,
        default=None,
        help="向量维度；不传则按 sentence-transformers 模型实际维度自动检测",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Prism DB 路径；未指定时按 cfg + user_id/profile 解析",
    )
    p.add_argument(
        "--user-id",
        default=DEFAULT_USER_ID,
        help=f"目标 user_id（默认 {DEFAULT_USER_ID!r}）— 仅 --db 未指定时生效",
    )
    p.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help=f"目标 profile（默认 {DEFAULT_PROFILE!r}）— 仅 --db 未指定时生效",
    )
    p.add_argument(
        "--data-home",
        default=None,
        help="数据根目录覆盖（默认 ~/.prism）— 仅 --db 未指定时生效",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="单批 encode_batch 大小（默认 64）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="仅扫描 + 报告待升量，不写 DB / 不重建 vstore",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="可选 prism YAML 配置；未指定走 default_config()",
    )
    return p


def _default_backend_factory(model: str, dim: int | None) -> SemanticBackend:
    """默认 backend 构造：LocalBgeBackend。延迟 import 避免 CLI 启动慢。"""
    from prism.semantic import BGE_SMALL_ZH_DIM, LocalBgeBackend

    return LocalBgeBackend(model_name=model, dim=dim if dim is not None else BGE_SMALL_ZH_DIM)


def main(
    argv: list[str] | None = None,
    *,
    backend_factory: Callable[[str, int | None], SemanticBackend] | None = None,
) -> int:
    """``python -m prism reindex ...`` 入口；返 0 成功 / 非 0 失败。

    Args:
        argv: 命令行参数；None 走 sys.argv
        backend_factory: 注入点，让单元测试可传 mock；默认 LocalBgeBackend
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg_path = args.config or discover_config_path()
    cfg = load_config(cfg_path) if cfg_path else default_config()

    db_path: Path
    if args.db is not None:
        db_path = args.db
    else:
        db_path = resolve_db_path_for_user(
            cfg.db, user_id=args.user_id, profile=args.profile,
            data_home=args.data_home,
        )

    if not db_path.exists():
        print(f"错误：Prism DB 不存在 {db_path}", file=sys.stderr)
        return 2

    factory = backend_factory or _default_backend_factory
    backend = factory(args.model, args.dim)

    vstore_path = db_path.with_suffix(".vstore.npz")

    print(
        f"reindex：{db_path} → model={args.model} dim={backend.dim} "
        f"batch={args.batch_size} dry_run={args.dry_run}",
        file=sys.stderr,
    )

    db = bootstrap(str(db_path))
    try:
        stats = reindex(
            db,
            backend,
            vstore_path=vstore_path,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    finally:
        db.close()

    print(
        f"完成：total_active={stats.total_active} "
        f"already={stats.already_on_target} reencoded={stats.reencoded} "
        f"failed={stats.failed} vstore_rebuilt={stats.vstore_rebuilt}",
        file=sys.stderr,
    )
    return 0 if stats.failed == 0 else 1


MANIFEST = single(
    "reindex",
    lambda argv: main(argv),
    "reindex             嵌入模型升级 / 重建",
)

if __name__ == "__main__":
    raise SystemExit(main())