"""``python -m prism vstore-migrate --to <backend>`` CLI。

把 ``facts.semantic_vector`` 从 SQLite 全量重建到指定 backend，并更新
``effective_backend``。直接从 SQLite 流式读 BLOB，不经 source vstore。

与 reindex 的区别：reindex 重编码 embedding，vstore-migrate 只切换 ANN 索引
backend，不动 SQLite 内容。
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

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
from prism.vstore import write_effective_backend

if TYPE_CHECKING:
    from collections.abc import Iterator

    from prism.config import PrismConfig
    from prism.vstore import VectorStore

log = logging.getLogger(__name__)

__all__ = [
    "MigrateStats",
    "build_target_vstore",
    "iter_facts_vectors",
    "main",
    "vstore_migrate",
]


_SUPPORTED_TARGETS = ("local_numpy", "hnswlib", "faiss", "pgvector", "qdrant")


@dataclass(frozen=True, slots=True)
class MigrateStats:
    """单次 vstore-migrate 的统计快照。"""

    target_backend: str
    """目标 backend 名（与 ``cfg.vector_store.backend`` 写入值一致）。"""
    sqlite_count: int
    """从 SQLite 流出的 ``(fact_id, vector)`` 对数。"""
    vstore_count: int
    """rebuild 后目标 vstore 的 ``len()``；应等于 sqlite_count。"""
    target_dim: int
    """目标 vstore 的 dim；用于诊断 dim mismatch。"""
    effective_backend_written: bool
    """是否更新了 ``prism_stats.vstore.effective_backend``。"""


# ─── 核心函数 ─────────────────────────────────────────────────────────────


def iter_facts_vectors(
    db: sqlite3.Connection,
) -> Iterator[tuple[int, np.ndarray]]:
    """流式产出 ``(fact_id, vector)`` 对 —— 来自 ``facts WHERE status='active'
    AND semantic_vector IS NOT NULL``。

    流式 / 不一次性 list：千万级数据下避免一次性 4GB+ 内存占用。
    """
    cur = db.execute(
        "SELECT fact_id, semantic_vector FROM facts "
        "WHERE status='active' AND semantic_vector IS NOT NULL "
        "ORDER BY fact_id ASC"
    )
    for row in cur:
        blob = row["semantic_vector"]
        if blob is None:
            continue
        # frombuffer 返 view → copy 让 vstore 拿到独立内存（add 的契约）
        vec = np.frombuffer(blob, dtype=np.float32).copy()
        yield int(row["fact_id"]), vec


def build_target_vstore(
    target: str,
    *,
    cfg: PrismConfig,
    dim: int,
    vstore_path: Path | None,
) -> VectorStore:
    """实例化干净的目标 backend（empty store）。

    Args:
        target: ``local_numpy`` / ``hnswlib`` / ``faiss`` / ``pgvector`` / ``qdrant``
        cfg: PrismConfig；pgvector/qdrant 从中读 env 变量名
        dim: 目标 vstore 的 dim（与 facts.semantic_vector 一致）
        vstore_path: 文件 backend 的持久化路径；外置 backend 忽略
    """
    if target == "local_numpy":
        from prism.vstore.local_numpy import LocalNumpyVectorStore

        return LocalNumpyVectorStore(
            dim=dim,
            path=str(vstore_path) if vstore_path else None,
            load=False,  # 全量重建：不能 load 旧文件（可能 dim 不一致）
        )
    if target == "hnswlib":
        from prism.vstore.hnswlib_store import HnswlibVectorStore

        return HnswlibVectorStore(
            dim=dim,
            path=str(vstore_path) if vstore_path else None,
            load=False,
        )
    if target == "faiss":
        from prism.vstore.faiss_store import FaissVectorStore

        return FaissVectorStore(
            dim=dim,
            path=str(vstore_path) if vstore_path else None,
            load=False,
        )
    if target == "pgvector":
        import os

        from prism.vstore.pgvector_store import PgVectorStore

        dsn = os.environ.get(cfg.vector_store.pgvector.dsn_env)
        if not dsn:
            raise ValueError(
                f"--to pgvector 需 env {cfg.vector_store.pgvector.dsn_env}=postgresql://..."
            )
        return PgVectorStore(
            dsn=dsn, dim=dim, table_name=cfg.vector_store.pgvector.table_name
        )
    if target == "qdrant":
        import os

        from prism.vstore.qdrant_store import QdrantVectorStore

        url = os.environ.get(cfg.vector_store.qdrant.url_env)
        if not url:
            raise ValueError(
                f"--to qdrant 需 env {cfg.vector_store.qdrant.url_env}=http://..."
            )
        api_key = os.environ.get(cfg.vector_store.qdrant.api_key_env)
        return QdrantVectorStore(
            url=url,
            api_key=api_key,
            dim=dim,
            collection_name=cfg.vector_store.qdrant.collection_name,
        )
    raise ValueError(
        f"未支持的 --to {target!r}；可选 {list(_SUPPORTED_TARGETS)}"
    )


def vstore_migrate(
    db: sqlite3.Connection,
    target: str,
    *,
    cfg: PrismConfig,
    vstore_path: Path | None,
    dim: int | None = None,
    dry_run: bool = False,
) -> MigrateStats:
    """全量 rebuild target backend 并持久化 effective_backend。

    Args:
        db: 已 ``init_schema`` 的 SQLite 连接
        target: 目标 backend 名
        cfg: PrismConfig
        vstore_path: 文件 backend 的 .npz/.hnsw/.bin 路径（None = 内存模式）
        dim: 目标 dim；None 时从首个 semantic_vector blob 推断
        dry_run: True 时不实例化 target、不写 effective_backend

    Returns:
        :class:`MigrateStats`
    """
    if target not in _SUPPORTED_TARGETS:
        raise ValueError(
            f"--to {target!r} 不在 {list(_SUPPORTED_TARGETS)}"
        )

    # 1) 推断 dim（如未传）+ count active vectors
    if dim is None:
        first = db.execute(
            "SELECT semantic_vector FROM facts "
            "WHERE status='active' AND semantic_vector IS NOT NULL LIMIT 1"
        ).fetchone()
        if first is None or first["semantic_vector"] is None:
            log.warning("DB 中无 active+semantic_vector 的 fact；migrate 退化为空 rebuild")
            dim = 512  # fallback — 空 rebuild 时 dim 仅影响 stats
        else:
            dim = int(
                np.frombuffer(first["semantic_vector"], dtype=np.float32).shape[0]
            )

    sqlite_count = int(
        db.execute(
            "SELECT COUNT(*) FROM facts "
            "WHERE status='active' AND semantic_vector IS NOT NULL"
        ).fetchone()[0]
    )

    if dry_run:
        log.info(
            "dry-run: target=%s dim=%d sqlite_count=%d", target, dim, sqlite_count
        )
        return MigrateStats(
            target_backend=target,
            sqlite_count=sqlite_count,
            vstore_count=0,
            target_dim=dim,
            effective_backend_written=False,
        )

    # 2) 实例化 target + 全量 rebuild
    target_vs = build_target_vstore(
        target, cfg=cfg, dim=dim, vstore_path=vstore_path
    )
    try:
        target_vs.rebuild_from_iter(iter_facts_vectors(db))
        target_vs.persist()
        vstore_count = len(target_vs)
    finally:
        close = getattr(target_vs, "close", None)
        if callable(close):
            close()

    # 3) 写 effective_backend（auto 模式下次启动会读）
    write_effective_backend(db, target)

    if vstore_count != sqlite_count:
        log.warning(
            "migrate: vstore_count=%d != sqlite_count=%d（部分行可能 dim 不一致被 vstore 拒）",
            vstore_count, sqlite_count,
        )

    log.info(
        "migrate 完成: target=%s sqlite_count=%d vstore_count=%d dim=%d",
        target, sqlite_count, vstore_count, dim,
    )
    return MigrateStats(
        target_backend=target,
        sqlite_count=sqlite_count,
        vstore_count=vstore_count,
        target_dim=dim,
        effective_backend_written=True,
    )


# ─── CLI argparse 入口 ────────────────────────────────────────────────────


_VSTORE_PATH_SUFFIX: dict[str, str] = {
    "local_numpy": ".vstore.npz",
    "hnswlib": ".vstore.hnsw",
    "faiss": ".vstore.faiss",
}


def _default_vstore_path(db_path: Path, target: str) -> Path | None:
    """为文件 backend 推断 .vstore.* 持久化路径；外置 backend 返 None。"""
    suffix = _VSTORE_PATH_SUFFIX.get(target)
    if suffix is None:
        return None
    return db_path.with_suffix(suffix)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m prism vstore-migrate",
        description=(
            "把 facts.semantic_vector 全量重建到指定 backend，并更新 "
            "prism_stats.vstore.effective_backend（auto 模式下次启动会读）。"
        ),
    )
    p.add_argument(
        "--to",
        required=True,
        choices=_SUPPORTED_TARGETS,
        help="目标 vector store backend",
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
        help=f"目标 profile（默认 {DEFAULT_PROFILE!r}）",
    )
    p.add_argument(
        "--data-home",
        default=None,
        help="数据根目录覆盖（默认 ~/.prism）— 仅 --db 未指定时生效",
    )
    p.add_argument(
        "--vstore-path",
        type=Path,
        default=None,
        help=(
            "文件 backend (local_numpy/hnswlib/faiss) 的持久化路径；"
            "默认 <db_path>.vstore.{npz|hnsw|faiss}"
        ),
    )
    p.add_argument(
        "--dim",
        type=int,
        default=None,
        help="目标 dim；不传则从 facts.semantic_vector 首行推断",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="仅扫描 + 报告，不实例化 target 也不写 effective_backend",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="可选 prism YAML 配置；未指定走 default_config()",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """``python -m prism vstore-migrate ...`` 入口；返 0 成功 / 非 0 失败。"""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

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

    vstore_path = args.vstore_path or _default_vstore_path(db_path, args.to)

    print(
        f"vstore-migrate：{db_path} → backend={args.to} "
        f"vstore_path={vstore_path} dry_run={args.dry_run}",
        file=sys.stderr,
    )

    db = bootstrap(str(db_path))
    try:
        stats = vstore_migrate(
            db,
            args.to,
            cfg=cfg,
            vstore_path=vstore_path,
            dim=args.dim,
            dry_run=args.dry_run,
        )
    finally:
        db.close()

    print(
        f"完成：target={stats.target_backend} sqlite={stats.sqlite_count} "
        f"vstore={stats.vstore_count} dim={stats.target_dim} "
        f"effective_backend_written={stats.effective_backend_written}",
        file=sys.stderr,
    )
    return 0 if stats.vstore_count == stats.sqlite_count else 1


MANIFEST = single(
    "vstore-migrate",
    lambda argv: main(argv),
    "vstore-migrate      vector store backend 切换",
)

if __name__ == "__main__":
    raise SystemExit(main())