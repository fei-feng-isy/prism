"""Backend 工厂 + 混合维度期 guard + auto 升级单调性。

根据 ``PrismConfig`` + SQLite 当前状态选择 ``VectorStore`` 实现。

混合维度期（``COUNT(DISTINCT embedding_model) > 1``）强制降级到
``LocalNumpyVectorStore``；auto 模式按 active fact 数量选 backend 并保持
升级单调性（一旦升级不自动降级，降级需手动 ``vstore-migrate``）。
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import TYPE_CHECKING, Final

from prism.semantic import BGE_SMALL_ZH_DIM
from prism.vstore.local_numpy import LocalNumpyVectorStore
from prism.vstore.protocol import VectorStore

if TYPE_CHECKING:
    from prism.config import PrismConfig

__all__ = [
    "AUTO_BACKEND_RANK",
    "count_active_facts",
    "count_distinct_embedding_models",
    "create_vstore",
    "decide_auto_backend",
    "read_effective_backend",
    "write_effective_backend",
]

log = logging.getLogger(__name__)


# auto 升级路径上的 backend 排名 —— 仅 local_numpy/hnswlib/faiss 参与
# （pgvector/qdrant 是用户 opt-in 的外置依赖，不进 auto）
AUTO_BACKEND_RANK: Final[dict[str, int]] = {
    "local_numpy": 0,
    "hnswlib": 1,
    "faiss": 2,
}

_EFFECTIVE_BACKEND_KEY: Final[str] = "vstore.effective_backend"


def create_vstore(
    cfg: PrismConfig,
    db: sqlite3.Connection,
    *,
    dim: int = BGE_SMALL_ZH_DIM,
    path: str | os.PathLike[str] | None = None,
) -> VectorStore:
    """根据配置 + DB 当前状态创建合适的 :class:`VectorStore` 实例。

    Args:
        cfg: 完整的 PrismConfig；读取 ``cfg.vector_store.*`` 字段。
        db: SQLite 连接（已通过 ``init_schema`` 建表），用于查 distinct
            embedding_model 计数（mixed-dim guard）与 active facts 计数
            （auto 决策）。
        dim: 向量维度，默认 :data:`~prism.semantic.BGE_SMALL_ZH_DIM` (512)。
        path: 文件 backend（local_numpy / hnswlib / faiss）的持久化路径；
            pgvector / qdrant 忽略。``None`` = 文件 backend 走纯内存。

    Returns:
        实现 :class:`VectorStore` 协议的实例。

    Raises:
        ValueError: pgvector 缺 DSN 或 qdrant 缺 URL 等配置错误。
    """
    if count_distinct_embedding_models(db) > 1:
        # 混合维度期 — 强制 local_numpy，无视 cfg.vector_store.backend
        return LocalNumpyVectorStore(
            dim=dim,
            path=path,
            forced_local_due_to_mixed_dim=True,
        )

    backend = cfg.vector_store.backend
    if backend == "auto":
        backend = decide_auto_backend(cfg, db)
        # auto 决策持久化（即使是 local_numpy 也写一次，便于运维查 stats）
        write_effective_backend(db, backend)

    if backend == "local_numpy":
        return LocalNumpyVectorStore(dim=dim, path=path)
    if backend == "hnswlib":
        from prism.vstore.hnswlib_store import HnswlibVectorStore

        return HnswlibVectorStore(dim=dim, path=path)
    if backend == "faiss":
        from prism.vstore.faiss_store import FaissVectorStore

        return FaissVectorStore(dim=dim, path=path)
    if backend == "pgvector":
        return _create_pgvector(cfg, dim)
    if backend == "qdrant":
        return _create_qdrant(cfg, dim)

    # _check_literal 已在 config 加载时拦截非法值；防御性兜底
    raise ValueError(f"未知 vector_store.backend: {backend!r}")


# ─── auto 决策 ─────────────────────────────────────────────────────────────


def decide_auto_backend(cfg: PrismConfig, db: sqlite3.Connection) -> str:
    """auto 模式下选哪个 backend，满足升级单调性。

    候选 = 按 ``active_facts`` + ``cfg.auto_thresholds`` 算出的最合适项
    （local_numpy / hnswlib / faiss）。

    最终 = ``max(候选, 持久化 effective_backend)`` —— 一旦升过就不再降。
    """
    n = count_active_facts(db)
    thresholds = cfg.vector_store.auto_thresholds
    if n > thresholds.faiss:
        candidate = "faiss"
    elif n > thresholds.hnswlib:
        candidate = "hnswlib"
    else:
        candidate = "local_numpy"

    persisted = read_effective_backend(db)
    if persisted is not None and persisted in AUTO_BACKEND_RANK:
        # 单调性：取 rank 较高者
        if AUTO_BACKEND_RANK[persisted] > AUTO_BACKEND_RANK[candidate]:
            log.info(
                "auto: monotonic guard keeps %s (active_facts=%d suggested %s)",
                persisted,
                n,
                candidate,
            )
            return persisted
    return candidate


# ─── 持久化 effective_backend（prism_stats kv 表）─────────────────────────


def read_effective_backend(db: sqlite3.Connection) -> str | None:
    """读 ``prism_stats`` 中持久化的 ``vstore.effective_backend``；不存在返 None。"""
    row = db.execute(
        "SELECT value FROM prism_stats WHERE key = ?",
        (_EFFECTIVE_BACKEND_KEY,),
    ).fetchone()
    return str(row[0]) if row and row[0] is not None else None


def write_effective_backend(db: sqlite3.Connection, backend: str) -> None:
    """upsert ``vstore.effective_backend`` 到 ``prism_stats``。"""
    db.execute(
        "INSERT INTO prism_stats (key, value, updated_at) "
        "VALUES (?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET "
        "  value = excluded.value, updated_at = CURRENT_TIMESTAMP",
        (_EFFECTIVE_BACKEND_KEY, backend),
    )


# ─── SQL helpers ───────────────────────────────────────────────────────────


def count_active_facts(db: sqlite3.Connection) -> int:
    """``COUNT(*) FROM facts WHERE status='active'`` —— auto 阈值判断依据。"""
    row = db.execute(
        "SELECT COUNT(*) FROM facts WHERE status = 'active'"
    ).fetchone()
    return int(row[0]) if row else 0


def count_distinct_embedding_models(db: sqlite3.Connection) -> int:
    """查 ``facts`` 表中当前存在的 DISTINCT ``embedding_model`` 数量。

    只统计 ``semantic_vector IS NOT NULL`` 的行（未生成向量的 fact 不算）；
    同时排除 ``embedding_model IS NULL``（理论上不应出现，但防御性过滤）。

    ``> 1`` 即为「混合维度期」— 外置 backend 不支持单实例多维度，需强制 local_numpy。
    """
    row = db.execute(
        "SELECT COUNT(DISTINCT embedding_model) FROM facts "
        "WHERE semantic_vector IS NOT NULL AND embedding_model IS NOT NULL"
    ).fetchone()
    return int(row[0]) if row else 0


# ─── 外置 backend 实例化（pgvector / qdrant 需读环境变量）─────────────────


def _create_pgvector(cfg: PrismConfig, dim: int) -> VectorStore:
    from prism.vstore.pgvector_store import PgVectorStore

    dsn = os.environ.get(cfg.vector_store.pgvector.dsn_env)
    if not dsn:
        raise ValueError(
            f"vector_store.backend='pgvector' 但环境变量 "
            f"{cfg.vector_store.pgvector.dsn_env!r} 未设置；"
            f"export {cfg.vector_store.pgvector.dsn_env}=postgresql://..."
        )
    return PgVectorStore(
        dsn=dsn,
        dim=dim,
        table_name=cfg.vector_store.pgvector.table_name,
    )


def _create_qdrant(cfg: PrismConfig, dim: int) -> VectorStore:
    from prism.vstore.qdrant_store import QdrantVectorStore

    url = os.environ.get(cfg.vector_store.qdrant.url_env)
    if not url:
        raise ValueError(
            f"vector_store.backend='qdrant' 但环境变量 "
            f"{cfg.vector_store.qdrant.url_env!r} 未设置；"
            f"export {cfg.vector_store.qdrant.url_env}=http://localhost:6333"
        )
    api_key = os.environ.get(cfg.vector_store.qdrant.api_key_env)
    return QdrantVectorStore(
        url=url,
        api_key=api_key,
        dim=dim,
        collection_name=cfg.vector_store.qdrant.collection_name,
    )
