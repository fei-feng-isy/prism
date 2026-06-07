"""``PgVectorStore`` -- PostgreSQL + pgvector 扩展的 ANN backend。

适用场景：多机部署、已有 PostgreSQL 基础设施、需要事务一致性。
支持原生 payload filter（``WHERE fact_id = ANY(...)``）；``persist`` 为 noop。

依赖：``pip install "psycopg[binary]>=3.1" pgvector``；PG 需 ``CREATE EXTENSION vector``。
向量要求：L2 归一化 float32；cosine 通过 ``<=>`` 距离算子计算。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from prism.semantic import BGE_SMALL_ZH_DIM

if TYPE_CHECKING:
    import psycopg

__all__ = ["PgVectorStore"]

log = logging.getLogger(__name__)

_REBUILD_BATCH_SIZE: Final[int] = 1000


class PgVectorStore:
    """PostgreSQL + pgvector 扩展的 KNN 索引。

    满足 :class:`~prism.vstore.VectorStore` 协议。

    Args:
        dsn: PostgreSQL 连接串（``postgresql://user:pass@host:port/db``）
        dim: 向量维度（默认 BGE-small-zh 512）；必须与表 schema 匹配
        table_name: 向量表名（默认 ``prism_vectors``）
        create_extension: 是否在初始化时执行 ``CREATE EXTENSION IF NOT EXISTS
            vector``（默认 True；若 PG 用户无 SUPERUSER 权限可设 False，由 DBA 预创建）
        create_schema: 是否在初始化时建表 + 建 HNSW 索引（默认 True）
        ef_search: pgvector HNSW 查询期探索宽度（默认 50；越大召回率越高、越慢）
        hnsw_m: HNSW 每节点连接数（默认 16；建表时生效，已存在表则忽略）
    """

    backend_name: Final[str] = "pgvector"

    def __init__(
        self,
        *,
        dsn: str,
        dim: int = BGE_SMALL_ZH_DIM,
        table_name: str = "prism_vectors",
        create_extension: bool = True,
        create_schema: bool = True,
        ef_search: int = 50,
        hnsw_m: int = 16,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim 必须为正：got {dim}")
        if not table_name.replace("_", "").isalnum():
            # 防御 SQL 注入：表名只允许 [A-Za-z0-9_]
            raise ValueError(f"非法 table_name: {table_name!r}")
        if ef_search <= 0 or hnsw_m <= 0:
            raise ValueError("ef_search / hnsw_m 必须为正")

        import psycopg
        from pgvector.psycopg import register_vector

        self.dim: int = dim
        self._table_name: str = table_name
        self._ef_search: int = ef_search
        self._hnsw_m: int = hnsw_m
        self._dsn: str = dsn

        # autocommit=True 让 CREATE EXTENSION / CREATE INDEX 立即生效
        self._conn: psycopg.Connection = psycopg.connect(dsn, autocommit=True)
        if create_extension:
            self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # register_vector 必须在 CREATE EXTENSION 之后，否则 vector type 未注册
        register_vector(self._conn)

        if create_schema:
            self._create_schema()

        self._conn.execute(f"SET hnsw.ef_search = {self._ef_search}")

        self._size: int = self._count_from_db()

    # ─── core operations ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._size

    def add(self, fact_id: int, vector: np.ndarray) -> None:
        vec = self._validate_and_cast(vector)
        try:
            self._conn.execute(
                f"INSERT INTO {self._table_name} (fact_id, embedding) "
                f"VALUES (%s, %s)",
                (int(fact_id), vec),
            )
        except Exception as e:
            # 重复 fact_id 触发主键冲突 → 暴露为 ValueError（与 local_numpy / hnswlib 一致）
            if "duplicate key" in str(e).lower() or "unique constraint" in str(e).lower():
                raise ValueError(
                    f"fact_id {fact_id} 已存在（PgVectorStore 主键冲突）"
                ) from e
            raise
        self._size += 1

    def remove(self, fact_id: int) -> None:
        cur = self._conn.execute(
            f"DELETE FROM {self._table_name} WHERE fact_id = %s",
            (int(fact_id),),
        )
        if cur.rowcount > 0:
            self._size -= 1
        # rowcount == 0 时静默 noop（协议契约）

    def update(self, fact_id: int, vector: np.ndarray) -> None:
        vec = self._validate_and_cast(vector)
        cur = self._conn.execute(
            f"UPDATE {self._table_name} SET embedding = %s WHERE fact_id = %s",
            (vec, int(fact_id)),
        )
        if cur.rowcount == 0:
            # 不存在 — 协议契约：等同 add
            self.add(fact_id, vector)

    def topk(
        self,
        query: np.ndarray,
        k: int,
        filter_fact_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if self._size == 0 or k <= 0:
            return []
        q = self._validate_and_cast(query)
        # pgvector cosine: <=> 算子返回距离（0 = identical, 2 = opposite）
        # similarity = 1 - distance，与 LocalNumpyVectorStore 的 dot product 对齐
        if filter_fact_ids is None:
            sql = (
                f"SELECT fact_id, 1 - (embedding <=> %s) AS score "
                f"FROM {self._table_name} "
                f"ORDER BY embedding <=> %s LIMIT %s"
            )
            rows = self._conn.execute(sql, (q, q, k)).fetchall()
        else:
            if not filter_fact_ids:
                return []
            # 原生 payload filter 下推 — pgvector 优势相对 hnswlib/faiss
            sql = (
                f"SELECT fact_id, 1 - (embedding <=> %s) AS score "
                f"FROM {self._table_name} "
                f"WHERE fact_id = ANY(%s) "
                f"ORDER BY embedding <=> %s LIMIT %s"
            )
            rows = self._conn.execute(
                sql,
                (q, list(filter_fact_ids), q, k),
            ).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]

    def fetch(self, fact_ids: list[int]) -> dict[int, np.ndarray]:
        if not fact_ids:
            return {}
        rows = self._conn.execute(
            f"SELECT fact_id, embedding FROM {self._table_name} "
            f"WHERE fact_id = ANY(%s)",
            (list(fact_ids),),
        ).fetchall()
        return {
            int(r[0]): np.asarray(r[1], dtype=np.float32)
            for r in rows
        }

    def rebuild_from_iter(self, pairs: Iterable[tuple[int, np.ndarray]]) -> None:
        ids: list[int] = []
        vecs: list[np.ndarray] = []
        for fid, vec in pairs:
            ids.append(int(fid))
            vecs.append(self._validate_and_cast(vec))
        n = len(ids)
        if len(set(ids)) != n:
            raise ValueError("rebuild_from_iter: 输入含重复 fact_id")

        # 在单事务内 TRUNCATE + batch INSERT —— 原子操作，失败回滚保留旧数据
        with self._conn.transaction():
            self._conn.execute(f"TRUNCATE {self._table_name}")
            # 分批写入避免单个语句体积过大（psycopg 默认 8MB packet 上限）
            for start in range(0, n, _REBUILD_BATCH_SIZE):
                end = min(start + _REBUILD_BATCH_SIZE, n)
                # executemany 把多条 INSERT 打包成 pipeline 减少 RTT
                self._conn.cursor().executemany(
                    f"INSERT INTO {self._table_name} (fact_id, embedding) "
                    f"VALUES (%s, %s)",
                    [(ids[i], vecs[i]) for i in range(start, end)],
                )
        self._size = n

    def persist(self) -> None:
        # 数据库 backend：数据已在事务中持久化，noop
        return

    def stats(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "dim": self.dim,
            "ntotal": self._size,
            "table_name": self._table_name,
            "ef_search": self._ef_search,
            "hnsw_m": self._hnsw_m,
            "fallback_filter": False,  # 原生 WHERE fact_id = ANY(...) 下推
            "dsn_host": self._dsn_host(),
        }

    def close(self) -> None:
        """关闭底层 PG 连接（测试 / shutdown hook 用）。"""
        if self._conn is not None and not self._conn.closed:
            self._conn.close()

    # ─── internals ───────────────────────────────────────────────────────

    def _create_schema(self) -> None:
        """建表 + HNSW 索引（IF NOT EXISTS，幂等）。"""
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self._table_name} ("
            f"  fact_id BIGINT PRIMARY KEY,"
            f"  embedding vector({self.dim}) NOT NULL"
            f")"
        )
        # HNSW 索引：m 在建表时生效；ef_construction 默认 64（pgvector 推荐）
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self._table_name}_embedding_idx "
            f"ON {self._table_name} USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = {self._hnsw_m})"
        )

    def _count_from_db(self) -> int:
        row = self._conn.execute(
            f"SELECT COUNT(*) FROM {self._table_name}"
        ).fetchone()
        return int(row[0]) if row else 0

    def _validate_and_cast(self, vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != self.dim:
            raise ValueError(
                f"vector shape mismatch: expected ({self.dim},), got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("vector contains NaN/Inf")
        return arr

    def _dsn_host(self) -> str | None:
        """从 DSN 抽出 host:port 用于 stats 日志（不暴露 user/password）。"""
        try:
            import psycopg

            info = psycopg.conninfo.conninfo_to_dict(self._dsn)
            host = info.get("host", "localhost")
            port = info.get("port", "5432")
            return f"{host}:{port}"
        except Exception:
            return None
