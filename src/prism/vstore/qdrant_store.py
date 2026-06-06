"""``QdrantVectorStore`` -- qdrant client ANN backend。

适用场景：独立向量服务、需要 payload 过滤 + 多 collection 隔离。
支持原生 ID 过滤（``HasIdCondition``）；``persist`` 为 noop。

依赖：``pip install qdrant-client``；生产环境需 qdrant server。
向量要求：L2 归一化 float32；score 即 cosine similarity。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from prism.semantic import BGE_SMALL_ZH_DIM

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

__all__ = ["QdrantVectorStore"]

log = logging.getLogger(__name__)

_UPSERT_BATCH_SIZE: Final[int] = 1000


class QdrantVectorStore:
    """qdrant client + cosine HNSW collection。

    满足 :class:`~prism.vstore.VectorStore` 协议。

    Args:
        url: qdrant 服务 URL（``http://host:6333`` / ``grpc://host:6334`` /
            ``":memory:"`` 测试模式）。client 不去管 client 本身是否远程。
        dim: 向量维度（默认 BGE-small-zh 512）；必须与 collection schema 匹配
        collection_name: collection 名（默认 ``prism``）
        api_key: qdrant cloud 的 API key（self-hosted 留空）
        create_collection: 是否在 init 时创建 collection（默认 True；幂等）
        client: 注入自定义 ``QdrantClient`` 实例（测试 / 复用连接池场景）
    """

    backend_name: Final[str] = "qdrant"

    def __init__(
        self,
        *,
        url: str = ":memory:",
        dim: int = BGE_SMALL_ZH_DIM,
        collection_name: str = "prism",
        api_key: str | None = None,
        create_collection: bool = True,
        client: QdrantClient | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim 必须为正：got {dim}")
        if not collection_name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"非法 collection_name: {collection_name!r}")

        from qdrant_client import QdrantClient

        self.dim: int = dim
        self._collection: str = collection_name
        self._url: str = url

        if client is not None:
            self._client = client
        elif url == ":memory:" or url.startswith("./") or url.startswith("/"):
            # 本地嵌入模式（测试 / 单机部署）—— 不是 HTTP URL
            self._client = QdrantClient(location=url)
        else:
            self._client = QdrantClient(url=url, api_key=api_key)

        if create_collection:
            self._ensure_collection()

        # 本地 _size 缓存（避免每次 __len__ 走 RTT）
        self._size: int = self._count_from_server()

    # ─── core operations ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._size

    def add(self, fact_id: int, vector: np.ndarray) -> None:
        from qdrant_client.models import PointStruct

        vec = self._validate_and_cast(vector)
        # qdrant upsert 在重复 id 时静默覆盖；我们需要先检查暴露重复
        if self._exists(fact_id):
            raise ValueError(
                f"fact_id {fact_id} 已存在（QdrantVectorStore 选择抛错以暴露 bug）"
            )
        self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=int(fact_id), vector=vec.tolist())],
        )
        self._size += 1

    def remove(self, fact_id: int) -> None:
        from qdrant_client.models import PointIdsList

        # 协议契约：noop 幂等；用 _exists 显式判断以维持 _size 准确
        if not self._exists(fact_id):
            return
        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=[int(fact_id)]),
        )
        self._size -= 1

    def update(self, fact_id: int, vector: np.ndarray) -> None:
        from qdrant_client.models import PointStruct

        vec = self._validate_and_cast(vector)
        existed = self._exists(fact_id)
        # upsert 同时处理 add 和 update
        self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=int(fact_id), vector=vec.tolist())],
        )
        if not existed:
            self._size += 1

    def topk(
        self,
        query: np.ndarray,
        k: int,
        filter_fact_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if self._size == 0 or k <= 0:
            return []
        q = self._validate_and_cast(query)

        if filter_fact_ids is not None and not filter_fact_ids:
            return []

        from qdrant_client.models import Filter, HasIdCondition

        qfilter = None
        if filter_fact_ids is not None:
            qfilter = Filter(
                must=[HasIdCondition(has_id=sorted(int(i) for i in filter_fact_ids))]
            )

        res = self._client.query_points(
            collection_name=self._collection,
            query=q.tolist(),
            query_filter=qfilter,
            limit=k,
            with_vectors=False,
            with_payload=False,
        )
        return [(int(p.id), float(p.score)) for p in res.points]

    def fetch(self, fact_ids: list[int]) -> dict[int, np.ndarray]:
        if not fact_ids:
            return {}
        points = self._client.retrieve(
            collection_name=self._collection,
            ids=[int(i) for i in fact_ids],
            with_vectors=True,
            with_payload=False,
        )
        return {
            int(p.id): np.asarray(p.vector, dtype=np.float32)
            for p in points
            if p.vector is not None
        }

    def rebuild_from_iter(self, pairs: Iterable[tuple[int, np.ndarray]]) -> None:
        from qdrant_client.models import PointStruct

        ids: list[int] = []
        vecs: list[np.ndarray] = []
        for fid, vec in pairs:
            ids.append(int(fid))
            vecs.append(self._validate_and_cast(vec))
        n = len(ids)
        if len(set(ids)) != n:
            raise ValueError("rebuild_from_iter: 输入含重复 fact_id")

        # qdrant 的"原子重建" = delete_collection + create_collection；
        # 失败时 collection 已删 → 无法回滚，但比保留半重建状态更可预测
        # 替代方案是 TRUNCATE 等价：先全删 → 再 upsert，缺点是中间窗口空
        self._client.delete_collection(self._collection)
        self._ensure_collection()

        for start in range(0, n, _UPSERT_BATCH_SIZE):
            end = min(start + _UPSERT_BATCH_SIZE, n)
            self._client.upsert(
                collection_name=self._collection,
                points=[
                    PointStruct(id=ids[i], vector=vecs[i].tolist())
                    for i in range(start, end)
                ],
            )
        self._size = n

    def persist(self) -> None:
        # 服务端 backend：数据已在 qdrant 持久化，noop
        return

    def stats(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "dim": self.dim,
            "ntotal": self._size,
            "collection_name": self._collection,
            "url": self._url,
            "fallback_filter": False,  # 原生 HasIdCondition 下推
        }

    def close(self) -> None:
        """关闭底层 qdrant client（测试 / shutdown hook 用）。"""
        if self._client is not None:
            self._client.close()

    # ─── internals ───────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self._client.collection_exists(self._collection):
            # 已存在：校验 dim 一致（防止 schema 漂移）
            info = self._client.get_collection(self._collection)
            existing_dim = info.config.params.vectors.size
            if existing_dim != self.dim:
                raise ValueError(
                    f"qdrant collection {self._collection!r} 已存在但 dim="
                    f"{existing_dim} != configured dim={self.dim}"
                )
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )

    def _count_from_server(self) -> int:
        return int(self._client.count(self._collection).count)

    def _exists(self, fact_id: int) -> bool:
        points = self._client.retrieve(
            collection_name=self._collection,
            ids=[int(fact_id)],
            with_vectors=False,
            with_payload=False,
        )
        return len(points) > 0

    def _validate_and_cast(self, vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != self.dim:
            raise ValueError(
                f"vector shape mismatch: expected ({self.dim},), got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("vector contains NaN/Inf")
        return arr
