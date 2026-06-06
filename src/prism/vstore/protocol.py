"""``VectorStore`` 协议。

向量存储与查询的统一接口，与 ``SemanticBackend``（编码器）正交解耦。

协议契约：
- ``add``：``fact_id`` 已存在视为实现错误
- ``remove``：不存在静默 noop（幂等）
- ``update``：存在则替换；不存在等同 ``add``
- ``rebuild_from_iter``：实现应分批处理
- ``persist``：文件 backend 必须落盘；DB/服务端 backend 为 noop
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

__all__ = ["VectorStore"]


@runtime_checkable
class VectorStore(Protocol):
    """所有向量索引 backend 必须满足此契约。

    属性：
        dim: 向量维度。同实例的所有 ``add/update/topk`` 输入必须等于此值。
        backend_name: 后端类型 — local_numpy / hnswlib / faiss / pgvector / qdrant。
            stats() 与日志按此字段区分实例来源。
    """

    dim: int
    backend_name: str

    def __len__(self) -> int:
        """当前活跃向量数量（不含 deleted_set 已标记删除的）。

        检索流程中用于 N=0 短路判断，避免每次调 ``stats()`` 解析 dict。
        """
        ...

    def add(self, fact_id: int, vector: np.ndarray) -> None:
        """新增向量并与 ``fact_id`` 关联。

        ``fact_id`` 已存在视为实现错误——调用方（mirror / reindex CLI）必须保证
        唯一性。各 backend 可根据自身能力选择「抛 ``ValueError``」或「静默覆盖」，
        但都不应假定后者是契约的一部分；推荐抛错以暴露调用方 bug。
        """
        ...

    def remove(self, fact_id: int) -> None:
        """按 ``fact_id`` 删除。

        幂等：``fact_id`` 不存在时静默 noop，不抛错。让批量清理路径不需要 try/except
        包裹（如幽灵清理）。
        """
        ...

    def update(self, fact_id: int, vector: np.ndarray) -> None:
        """按 ``fact_id`` 替换向量。

        幂等：``fact_id`` 存在则替换；不存在则行为等同 :meth:`add`。
        不支持原地更新的 backend 用 ``remove + add`` 满足契约。
        """
        ...

    def topk(
        self,
        query: np.ndarray,
        k: int,
        filter_fact_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """返回 ``[(fact_id, cosine_score), ...]``，按分数降序。

        Args:
            query: 查询向量，必须维度 == ``self.dim`` 且 L2 归一化。
            k: 期望返回 top-k；实际返回数 ≤ k（候选不足时短）。
            filter_fact_ids: 元数据预过滤后的候选集（SQLite 按 category/trust/status
                过滤得到）。``None`` 表示不过滤。

        实现要求：
            * 支持原生 payload filter 的 backend（qdrant / pgvector）应直接下推
            * 不支持的 backend 降级「召回 k×10 候选 + Python 侧过滤」并在
              ``stats()`` 暴露此特征（如 ``fallback_filter=True``）
        """
        ...

    def fetch(self, fact_ids: list[int]) -> dict[int, np.ndarray]:
        """按 ``fact_id`` 批量取回向量。

        矛盾检测等场景需要原始向量做 cosine 计算时调用。

        返回 dict 不保证包含所有请求的 ``fact_id``；调用方需做缺失 guard。
        """
        ...

    def rebuild_from_iter(self, pairs: Iterable[tuple[int, np.ndarray]]) -> None:
        """全量重建（首次初始化、模型升级、backend 切换）。

        实现应分批处理（推荐 ``batch_size=1000``），避免：

        * 逐条 Python 循环开销（local_numpy）
        * 外部服务过多小请求（pgvector / qdrant 的 round-trip）

        ``Iterable`` 而非 ``list``：让调用方（``prism vstore-migrate`` CLI）可以
        流式读 SQLite，不必一次性把 1M facts 装内存。
        """
        ...

    def persist(self) -> None:
        """将内存状态写入持久化存储。

        * 依赖本地文件的 backend（``local_numpy`` / ``hnswlib`` / ``faiss``）必须落盘
        * 数据库 / 服务端 backend（``pgvector`` / ``qdrant``）为 noop
        """
        ...

    def stats(self) -> dict[str, Any]:
        """运维指标：backend / ntotal / dim / memory_bytes / index_type / deleted_count 等。

        ``prism_admin(stats)`` 与 Grafana 抓取的字段来源。具体字段不做强约束（不同
        backend 暴露不同维度），但 ``backend_name`` / ``dim`` / ``ntotal`` 三项推荐
        所有实现都提供以便统一观测。
        """
        ...
