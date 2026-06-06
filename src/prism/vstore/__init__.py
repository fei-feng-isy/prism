"""Prism 向量存储模块。

暴露 :class:`VectorStore` 协议、:class:`LocalNumpyVectorStore` 默认实现、
:func:`create_vstore` 工厂。外置实现：hnswlib / faiss / pgvector / qdrant。
"""

from __future__ import annotations

from .factory import (
    AUTO_BACKEND_RANK,
    count_active_facts,
    count_distinct_embedding_models,
    create_vstore,
    decide_auto_backend,
    read_effective_backend,
    write_effective_backend,
)
from .local_numpy import LocalNumpyVectorStore
from .protocol import VectorStore

__all__ = [
    "AUTO_BACKEND_RANK",
    "LocalNumpyVectorStore",
    "VectorStore",
    "count_active_facts",
    "count_distinct_embedding_models",
    "create_vstore",
    "decide_auto_backend",
    "read_effective_backend",
    "write_effective_backend",
]
