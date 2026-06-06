"""Prism 语义编码模块。

暴露：
    - :class:`SemanticBackend` 协议
    - :class:`DegradedSemanticBackend` 占位（sentence-transformers 缺失场景）
    - :class:`LocalBgeBackend` / :class:`CloudEmbeddingBackend` / :class:`HybridRerankBackend`
    - :func:`check_sentence_transformers_available` / :func:`warn_if_sentence_transformers_missing`
    - 正常/降级路径的三路融合权重常量与 helper
"""

from __future__ import annotations

from .backend import (
    DEFAULT_FTS_WEIGHT,
    DEFAULT_JACCARD_WEIGHT,
    DEFAULT_SEMANTIC_WEIGHT,
    DEGRADED_FTS_WEIGHT,
    DEGRADED_JACCARD_WEIGHT,
    DEGRADED_SEMANTIC_WEIGHT,
    DegradedSemanticBackend,
    SemanticBackend,
    SemanticUnavailable,
    check_sentence_transformers_available,
    default_weights,
    degraded_weights,
    warn_if_sentence_transformers_missing,
)
from .cloud_embedding import CloudEmbeddingBackend
from .hybrid_rerank import HybridRerankBackend, RerankCandidate, RerankResult
from .local_bge import BGE_SMALL_ZH_DIM, DEFAULT_BGE_MODEL, LocalBgeBackend

__all__ = [
    "BGE_SMALL_ZH_DIM",
    "DEFAULT_BGE_MODEL",
    "DEFAULT_FTS_WEIGHT",
    "DEFAULT_JACCARD_WEIGHT",
    "DEFAULT_SEMANTIC_WEIGHT",
    "DEGRADED_FTS_WEIGHT",
    "DEGRADED_JACCARD_WEIGHT",
    "DEGRADED_SEMANTIC_WEIGHT",
    "CloudEmbeddingBackend",
    "DegradedSemanticBackend",
    "HybridRerankBackend",
    "LocalBgeBackend",
    "RerankCandidate",
    "RerankResult",
    "SemanticBackend",
    "SemanticUnavailable",
    "check_sentence_transformers_available",
    "default_weights",
    "degraded_weights",
    "warn_if_sentence_transformers_missing",
]