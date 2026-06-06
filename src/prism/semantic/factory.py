"""Semantic backend 工厂。

返回 :class:`SemanticAssembly`（write + query 两个 backend 引用）：

* **write**：mirror 用，永远是 ``LocalBgeBackend`` 或降级
* **query**：按 ``cfg.semantic.backend`` 选择 local_bge / cloud_embedding / hybrid_rerank
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from prism.config import PrismConfig
from prism.llm import LLMClient

from .backend import (
    DegradedSemanticBackend,
    SemanticBackend,
    check_sentence_transformers_available,
)
from .cloud_embedding import CloudEmbeddingBackend
from .hybrid_rerank import HybridRerankBackend
from .local_bge import BGE_SMALL_ZH_DIM, LocalBgeBackend

log = logging.getLogger(__name__)

__all__ = ["SemanticAssembly", "create_semantic_assembly"]


@dataclass(frozen=True)
class SemanticAssembly:
    """工厂输出：write + query 两个 backend 引用。

    Attributes:
        write: mirror 用（写路径），永远是 LocalBge 或 Degraded
        query: pipeline / prefetch / recall 用，按 cfg 决定
        dim: 嵌入空间维度（write 与 query 必须一致；factory 已校验）
    """

    write: SemanticBackend
    query: SemanticBackend
    dim: int


def create_semantic_assembly(cfg: PrismConfig) -> SemanticAssembly:
    """按 ``cfg.semantic.backend`` 装配 write + query backend。

    决策树：

    1. 构造 write backend
       - ``sentence-transformers`` 可用 → ``LocalBgeBackend(model_name=cfg.semantic.local_model)``
       - 缺包 → ``DegradedSemanticBackend`` + log WARNING；dim 兜底 BGE_SMALL_ZH_DIM=512
       - LocalBge 构造异常 → 同上降级（不抛）

    2. 按 ``cfg.semantic.backend`` 构造 query backend
       - ``"local_bge"``：query = write
       - ``"cloud_embedding"``：构造 CloudEmbeddingBackend；is_available()==False
         时 fallback 到 write 并 log WARNING
       - ``"hybrid_rerank"``：用 write 作 wrapped 构造 HybridRerankBackend
         （encode 仍走 LocalBge —— 无写延迟变化；仅 rerank 时调 LLM）

    Returns:
        :class:`SemanticAssembly`；调用方按 write / query 字段注入对应组件
    """
    # 1) write backend（永远 local）
    write: SemanticBackend
    if check_sentence_transformers_available():
        try:
            write = LocalBgeBackend(
                model_name=cfg.semantic.local_model,
                hf_endpoint_strategy=cfg.semantic.hf_endpoint_strategy,
                hf_mirror_url=cfg.semantic.hf_mirror_url,
            )
            dim = write.dim
        except Exception as e:
            log.warning("LocalBgeBackend 构造失败，写路径降级：%s", e)
            write = DegradedSemanticBackend()
            dim = BGE_SMALL_ZH_DIM
    else:
        log.warning(
            "sentence-transformers 未安装；Prism 走降级路径 "
            "（fts + jaccard 0.65/0.35），不影响数据正确性"
        )
        write = DegradedSemanticBackend()
        dim = BGE_SMALL_ZH_DIM

    # 2) query backend
    backend_name = cfg.semantic.backend
    query: SemanticBackend

    if backend_name == "local_bge":
        query = write

    elif backend_name == "cloud_embedding":
        try:
            cloud = CloudEmbeddingBackend(
                provider=cfg.semantic.cloud.provider,
                model=cfg.semantic.cloud.model,
                api_key_env=cfg.semantic.cloud.api_key_env,
                timeout_ms=cfg.semantic.cloud.timeout_ms,
            )
        except ValueError as e:
            log.warning(
                "CloudEmbeddingBackend 构造失败（provider=%s model=%s）："
                "%s — query 路径 fallback 到 write",
                cfg.semantic.cloud.provider, cfg.semantic.cloud.model, e,
            )
            query = write
        else:
            if cloud.is_available():
                # 维度协调由 vstore 工厂负责（cloud dim != write dim 时强制 local_numpy）
                log.info(
                    "Semantic query backend = CloudEmbedding(%s:%s, dim=%d)",
                    cloud.provider, cloud.model, cloud.dim,
                )
                query = cloud
            else:
                log.warning(
                    "CloudEmbeddingBackend 不可用（env %s 未设置或 httpx 缺）；"
                    "query 路径 fallback 到 write (%s)",
                    cfg.semantic.cloud.api_key_env, write.backend_name,
                )
                query = write

    elif backend_name == "hybrid_rerank":
        # hybrid 的 encode 透传给 wrapped(=write) → 写路径无网络
        # rerank 只在显式 recall 时被 pipeline 调用
        try:
            llm_client = LLMClient(
                model=cfg.semantic.rerank.llm_model,
                endpoint=cfg.semantic.rerank.llm_endpoint
                + "/chat/completions",
                api_key_env=cfg.semantic.rerank.api_key_env,
                timeout_seconds=cfg.semantic.rerank.timeout_ms / 1000.0,
            )
            hybrid = HybridRerankBackend(
                write,
                llm_model=cfg.semantic.rerank.llm_model,
                endpoint=cfg.semantic.rerank.llm_endpoint
                + "/chat/completions",
                api_key_env=cfg.semantic.rerank.api_key_env,
                timeout_ms=cfg.semantic.rerank.timeout_ms,
                candidate_n=cfg.semantic.rerank.candidate_n,
                final_k=cfg.semantic.rerank.final_k,
                fallback_on_error=cfg.semantic.rerank.fallback_on_error,
                llm_client=llm_client,
            )
        except ValueError as e:
            log.warning(
                "HybridRerankBackend 构造失败（%s）— query 路径 fallback 到 write",
                e,
            )
            query = write
        else:
            log.info(
                "Semantic query backend = HybridRerank(wrapping %s, llm=%s)",
                write.backend_name, cfg.semantic.rerank.llm_model,
            )
            query = hybrid

    else:
        # _validate_literals 已在 load_config 时校验；到这里只可能因测试直接
        # 构造非法 PrismConfig，防御性 fallback
        log.warning(
            "未知 semantic.backend=%r，fallback 到 local_bge 行为", backend_name
        )
        query = write

    return SemanticAssembly(write=write, query=query, dim=dim)
