"""Prism 配置 schema — dataclass 定义 + 类型别名。

所有配置结构的单一定义源。其他子模块（loader / patcher / paths）
从这里导入 dataclass / Literal 类型。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import yaml

# ─── 异常 ────────────────────────────────────────────────────────────────────


class ConfigError(ValueError):
    """配置加载或校验失败。"""


# ─── Literal 类型别名 ───────────────────────────────────────────────────────

SemanticBackendName = Literal["local_bge", "cloud_embedding", "hybrid_rerank"]
VectorBackendName = Literal["auto", "local_numpy", "hnswlib", "faiss", "pgvector", "qdrant"]
RerankApplyTo = Literal["recall_tool", "prefetch", "both"]
RerankFallback = Literal["local", "error"]
# BGE 模型首次下载的 HF endpoint 策略
#   respect_env  — 沿用用户 env / sentence_transformers 默认（向后兼容）
#   mirror_first — 先走镜像下载，失败回退官方（推荐：国内代理环境首装体验）
#   mirror_only  — 仅走镜像，不回退（用户明确强镜像场景）
HfEndpointStrategy = Literal["respect_env", "mirror_first", "mirror_only"]

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


# ─── 子配置 dataclasses ───────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CloudEmbeddingConfig:
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_ms: int = 3000


@dataclass(frozen=True, slots=True)
class RerankConfig:
    llm_model: str = "deepseek-v4-flash"
    llm_endpoint: str = "https://api.deepseek.com/v1"
    api_key_env: str = "DEEPSEEK_API_KEY"
    candidate_n: int = 50
    final_k: int = 5
    apply_to: RerankApplyTo = "recall_tool"
    timeout_ms: int = 2000
    fallback_on_error: RerankFallback = "local"


@dataclass(frozen=True, slots=True)
class SemanticConfig:
    backend: SemanticBackendName = "local_bge"
    local_model: str = "BAAI/bge-small-zh-v1.5"
    # 仅在首次下载（HF cache miss）生效；cache 命中直接走本地路径
    hf_endpoint_strategy: HfEndpointStrategy = "respect_env"
    hf_mirror_url: str = "https://hf-mirror.com"
    cloud: CloudEmbeddingConfig = field(default_factory=CloudEmbeddingConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)


@dataclass(frozen=True, slots=True)
class AutoThresholds:
    hnswlib: int = 2000
    faiss: int = 100_000


@dataclass(frozen=True, slots=True)
class PgVectorConfig:
    dsn_env: str = "PRISM_PGVECTOR_DSN"
    table_name: str = "prism_vectors"


@dataclass(frozen=True, slots=True)
class QdrantConfig:
    url_env: str = "PRISM_QDRANT_URL"
    api_key_env: str = "PRISM_QDRANT_API_KEY"
    collection_name: str = "prism"


@dataclass(frozen=True, slots=True)
class VectorStoreConfig:
    backend: VectorBackendName = "auto"
    auto_thresholds: AutoThresholds = field(default_factory=AutoThresholds)
    pgvector: PgVectorConfig = field(default_factory=PgVectorConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)


@dataclass(frozen=True, slots=True)
class PrefetchConfig:
    top_k: int = 5
    p95_target_ms: int = 100
    min_trust: float = 0.3


@dataclass(frozen=True, slots=True)
class RetrieverConfig:
    weight_semantic: float = 0.55
    weight_fts: float = 0.30
    weight_jaccard: float = 0.15
    prefetch: PrefetchConfig = field(default_factory=PrefetchConfig)


@dataclass(frozen=True, slots=True)
class BankConfig:
    remove_debounce_ms: int = 50
    calibration_threshold_min: int = 10
    calibration_threshold_pct: float = 0.15
    snr_warn_factor: float = 1.2


@dataclass(frozen=True, slots=True)
class HrrConfig:
    dim: int = 1024
    bank: BankConfig = field(default_factory=BankConfig)


@dataclass(frozen=True, slots=True)
class EntitiesConfig:
    auto_enrich: bool = True
    stage1_min_entities: int = 2


@dataclass(frozen=True, slots=True)
class CategoryDecay:
    decay_per_day: float = 0.995
    min_trust_floor: float = 0.0


@dataclass(frozen=True, slots=True)
class LifecycleConfig:
    archive_after_days: int = 90
    decay_by_category: dict[str, CategoryDecay] = field(
        default_factory=lambda: {
            "user_pref": CategoryDecay(decay_per_day=1.0, min_trust_floor=0.4),
            "user_env": CategoryDecay(decay_per_day=1.0, min_trust_floor=0.4),
            "project": CategoryDecay(decay_per_day=0.999, min_trust_floor=0.2),
            "general": CategoryDecay(decay_per_day=0.995, min_trust_floor=0.0),
        }
    )


@dataclass(frozen=True, slots=True)
class DbConfig:
    # 路径模板支持 {data_home} / {profile} / {user_hash} 三个占位
    path_template: str = "{data_home}/{profile}/{user_hash}.db"
    data_home_default: str = "~/.prism"


@dataclass(frozen=True, slots=True)
class CallTrackingConfig:
    enabled: bool = True
    file_logging: bool = True
    max_bytes: int = 5_000_000
    backup_count: int = 3
    buffer_size: int = 10_000


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    level: LogLevel = "INFO"
    call_tracking: CallTrackingConfig = field(default_factory=CallTrackingConfig)


@dataclass(frozen=True, slots=True)
class PrismConfig:
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    hrr: HrrConfig = field(default_factory=HrrConfig)
    entities: EntitiesConfig = field(default_factory=EntitiesConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    db: DbConfig = field(default_factory=DbConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def default_config() -> PrismConfig:
    return PrismConfig()


# ─── 默认 YAML 模板 ─────────────────────────────────────────────────────────

_DEFAULT_CONFIG_HEADER = """\
# Prism Memory plugin config — auto-generated on first activation.
# Edit any field below; restart agent to apply changes.
#
# Documentation:    https://github.com/fei-feng-isy/prism/tree/master/doc
# Schema reference: src/prism/config/schema.py (PrismConfig dataclass tree)
#
# Top-level sections:
#   semantic       — embedding backend (local_bge / cloud_embedding / hybrid_rerank)
#   vector_store   — ANN backend (auto / local_numpy / hnswlib / faiss / pgvector / qdrant)
#   retriever      — three-way fusion weights (semantic / fts / jaccard)
#   hrr            — holographic reduced representation (dim, calibration)
#   entities       — async LLM enrichment when stage1 < 2 entities
#   lifecycle      — per-category trust decay + archive
#   db             — DB path template ({data_home}/{profile}/{user_hash}.db)
#   logging        — log level + call_tracking (API 调用追踪开关)
#
# Env var overrides take precedence over this YAML — see ENV_OVERRIDES in
# src/prism/config/loader.py for the full list (PRISM_SEMANTIC_BACKEND, PRISM_HRR_DIM, ...).
#
# To regenerate this file with current defaults: delete it and re-run agent.
"""


def dump_default_config_yaml() -> str:
    """``default_config()`` 序列化为带头部注释的 YAML 字符串。

    用于初次激活时写一份包含全部默认值的 YAML 模板，
    让用户基于实际默认值修改而不是面对空文件 + 翻文档。
    """
    body = yaml.safe_dump(
        asdict(default_config()),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    return _DEFAULT_CONFIG_HEADER + "\n" + body
