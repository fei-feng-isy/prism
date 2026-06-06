"""Prism 配置包 — 统一导出 + 单例管理。

外部模块统一通过 ``from prism.config import X`` 使用，无需关心内部子模块划分。

**单例 API**：

    from prism.config import init_config, get_config

    init_config(path)       # 入口点调用一次（CLI main / MCP main / agent initialize）
    cfg = get_config()      # 任何模块随时获取已加载的配置

``init_config`` 只允许调用一次（重复调用需先 ``reset_config``）。保证全进程共享
同一份 ``PrismConfig``，消除多处独立 ``load_config`` 导致的路径不一致。

``set_agent_home`` / ``get_agent_home``：agent 插件在初始化时注入根目录，核心代码
通过 ``get_agent_home()`` 读取，不直接感知具体 agent 类型。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from prism.config.env import (
    PRISM_ENV_CONFIG,
    PRISM_ENV_DATA_HOME,
    PRISM_ENV_DB_PATH,
    PRISM_ENV_LOG_LEVEL,
    PRISM_ENV_PROFILE,
    PRISM_ENV_USER_ID,
    read_env_options,
)
from prism.config.loader import ENV_OVERRIDES, load_config
from prism.config.patcher import patch_config_file
from prism.config.paths import (
    DEFAULT_PROFILE,
    DEFAULT_USER_HASH_LENGTH,
    DEFAULT_USER_ID,
    compute_user_hash,
    discover_config_path,
    get_agent_home,
    reset_agent_home,
    resolve_config_path,
    resolve_data_home,
    resolve_db_path,
    resolve_db_path_for_user,
    set_agent_home,
)
from prism.config.schema import (
    AutoThresholds,
    BankConfig,
    CallTrackingConfig,
    CategoryDecay,
    CloudEmbeddingConfig,
    ConfigError,
    DbConfig,
    EntitiesConfig,
    HfEndpointStrategy,
    HrrConfig,
    LifecycleConfig,
    LoggingConfig,
    LogLevel,
    PgVectorConfig,
    PrefetchConfig,
    PrismConfig,
    QdrantConfig,
    RerankApplyTo,
    RerankConfig,
    RerankFallback,
    RetrieverConfig,
    SemanticBackendName,
    SemanticConfig,
    VectorBackendName,
    VectorStoreConfig,
    default_config,
    dump_default_config_yaml,
)

if TYPE_CHECKING:
    import os

# ─── 单例管理 ──────────────────────────────────────────────────────────────

_config: PrismConfig | None = None
_config_path: Path | None = None


def init_config(
    path: str | os.PathLike[str] | None = None,
) -> PrismConfig:
    """加载配置并缓存为进程单例。

    Args:
        path: 显式配置文件路径。``None`` 时自动调用 :func:`discover_config_path`
            搜索 well-known 位置；仍未找到则用 :func:`default_config`。

    Returns:
        已加载的 :class:`PrismConfig` 实例（与后续 :func:`get_config` 返回同一对象）。

    Raises:
        RuntimeError: 重复调用且未先 :func:`reset_config`。

    Typical call sites: CLI ``main()`` / MCP ``__main__`` / agent plugin ``initialize()``。
    """
    global _config, _config_path
    if _config is not None:
        raise RuntimeError(
            "init_config 重复调用；如需重新加载请先 reset_config()。"
            f"已加载配置来源：{_config_path}"
        )
    resolved = Path(path) if path else discover_config_path()
    if resolved is not None and resolved.exists():
        _config = load_config(resolved)
        _config_path = resolved
    else:
        _config = default_config()
        _config_path = None
    return _config


def get_config() -> PrismConfig:
    """获取已加载的配置单例。

    必须在 :func:`init_config` 之后调用。
    """
    if _config is None:
        raise RuntimeError(
            "get_config 调用前未 init_config；"
            "请在入口点（CLI main / MCP main / Hermes initialize）先调用 init_config()"
        )
    return _config


def get_config_path() -> Path | None:
    """返回当前配置文件路径，未初始化或用默认配置时返回 ``None``。"""
    return _config_path


def reset_config() -> None:
    """清除配置单例（仅测试用）。"""
    global _config, _config_path
    _config = None
    _config_path = None


__all__ = [
    "AutoThresholds",
    "BankConfig",
    "CallTrackingConfig",
    "CategoryDecay",
    "CloudEmbeddingConfig",
    "ConfigError",
    "DbConfig",
    "DEFAULT_PROFILE",
    "DEFAULT_USER_HASH_LENGTH",
    "DEFAULT_USER_ID",
    "EntitiesConfig",
    "ENV_OVERRIDES",
    "HfEndpointStrategy",
    "HrrConfig",
    "LifecycleConfig",
    "LoggingConfig",
    "LogLevel",
    "PgVectorConfig",
    "PRISM_ENV_CONFIG",
    "PRISM_ENV_DATA_HOME",
    "PRISM_ENV_DB_PATH",
    "PRISM_ENV_LOG_LEVEL",
    "PRISM_ENV_PROFILE",
    "PRISM_ENV_USER_ID",
    "PrefetchConfig",
    "PrismConfig",
    "QdrantConfig",
    "RerankApplyTo",
    "RerankConfig",
    "RerankFallback",
    "RetrieverConfig",
    "SemanticBackendName",
    "SemanticConfig",
    "VectorBackendName",
    "VectorStoreConfig",
    "compute_user_hash",
    "default_config",
    "discover_config_path",
    "dump_default_config_yaml",
    "get_agent_home",
    "get_config",
    "get_config_path",
    "init_config",
    "load_config",
    "patch_config_file",
    "read_env_options",
    "reset_agent_home",
    "reset_config",
    "resolve_config_path",
    "resolve_data_home",
    "resolve_db_path",
    "resolve_db_path_for_user",
    "set_agent_home",
]
