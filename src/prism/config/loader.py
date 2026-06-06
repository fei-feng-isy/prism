"""Prism 配置加载 — YAML + 环境变量覆盖 + 校验。

三层优先级（右覆盖左）：
    内置默认值  →  YAML 文件  →  环境变量
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml

from .schema import (
    CategoryDecay,
    ConfigError,
    HfEndpointStrategy,
    LogLevel,
    PrismConfig,
    RerankApplyTo,
    RerankFallback,
    SemanticBackendName,
    VectorBackendName,
    default_config,
)

__all__ = [
    "ENV_OVERRIDES",
    "load_config",
]


# ─── 环境变量映射 ────────────────────────────────────────────────────────────
#
# 每条 (env_var, 路径, 转换器)；路径用 "." 分隔的字段名定位到嵌套 dataclass。
# 仅暴露最常被运维覆盖的字段；其他需要改的请用 YAML。

_EnvCoerce = type | None


def _as_bool(s: str) -> bool:
    v = s.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"无法将 {s!r} 解析为布尔值")


ENV_OVERRIDES: list[tuple[str, str, _EnvCoerce]] = [
    ("PRISM_SEMANTIC_BACKEND", "semantic.backend", str),
    ("PRISM_SEMANTIC_LOCAL_MODEL", "semantic.local_model", str),
    ("PRISM_VECTOR_STORE_BACKEND", "vector_store.backend", str),
    ("PRISM_HRR_DIM", "hrr.dim", int),
    ("PRISM_ENTITIES_AUTO_ENRICH", "entities.auto_enrich", _as_bool),
    ("PRISM_RETRIEVER_WEIGHT_SEMANTIC", "retriever.weight_semantic", float),
    ("PRISM_RETRIEVER_WEIGHT_FTS", "retriever.weight_fts", float),
    ("PRISM_RETRIEVER_WEIGHT_JACCARD", "retriever.weight_jaccard", float),
    ("PRISM_LOG_LEVEL", "logging.level", str),
    ("PRISM_DB_PATH_TEMPLATE", "db.path_template", str),
    ("PRISM_CALL_TRACKING_ENABLED", "logging.call_tracking.enabled", _as_bool),
    ("PRISM_CALL_TRACKING_FILE", "logging.call_tracking.file_logging", _as_bool),
]


# ─── 加载入口 ────────────────────────────────────────────────────────────────


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> PrismConfig:
    """加载配置：默认 → YAML → env。

    Args:
        path: YAML 文件路径；None 时只用默认 + env
        env: 用于测试注入；None 时用 os.environ

    Raises:
        ConfigError: 文件不存在、YAML 解析失败、字段类型不匹配、Literal 值非法
    """
    cfg = default_config()

    if path is not None:
        cfg = _apply_yaml(cfg, Path(path))

    cfg = _apply_env(cfg, env if env is not None else os.environ)
    _validate(cfg)
    return cfg


# ─── YAML 加载 ───────────────────────────────────────────────────────────────


def _apply_yaml(cfg: PrismConfig, path: Path) -> PrismConfig:
    if not path.exists():
        raise ConfigError(f"配置文件不存在：{path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML 解析失败 ({path}): {e}") from e
    if raw is None:
        return cfg
    if not isinstance(raw, dict):
        raise ConfigError(f"YAML 根节点必须是 mapping，实际是 {type(raw).__name__}")

    # 兼容两种顶层：直接字段或包在 `prism:` 下
    if "prism" in raw and len(raw) == 1:
        section = raw["prism"]
        if not isinstance(section, dict):
            raise ConfigError("`prism:` 必须是 mapping")
        raw = section

    return _merge_dataclass(cfg, raw, path="prism")


def _merge_dataclass(instance: Any, overrides: Mapping[str, Any], *, path: str) -> Any:
    """把 dict 覆盖到 dataclass 实例上，递归处理嵌套 dataclass。"""
    assert is_dataclass(instance), f"{path}: 不是 dataclass"
    type_hints = {f.name: f.type for f in fields(instance)}
    type_objs = {f.name: type(getattr(instance, f.name)) for f in fields(instance)}

    updates: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in type_hints:
            raise ConfigError(f"{path}: 未知配置项 {key!r}")
        current = getattr(instance, key)
        sub_path = f"{path}.{key}"

        if is_dataclass(current) and isinstance(value, dict):
            updates[key] = _merge_dataclass(current, value, path=sub_path)
        elif key == "decay_by_category" and isinstance(value, dict):
            # 特殊：dict[str, CategoryDecay]
            merged = dict(current)
            for cat, body in value.items():
                if not isinstance(body, dict):
                    raise ConfigError(f"{sub_path}.{cat}: 必须是 mapping")
                base = merged.get(cat, CategoryDecay())
                merged[cat] = _merge_dataclass(base, body, path=f"{sub_path}.{cat}")
            updates[key] = merged
        else:
            updates[key] = _coerce(value, type_objs[key], path=sub_path)

    return replace(instance, **updates)


def _coerce(value: Any, expected: type, *, path: str) -> Any:
    """轻量类型校验 / 转换。"""
    # bool 必须严格判断，因为 isinstance(True, int) == True
    if expected is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: 期望 bool，得到 {type(value).__name__}")
        return value
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: 期望 int，得到 {type(value).__name__}")
        return value
    if expected is float:
        if isinstance(value, bool):
            raise ConfigError(f"{path}: 期望 float，得到 bool")
        if isinstance(value, int | float):
            return float(value)
        raise ConfigError(f"{path}: 期望 float，得到 {type(value).__name__}")
    if expected is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: 期望 str，得到 {type(value).__name__}")
        return value
    if expected is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: 期望 dict，得到 {type(value).__name__}")
        return value
    # 兜底
    return value


# ─── env 覆盖 ────────────────────────────────────────────────────────────────


def _apply_env(cfg: PrismConfig, env: Mapping[str, str]) -> PrismConfig:
    for env_key, dotted, coerce in ENV_OVERRIDES:
        if env_key not in env:
            continue
        raw = env[env_key]
        try:
            value = coerce(raw) if coerce is not None else raw
        except (ValueError, TypeError) as e:
            raise ConfigError(f"环境变量 {env_key}={raw!r} 解析失败: {e}") from e
        cfg = _set_path(cfg, dotted, value)
    return cfg


def _set_path(cfg: Any, dotted: str, value: Any) -> Any:
    head, _, tail = dotted.partition(".")
    if not tail:
        return replace(cfg, **{head: value})
    sub = getattr(cfg, head)
    return replace(cfg, **{head: _set_path(sub, tail, value)})


# ─── 校验 ────────────────────────────────────────────────────────────────────


def _validate(cfg: PrismConfig) -> None:
    _validate_literals(cfg)

    weights = (
        cfg.retriever.weight_semantic,
        cfg.retriever.weight_fts,
        cfg.retriever.weight_jaccard,
    )
    for name, w in zip(("semantic", "fts", "jaccard"), weights, strict=True):
        if w < 0 or w > 1:
            raise ConfigError(f"retriever.weight_{name} 必须在 [0,1]，得到 {w}")
    total = sum(weights)
    if not (0.99 <= total <= 1.01):
        raise ConfigError(f"retriever 权重和必须 ≈ 1.0，得到 {total:.4f}")

    if cfg.hrr.dim <= 0 or cfg.hrr.dim % 2:
        raise ConfigError(f"hrr.dim 必须是正偶数，得到 {cfg.hrr.dim}")
    if cfg.hrr.bank.remove_debounce_ms < 0:
        raise ConfigError("hrr.bank.remove_debounce_ms 必须 ≥ 0")
    if not (0 < cfg.hrr.bank.calibration_threshold_pct <= 1):
        raise ConfigError("hrr.bank.calibration_threshold_pct 必须在 (0,1]")

    if cfg.retriever.prefetch.p95_target_ms <= 0:
        raise ConfigError("retriever.prefetch.p95_target_ms 必须 > 0")
    if not (0 <= cfg.retriever.prefetch.min_trust <= 1):
        raise ConfigError("retriever.prefetch.min_trust 必须在 [0,1]")

    if cfg.lifecycle.archive_after_days <= 0:
        raise ConfigError("lifecycle.archive_after_days 必须 > 0")
    for cat, decay in cfg.lifecycle.decay_by_category.items():
        if not (0 < decay.decay_per_day <= 1.0):
            raise ConfigError(
                f"lifecycle.decay_by_category.{cat}.decay_per_day 必须在 (0,1]"
            )
        if not (0 <= decay.min_trust_floor <= 1):
            raise ConfigError(
                f"lifecycle.decay_by_category.{cat}.min_trust_floor 必须在 [0,1]"
            )

    template = cfg.db.path_template
    for token in ("{data_home}", "{profile}", "{user_hash}"):
        if token not in template:
            raise ConfigError(f"db.path_template 必须包含 {token}")


def _validate_literals(cfg: PrismConfig) -> None:
    """校验所有 Literal 字段取值合法。"""
    _check_literal("semantic.backend", cfg.semantic.backend, SemanticBackendName)
    _check_literal(
        "semantic.hf_endpoint_strategy",
        cfg.semantic.hf_endpoint_strategy,
        HfEndpointStrategy,
    )
    _check_literal("vector_store.backend", cfg.vector_store.backend, VectorBackendName)
    _check_literal("semantic.rerank.apply_to", cfg.semantic.rerank.apply_to, RerankApplyTo)
    _check_literal(
        "semantic.rerank.fallback_on_error",
        cfg.semantic.rerank.fallback_on_error,
        RerankFallback,
    )
    _check_literal("logging.level", cfg.logging.level, LogLevel)


def _check_literal(path: str, value: Any, alias: Any) -> None:
    if get_origin(alias) is None:
        args = get_args(alias)
    else:
        args = get_args(alias)
    if value not in args:
        raise ConfigError(f"{path}={value!r} 非法；允许 {list(args)}")
