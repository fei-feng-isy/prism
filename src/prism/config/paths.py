"""Prism 路径解析 — DB 路径模板渲染 + 数据目录 / 配置文件发现。

所有「从配置推导出文件系统路径」的逻辑集中在此，避免散落到 db / cli / plugin。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Final

from .schema import DbConfig

__all__ = [
    "DEFAULT_PROFILE",
    "DEFAULT_USER_HASH_LENGTH",
    "DEFAULT_USER_ID",
    "compute_user_hash",
    "discover_config_path",
    "get_agent_home",
    "reset_agent_home",
    "resolve_config_path",
    "resolve_data_home",
    "resolve_db_path",
    "resolve_db_path_for_user",
    "set_agent_home",
]


# ─── 常量 ───────────────────────────────────────────────────────────────────

DEFAULT_PROFILE: Final[str] = "default"
"""单 profile 默认值。"""

DEFAULT_USER_ID: Final[str] = "local_default"
"""单用户场景下用的稳定 user_id（与 wire/plugin 默认对齐）。"""

DEFAULT_USER_HASH_LENGTH: Final[int] = 16
"""``compute_user_hash`` 默认取 sha256 前 16 hex（足以避免冲突且短到适合路径）。"""


# ─── DB 路径 ─────────────────────────────────────────────────────────────────


def compute_user_hash(user_id: str, *, length: int = DEFAULT_USER_HASH_LENGTH) -> str:
    """对 ``user_id`` 取 SHA-256 截断 hex — 多 user 路径段统一脱敏。

    Args:
        user_id: 任意字符串（设计推荐使用稳定标识，如 OS 用户名 / 设备 ID）
        length: 截断长度（hex 字符数）；默认 16 = 64 bit 命名空间

    Raises:
        ValueError: ``user_id`` 为空 / ``length`` ≤ 0 或 > 64
    """
    if not user_id:
        raise ValueError("user_id 不能为空字符串")
    if length <= 0 or length > 64:
        raise ValueError(f"length 必须在 (0, 64]，实际：{length}")
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:length]


def resolve_db_path(
    cfg: DbConfig,
    *,
    profile: str,
    user_hash: str,
    data_home: str | None = None,
) -> Path:
    """把 ``path_template`` 渲染成绝对 Path。

    占位符：
        - ``{data_home}`` — 优先用 ``data_home`` 参数，缺失则用
          ``cfg.data_home_default``（默认 ``~/.prism``）
        - ``{profile}`` — 入参
        - ``{user_hash}`` — 入参（调用方负责脱敏）

    Args:
        cfg: 已校验过的 DbConfig
        profile: 例如 "default" / "work" / "personal"
        user_hash: 例如对 user_id 取 SHA-256 前 12 hex
        data_home: 数据根目录覆盖；None 时用 ``cfg.data_home_default``
    """
    home = data_home if data_home is not None else cfg.data_home_default
    raw = cfg.path_template.format(
        data_home=home,
        profile=profile,
        user_hash=user_hash,
    )
    return Path(raw).expanduser()


def resolve_db_path_for_user(
    cfg: DbConfig,
    *,
    user_id: str = DEFAULT_USER_ID,
    profile: str = DEFAULT_PROFILE,
    data_home: str | None = None,
    user_hash_length: int = DEFAULT_USER_HASH_LENGTH,
) -> Path:
    """便利封装：``compute_user_hash(user_id) + resolve_db_path``。

    Args:
        cfg: 已校验的 DbConfig
        user_id: 用户标识；缺省 ``"local_default"`` — 单 user 场景默认
        profile: 多 profile 隔离用；缺省 ``"default"``
        data_home: 数据根目录覆盖；None 时用 ``cfg.data_home_default``
        user_hash_length: 见 :func:`compute_user_hash`

    Returns:
        渲染好的 :class:`Path`（未 mkdir；调用 ``connect`` / ``bootstrap`` 时
        ``connect()`` 会自动创建父目录）
    """
    user_hash = compute_user_hash(user_id, length=user_hash_length)
    return resolve_db_path(cfg, profile=profile, user_hash=user_hash, data_home=data_home)


# ─── agent_home 注入 ────────────────────────────────────────────────────────

_agent_home: Path | None = None


def set_agent_home(path: str | os.PathLike[str]) -> None:
    """由 agent 插件在初始化时调用，设置 agent 根目录。

    Prism 核心代码通过 :func:`get_agent_home` 读取，不直接感知具体 agent。
    """
    global _agent_home
    _agent_home = Path(path).expanduser()


def get_agent_home() -> Path | None:
    """获取已设置的 agent 根目录；未设置（standalone / CLI）时返回 ``None``。"""
    return _agent_home


def reset_agent_home() -> None:
    """清除 agent home（测试 / shutdown 用）。"""
    global _agent_home
    _agent_home = None


# ─── 数据目录 / 配置文件发现 ─────────────────────────────────────────────────


def resolve_data_home(
    *,
    agent_home: str | os.PathLike[str] | None = None,
) -> Path:
    """推导 agent 插件场景下的 data_home。

    - 有 agent_home 参数 → ``agent_home/prism``
    - 无参数 → ``get_agent_home() / "prism"``

    MCP / 独立使用时不需要此函数（直接用 ``PRISM_DATA_HOME`` 或默认 ``~/.prism``）。
    """
    if agent_home:
        hh = Path(agent_home).expanduser()
    else:
        ah = get_agent_home()
        if ah is None:
            raise RuntimeError(
                "agent_home 未设置且未传参；请先 set_agent_home() 或传入 agent_home"
            )
        hh = ah
    return hh / "prism"


def discover_config_path() -> Path | None:
    """在 well-known 位置搜索已有的 ``config.yaml``。

    搜索顺序：
        1. ``$PRISM_CONFIG`` 环境变量
        2. ``<agent_home>/prism/config.yaml``（如 agent_home 已设置）
        3. ``~/.prism/config.yaml``

    返回第一个存在的路径，全不存在返回 ``None``。
    CLI 命令在用户未传 ``--config`` 时调用此函数，避免回退到 ``default_config()``
    而丢失 agent 场景下的 ``data_home_default`` 等已持久化配置。
    """
    env = os.environ.get("PRISM_CONFIG")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p

    agent_home = get_agent_home()
    if agent_home:
        agent_cfg = agent_home / "prism" / "config.yaml"
        if agent_cfg.exists():
            return agent_cfg

    standalone_cfg = Path("~/.prism/config.yaml").expanduser()
    if standalone_cfg.exists():
        return standalone_cfg

    return None


def resolve_config_path(
    *,
    override: str | os.PathLike[str] | None = None,
    data_home: str | os.PathLike[str] | None = None,
) -> Path:
    """定位 ``config.yaml``。

    优先级：override > data_home/config.yaml > ~/.prism/config.yaml。
    返回的路径不保证存在；调用方负责后续 mkdir + write。
    """
    if override is not None:
        return Path(override).expanduser()
    if data_home is not None:
        return Path(data_home).expanduser() / "config.yaml"
    return Path("~/.prism/config.yaml").expanduser()
