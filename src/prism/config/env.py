"""Prism 环境变量常量 + 统一读取。

所有 ``PRISM_*`` 环境变量的名称和读取逻辑集中在此，外部模块禁止直接
``os.environ.get("PRISM_...")``。
"""

from __future__ import annotations

import os

__all__ = [
    "PRISM_ENV_CONFIG",
    "PRISM_ENV_DATA_HOME",
    "PRISM_ENV_DB_PATH",
    "PRISM_ENV_LOG_LEVEL",
    "PRISM_ENV_PROFILE",
    "PRISM_ENV_USER_ID",
    "read_env_options",
]

PRISM_ENV_DATA_HOME: str = "PRISM_DATA_HOME"
PRISM_ENV_CONFIG: str = "PRISM_CONFIG"
PRISM_ENV_PROFILE: str = "PRISM_PROFILE"
PRISM_ENV_USER_ID: str = "PRISM_USER_ID"
PRISM_ENV_DB_PATH: str = "PRISM_DB_PATH"
PRISM_ENV_LOG_LEVEL: str = "PRISM_LOG_LEVEL"


def read_env_options() -> dict[str, str | None]:
    """读取所有 ``PRISM_*`` 运行时环境变量。

    返回 dict，key 与 :class:`prism.mcp.wire.RuntimeOptions` 字段对齐。
    MCP 入口和需要批量读取环境变量的场景调用此函数。
    """
    return {
        "data_home": os.environ.get(PRISM_ENV_DATA_HOME),
        "config_path": os.environ.get(PRISM_ENV_CONFIG),
        "profile": os.environ.get(PRISM_ENV_PROFILE, "default"),
        "user_id": os.environ.get(PRISM_ENV_USER_ID, "local_default"),
        "db_path_override": os.environ.get(PRISM_ENV_DB_PATH),
        "log_level": os.environ.get(PRISM_ENV_LOG_LEVEL, "INFO"),
    }
