"""``python -m prism.mcp`` — stdio MCP server 入口。

读环境变量 / CLI 参数构造 :class:`prism.mcp.wire.RuntimeOptions`，装配
:class:`PrismRuntime`，挂到 stdio transport：

环境变量：
    - ``PRISM_CONFIG``         — YAML 配置路径（可选；未设则自动探测 data_home/config.yaml）
    - ``PRISM_PROFILE``        — DB 路径模板的 ``{profile}``，默认 "default"
    - ``PRISM_USER_ID``        — 用户隔离 ID（sha256 → user_hash），默认 "local_default"
    - ``PRISM_DATA_HOME``      — 数据根目录（默认 ``~/.prism``）
    - ``PRISM_DB_PATH``        — 直传 DB 路径（绕过 path_template；测试用）

LLM chat_fn 注入暂不暴露 env（避免 stdio 入口耦合 OpenAI/DeepSeek SDK）；
生产场景由 wrapping process 调 :func:`prism.mcp.wire.build_runtime` 显式
传入 ``llm_chat_fn``（如自定义脚本 ``run_prism_mcp.py``）。

启动后阻塞读 stdio；按下 Ctrl-C 或 stdin EOF 退出，shutdown 自动 persist
vstore + close DB。
"""

from __future__ import annotations

import asyncio
import logging
import sys

import mcp.server.stdio
from mcp.server.lowlevel import NotificationOptions
from mcp.server.models import InitializationOptions

from prism.mcp.server import create_server
from prism.mcp.wire import RuntimeOptions, build_runtime


def _setup_logging() -> None:
    """所有日志写 stderr（stdio transport 的 stdout 是 MCP 协议帧）。

    用 ``force=True`` 覆盖已有 handler，否则在已配置过 root logger 的环境
    （如 pytest）下静默 no-op。
    """
    from prism.config import read_env_options

    level_name = read_env_options()["log_level"] or "INFO"
    level = getattr(logging, level_name.upper(), logging.INFO)
    if not isinstance(level, int):
        level = logging.INFO
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        force=True,
    )


def _opts_from_env() -> RuntimeOptions:
    from prism.config import read_env_options, resolve_config_path

    env = read_env_options()
    config_path = env["config_path"]
    if config_path is None:
        candidate = resolve_config_path(data_home=env["data_home"])
        if candidate.exists():
            config_path = str(candidate)

    return RuntimeOptions(
        config_path=config_path,
        profile=env["profile"] or "default",
        user_id=env["user_id"] or "local_default",
        data_home=env["data_home"],
        db_path_override=env["db_path_override"],
        call_source="mcp",
    )


async def _run() -> None:
    runtime = build_runtime(_opts_from_env())
    server = create_server(runtime)

    init_opts = InitializationOptions(
        server_name=server.name,
        server_version=server.version or "0.0.0",
        capabilities=server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        ),
        instructions=server.instructions,
    )

    try:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, init_opts)
    finally:
        runtime.shutdown()


def main() -> None:
    _setup_logging()
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        # 优雅退出（stderr 不需要堆栈）
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
