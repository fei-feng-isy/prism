"""MCP server 工厂。

:func:`create_server` 把一个已装配的 :class:`PrismRuntime` 包装成标准
:class:`mcp.server.Server`：

    - 注册 ``list_tools`` → 返回 :data:`prism.mcp.tools.PRISM_TOOLS`
    - 注册 ``call_tool`` → 调 :func:`call_prism_tool` dispatch → TextContent
    - SDK 的 ``call_tool`` 装饰器会按返回值类型自动包 ``CallToolResult``

transport 层（stdio / SSE / streamable HTTP）由 :mod:`prism.mcp.__main__`
选择；server 实例与 transport 解耦。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mcp.types as mcp_types
from mcp.server import Server

from prism.mcp.tools import PRISM_TOOLS, call_prism_tool, to_text_content

if TYPE_CHECKING:
    from prism.mcp.wire import PrismRuntime

log = logging.getLogger(__name__)

__all__ = ["create_server"]


# 与 pyproject.toml 同步；MCP server.version 字段
_PRISM_VERSION = "0.2.0"

_SERVER_INSTRUCTIONS = (
    "Prism — Chinese-optimized hybrid memory (FTS + entity-resolved + HRR + "
    "semantic). Three tools: prism_remember (write durable facts), "
    "prism_recall (search / probe / reason / related), prism_admin (stats)."
)


def create_server(
    runtime: PrismRuntime,
    *,
    name: str = "prism",
    version: str = _PRISM_VERSION,
    instructions: str = _SERVER_INSTRUCTIONS,
) -> Server:
    """构造已注册三个工具的 MCP :class:`Server` 实例。

    调用方负责 server.run(read_stream, write_stream, init_opts) 与 transport
    （stdio / SSE）的接续。本函数不做 IO。

    Args:
        runtime: 已就绪的 :class:`PrismRuntime`（构造期完成 BGE / DB / wire）
        name: MCP server 名（client 端显示）
        version: 版本字符串
        instructions: server 级提示（部分 client 会注入 system prompt）

    Returns:
        :class:`mcp.server.Server` 实例
    """
    server: Server = Server(name=name, version=version, instructions=instructions)

    @server.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:
        return list(PRISM_TOOLS)

    @server.call_tool()
    async def _call_tool(
        name: str, arguments: dict | None
    ) -> list[mcp_types.TextContent]:
        json_text, is_error = call_prism_tool(runtime, name, arguments)
        if is_error:
            # SDK 的 call_tool 装饰器对抛出的异常会自动设置 isError=True；
            # 这里把 dispatch 已捕获的错误也转回异常，避免静默成功
            raise RuntimeError(json_text)
        return to_text_content(json_text)

    return server
