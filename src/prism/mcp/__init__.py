"""Prism MCP server 适配。

把 Prism 暴露为标准 MCP server（stdio transport），注册三个工具：
``prism_remember`` / ``prism_recall`` / ``prism_admin``，供 Claude Desktop /
Cline / Continue 等 MCP client 直接挂载。

子模块：
    - :mod:`prism.mcp.wire`     — :class:`PrismRuntime` + :func:`build_runtime`
    - :mod:`prism.mcp.tools`    — Tool schema 与 dispatch helper
    - :mod:`prism.mcp.server`   — :func:`create_server`
    - :mod:`prism.mcp.__main__` — ``python -m prism.mcp`` stdio 入口
"""

from prism.mcp.wire import PrismRuntime, build_runtime

__all__ = ["PrismRuntime", "build_runtime"]