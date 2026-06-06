"""MCP tool 定义与 dispatch。

定义三个 :class:`mcp.types.Tool`（prism_remember / prism_recall / prism_admin）
的 inputSchema，并提供 ``call_prism_tool`` 统一 dispatch 入口。返回
``(json_text, is_error)``，MCP server 层包装为 TextContent。
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import mcp.types as mcp_types

if TYPE_CHECKING:
    from prism.mcp.wire import PrismRuntime

log = logging.getLogger(__name__)

__all__ = [
    "PRISM_ADMIN_TOOL",
    "PRISM_RECALL_TOOL",
    "PRISM_REMEMBER_TOOL",
    "PRISM_TOOLS",
    "call_prism_tool",
    "to_text_content",
]


# ─── inputSchema（与 plugins/hermes 同源）─────────────────────────────────────

_REMEMBER_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["add", "update", "remove", "helpful", "unhelpful"],
            "description": "Action to perform: add / update / remove / helpful / unhelpful.",
        },
        "content": {
            "type": "string",
            "description": "Fact text (required for 'add'). Be concise and self-contained.",
        },
        "category": {
            "type": "string",
            "enum": ["user_pref", "user_env", "project", "tool", "general"],
            "description": "Fact category (defaults to 'general').",
        },
    },
    "required": ["action"],
}

_RECALL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["search", "probe", "reason", "related", "contradict"],
            "description": "search=free-text; probe=single entity; reason=AND-join entities; related=co-occurrence; contradict=check.",
        },
        "query": {"type": "string", "description": "Free-text query (search)."},
        "entity": {"type": "string", "description": "Single entity (probe / related)."},
        "entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Multiple entities (reason, AND-joined).",
        },
        "category": {
            "type": "string",
            "enum": ["user_pref", "user_env", "project", "tool", "general"],
        },
        "limit": {"type": "integer", "description": "Max results."},
        "min_trust": {"type": "number", "description": "Min trust filter (search, as_markdown=false)."},
        "as_markdown": {
            "type": "boolean",
            "description": "search only: True (default) returns markdown; False returns structured list.",
        },
    },
    "required": ["action"],
}

_ADMIN_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["stats", "list", "archive", "restore", "enrichment_diagnose", "enrichment_fix"],
            "description": "stats=health JSON; list/archive/restore; enrichment_diagnose=deep enrichment diagnostics; enrichment_fix=fix missing embeddings + clear queue + rebuild vstore.",
        },
        "category": {
            "type": "string",
            "enum": ["user_pref", "user_env", "project", "tool", "general"],
            "description": "Optional category filter for 'stats'.",
        },
        "dry_run": {
            "type": "boolean",
            "description": "enrichment_fix only: True=report without modifying; default False.",
        },
    },
    "required": ["action"],
}


PRISM_REMEMBER_TOOL: mcp_types.Tool = mcp_types.Tool(
    name="prism_remember",
    description=(
        "Write or modify facts in Prism memory (Chinese-optimized hybrid memory).\n\n"
        "ACTIONS:\n"
        "• add — Store a new fact the user would expect you to remember later.\n"
        "  Examples: 'Alice 喜欢 PostgreSQL 14', 'project deadline is 2026-07-01'.\n"
        "• update / remove / helpful / unhelpful — Modify existing facts or provide feedback.\n\n"
        "WHEN TO USE: Whenever the user shares durable preferences, facts about people, "
        "projects, decisions, or environment that would be useful in future turns."
    ),
    inputSchema=_REMEMBER_INPUT_SCHEMA,
)

PRISM_RECALL_TOOL: mcp_types.Tool = mcp_types.Tool(
    name="prism_recall",
    description=(
        "Query Prism memory (Chinese-optimized hybrid search: semantic + FTS + entity).\n\n"
        "ACTIONS:\n"
        "• search — Free-text query, returns markdown snippet of top-k facts.\n"
        "• probe — All facts mentioning an entity.\n"
        "• reason — Facts mentioning ALL of the given entities (AND-join).\n"
        "• related — Entities co-occurring with the given entity.\n"
        "• contradict — Find facts that contradict each other.\n\n"
        "WHEN TO USE: Before answering questions about the user / project / past decisions."
    ),
    inputSchema=_RECALL_INPUT_SCHEMA,
)

PRISM_ADMIN_TOOL: mcp_types.Tool = mcp_types.Tool(
    name="prism_admin",
    description=(
        "Inspect / manage Prism memory store (operator-facing).\n\n"
        "ACTIONS:\n"
        "• stats — Return health panel JSON: db counts / vstore capacity / "
        "semantic backend status / prefetch warm-up / retriever weights.\n"
        "• list / archive / restore — Browse and manage stored facts.\n"
        "• enrichment_diagnose — Deep enrichment diagnostics: queue items, "
        "status distribution, missing vectors.\n"
        "• enrichment_fix — Fix missing embeddings, clear pending queue, "
        "rebuild vstore. Pass dry_run=true to preview."
    ),
    inputSchema=_ADMIN_INPUT_SCHEMA,
)


PRISM_TOOLS: list[mcp_types.Tool] = [
    PRISM_REMEMBER_TOOL,
    PRISM_RECALL_TOOL,
    PRISM_ADMIN_TOOL,
]


# ─── dispatch ────────────────────────────────────────────────────────────────


def _to_json(payload: Any) -> str:
    """``json.dumps`` 兜底 ndarray / set / 异常等非原生类型。"""
    return json.dumps(payload, ensure_ascii=False, default=str)


def _error_payload(msg: str) -> dict[str, str]:
    return {"error": msg}


def call_prism_tool(
    runtime: PrismRuntime | None,
    tool_name: str,
    arguments: dict[str, Any] | None,
) -> tuple[str, bool]:
    """统一 dispatch 入口。

    Returns:
        (json_text, is_error) —— json_text 永远是 ``json.dumps`` 结果（保证
        MCP TextContent 一致），is_error 为 True 时 MCP server 层应设置
        TextContent.annotations 或返 isError=True
    """
    args = arguments or {}

    if runtime is None:
        return _to_json(_error_payload("Prism runtime not initialized")), True

    if tool_name == "prism_remember":
        return _handle_remember(runtime, args)
    if tool_name == "prism_recall":
        return _handle_recall(runtime, args)
    if tool_name == "prism_admin":
        return _handle_admin(runtime, args)
    return _to_json(_error_payload(f"Unknown tool: {tool_name}")), True


def _split_action(args: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    action = args.get("action")
    rest = {k: v for k, v in args.items() if k != "action"}
    return action, rest


def _handle_remember(
    runtime: PrismRuntime, args: dict[str, Any]
) -> tuple[str, bool]:
    try:
        action, kwargs = _split_action(args)
        if not action:
            return _to_json(_error_payload("Missing required argument: action")), True
        return _to_json(runtime.remember(action, **kwargs)), False
    except (ValueError, TypeError) as e:
        return _to_json(_error_payload(str(e))), True
    except NotImplementedError as e:
        return _to_json(_error_payload(str(e))), True
    except Exception as e:
        log.warning("prism_remember 异常：%s", e)
        return _to_json(_error_payload(str(e))), True


def _handle_recall(
    runtime: PrismRuntime, args: dict[str, Any]
) -> tuple[str, bool]:
    try:
        action, kwargs = _split_action(args)
        if not action:
            return _to_json(_error_payload("Missing required argument: action")), True
        result = runtime.recall(action, **kwargs)
        # search(as_markdown=True) 返字符串；其余返 list[dict] 或 None
        if isinstance(result, str):
            return _to_json({"markdown": result}), False
        if result is None:
            return _to_json({"results": [], "count": 0}), False
        count = len(result) if hasattr(result, "__len__") else 0
        return _to_json({"results": result, "count": count}), False
    except (ValueError, TypeError) as e:
        return _to_json(_error_payload(str(e))), True
    except NotImplementedError as e:
        return _to_json(_error_payload(str(e))), True
    except Exception as e:
        log.warning("prism_recall 异常：%s", e)
        return _to_json(_error_payload(str(e))), True


def _handle_admin(
    runtime: PrismRuntime, args: dict[str, Any]
) -> tuple[str, bool]:
    try:
        action, kwargs = _split_action(args)
        if not action:
            return _to_json(_error_payload("Missing required argument: action")), True
        return _to_json(runtime.admin(action, **kwargs)), False
    except (ValueError, TypeError) as e:
        return _to_json(_error_payload(str(e))), True
    except NotImplementedError as e:
        return _to_json(_error_payload(str(e))), True
    except Exception as e:
        log.warning("prism_admin 异常：%s", e)
        return _to_json(_error_payload(str(e))), True


def to_text_content(json_text: str) -> list[mcp_types.TextContent]:
    """把 :func:`call_prism_tool` 的 json_text 包成 MCP TextContent 列表。"""
    return [mcp_types.TextContent(type="text", text=json_text)]
