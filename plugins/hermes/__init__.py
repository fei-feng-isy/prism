"""Prism -- Hermes MemoryProvider 适配层。

把 Prism 包装成符合 Hermes ``MemoryProvider`` ABC 的插件，注册三个 LLM tool：
``prism_remember`` / ``prism_recall`` / ``prism_admin``。

部署::

    ln -s /path/to/prism/plugins/hermes "$HERMES_HOME/plugins/prism"
    hermes memory setup prism
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

# 让 `from prism.X import ...` 在 user-installed 部署模式下工作 —
# $HERMES_HOME/plugins/prism/ 通过 symlink 指向本仓库的 plugins/hermes，
# 真正的 prism 包在仓库的 src/prism/。
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from agent.memory_provider import MemoryProvider  # noqa: E402

from prism.api import PrismAdmin, PrismRecall, PrismRemember  # noqa: E402
from prism.config import (  # noqa: E402
    dump_default_config_yaml,
    patch_config_file,
    reset_agent_home,
    resolve_config_path,
    resolve_data_home,
    set_agent_home,
)
from prism.entities.regex_extractor import preload_jieba  # noqa: E402
from prism.mcp.wire import PrismRuntime, RuntimeOptions, build_runtime  # noqa: E402
from prism.mirror import PrismMirror  # noqa: E402
from prism.retriever import SmartPrefetch  # noqa: E402

log = logging.getLogger(__name__)


# ─── Tool schemas（OpenAI function calling 格式）──────────────────────

PRISM_REMEMBER_SCHEMA: dict[str, Any] = {
    "name": "prism_remember",
    "description": (
        "Write or modify facts in Prism memory (Chinese-optimized hybrid memory).\n\n"
        "ACTIONS:\n"
        "• add — Store a new fact the user would expect you to remember later.\n"
        "  Examples: 'Alice 喜欢 PostgreSQL 14', 'project deadline is 2026-07-01'.\n"
        "• update / remove / helpful / unhelpful — Modify existing facts or provide feedback.\n\n"
        "WHEN TO USE: Whenever the user shares durable preferences, facts about people, "
        "projects, decisions, or environment that would be useful in future turns. "
        "Prefer prism_remember(add) over the built-in memory tool for structured/entity-rich "
        "facts because Prism enables entity probe and multi-entity reasoning."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "update", "remove", "helpful", "unhelpful"],
                "description": "Action to perform. MVP only supports 'add'.",
            },
            "content": {
                "type": "string",
                "description": "Fact text (required for 'add'). Be concise and self-contained.",
            },
            "category": {
                "type": "string",
                "enum": ["user_pref", "user_env", "project", "tool", "general"],
                "description": "Fact category (defaults to 'general'). user_pref = user preferences; user_env = user environment/tooling; project = project facts; tool = tool config.",
            },
        },
        "required": ["action"],
    },
}


PRISM_RECALL_SCHEMA: dict[str, Any] = {
    "name": "prism_recall",
    "description": (
        "Query Prism memory (Chinese-optimized hybrid search: semantic + FTS + entity).\n\n"
        "ACTIONS (use the simplest one that fits):\n"
        "• search — Free-text query. Returns top-k most relevant facts as a markdown snippet.\n"
        "  Example: prism_recall(action='search', query='用户的沟通风格')\n"
        "• probe — Find all facts that mention an entity (person / project / concept).\n"
        "  Example: prism_recall(action='probe', entity='张三')\n"
        "• reason — Find facts that mention ALL of the given entities (AND-join).\n"
        "  Example: prism_recall(action='reason', entities=['ffeng', 'PostgreSQL'])\n"
        "• related — Find entities that co-occur with the given entity in the same fact.\n"
        "  Example: prism_recall(action='related', entity='张三')\n"
        "• contradict — Find facts that contradict each other.\n\n"
        "WHEN TO USE: Before answering questions about the user / project / past decisions, "
        "ALWAYS call prism_recall first. Prefetched context is auto-injected, but explicit "
        "queries get more targeted results."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "probe", "reason", "related", "contradict"],
            },
            "query": {
                "type": "string",
                "description": "Free-text query (required for 'search').",
            },
            "entity": {
                "type": "string",
                "description": "Single entity name (required for 'probe' / 'related').",
            },
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Multiple entity names (required for 'reason'; AND-joined).",
            },
            "category": {
                "type": "string",
                "enum": ["user_pref", "user_env", "project", "tool", "general"],
                "description": "Optional category filter.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default: 5 for search, 10 for probe/reason/related).",
            },
            "min_trust": {
                "type": "number",
                "description": "Minimum trust score filter (search only, requires as_markdown=false).",
            },
            "as_markdown": {
                "type": "boolean",
                "description": "search only: True (default) returns LLM-friendly markdown; False returns structured list[dict] with path_scores.",
            },
        },
        "required": ["action"],
    },
}


PRISM_ADMIN_SCHEMA: dict[str, Any] = {
    "name": "prism_admin",
    "description": (
        "Inspect / manage Prism memory store (operator-facing).\n\n"
        "ACTIONS:\n"
        "• stats — Return health panel JSON: db counts / vstore capacity / "
        "semantic backend status / prefetch warm-up state / retriever weights.\n"
        "• list / archive / restore — Browse and manage stored facts.\n"
        "• enrichment_diagnose — Deep enrichment diagnostics: queue items, "
        "status distribution, missing vectors.\n"
        "• enrichment_fix — Fix missing embeddings, clear pending queue, "
        "rebuild vstore. Pass dry_run=true to preview.\n\n"
        "WHEN TO USE: When the user asks about memory state ('how many facts do you "
        "remember?', '记忆库里有多少条事实?') or to debug retrieval issues."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["stats", "list", "archive", "restore", "enrichment_diagnose", "enrichment_fix"],
            },
            "category": {
                "type": "string",
                "enum": ["user_pref", "user_env", "project", "tool", "general"],
                "description": "Optional category filter for 'stats' (only affects db section counts).",
            },
            "dry_run": {
                "type": "boolean",
                "description": "enrichment_fix only: True=report without modifying; default False.",
            },
        },
        "required": ["action"],
    },
}


# ─── slash 命令会话上下文（module-level，供 slash.py 读取）──────────────────

_active_db_path: str | None = None


# ─── 配置加载 ────────────────────────────────────────────────────────────────



def _sync_deps_skills(data_home: Path) -> None:
    """将 deps/hermes/skills 复制到 data_home/skills（仅目标缺失时触发）。"""
    dst = data_home / "skills"
    if dst.is_dir() and (dst / "SKILL.md").exists():
        return
    import shutil

    src = _REPO_ROOT / "deps" / "hermes" / "skills"
    if not src.is_dir():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    log.debug("Prism skills synced: %s → %s", src, dst)


def _ensure_config_file(path: Path, *, data_home: Path | None = None) -> None:
    """``path`` 不存在则写入默认 YAML 模板；已存在则补全缺失配置段。

    Args:
        data_home: Hermes 场景下传入实际 data_home（如 ``~/.hermes/prism``），
            替换默认的 ``~/.prism``，让 CLI 命令读 config 时解析到正确路径。
    """
    if path.exists():
        added = patch_config_file(path)
        if added:
            log.info("Prism config 已补全缺失段：%s → %s", path, ", ".join(added))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    from prism.config import DbConfig

    content = dump_default_config_yaml()
    if data_home is not None:
        content = content.replace(
            f"data_home_default: {DbConfig.data_home_default}",
            f"data_home_default: {data_home}",
        )
    path.write_text(content, encoding="utf-8")
    log.info("Prism 默认 config 已生成：%s", path)


# ─── 工具：JSON 兜底 ────────────────────────────────────────────────────────


def _to_json(payload: Any) -> str:
    """``json.dumps`` 兜底 ndarray / set 等非原生类型，防单点序列化失败。"""
    return json.dumps(payload, ensure_ascii=False, default=str)


def _tool_error(msg: str) -> str:
    """与 holographic 一致的 error JSON 包装（``tools.registry.tool_error`` 同等效果）。"""
    return _to_json({"error": msg})


# ─── Provider 主类 ──────────────────────────────────────────────────────────


class PrismMemoryProvider(MemoryProvider):
    """Hermes 端 Prism memory provider 包装。

    生命周期由 ``MemoryManager`` 调用：
        ``__init__`` → ``is_available`` → ``initialize`` → ``system_prompt_block``
        → 每轮 ``prefetch`` → ``handle_tool_call`` （按需） → ``on_memory_write``
        （内置 memory 镜像） → 退出 ``shutdown``

    Args:
        config_path: 可选 prism config YAML 路径覆盖（仅测试用）。生产路径在
            :meth:`initialize` 里从 ``$HERMES_HOME/prism/config.yaml`` 推导，
            不存在自动生成默认模板。

    Attributes:
        _remember / _recall / _admin: 各工具 dispatch 入口（initialize 后非空）
        _mirror: PrismMirror 实例（on_memory_write 转发目标）
    """

    def __init__(self, *, config_path: str | os.PathLike[str] | None = None) -> None:
        self._config_path: Path | None = (
            Path(config_path).expanduser() if config_path else None
        )
        self._runtime: PrismRuntime | None = None
        self._mirror: PrismMirror | None = None
        self._remember: PrismRemember | None = None
        self._recall: PrismRecall | None = None
        self._admin: PrismAdmin | None = None
        self._prefetch: SmartPrefetch | None = None
        self._session_id: str = ""
        self._jieba_thread: threading.Thread | None = None

    @property
    def name(self) -> str:
        return "prism"

    # ─── 生命周期 ────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """jieba + numpy + sqlite3 是必备依赖（pip install prism-memory 已带）；
        sentence-transformers 为可选 — 缺失走降级路径（fts + jaccard 0.65/0.35）。
        """
        return True

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """构造完整 pipeline，委托 :func:`build_runtime`。

        kwargs:
            hermes_home: Hermes 根目录，Prism 数据放在其下 ``prism/`` 子目录。
            agent_identity: profile 名（如 "coder"），无则 "default"。
            user_id: 用于 user_hash；无则 "local_default"。
        """
        if self._runtime is not None:
            log.warning(
                "PrismMemoryProvider.initialize 被重复调用（session_id=%s）；"
                "先 shutdown 旧实例再重建",
                session_id,
            )
            self.shutdown()

        t0 = time.monotonic()
        self._session_id = session_id

        # 1) 注入 agent_home + 解析 data_home（→ agent_home/prism）+ config
        hermes_home_kw = kwargs.get("hermes_home")
        hermes_home = (
            Path(hermes_home_kw).expanduser()
            if hermes_home_kw
            else Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
        )
        set_agent_home(hermes_home)
        data_home = resolve_data_home()
        cfg_path = resolve_config_path(
            override=self._config_path, data_home=data_home
        )
        _ensure_config_file(cfg_path, data_home=data_home)
        _sync_deps_skills(data_home)
        t1 = time.monotonic()
        log.info("Prism init step 1/3 config: %.0fms", (t1 - t0) * 1000)

        # 2) build_runtime 统一装配（db/bank/semantic/vstore/mirror/pipeline/prefetch/service/APIs）
        profile = kwargs.get("agent_identity") or "default"
        user_id = kwargs.get("user_id") or "local_default"
        self._runtime = build_runtime(RuntimeOptions(
            config_path=str(cfg_path),
            profile=profile,
            user_id=user_id,
            data_home=str(data_home),
            start_worker=False,
            warmup_prefetch=False,
            call_source="hermes",
        ))
        self._mirror = self._runtime.mirror
        self._prefetch = self._runtime.prefetch
        self._remember = self._runtime.remember
        self._recall = self._runtime.recall
        self._admin = self._runtime.admin
        t2 = time.monotonic()
        log.info("Prism init step 2/3 build_runtime: %.0fms", (t2 - t1) * 1000)

        # 3) 异步后台预热 jieba（BGE warmup 由 runtime 自身或此处触发）
        def _warmup_jieba() -> None:
            try:
                preload_jieba()
                log.info("Prism jieba 预加载完成（后台）")
            except Exception:
                log.exception("jieba 预加载异常（后台）")
            try:
                ok = self._prefetch.warmup()
                log.info("Prism BGE warmup 完成（后台）：loaded=%s", ok)
            except Exception:
                log.exception("prefetch warmup 异常（后台）")

        self._jieba_thread = threading.Thread(
            target=_warmup_jieba, daemon=True, name="prism-warmup"
        )
        self._jieba_thread.start()
        t3 = time.monotonic()
        log.info(
            "Prism init step 3/3 warmup-thread-spawn: %.0fms (total: %.0fms)",
            (t3 - t2) * 1000, (t3 - t0) * 1000,
        )

        global _active_db_path
        _active_db_path = str(self._runtime.db_path)
        os.environ["_PRISM_HERMES_DB_PATH"] = _active_db_path

    def shutdown(self) -> None:
        """委托 :meth:`PrismRuntime.shutdown`（幂等关 worker → vstore → DB）。"""
        if self._jieba_thread is not None and self._jieba_thread.is_alive():
            self._jieba_thread.join(timeout=2.0)
            if self._jieba_thread.is_alive():
                log.warning(
                    "prism-warmup 线程未在 2s 内退出；继续 shutdown（daemon 线程会被进程清理）"
                )

        global _active_db_path
        _active_db_path = None
        os.environ.pop("_PRISM_HERMES_DB_PATH", None)
        reset_agent_home()

        if self._runtime is not None:
            self._runtime.shutdown()

        self._runtime = None
        self._mirror = None
        self._remember = None
        self._recall = None
        self._admin = None
        self._prefetch = None
        self._jieba_thread = None

    # ─── 系统 prompt 块 ──────────────────────────────────────────────────

    def system_prompt_block(self) -> str:
        if self._admin is None or self._prefetch is None:
            return ""
        try:
            stats = self._admin.stats()
            active = stats["facts"]["active"]
            retr = stats["retriever"]
            # system_prompt 只反映 *永久* 降级（包缺失）；
            # transient（异步 warmup 中）不写入 session — 否则该字符串会被
            # session DB 缓存，warmup 完成后用户仍看到 "degraded"。
            degraded_permanent = retr.get("degraded_permanent", retr.get("degraded", False))
        except Exception:
            return ""

        mode = " (degraded: fts+jaccard only)" if degraded_permanent else ""
        ops_hint = (
            "Ops commands available in-chat via `/prism <sub>` "
            "(migrate / reindex / vstore-migrate / memory; equivalent to `hermes prism ...`)."
        )
        if active == 0:
            return (
                "# Prism Memory\n"
                f"Active{mode}. Empty fact store — proactively call prism_remember(add) "
                "for facts the user would expect you to remember.\n"
                "Use prism_recall(search|probe|reason|related) before answering questions "
                "about the user / project.\n"
                f"{ops_hint}"
            )
        return (
            f"# Prism Memory\n"
            f"Active{mode}. {active} facts stored (entity-resolved + HRR-encoded).\n"
            f"Use prism_recall before answering user-specific questions; use prism_remember "
            f"to store durable facts; use prism_admin(stats) to inspect the store.\n"
            f"{ops_hint}"
        )

    # ─── 每轮 prefetch（LLM 注入 markdown）────────────────────────────────

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch is None or not query:
            return ""
        try:
            return self._prefetch.prefetch(query)
        except Exception as e:
            log.debug("Prism prefetch 失败：%s", e)
            return ""

    # ─── tool schemas + dispatch ─────────────────────────────────────────

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [PRISM_REMEMBER_SCHEMA, PRISM_RECALL_SCHEMA, PRISM_ADMIN_SCHEMA]

    def handle_tool_call(
        self, tool_name: str, args: dict[str, Any], **kwargs: Any
    ) -> str:
        if tool_name == "prism_remember":
            return self._handle_remember(args)
        if tool_name == "prism_recall":
            return self._handle_recall(args)
        if tool_name == "prism_admin":
            return self._handle_admin(args)
        return _tool_error(f"Unknown tool: {tool_name}")

    # ─── 内置 memory 镜像 ────────────────────────────────────────────────

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._mirror is None:
            return
        self._mirror.on_memory_write(action, target, content, metadata=metadata)

    # ─── 内部 dispatch helpers ───────────────────────────────────────────

    def _handle_remember(self, args: dict[str, Any]) -> str:
        if self._remember is None:
            return _tool_error("Prism not initialized")
        try:
            action = args.get("action")
            if not action:
                return _tool_error("Missing required argument: action")
            kwargs = {k: v for k, v in args.items() if k != "action"}
            return _to_json(self._remember(action, **kwargs))
        except (ValueError, TypeError) as e:
            return _tool_error(str(e))
        except NotImplementedError as e:
            return _tool_error(str(e))
        except Exception as e:
            log.warning("prism_remember 异常：%s", e)
            return _tool_error(str(e))

    def _handle_recall(self, args: dict[str, Any]) -> str:
        if self._recall is None:
            return _tool_error("Prism not initialized")
        try:
            action = args.get("action")
            if not action:
                return _tool_error("Missing required argument: action")
            kwargs = {k: v for k, v in args.items() if k != "action"}
            result = self._recall(action, **kwargs)
            # search(as_markdown=True) 返字符串 → 直接 wrap；其它返 list[dict]
            if isinstance(result, str):
                return _to_json({"markdown": result})
            return _to_json({"results": result, "count": len(result) if hasattr(result, "__len__") else 0})
        except (ValueError, TypeError) as e:
            return _tool_error(str(e))
        except NotImplementedError as e:
            return _tool_error(str(e))
        except Exception as e:
            log.warning("prism_recall 异常：%s", e)
            return _tool_error(str(e))

    def _handle_admin(self, args: dict[str, Any]) -> str:
        if self._admin is None:
            return _tool_error("Prism not initialized")
        try:
            action = args.get("action")
            if not action:
                return _tool_error("Missing required argument: action")
            kwargs = {k: v for k, v in args.items() if k != "action"}
            return _to_json(self._admin(action, **kwargs))
        except (ValueError, TypeError) as e:
            return _tool_error(str(e))
        except NotImplementedError as e:
            return _tool_error(str(e))
        except Exception as e:
            log.warning("prism_admin 异常：%s", e)
            return _tool_error(str(e))


# ─── 插件入口 ────────────────────────────────────────────────────────────────


def register(ctx: Any) -> None:
    """Hermes 插件入口 -- 按 ctx 能力分流注册 memory provider 和 slash 命令。"""
    if hasattr(ctx, "register_memory_provider"):
        provider = PrismMemoryProvider()
        ctx.register_memory_provider(provider)

    if hasattr(ctx, "register_command"):
        # 懒 import：仅 PluginManager 路径需要 slash 模块。
        # 必须用相对 import：hermes 的 plugins/ 占用了 sys.modules['plugins']。
        from .slash import SLASH_DESCRIPTION, handle_prism_slash
        ctx.register_command(
            "prism",
            handler=handle_prism_slash,
            description=SLASH_DESCRIPTION,
            args_hint="<subcommand> [args...]",
        )
