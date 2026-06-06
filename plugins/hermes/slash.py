"""``/prism`` 对话内 slash 命令 -- Hermes REPL 里的运维入口。

与 ``hermes prism <sub>`` CLI 桥并存，最终都 dispatch 到
``prism.cli.__main__.main(argv)``。输出通过 ``redirect_stdout/stderr``
缓冲后一次性返回给 REPL；异常全部捕获，不向 REPL 抛。
"""

from __future__ import annotations

import io
import os
import shlex
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout

__all__ = ["handle_prism_slash", "SLASH_DESCRIPTION"]

SLASH_DESCRIPTION = "Prism 运维命令（doctor / migrate / reindex / vstore-migrate / memory / eval / export）"

_DB_SUBCOMMANDS = frozenset({"memory", "reindex", "vstore-migrate", "export"})


def _ensure_agent_home() -> None:
    """确保 agent_home 已设置，即使 initialize() 尚未被调用。

    用户在 Hermes 启动后未进行对话就直接执行 ``/prism`` 命令时，
    ``PrismMemoryProvider.initialize()`` 还没跑过，``agent_home`` 为 None，
    导致 ``discover_config_path()`` 找不到 ``<agent_home>/prism/config.yaml``，
    CLI 回退到 ``default_config()``，DB 路径错误。

    此函数在 slash handler 入口处调用，轻量设置 agent_home 即可让
    CLI 的配置发现链路正确工作。
    """
    from prism.config import get_agent_home, set_agent_home

    if get_agent_home() is not None:
        return
    hermes_home = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
    set_agent_home(hermes_home)


def _inject_session_context(argv: list[str]) -> list[str]:
    """从环境变量注入 ``--db``，让 slash 命令操作与 Hermes 会话相同的 DB。

    ``--config`` 不需要注入 —— CLI 通过 ``discover_config_path()`` 自行发现。
    ``--db`` 仍需注入，因为 DB path 依赖 session 级 ``user_id``，CLI 默认
    ``local_default`` 可能不匹配。

    环境变量由 ``PrismMemoryProvider.initialize()`` 设置，进程全局可见，
    不受 Hermes 双重模块加载（memory loader vs PluginManager）影响。
    """
    if not argv:
        return argv
    db_path = os.environ.get("_PRISM_HERMES_DB_PATH")
    if db_path and argv[0] in _DB_SUBCOMMANDS and "--db" not in argv:
        argv = [*argv, "--db", db_path]
    return argv


def handle_prism_slash(raw_args: str) -> str:
    """Hermes 插件 slash 命令回调；签名固定为 ``(raw_args: str) -> str | None``。"""
    _ensure_agent_home()
    argv = shlex.split(raw_args.strip()) if raw_args and raw_args.strip() else []
    argv = _inject_session_context(argv)

    from prism.cli.__main__ import main as _prism_main

    out = io.StringIO()
    err = io.StringIO()
    rc: int
    try:
        with redirect_stdout(out), redirect_stderr(err):
            rc = _prism_main(argv) or 0
    except SystemExit as e:
        rc = int(e.code) if isinstance(e.code, int) else (1 if e.code else 0)
    except Exception as e:
        tail = (err.getvalue() or out.getvalue()).rstrip()
        return (
            f"[/prism] 异常：{type(e).__name__}: {e}"
            + (f"\n{tail}" if tail else "")
        )

    body = (out.getvalue() + err.getvalue()).rstrip()
    if rc != 0:
        body = f"{body}\n[/prism] exit code: {rc}" if body else f"[/prism] exit code: {rc}"
    return body or "(no output)"
