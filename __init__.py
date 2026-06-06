"""Hermes plugin entrypoint when installed via ``hermes plugins install <git>``.

Hermes clones the whole repo into ``$HERMES_HOME/plugins/prism/``; its loader
expects ``__init__.py`` + ``plugin.yaml`` at the top level. The actual
``PrismMemoryProvider`` lives in ``plugins/hermes/`` so that local symlink
deployment (``ln -sfn $(pwd)/plugins/hermes $HERMES_HOME/plugins/prism``)
also keeps working — this file just bridges the two layouts.

**Lazy import (PEP 562 ``__getattr__`` / ``__dir__``)**: this module is
imported eagerly by pytest as part of package walk (because making the
repo root a package puts ``<repo>`` on the import path), but ``plugins.hermes``
pulls heavy runtime deps (numpy, jieba, ``agent.memory_provider`` ABC).
We defer the import until hermes loader actually does
``getattr(mod, "PrismMemoryProvider")`` after exec_module.

暴露顶层 ``register(ctx)`` 让 hermes 通用 PluginManager 能挂上
``/prism`` slash 命令；函数体内才懒 import 实现，保留启动期零 IO。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_PUBLIC_NAMES = (
    "PrismMemoryProvider",
    "PRISM_REMEMBER_SCHEMA",
    "PRISM_RECALL_SCHEMA",
    "PRISM_ADMIN_SCHEMA",
)


def register(ctx: Any) -> None:
    """Hermes plugin 入口 — memory loader 与通用 PluginManager 都会调用此函数。

    能力分流（memory provider 注册 / slash 命令注册）由
    ``plugins.hermes.register`` 内部的 ``hasattr(ctx, ...)`` 判断处理。
    本顶层入口只做懒 import + 委托，避免任何启动期 IO。

    放在顶层 ``register`` 而非 ``__getattr__`` 里：hermes 用
    ``getattr(module, "register", None)`` 探测，走 PEP562 路径会触发实现的
    完整 import，反而把懒加载收益吃掉。

    用**相对** import（``.plugins.hermes``）而非 absolute（``plugins.hermes``）：
    hermes 通用 PluginManager 把本 module 加载为 ``hermes_plugins.prism``，
    而 hermes 自己的 ``plugins/`` 已占用 ``sys.modules['plugins']`` 这个 key
    且 ``__path__`` 不含 prism repo —— 绝对 import 必失败并写入
    ``sys.modules['plugins.hermes'] = None`` negative cache，连带把 memory
    loader 路径也带崩（v1.0.7 P0 regression 教训）。相对 import 解析为
    ``hermes_plugins.prism.plugins.hermes`` / ``_hermes_user_memory.prism.plugins.hermes``，
    两条 namespace 都不冲突。
    """
    from .plugins.hermes import register as _impl
    _impl(ctx)


def __getattr__(name: str) -> Any:
    if name in _PUBLIC_NAMES:
        from .plugins.hermes import (
            PRISM_ADMIN_SCHEMA,
            PRISM_RECALL_SCHEMA,
            PRISM_REMEMBER_SCHEMA,
            PrismMemoryProvider,
        )
        resolved = {
            "PrismMemoryProvider": PrismMemoryProvider,
            "PRISM_REMEMBER_SCHEMA": PRISM_REMEMBER_SCHEMA,
            "PRISM_RECALL_SCHEMA": PRISM_RECALL_SCHEMA,
            "PRISM_ADMIN_SCHEMA": PRISM_ADMIN_SCHEMA,
        }
        for key, value in resolved.items():
            globals()[key] = value
        return resolved[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_PUBLIC_NAMES) | {"register"})
