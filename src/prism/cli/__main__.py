"""``python -m prism`` 入口 — 自动发现 cli 子命令。"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable

from prism.cli.interface import CliManifest

# ── 自动发现 ─────────────────────────────────────────────────

_CLI_MODULES: list[str] = [
    "prism.cli.doctor",
    "prism.cli.migrate",
    "prism.cli.reindex",
    "prism.cli.vstore_migrate",
    "prism.cli.memory",
    "prism.cli.eval",
    "prism.cli.export",
]

_registry: dict[str, tuple[Callable[[list[str]], int], str]] = {}
_usage_lines: list[str] = []

for _mod_name in _CLI_MODULES:
    _mod = importlib.import_module(_mod_name)
    _manifest: CliManifest = getattr(_mod, "MANIFEST")
    for _reg in _manifest.registrations:
        _registry[_reg.name] = (_reg.handler, _reg.usage_line)
        _usage_lines.append(f"  {_reg.usage_line}")

USAGE = (
    "────────────────────────────────────────"
    "\n可用子命令：\n"
    + "\n".join(_usage_lines)
    + "\n\n完整 help：python -m prism <sub> --help\n"
    "────────────────────────────────────────"
)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(USAGE, file=sys.stderr)
        return 0 if argv else 1

    sub, rest = argv[0], argv[1:]
    entry = _registry.get(sub)
    if entry is not None:
        handler, _ = entry
        return handler(rest)

    print(f"未知子命令：{sub}\n\n{USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
