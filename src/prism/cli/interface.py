"""CLI 接口协议 — 每个 CLI 模块返回命令注册信息，__main__ 自动发现。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CliRegistration:
    """单个 CLI 命令的注册信息。

    Attributes:
        name: 子命令名，如 ``\"doctor\"``、``\"migrate\"``。
        handler: 入口函数，签名为 ``(argv: list[str]) -> int``。
        usage_line: 单行 usage，如 ``\"doctor  健康自检\"``。
            子命令（如 memory）可传多行续写，首行缩进后就是展示行。
    """

    name: str
    handler: Callable[[list[str]], int]
    usage_line: str


@dataclass(frozen=True, slots=True)
class CliManifest:
    """一个 CLI 模块返回的完整清单。

    ``registrations`` 中的每条对应一个独立的 ``python -m prism <name>`` 入口。
    """

    registrations: list[CliRegistration] = field(default_factory=list)


def single(
    name: str,
    handler: Callable[[list[str]], int],
    usage_line: str,
) -> CliManifest:
    """便捷构造：单个命令的注册。"""
    return CliManifest([CliRegistration(name=name, handler=handler, usage_line=usage_line)])
