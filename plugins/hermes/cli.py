"""Hermes CLI 桥 -- 把 prism 运维 CLI 接到 ``hermes prism ...`` 子命令下。

薄壳 forwarder：用 ``argparse.REMAINDER`` 兜住 ``hermes prism`` 之后的全部
argv，透传给 ``prism.cli.__main__.main``，进程内调用，不走 subprocess。
"""

from __future__ import annotations

import argparse

__all__ = ["prism_command", "register_cli"]


# ─── Hermes 协议入口 ──────────────────────────────────────────────────────


def register_cli(subparser: argparse.ArgumentParser) -> None:
    """挂载 ``hermes prism ...`` 的 REMAINDER 位置参数，把后续 argv 透传给 prism CLI。"""
    subparser.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help=(
            "prism 子命令 + 参数；与 ``python -m prism`` 完全一致 "
            "（doctor / migrate / reindex / vstore-migrate / memory / eval / export）"
        ),
    )
    subparser.set_defaults(func=prism_command)


def prism_command(args: argparse.Namespace) -> None:
    """``hermes prism ...`` dispatch 入口，转发给 ``prism.cli.__main__.main``。"""
    rest: list[str] = list(getattr(args, "rest", None) or [])

    from prism.cli.__main__ import main as _prism_main

    raise SystemExit(_prism_main(rest))
