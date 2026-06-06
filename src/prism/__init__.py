"""Prism — 中文优先的多模态记忆系统。"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("prism-memory")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = ["__version__"]
