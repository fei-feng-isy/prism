"""Prism HRR 层 — Plate 1995 风格的全息归约表示。

暴露的原语：
    - :func:`atom` / :func:`atom_batch` — SHA-256 派生的确定性原子向量
    - :func:`bind` / :func:`unbind` / :func:`bundle` — 相位代数
    - :class:`IncrementalBank` / :class:`BankState` — per-category 增量 Bank
    - :class:`RebuildDebouncer` — 50ms 窗口去抖工具，配合 ``remove`` 合并校准
    - :func:`is_valid_atom` — 输入校验工具（DB / 网络反序列化用）
    - :exc:`AtomDimensionError` / :exc:`ShapeMismatchError` / :exc:`BankConfigError`
"""

from __future__ import annotations

from .algebra import ShapeMismatchError, bind, bundle, unbind
from .atoms import (
    TWO_PI,
    AtomDimensionError,
    atom,
    atom_batch,
    is_valid_atom,
)
from .bank import BankConfigError, BankState, IncrementalBank, RebuildDebouncer

__all__ = [
    "TWO_PI",
    "AtomDimensionError",
    "BankConfigError",
    "BankState",
    "IncrementalBank",
    "RebuildDebouncer",
    "ShapeMismatchError",
    "atom",
    "atom_batch",
    "bind",
    "bundle",
    "is_valid_atom",
    "unbind",
]
