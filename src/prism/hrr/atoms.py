"""HRR 原子向量 — Plate 1995 风格的相位向量派生。

atom 向量由 SHA-256(name) 派生，不承载语义；唯一职责是在 bind/unbind/bundle
代数下保持身份可识别。语义相似度交给 SemanticBackend。

向量形状：dtype=float64, shape=(dim,), dim 为正偶数, 取值 [0, 2π)。
相同 (name, dim) 在任意环境下产出完全相同的向量（跨平台确定性）。
"""

from __future__ import annotations

import hashlib
from typing import Final

import numpy as np

__all__ = [
    "AtomDimensionError",
    "atom",
    "atom_batch",
    "is_valid_atom",
]


class AtomDimensionError(ValueError):
    """维度参数非法：必须是正偶数。"""


# 2π 的高精度常量；np.float64(2π) 在不同平台位级一致（IEEE 754）
TWO_PI: Final[float] = float(2.0 * np.pi)

# SHA-256 摘要长度（字节）
_DIGEST_BYTES: Final[int] = 32


def _validate_dim(dim: int) -> None:
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise AtomDimensionError(f"dim 必须是 int，实际：{type(dim).__name__}")
    if dim <= 0:
        raise AtomDimensionError(f"dim 必须是正整数，实际：{dim}")
    if dim % 2 != 0:
        raise AtomDimensionError(f"dim 必须是偶数（HRR 代数约束），实际：{dim}")


def atom(name: str, dim: int = 1024) -> np.ndarray:
    """对 ``name`` 派生确定性相位向量。

    构造：
        1. SHA-256(name) → 32 字节摘要 ``d``
        2. 滚动 hash 块：第 ``i`` 块为 ``SHA-256(d || i.to_bytes(4, "big"))``
           其中 ``i = 0, 1, 2, ...``，按需取直到填满 ``dim * 8`` 字节
        3. 将字节流按 uint64 little-endian 解码为 ``dim`` 个 64bit 整数
        4. 归一化到 ``[0, 2π)``：``phase = (u / 2**64) * 2π``

    Args:
        name: 实体名 / role 名 / 任意字符串标识。必须非空。
        dim: 向量维度，必须是正偶数。默认 1024（与 ``HrrConfig.dim`` 对齐）。

    Returns:
        shape=(dim,) dtype=float64 的相位向量，取值 [0, 2π)。

    Raises:
        AtomDimensionError: dim 非正偶数。
        ValueError: name 为空字符串。
    """
    if not isinstance(name, str):
        raise ValueError(f"name 必须是 str，实际：{type(name).__name__}")
    if not name:
        raise ValueError("name 不能是空字符串")
    _validate_dim(dim)

    # 1) base digest
    base = hashlib.sha256(name.encode("utf-8")).digest()

    # 2) 滚动派生足够字节：每相位 8 字节
    bytes_needed = dim * 8
    chunks: list[bytes] = []
    collected = 0
    counter = 0
    while collected < bytes_needed:
        block = hashlib.sha256(base + counter.to_bytes(4, "big")).digest()
        chunks.append(block)
        collected += _DIGEST_BYTES
        counter += 1
    raw = b"".join(chunks)[:bytes_needed]

    # 3) 解码为 uint64
    uints = np.frombuffer(raw, dtype="<u8")
    assert uints.shape == (dim,)

    # 4) 归一化到 [0, 2π)
    # 用 float64 除法；2**64 在 float64 中精确表示（< 2^53 阈值但 2^64 也无误差）
    phases = (uints.astype(np.float64) / float(1 << 64)) * TWO_PI
    return phases


def atom_batch(names: list[str], dim: int = 1024) -> np.ndarray:
    """批量派生原子向量；返回 shape=(len(names), dim)。

    与逐个调用 :func:`atom` 在数值上完全等价；提供仅为减少分配。

    Args:
        names: 字符串列表（可空）。空列表返回 shape=(0, dim) 的空矩阵。
        dim: 向量维度。
    """
    _validate_dim(dim)
    if not names:
        return np.empty((0, dim), dtype=np.float64)
    out = np.empty((len(names), dim), dtype=np.float64)
    for i, n in enumerate(names):
        out[i] = atom(n, dim)
    return out


def is_valid_atom(vec: np.ndarray) -> bool:
    """``vec`` 形状 / dtype / 取值范围合法？

    用于防御性断言：从 DB / 网络反序列化的向量在进入 ``bind/unbind`` 前可调用本检查。
    """
    if not isinstance(vec, np.ndarray):
        return False
    if vec.ndim != 1 or vec.shape[0] == 0 or vec.shape[0] % 2 != 0:
        return False
    if vec.dtype != np.float64:
        return False
    return bool(np.all((vec >= 0.0) & (vec < TWO_PI)))
