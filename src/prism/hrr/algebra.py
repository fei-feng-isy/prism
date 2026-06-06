"""HRR 代数 — bind / unbind / bundle（Plate 1995 相位向量版本）。

三个原语：
    - ``bind(a, b)   = (a + b) mod 2π``  — 角度合成；O(dim)
    - ``unbind(a, b) = (a - b) mod 2π``  — bind 的逆；``unbind(bind(a, b), b) == a``
    - ``bundle(*v)   = angle(Σ e^iv)``    — 单位复向量循环均值；归一化回 [0, 2π)

所有输入必须为 ``np.ndarray``、``shape=(dim,)``、``dtype=float64``、取值 ``[0, 2π)``。
返回值同样满足此约束。热路径仅做 shape/dtype 检查，完整校验由反序列化层负责。
"""

from __future__ import annotations

import numpy as np

from .atoms import TWO_PI

__all__ = [
    "ShapeMismatchError",
    "bind",
    "bundle",
    "unbind",
]


class ShapeMismatchError(ValueError):
    """两个或以上 HRR 向量在 shape / dtype 上不一致。"""


def _check_phase_vector(v: object, *, role: str) -> None:
    if not isinstance(v, np.ndarray):
        raise ShapeMismatchError(f"{role} 必须是 np.ndarray，实际：{type(v).__name__}")
    if v.ndim != 1:
        raise ShapeMismatchError(f"{role} 必须是 1D 向量，实际 ndim={v.ndim}")
    if v.dtype != np.float64:
        raise ShapeMismatchError(f"{role} dtype 必须是 float64，实际：{v.dtype}")


def _check_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ShapeMismatchError(f"shape 不一致：a={a.shape} vs b={b.shape}")


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """合成两个相位向量：``(a + b) mod 2π``。

    交换性：``bind(a, b) == bind(b, a)`` 在浮点精度下严格成立（加法可交换）。
    结合性：``bind(bind(a, b), c) == bind(a, bind(b, c))`` 仅在 mod 2π 下成立，
    单步浮点尾数可能差 1 ulp，不要用于身份比对。

    Args:
        a, b: 相位向量，必须 shape / dtype 一致。

    Returns:
        新数组（不就地修改输入），shape/dtype 与输入相同，取值 [0, 2π)。

    Raises:
        ShapeMismatchError: 类型 / 维度 / dtype 不符。
    """
    _check_phase_vector(a, role="a")
    _check_phase_vector(b, role="b")
    _check_same_shape(a, b)
    return np.mod(a + b, TWO_PI)


def unbind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """bind 的精确逆：``(a - b) mod 2π``。

    核心 invariant：``unbind(bind(a, b), b) == a`` 在 mod 2π 下成立；浮点尾数
    误差量级 < 1e-12（用 :func:`numpy.allclose` 比较即可）。

    Args:
        a, b: 相位向量。

    Returns:
        新数组，shape/dtype 同输入，取值 [0, 2π)。

    Raises:
        ShapeMismatchError: 类型 / 维度 / dtype 不符。
    """
    _check_phase_vector(a, role="a")
    _check_phase_vector(b, role="b")
    _check_same_shape(a, b)
    return np.mod(a - b, TWO_PI)


def bundle(*vectors: np.ndarray) -> np.ndarray:
    """循环均值叠加：``angle(Σ_i exp(1j · v_i))``，归一化回 [0, 2π)。

    几何含义：把每个相位映射到单位复圆 ``z_i = e^{i v_i}``，求矢量和的辐角。
    Plate 1995 中这是 HRR "概率叠加 / 模糊集合并" 的原语。

    与 add：
        本函数与逐项 ``bind`` 不同 —— bind 是合成单条 (role, filler)，
        bundle 是把若干条 fact 聚到同一 Bank 状态向量里。

    奇异点：
        若所有 vectors 在元素 ``k`` 上完美对称抵消（合矢量为 0+0j），
        ``np.angle(0)`` 返回 0；此时该维度无法恢复任何信息，但向量本身仍合法。
        Bank SNR 监控会捕获此类退化。

    Args:
        *vectors: 至少一个相位向量；所有向量 shape / dtype 必须一致。

    Returns:
        合成相位向量，shape/dtype 同输入，取值 [0, 2π)。

    Raises:
        ValueError: 未传入任何 vector（无法推断 dim；初始零相位状态请显式
            ``np.zeros(dim, dtype=np.float64)``）。
        ShapeMismatchError: 任一向量类型 / 维度 / dtype 不符或 shape 不一致。
    """
    if not vectors:
        raise ValueError("bundle 至少需要 1 个向量；零相位初态请用 np.zeros(dim, dtype=np.float64)")

    first = vectors[0]
    _check_phase_vector(first, role="vectors[0]")
    shape = first.shape
    for i, v in enumerate(vectors[1:], start=1):
        _check_phase_vector(v, role=f"vectors[{i}]")
        if v.shape != shape:
            raise ShapeMismatchError(f"shape 不一致：vectors[0]={shape} vs vectors[{i}]={v.shape}")

    # 累加单位复向量；complex128 与 float64 实部精度匹配
    z = np.zeros(shape, dtype=np.complex128)
    for v in vectors:
        z += np.exp(1j * v)
    # np.angle 返回 (-π, π]；归一化到 [0, 2π) 以保持与 atom 输出域一致
    return np.mod(np.angle(z), TWO_PI)
