"""HRR 增量 Bank — per-category 状态向量 + 校准触发器 + 去抖工具。

每个 category 维护一个 BankState，支持：
    - ``add``：O(dim) 增量累加单位复矢量到内部 z_sum
    - ``remove``：O(dim) 立即从 z_sum 减去（防止 probe 返回幽灵 fact）
    - ``calibrate``：从权威 fact 列表全量重建 z_sum，重置 dirty_count
    - ``needs_calibration``：纯查询函数

remove 采用"先减再校准"策略：立即 z_sum -= exp(i*v) 保证查询正确性，
调用方配合 RebuildDebouncer 在去抖窗口后全量 calibrate 修正浮点漂移。

内部维护 complex128 累加器 _z_sum，使增量 bundle 严格等价于全量 bundle。
SNR 触发默认关闭（snr_warn=0.0），仅用 dirty_count 触发校准。
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

import numpy as np

from .atoms import TWO_PI

__all__ = [
    "BankConfigError",
    "BankState",
    "IncrementalBank",
    "RebuildDebouncer",
]


class BankConfigError(ValueError):
    """构造 IncrementalBank 时参数非法。"""


@dataclass
class BankState:
    """单个 category 的 Bank 状态快照。

    Attributes:
        bank_vector: 当前 bundle 的相位向量；``np.mod(np.angle(z_sum), 2π)`` 派生。
            shape=(dim,) dtype=float64，取值 [0, 2π)。
        fact_count: 当前 Bank 包含的 fact 数。
        dirty_count: 自上次 calibrate 起的累计 add 数；calibrate 后归零。
        last_calibrated_at: 最后一次 calibrate 的 UTC 时间；未校准则 None。
        snr_estimate: 当前 SNR 估计 = mean(|z_sum| / fact_count)，[0, 1]。
            完美对齐 → 1.0；独立随机相位 fact → ≈ 1/√fact_count。

    Notes:
        - ``_z_sum`` 是内部 complex128 累加器，**不**对外承诺接口稳定。
          调用方需要 phase 向量请用 ``bank_vector``。
    """

    bank_vector: np.ndarray
    fact_count: int = 0
    dirty_count: int = 0
    last_calibrated_at: datetime | None = None
    snr_estimate: float = 1.0
    _z_sum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.complex128))


class IncrementalBank:
    """per-category HRR Bank，提供增量 add 与全量 calibrate。

    线程安全性：本类**不**做任何同步；调用方（mirror）保证按 category
    串行访问，或自行加锁。

    Args:
        dim: 相位向量维度，必须正偶数（与 :func:`prism.hrr.atom` 一致）。
        snr_warn: SNR 触发阈值；snr_estimate < snr_warn × 1.2 触发 ``needs_calibration``。
            默认 0.0（关闭）。
        dirty_pct: dirty_count 百分比阈值，默认 0.15。
        dirty_floor: dirty_count 绝对下限，默认 10。
    """

    def __init__(
        self,
        dim: int,
        *,
        snr_warn: float = 0.0,
        dirty_pct: float = 0.15,
        dirty_floor: int = 10,
    ) -> None:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 or dim % 2 != 0:
            raise BankConfigError(f"dim 必须是正偶数 int，实际：{dim!r}")
        if not (0.0 <= snr_warn <= 1.0):
            raise BankConfigError(f"snr_warn 必须在 [0, 1]，实际：{snr_warn}")
        if not (0.0 < dirty_pct <= 1.0):
            raise BankConfigError(f"dirty_pct 必须在 (0, 1]，实际：{dirty_pct}")
        if isinstance(dirty_floor, bool) or not isinstance(dirty_floor, int) or dirty_floor <= 0:
            raise BankConfigError(f"dirty_floor 必须是正 int，实际：{dirty_floor!r}")

        self.dim: Final[int] = dim
        self.snr_warn: Final[float] = snr_warn
        self.dirty_pct: Final[float] = dirty_pct
        self.dirty_floor: Final[int] = dirty_floor
        self._states: dict[str, BankState] = {}

    # ─── 写入路径 ─────────────────────────────────────────────────────

    def add(self, category: str, fact_vector: np.ndarray) -> None:
        """增量叠加单条 fact 向量到指定 category。

        Args:
            category: 分类标识；不存在会自动初始化。
            fact_vector: 已 bind 好的 fact 向量，shape=(dim,) dtype=float64。

        Raises:
            ValueError: shape / dtype 不符。
        """
        self._check_vector(fact_vector)
        z_new = np.exp(1j * fact_vector)
        state = self._states.get(category)
        if state is None:
            state = BankState(
                bank_vector=np.mod(np.angle(z_new), TWO_PI),
                fact_count=1,
                dirty_count=1,
                snr_estimate=1.0,  # |z_sum|=1, N=1 → 1.0
                _z_sum=z_new,
            )
            self._states[category] = state
            return

        state._z_sum = state._z_sum + z_new
        state.bank_vector = np.mod(np.angle(state._z_sum), TWO_PI)
        state.fact_count += 1
        state.dirty_count += 1
        state.snr_estimate = float(np.mean(np.abs(state._z_sum))) / state.fact_count

    def remove(self, category: str, fact_vector: np.ndarray) -> None:
        """从指定 category 立即减去单条 fact，更新 bank_vector 防幽灵。

        语义：``z_sum -= exp(i·fact_vector)``，O(dim)。
        移除最后一条 fact 时 category 状态被删除（与 calibrate([]) 一致）。
        调用方应配合 :class:`RebuildDebouncer` 在窗口结束后调用 ``calibrate``
        以修正浮点漂移并归零 dirty_count。

        Args:
            category: 分类标识。
            fact_vector: 要移除的 fact 向量；调用方负责保证它确实在过去
                ``add`` 过（否则减完 z_sum 偏离真实和）。

        Raises:
            ValueError: shape / dtype / ndarray 不符；category 不存在。
        """
        self._check_vector(fact_vector)
        state = self._states.get(category)
        if state is None:
            raise ValueError(f"category 不存在：{category!r}")

        z_removed = np.exp(1j * fact_vector)
        state._z_sum = state._z_sum - z_removed
        state.fact_count -= 1
        state.dirty_count += 1

        if state.fact_count <= 0:
            # 最后一条移除：清空 category（与 calibrate([]) 语义一致）
            self._states.pop(category)
            return

        state.bank_vector = np.mod(np.angle(state._z_sum), TWO_PI)
        state.snr_estimate = float(np.mean(np.abs(state._z_sum))) / state.fact_count

    # ─── 校准路径 ─────────────────────────────────────────────────────

    def calibrate(self, category: str, fact_vectors: Sequence[np.ndarray]) -> None:
        """从权威 fact 列表全量重建该 category 的 bank。

        语义上等价于"批量 add 一遍"（内部 z_sum 是真正的累加和，所以一致）。
        本方法存在的意义是：

            1. 重置 ``dirty_count = 0``、设置 ``last_calibrated_at``
            2. 修正长期增量过程中累积的浮点尾数漂移（z_sum 是大数和）
            3. 配合 ``remove`` 路径的全量重建

        空列表 → 该 category 的 state 被删除。

        Args:
            category: 分类标识。
            fact_vectors: 该 category 当前全部活跃 fact 的向量列表。

        Raises:
            ValueError: 任一 vector shape / dtype 不符。
        """
        if not fact_vectors:
            self._states.pop(category, None)
            return

        for v in fact_vectors:
            self._check_vector(v)

        z_sum = np.zeros(self.dim, dtype=np.complex128)
        for v in fact_vectors:
            z_sum += np.exp(1j * v)
        n = len(fact_vectors)
        coherence = float(np.mean(np.abs(z_sum))) / n

        self._states[category] = BankState(
            bank_vector=np.mod(np.angle(z_sum), TWO_PI),
            fact_count=n,
            dirty_count=0,
            last_calibrated_at=datetime.now(UTC),
            snr_estimate=coherence,
            _z_sum=z_sum,
        )

    # ─── 触发器查询 ───────────────────────────────────────────────────

    def needs_calibration(self, category: str) -> bool:
        """该 category 是否到达校准条件？

        OR 触发：
            - ``dirty_count >= max(dirty_floor, fact_count × dirty_pct)``
            - ``snr_estimate < snr_warn × 1.2``  （默认 snr_warn=0 即关闭）

        Returns:
            ``False`` 当 category 不存在或两条件都未达。
        """
        state = self._states.get(category)
        if state is None:
            return False
        threshold = max(self.dirty_floor, int(state.fact_count * self.dirty_pct))
        return state.dirty_count >= threshold or state.snr_estimate < self.snr_warn * 1.2

    # ─── 状态查询 ─────────────────────────────────────────────────────

    def get(self, category: str) -> BankState | None:
        """返回 category 的状态（实例引用，不复制）；不存在返回 None。"""
        return self._states.get(category)

    def categories(self) -> list[str]:
        """当前已存在的所有 category 名（任意顺序）。"""
        return list(self._states)

    def stats(self) -> dict[str, dict[str, object]]:
        """所有 category 的运维快照（不含 bank_vector / z_sum 本体）。"""
        return {
            cat: {
                "fact_count": s.fact_count,
                "dirty_count": s.dirty_count,
                "snr_estimate": s.snr_estimate,
                "needs_calibration": self.needs_calibration(cat),
                "last_calibrated_at": (
                    s.last_calibrated_at.isoformat() if s.last_calibrated_at is not None else None
                ),
            }
            for cat, s in self._states.items()
        }

    # ─── 内部 ──────────────────────────────────────────────────────────

    def _check_vector(self, v: np.ndarray) -> None:
        if not isinstance(v, np.ndarray):
            raise ValueError(f"vector 必须是 np.ndarray，实际：{type(v).__name__}")
        if v.shape != (self.dim,):
            raise ValueError(f"vector shape 必须是 ({self.dim},)，实际：{v.shape}")
        if v.dtype != np.float64:
            raise ValueError(f"vector dtype 必须是 float64，实际：{v.dtype}")


# ─── 50ms 去抖工具 ────────────────────────────────────────────────────────


class RebuildDebouncer:
    """合并连续 ``schedule()`` 为一次 ``callback`` 调用的去抖器。

    调用方在每次 ``bank.remove()`` 后调用 ``schedule()``；最后一次
    ``schedule()`` 后 ``window_seconds`` 触发一次 callback。
    内部用 ``threading.Timer``，线程安全（lock 保护 timer 引用）。

    Args:
        callback: 窗口结束后执行的零参回调。
        window_seconds: 去抖窗口，默认 0.050s。必须正数。

    Raises:
        ValueError: ``window_seconds`` 非正数。
    """

    def __init__(
        self,
        callback: Callable[[], None],
        *,
        window_seconds: float = 0.050,
    ) -> None:
        if not (window_seconds > 0):
            raise ValueError(f"window_seconds 必须是正数，实际：{window_seconds}")
        self._callback = callback
        self._window: Final[float] = window_seconds
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def schedule(self) -> None:
        """重置去抖定时器；callback 将在最后一次 schedule 后 ``window_seconds`` 触发。"""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            timer = threading.Timer(self._window, self._fire)
            timer.daemon = True
            self._timer = timer
            timer.start()

    def cancel(self) -> None:
        """取消待触发的 callback；已触发则无效。"""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def flush(self) -> None:
        """立即触发 callback 并清空 timer；无 pending 时是 noop。"""
        with self._lock:
            if self._timer is None:
                return
            self._timer.cancel()
            self._timer = None
        self._callback()

    @property
    def is_pending(self) -> bool:
        """当前是否有 timer 等待触发。"""
        with self._lock:
            return self._timer is not None

    def _fire(self) -> None:
        with self._lock:
            self._timer = None
        self._callback()
