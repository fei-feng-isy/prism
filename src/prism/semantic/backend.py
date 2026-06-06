"""SemanticBackend 协议 + 降级路径桩。

* 定义 ``SemanticBackend`` 协议（运行时可 ``isinstance`` 检查）
* 提供 ``check_sentence_transformers_available()`` 与 ``DegradedSemanticBackend``：
  缺包时发出 WARNING 级日志，暴露不可用的 backend 占位，上层走降级权重
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_FTS_WEIGHT",
    "DEFAULT_JACCARD_WEIGHT",
    "DEFAULT_SEMANTIC_WEIGHT",
    "DEGRADED_FTS_WEIGHT",
    "DEGRADED_JACCARD_WEIGHT",
    "DEGRADED_SEMANTIC_WEIGHT",
    "DegradedSemanticBackend",
    "SemanticBackend",
    "SemanticUnavailable",
    "check_sentence_transformers_available",
    "default_weights",
    "degraded_weights",
    "warn_if_sentence_transformers_missing",
]


# ─── 三路融合权重 ─────────────────────────────────────────────────────────

# 正常路径
DEFAULT_SEMANTIC_WEIGHT: float = 0.55
DEFAULT_FTS_WEIGHT: float = 0.30
DEFAULT_JACCARD_WEIGHT: float = 0.15

# sentence-transformers 缺失时：把 semantic 权重重分配给 FTS + Jaccard
DEGRADED_SEMANTIC_WEIGHT: float = 0.0
DEGRADED_FTS_WEIGHT: float = 0.65
DEGRADED_JACCARD_WEIGHT: float = 0.35


def default_weights() -> tuple[float, float, float]:
    """正常路径下的（semantic, fts, jaccard）权重，和必须为 1。"""
    return (DEFAULT_SEMANTIC_WEIGHT, DEFAULT_FTS_WEIGHT, DEFAULT_JACCARD_WEIGHT)


def degraded_weights() -> tuple[float, float, float]:
    """降级路径下的（semantic, fts, jaccard）权重。semantic=0，其余重分配。"""
    return (DEGRADED_SEMANTIC_WEIGHT, DEGRADED_FTS_WEIGHT, DEGRADED_JACCARD_WEIGHT)


# ─── 异常 ──────────────────────────────────────────────────────────────────


class SemanticUnavailable(RuntimeError):
    """语义后端不可用（缺包 / 模型未加载 / 凭证缺失等）。

    DegradedSemanticBackend.encode 等会抛出此异常；上层捕获后应走降级路径
    （见 :func:`degraded_weights`），而不是把异常透出给调用方。
    """


# ─── 协议 ─────────────────────────────────────────────────────────────────


@runtime_checkable
class SemanticBackend(Protocol):
    """文本编码器协议；所有 backend（local_bge / cloud_embedding / hybrid_rerank）
    必须满足此契约。

    属性：
        name: 实例标识（写日志 / stats 用）。同一进程可有多个同 ``backend_name``
            的不同实例（比如不同 model_name 的两个 local_bge）。
        backend_name: 后端类型 — local_bge / cloud_embedding / hybrid_rerank / degraded。
        dim: 输出向量维度。降级实例为 0。
        is_loaded: 模型/客户端是否已加载就绪（Protocol 必需）。
            纯网络 backend（cloud_embedding）/ 装饰器（hybrid_rerank）按 wrapped
            语义透传；本地 backend（local_bge）由 ``_lazy_load`` 翻转。
            用于区分"永久不可用"与"暂态加载中"，让上层在 warmup 进行时走降级
            路径而不阻塞主线程触发模型加载。
    """

    name: str
    backend_name: str
    dim: int
    is_loaded: bool

    def encode(self, text: str) -> np.ndarray:
        """单条编码，返回 L2 归一化向量。"""
        ...

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """批量编码，返回 (N, dim) 的归一化矩阵。"""
        ...

    def is_available(self) -> bool:
        """依赖与凭证检查；**不发起网络请求**。"""
        ...


# ─── 缺包检测 ──────────────────────────────────────────────────────────────


def check_sentence_transformers_available() -> bool:
    """``sentence-transformers`` 是否可 import（不实例化模型、不下载权重）。

    用 ``importlib.util.find_spec`` 探查，避免执行重量级的传递依赖。
    已加载的 ``sys.modules`` 条目优先返回。
    """
    import sys

    sentinel = object()
    mod = sys.modules.get("sentence_transformers", sentinel)
    if mod is None:
        # 测试 / 上层显式标记为缺失
        return False
    if mod is not sentinel:
        # 已 import 或测试注入的替身模块
        return True
    try:
        import importlib.util

        return importlib.util.find_spec("sentence_transformers") is not None
    except (ImportError, ValueError):
        return False


def warn_if_sentence_transformers_missing(
    *,
    logger: logging.Logger | None = None,
) -> bool:
    """启动期检查：缺包则打 WARNING 级别日志。

    Args:
        logger: 自定义 logger，默认本模块的 ``log``。

    Returns:
        True 表示可用；False 表示已缺包并已记录 WARNING。
    """
    target = logger if logger is not None else log
    if check_sentence_transformers_available():
        return True
    target.warning(
        "sentence-transformers 未安装；语义编码降级到 regex+FTS+Jaccard 模式（"
        "semantic 权重 0，FTS+Jaccard 重分配为 %.2f/%.2f）。"
        "如需启用，请执行：pip install 'prism-memory[semantic]'",
        DEGRADED_FTS_WEIGHT,
        DEGRADED_JACCARD_WEIGHT,
    )
    return False


# ─── 降级 backend ──────────────────────────────────────────────────────────


class DegradedSemanticBackend:
    """语义不可用时的占位实例。

    - ``is_available()`` 永远返回 ``False``
    - ``is_loaded`` 永远返回 ``False``
    - ``encode`` / ``encode_batch`` 抛 :class:`SemanticUnavailable`
    - ``backend_name = "degraded"``，``dim = 0``
    """

    backend_name = "degraded"

    def __init__(self, *, name: str = "degraded-default") -> None:
        self.name = name
        self.dim = 0

    @property
    def is_loaded(self) -> bool:
        """降级 backend 永远返回 False——模型永不会被加载。"""
        return False

    def encode(self, text: str) -> np.ndarray:
        raise SemanticUnavailable(
            "sentence-transformers 未安装；DegradedSemanticBackend 不能编码。"
        )

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        raise SemanticUnavailable(
            "sentence-transformers 未安装；DegradedSemanticBackend 不能编码。"
        )

    def is_available(self) -> bool:
        return False