"""``LocalBgeBackend`` — 本地 ``BAAI/bge-small-zh-v1.5`` 嵌入实现。

512 维 float32 句向量，输出 L2 归一化（cos 相似度 = 内积）。

* 惰性加载：首次 ``encode`` 时才加载模型，``is_available()`` 不触发加载
* 线程安全：``_lazy_load`` 用 ``threading.Lock`` 保护
* 离线优先：cache 命中时传 ``local_files_only=True``，跳过 HF HEAD 探测
* Cache 自动修复：cache 存在但加载失败时，尝试一次 ``force_download`` + retry
* 镜像策略：首次下载按 ``hf_endpoint_strategy`` 切换
  (``respect_env`` / ``mirror_first`` / ``mirror_only``)
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .backend import SemanticUnavailable, check_sentence_transformers_available

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

__all__ = ["BGE_SMALL_ZH_DIM", "DEFAULT_BGE_MODEL", "LocalBgeBackend"]


DEFAULT_BGE_MODEL: Final[str] = "BAAI/bge-small-zh-v1.5"
BGE_SMALL_ZH_DIM: Final[int] = 512
DEFAULT_HF_MIRROR_URL: Final[str] = "https://hf-mirror.com"


class LocalBgeBackend:
    """本地 sentence-transformers backend（满足 :class:`SemanticBackend` 协议）。

    Args:
        model_name: HuggingFace 模型名或本地路径，默认 ``BAAI/bge-small-zh-v1.5``
        dim: 期望输出维度（默认 512，bge-small-zh-v1.5 固定）。
            实际加载后会断言与模型 dim 一致；不一致抛 :class:`ValueError`
        name: 实例标识（写日志 / stats 用），默认与 ``model_name`` 一致
        device: 推理设备（``'cpu'`` / ``'cuda'`` / ``'mps'`` / ``None`` 自动）

    Raises:
        ValueError: ``dim`` 与模型实际维度不一致（在首次加载完成后才能检测）
        SemanticUnavailable: ``sentence_transformers`` 未安装且调用了 encode

    Example:
        >>> backend = LocalBgeBackend()
        >>> backend.is_available()  # 不触发模型加载
        True
        >>> vec = backend.encode("用户的母语是中文")  # 首次：触发加载
        >>> vec.shape
        (512,)
        >>> np.linalg.norm(vec)  # L2 归一化
        1.0
    """

    backend_name = "local_bge"

    def __init__(
        self,
        model_name: str = DEFAULT_BGE_MODEL,
        *,
        dim: int = BGE_SMALL_ZH_DIM,
        name: str | None = None,
        device: str | None = None,
        auto_repair_cache: bool = True,
        hf_endpoint_strategy: str = "respect_env",
        hf_mirror_url: str = DEFAULT_HF_MIRROR_URL,
    ) -> None:
        self.model_name = model_name
        self.dim = dim
        self.name = name if name is not None else model_name
        self._device = device
        self._auto_repair_cache = auto_repair_cache
        self._hf_endpoint_strategy = hf_endpoint_strategy
        self._hf_mirror_url = hf_mirror_url
        self._model: SentenceTransformer | None = None
        self._load_lock = threading.Lock()

    # ─── 协议方法 ────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """``sentence_transformers`` 是否可 import；**不触发模型加载、不走网络**。"""
        return check_sentence_transformers_available()

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载到内存（``self._model is not None``）。"""
        return self._model is not None

    def encode(self, text: str) -> np.ndarray:
        """单条编码 → ``(dim,)`` L2 归一化 float32 向量。"""
        model = self._lazy_load()
        vec = model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # sentence-transformers 单条返回 (dim,) ndarray；保险起见 squeeze
        arr = np.asarray(vec, dtype=np.float32).reshape(self.dim)
        return arr

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """批量编码 → ``(N, dim)`` L2 归一化 float32 矩阵。

        空列表返回形状 ``(0, dim)`` 的空数组（不抛），方便调用方与单条路径统一处理。
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self._lazy_load()
        mat = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(mat, dtype=np.float32).reshape(len(texts), self.dim)

    # ─── 内部 ────────────────────────────────────────────────────────────

    def _lazy_load(self) -> SentenceTransformer:
        """首次调用加载模型；后续直接返回缓存实例。

        Raises:
            SemanticUnavailable: ``sentence_transformers`` 未安装或模型加载失败
                （网络断、HuggingFace 限流、磁盘满等都归入此异常）
            ValueError: 加载成功但模型维度与 ``self.dim`` 不一致
        """
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:  # 双检：另一线程已加载
                return self._model
            if not check_sentence_transformers_available():
                raise SemanticUnavailable("sentence-transformers 未安装；无法加载 LocalBgeBackend")
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise SemanticUnavailable(f"sentence-transformers import 失败: {e}") from e
            cache_exists = _has_hf_cache(self.model_name)
            load_kwargs: dict[str, Any] = {"device": self._device}
            if cache_exists:
                # 本地 cache 命中 → 跳过 HF HEAD 探测，避免代理 / 内网环境下挂死
                load_kwargs["local_files_only"] = True
            try:
                if cache_exists:
                    # offline 路径：不接入 endpoint 策略，信任本地副本
                    model = SentenceTransformer(self.model_name, **load_kwargs)
                else:
                    # 首次下载：应用 endpoint 策略（mirror_first / mirror_only / respect_env）
                    with _hf_endpoint_context(
                        self._hf_endpoint_strategy, self._hf_mirror_url
                    ):
                        model = SentenceTransformer(self.model_name, **load_kwargs)
            except Exception as first_err:
                # 首次下载失败 + mirror_first → 回退官方 endpoint 重试一次
                if not cache_exists and self._hf_endpoint_strategy == "mirror_first":
                    log.warning(
                        "LocalBgeBackend 镜像 (%s) 首次下载失败，回退官方 endpoint。"
                        "原始异常：%s",
                        self._hf_mirror_url,
                        first_err,
                    )
                    try:
                        # 不带 context → 走用户 env / 默认官方 endpoint
                        model = SentenceTransformer(self.model_name, **load_kwargs)
                    except Exception as fallback_err:
                        raise SemanticUnavailable(
                            f"LocalBgeBackend 镜像与官方端点均下载失败 "
                            f"(model={self.model_name!r}): "
                            f"mirror={first_err}; official={fallback_err}"
                        ) from fallback_err
                    log.info(
                        "LocalBgeBackend 官方 endpoint 回退下载成功 (model=%r)",
                        self.model_name,
                    )
                else:
                    # 仅当 cache 已存在 → 判定为损坏，尝试一次 force_download + retry。
                    # cache 不存在 + 非 mirror_first → 不在自愈范围（盲目 force_download 只会
                    # 再次走网络又挂一遍）。
                    if not (self._auto_repair_cache and cache_exists):
                        raise SemanticUnavailable(
                            f"LocalBgeBackend 模型加载失败 "
                            f"(model={self.model_name!r}): {first_err}"
                        ) from first_err
                    log.warning(
                        "LocalBgeBackend 检测到 HF cache 加载失败 (model=%r)，"
                        "尝试自动修复：snapshot_download(force_download=True)。原始异常：%s",
                        self.model_name,
                        first_err,
                    )
                    try:
                        _force_redownload(self.model_name)
                    except Exception as repair_err:
                        raise SemanticUnavailable(
                            f"LocalBgeBackend 模型加载失败 (model={self.model_name!r}): "
                            f"{first_err}；自动修复亦失败：{repair_err}"
                        ) from first_err
                    try:
                        # cache 已刷新 → 仍走 local_files_only=True 避免再次 HEAD
                        model = SentenceTransformer(
                            self.model_name, device=self._device, local_files_only=True
                        )
                    except Exception as retry_err:
                        raise SemanticUnavailable(
                            f"LocalBgeBackend 自动修复后重试仍失败 "
                            f"(model={self.model_name!r}): {retry_err}"
                        ) from retry_err
                    log.info(
                        "LocalBgeBackend HF cache 自动修复成功 (model=%r)", self.model_name
                    )
            # 校验维度（一旦模型加载成功，dim 应与配置一致）
            actual_dim = _model_dim(model)
            if actual_dim != self.dim:
                raise ValueError(
                    f"模型 {self.model_name!r} 实际维度 {actual_dim} 与配置 dim={self.dim} 不一致"
                )
            log.info(
                "LocalBgeBackend 加载完成: model=%s dim=%d device=%s",
                self.model_name,
                actual_dim,
                self._device or "auto",
            )
            self._model = model
            return model


def _model_dim(model: SentenceTransformer) -> int:
    """兼容 sentence-transformers v2/v3/v5 的 dim 获取（API 多次重命名）。"""
    # v5+ 使用 get_embedding_dimension；v2-v4 使用 get_sentence_embedding_dimension
    for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        fn = getattr(model, attr, None)
        if callable(fn):
            return int(fn())
    raise RuntimeError(
        "无法从 SentenceTransformer 实例获取 dim "
        "（既无 get_embedding_dimension 也无 get_sentence_embedding_dimension）"
    )


def _has_hf_cache(model_name: str) -> bool:
    """该 repo 是否在 HuggingFace cache 中存在快照（用 ``config.json`` 探针）。

    本地路径直接返回 False（用户手工管理的模型目录不走 cache 修复逻辑）。
    """
    if Path(model_name).is_dir():
        return False
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    cached = try_to_load_from_cache(repo_id=model_name, filename="config.json")
    return isinstance(cached, str)


def _force_redownload(model_name: str) -> None:
    """对 HF repo 触发一次 ``snapshot_download(force_download=True)``。异常透出。"""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=model_name, force_download=True)


@contextmanager
def _hf_endpoint_context(strategy: str, mirror_url: str) -> Iterator[None]:
    """临时 set ``HF_ENDPOINT`` 走镜像；用户已显式 set 时尊重用户不覆盖。

    * ``respect_env``: no-op
    * ``mirror_first`` / ``mirror_only``: 仅当用户未设 ``HF_ENDPOINT`` 时 set 镜像 URL
    """
    if strategy == "respect_env" or os.environ.get("HF_ENDPOINT"):
        yield
        return
    os.environ["HF_ENDPOINT"] = mirror_url
    try:
        yield
    finally:
        os.environ.pop("HF_ENDPOINT", None)
