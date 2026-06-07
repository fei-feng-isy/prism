"""``CloudEmbeddingBackend`` — 云端嵌入服务（OpenAI / voyageai）。

仅用于 query 路径；写路径始终走 ``LocalBgeBackend``。

* lazy httpx 客户端：首次 ``encode`` 才构造，``is_available()`` 无副作用
* ``is_available`` 只检查 ``httpx`` 可 import + env 中存在 API key，不发网络请求
* 输出始终 L2 归一化（即使 provider 已归一化，仍显式归一化兜底）
* 超时 / HTTP 错误统一抛 :class:`SemanticUnavailable`

支持的 provider：``"openai"``、``"voyageai"``。
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .backend import SemanticUnavailable

if TYPE_CHECKING:
    import httpx

log = logging.getLogger(__name__)

__all__ = [
    "OPENAI_ENDPOINT",
    "PROVIDER_ENDPOINTS",
    "VOYAGEAI_ENDPOINT",
    "CloudEmbeddingBackend",
]

OPENAI_ENDPOINT: Final[str] = "https://api.openai.com/v1/embeddings"
VOYAGEAI_ENDPOINT: Final[str] = "https://api.voyageai.com/v1/embeddings"

PROVIDER_ENDPOINTS: Final[dict[str, str]] = {
    "openai": OPENAI_ENDPOINT,
    "voyageai": VOYAGEAI_ENDPOINT,
}

# 模型 → 默认维度（用于 is_available 早期校验 + dim 没显式传时的兜底）
# 列表不求完整，未列出的 model 必须由配置显式给 dim
_DEFAULT_DIMS: Final[dict[str, int]] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-large-2": 1536,
}


class CloudEmbeddingBackend:
    """云端嵌入服务 backend（满足 :class:`SemanticBackend` 协议）。

    Args:
        provider: ``"openai"`` 或 ``"voyageai"``；其他值抛 :class:`ValueError`
        model: 模型名（OpenAI ``text-embedding-3-small``，voyage ``voyage-3`` 等）
        api_key_env: 读取 API key 的环境变量名（默认对应 provider 的常用 env）
        timeout_ms: 单次请求超时（毫秒），默认 3000
        dim: 期望输出维度；未传则按 ``_DEFAULT_DIMS`` 查 model；都没有时
            必须等首次 encode 拿到响应后才能知道
        name: 实例标识；默认 ``"{provider}:{model}"``
        endpoint: 显式覆盖 endpoint（测试 / 私有部署用）

    Raises:
        ValueError: provider 未知 / dim 既未传也无默认且 model 不在 _DEFAULT_DIMS
    """

    backend_name = "cloud_embedding"
    # 云端 backend 没有"加载"概念，连接由 httpx 在首次 encode 时建立。
    # is_loaded 永远 True 让上层不要把云端误判为 transient degraded。
    is_loaded = True

    def __init__(
        self,
        *,
        provider: str = "openai",
        model: str = "text-embedding-3-small",
        api_key_env: str = "OPENAI_API_KEY",
        timeout_ms: int = 3000,
        dim: int | None = None,
        name: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        if provider not in PROVIDER_ENDPOINTS and endpoint is None:
            raise ValueError(
                f"unknown provider {provider!r}; supported: "
                f"{sorted(PROVIDER_ENDPOINTS)} or pass explicit `endpoint=`"
            )
        if timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be > 0, got {timeout_ms}")

        if dim is None:
            dim = _DEFAULT_DIMS.get(model)
        if dim is None or dim <= 0:
            raise ValueError(
                f"无法确定 dim：model={model!r} 不在默认维度表 "
                f"{sorted(_DEFAULT_DIMS)}，请显式传 dim="
            )

        self.provider = provider
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_ms / 1000.0
        self.dim = int(dim)
        self.name = name if name is not None else f"{provider}:{model}"
        self.endpoint = endpoint or PROVIDER_ENDPOINTS[provider]

        self._client: httpx.Client | None = None
        self._client_lock = threading.Lock()
        # 测试注入点：MockTransport 走这里
        self._transport: httpx.BaseTransport | None = None

    # ─── 协议方法 ────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """``httpx`` 可 import 且 env 中存在 api key；**不发起网络请求**。

        - 缺包 / 缺 key 都返回 ``False``，但不抛
        - 配置错误（provider 未知）在 ``__init__`` 已抛 ValueError，不到这里
        """
        try:
            import importlib

            importlib.import_module("httpx")
        except ImportError:
            return False
        return bool(os.environ.get(self.api_key_env, "").strip())

    def encode(self, text: str) -> np.ndarray:
        """单条编码 → (dim,) L2 归一化 float32。"""
        out = self.encode_batch([text])
        return out[0]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """批量编码 → (N, dim) L2 归一化 float32。

        空列表返回 (0, dim) 不发请求，避免无谓 API 调用。
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        client = self._lazy_client()
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise SemanticUnavailable(
                f"env {self.api_key_env} 未设置；CloudEmbeddingBackend "
                f"({self.provider}:{self.model}) 不可用"
            )

        try:
            payload = self._build_payload(texts)
            response = client.post(
                self.endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout_seconds,
            )
        except Exception as e:
            # httpx.TimeoutException / ConnectError / TransportError 等
            raise SemanticUnavailable(
                f"{self.provider} embeddings 请求失败: {type(e).__name__}: {e}"
            ) from e

        if response.status_code >= 400:
            # 401 / 429 / 5xx 等
            body_excerpt = response.text[:300]
            raise SemanticUnavailable(
                f"{self.provider} embeddings HTTP {response.status_code}: "
                f"{body_excerpt}"
            )

        try:
            data = response.json()
        except Exception as e:
            raise SemanticUnavailable(
                f"{self.provider} 响应非 JSON: {response.text[:200]!r}"
            ) from e

        vectors = self._parse_response(data, expected_n=len(texts))
        # L2 归一化兜底（即使 provider 已归一）
        return _l2_normalize(vectors).astype(np.float32, copy=False)

    # ─── 内部 ────────────────────────────────────────────────────────────

    def _build_payload(self, texts: list[str]) -> dict[str, Any]:
        """provider-specific 请求体。"""
        body: dict[str, Any] = {"model": self.model, "input": texts}
        if self.provider == "openai":
            # 强制 float 编码（不要 base64）+ 显式指定维度（3-* 系列支持降维）
            body["encoding_format"] = "float"
            if self.model.startswith("text-embedding-3"):
                # 可降维但保持默认；为兼容 _DEFAULT_DIMS 不主动设置 dimensions
                pass
        elif self.provider == "voyageai":
            # voyage 推荐传 input_type 区分 query/document；不传时服务端走通用模式
            pass
        return body

    def _parse_response(self, data: dict[str, Any], *, expected_n: int) -> np.ndarray:
        """OpenAI / voyageai 响应都是 ``{"data": [{"embedding": [...], "index": k}]}``。"""
        rows = data.get("data")
        if not isinstance(rows, list) or len(rows) != expected_n:
            raise SemanticUnavailable(
                f"{self.provider} 响应 data 字段缺失或长度不符："
                f"期望 {expected_n} 条，实得 {len(rows) if isinstance(rows, list) else 'N/A'}"
            )
        # 按 index 排序还原顺序（API 保证有序但保险）
        try:
            rows_sorted = sorted(rows, key=lambda r: int(r.get("index", 0)))
        except Exception as e:
            raise SemanticUnavailable(
                f"{self.provider} data row 缺 index 字段: {e}"
            ) from e

        vecs: list[np.ndarray] = []
        for row in rows_sorted:
            emb = row.get("embedding")
            if not isinstance(emb, list):
                raise SemanticUnavailable(
                    f"{self.provider} embedding 字段缺失或非数组：{row!r}"
                )
            if len(emb) != self.dim:
                raise SemanticUnavailable(
                    f"{self.provider} 返回维度 {len(emb)} != 期望 {self.dim} "
                    f"（model={self.model}）"
                )
            vecs.append(np.asarray(emb, dtype=np.float32))
        return np.stack(vecs, axis=0)

    def _lazy_client(self) -> httpx.Client:
        """首次调用构造 :class:`httpx.Client`；后续复用（连接池）。"""
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                import httpx
            except ImportError as e:
                raise SemanticUnavailable(
                    f"httpx 未安装；CloudEmbeddingBackend 不可用: {e}"
                ) from e
            self._client = httpx.Client(transport=self._transport)
            return self._client

    def close(self) -> None:
        """关闭 httpx 连接池；幂等。"""
        with self._client_lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:  # pragma: no cover
                    pass
                self._client = None


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """按行 L2 归一化；零向量保留为零（不抛）。"""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    # 避免 0 除：把 0 范数置为 1，结果行仍为零向量
    safe = np.where(norms > 0, norms, 1.0)
    return mat / safe
