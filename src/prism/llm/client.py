"""OpenAI-compatible chat/completions 统一客户端。

线程安全（lazy httpx.Client + Lock），一个实例绑定一组
(model, endpoint, api_key_env, timeout)。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any, Final

log = logging.getLogger(__name__)

__all__ = ["LLMClient", "LLMClientError"]

_RESPONSE_TEXT_PREVIEW: Final[int] = 300


class LLMClientError(Exception):
    """LLM 请求级错误（env 缺失 / httpx 未装 / 网络 / HTTP 4xx-5xx / 响应格式异常）。"""


class LLMClient:
    """OpenAI-compatible ``/chat/completions`` 客户端。

    Args:
        model: 模型名（如 ``deepseek-v4-flash``）
        endpoint: 完整 URL，含 ``/chat/completions`` 后缀
        api_key_env: 读 API key 的环境变量名
        timeout_seconds: 单次请求超时（秒）
        transport: 可选 httpx.BaseTransport，测试注入用
    """

    def __init__(
        self,
        model: str,
        endpoint: str,
        api_key_env: str,
        *,
        timeout_seconds: float = 30.0,
        transport: Any = None,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self._transport = transport
        self._client: Any = None
        self._client_lock = threading.Lock()

    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """发起 chat/completions 调用，返回 ``choices[0].message.content``。

        Raises:
            LLMClientError: env 缺失 / httpx 未装 / 网络失败 / HTTP 错 / 响应异常
        """
        client = self._lazy_client()
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise LLMClientError(
                f"env {self.api_key_env} 未设置"
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        t0 = time.monotonic()
        try:
            resp = client.post(
                self.endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout_seconds,
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            log.warning(
                "LLM 请求失败 model=%s elapsed=%.0fms: %s: %s",
                self.model, elapsed_ms, type(e).__name__, e,
            )
            raise LLMClientError(
                f"LLM 请求失败: {type(e).__name__}: {e}"
            ) from e

        elapsed_ms = (time.monotonic() - t0) * 1000

        if resp.status_code >= 400:
            log.warning(
                "LLM HTTP 错误 model=%s status=%s elapsed=%.0fms: %s",
                self.model, resp.status_code, elapsed_ms,
                resp.text[:_RESPONSE_TEXT_PREVIEW],
            )
            raise LLMClientError(
                f"LLM HTTP {resp.status_code}: {resp.text[:_RESPONSE_TEXT_PREVIEW]}"
            )

        try:
            data = resp.json()
        except Exception as e:
            log.warning(
                "LLM 响应非 JSON model=%s elapsed=%.0fms: %s",
                self.model, elapsed_ms, resp.text[:_RESPONSE_TEXT_PREVIEW],
            )
            raise LLMClientError(
                f"LLM 响应非 JSON: {resp.text[:_RESPONSE_TEXT_PREVIEW]!r}"
            ) from e

        choices = data.get("choices") or []
        if not choices:
            log.warning(
                "LLM 响应缺 choices model=%s elapsed=%.0fms",
                self.model, elapsed_ms,
            )
            raise LLMClientError(f"LLM 响应缺 choices: {data!r}")

        content = (
            choices[0].get("message", {}).get("content")
            if isinstance(choices[0], dict)
            else None
        )
        if not isinstance(content, str):
            log.warning(
                "LLM 响应 content 非字符串 model=%s elapsed=%.0fms",
                self.model, elapsed_ms,
            )
            raise LLMClientError(
                f"LLM 响应 content 非字符串: {choices[0]!r}"
            )

        log.debug(
            "LLM 调用成功 model=%s elapsed=%.0fms content_len=%d",
            self.model, elapsed_ms, len(content),
        )
        return content

    def as_chat_fn(
        self,
        *,
        system: str,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> Callable[[str], str]:
        """返回 ``ChatFn`` 闭包：``prompt → content str``。

        固定 system prompt 和参数，只暴露 user message。
        供 :class:`~prism.entities.llm_extractor.LLMExtractor` 等注入。
        """

        def chat_fn(prompt: str) -> str:
            return self.chat(
                system=system,
                user=prompt,
                temperature=temperature,
                response_format=response_format,
            )

        return chat_fn

    def close(self) -> None:
        """关闭 httpx 连接池；幂等。"""
        with self._client_lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    def _lazy_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                import httpx
            except ImportError as e:
                raise LLMClientError(
                    f"httpx 未安装: {e}（pip install httpx）"
                ) from e
            if self._transport is not None:
                self._client = httpx.Client(transport=self._transport)
            else:
                self._client = httpx.Client()
            return self._client
