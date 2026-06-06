"""``HybridRerankBackend`` — 本地召回 + LLM rerank 装饰器。

套在已有 ``SemanticBackend``（通常是 ``LocalBgeBackend``）外面的重排装饰器：

* ``encode`` / ``encode_batch`` 透传给 wrapped backend，嵌入空间不变
* ``rerank(query, candidates, *, final_k)`` 向 LLM 发问后按相关性重排
* LLM 失败时按 ``fallback_on_error`` 决定：``"local"`` 返回本地排序（默认），
  ``"error"`` 抛 :class:`SemanticUnavailable`
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from ..llm import LLMClient, LLMClientError
from .backend import SemanticUnavailable

if TYPE_CHECKING:
    from .backend import SemanticBackend

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_RERANK_ENDPOINT",
    "RERANK_SYSTEM_PROMPT",
    "HybridRerankBackend",
    "RerankCandidate",
    "RerankResult",
    "build_rerank_user_prompt",
    "parse_rerank_response",
]


DEFAULT_RERANK_ENDPOINT: Final[str] = "https://api.deepseek.com/v1/chat/completions"


RERANK_SYSTEM_PROMPT: Final[str] = (
    "你是一个相关性重排器。给定一个查询和若干候选事实，"
    "请按照与查询的相关性从高到低排序，仅返回相关候选的编号。"
    "只输出严格的 JSON 对象，格式：{\"ranked\": [int, int, ...]}。"
    "ranked 数组中的整数必须是候选编号，不要解释、不要额外文字。"
)


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    """rerank 输入候选；fact_id 是 SQLite 行键，content 是事实文本。"""

    fact_id: int
    content: str
    base_score: float  # 本地融合分数，fallback 时用于排序


@dataclass(frozen=True, slots=True)
class RerankResult:
    """rerank 单条结果；rerank_score 为 None 表示走 fallback。"""

    fact_id: int
    rerank_score: float | None
    source: str  # "llm" | "fallback_local"


class HybridRerankBackend:
    """本地嵌入 + LLM rerank 装饰器。

    Args:
        wrapped: 被装饰的 :class:`SemanticBackend`（通常是 ``LocalBgeBackend``）
        llm_model: LLM 模型名（默认 ``deepseek-v4-flash``）
        endpoint: LLM 端点；默认 ``DEFAULT_RERANK_ENDPOINT``（deepseek）
        api_key_env: 读 API key 的 env 名
        timeout_ms: 单次 LLM 调用超时
        candidate_n: 从本地召回的候选数（pipeline 取多少送给 rerank）
        final_k: rerank 后输出的最终条数
        fallback_on_error: ``"local"`` → 失败时按本地分数返回前 final_k；
            ``"error"`` → 抛 :class:`SemanticUnavailable`
        name: 实例标识

    Raises:
        ValueError: candidate_n < final_k / timeout 非正
    """

    backend_name = "hybrid_rerank"

    def __init__(
        self,
        wrapped: SemanticBackend,
        *,
        llm_model: str = "deepseek-v4-flash",
        endpoint: str = DEFAULT_RERANK_ENDPOINT,
        api_key_env: str = "DEEPSEEK_API_KEY",
        timeout_ms: int = 2000,
        candidate_n: int = 50,
        final_k: int = 5,
        fallback_on_error: str = "local",
        name: str | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        if candidate_n < final_k:
            raise ValueError(
                f"candidate_n ({candidate_n}) 必须 >= final_k ({final_k})"
            )
        if timeout_ms <= 0:
            raise ValueError(f"timeout_ms 必须 > 0，得到 {timeout_ms}")
        if fallback_on_error not in ("local", "error"):
            raise ValueError(
                f"fallback_on_error 必须是 'local' 或 'error'，得到 {fallback_on_error!r}"
            )

        self.wrapped = wrapped
        self.llm_model = llm_model
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_ms / 1000.0
        self.candidate_n = int(candidate_n)
        self.final_k = int(final_k)
        self.fallback_on_error = fallback_on_error
        self.dim = wrapped.dim
        self.name = (
            name if name is not None else f"hybrid_rerank({wrapped.name},{llm_model})"
        )

        if llm_client is not None:
            self._llm_client = llm_client
            self._owns_client = False
        else:
            self._llm_client = LLMClient(
                model=llm_model,
                endpoint=endpoint,
                api_key_env=api_key_env,
                timeout_seconds=self.timeout_seconds,
            )
            self._owns_client = True

    # ─── SemanticBackend Protocol：透传给 wrapped ────────────────────────

    def is_available(self) -> bool:
        """wrapped 可用即装饰器可用；rerank LLM 不在此检查（LLM 失败按
        ``fallback_on_error`` 处理，不影响 *是否能跑*）。"""
        return self.wrapped.is_available()

    @property
    def is_loaded(self) -> bool:
        """透传 wrapped backend 的 is_loaded。"""
        return bool(getattr(self.wrapped, "is_loaded", True))

    def encode(self, text: str) -> np.ndarray:
        return self.wrapped.encode(text)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return self.wrapped.encode_batch(texts)

    # ─── 新增 rerank 接口 ────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        final_k: int | None = None,
    ) -> list[RerankResult]:
        """LLM 重排。

        Args:
            query: 用户查询文本
            candidates: 本地召回的候选（已按 base_score 排序）
            final_k: 覆盖 self.final_k（按召回端临时调整）；不传则用默认

        Returns:
            前 final_k 条 :class:`RerankResult`。``source="llm"`` 表示 LLM 排序，
            ``source="fallback_local"`` 表示走了本地降级。
        """
        k = final_k if final_k is not None else self.final_k
        if not candidates:
            return []

        # 截断到 candidate_n（保护 LLM prompt 大小）
        truncated = candidates[: self.candidate_n]

        # final_k 必须 <= len(truncated)，否则只能返回所有
        effective_k = min(k, len(truncated))

        try:
            ranked_ids = self._call_llm(query, truncated)
        except SemanticUnavailable:
            if self.fallback_on_error == "error":
                raise
            log.warning(
                "rerank LLM 不可用，按本地分数 fallback：query=%r candidates=%d",
                query[:50], len(truncated),
            )
            return self._fallback_local(truncated, effective_k)
        except Exception as e:
            # 任何意料外异常都归一到 fallback / 抛
            if self.fallback_on_error == "error":
                raise SemanticUnavailable(f"rerank 未知错误: {e}") from e
            log.warning("rerank 未知错误，按本地 fallback：%s", e)
            return self._fallback_local(truncated, effective_k)

        # 用 LLM 排序结果重组（保留出现在 ranked_ids 中的候选，按顺序）
        by_id: dict[int, RerankCandidate] = {c.fact_id: c for c in truncated}
        results: list[RerankResult] = []
        seen: set[int] = set()
        # rerank_score = 1.0 / rank（rank=1..n），保留 ordering 信号
        for rank, fid in enumerate(ranked_ids, start=1):
            if fid in by_id and fid not in seen:
                seen.add(fid)
                results.append(
                    RerankResult(
                        fact_id=fid,
                        rerank_score=1.0 / rank,
                        source="llm",
                    )
                )
                if len(results) >= effective_k:
                    break

        # LLM 可能漏掉部分候选；用本地分数补齐到 effective_k
        if len(results) < effective_k:
            for c in truncated:
                if c.fact_id in seen:
                    continue
                results.append(
                    RerankResult(
                        fact_id=c.fact_id,
                        rerank_score=None,  # 未被 LLM 评分
                        source="fallback_local",
                    )
                )
                seen.add(c.fact_id)
                if len(results) >= effective_k:
                    break

        return results

    # ─── 内部 ────────────────────────────────────────────────────────────

    def _call_llm(
        self, query: str, candidates: list[RerankCandidate]
    ) -> list[int]:
        """发起 LLM chat/completions 调用，返回排序后的 fact_id 列表。

        Raises:
            SemanticUnavailable: 缺 env / 请求失败 / 非 JSON / 越界
        """
        prompt = build_rerank_user_prompt(query, candidates)
        try:
            raw = self._llm_client.chat(
                system=RERANK_SYSTEM_PROMPT,
                user=prompt,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except LLMClientError as e:
            raise SemanticUnavailable(str(e)) from e

        return _parse_rerank_content(raw, valid_ids={c.fact_id for c in candidates})

    def _fallback_local(
        self, candidates: list[RerankCandidate], k: int
    ) -> list[RerankResult]:
        """LLM 失败时按 base_score 排序返回前 k 条。"""
        ordered = sorted(candidates, key=lambda c: c.base_score, reverse=True)[:k]
        return [
            RerankResult(
                fact_id=c.fact_id,
                rerank_score=None,
                source="fallback_local",
            )
            for c in ordered
        ]

    def close(self) -> None:
        """关闭 LLM 客户端连接池；幂等。"""
        if self._owns_client:
            self._llm_client.close()


# ─── prompt + response helpers（独立函数便于单测） ──────────────────────────


def build_rerank_user_prompt(
    query: str, candidates: list[RerankCandidate]
) -> str:
    """构造 LLM user message：候选用 fact_id 编号呈现。"""
    lines = [f"查询：{query}", "", "候选事实："]
    for c in candidates:
        # 截断单条 content 防 prompt 爆（中文一般 < 200 字够）
        content = c.content if len(c.content) <= 200 else c.content[:200] + "…"
        lines.append(f"[{c.fact_id}] {content}")
    lines.append("")
    lines.append(
        '按相关性从高到低返回 fact_id（最多 ' f'{len(candidates)} 个），'
        '格式：{"ranked": [int, int, ...]}'
    )
    return "\n".join(lines)


# 兼容 LLM 偶尔返回 ```json ... ``` 包裹的代码块
_CODE_FENCE_RE: Final[re.Pattern[str]] = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL
)


def _parse_rerank_content(
    content: str, *, valid_ids: set[int]
) -> list[int]:
    """从 LLM 返回的 content 字符串解析 ``ranked`` 数组，过滤无效 ID。

    Args:
        content: LLM message content（JSON 字符串或 code-fence 包裹）
        valid_ids: 候选合法 fact_id 集合

    Returns:
        合法 fact_id 列表，保留 LLM 排序

    Raises:
        SemanticUnavailable: 非 JSON / ranked 不是数组
    """
    parsed: Any
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        fence_match = _CODE_FENCE_RE.search(content)
        if fence_match is None:
            raise SemanticUnavailable(
                f"rerank LLM content 非 JSON: {content[:200]!r}"
            ) from None
        try:
            parsed = json.loads(fence_match.group(1))
        except json.JSONDecodeError as e:
            raise SemanticUnavailable(
                f"rerank LLM JSON 代码块解析失败: {e}"
            ) from e

    if not isinstance(parsed, dict):
        raise SemanticUnavailable(
            f"rerank LLM JSON 非 object: {type(parsed).__name__}"
        )
    raw_ranked = parsed.get("ranked")
    if not isinstance(raw_ranked, list):
        raise SemanticUnavailable(
            f"rerank LLM 缺 ranked 数组: {parsed!r}"
        )

    out: list[int] = []
    seen: set[int] = set()
    for item in raw_ranked:
        try:
            fid = int(item)
        except (TypeError, ValueError):
            continue
        if fid in valid_ids and fid not in seen:
            out.append(fid)
            seen.add(fid)
    return out


def parse_rerank_response(
    data: dict[str, Any], *, valid_ids: set[int]
) -> list[int]:
    """从 chat/completions 完整响应里抽 ``ranked`` 数组，过滤无效 ID。

    Args:
        data: LLM 响应 dict（chat/completions 格式）
        valid_ids: 候选合法 fact_id 集合

    Returns:
        合法 fact_id 列表，保留 LLM 排序

    Raises:
        SemanticUnavailable: 响应结构损坏 / 无法解析 / ranked 不是数组
    """
    try:
        choices = data["choices"]
        message = choices[0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise SemanticUnavailable(
            f"rerank LLM 响应结构异常（缺 choices/message/content）: {e}"
        ) from e

    if not isinstance(content, str):
        raise SemanticUnavailable(
            f"rerank LLM content 非字符串: {type(content).__name__}"
        )

    return _parse_rerank_content(content, valid_ids=valid_ids)
