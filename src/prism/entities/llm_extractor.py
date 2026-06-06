"""Stage 2 异步 LLM 实体抽取。

提供 ``LLMExtractor`` — 符合 ``ExtractorFn`` 协议的可调用对象。调用方注入
``chat_fn: Callable[[str], str]``（prompt -> text response），LLM 后端选型
完全解耦。

异常语义：``chat_fn`` 抛异常则上抛（worker 重试）；JSON 解析失败返回 ``[]``
（确定性错误重试无意义）。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Final

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PROMPT_TEMPLATE",
    "ChatFn",
    "LLMExtractor",
    "parse_llm_entities",
]


ChatFn = Callable[[str], str]
"""签名：(prompt: str) -> response_text: str。可抛 (网络 / 限流 / timeout / API 错)。"""


DEFAULT_PROMPT_TEMPLATE: Final[str] = (
    "从下面这段记忆中提取关键实体（人名、地点、产品、版本、概念），用 JSON 数组返回。\n"
    "\n"
    "约束：\n"
    "- 中文实体按词语切分，如「老王」是一个实体而不是「老」和「王」两个字符\n"
    "- 不提取通用词汇（如「用户」「系统」「项目」「今天」「我们」）\n"
    "- 版本号与产品名应保持完整，如「PostgreSQL 14」是一个实体\n"
    "- 实体名 strip 前后空白，长度 ≥ 2 字符\n"
    "\n"
    "文本：{content}\n"
    "\n"
    "只返回 JSON，不要其他解释。格式：{{\"entities\": [\"...\", \"...\"]}}"
)


# 容忍 ```json ... ``` / ``` ... ``` markdown 包裹
_FENCED_BLOCK_RE: Final[re.Pattern[str]] = re.compile(
    r"```(?:json)?\s*(.+?)\s*```",
    re.DOTALL | re.IGNORECASE,
)

# 直接寻找第一个 JSON object（贪到末尾的最后 `}`）
_JSON_OBJECT_RE: Final[re.Pattern[str]] = re.compile(
    r"\{.*\}",
    re.DOTALL,
)


class LLMExtractor:
    """LLM 实体抽取器（Stage 2）；调用方注入 ``chat_fn``。

    Args:
        chat_fn: prompt -> text response 的可调用对象
        prompt_template: 必须含 ``{content}``，默认 ``DEFAULT_PROMPT_TEMPLATE``
        max_entities: 单次抽取实体数上限
        content_max_chars: prompt 中 content 的截断阈值

    Raises:
        ValueError: prompt_template 缺少占位符 / 数值参数 <= 0
    """

    DEFAULT_MAX_ENTITIES = 10
    DEFAULT_CONTENT_MAX_CHARS = 2000

    def __init__(
        self,
        chat_fn: ChatFn,
        *,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        max_entities: int = DEFAULT_MAX_ENTITIES,
        content_max_chars: int = DEFAULT_CONTENT_MAX_CHARS,
    ) -> None:
        if "{content}" not in prompt_template:
            raise ValueError(
                "prompt_template must contain '{content}' placeholder"
            )
        if max_entities <= 0:
            raise ValueError(
                f"max_entities must be > 0, got {max_entities}"
            )
        if content_max_chars <= 0:
            raise ValueError(
                f"content_max_chars must be > 0, got {content_max_chars}"
            )
        self._chat_fn = chat_fn
        self._prompt_template = prompt_template
        self._max_entities = max_entities
        self._content_max_chars = content_max_chars

    def __call__(self, content: str) -> list[str]:
        """符合 `ExtractorFn` 协议。失败语义见模块 docstring。"""
        if not content or not content.strip():
            return []
        # 截断超长内容（rare path；防 prompt 爆 token）
        truncated = content[: self._content_max_chars]
        prompt = self._prompt_template.format(content=truncated)

        # chat_fn 抛异常 → 上抛（worker mark_failed 触发重试）
        raw = self._chat_fn(prompt)

        # parse 失败 → fallback []（确定性错误重试无意义）
        entities = parse_llm_entities(raw)
        return entities[: self._max_entities]


def parse_llm_entities(raw: str) -> list[str]:
    """从 LLM 原文本解析 entities list。健壮性目标：

    - 容忍 ```json ... ``` / ``` ... ``` markdown 包裹
    - 容忍前后散文（"以下是..." / "Result:" 等），抓第一个 `{...}` 块
    - schema 不符（缺 entities key / entities 非 list / 元素非 string）→ []
    - 元素去重（保序）、strip、过滤空白与 < 2 字符
    - JSON 解析失败 → log.warning + []
    """
    if not isinstance(raw, str) or not raw.strip():
        return []

    text = raw.strip()

    # 1) 先剥 ```json ... ``` 围栏
    m = _FENCED_BLOCK_RE.search(text)
    if m is not None:
        text = m.group(1).strip()

    # 2) 抓第一个 JSON object（最外层 {}）— 容忍 LLM 在前后加散文
    if not text.startswith("{"):
        m2 = _JSON_OBJECT_RE.search(text)
        if m2 is None:
            log.warning("LLM 输出无 JSON object：%r", _truncate(raw))
            return []
        text = m2.group(0)

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning(
            "LLM 输出 JSON 解析失败 (%s)：%r", exc, _truncate(raw)
        )
        return []

    if not isinstance(parsed, dict):
        log.warning("LLM 输出非 object：%r", _truncate(raw))
        return []

    raw_entities = parsed.get("entities")
    if not isinstance(raw_entities, list):
        log.warning(
            "LLM 输出缺少 entities list (got %s)：%r",
            type(raw_entities).__name__,
            _truncate(raw),
        )
        return []

    seen: set[str] = set()
    out: list[str] = []
    for item in raw_entities:
        if not isinstance(item, str):
            continue
        name = item.strip()
        if len(name) < 2:
            continue
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _truncate(s: str, n: int = 200) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "..."
