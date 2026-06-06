"""实体抽取 Stage 1 — 同步 regex + jieba 词性标注，写入路径上调用。

jieba 可选：未安装时降级为 regex-only 模式。
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Final

log = logging.getLogger(__name__)

__all__ = [
    "ExtractedEntity",
    "extract_entities",
    "jieba_available",
    "preload_jieba",
]


# ─── 模型 ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, order=True)
class ExtractedEntity:
    """抽取出的单一实体。``name`` 已 strip；按 ``name`` 排序便于测试稳定。"""

    name: str
    entity_type: str  # person / org / concept / product / version / identifier / unknown
    method: str       # regex / jieba


# ─── jieba 预加载 ──────────────────────────────────────────────────────────

# 词典加载一次性、线程安全；不持有 jieba 模块引用（按需 import）
_PRELOAD_LOCK = threading.Lock()
_PRELOADED: bool = False
_JIEBA_AVAILABLE: bool = False


def jieba_available() -> bool:
    """jieba 是否成功 import + 预加载（``preload_jieba()`` 后才有意义）。"""
    return _JIEBA_AVAILABLE


def preload_jieba(*, warmup_text: str = "预热") -> bool:
    """一次性预加载 jieba 词典（线程安全 + 幂等）。

    应在 provider 启动期调用，避免首次 add_fact 时的词典加载延迟。

    Returns:
        True if jieba 已可用；False if 未安装（不抛异常，调用方可降级）。
    """
    global _PRELOADED, _JIEBA_AVAILABLE
    if _PRELOADED:
        return _JIEBA_AVAILABLE
    with _PRELOAD_LOCK:
        if _PRELOADED:
            return _JIEBA_AVAILABLE
        try:
            import jieba
            import jieba.posseg as pseg
            jieba.setLogLevel(60)  # 抑制 "Building prefix dict" 之类的提示
            # 真正触发词典加载
            list(pseg.cut(warmup_text))
            _JIEBA_AVAILABLE = True
        except ImportError:
            log.warning("jieba 未安装；实体抽取降级到 regex-only 模式")
            _JIEBA_AVAILABLE = False
        except Exception as e:
            log.warning("jieba 预加载失败 (%s)；降级到 regex-only 模式", e)
            _JIEBA_AVAILABLE = False
        _PRELOADED = True
        return _JIEBA_AVAILABLE


# ─── 正则规则 ──────────────────────────────────────────────────────────────

# 引号内容（中英文引号）
_QUOTED_RE: Final[re.Pattern[str]] = re.compile(
    r'["“”‘’「『](.+?)["“”‘’」』]'
)

# 英文大写多词：Steve Jobs / Project Lighthouse
_CAPS_MULTIWORD_RE: Final[re.Pattern[str]] = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\b"
)

# 别名标记之后的连续非空白片段
_ALIAS_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:aka|又称|也叫|别名|又名)[\s:：]+([^\s,，。；;]+)",
    re.IGNORECASE,
)

# 标识符：含连字符 / 下划线 的 ASCII 名（vim-mode / kebab_case_123）
_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(
    r"\b[A-Za-z][\w]*[-_][\w-]+\b"
)

# CamelCase 多段；同时支持以全大写缩写收尾（OpenAPI / ParseHTML）
_CAMEL_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:"
    r"[A-Z][a-z]+(?:[A-Z][a-z]+)+"      # ProjectPhoenix
    r"|"
    r"(?:[A-Z][a-z]+)+[A-Z]{2,}"        # OpenAPI / ParseHTML
    r")\b"
)

# 产品 + 版本号：PostgreSQL 14 / Python 3.12.1 / SQLite 3.46
_VERSIONED_RE: Final[re.Pattern[str]] = re.compile(
    r"\b([A-Za-z][A-Za-z0-9]*)\s+(\d+(?:\.\d+)*)\b"
)


# 常见低信号词（中英），过滤掉
_STOPWORDS: Final[frozenset[str]] = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "are", "was",
    "用户", "系统", "项目", "今天", "明天", "昨天", "我们", "他们", "你们",
    "什么", "怎么", "为什么", "可以", "应该", "需要",
})


# jieba flag → entity_type
_JIEBA_FLAG_TO_TYPE: Final[dict[str, str]] = {
    "nr": "person",     # 人名
    "nrt": "person",    # 翻译人名（jieba 也常用于专有名词）
    "nrfg": "person",   # 其他专名
    "nt": "org",        # 机构
    "nz": "concept",    # 其他专有
}


# ─── 主入口 ───────────────────────────────────────────────────────────────


def extract_entities(text: str) -> list[ExtractedEntity]:
    """同步抽取实体；jieba 可用时叠加词性标注。

    返回按 ``(name)`` 升序、去重的列表（同名优先保留更具体的 type）。

    Args:
        text: 原始事实内容

    Note:
        函数本身不调用 :func:`preload_jieba`；调用方应在 provider 启动期
        预加载一次。未预加载时也不报错，仅退化为 regex-only。
    """
    if not text or not text.strip():
        return []

    # 用 dict 做"按 name 去重 + 保留最优 type"
    # 优先级：person > org > product > version > concept > identifier > unknown
    found: dict[str, ExtractedEntity] = {}

    def _push(name: str, etype: str, method: str) -> None:
        n = name.strip()
        if len(n) < 2 or n.lower() in _STOPWORDS:
            return
        prev = found.get(n)
        if prev is None or _TYPE_RANK[etype] < _TYPE_RANK[prev.entity_type]:
            found[n] = ExtractedEntity(name=n, entity_type=etype, method=method)

    # ── regex 规则 ──
    for m in _QUOTED_RE.findall(text):
        _push(m, "unknown", "regex")
    for m in _ALIAS_RE.findall(text):
        _push(m, "unknown", "regex")
    for m in _CAPS_MULTIWORD_RE.findall(text):
        _push(m, "concept", "regex")
    for m in _CAMEL_RE.findall(text):
        _push(m, "concept", "regex")
    for m in _IDENTIFIER_RE.findall(text):
        _push(m, "identifier", "regex")
    for stem, ver in _VERSIONED_RE.findall(text):
        _push(f"{stem} {ver}", "version", "regex")

    # ── jieba 词性 ──
    if _JIEBA_AVAILABLE:
        try:
            import jieba.posseg as pseg
            for word, flag in pseg.cut(text):
                if flag in _JIEBA_FLAG_TO_TYPE:
                    _push(word, _JIEBA_FLAG_TO_TYPE[flag], "jieba")
        except Exception as e:
            log.warning("jieba 抽取异常 (%s)，本次降级", e)

    return sorted(found.values())


# 数值越小越优先
_TYPE_RANK: Final[dict[str, int]] = {
    "person": 0,
    "org": 1,
    "product": 2,
    "version": 3,
    "concept": 4,
    "identifier": 5,
    "unknown": 6,
}
