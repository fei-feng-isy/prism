"""Prism 检索层。

暴露：

* :class:`RecallResult` — 单条召回结果（含融合分 + 三路分项）
* :class:`RetrievalPipeline` — semantic + FTS5 trigram + jieba/Jaccard 三路加权融合
* :func:`compute_jaccard` / :func:`min_max_normalize` — 融合用 helper（也方便单测）
* :class:`SmartPrefetch` — 把 recall 结果格式化为 LLM 注入 markdown，含 warm-up 钩子

"""

from __future__ import annotations

from .fusion import (
    RecallResult,
    RetrievalPipeline,
    compute_jaccard,
    min_max_normalize,
)
from .prefetch import SmartPrefetch

__all__ = [
    "RecallResult",
    "RetrievalPipeline",
    "SmartPrefetch",
    "compute_jaccard",
    "min_max_normalize",
]
