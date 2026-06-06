"""异步实体富化子包。

公开 API：

- `EnrichmentQueue` — SQLite 队列封装（crash-safe at-least-once 语义）
- `QueueItem` — 队列 pop 返回的不可变记录
- `EnrichmentWorker` — 单线程 worker，注入 extractor + merge_callback
- `ExtractorFn` / `MergeCallbackFn` — 类型别名
"""

from prism.enrichment.queue import EnrichmentQueue, QueueItem
from prism.enrichment.worker import (
    EnrichmentWorker,
    ExtractorFn,
    MergeCallbackFn,
)

__all__ = [
    "EnrichmentQueue",
    "EnrichmentWorker",
    "ExtractorFn",
    "MergeCallbackFn",
    "QueueItem",
]
