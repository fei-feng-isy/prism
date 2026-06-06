"""异步富化 worker：从队列取 fact、调 extractor、合并实体、标 done/failed。

通过依赖注入解耦：队列由 ``EnrichmentQueue`` 管理，实体抽取由 ``ExtractorFn``
提供，实体合并由 ``merge_callback`` 完成。extractor / merge_callback 异常会
``mark_failed`` 而非上抛，保证单条 fact 失败不影响整个 loop。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable

from prism.enrichment.queue import EnrichmentQueue, QueueItem

log = logging.getLogger(__name__)


ExtractorFn = Callable[[str], Iterable[str]]
"""签名：(content: str) -> Iterable[str]，可抛任意异常。"""

MergeCallbackFn = Callable[[int, list[str]], None]
"""签名：(fact_id: int, entities: list[str]) -> None，可抛任意异常。"""


class EnrichmentWorker:
    """异步富化 worker — 单线程消费 EnrichmentQueue。"""

    def __init__(
        self,
        queue: EnrichmentQueue,
        extractor: ExtractorFn,
        *,
        merge_callback: MergeCallbackFn | None = None,
        poll_interval: float = 1.0,
        timeout_seconds: float = 10.0,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError(
                f"poll_interval must be > 0, got {poll_interval}"
            )
        if timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be > 0, got {timeout_seconds}"
            )
        self._queue = queue
        self._extractor = extractor
        self._merge_callback = merge_callback
        self._poll_interval = poll_interval
        self._timeout_seconds = timeout_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run_once(self) -> bool:
        """处理一个队列项。返回 True 表示有处理（成功 or 失败），False 表示队列空。"""
        item = self._queue.pop_next()
        if item is None:
            return False
        self._process(item)
        return True

    def _process(self, item: QueueItem) -> None:
        try:
            extracted = self._extractor(item.content)
            entities = [str(e).strip() for e in extracted if str(e).strip()]
        except Exception as exc:
            log.warning(
                "enrichment 抽取异常 fact_id=%s attempt=%s: %s",
                item.fact_id,
                item.attempts,
                exc,
            )
            self._queue.mark_failed(item.fact_id, f"extract: {exc}")
            return

        if self._merge_callback is not None:
            try:
                self._merge_callback(item.fact_id, entities)
            except Exception as exc:
                log.warning(
                    "enrichment merge_callback 异常 fact_id=%s: %s",
                    item.fact_id,
                    exc,
                )
                self._queue.mark_failed(item.fact_id, f"merge: {exc}")
                return

        self._queue.mark_done(item.fact_id)

    def start(self) -> None:
        """启动后台 daemon thread。如已在跑则 no-op。"""
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="prism-enrichment-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        """通知 worker 停止；join 等 timeout 秒。幂等。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                processed = self.run_once()
            except Exception as exc:
                # queue.pop_next / mark_* 自身异常 → log + 暂停 poll_interval 后重试
                log.error(
                    "enrichment worker loop 异常（队列层）：%s", exc
                )
                processed = False
            if not processed:
                # 队列空（或 loop 异常）：sleep poll_interval 或直到 stop signal
                self._stop_event.wait(self._poll_interval)
