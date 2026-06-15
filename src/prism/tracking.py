"""Prism API 调用追踪模块。

独立模块，集中管理全部追踪逻辑：调用频次、耗时分布、来源标识、
retrieval_count 自增。通过 :meth:`CallTracker.wrap_service` 在
``build_runtime()`` 装配阶段一次性包装 service 方法，service 层本身
零修改。

日志输出由 ``cfg.logging.call_tracking`` 控制：
    - ``enabled=False``：完全不追踪，零开销
    - ``file_logging=False``：仅内存缓冲，不写文件
    - ``file_logging=True``：内存缓冲 + RotatingFileHandler → ``{db_dir}/logs/api_calls.jsonl``
"""

from __future__ import annotations

import json
import logging
import sqlite3
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Final

__all__ = ["CallTracker"]

log = logging.getLogger(__name__)

_LOG_FILE_NAME: Final[str] = "api_calls.jsonl"
_DEFAULT_BUFFER_SIZE: Final[int] = 10_000
_DEFAULT_MAX_BYTES: Final[int] = 5_000_000
_DEFAULT_BACKUP_COUNT: Final[int] = 3


@dataclass(frozen=True, slots=True)
class CallRecord:
    """单次 API 调用记录。"""

    timestamp: str
    service: str
    action: str
    source: str
    latency_ms: float
    success: bool
    detail: dict[str, Any] | None = None


class CallTracker:
    """线程安全的 API 调用追踪器。

    Args:
        log_dir: 日志文件目录；None 时仅内存缓冲
        source: 调用来源标识（hermes / mcp / cli）
        db: 可选 SQLite 连接，用于 retrieval_count 自增
        buffer_size: 内存环形缓冲上限
        max_bytes: 单个日志文件大小上限
        backup_count: 轮转保留文件数
    """

    def __init__(
        self,
        *,
        log_dir: Path | None = None,
        source: str = "unknown",
        db: sqlite3.Connection | None = None,
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        backup_count: int = _DEFAULT_BACKUP_COUNT,
    ) -> None:
        self._source = source
        self._db = db
        self._lock = threading.Lock()
        self._records: deque[CallRecord] = deque(maxlen=buffer_size)
        self._file_logger: logging.Logger | None = None
        if log_dir is not None:
            self._setup_file_logger(log_dir, max_bytes, backup_count)

    # ─── 核心：记录 ──────────────────────────────────────────────────────

    def record(
        self,
        service: str,
        action: str,
        latency_ms: float,
        success: bool,
        detail: dict[str, Any] | None = None,
    ) -> None:
        rec = CallRecord(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            service=service,
            action=action,
            source=self._source,
            latency_ms=round(latency_ms, 3),
            success=success,
            detail=detail,
        )
        with self._lock:
            self._records.append(rec)
        if self._file_logger is not None:
            self._file_logger.info(
                json.dumps(asdict(rec), ensure_ascii=False, default=str)
            )

    # ─── 统计 ────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """从内存缓冲区计算 p50/p95/count 统计。"""
        with self._lock:
            snapshot = list(self._records)

        if not snapshot:
            return {
                "total_calls": 0,
                "search_p50": None,
                "search_p95": None,
                "add_p50": None,
                "add_p95": None,
                "by_action": {},
                "by_source": {},
            }

        by_action: dict[str, list[float]] = defaultdict(list)
        by_source: dict[str, int] = defaultdict(int)
        for r in snapshot:
            by_action[r.action].append(r.latency_ms)
            by_source[r.source] += 1

        def _percentile(data: list[float], p: int) -> float | None:
            if not data:
                return None
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * p / 100
            f = int(k)
            c = f + 1
            if c >= len(sorted_data):
                return round(sorted_data[f], 3)
            return round(sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f]), 3)

        search_latencies = by_action.get("search", []) + by_action.get("search_markdown", [])
        add_latencies = by_action.get("add", [])

        action_counts = {k: len(v) for k, v in by_action.items()}

        return {
            "total_calls": len(snapshot),
            "search_p50": _percentile(search_latencies, 50),
            "search_p95": _percentile(search_latencies, 95),
            "add_p50": _percentile(add_latencies, 50),
            "add_p95": _percentile(add_latencies, 95),
            "by_action": action_counts,
            "by_source": dict(by_source),
        }

    # ─── service 包装 ────────────────────────────────────────────────────

    def wrap_service(
        self,
        service: object,
        service_name: str,
        methods: list[str],
    ) -> None:
        """一次性包装 service 实例的指定方法，注入计时 + 记录。

        对 search/search_markdown 额外自增 retrieval_count。
        """
        for method_name in methods:
            original = getattr(service, method_name)
            if service_name == "search" and method_name in ("search", "search_markdown"):
                wrapped = self._make_search_wrapper(original, method_name)
            else:
                wrapped = self._make_wrapper(original, service_name, method_name)
            setattr(service, method_name, wrapped)

    def _make_wrapper(
        self, fn: Callable[..., Any], service_name: str, action: str
    ) -> Callable[..., Any]:
        tracker = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.monotonic()
            ok = False
            try:
                result = fn(*args, **kwargs)
                ok = True
                return result
            finally:
                elapsed = (time.monotonic() - t0) * 1000
                tracker.record(service_name, action, elapsed, ok)

        wrapper.__name__ = fn.__name__  # type: ignore[attr-defined]
        wrapper.__qualname__ = fn.__qualname__  # type: ignore[attr-defined]
        return wrapper

    def _make_search_wrapper(
        self, fn: Callable[..., Any], action: str
    ) -> Callable[..., Any]:
        tracker = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.monotonic()
            ok = False
            try:
                result = fn(*args, **kwargs)
                ok = True
                return result
            finally:
                elapsed = (time.monotonic() - t0) * 1000
                detail = None
                if ok and result is not None:
                    fact_ids = _extract_fact_ids(result)
                    if fact_ids:
                        detail = {"hit_count": len(fact_ids)}
                        tracker._increment_retrieval_count(fact_ids)
                tracker.record("search", action, elapsed, ok, detail)

        wrapper.__name__ = fn.__name__  # type: ignore[attr-defined]
        wrapper.__qualname__ = fn.__qualname__  # type: ignore[attr-defined]
        return wrapper

    def _increment_retrieval_count(self, fact_ids: list[int]) -> None:
        if not self._db or not fact_ids:
            return
        try:
            from .db import FactsRepository
            FactsRepository(self._db).increment_retrieval_count(fact_ids)
        except Exception as e:
            log.debug("retrieval_count 更新失败：%s", e)

    # ─── 文件日志 ────────────────────────────────────────────────────────

    def _setup_file_logger(
        self, log_dir: Path, max_bytes: int, backup_count: int
    ) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_path = log_dir / _LOG_FILE_NAME

        logger = logging.getLogger(f"prism.tracking.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        handler = RotatingFileHandler(
            str(file_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        self._file_logger = logger

        log.info("调用追踪日志：%s", file_path)


def _extract_fact_ids(result: Any) -> list[int]:
    """从 search/search_markdown 结果中提取 fact_ids。"""
    if isinstance(result, list):
        return [getattr(h, "fact_id", 0) for h in result if hasattr(h, "fact_id")]
    if isinstance(result, str):
        return []
    return []
