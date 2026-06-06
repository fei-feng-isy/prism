"""Prism 运行时装配（与 transport 无关）。

:func:`build_runtime` 集中执行九步装配（config → DB → bank → semantic →
vstore → mirror → APIs → 可选 LLM 富化 + 去抖 calibrate），返回
:class:`PrismRuntime` dataclass。transport 层只持有 runtime 引用即可。
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prism.api import PrismAdmin, PrismRecall, PrismRemember
from prism.config import (
    PrismConfig,
    default_config,
    discover_config_path,
    load_config,
    resolve_db_path_for_user,
)
from prism.db import bootstrap
from prism.enrichment import EnrichmentQueue, EnrichmentWorker
from prism.hrr import IncrementalBank, RebuildDebouncer
from prism.mirror import PrismMirror
from prism.retriever import RetrievalPipeline, SmartPrefetch
from prism.semantic.factory import create_semantic_assembly
from prism.service import AdminService, FactService, SearchService
from prism.tracking import CallTracker
from prism.vstore.factory import create_vstore

if TYPE_CHECKING:
    from prism.semantic import SemanticBackend
    from prism.vstore import VectorStore

log = logging.getLogger(__name__)

__all__ = [
    "PrismRuntime",
    "RuntimeOptions",
    "build_runtime",
]


# ─── 选项 ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RuntimeOptions:
    """``build_runtime`` 的所有可调参数集中点。

    Attributes:
        config_path: 可选 YAML 配置文件路径；None 时走 :func:`default_config`
        profile: 多 profile 隔离用（对应 cfg.db.path_template ``{profile}``）
        user_id: 多用户隔离用，内部 sha256 → ``{user_hash}``
        data_home: 数据根目录覆盖；None 时用 cfg.db.data_home_default
        llm_chat_fn: 可选 LLM chat_fn；提供则启动 :class:`EnrichmentWorker`
            daemon 线程异步执行 Stage 2 富化
        bank_window_seconds: :class:`RebuildDebouncer` 去抖窗口；默认 50ms
        start_worker: 是否真启 daemon 线程；False 时仍构造 worker 但不 start
            （测试期 run_once 手动驱动用）
        warmup_prefetch: ``initialize`` 末尾是否调 ``prefetch.warmup()``；
            生产建议 True（预热 BGE）；测试 / 启动延迟敏感场景可关
    """

    config_path: str | None = None
    profile: str = "default"
    user_id: str = "local_default"
    data_home: str | None = None
    llm_chat_fn: Callable[[str], str] | None = None
    bank_window_seconds: float = 0.050
    start_worker: bool = True
    warmup_prefetch: bool = True
    db_path_override: str | None = None  # 测试场景直传 :memory: 或 tmp 文件
    call_source: str = "unknown"  # 调用来源标识（hermes / mcp / cli）


# ─── runtime 容器 ────────────────────────────────────────────────────────────


@dataclass
class PrismRuntime:
    """所有装配好的运行时部件（mutable，含 cleanup state）。

    transport 层（MCP server）只读这些字段调用相应 API。``shutdown()``
    幂等关闭 worker / DB / vstore。
    """

    cfg: PrismConfig
    db: sqlite3.Connection
    bank: IncrementalBank
    semantic: SemanticBackend
    vstore: VectorStore
    mirror: PrismMirror
    pipeline: RetrievalPipeline
    prefetch: SmartPrefetch
    fact_service: FactService
    search_service: SearchService
    admin_service: AdminService
    remember: PrismRemember
    recall: PrismRecall
    admin: PrismAdmin
    enrichment_queue: EnrichmentQueue
    debouncer: RebuildDebouncer
    db_path: Path
    vstore_path: Path
    call_tracker: CallTracker | None = None
    enrichment_worker: EnrichmentWorker | None = None
    warmup_thread: threading.Thread | None = None
    _shutdown: bool = field(default=False, repr=False)

    def shutdown(self) -> None:
        """关 worker → join warmup → persist vstore → close DB。幂等。"""
        if self._shutdown:
            return
        self._shutdown = True

        # 1) 停 worker 线程（如启用）
        if self.enrichment_worker is not None:
            try:
                self.enrichment_worker.stop()
            except Exception as e:
                log.debug("worker stop 异常：%s", e)

        # 2) 等待 warmup 线程退出（避免后台仍在写 vstore 时被 persist 抢路径）
        if self.warmup_thread is not None and self.warmup_thread.is_alive():
            self.warmup_thread.join(timeout=2.0)
            if self.warmup_thread.is_alive():
                log.warning(
                    "prism-warmup 线程未在 2s 内退出；继续 shutdown（daemon 线程会被进程清理）"
                )

        # 3) persist vstore
        try:
            self.vstore.persist()
        except Exception as e:
            log.warning("vstore persist 失败：%s", e)

        # 4) close DB
        try:
            self.db.close()
        except Exception as e:
            log.debug("DB close 异常：%s", e)


# ─── builder ─────────────────────────────────────────────────────────────────


def _make_calibrate_callback(
    bank: IncrementalBank,
    db: sqlite3.Connection,
    hrr_dim: int,
) -> Callable[[], None]:
    """生成 :class:`RebuildDebouncer` 回调：触发时按 bank 内 categories 全量
    校准（重新从 active facts 的 hrr_vector 计算 z_sum），防 z_sum 漂移。
    """
    import numpy as np

    def _cb() -> None:
        try:
            cats = list(bank.categories())
        except Exception as e:
            log.warning("calibrate 取 categories 失败：%s", e)
            return
        for category in cats:
            try:
                rows = db.execute(
                    "SELECT hrr_vector FROM facts "
                    "WHERE status='active' AND category=? AND hrr_vector IS NOT NULL",
                    (category,),
                ).fetchall()
                vectors: list[np.ndarray] = []
                for row in rows:
                    blob = row["hrr_vector"]
                    if blob is None:
                        continue
                    v = np.frombuffer(blob, dtype=np.float64)
                    if v.shape[0] == hrr_dim:
                        vectors.append(v)
                bank.calibrate(category, vectors)
            except Exception as e:
                log.warning("calibrate category=%s 失败：%s", category, e)

    return _cb


def _maybe_start_worker(
    queue: EnrichmentQueue,
    mirror: PrismMirror,
    llm_chat_fn: Callable[[str], str] | None,
    *,
    start: bool,
) -> EnrichmentWorker | None:
    """提供 chat_fn 时构造 LLMExtractor + EnrichmentWorker；start=True 时启
    daemon 线程（用 :meth:`EnrichmentWorker.start` 内置循环）。

    返回 worker（或 None）；停止由 :meth:`PrismRuntime.shutdown` 统一处理。
    """
    if llm_chat_fn is None:
        return None

    from prism.entities.llm_extractor import LLMExtractor

    extractor = LLMExtractor(llm_chat_fn)
    worker = EnrichmentWorker(
        queue=queue,
        extractor=extractor,
        merge_callback=mirror.enrichment_merge,
    )
    if start:
        worker.start()
    return worker


def build_runtime(opts: RuntimeOptions | None = None, **kwargs: Any) -> PrismRuntime:
    """装配并返回 :class:`PrismRuntime`。

    Args:
        opts: 全部参数的封装；若 None，用 ``kwargs`` 构造 RuntimeOptions
        **kwargs: opts is None 时的便捷参数（与 RuntimeOptions 字段同名）

    Returns:
        已就绪的 PrismRuntime；调用方负责后续 :meth:`PrismRuntime.shutdown`
    """
    if opts is None:
        opts = RuntimeOptions(**kwargs)

    # 1) 配置
    if opts.config_path:
        cfg_path: Path | None = Path(opts.config_path)
    else:
        cfg_path = discover_config_path()
    cfg = load_config(cfg_path) if cfg_path else default_config()

    # 2) DB 路径
    if opts.db_path_override is not None:
        db_path = Path(opts.db_path_override)
    else:
        db_path = resolve_db_path_for_user(
            cfg.db,
            user_id=opts.user_id,
            profile=opts.profile,
            data_home=opts.data_home,
        )

    log.info(
        "Prism runtime initializing: profile=%s db_path=%s llm=%s",
        opts.profile, db_path, opts.llm_chat_fn is not None,
    )

    # 3) DB + bank
    db = bootstrap(str(db_path))
    bank = IncrementalBank(dim=cfg.hrr.dim)

    # 4) semantic backend（write + query 分离，按 cfg.semantic.backend 装配）
    assembly = create_semantic_assembly(cfg)
    write_semantic = assembly.write
    semantic = assembly.query  # pipeline / prefetch / recall 用
    dim = assembly.dim

    # 5) vstore
    vstore_path = db_path.with_suffix(".vstore.npz") if str(db_path) != ":memory:" \
        else Path(":memory:")
    vstore = create_vstore(
        cfg,
        db,
        dim=dim,
        path=str(vstore_path) if str(vstore_path) != ":memory:" else None,
    )

    # 6) enrichment queue + RebuildDebouncer
    queue = EnrichmentQueue(db)
    debouncer = RebuildDebouncer(
        _make_calibrate_callback(bank, db, cfg.hrr.dim),
        window_seconds=opts.bank_window_seconds,
    )

    # 7) mirror（写路径强制 LocalBge）
    if write_semantic.is_available():
        mirror = PrismMirror(
            db,
            bank,
            hrr_dim=cfg.hrr.dim,
            semantic=write_semantic,
            vstore=vstore,
            enrichment_queue=queue,
            entities_config=cfg.entities,
            rebuild_debouncer=debouncer,
        )
    else:
        mirror = PrismMirror(
            db,
            bank,
            hrr_dim=cfg.hrr.dim,
            enrichment_queue=queue,
            entities_config=cfg.entities,
            rebuild_debouncer=debouncer,
        )

    # 8) pipeline + prefetch
    pipeline = RetrievalPipeline(cfg=cfg, db=db, semantic=semantic, vstore=vstore)
    prefetch = SmartPrefetch(pipeline, max_results=cfg.retriever.prefetch.top_k)

    # 9) service 层
    fact_service = FactService(db, mirror)
    search_service = SearchService(db, pipeline, prefetch)

    # 9.5) 调用追踪（需在 AdminService 之前构造，以便注入 tracker）
    tracking_cfg = cfg.logging.call_tracking
    call_tracker: CallTracker | None = None
    if tracking_cfg.enabled:
        if tracking_cfg.file_logging and str(db_path) != ":memory:":
            tracker_log_dir: Path | None = db_path.parent / "logs"
        else:
            tracker_log_dir = None
        call_tracker = CallTracker(
            log_dir=tracker_log_dir,
            source=opts.call_source,
            db=db,
            buffer_size=tracking_cfg.buffer_size,
            max_bytes=tracking_cfg.max_bytes,
            backup_count=tracking_cfg.backup_count,
        )
        call_tracker.wrap_service(
            fact_service, "fact",
            ["add", "edit", "remove", "helpful", "unhelpful",
             "archive", "restore", "list", "show"],
        )
        call_tracker.wrap_service(
            search_service, "search",
            ["search", "search_markdown", "probe", "reason",
             "related", "contradict"],
        )

    admin_service = AdminService(
        pipeline, prefetch, mirror=mirror, bank=bank, tracker=call_tracker,
    )

    # 10) APIs（薄壳，委托 service）
    remember = PrismRemember(fact_service)
    recall = PrismRecall(search_service)
    admin = PrismAdmin(admin_service, fact_service)

    # 11) 可选 LLM worker
    worker = _maybe_start_worker(
        queue, mirror, opts.llm_chat_fn, start=opts.start_worker,
    )

    # 12) warmup BGE — 异步 daemon 线程
    #    与 hermes plugin 对齐：让 build_runtime 立即返回，BGE 加载（500ms-2s）
    #    在后台完成；RetrievalPipeline 通过 semantic.is_loaded 判定走降级路径，
    #    模型加载完成后自动切回完整语义路径。
    warmup_thread: threading.Thread | None = None
    if opts.warmup_prefetch:
        def _warmup_background() -> None:
            try:
                prefetch.warmup()
                log.info("Prism BGE warmup 完成（后台）")
            except Exception as e:
                log.warning("prefetch warmup 异常（后台）：%s", e)

        warmup_thread = threading.Thread(
            target=_warmup_background, daemon=True, name="prism-warmup"
        )
        warmup_thread.start()

    return PrismRuntime(
        cfg=cfg,
        db=db,
        bank=bank,
        semantic=semantic,
        vstore=vstore,
        mirror=mirror,
        pipeline=pipeline,
        prefetch=prefetch,
        fact_service=fact_service,
        search_service=search_service,
        admin_service=admin_service,
        remember=remember,
        recall=recall,
        admin=admin,
        enrichment_queue=queue,
        debouncer=debouncer,
        db_path=db_path,
        vstore_path=vstore_path,
        call_tracker=call_tracker,
        enrichment_worker=worker,
        warmup_thread=warmup_thread,
    )
