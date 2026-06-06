"""运维业务逻辑门面（Service 层）。

向后兼容门面 — 委托到 :class:`StatsService`、:class:`RepairService`、
:class:`ImportService`。新代码应直接使用子 Service。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .import_service import ImportService
from .repair_service import RepairService
from .stats_service import StatsService
from .types import EnrichmentDiagnosis, EnrichmentFixResult, MirrorMdStats

if TYPE_CHECKING:
    from prism.hrr.bank import IncrementalBank
    from prism.mirror import PrismMirror
    from prism.retriever import RetrievalPipeline, SmartPrefetch
    from prism.tracking import CallTracker

log = logging.getLogger(__name__)

__all__ = ["AdminService"]


class AdminService:
    """运维门面 — 委托到 StatsService / RepairService / ImportService。

    保留此类是为了向后兼容现有 API 层和 wire.py 的构造方式。
    新功能应直接添加到对应子 Service。

    Args:
        pipeline: :class:`RetrievalPipeline`
        prefetch: :class:`SmartPrefetch`
        mirror: 可选 :class:`PrismMirror`（ImportService 需要）
        bank: 可选 :class:`IncrementalBank`（StatsService 需要）
        tracker: 可选 :class:`CallTracker`（StatsService 需要）
        eval_baseline_path: 可选 eval baseline JSON 路径
    """

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        prefetch: SmartPrefetch,
        *,
        mirror: PrismMirror | None = None,
        bank: IncrementalBank | None = None,
        tracker: CallTracker | None = None,
        eval_baseline_path: str | Path | None = None,
    ) -> None:
        self._stats_service = StatsService(
            db=pipeline.db,
            cfg=pipeline.cfg,
            semantic=pipeline.semantic,
            vstore=pipeline.vstore,
            prefetch=prefetch,
            bank=bank,
            tracker=tracker,
            eval_baseline_path=(
                Path(eval_baseline_path) if eval_baseline_path is not None else None
            ),
        )
        self._repair_service = RepairService(
            db=pipeline.db,
            semantic=pipeline.semantic,
        )
        self._import_service: ImportService | None = (
            ImportService(mirror) if mirror is not None else None
        )

    @property
    def stats_service(self) -> StatsService:
        return self._stats_service

    @property
    def repair_service(self) -> RepairService:
        return self._repair_service

    @property
    def import_service(self) -> ImportService | None:
        return self._import_service

    # ─── 委托方法（向后兼容） ────────────────────────────────────────────

    def stats(self, category: str | None = None) -> dict[str, Any]:
        return self._stats_service.stats(category=category)

    def enrichment_diagnose(self) -> EnrichmentDiagnosis:
        return self._repair_service.enrichment_diagnose()

    def enrichment_fix(self, *, dry_run: bool = False) -> EnrichmentFixResult:
        return self._repair_service.enrichment_fix(dry_run=dry_run)

    def mirror_memory_md(
        self,
        md_path: Path,
        *,
        prune: bool = False,
    ) -> MirrorMdStats:
        if self._import_service is None:
            raise RuntimeError("AdminService 未注入 mirror，无法执行 mirror_memory_md")
        return self._import_service.mirror_memory_md(md_path, prune=prune)
