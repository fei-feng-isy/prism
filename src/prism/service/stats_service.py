"""系统统计聚合业务逻辑（Service 层）。"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from prism import __version__
from prism.vstore.factory import count_distinct_embedding_models

from .types import EnrichmentStatusEntry

if TYPE_CHECKING:
    from prism.config import PrismConfig
    from prism.hrr.bank import IncrementalBank
    from prism.retriever import RetrievalPipeline, SmartPrefetch
    from prism.semantic import SemanticBackend
    from prism.tracking import CallTracker
    from prism.vstore import VectorStore

log = logging.getLogger(__name__)

__all__ = ["StatsService"]

_FTS_TOKENIZER: Final[str] = "trigram"
_TRUST_HIGH_MIN: Final[float] = 0.7
_TRUST_MID_MIN: Final[float] = 0.3


class StatsService:
    """系统统计聚合（只读）。

    Args:
        db: 已初始化 schema 的 SQLite 连接
        cfg: 完整 PrismConfig
        semantic: SemanticBackend 实例
        vstore: VectorStore 实例
        prefetch: SmartPrefetch 实例
        bank: 可选 IncrementalBank
        tracker: 可选 CallTracker
        eval_baseline_path: 可选 eval baseline JSON 路径
    """

    def __init__(
        self,
        db: sqlite3.Connection,
        cfg: PrismConfig,
        semantic: SemanticBackend,
        vstore: VectorStore,
        prefetch: SmartPrefetch,
        *,
        bank: IncrementalBank | None = None,
        tracker: CallTracker | None = None,
        eval_baseline_path: Path | None = None,
    ) -> None:
        self._db = db
        self._cfg = cfg
        self._semantic = semantic
        self._vstore = vstore
        self._prefetch = prefetch
        self._bank = bank
        self._tracker = tracker
        self._eval_baseline_path = eval_baseline_path

    def stats(self, category: str | None = None) -> dict[str, Any]:
        sem = self._semantic
        cfg = self._cfg
        sem_available = sem.is_available()
        sem_loaded = getattr(sem, "is_loaded", True)
        from prism.db import EntitiesRepository
        entities_repo = EntitiesRepository(self._db)
        return {
            "version": __version__,
            "db_path": self._db_path(),
            "semantic_backend": cfg.semantic.backend,
            "embedding_model": sem.name,
            "embedding_available": sem_available and sem_loaded,
            "rerank_enabled": cfg.semantic.backend == "hybrid_rerank",
            "fts_tokenizer": _FTS_TOKENIZER,
            "facts": self._facts_stats(category),
            "trust": self._trust_stats(),
            "banks": self._banks_stats(),
            "enrichment": self._enrichment_stats(),
            "contradictions": self._contradictions_stats(),
            "performance": self._performance_stats(),
            "eval_baseline": self._eval_baseline(),
            "vstore": self._vstore.stats(),
            "prefetch": {"warmed": self._prefetch.warmed},
            "retriever": self._retriever_stats(),
            "entities": {
                "entities_count": entities_repo.count_entities(),
                "fact_entities_count": entities_repo.count_fact_entities(),
                "distinct_embedding_models": count_distinct_embedding_models(self._db),
            },
        }

    def _facts_stats(self, category: str | None) -> dict[str, Any]:
        from prism.db import FactsRepository
        repo = FactsRepository(self._db)
        return {
            "total": repo.count_total(category),
            "active": repo.count_active(category),
            "archived": repo.count_archived(category),
            "by_category": repo.count_by_category(),
        }

    def _trust_stats(self) -> dict[str, Any]:
        from prism.db import FactsRepository
        return FactsRepository(self._db).get_trust_aggregates()

    def _banks_stats(self) -> dict[str, Any]:
        if self._bank is None:
            return {}
        return {
            cat: {
                "fact_count": int(s["fact_count"]),
                "snr": float(s["snr_estimate"]),
                "dirty": int(s["dirty_count"]),
            }
            for cat, s in self._bank.stats().items()
        }

    def _enrichment_stats(self) -> dict[str, int]:
        from prism.db import EnrichmentQueueRepository
        return EnrichmentQueueRepository(self._db).stats()

    def _contradictions_stats(self) -> dict[str, int]:
        from prism.db import ContradictionRepository
        repo = ContradictionRepository(self._db)
        return {
            "detected": repo.count_total(),
            "unresolved": repo.count_unresolved(),
        }

    def _performance_stats(self) -> dict[str, Any]:
        if self._tracker is None:
            return {
                "search_p50_ms": None,
                "search_p95_ms": None,
                "add_p50_ms": None,
                "add_p95_ms": None,
                "total_calls": None,
                "calls_by_action": None,
                "calls_by_source": None,
            }
        s = self._tracker.stats()
        return {
            "search_p50_ms": s.get("search_p50"),
            "search_p95_ms": s.get("search_p95"),
            "add_p50_ms": s.get("add_p50"),
            "add_p95_ms": s.get("add_p95"),
            "total_calls": s.get("total_calls"),
            "calls_by_action": s.get("by_action"),
            "calls_by_source": s.get("by_source"),
        }

    def _eval_baseline(self) -> dict[str, Any] | None:
        if self._eval_baseline_path is None:
            return None
        if not self._eval_baseline_path.exists():
            log.warning(
                "eval_baseline_path 不存在：%s — 返 None", self._eval_baseline_path
            )
            return None
        try:
            return json.loads(self._eval_baseline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("解析 eval_baseline 失败 %s：%s", self._eval_baseline_path, e)
            return None

    def _retriever_stats(self) -> dict[str, Any]:
        cfg_r = self._cfg.retriever
        sem = self._semantic
        sem_available = sem.is_available()
        sem_loaded = getattr(sem, "is_loaded", True)
        degraded_permanent = not sem_available
        degraded_transient = sem_available and not sem_loaded
        return {
            "weight_semantic": cfg_r.weight_semantic,
            "weight_fts": cfg_r.weight_fts,
            "weight_jaccard": cfg_r.weight_jaccard,
            "degraded": degraded_permanent or degraded_transient,
            "degraded_permanent": degraded_permanent,
            "degraded_transient": degraded_transient,
            "semantic_available": sem_available,
            "semantic_loaded": sem_loaded,
        }

    def _db_path(self) -> str:
        try:
            for row in self._db.execute("PRAGMA database_list").fetchall():
                if row["name"] == "main":
                    return str(row["file"]) or ":memory:"
        except sqlite3.Error as e:
            log.warning("读取 db_path 失败：%s", e)
        return ":memory:"

    def _count(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        row = self._db.execute(sql, params).fetchone()
        return int(row[0]) if row else 0
