"""三路融合检索 pipeline。

接口围绕 :class:`RetrievalPipeline.recall`，按 ``cfg.retriever.weight_*`` 加权融合：

1. **semantic 路径**：``semantic.encode(query)`` → ``vstore.topk(vec, k*over_fetch)``
2. **fts 路径**：SQLite FTS5 ``facts_fts MATCH ?``（trigram tokenizer，≥3 字符要求）
3. **entity 路径**：``extract_entities(query)`` → 与 ``fact_entities`` 求 Jaccard

三路各自 ``min-max`` 归一化到 ``[0, 1]``（避免 cosine ∈ [-1,1] 与 BM25 自然分
量级混杂），缺失路径补 0；按 ``cfg.retriever.weight_semantic / weight_fts /
weight_jaccard`` 加权求和，最终按融合分降序取 ``top k``。

降级路径（``semantic.is_available() == False``）：跳过 semantic 路径，权重切
:func:`semantic.degraded_weights` ``(0, 0.65, 0.35)``。
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from prism.entities.regex_extractor import extract_entities
from prism.semantic import (
    DEGRADED_FTS_WEIGHT,
    DEGRADED_JACCARD_WEIGHT,
    DEGRADED_SEMANTIC_WEIGHT,
    SemanticUnavailable,
)

if TYPE_CHECKING:
    from prism.config import PrismConfig
    from prism.semantic import SemanticBackend
    from prism.vstore import VectorStore

log = logging.getLogger(__name__)

__all__ = [
    "RecallResult",
    "RetrievalPipeline",
    "compute_jaccard",
    "min_max_normalize",
]


# trigram tokenizer 要求查询 ≥ 3 字符；不足时 fts 路径返回空（不抛）
_FTS_MIN_CHUNK_LEN: Final[int] = 3

# 过取倍数：每路 over_fetch_factor × k 候选送入融合（默认 5 → 取 50 候选融合出 10）
_DEFAULT_OVER_FETCH_FACTOR: Final[int] = 5

# 切 FTS5 query 的分隔字符：whitespace + FTS5 保留字符（避免 phrase 内 mixed 内容
# 让 trigram 要求严格连续匹配 — 例如 query "中文回答 \"UniqueEntity\"" 整段作为 phrase
# 会要求所有 trigrams 按顺序连续出现，但实际期望是「任一 chunk 命中即可」）
_FTS_CHUNK_SPLIT_RE: Final[re.Pattern[str]] = re.compile(r'[\s"*+()\-:^]+')


@dataclass(frozen=True, slots=True)
class RecallResult:
    """单条召回结果。

    Attributes:
        fact_id: facts 主键
        content: facts.content
        score: 融合后总分（``[0, 1]``，越大越相关）
        path_scores: 三路各自归一化后分项 ``{"semantic": .., "fts": .., "entity": ..}``。
            未参与某路（无候选）的 fact 在该路下分为 ``0.0``；degraded 模式
            ``"semantic"`` 永远为 ``0.0``。
    """

    fact_id: int
    content: str
    score: float
    path_scores: dict[str, float]


# ─── helpers（公开便于单测 + 复用） ─────────────────────────────────────────


def min_max_normalize(scores: dict[int, float]) -> dict[int, float]:
    """把任意范围的分数线性映射到 ``[0, 1]``；最大最小相同时全 1.0。

    空 dict 返回空 dict；单元素 dict 返回 ``{id: 1.0}``。
    """
    if not scores:
        return {}
    values = scores.values()
    lo = min(values)
    hi = max(values)
    if hi == lo:
        # 所有分数相等：全部视为最高（避免除零，也避免全 0 让该路被融合权重抹掉）
        return dict.fromkeys(scores.keys(), 1.0)
    span = hi - lo
    return {fid: (s - lo) / span for fid, s in scores.items()}


def _build_fts_or_query(query: str) -> str | None:
    """把任意用户 query 切成 ≥3 字符 chunk，OR 连接成合法 FTS5 phrase 查询。

    ``None`` 表示无可用 chunk（短 query 或全是分隔符 / 特殊字符）— 调用方应跳过。
    """
    if not query:
        return None
    chunks = _FTS_CHUNK_SPLIT_RE.split(query)
    valid = [c for c in chunks if len(c) >= _FTS_MIN_CHUNK_LEN]
    if not valid:
        return None
    return " OR ".join(f'"{c}"' for c in valid)


def compute_jaccard(
    query_entity_ids: set[int],
    fact_entity_ids: set[int],
) -> float:
    """``|A ∩ B| / |A ∪ B|``；两侧都空 → 0（避免假阳性高分）。"""
    if not query_entity_ids and not fact_entity_ids:
        return 0.0
    inter = len(query_entity_ids & fact_entity_ids)
    union = len(query_entity_ids | fact_entity_ids)
    return inter / union if union > 0 else 0.0


# ─── 主类 ────────────────────────────────────────────────────────────────────


class RetrievalPipeline:
    """三路融合检索 pipeline。

    Args:
        cfg: 完整 PrismConfig；读 ``cfg.retriever.weight_*``
        db: 已 ``init_schema`` 的 SQLite 连接
        semantic: :class:`SemanticBackend` 实例（degraded 实例触发降级路径）
        vstore: :class:`VectorStore` 实例（degraded 时不调用，但仍要求传入避免可选分支）
        over_fetch_factor: 每路过取倍数（默认 5；返回前融合时取并集再切前 k）

    Note:
        本类不维护状态、不缓存；每次 :meth:`recall` 都直接打 SQLite + vstore + semantic。
        缓存交给 smart_prefetch。
    """

    def __init__(
        self,
        *,
        cfg: PrismConfig,
        db: sqlite3.Connection,
        semantic: SemanticBackend,
        vstore: VectorStore,
        over_fetch_factor: int = _DEFAULT_OVER_FETCH_FACTOR,
    ) -> None:
        if over_fetch_factor < 1:
            raise ValueError(f"over_fetch_factor 必须 ≥ 1：got {over_fetch_factor}")
        self._cfg = cfg
        self._db = db
        self._semantic = semantic
        self._vstore = vstore
        self._over_fetch = over_fetch_factor

    @property
    def cfg(self) -> PrismConfig:
        return self._cfg

    @property
    def db(self) -> sqlite3.Connection:
        return self._db

    @property
    def semantic(self) -> SemanticBackend:
        return self._semantic

    @property
    def vstore(self) -> VectorStore:
        return self._vstore

    # ─── 公共入口 ────────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        *,
        k: int = 10,
        category: str | None = None,
        status: str = "active",
    ) -> list[RecallResult]:
        """三路召回 + 融合。

        Args:
            query: 用户查询文本（会 strip）
            k: 最终返回条数
            category: 限定 facts.category（``None`` = 不限）
            status: facts.status 过滤（默认 ``"active"`` 排除归档）

        Returns:
            按融合分降序的 ``RecallResult`` 列表；最多 ``k`` 条。
            空查询或三路均无候选时返回 ``[]``。
        """
        normalized = query.strip()
        if not normalized or k <= 0:
            return []

        # 两层降级判定 —
        #   (a) 包不可用（sentence-transformers 未安装）→ 永久降级
        #   (b) 模型未加载（异步 warmup 进行中）      → 临时降级，加载完成后自动恢复
        # is_loaded 已经是 SemanticBackend Protocol 必需属性，不再用 getattr 兜底。
        _sem_available = self._semantic.is_available()
        _sem_loaded = self._semantic.is_loaded
        is_degraded = not _sem_available or not _sem_loaded
        if _sem_available and not _sem_loaded:
            log.debug("语义模型尚未加载（异步 warmup 进行中），本次召回走 fts+entity")

        weights = self._resolve_weights(is_degraded)

        # 每路独立召回 → 原始分数（fact_id -> raw_score）
        per_path: dict[str, dict[int, float]] = {
            "semantic": (
                {} if is_degraded
                else self._semantic_path(normalized, k * self._over_fetch)
            ),
            "fts": self._fts_path(normalized, k * self._over_fetch),
            "entity": self._entity_path(normalized, k * self._over_fetch),
        }

        # 候选并集 → SQLite 过滤 status / category 后取 content
        candidate_ids = set().union(*(d.keys() for d in per_path.values()))
        if not candidate_ids:
            return []

        contents = self._fetch_filtered_contents(candidate_ids, status, category)
        if not contents:
            return []

        # 各路归一化 + 加权融合
        normalized_per_path = {
            path: min_max_normalize(scores) for path, scores in per_path.items()
        }
        fused: list[RecallResult] = []
        for fid, content in contents.items():
            path_scores = {
                path: normalized_per_path[path].get(fid, 0.0)
                for path in ("semantic", "fts", "entity")
            }
            score = (
                weights["semantic"] * path_scores["semantic"]
                + weights["fts"] * path_scores["fts"]
                + weights["entity"] * path_scores["entity"]
            )
            fused.append(
                RecallResult(
                    fact_id=fid,
                    content=content,
                    score=score,
                    path_scores=path_scores,
                )
            )

        # 按融合分降序；分数并列按 fact_id 升序（稳定可预测）
        fused.sort(key=lambda r: (-r.score, r.fact_id))
        return fused[:k]

    # ─── 三路实现 ────────────────────────────────────────────────────────

    def _semantic_path(self, query: str, n: int) -> dict[int, float]:
        """vstore.topk over-fetch → ``{fact_id: cos_score}``（cos 已 ∈ [-1, 1]）。

        encode 失败 / degraded encode 抛 ``SemanticUnavailable``：返回 ``{}``，不抛。
        """
        if len(self._vstore) == 0:
            return {}
        try:
            vec = self._semantic.encode(query)
        except SemanticUnavailable:
            # 运行期失败（is_available() 返 True 但 encode 时模型加载崩了）
            log.warning("semantic.encode 运行期失败；本次 recall 走 fts+entity")
            return {}
        pairs = self._vstore.topk(vec, k=n)
        return dict(pairs)

    def _fts_path(self, query: str, n: int) -> dict[int, float]:
        """FTS5 trigram MATCH → ``{fact_id: -bm25}``（bm25 越负越相关，取反让大即好）。

        实现细节：
            * 按 whitespace + FTS5 保留字符切 chunk（``"``/``*``/``+``/``(``/``)`` 等）
            * 长度 ≥ 3 的 chunk 各自包成 phrase（``"chunk"``），用 OR 连接
            * 全部 chunk < 3 或 query 空 → 返回 ``{}`` 不抛
            * bm25 越小越好（SQLite FTS5 惯例），取负让 fusion「越大越好」统一
        """
        fts_query = _build_fts_or_query(query)
        if fts_query is None:
            return {}
        try:
            rows = self._db.execute(
                "SELECT rowid AS fact_id, bm25(facts_fts) AS score "
                "FROM facts_fts WHERE facts_fts MATCH ? "
                "ORDER BY score LIMIT ?",
                (fts_query, n),
            ).fetchall()
        except sqlite3.OperationalError as e:
            # 退化保护：极端 query 让 FTS5 解析失败（理论上 chunk 切+phrase 已避免）
            log.warning("FTS5 MATCH 失败 (query=%r): %s", fts_query, e)
            return {}
        return {int(r["fact_id"]): -float(r["score"]) for r in rows}

    def _entity_path(self, query: str, n: int) -> dict[int, float]:
        """提取 query 实体 → 与每条 fact 的 entities 求 Jaccard → top n。

        实现策略：
            1. ``extract_entities(query)`` 取 query 实体名
            2. SQLite 查这些 name 对应的 entity_id（未入库的 entity 名直接放弃，
               因为它不可能与任何 fact 共享 → Jaccard 分子必为 0）
            3. 找含 ≥1 个 query entity 的候选 fact，一次 SQL 取 fact_entities
               全量；Python 端按 fact_id 分组 → 计算 Jaccard
        """
        extracted = extract_entities(query)
        query_names = {e.name for e in extracted}
        if not query_names:
            return {}

        # query 实体名 → entity_id
        from prism.db import EntitiesRepository
        entities_repo = EntitiesRepository(self._db)
        name_map = entities_repo.get_entity_ids_by_names(query_names)
        query_eids: set[int] = set(name_map.values())
        if not query_eids:
            return {}

        # 找含 ≥ 1 个 query entity 的 fact，再取它们的全部 entity_ids
        eid_list = list(query_eids)
        eid_placeholders = ",".join("?" * len(eid_list))
        rows = self._db.execute(
            f"SELECT fact_id, entity_id FROM fact_entities WHERE fact_id IN ("
            f"SELECT DISTINCT fact_id FROM fact_entities WHERE entity_id IN ({eid_placeholders})"
            f")",
            tuple(eid_list),
        ).fetchall()

        fact_eids: dict[int, set[int]] = {}
        for r in rows:
            fact_eids.setdefault(int(r["fact_id"]), set()).add(int(r["entity_id"]))

        scored = {
            fid: compute_jaccard(query_eids, eids)
            for fid, eids in fact_eids.items()
        }
        # 取 top n
        top = sorted(scored.items(), key=lambda kv: -kv[1])[:n]
        return dict(top)

    # ─── 内部 helpers ────────────────────────────────────────────────────

    def _resolve_weights(self, is_degraded: bool) -> dict[str, float]:
        """正常 → cfg；degraded → 固定 :func:`degraded_weights`。"""
        if is_degraded:
            return {
                "semantic": DEGRADED_SEMANTIC_WEIGHT,
                "fts": DEGRADED_FTS_WEIGHT,
                "entity": DEGRADED_JACCARD_WEIGHT,
            }
        r = self._cfg.retriever
        return {
            "semantic": r.weight_semantic,
            "fts": r.weight_fts,
            "entity": r.weight_jaccard,
        }

    def _fetch_filtered_contents(
        self,
        candidate_ids: set[int],
        status: str,
        category: str | None,
    ) -> dict[int, str]:
        """按 status / category 过滤候选 fact_ids，返回 ``{fact_id: content}``。

        调用方在 :meth:`recall` 已 guard ``candidate_ids`` 非空 — 此处不再重复判空。
        """
        placeholders = ",".join("?" * len(candidate_ids))
        params: list[object] = [*candidate_ids, status]
        sql = (
            f"SELECT fact_id, content FROM facts "
            f"WHERE fact_id IN ({placeholders}) AND status = ?"
        )
        if category is not None:
            sql += " AND category = ?"
            params.append(category)
        rows = self._db.execute(sql, tuple(params)).fetchall()
        return {int(r["fact_id"]): str(r["content"]) for r in rows}
