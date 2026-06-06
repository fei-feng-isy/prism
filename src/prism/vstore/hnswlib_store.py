"""``HnswlibVectorStore`` -- hnswlib cosine HNSW ANN backend。

适用规模：``2000 < N <= 100k``。使用逻辑删除（``mark_deleted``）+
``allow_replace_deleted``；``deleted_count / ntotal > 0.2`` 标 ``dirty``
需要 ``rebuild_from_iter`` 重建。持久化为 ``.bin`` 索引 + sidecar JSON。

向量要求：L2 归一化 float32；NaN/Inf 拒收；dim 不匹配 fail-fast。
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

import numpy as np

from prism.semantic import BGE_SMALL_ZH_DIM

__all__ = ["HnswlibVectorStore"]

log = logging.getLogger(__name__)

_DEFAULT_INITIAL_MAX_ELEMENTS: Final[int] = 10_000
_DEFAULT_EF_CONSTRUCTION: Final[int] = 200
_DEFAULT_M: Final[int] = 16
_DEFAULT_EF: Final[int] = 50
_REBUILD_DELETED_RATIO: Final[float] = 0.2
_FILTER_OVERFETCH_FACTOR: Final[int] = 10
_REBUILD_BATCH_SIZE: Final[int] = 1000


class HnswlibVectorStore:
    """hnswlib cosine 索引 + 逻辑删除集 + sidecar JSON 持久化。

    满足 :class:`~prism.vstore.VectorStore` 协议。

    Args:
        dim: 向量维度（默认 BGE-small-zh 512）
        path: 持久化路径前缀；实际写两个文件 ``{path}`` (hnswlib 索引) +
            ``{path}.meta.json`` (deleted/present 集合)；``None`` = 纯内存
        load: ``True`` 且 ``path`` 存在时自动加载
        initial_max_elements: hnswlib ``init_index`` 容量；超时自动 ``resize_index``
        ef_construction: 建索引时探索宽度；默认 200（hnswlib 推荐）
        M: 每节点连接数；默认 16（hnswlib 推荐 12-48）
        ef: 查询期探索宽度；默认 50；自动 ``max(ef, k)``
    """

    backend_name: Final[str] = "hnswlib"

    def __init__(
        self,
        *,
        dim: int = BGE_SMALL_ZH_DIM,
        path: str | os.PathLike[str] | None = None,
        load: bool = True,
        initial_max_elements: int = _DEFAULT_INITIAL_MAX_ELEMENTS,
        ef_construction: int = _DEFAULT_EF_CONSTRUCTION,
        M: int = _DEFAULT_M,
        ef: int = _DEFAULT_EF,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim 必须为正：got {dim}")
        if initial_max_elements <= 0:
            raise ValueError(
                f"initial_max_elements 必须为正：got {initial_max_elements}"
            )
        if ef_construction <= 0 or M <= 0 or ef <= 0:
            raise ValueError("ef_construction / M / ef 必须为正")

        # 延迟到 _init_index 真建索引（也用于 rebuild_from_iter 复用同套参数）
        self.dim: int = dim
        self._path: Path | None = Path(path) if path is not None else None
        self._ef_construction = ef_construction
        self._M = M
        self._ef = ef
        self._max_elements = initial_max_elements

        self._deleted_ids: set[int] = set()
        self._present_ids: set[int] = set()

        self._init_index(initial_max_elements)

        if load and self._path is not None and self._path.exists():
            self._load_from_disk()

    # ─── core operations ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._present_ids)

    def add(self, fact_id: int, vector: np.ndarray) -> None:
        if fact_id in self._present_ids:
            raise ValueError(
                f"fact_id {fact_id} 已存在（HnswlibVectorStore 选择抛错以暴露 bug）"
            )
        vec = self._validate_and_cast(vector)
        # 估算容量：present + 待处理；deleted 槽位通过 replace_deleted 复用
        # 但首次 add 一个全新 fact_id 时若总插入数 >= max_elements 仍需 resize
        # 用 get_current_count 作权威（含 deleted）
        if self._index.get_current_count() + 1 > self._max_elements:
            self._grow_capacity(self._max_elements * 2)

        replace = fact_id in self._deleted_ids
        self._index.add_items(
            vec.reshape(1, -1), np.asarray([fact_id], dtype=np.int64),
            replace_deleted=replace,
        )
        self._present_ids.add(fact_id)
        self._deleted_ids.discard(fact_id)

    def remove(self, fact_id: int) -> None:
        if fact_id not in self._present_ids:
            return  # 协议契约：不存在静默 noop
        self._index.mark_deleted(fact_id)
        self._present_ids.discard(fact_id)
        self._deleted_ids.add(fact_id)

    def update(self, fact_id: int, vector: np.ndarray) -> None:
        if fact_id not in self._present_ids:
            # 协议契约：不存在等同 add（含已删的 fact_id 复用）
            self.add(fact_id, vector)
            return
        vec = self._validate_and_cast(vector)
        # hnswlib 不支持原地更新；mark_deleted + replace_deleted=True 即覆盖
        self._index.mark_deleted(fact_id)
        # 由于 mark_deleted 已让节点变 deleted，replace_deleted=True 复用槽位
        self._index.add_items(
            vec.reshape(1, -1), np.asarray([fact_id], dtype=np.int64),
            replace_deleted=True,
        )
        # _present_ids 保持不变；_deleted_ids 不增（add_items 已重激活）

    def topk(
        self,
        query: np.ndarray,
        k: int,
        filter_fact_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        n_present = len(self._present_ids)
        if n_present == 0 or k <= 0:
            return []
        q = self._validate_and_cast(query)

        # 1) 决定召回数：有 filter 时按 k * factor 多取（Python 侧过滤）
        if filter_fact_ids is None:
            target = min(k, n_present)
        else:
            target = min(k * _FILTER_OVERFETCH_FACTOR, n_present)

        # 2) ef 必须 ≥ k（hnswlib 内部要求）
        eff_ef = max(self._ef, target)
        self._index.set_ef(eff_ef)

        try:
            labels, dists = self._index.knn_query(q.reshape(1, -1), k=target)
        except RuntimeError as e:
            # "Cannot return the results in a contiguous 2D array" → 索引候选不够
            log.debug("hnswlib knn_query 警告（candidates 不足）：%s", e)
            return []

        # hnswlib cosine 距离 = 1 - cos；分数 = 1 - dist
        out: list[tuple[int, float]] = []
        for lbl, dist in zip(labels[0].tolist(), dists[0].tolist(), strict=True):
            fid = int(lbl)
            if fid not in self._present_ids:
                continue  # 防御：mark_deleted 期间的过期结果
            if filter_fact_ids is not None and fid not in filter_fact_ids:
                continue
            out.append((fid, float(1.0 - dist)))
            if len(out) >= k:
                break
        return out

    def fetch(self, fact_ids: list[int]) -> dict[int, np.ndarray]:
        present = [fid for fid in fact_ids if fid in self._present_ids]
        if not present:
            return {}
        # hnswlib get_items 返回 list[list[float]]；统一转 np.array
        raw = self._index.get_items(present)
        arr = np.asarray(raw, dtype=np.float32)
        return {fid: arr[i].copy() for i, fid in enumerate(present)}

    def rebuild_from_iter(self, pairs: Iterable[tuple[int, np.ndarray]]) -> None:
        ids: list[int] = []
        vecs: list[np.ndarray] = []
        for fid, vec in pairs:
            ids.append(fid)
            vecs.append(self._validate_and_cast(vec))
        n = len(ids)
        if len(set(ids)) != n:
            raise ValueError("rebuild_from_iter: 输入含重复 fact_id")

        new_capacity = max(_DEFAULT_INITIAL_MAX_ELEMENTS, n * 2)
        self._max_elements = new_capacity
        self._init_index(new_capacity)
        self._present_ids = set()
        self._deleted_ids = set()

        # 分批写入（避免一次性 N×dim 大矩阵 RSS 峰值）
        for start in range(0, n, _REBUILD_BATCH_SIZE):
            end = min(start + _REBUILD_BATCH_SIZE, n)
            batch_vecs = np.stack(vecs[start:end])
            batch_ids = np.asarray(ids[start:end], dtype=np.int64)
            self._index.add_items(batch_vecs, batch_ids)
            self._present_ids.update(int(i) for i in batch_ids.tolist())

    def persist(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_idx = self._path.with_suffix(self._path.suffix + ".tmp")
        meta_path = self._path.with_suffix(self._path.suffix + ".meta.json")
        tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")

        # 1) 索引落盘 — hnswlib 自带 atomic-ish 写（不保证 POSIX 原子），手工
        # 走临时文件 + os.replace 兜底
        self._index.save_index(str(tmp_idx))
        os.replace(tmp_idx, self._path)

        # 2) sidecar JSON — hnswlib 不保留逻辑删除/活跃集合，必须自己存
        meta = {
            "dim": self.dim,
            "max_elements": self._max_elements,
            "ef_construction": self._ef_construction,
            "M": self._M,
            "ef": self._ef,
            "deleted_ids": sorted(self._deleted_ids),
            "present_ids": sorted(self._present_ids),
        }
        with open(tmp_meta, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, separators=(",", ":"))
        os.replace(tmp_meta, meta_path)

    def stats(self) -> dict[str, Any]:
        ntotal = len(self._present_ids)
        deleted = len(self._deleted_ids)
        denom = max(ntotal + deleted, 1)
        return {
            "backend_name": self.backend_name,
            "dim": self.dim,
            "ntotal": ntotal,
            "deleted_count": deleted,
            "effective_ntotal": ntotal,
            "index_count": int(self._index.get_current_count()),
            "max_elements": self._max_elements,
            "ef": self._ef,
            "ef_construction": self._ef_construction,
            "M": self._M,
            "dirty": (deleted / denom) > _REBUILD_DELETED_RATIO,
            "fallback_filter": True,  # 无 native predicate；用 Python 侧过滤
            "path": str(self._path) if self._path else None,
        }

    # ─── internals ───────────────────────────────────────────────────────

    def _init_index(self, max_elements: int) -> None:
        """新建一个空 hnswlib 索引（参数复用 __init__ 持的字段）。"""
        import hnswlib  # 延迟 import：可选依赖

        self._index = hnswlib.Index(space="cosine", dim=self.dim)
        self._index.init_index(
            max_elements=max_elements,
            ef_construction=self._ef_construction,
            M=self._M,
            allow_replace_deleted=True,
        )
        self._index.set_ef(self._ef)
        self._max_elements = max_elements

    def _grow_capacity(self, new_capacity: int) -> None:
        """容量不足时倍增。hnswlib ``resize_index`` 是原地扩容（保留向量）。"""
        self._index.resize_index(new_capacity)
        self._max_elements = new_capacity

    def _validate_and_cast(self, vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != self.dim:
            raise ValueError(
                f"vector shape mismatch: expected ({self.dim},), got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("vector contains NaN/Inf")
        return arr

    def _load_from_disk(self) -> None:
        assert self._path is not None
        meta_path = self._path.with_suffix(self._path.suffix + ".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"hnswlib 索引 {self._path} 存在但 sidecar {meta_path} 缺失；"
                f"无法恢复 deleted/present 集合"
            )
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        saved_dim = int(meta["dim"])
        if saved_dim != self.dim:
            raise ValueError(
                f"persisted dim {saved_dim} != configured dim {self.dim}"
            )
        self._max_elements = int(meta["max_elements"])
        self._ef_construction = int(meta["ef_construction"])
        self._M = int(meta["M"])
        self._ef = int(meta["ef"])
        # load_index 要求 max_elements 显式传入
        import hnswlib

        self._index = hnswlib.Index(space="cosine", dim=self.dim)
        self._index.load_index(
            str(self._path),
            max_elements=self._max_elements,
            allow_replace_deleted=True,
        )
        self._index.set_ef(self._ef)
        self._deleted_ids = set(int(i) for i in meta["deleted_ids"])
        self._present_ids = set(int(i) for i in meta["present_ids"])
