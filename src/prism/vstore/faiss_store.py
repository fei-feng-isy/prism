"""``FaissVectorStore`` -- faiss ``IndexFlatIP`` + ``IndexIDMap2`` ANN backend。

适用规模：``N > 100k``。支持原生 ``remove_ids``、``reconstruct``、
``write_index`` / ``read_index`` 持久化。当前仅支持 ``index_type="flat"``。

向量要求：L2 归一化 float32；NaN/Inf 拒收；dim 不匹配 fail-fast。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

import numpy as np

from prism.semantic import BGE_SMALL_ZH_DIM

__all__ = ["FaissVectorStore"]

log = logging.getLogger(__name__)

_FILTER_OVERFETCH_FACTOR: Final[int] = 10
_REBUILD_BATCH_SIZE: Final[int] = 1000
_SUPPORTED_INDEX_TYPES: Final[frozenset[str]] = frozenset({"flat"})


class FaissVectorStore:
    """faiss ``IndexFlatIP`` + ``IndexIDMap2`` 持久化索引。

    满足 :class:`~prism.vstore.VectorStore` 协议。

    Args:
        dim: 向量维度（默认 BGE-small-zh 512）
        path: 持久化路径；``None`` = 纯内存。faiss ``write_index`` / ``read_index``
            原生持久化 fact_id 与向量，不需要 sidecar
        load: ``True`` 且 ``path`` 存在时自动加载
        index_type: 当前只支持 ``"flat"``（精确 IP）；IVF / HNSW 由工厂
            auto 路径按 ntotal 阈值启用
    """

    backend_name: Final[str] = "faiss"

    def __init__(
        self,
        *,
        dim: int = BGE_SMALL_ZH_DIM,
        path: str | os.PathLike[str] | None = None,
        load: bool = True,
        index_type: str = "flat",
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim 必须为正：got {dim}")
        if index_type not in _SUPPORTED_INDEX_TYPES:
            raise ValueError(
                f"index_type={index_type!r} 不支持；当前仅 "
                f"{sorted(_SUPPORTED_INDEX_TYPES)}（IVF/HNSW 由工厂路径上线）"
            )

        self.dim: int = dim
        self._path: Path | None = Path(path) if path is not None else None
        self._index_type: str = index_type
        self._present_ids: set[int] = set()

        self._init_index()

        if load and self._path is not None and self._path.exists():
            self._load_from_disk()

    # ─── core operations ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._present_ids)

    def add(self, fact_id: int, vector: np.ndarray) -> None:
        if fact_id in self._present_ids:
            raise ValueError(
                f"fact_id {fact_id} 已存在（FaissVectorStore 选择抛错以暴露 bug）"
            )
        vec = self._validate_and_cast(vector)
        self._index.add_with_ids(
            vec.reshape(1, -1),
            np.asarray([fact_id], dtype=np.int64),
        )
        self._present_ids.add(fact_id)

    def remove(self, fact_id: int) -> None:
        if fact_id not in self._present_ids:
            return  # 协议契约：不存在静默 noop
        import faiss

        sel = faiss.IDSelectorBatch(np.asarray([fact_id], dtype=np.int64))
        self._index.remove_ids(sel)
        self._present_ids.discard(fact_id)

    def update(self, fact_id: int, vector: np.ndarray) -> None:
        if fact_id not in self._present_ids:
            # 协议契约：不存在等同 add
            self.add(fact_id, vector)
            return
        # faiss 不支持原地更新；remove + add 满足契约
        self.remove(fact_id)
        self.add(fact_id, vector)

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

        # 决定召回数：有 filter 时按 max(k×10, len(filter)) 多取
        if filter_fact_ids is None:
            target = min(k, n_present)
        else:
            target = min(
                max(k * _FILTER_OVERFETCH_FACTOR, len(filter_fact_ids)),
                n_present,
            )

        scores, ids = self._index.search(q.reshape(1, -1), target)

        # faiss search: 候选不足时返回 -1 填充，必须过滤
        out: list[tuple[int, float]] = []
        for fid_raw, score in zip(ids[0].tolist(), scores[0].tolist(), strict=True):
            fid = int(fid_raw)
            if fid == -1:
                continue
            if fid not in self._present_ids:
                continue  # 防御：刚 remove 还未同步
            if filter_fact_ids is not None and fid not in filter_fact_ids:
                continue
            # IP metric + L2 归一化 == cosine；score 直接是 cosine
            out.append((fid, float(score)))
            if len(out) >= k:
                break
        return out

    def fetch(self, fact_ids: list[int]) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for fid in fact_ids:
            if fid not in self._present_ids:
                continue
            try:
                vec = self._index.reconstruct(int(fid))
            except RuntimeError:
                # 防御：present_ids 与 faiss 内部 id_map 不一致（理论上不应发生）
                log.debug("reconstruct(%d) 失败 — 跳过", fid)
                continue
            # reconstruct 返回的是 numpy view（copy 防止调用方污染索引）
            out[fid] = np.asarray(vec, dtype=np.float32).copy()
        return out

    def rebuild_from_iter(self, pairs: Iterable[tuple[int, np.ndarray]]) -> None:
        ids: list[int] = []
        vecs: list[np.ndarray] = []
        for fid, vec in pairs:
            ids.append(fid)
            vecs.append(self._validate_and_cast(vec))
        n = len(ids)
        if len(set(ids)) != n:
            raise ValueError("rebuild_from_iter: 输入含重复 fact_id")

        self._init_index()
        self._present_ids = set()

        # 分批写入（避免一次性 N×dim 大矩阵 RSS 峰值）
        for start in range(0, n, _REBUILD_BATCH_SIZE):
            end = min(start + _REBUILD_BATCH_SIZE, n)
            batch_vecs = np.stack(vecs[start:end])
            batch_ids = np.asarray(ids[start:end], dtype=np.int64)
            self._index.add_with_ids(batch_vecs, batch_ids)
            self._present_ids.update(int(i) for i in batch_ids.tolist())

    def persist(self) -> None:
        if self._path is None:
            return
        import faiss

        self._path.parent.mkdir(parents=True, exist_ok=True)
        # faiss 的 write_index 非 POSIX 原子；走临时文件 + os.replace 兜底
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        faiss.write_index(self._index, str(tmp_path))
        os.replace(tmp_path, self._path)

    def stats(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "dim": self.dim,
            "ntotal": len(self._present_ids),
            "index_ntotal": int(self._index.ntotal),
            "index_type": self._index_type,
            "fallback_filter": True,  # 无 native predicate；用 Python 侧过滤
            "path": str(self._path) if self._path else None,
        }

    # ─── internals ───────────────────────────────────────────────────────

    def _init_index(self) -> None:
        """新建一个空 faiss 索引（按 index_type 选基础类）。"""
        import faiss

        if self._index_type == "flat":
            base = faiss.IndexFlatIP(self.dim)
        else:  # pragma: no cover - guarded by __init__ validation
            raise ValueError(f"unsupported index_type: {self._index_type}")
        # IndexIDMap2 允许任意 int64 fact_id 作主键 + 支持 reconstruct(id)
        self._index = faiss.IndexIDMap2(base)

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
        import faiss

        loaded = faiss.read_index(str(self._path))
        if not isinstance(loaded, faiss.IndexIDMap2):
            raise ValueError(
                f"persisted faiss index 不是 IndexIDMap2：got {type(loaded).__name__}"
            )
        if loaded.d != self.dim:
            raise ValueError(
                f"persisted dim {loaded.d} != configured dim {self.dim}"
            )
        self._index = loaded
        # 从 id_map 还原 _present_ids（faiss 内部 Int64Vector）
        id_map = self._index.id_map
        self._present_ids = {int(id_map.at(i)) for i in range(id_map.size())}
