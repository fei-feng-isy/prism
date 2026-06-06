"""``LocalNumpyVectorStore`` -- 内存 numpy 矩阵 + ``.npz`` 持久化。

默认 backend（N <= 2000）。混合维度期也强制走本实现。
向量要求：L2 归一化 float32；NaN/Inf 拒收；dim 不匹配 fail-fast。
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

import numpy as np

from prism.semantic import BGE_SMALL_ZH_DIM

__all__ = ["LocalNumpyVectorStore"]

_DEFAULT_INITIAL_CAPACITY: Final[int] = 64
_GROWTH_FACTOR: Final[int] = 2


class LocalNumpyVectorStore:
    """内存 numpy 矩阵 + ``.npz`` 持久化实现。

    满足 :class:`~prism.vstore.VectorStore` 协议。

    Args:
        dim: 向量维度，默认 :data:`~prism.semantic.BGE_SMALL_ZH_DIM` (512)。
        path: ``.npz`` 持久化路径。``None`` 表示纯内存模式
            （:meth:`persist` 为 noop，测试与混合维度临时实例用）。
        load: ``True`` 且 ``path`` 存在时自动加载已存档状态。
        initial_capacity: 预分配行数，写满后倍增扩容。
        forced_local_due_to_mixed_dim: 由 backend 工厂设置；表示当前
            实例并非按 ``cfg.vector_store.backend`` 配置选出来，而是因混合维度期
            （``COUNT(DISTINCT embedding_model) > 1``）强制降级。:meth:`stats` 暴露
            此字段供 ``prism_admin(stats)`` 与日志读取。
    """

    backend_name: Final[str] = "local_numpy"

    def __init__(
        self,
        *,
        dim: int = BGE_SMALL_ZH_DIM,
        path: str | os.PathLike[str] | None = None,
        load: bool = True,
        initial_capacity: int = _DEFAULT_INITIAL_CAPACITY,
        forced_local_due_to_mixed_dim: bool = False,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim 必须为正：got {dim}")
        if initial_capacity <= 0:
            raise ValueError(f"initial_capacity 必须为正：got {initial_capacity}")

        self.dim: int = dim
        self._path: Path | None = Path(path) if path is not None else None
        self._matrix: np.ndarray = np.zeros((initial_capacity, dim), dtype=np.float32)
        self._fact_to_row: dict[int, int] = {}
        # 与 _matrix 活跃行一一对应；len(_row_to_fact) == _size
        self._row_to_fact: list[int] = []
        self._size: int = 0
        self._forced_local_due_to_mixed_dim: bool = forced_local_due_to_mixed_dim

        if load and self._path is not None and self._path.exists():
            self._load_from_disk()

    # ─── core operations ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._size

    def add(self, fact_id: int, vector: np.ndarray) -> None:
        if fact_id in self._fact_to_row:
            raise ValueError(
                f"fact_id {fact_id} 已存在（LocalNumpyVectorStore 选择抛错以暴露 bug）"
            )
        vec = self._validate_and_cast(vector)
        self._ensure_capacity(self._size + 1)
        row = self._size
        self._matrix[row] = vec
        self._fact_to_row[fact_id] = row
        self._row_to_fact.append(fact_id)
        self._size += 1

    def remove(self, fact_id: int) -> None:
        row = self._fact_to_row.pop(fact_id, None)
        if row is None:
            return  # 协议契约：不存在静默 noop
        last_row = self._size - 1
        if row != last_row:
            # swap-with-last：避免 O(N) memmove
            self._matrix[row] = self._matrix[last_row]
            last_fact_id = self._row_to_fact[last_row]
            self._row_to_fact[row] = last_fact_id
            self._fact_to_row[last_fact_id] = row
        self._row_to_fact.pop()
        self._matrix[last_row].fill(0.0)  # 清空回收行，便于内存检视与持久化精简
        self._size -= 1

    def update(self, fact_id: int, vector: np.ndarray) -> None:
        row = self._fact_to_row.get(fact_id)
        if row is None:
            # 协议契约：不存在等同 add
            self.add(fact_id, vector)
            return
        vec = self._validate_and_cast(vector)
        self._matrix[row] = vec

    def topk(
        self,
        query: np.ndarray,
        k: int,
        filter_fact_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if self._size == 0 or k <= 0:
            return []
        q = self._validate_and_cast(query)

        if filter_fact_ids is None:
            active_rows = np.arange(self._size, dtype=np.int64)
        else:
            active_rows = np.fromiter(
                (r for r, fid in enumerate(self._row_to_fact) if fid in filter_fact_ids),
                dtype=np.int64,
            )
            if active_rows.size == 0:
                return []

        # 双方均 L2 归一化 → 点积即 cosine
        scores = self._matrix[active_rows] @ q

        n = int(scores.size)
        if k >= n:
            order = np.argsort(-scores, kind="stable")
        else:
            # argpartition O(N) 取 top-k 候选；候选内再 argsort 拿严格降序
            part = np.argpartition(-scores, k)[:k]
            order = part[np.argsort(-scores[part], kind="stable")]

        return [
            (self._row_to_fact[int(active_rows[i])], float(scores[i]))
            for i in order
        ]

    def fetch(self, fact_ids: list[int]) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for fid in fact_ids:
            row = self._fact_to_row.get(fid)
            if row is not None:
                # copy：避免调用方就地修改污染内部 matrix
                out[fid] = self._matrix[row].copy()
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

        capacity = max(_DEFAULT_INITIAL_CAPACITY, n)
        self._matrix = np.zeros((capacity, self.dim), dtype=np.float32)
        if n > 0:
            self._matrix[:n] = np.stack(vecs)
        self._fact_to_row = {fid: i for i, fid in enumerate(ids)}
        self._row_to_fact = list(ids)
        self._size = n

    def persist(self) -> None:
        if self._path is None:
            return  # in-memory mode（测试 / 混合维度临时实例）
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        # 通过 open() 控制后缀（np.savez 否则会自动追加 .npz）
        with open(tmp_path, "wb") as fh:
            np.savez(
                fh,
                matrix=self._matrix[: self._size],
                fact_ids=np.asarray(self._row_to_fact, dtype=np.int64),
                dim=np.int64(self.dim),
            )
        os.replace(tmp_path, self._path)  # POSIX 原子 rename

    def stats(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "dim": self.dim,
            "ntotal": self._size,
            "capacity": int(self._matrix.shape[0]),
            "memory_bytes": int(self._matrix.nbytes),
            "path": str(self._path) if self._path else None,
            "forced_local_due_to_mixed_dim": self._forced_local_due_to_mixed_dim,
        }

    # ─── internals ───────────────────────────────────────────────────────

    def _validate_and_cast(self, vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != self.dim:
            raise ValueError(
                f"vector shape mismatch: expected ({self.dim},), got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("vector contains NaN/Inf")
        return arr

    def _ensure_capacity(self, needed: int) -> None:
        current = int(self._matrix.shape[0])
        if needed <= current:
            return
        new_capacity = current
        while new_capacity < needed:
            new_capacity *= _GROWTH_FACTOR
        new_matrix = np.zeros((new_capacity, self.dim), dtype=np.float32)
        new_matrix[: self._size] = self._matrix[: self._size]
        self._matrix = new_matrix

    def _load_from_disk(self) -> None:
        assert self._path is not None
        with np.load(self._path, allow_pickle=False) as data:
            matrix = np.asarray(data["matrix"], dtype=np.float32)
            fact_ids_arr = np.asarray(data["fact_ids"], dtype=np.int64)
            saved_dim = int(data["dim"])
        if saved_dim != self.dim:
            raise ValueError(
                f"persisted dim {saved_dim} != configured dim {self.dim}"
            )
        n = int(matrix.shape[0])
        if fact_ids_arr.shape[0] != n:
            raise ValueError(
                f"persisted matrix rows {n} != fact_ids length {fact_ids_arr.shape[0]}"
            )
        capacity = max(_DEFAULT_INITIAL_CAPACITY, n)
        self._matrix = np.zeros((capacity, self.dim), dtype=np.float32)
        if n > 0:
            self._matrix[:n] = matrix
        ids = [int(fid) for fid in fact_ids_arr.tolist()]
        self._fact_to_row = {fid: i for i, fid in enumerate(ids)}
        self._row_to_fact = ids
        self._size = n
