"""Prism 镜像层 — 内置 memory write → Prism 自动同步入库。

:class:`PrismMirror` 暴露 :meth:`on_memory_write` 单一入口，按 ``action``
分派到 add / replace / remove：实体抽取 → HRR 编码 → SQLite 写入 → Bank 同步。

关键行为：
    - 异常吞掉 + WARN：mirror 失败不阻塞调用方写入。
    - 幂等：``facts.content UNIQUE``，重复 add 返回 ``MirrorResult(is_new=False)``。
    - bank 纯内存，可由 ``calibrate(category, all_active_facts)`` 重建。
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .config import EntitiesConfig
from .entities.regex_extractor import extract_entities
from .hrr import IncrementalBank, atom, bind, bundle
from .semantic import SemanticUnavailable

if TYPE_CHECKING:
    from .enrichment import EnrichmentQueue
    from .hrr import RebuildDebouncer
    from .semantic import SemanticBackend
    from .vstore import VectorStore

log = logging.getLogger(__name__)

__all__ = ["MIRROR_SOURCE_BUILTIN", "MIRROR_SOURCE_USER", "MirrorResult", "PrismMirror"]


# 固定 ROLE atom 名（跨进程确定性）
_ROLE_CONTENT_NAME: Final[str] = "__role_content__"
_ROLE_ENTITY_NAME: Final[str] = "__role_entity__"

# mirror_source 标签
MIRROR_SOURCE_BUILTIN: Final[str] = "builtin_memory"
MIRROR_SOURCE_USER: Final[str] = "user"

# archive_reason 取值
_ARCHIVE_REASON_REPLACED: Final[str] = "replaced"
_ARCHIVE_REASON_MANUAL: Final[str] = "manual"
_ARCHIVE_REASON_BUILTIN_REMOVED: Final[str] = "builtin_removed"
_ARCHIVE_REASON_GHOST: Final[str] = "ghost"


@dataclass(frozen=True, slots=True)
class MirrorResult:
    """单次 mirror 写入的结果。

    Attributes:
        fact_id: facts 主键
        is_new: ``True`` 新插入；``False`` content UNIQUE 命中现有 fact（幂等）
        entities: 抽取出的实体名元组（按 ``ExtractedEntity`` 默认排序）
        category: 实际写入的分类（含 metadata 覆盖后）
        semantic_indexed: ``True`` 时 ``facts.semantic_vector`` 已写入 +
            ``vstore.add(fact_id, vec)`` 成功。``False`` 表示无 semantic
            注入、降级 backend、encode 异常被吞、或 ``is_new=False`` 跳过。
    """

    fact_id: int
    is_new: bool
    entities: tuple[str, ...]
    category: str
    semantic_indexed: bool = False


class PrismMirror:
    """内置 memory write → SQLite + HRR Bank 镜像。

    Args:
        conn: 已 ``init_schema`` 的 SQLite 连接（调用方负责生命周期）
        bank: :class:`IncrementalBank` 实例（``bank.dim`` 必须等于 ``hrr_dim``）
        hrr_dim: HRR 向量维度，默认 1024
        default_category: ``metadata`` 未提供 ``category`` 时使用
        semantic / vstore: 同时注入用于 semantic 索引；同时省略走纯 HRR
        enrichment_queue: 可选，Stage 2 异步实体富化队列
        entities_config: 可选；控制富化触发阈值
        rebuild_debouncer: 可选；注入后 ``mirror_remove`` 在 ``bank.remove``
            后调 ``schedule()``，去抖窗口合并连续 remove → 单次 calibrate

    Raises:
        ValueError: ``bank.dim != hrr_dim``
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        bank: IncrementalBank,
        *,
        hrr_dim: int = 1024,
        default_category: str = "general",
        semantic: SemanticBackend | None = None,
        vstore: VectorStore | None = None,
        enrichment_queue: EnrichmentQueue | None = None,
        entities_config: EntitiesConfig | None = None,
        rebuild_debouncer: RebuildDebouncer | None = None,
    ) -> None:
        if bank.dim != hrr_dim:
            raise ValueError(f"bank.dim={bank.dim} 与 hrr_dim={hrr_dim} 不一致")
        if (semantic is None) != (vstore is None):
            raise ValueError(
                "semantic 与 vstore 必须同时提供或同时省略 — 单独注入一侧没有意义"
            )
        if semantic is not None and vstore is not None and semantic.dim != vstore.dim:
            raise ValueError(
                f"semantic.dim={semantic.dim} 与 vstore.dim={vstore.dim} 不一致"
            )
        self._conn = conn
        self._bank = bank
        self._dim: Final[int] = hrr_dim
        self._default_category: Final[str] = default_category
        self._semantic = semantic
        self._vstore = vstore
        self._enrichment_queue = enrichment_queue
        self._entities_config: Final[EntitiesConfig] = (
            entities_config if entities_config is not None else EntitiesConfig()
        )
        self._rebuild_debouncer = rebuild_debouncer
        # 预算 ROLE atom（构造一次复用）
        self._role_content = atom(_ROLE_CONTENT_NAME, dim=hrr_dim)
        self._role_entity = atom(_ROLE_ENTITY_NAME, dim=hrr_dim)

    # ─── 公共入口 ────────────────────────────────────────────────────────

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> MirrorResult | None:
        """处理一次内置 memory 写入。

        Args:
            action: ``'add'`` / ``'replace'`` / ``'remove'``
            target: 写入目标（'memory' / 'user' 等），写入 ``facts.mirror_target``
            content: fact 内容
            metadata: 可选元数据；支持 ``category`` 覆盖默认分类

        Returns:
            ``MirrorResult`` 成功；``None`` 表示异常被吞（WARN 日志已记）、
            未支持的 action、或 content 为空。
        """
        try:
            if action == "add":
                return self.mirror_add(
                    content,
                    metadata=metadata,
                    source=MIRROR_SOURCE_BUILTIN,
                    target=target,
                )
            if action == "replace":
                return self.mirror_replace(
                    content,
                    metadata=metadata,
                    source=MIRROR_SOURCE_BUILTIN,
                    target=target,
                )
            if action == "remove":
                return self.mirror_remove(
                    content=content,
                    metadata=metadata,
                )
            log.debug(
                "Prism mirror skipping unknown action=%s", action,
            )
            return None
        except Exception as e:
            # mirror 失败仅 WARN，不阻塞调用方
            log.warning(
                "Prism mirror failed: action=%s target=%s err=%s",
                action,
                target,
                e,
            )
            return None

    # ─── add 路径 ────────────────────────────────────────────────────────

    def mirror_add(
        self,
        content: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        source: str = MIRROR_SOURCE_BUILTIN,
        target: str | None = None,
    ) -> MirrorResult | None:
        """单条 fact 写入内核：实体抽取 → HRR 编码 → SQLite → Bank。

        与 :meth:`on_memory_write` 不同：异常向上抛，调用方负责处理。

        Args:
            content: fact 内容（会 strip）
            metadata: 可选；支持 ``category`` 覆盖默认分类
            source: 写入来源标签（``builtin_memory`` / ``user`` / ``async_extract``）
            target: 写入目标（``memory`` / ``user``）；非镜像场景可为 ``None``

        Returns:
            ``MirrorResult`` 成功；``None`` 表示 content 为空（不视为异常）
        """
        meta = metadata or {}
        normalized = content.strip()
        if not normalized:
            log.debug("Prism mirror_add: empty content, skip")
            return None

        category = str(meta.get("category", self._default_category))

        # 1. 实体抽取（已预加载 jieba）
        entities = extract_entities(normalized)
        entity_names: tuple[str, ...] = tuple(e.name for e in entities)

        # 2. HRR fact_vector
        fact_vector = self._encode_fact(normalized, entity_names)

        # 3. SQLite 写入（单事务）
        fact_id, is_new = self._persist(
            content=normalized,
            category=category,
            source=source,
            target=target,
            fact_vector=fact_vector,
            entities=entities,
        )

        # 4. Bank 增量（事务外；幂等命中跳过避免双倍计数）
        if is_new:
            self._bank.add(category, fact_vector)

        # 5. semantic 索引（新插入 + 注入了 backend 才走）
        semantic_indexed = False
        if is_new:
            semantic_indexed = self._index_semantic(fact_id, normalized)

        # 6. 异步实体富化触发（新插入且 Stage 1 实体不足时入队）
        if is_new:
            self._maybe_enqueue_enrichment(fact_id, len(entity_names))

        return MirrorResult(
            fact_id=fact_id,
            is_new=is_new,
            entities=entity_names,
            category=category,
            semantic_indexed=semantic_indexed,
        )

    # ─── replace 路径 ────────────────────────────────────────────────────

    def mirror_replace(
        self,
        content: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        source: str = MIRROR_SOURCE_BUILTIN,
        target: str | None = None,
    ) -> MirrorResult | None:
        """归档旧 fact + 写入新 fact + 建立 supersedes 链。

        旧 fact 定位优先级：

        1. ``metadata['supersedes_id']: int`` — 显式 fact_id 引用（最可靠）
        2. ``metadata['old_content']: str`` — 按完整 content SELECT 一次

        二者皆缺 → WARN 返回 ``None``。

        Args:
            content: 新 fact 内容（strip）
            metadata: 必含 ``supersedes_id`` 或 ``old_content`` 之一；支持
                ``category`` 覆盖
            source / target: 同 :meth:`mirror_add`

        Returns:
            ``MirrorResult`` 成功；``None`` 表示空 content / 旧 fact 未找到 /
            未提供 supersedes_id 与 old_content。

            新旧 content 完全相同的退化场景：no-op，返回旧 fact 的
            ``MirrorResult(is_new=False)``。
        """
        meta = metadata or {}
        normalized = content.strip()
        if not normalized:
            log.debug("Prism mirror_replace: empty content, skip")
            return None

        old_fact_id = self._locate_old_fact(meta)
        if old_fact_id is None:
            log.warning(
                "mirror_replace: metadata 需要 'supersedes_id' 或 'old_content' 之一"
            )
            return None

        # 校验旧 fact 存在 + 取旧 content（用于同内容 no-op 检测）
        old_row = self._conn.execute(
            "SELECT content, status FROM facts WHERE fact_id = ?",
            (old_fact_id,),
        ).fetchone()
        if old_row is None:
            log.warning("mirror_replace: 旧 fact_id=%s 不存在", old_fact_id)
            return None
        old_content = str(old_row["content"])
        old_status = str(old_row["status"])

        category = str(meta.get("category", self._default_category))

        # 退化：新旧 content 相同 → 跳过 archive + insert，回报旧 fact
        if normalized == old_content:
            log.debug(
                "mirror_replace: 新旧 content 相同 (fact_id=%s)，no-op",
                old_fact_id,
            )
            return MirrorResult(
                fact_id=old_fact_id,
                is_new=False,
                entities=(),
                category=category,
                semantic_indexed=False,
            )

        entities = extract_entities(normalized)
        entity_names: tuple[str, ...] = tuple(e.name for e in entities)
        fact_vector = self._encode_fact(normalized, entity_names)

        new_fact_id, is_new = self._persist(
            content=normalized,
            category=category,
            source=source,
            target=target,
            fact_vector=fact_vector,
            entities=entities,
            supersedes_id=old_fact_id,
            archive_old_if_active=(old_status == "active"),
        )

        if is_new:
            self._bank.add(category, fact_vector)

        semantic_indexed = False
        if is_new:
            semantic_indexed = self._index_semantic(new_fact_id, normalized)

        if is_new:
            self._maybe_enqueue_enrichment(new_fact_id, len(entity_names))

        return MirrorResult(
            fact_id=new_fact_id,
            is_new=is_new,
            entities=entity_names,
            category=category,
            semantic_indexed=semantic_indexed,
        )

    # ─── remove 路径 ─────────────────────────────────────────────────────

    def mirror_remove(
        self,
        content: str | None = None,
        *,
        fact_id: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        reason: str = _ARCHIVE_REASON_MANUAL,
    ) -> MirrorResult | None:
        """软删除 fact：status='archived' + bank.remove + 去抖 calibrate。

        定位优先级：

        1. ``fact_id`` 关键字参数（显式 ID）
        2. ``metadata['fact_id']: int``
        3. ``content`` 位置参数 / ``metadata['content']`` 按 content SELECT
        4. ``metadata['old_content']: str``（与 replace 一致，便于 builtin 镜像）

        Args:
            content: 按 content 定位（与 ``mirror_add`` 参数对称，便于
                ``on_memory_write(remove)`` 路径透传）
            fact_id: 显式 fact_id（最可靠，优先于 content）
            metadata: 可选；支持 ``fact_id`` / ``old_content`` / ``content`` /
                ``reason`` 覆盖
            reason: ``archive_reason`` 列写入值，默认 ``'manual'``。常用：
                ``'manual'``（用户显式删）、``'builtin_removed'``（内置 memory
                镜像反向同步）、``'ghost'``（cron 兜底）

        Returns:
            ``MirrorResult(is_new=False, ...)`` 成功（无论 fact 是刚归档还是
            已经 archived）；``None`` 未找到 / 无定位信息 / 空 content
        """
        meta = metadata or {}
        target_id = self._locate_for_remove(content, fact_id, meta)
        if target_id is None:
            log.warning(
                "mirror_remove: 无法定位 fact（需要 fact_id / metadata.fact_id / "
                "content / metadata.old_content）",
            )
            return None

        row = self._conn.execute(
            "SELECT content, category, status, hrr_vector "
            "FROM facts WHERE fact_id = ?",
            (target_id,),
        ).fetchone()
        if row is None:
            log.warning("mirror_remove: fact_id=%s 不存在", target_id)
            return None

        existing_status = str(row["status"])
        category = str(row["category"]) if row["category"] is not None else self._default_category
        actual_reason = str(meta.get("reason", reason))

        # 已 archived：no-op（不动 archive_reason，不重启 debouncer）
        if existing_status != "active":
            log.debug(
                "mirror_remove: fact_id=%s 已 %s，no-op", target_id, existing_status
            )
            return MirrorResult(
                fact_id=target_id,
                is_new=False,
                entities=(),
                category=category,
                semantic_indexed=False,
            )

        with _txn(self._conn):
            self._conn.execute(
                "UPDATE facts SET status = 'archived', "
                "archived_at = CURRENT_TIMESTAMP, archive_reason = ? "
                "WHERE fact_id = ? AND status = 'active'",
                (actual_reason, target_id),
            )

        # 清队列条目（避免 worker 处理幽灵 fact；pop_next 的 active guard 是兜底）
        if self._enrichment_queue is not None:
            try:
                self._enrichment_queue.mark_done(target_id)
            except Exception as e:
                log.warning(
                    "mark_done on remove fact_id=%s 失败：%s", target_id, e
                )

        # bank.remove + 去抖 calibrate（注入了 debouncer 才走）
        if self._rebuild_debouncer is not None and row["hrr_vector"] is not None:
            try:
                vec = np.frombuffer(row["hrr_vector"], dtype=np.float64)
                if vec.shape == (self._dim,):
                    self._bank.remove(category, vec)
                else:
                    log.warning(
                        "mirror_remove: hrr_vector shape 异常 fact_id=%s",
                        target_id,
                    )
            except Exception as e:
                # bank.remove 抛（dim 不匹配 / category 不存在）— 不阻塞 DB 归档
                log.warning(
                    "bank.remove 失败 fact_id=%s category=%s: %s",
                    target_id, category, e,
                )
            try:
                self._rebuild_debouncer.schedule()
            except Exception as e:
                log.warning(
                    "rebuild_debouncer.schedule 失败 fact_id=%s: %s",
                    target_id, e,
                )

        return MirrorResult(
            fact_id=target_id,
            is_new=False,
            entities=(),
            category=category,
            semantic_indexed=False,
        )

    def restore_vectors(
        self,
        fact_id: int,
        category: str,
        hrr_blob: bytes | None,
        semantic_blob: bytes | None,
    ) -> None:
        """恢复归档 fact 的 HRR bank 与 vstore 向量（供 FactService.restore 调用）。"""
        if hrr_blob is not None:
            try:
                vec = np.frombuffer(hrr_blob, dtype=np.float64)
                if vec.shape == (self._dim,):
                    self._bank.add(category, vec)
                    if self._rebuild_debouncer is not None:
                        self._rebuild_debouncer.schedule()
                else:
                    log.warning(
                        "restore: hrr_vector shape 异常 fact_id=%s shape=%s",
                        fact_id, vec.shape,
                    )
            except Exception as e:
                log.warning("restore: bank.add 失败 fact_id=%s: %s", fact_id, e)

        if semantic_blob is not None and self._vstore is not None:
            try:
                vec = np.frombuffer(semantic_blob, dtype=np.float32)
                if vec.shape == (self._vstore.dim,):
                    self._vstore.add(fact_id, vec)
                else:
                    log.warning(
                        "restore: semantic_vector shape 异常 fact_id=%s shape=%s",
                        fact_id, vec.shape,
                    )
            except Exception as e:
                log.warning("restore: vstore.add 失败 fact_id=%s: %s", fact_id, e)

    def _locate_for_remove(
        self,
        content: str | None,
        fact_id: int | None,
        meta: Mapping[str, Any],
    ) -> int | None:
        # 1) 显式 fact_id 关键字参数
        if isinstance(fact_id, int) and fact_id > 0:
            return fact_id
        # 2) metadata.fact_id
        mid = meta.get("fact_id")
        if isinstance(mid, int) and mid > 0:
            return mid
        # 3) content 位置参数 / metadata.content
        c = content if content is not None else meta.get("content")
        # 4) metadata.old_content 兜底
        if c is None:
            c = meta.get("old_content")
        if isinstance(c, str) and c.strip():
            row = self._conn.execute(
                "SELECT fact_id FROM facts WHERE content = ?",
                (c.strip(),),
            ).fetchone()
            if row is not None:
                return int(row["fact_id"])
        return None

    def archive_ghost_facts(
        self,
        is_alive: Callable[[str], bool],
        *,
        mirror_source: str = MIRROR_SOURCE_BUILTIN,
        reason: str = _ARCHIVE_REASON_GHOST,
        batch_size: int = 500,
    ) -> int:
        """扫指定 mirror_source 的 active fact，对每条调 ``is_alive(content)``；
        False 即归档。``is_alive`` 抛异常的 fact 跳过不阻塞批次。

        Args:
            is_alive: ``(content: str) -> bool``；True 保留，False 归档
            mirror_source: 只扫指定来源的 fact，默认 ``'builtin_memory'``
            reason: 归档时写入的 ``archive_reason``，默认 ``'ghost'``
            batch_size: 单批 SELECT 上限

        Returns:
            本次归档的 fact 数量
        """
        archived = 0
        rows = self._conn.execute(
            "SELECT fact_id, content FROM facts "
            "WHERE status = 'active' AND mirror_source = ? "
            "ORDER BY fact_id LIMIT ?",
            (mirror_source, batch_size),
        ).fetchall()
        for row in rows:
            fid = int(row["fact_id"])
            content = str(row["content"])
            try:
                alive = bool(is_alive(content))
            except Exception as e:
                log.warning(
                    "archive_ghost_facts: is_alive 抛 fact_id=%s: %s — 跳过",
                    fid, e,
                )
                continue
            if alive:
                continue
            result = self.mirror_remove(fact_id=fid, reason=reason)
            if result is not None:
                archived += 1
        return archived

    def _locate_old_fact(self, meta: Mapping[str, Any]) -> int | None:
        """按 metadata 定位待替换 fact。无匹配返回 ``None``。"""
        sid = meta.get("supersedes_id")
        if isinstance(sid, int) and sid > 0:
            return sid
        old = meta.get("old_content")
        if isinstance(old, str) and old.strip():
            row = self._conn.execute(
                "SELECT fact_id FROM facts WHERE content = ?",
                (old.strip(),),
            ).fetchone()
            if row is not None:
                return int(row["fact_id"])
        return None

    def _persist(
        self,
        *,
        content: str,
        category: str,
        source: str,
        target: str | None,
        fact_vector: np.ndarray,
        entities: list,
        supersedes_id: int | None = None,
        archive_old_if_active: bool = False,
    ) -> tuple[int, bool]:
        """事务内写 facts + entities + fact_entities；返回 (fact_id, is_new)。

        - content UNIQUE 冲突走 ON CONFLICT IGNORE，定位现有 fact_id 返回
        - 已存在的 fact 不更新 entity 链接（避免重复写入路径里污染历史关联）
        - ``supersedes_id`` 非空：新 fact 的 ``supersedes_id`` 列写入旧 fact_id；
          若 ``archive_old_if_active=True``，同一事务内将旧 fact 标记为
          ``status='archived'``、``archived_at=CURRENT_TIMESTAMP``、
          ``archive_reason='replaced'``。两步同事务保证 supersedes 链不出现
          "新 fact 已写 + 旧 fact 仍 active" 的中间态。
        """
        with _txn(self._conn):
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO facts "
                "(content, category, hrr_vector, mirror_source, mirror_target, supersedes_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    content,
                    category,
                    fact_vector.tobytes(),
                    source,
                    target,
                    supersedes_id,
                ),
            )
            if cur.rowcount == 0:
                # content UNIQUE 命中现有
                row = self._conn.execute(
                    "SELECT fact_id FROM facts WHERE content = ?", (content,)
                ).fetchone()
                return int(row["fact_id"]), False

            assert cur.lastrowid is not None
            fact_id = int(cur.lastrowid)

            # entities + fact_entities
            for ent in entities:
                self._conn.execute(
                    "INSERT OR IGNORE INTO entities "
                    "(name, entity_type, extraction_method) VALUES (?, ?, ?)",
                    (ent.name, ent.entity_type, ent.method),
                )
                ent_row = self._conn.execute(
                    "SELECT entity_id FROM entities WHERE name = ?", (ent.name,)
                ).fetchone()
                # ent_row 一定存在：上面 INSERT OR IGNORE 后必能 SELECT 到
                self._conn.execute(
                    "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                    (fact_id, int(ent_row["entity_id"])),
                )

            # 归档旧 fact（replace 路径）
            if supersedes_id is not None and archive_old_if_active:
                self._conn.execute(
                    "UPDATE facts SET status = 'archived', "
                    "archived_at = CURRENT_TIMESTAMP, archive_reason = ? "
                    "WHERE fact_id = ? AND status = 'active'",
                    (_ARCHIVE_REASON_REPLACED, supersedes_id),
                )

            return fact_id, True

    # ─── 异步富化（钩子）────────────────────────────────────────────────

    def _maybe_enqueue_enrichment(self, fact_id: int, entity_count: int) -> None:
        """触发条件：注入了队列 + ``auto_enrich`` + Stage 1 实体数 < 阈值。"""
        if self._enrichment_queue is None:
            return
        if not self._entities_config.auto_enrich:
            return
        if entity_count >= self._entities_config.stage1_min_entities:
            return
        try:
            self._enrichment_queue.enqueue(fact_id)
        except Exception as e:
            # 富化是最佳努力 — 入队失败不阻塞主写入路径
            log.warning(
                "enrichment enqueue 失败 fact_id=%s: %s", fact_id, e
            )

    def enrichment_merge(self, fact_id: int, entity_names: list[str]) -> None:
        """worker `merge_callback` 入口：把 LLM 抽出的实体合并到 entities + fact_entities。

        - 去重 + strip + 过滤空白 / < 2 字符（与 :func:`parse_llm_entities`
          的输出格式一致，但二次过滤兜底）
        - INSERT OR IGNORE 两表，幂等
        - ``entity_type='llm_extracted'`` / ``extraction_method='llm'`` 用于区分
          Stage 1 (jieba/regex) 与 Stage 2 (LLM) 来源
        - 单事务保证「entities 已写 + fact_entities 未写」的中间态不可见

        ``facts.enrichment_status='done'`` 由 worker `mark_done` 设置，
        本方法不动 facts 表。
        """
        # 去重 + 规范化
        seen: set[str] = set()
        clean: list[str] = []
        for name in entity_names:
            if not isinstance(name, str):
                continue
            s = name.strip()
            if len(s) < 2 or s in seen:
                continue
            seen.add(s)
            clean.append(s)
        if not clean:
            return

        with _txn(self._conn):
            for name in clean:
                self._conn.execute(
                    "INSERT OR IGNORE INTO entities "
                    "(name, entity_type, extraction_method) VALUES (?, ?, ?)",
                    (name, "llm_extracted", "llm"),
                )
                row = self._conn.execute(
                    "SELECT entity_id FROM entities WHERE name = ?", (name,)
                ).fetchone()
                self._conn.execute(
                    "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) "
                    "VALUES (?, ?)",
                    (fact_id, int(row["entity_id"])),
                )

    # ─── semantic 索引 ──────────────────────────────────────────────────

    def _index_semantic(self, fact_id: int, content: str) -> bool:
        """encode → 更新 ``facts.semantic_vector`` + ``embedding_model`` → ``vstore.add``。

        失败路径全部静默 WARN：semantic 是检索增强而非数据正确性 — 索引失败时
        fact 已经入 DB / bank，下次 reindex 会补上。

        Returns:
            ``True`` semantic 写入成功；``False`` 未注入 backend / backend 不可用 /
            encode 异常被吞。
        """
        if self._semantic is None or self._vstore is None:
            return False
        if not self._semantic.is_available():
            # degraded 路径主流场景：sentence-transformers 缺失；不打日志（每次写都打太吵）
            return False
        # 异步 warmup 进行中（包可用但模型未加载）→ 跳过本次 semantic 索引。
        # 避免在主线程触发同步模型加载阻塞写入；后续 reindex 会补上未索引的 fact。
        if not getattr(self._semantic, "is_loaded", True):
            log.debug(
                "semantic 模型尚未加载（异步 warmup 进行中），跳过 fact_id=%s 的 semantic 索引",
                fact_id,
            )
            return False
        try:
            vec = self._semantic.encode(content)
        except SemanticUnavailable as e:
            log.warning("semantic.encode 失败 fact_id=%s: %s", fact_id, e)
            return False
        except Exception as e:
            log.warning("semantic.encode 异常 fact_id=%s: %s", fact_id, e)
            return False

        try:
            self._conn.execute(
                "UPDATE facts SET semantic_vector = ?, embedding_model = ? "
                "WHERE fact_id = ?",
                (vec.tobytes(), self._semantic.backend_name, fact_id),
            )
        except sqlite3.Error as e:
            log.warning("更新 facts.semantic_vector 失败 fact_id=%s: %s", fact_id, e)
            return False

        try:
            self._vstore.add(fact_id, vec)
        except Exception as e:
            log.warning("vstore.add 失败 fact_id=%s: %s", fact_id, e)
            return False

        return True

    # ─── HRR 编码 ────────────────────────────────────────────────────────

    def _encode_fact(self, content: str, entity_names: tuple[str, ...]) -> np.ndarray:
        """``bundle(bind(ROLE_CONTENT, atom(content)), bind(ROLE_ENTITY, atom(e_i))...)``。

        无实体时退化为单个 ``bind(ROLE_CONTENT, atom(content))``（bind 输出已 mod 2π）。
        """
        content_atom = atom(content, dim=self._dim)
        bound_content = bind(self._role_content, content_atom)
        if not entity_names:
            return bound_content
        bound_entities = [
            bind(self._role_entity, atom(name, dim=self._dim)) for name in entity_names
        ]
        return bundle(bound_content, *bound_entities)


# ─── 内部事务工具 ─────────────────────────────────────────────────────────


@contextmanager
def _txn(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """``isolation_level=None`` 连接专用的轻量事务：成功 COMMIT / 异常 ROLLBACK。

    ``db._transaction`` 是 db 模块私有；mirror 内联以避免跨模块导入下划线名。
    """
    conn.execute("BEGIN")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
