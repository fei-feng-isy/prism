"""Prism 数据层 — SQLite schema + 连接管理 + Repository 模式封装。

每个 profile/user 一个独立 .db 文件。FTS5 使用 trigram tokenizer（对中日韩友好），
facts_fts 通过 trigger 与 facts 主表自动同步。连接默认开启 foreign_keys / WAL /
busy_timeout=1s。路径解析见 :mod:`prism.config.paths`。

Repository 层封装所有表操作，禁止外部模块直接拼 SQL。
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

__all__ = [
    "DDL_STATEMENTS",
    "FTS_TRIGGERS",
    "MEMORY_DB",
    "SCHEMA_VERSION",
    "DatabaseError",
    "DatabaseRepository",
    "FactsRepository",
    "EntitiesRepository",
    "EnrichmentQueueRepository",
    "ContradictionRepository",
    "StatsRepository",
    "bootstrap",
    "connect",
    "init_schema",
    "schema_version",
    "verify_chinese_tokenizer",
]

log = logging.getLogger(__name__)


class DatabaseError(RuntimeError):
    """数据库初始化或操作失败。"""


# ─── 常量 ───────────────────────────────────────────────────────────────────

SCHEMA_VERSION: Final[int] = 1
"""当前 schema 版本；后续迁移须 +1 并在 ``_MIGRATIONS`` 中注册步骤。"""

MEMORY_DB: Final[str] = ":memory:"
"""sqlite3 内存数据库标识；测试常用。"""


# 主表与辅助表（顺序敏感：含外键引用必须后建）
DDL_STATEMENTS: Final[tuple[str, ...]] = (
    # ─ facts 主表 ─
    """
    CREATE TABLE IF NOT EXISTS facts (
        fact_id           INTEGER PRIMARY KEY AUTOINCREMENT,
        content           TEXT NOT NULL UNIQUE,
        category          TEXT DEFAULT 'general',
        tags              TEXT DEFAULT '',
        trust_score       REAL DEFAULT 0.5,
        retrieval_count   INTEGER DEFAULT 0,
        helpful_count     INTEGER DEFAULT 0,
        last_retrieved_at TIMESTAMP,
        hrr_vector        BLOB,
        semantic_vector   BLOB,
        embedding_model   TEXT,
        vector_store      TEXT DEFAULT 'local_numpy',
        status            TEXT DEFAULT 'active',
        supersedes_id     INTEGER REFERENCES facts(fact_id),
        archived_at       TIMESTAMP,
        archive_reason    TEXT,
        ttl_days          INTEGER DEFAULT 0,
        enrichment_status TEXT DEFAULT 'pending',
        mirror_source     TEXT,
        mirror_target     TEXT,
        created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_facts_status_category ON facts(status, category)",
    "CREATE INDEX IF NOT EXISTS idx_facts_supersedes ON facts(supersedes_id)",
    # ─ FTS5 虚表 ─
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
        content,
        content='facts',
        content_rowid='fact_id',
        tokenize='trigram'
    )
    """,
    # ─ 实体 ─
    """
    CREATE TABLE IF NOT EXISTS entities (
        entity_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        name              TEXT NOT NULL UNIQUE,
        entity_type       TEXT DEFAULT 'unknown',
        aliases           TEXT DEFAULT '',
        extraction_method TEXT DEFAULT 'regex',
        created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fact_entities (
        fact_id   INTEGER REFERENCES facts(fact_id) ON DELETE CASCADE,
        entity_id INTEGER REFERENCES entities(entity_id) ON DELETE CASCADE,
        PRIMARY KEY (fact_id, entity_id)
    )
    """,
    # ─ 异步富化队列 ─
    """
    CREATE TABLE IF NOT EXISTS enrichment_queue (
        fact_id         INTEGER PRIMARY KEY REFERENCES facts(fact_id) ON DELETE CASCADE,
        enqueued_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        attempts        INTEGER DEFAULT 0,
        last_error      TEXT,
        last_attempt_at TIMESTAMP
    )
    """,
    # ─ 增量 bank ─
    """
    CREATE TABLE IF NOT EXISTS bank_state (
        bank_name          TEXT PRIMARY KEY,
        vector             BLOB NOT NULL,
        fact_count         INTEGER DEFAULT 0,
        dirty_count        INTEGER DEFAULT 0,
        last_calibrated_at TIMESTAMP,
        snr_current        REAL,
        updated_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    # ─ 矛盾日志 ─
    """
    CREATE TABLE IF NOT EXISTS contradiction_log (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        fact_a        INTEGER REFERENCES facts(fact_id) ON DELETE CASCADE,
        fact_b        INTEGER REFERENCES facts(fact_id) ON DELETE CASCADE,
        score         REAL,
        detected_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        resolved      INTEGER DEFAULT 0,
        loser_fact_id INTEGER REFERENCES facts(fact_id),
        resolution    TEXT,
        resolved_at   TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_contradict_loser ON contradiction_log(loser_fact_id)",
    # ─ 统计 & 评估 ─
    """
    CREATE TABLE IF NOT EXISTS prism_stats (
        key        TEXT PRIMARY KEY,
        value      TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_results (
        eval_run_id    TEXT,
        query_id       TEXT,
        expected_ids   TEXT,
        actual_ids     TEXT,
        precision_at_k REAL,
        recall_at_k    REAL,
        mrr            REAL,
        ran_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (eval_run_id, query_id)
    )
    """,
)


# FTS5 触发器：把 facts 内容同步进 facts_fts
# 用 contentless-style 的 "delete" 指令撤回旧值（FTS5 外部内容表模式）
FTS_TRIGGERS: Final[tuple[str, ...]] = (
    """
    CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
        INSERT INTO facts_fts(rowid, content) VALUES (new.fact_id, new.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
        INSERT INTO facts_fts(facts_fts, rowid, content)
            VALUES ('delete', old.fact_id, old.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE OF content ON facts BEGIN
        INSERT INTO facts_fts(facts_fts, rowid, content)
            VALUES ('delete', old.fact_id, old.content);
        INSERT INTO facts_fts(rowid, content) VALUES (new.fact_id, new.content);
    END
    """,
)


# ─── 连接 ───────────────────────────────────────────────────────────────────


def connect(
    path: str | os.PathLike[str],
    *,
    busy_timeout_ms: int = 1000,
) -> sqlite3.Connection:
    """创建一个配置好 PRAGMA 的 sqlite3 连接。

    Args:
        path: 数据库文件路径，或 ``MEMORY_DB`` 走内存库
        busy_timeout_ms: SQLite 锁等待超时

    Raises:
        DatabaseError: 父目录创建失败，或 sqlite3.connect 失败
    """
    target = str(path)
    if target != MEMORY_DB:
        p = Path(target).expanduser()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise DatabaseError(f"无法创建父目录 {p.parent}: {e}") from e
        target = str(p)

    try:
        conn = sqlite3.connect(target, isolation_level=None, check_same_thread=False)
    except sqlite3.Error as e:
        raise DatabaseError(f"打开数据库失败 {target}: {e}") from e

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    if target != MEMORY_DB:
        # 内存库不支持 WAL；仅对落盘库启用
        conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


# ─── schema 初始化 / 迁移 ───────────────────────────────────────────────────


def schema_version(conn: sqlite3.Connection) -> int:
    """读取当前 ``PRAGMA user_version``。"""
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row is not None else 0


def init_schema(conn: sqlite3.Connection) -> int:
    """幂等创建全部表 / 索引 / 触发器；推进 ``user_version`` 至 ``SCHEMA_VERSION``。

    重复调用安全：所有 DDL 均使用 IF NOT EXISTS。
    Returns:
        迁移后的 schema 版本号
    """
    current = schema_version(conn)
    if current > SCHEMA_VERSION:
        raise DatabaseError(
            f"数据库 schema 版本 {current} 高于代码支持的 {SCHEMA_VERSION}，"
            "请升级 Prism 或回滚 DB"
        )

    try:
        with _transaction(conn):
            for ddl in DDL_STATEMENTS:
                conn.execute(ddl)
            for trig in FTS_TRIGGERS:
                conn.execute(trig)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    except sqlite3.Error as e:
        raise DatabaseError(f"schema 初始化失败: {e}") from e

    return SCHEMA_VERSION


class _transaction:
    """轻量事务上下文：成功 commit / 失败 rollback。

    必须配合 ``isolation_level=None`` 的连接使用（已在 :func:`connect` 设好）。
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def __enter__(self) -> sqlite3.Connection:
        self._conn.execute("BEGIN")
        return self._conn

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc is None:
            self._conn.execute("COMMIT")
        else:
            self._conn.execute("ROLLBACK")


# ─── 一站式 bootstrap ───────────────────────────────────────────────────────


def bootstrap(path: str | os.PathLike[str]) -> sqlite3.Connection:
    """连接 + 初始化 schema。便利封装，测试常用。"""
    conn = connect(path)
    init_schema(conn)
    return conn


# ─── 中文 tokenizer 自检 ────────────────────────────────────────────────────


# 用 ``trigram`` 必然命中的 3-字符切片 + 一段未出现的 3-字符切片
_PROBE_SAMPLE: Final[str] = "我喜欢简洁的中文回答"
_PROBE_HIT: Final[str] = "简洁的"
_PROBE_MISS: Final[str] = "繁琐的"


def verify_chinese_tokenizer(conn: sqlite3.Connection) -> None:
    """证明 ``facts_fts`` 能按 n-gram 切分中文，而非把整段视为单 token。

    判定（基于内置 ``trigram`` tokenizer，查询长度需 ≥ 3）：
        - 命中：含 ``简洁的`` 的事实能被 ``MATCH '简洁的'`` 召回
        - 反例：未出现的 ``繁琐的`` 不会误命中

    Raises:
        DatabaseError: 行为不符（提示重建 FTS5 表或更换 tokenizer）
    """
    try:
        with _transaction(conn):
            # 写入一条带独占 category 的探针；事务收尾时一律 rollback，不污染数据
            conn.execute(
                "INSERT INTO facts(content, category) VALUES (?, ?)",
                (_PROBE_SAMPLE, "_probe"),
            )
            hit = conn.execute(
                "SELECT COUNT(*) FROM facts_fts WHERE facts_fts MATCH ?",
                (_PROBE_HIT,),
            ).fetchone()[0]
            miss = conn.execute(
                "SELECT COUNT(*) FROM facts_fts WHERE facts_fts MATCH ?",
                (_PROBE_MISS,),
            ).fetchone()[0]
            if hit < 1:
                raise _ProbeFailure(
                    f"FTS5 未能匹配中文 trigram {_PROBE_HIT!r}；"
                    "请确认 tokenizer 配置为 'trigram'"
                )
            if miss > 0:
                raise _ProbeFailure(
                    f"FTS5 在不应命中的 trigram {_PROBE_MISS!r} 上召回 {miss} 条"
                )
            raise _ProbeRollback  # 故意触发 rollback 撤销 marker 行
    except _ProbeRollback:
        return
    except _ProbeFailure as e:
        raise DatabaseError(str(e)) from None


class _ProbeFailure(Exception):
    """tokenizer 自检失败，外层转为 DatabaseError。"""


class _ProbeRollback(Exception):
    """成功路径：故意触发 rollback 撤销 probe 行。"""


# ═══════════════════════════════════════════════════════════════════════════════
# Repository 层 — 统一数据库访问接口
# ═══════════════════════════════════════════════════════════════════════════════


# ─── FactsRepository ─────────────────────────────────────────────────────────


class FactsRepository:
    """facts 表的统一读写接口。

    所有 INSERT / UPDATE / SELECT / DELETE 必须经过此类的公开方法，
    禁止外部模块拼装 SQL 操作 facts 表。

    Repository 方法不管理事务 —— 由调用方负责 BEGIN/COMMIT/ROLLBACK。
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        # embedding_model 标签一致性缓存（实例作用域，避免跨连接/跨测试污染）
        self._distinct_models: set[str] | None = None

    def _consolidate_model(self, new_model: str) -> str:
        """检查 embedding_model 一致性：如果 facts 表中已有其他标签，
        自动对齐到已有标签（并 WARN 日志记录逃逸路径）。

        缓存基于实例，每次新连接自动失效。
        """
        if self._distinct_models is None:
            rows = self._conn.execute(
                "SELECT DISTINCT embedding_model FROM facts "
                "WHERE embedding_model IS NOT NULL AND semantic_vector IS NOT NULL"
            ).fetchall()
            self._distinct_models = {str(r[0]) for r in rows} if rows else set()

        existing = self._distinct_models
        if not existing:
            existing.add(new_model)
            return new_model
        if new_model in existing:
            return new_model

        # 标签不一致！WARN + 自动对齐
        target = next(iter(existing))
        log.warning(
            "embedding_model 标签不一致：试图写入 '%s'，自动对齐到已有标签 '%s'。",
            new_model, target,
        )
        return target

    # ── 写入 ──────────────────────────────────

    def insert_fact(
        self,
        content: str,
        category: str,
        hrr_vector: bytes,
        mirror_source: str,
        mirror_target: str | None = None,
        supersedes_id: int | None = None,
    ) -> tuple[int, bool]:
        """INSERT OR IGNORE；返回 (fact_id, is_new)。

        is_new=False 表示 content UNIQUE 冲突命中已有 fact。
        """
        assert isinstance(content, str) and content.strip(), "content 不可为空"
        assert isinstance(category, str), "category 必须为字符串"
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO facts "
            "(content, category, hrr_vector, mirror_source, mirror_target, supersedes_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (content, category, hrr_vector, mirror_source, mirror_target, supersedes_id),
        )
        if cur.rowcount == 0:
            row = self._conn.execute(
                "SELECT fact_id FROM facts WHERE content = ?", (content,)
            ).fetchone()
            return int(row["fact_id"]), False
        assert cur.lastrowid is not None
        return int(cur.lastrowid), True

    def upsert_semantic_vector(
        self, fact_id: int, vec_bytes: bytes, embedding_model: str
    ) -> None:
        """写入 semantic_vector + embedding_model（含标签一致性断言）。"""
        consolidated = self._consolidate_model(embedding_model)
        self._conn.execute(
            "UPDATE facts SET semantic_vector = ?, embedding_model = ? "
            "WHERE fact_id = ?",
            (vec_bytes, consolidated, fact_id),
        )

    def archive_fact(
        self, fact_id: int, reason: str, now: str | None = None
    ) -> bool:
        """status='archived' + archived_at + archive_reason。
        Returns True 真正执行了 UPDATE，False 已是 archived。
        """
        if now is None:
            cur = self._conn.execute(
                "UPDATE facts SET status = 'archived', "
                "archived_at = CURRENT_TIMESTAMP, archive_reason = ? "
                "WHERE fact_id = ? AND status = 'active'",
                (reason, fact_id),
            )
        else:
            cur = self._conn.execute(
                "UPDATE facts SET status = 'archived', "
                "archived_at = ?, archive_reason = ? "
                "WHERE fact_id = ? AND status = 'active'",
                (now, reason, fact_id),
            )
        return cur.rowcount > 0

    def restore_fact(self, fact_id: int) -> None:
        """status='active' + archived_at=NULL + archive_reason=NULL。"""
        self._conn.execute(
            "UPDATE facts SET status = 'active', "
            "archived_at = NULL, archive_reason = NULL "
            "WHERE fact_id = ?",
            (fact_id,),
        )

    def update_trust_score(self, fact_id: int, score: float) -> None:
        self._conn.execute(
            "UPDATE facts SET trust_score = ? WHERE fact_id = ?",
            (float(score), fact_id),
        )

    def update_trust_and_helpful(
        self, fact_id: int, score: float, helpful_count: int
    ) -> None:
        self._conn.execute(
            "UPDATE facts SET trust_score = ?, helpful_count = ? WHERE fact_id = ?",
            (float(score), int(helpful_count), fact_id),
        )

    def increment_retrieval_count(self, fact_ids: list[int]) -> None:
        if not fact_ids:
            return
        placeholders = ",".join("?" for _ in fact_ids)
        self._conn.execute(
            f"UPDATE facts SET retrieval_count = retrieval_count + 1 "
            f"WHERE fact_id IN ({placeholders})",
            fact_ids,
        )

    # ── 单条读取 ──────────────────────────────────

    def get_fact_by_id(self, fact_id: int) -> dict | None:
        """按 fact_id 取完整行。"""
        row = self._conn.execute(
            "SELECT * FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def get_active_fact(self, fact_id: int) -> dict | None:
        """status='active' 的完整行。"""
        row = self._conn.execute(
            "SELECT content, category, status, hrr_vector, semantic_vector, "
            "embedding_model, trust_score, helpful_count, mirror_source, mirror_target "
            "FROM facts WHERE fact_id = ? AND status = 'active'",
            (fact_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def get_fact_by_content(self, content: str) -> dict | None:
        row = self._conn.execute(
            "SELECT fact_id, category, status FROM facts WHERE content = ?",
            (content,),
        ).fetchone()
        return dict(row) if row is not None else None

    def get_fact_ids_by_content_bulk(self, contents: list[str]) -> dict[str, int]:
        if not contents:
            return {}
        placeholders = ",".join("?" for _ in contents)
        rows = self._conn.execute(
            f"SELECT fact_id, content FROM facts WHERE content IN ({placeholders})",
            list(contents),
        ).fetchall()
        return {str(r["content"]): int(r["fact_id"]) for r in rows}

    def get_active_facts_by_source(
        self, source: str, batch_size: int = 500, last_id: int = 0
    ) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM facts "
            "WHERE status = 'active' AND mirror_source = ? "
            "AND fact_id > ? ORDER BY fact_id LIMIT ?",
            (source, last_id, batch_size),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_expired_ttl_ids(self, now: str, batch_size: int) -> list[int]:
        rows = self._conn.execute(
            "SELECT fact_id FROM facts "
            "WHERE status = 'active' AND ttl_days > 0 "
            "AND created_at IS NOT NULL "
            "AND datetime(created_at, '+' || ttl_days || ' days') <= ? "
            "ORDER BY fact_id LIMIT ?",
            (now, batch_size),
        ).fetchall()
        return [int(r["fact_id"]) for r in rows]

    def get_purge_candidates(
        self, cutoff: str, retention_days: int, batch_size: int
    ) -> list[int]:
        rows = self._conn.execute(
            "SELECT fact_id FROM facts "
            "WHERE status = 'archived' AND archived_at IS NOT NULL "
            "AND datetime(archived_at, '+' || ? || ' days') <= ? "
            "ORDER BY fact_id LIMIT ?",
            (retention_days, cutoff, batch_size),
        ).fetchall()
        return [int(r["fact_id"]) for r in rows]

    # ── 列表查询 ──────────────────────────────────

    def list_facts(
        self,
        *,
        category: str | None = None,
        status: str | None = "active",
        mirror_source: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], bool]:
        """参数化列表查询。返回 (rows, truncated)。"""
        clauses: list[str] = []
        params: list[Any] = []
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if mirror_source is not None:
            clauses.append("mirror_source = ?")
            params.append(mirror_source)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        sql = (
            "SELECT fact_id, content, category, status, trust_score, "
            "helpful_count, mirror_source, created_at, archived_at, archive_reason "
            f"FROM facts{where} ORDER BY created_at DESC, fact_id DESC "
            "LIMIT ? OFFSET ?"
        )
        params.extend([limit + 1, offset])
        rows = self._conn.execute(sql, params).fetchall()
        truncated = len(rows) > limit
        return [dict(r) for r in rows[:limit]], truncated

    def filter_ids_by_min_trust(
        self, fact_ids: list[int], min_trust: float
    ) -> set[int]:
        if not fact_ids:
            return set()
        placeholders = ",".join("?" for _ in fact_ids)
        rows = self._conn.execute(
            f"SELECT fact_id FROM facts "
            f"WHERE fact_id IN ({placeholders}) AND trust_score >= ?",
            [*fact_ids, float(min_trust)],
        ).fetchall()
        return {int(r["fact_id"]) for r in rows}

    def get_active_facts_with_semantic_vector(
        self, embedding_model: str
    ) -> list[dict]:
        """用于 vstore 重建。"""
        rows = self._conn.execute(
            "SELECT fact_id, semantic_vector FROM facts "
            "WHERE status='active' AND embedding_model = ? "
            "AND semantic_vector IS NOT NULL",
            (embedding_model,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── 聚合查询 ──────────────────────────────────

    def count_active(self, category: str | None = None) -> int:
        if category is None:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE status = 'active'"
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE status = 'active' AND category = ?",
                (category,),
            ).fetchone()
        return int(row[0]) if row else 0

    def count_archived(self, category: str | None = None) -> int:
        if category is None:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE status = 'archived'"
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE status = 'archived' AND category = ?",
                (category,),
            ).fetchone()
        return int(row[0]) if row else 0

    def count_total(self, category: str | None = None) -> int:
        if category is None:
            row = self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE category = ?", (category,)
            ).fetchone()
        return int(row[0]) if row else 0

    def count_by_category(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT category, COUNT(*) AS n FROM facts "
            "WHERE status = 'active' "
            "GROUP BY category ORDER BY n DESC"
        ).fetchall()
        return {str(r["category"]): int(r["n"]) for r in rows}

    def count_by_enrichment_status(self) -> dict[str, dict]:
        rows = self._conn.execute(
            "SELECT enrichment_status, COUNT(*) AS cnt, "
            "SUM(CASE WHEN semantic_vector IS NULL THEN 1 ELSE 0 END) AS null_vec "
            "FROM facts GROUP BY enrichment_status"
        ).fetchall()
        return {
            str(r["enrichment_status"]): {"count": int(r["cnt"]), "null_vector": int(r["null_vec"])}
            for r in rows
        }

    def get_trust_aggregates(self) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT AVG(trust_score) AS avg, "
            "SUM(CASE WHEN trust_score > 0.7 THEN 1 ELSE 0 END) AS high, "
            "SUM(CASE WHEN trust_score >= 0.3 AND trust_score <= 0.7 THEN 1 ELSE 0 END) AS mid, "
            "SUM(CASE WHEN trust_score < 0.3 THEN 1 ELSE 0 END) AS low "
            "FROM facts WHERE status = 'active'"
        ).fetchone()
        return {
            "avg": float(row["avg"]) if row and row["avg"] is not None else None,
            "high (>0.7)": int(row["high"] or 0) if row else 0,
            "mid (0.3-0.7)": int(row["mid"] or 0) if row else 0,
            "low (<0.3)": int(row["low"] or 0) if row else 0,
        }

    def count_by_enrichment_status_simple(self, status: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM facts WHERE enrichment_status = ?", (status,)
        ).fetchone()
        return int(row[0]) if row else 0

    # ── 物理删除（purge 路径）───────────────────────

    def delete_facts(self, fact_ids: list[int]) -> None:
        if not fact_ids:
            return
        placeholders = ",".join("?" for _ in fact_ids)
        self._conn.execute(
            f"DELETE FROM facts WHERE fact_id IN ({placeholders})", fact_ids
        )

    def unlink_supersedes(self, fact_ids: list[int]) -> None:
        """purge 前的 FK 解除 — supersedes_id 反向引用。"""
        if not fact_ids:
            return
        placeholders = ",".join("?" for _ in fact_ids)
        self._conn.execute(
            f"UPDATE facts SET supersedes_id = NULL "
            f"WHERE supersedes_id IN ({placeholders})",
            fact_ids,
        )


# ─── EntitiesRepository ──────────────────────────────────────────────────────


# 合法的 extraction_method 取值
_VALID_EXTRACTION_METHODS: Final[frozenset[str]] = frozenset({
    "regex", "jieba", "llm", "manual",
})


class EntitiesRepository:
    """entities + fact_entities 表的统一接口。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # ── 写入 ──────────────────────────────────

    def insert_entity(
        self, name: str, entity_type: str = "unknown", method: str = "regex"
    ) -> int:
        """INSERT OR IGNORE；返回 entity_id。"""
        assert isinstance(name, str) and name.strip(), "name 不可为空"
        assert method in _VALID_EXTRACTION_METHODS, f"method 取值非法: {method!r}"
        self._conn.execute(
            "INSERT OR IGNORE INTO entities "
            "(name, entity_type, extraction_method) VALUES (?, ?, ?)",
            (name, entity_type, method),
        )
        row = self._conn.execute(
            "SELECT entity_id FROM entities WHERE name = ?", (name,)
        ).fetchone()
        assert row is not None
        return int(row["entity_id"])

    def link_entity(self, fact_id: int, entity_id: int) -> None:
        """INSERT OR IGNORE INTO fact_entities。"""
        self._conn.execute(
            "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
            (fact_id, entity_id),
        )

    def ensure_entities(
        self, names: list[tuple[str, str, str]]
    ) -> dict[str, int]:
        """批量确保实体存在，返回 name→entity_id 映射。

        names: [(name, entity_type, method), ...]。
        """
        result: dict[str, int] = {}
        for name, etype, method in names:
            self._conn.execute(
                "INSERT OR IGNORE INTO entities "
                "(name, entity_type, extraction_method) VALUES (?, ?, ?)",
                (name, etype, method),
            )
        # 批量 SELECT 取回所有 ID
        for name, _, _ in names:
            row = self._conn.execute(
                "SELECT entity_id FROM entities WHERE name = ?", (name,)
            ).fetchone()
            if row is not None:
                result[name] = int(row["entity_id"])
        return result

    def link_entities_bulk(
        self, fact_id: int, entity_ids: list[int]
    ) -> None:
        """批量链接 fact_entities。"""
        if not entity_ids:
            return
        self._conn.executemany(
            "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
            [(fact_id, eid) for eid in entity_ids],
        )

    # ── 读取 ──────────────────────────────────

    def get_entity_ids_by_names(self, names: set[str]) -> dict[str, int]:
        if not names:
            return {}
        placeholders = ",".join("?" for _ in names)
        rows = self._conn.execute(
            f"SELECT entity_id, name FROM entities WHERE name IN ({placeholders})",
            list(names),
        ).fetchall()
        return {str(r["name"]): int(r["entity_id"]) for r in rows}

    def get_entity_names_by_fact_id(self, fact_id: int) -> list[str]:
        rows = self._conn.execute(
            "SELECT e.name FROM fact_entities fe "
            "JOIN entities e ON e.entity_id = fe.entity_id "
            "WHERE fe.fact_id = ? ORDER BY e.name",
            (fact_id,),
        ).fetchall()
        return [str(r["name"]) for r in rows]

    def get_fact_entity_ids(self, fact_id: int) -> set[int]:
        rows = self._conn.execute(
            "SELECT entity_id FROM fact_entities WHERE fact_id = ?", (fact_id,)
        ).fetchall()
        return {int(r["entity_id"]) for r in rows}

    # ── 关联查询 ──────────────────────────────────

    def get_fact_ids_by_entity(
        self, entity: str, category: str | None = None, limit: int = 10
    ) -> list[dict]:
        """probe 查询：三表 JOIN，按实体名找关联 fact。"""
        params: list[Any] = [entity.strip()]
        sql = (
            "SELECT f.fact_id, f.content, f.category, f.trust_score "
            "FROM facts f "
            "JOIN fact_entities fe ON fe.fact_id = f.fact_id "
            "JOIN entities e ON e.entity_id = fe.entity_id "
            "WHERE e.name = ? AND f.status = 'active'"
        )
        if category is not None:
            sql += " AND f.category = ?"
            params.append(category)
        sql += " ORDER BY f.fact_id DESC LIMIT ?"
        params.append(int(limit))
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_fact_ids_by_entities(
        self, entity_names: list[str], category: str | None = None, limit: int = 10
    ) -> list[dict]:
        """reason 查询：HAVING COUNT(DISTINCT e.entity_id) = len(names)。"""
        if not entity_names:
            return []
        placeholders = ",".join("?" for _ in entity_names)
        params: list[Any] = list(entity_names)
        sql = (
            "SELECT f.fact_id, f.content, f.category, f.trust_score "
            "FROM facts f "
            "JOIN fact_entities fe ON fe.fact_id = f.fact_id "
            "JOIN entities e ON e.entity_id = fe.entity_id "
            f"WHERE e.name IN ({placeholders}) AND f.status = 'active'"
        )
        if category is not None:
            sql += " AND f.category = ?"
            params.append(category)
        sql += (
            " GROUP BY f.fact_id "
            "HAVING COUNT(DISTINCT e.entity_id) = ? "
            "ORDER BY f.fact_id DESC LIMIT ?"
        )
        params.extend([len(entity_names), int(limit)])
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_co_occurring_entities(
        self, entity: str, category: str | None = None, limit: int = 10
    ) -> list[dict]:
        """related 查询：自联表 JOIN。"""
        anchor = entity.strip()
        params: list[Any] = [anchor]
        sql = (
            "SELECT e2.name AS name, COUNT(*) AS co "
            "FROM fact_entities fe1 "
            "JOIN entities e1 ON e1.entity_id = fe1.entity_id "
            "JOIN fact_entities fe2 ON fe2.fact_id = fe1.fact_id "
            "JOIN entities e2 ON e2.entity_id = fe2.entity_id "
            "JOIN facts f ON f.fact_id = fe1.fact_id "
            "WHERE e1.name = ? AND e2.name != e1.name AND f.status = 'active'"
        )
        if category is not None:
            sql += " AND f.category = ?"
            params.append(category)
        sql += " GROUP BY e2.name ORDER BY co DESC, e2.name ASC LIMIT ?"
        params.append(int(limit))
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def load_facts_with_entities(
        self, fact_ids: list[int], active_only: bool = False, limit: int = 2000
    ) -> list[dict]:
        """LEFT JOIN facts→fact_entities→entities，按 fact_id 分组带 entity 列表。

        contradict 模块核心查询。返回 list[dict]，每项含 fact_id + entities(set)。
        """
        if not fact_ids:
            return []
        placeholders = ",".join("?" for _ in fact_ids)
        params: list[Any] = list(fact_ids)
        if active_only:
            sql = (
                f"SELECT f.fact_id, e.name AS entity FROM facts f "
                f"LEFT JOIN fact_entities fe ON fe.fact_id = f.fact_id "
                f"LEFT JOIN entities e ON e.entity_id = fe.entity_id "
                f"WHERE f.fact_id IN ({placeholders}) AND f.status = 'active' "
                f"LIMIT ?"
            )
            params.append(int(limit))
        else:
            sql = (
                f"SELECT f.fact_id, e.name AS entity FROM facts f "
                f"LEFT JOIN fact_entities fe ON fe.fact_id = f.fact_id "
                f"LEFT JOIN entities e ON e.entity_id = fe.entity_id "
                f"WHERE f.fact_id IN ({placeholders}) "
                f"LIMIT ?"
            )
            params.append(int(limit))
        rows = self._conn.execute(sql, params).fetchall()
        # 按 fact_id 分组
        grouped: dict[int, set[str]] = {}
        for r in rows:
            fid = int(r["fact_id"])
            ent = r["entity"]
            bucket = grouped.setdefault(fid, set())
            if ent is not None:
                bucket.add(str(ent))
        return [
            {"fact_id": fid, "entities": ents}
            for fid, ents in sorted(grouped.items())
        ]

    def load_active_corpus_with_entities(self, limit: int = 2000) -> list[dict]:
        """加载 active fact 语料 + 关联实体（contradict 用）。"""
        rows = self._conn.execute(
            "SELECT f.fact_id, e.name AS entity FROM facts f "
            "LEFT JOIN fact_entities fe ON fe.fact_id = f.fact_id "
            "LEFT JOIN entities e ON e.entity_id = fe.entity_id "
            "WHERE f.fact_id IN ("
            "  SELECT fact_id FROM facts WHERE status = 'active' "
            "  ORDER BY fact_id DESC LIMIT ?"
            ")",
            (int(limit),),
        ).fetchall()
        grouped: dict[int, set[str]] = {}
        for r in rows:
            fid = int(r["fact_id"])
            ent = r["entity"]
            bucket = grouped.setdefault(fid, set())
            if ent is not None:
                bucket.add(str(ent))
        return [
            {"fact_id": fid, "entities": ents}
            for fid, ents in sorted(grouped.items())
        ]

    # ── 聚合 ──────────────────────────────────

    def count_entities(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()
        return int(row[0]) if row else 0

    def count_fact_entities(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM fact_entities").fetchone()
        return int(row[0]) if row else 0


# ─── EnrichmentQueueRepository ────────────────────────────────────────────────


class EnrichmentQueueRepository:
    """enrichment_queue + facts.enrichment_status 的统一接口。"""

    DEFAULT_MAX_ATTEMPTS: Final[int] = 3
    _MAX_ERROR_LEN: Final[int] = 1024

    def __init__(self, conn: sqlite3.Connection, *, max_attempts: int = 3) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        self._conn = conn
        self._max_attempts = max_attempts

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    def enqueue(self, fact_id: int) -> bool:
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO enrichment_queue (fact_id) VALUES (?)",
            (fact_id,),
        )
        return cur.rowcount > 0

    def pop_next(self) -> dict | None:
        """取最旧的待处理 fact，原子 attempts++。返回 dict 或 None。"""
        row = self._conn.execute(
            """
            SELECT q.fact_id, f.content, q.attempts, q.last_error
            FROM enrichment_queue q
            JOIN facts f ON f.fact_id = q.fact_id
            WHERE f.status = 'active'
            ORDER BY q.enqueued_at ASC, q.fact_id ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        fact_id = int(row["fact_id"])
        new_attempts = int(row["attempts"]) + 1
        self._conn.execute(
            "UPDATE enrichment_queue SET attempts = ?, "
            "last_attempt_at = CURRENT_TIMESTAMP WHERE fact_id = ?",
            (new_attempts, fact_id),
        )
        return {
            "fact_id": fact_id,
            "content": str(row["content"]),
            "attempts": new_attempts,
            "last_error": row["last_error"],
        }

    def mark_done(self, fact_id: int) -> None:
        self._conn.execute(
            "DELETE FROM enrichment_queue WHERE fact_id = ?", (fact_id,)
        )
        self._conn.execute(
            "UPDATE facts SET enrichment_status = 'done', "
            "updated_at = CURRENT_TIMESTAMP WHERE fact_id = ?",
            (fact_id,),
        )

    def mark_failed(self, fact_id: int, error: str) -> None:
        row = self._conn.execute(
            "SELECT attempts FROM enrichment_queue WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        if row is None:
            return
        attempts = int(row[0])
        truncated = error[: self._MAX_ERROR_LEN]
        if attempts >= self._max_attempts:
            self._conn.execute(
                "DELETE FROM enrichment_queue WHERE fact_id = ?", (fact_id,)
            )
            self._conn.execute(
                "UPDATE facts SET enrichment_status = 'failed', "
                "updated_at = CURRENT_TIMESTAMP WHERE fact_id = ?",
                (fact_id,),
            )
            log.warning(
                "enrichment fact_id=%s 达到 max_attempts=%s 放弃；error=%s",
                fact_id, self._max_attempts, truncated,
            )
        else:
            self._conn.execute(
                "UPDATE enrichment_queue SET last_error = ?, "
                "last_attempt_at = CURRENT_TIMESTAMP WHERE fact_id = ?",
                (truncated, fact_id),
            )

    def pending_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM enrichment_queue"
        ).fetchone()
        return int(row[0])

    def stats(self) -> dict[str, int]:
        """返回 {pending, done, failed} 三段计数。"""
        pending = self.pending_count()
        done_row = self._conn.execute(
            "SELECT COUNT(*) FROM facts WHERE enrichment_status = 'done'"
        ).fetchone()
        failed_row = self._conn.execute(
            "SELECT COUNT(*) FROM facts WHERE enrichment_status = 'failed'"
        ).fetchone()
        return {
            "pending": pending,
            "done": int(done_row[0]) if done_row else 0,
            "failed": int(failed_row[0]) if failed_row else 0,
        }

    def list_items(self, limit: int = 100) -> list[dict]:
        rows = self._conn.execute(
            "SELECT fact_id, attempts, last_error, enqueued_at, last_attempt_at "
            "FROM enrichment_queue ORDER BY enqueued_at ASC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]

    def clear_all(self) -> int:
        """DELETE FROM enrichment_queue；返回删除行数。"""
        cur = self._conn.execute("DELETE FROM enrichment_queue")
        return cur.rowcount

    def bulk_update_enrichment_status(
        self, from_status: str, to_status: str
    ) -> int:
        """批量 UPDATE facts SET enrichment_status。"""
        cur = self._conn.execute(
            "UPDATE facts SET enrichment_status = ?, "
            "updated_at = CURRENT_TIMESTAMP "
            "WHERE enrichment_status = ?",
            (to_status, from_status),
        )
        return cur.rowcount


# ─── ContradictionRepository ──────────────────────────────────────────────────


class ContradictionRepository:
    """contradiction_log 表的统一接口。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def log_contradiction(self, fact_a: int, fact_b: int, score: float) -> None:
        self._conn.execute(
            "INSERT INTO contradiction_log (fact_a, fact_b, score) VALUES (?, ?, ?)",
            (fact_a, fact_b, float(score)),
        )

    def get_open_pairs(self) -> set[tuple[int, int]]:
        rows = self._conn.execute(
            "SELECT fact_a, fact_b FROM contradiction_log WHERE resolved = 0"
        ).fetchall()
        return {(int(r["fact_a"]), int(r["fact_b"])) for r in rows}

    def list_contradictions(
        self,
        resolved: int = 0,
        category: str | None = None,
        threshold: float | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """四表 JOIN：contradiction_log + facts a + facts b。"""
        params: list[Any] = [int(resolved)]
        sql = (
            "SELECT cl.id AS cid, cl.fact_a, cl.fact_b, cl.score, cl.detected_at, "
            "       fa.content AS content_a, fb.content AS content_b, "
            "       fa.category AS category_a, fb.category AS category_b "
            "FROM contradiction_log cl "
            "JOIN facts fa ON fa.fact_id = cl.fact_a "
            "JOIN facts fb ON fb.fact_id = cl.fact_b "
            "WHERE cl.resolved = ?"
        )
        if threshold is not None:
            sql += " AND cl.score >= ?"
            params.append(float(threshold))
        if category is not None:
            sql += " AND (fa.category = ? OR fb.category = ?)"
            params.extend([category, category])
        sql += " ORDER BY cl.score DESC, cl.detected_at DESC LIMIT ?"
        params.append(int(limit))
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def unlink_loser_facts(self, fact_ids: list[int]) -> None:
        """purge 前清理 loser_fact_id FK 引用。"""
        if not fact_ids:
            return
        placeholders = ",".join("?" for _ in fact_ids)
        self._conn.execute(
            f"UPDATE contradiction_log SET loser_fact_id = NULL "
            f"WHERE loser_fact_id IN ({placeholders})",
            fact_ids,
        )

    def count_total(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM contradiction_log"
        ).fetchone()
        return int(row[0]) if row else 0

    def count_unresolved(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM contradiction_log WHERE resolved = 0"
        ).fetchone()
        return int(row[0]) if row else 0


# ─── StatsRepository ──────────────────────────────────────────────────────────


class StatsRepository:
    """prism_stats 表的统一接口。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM prism_stats WHERE key = ?", (key,)
        ).fetchone()
        return str(row["value"]) if row is not None else None

    def set(self, key: str, value: str) -> None:
        """UPSERT。"""
        self._conn.execute(
            "INSERT INTO prism_stats (key, value, updated_at) "
            "VALUES (?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(key) DO UPDATE SET "
            "value = excluded.value, updated_at = CURRENT_TIMESTAMP",
            (key, value),
        )

    def set_many(self, items: dict[str, str]) -> None:
        for key, value in items.items():
            self.set(key, value)


# ─── DatabaseRepository（聚合工厂）────────────────────────────────────────────


class DatabaseRepository:
    """聚合所有表仓库，单点获取 DB 访问。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.facts = FactsRepository(conn)
        self.entities = EntitiesRepository(conn)
        self.enrichment = EnrichmentQueueRepository(conn)
        self.contradiction = ContradictionRepository(conn)
        self.stats = StatsRepository(conn)
