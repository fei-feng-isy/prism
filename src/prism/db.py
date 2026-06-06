"""Prism 数据层 — SQLite schema + 连接管理。

每个 profile/user 一个独立 .db 文件。FTS5 使用 trigram tokenizer（对中日韩友好），
facts_fts 通过 trigger 与 facts 主表自动同步。连接默认开启 foreign_keys / WAL /
busy_timeout=1s。路径解析见 :mod:`prism.config.paths`。
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Final

__all__ = [
    "DDL_STATEMENTS",
    "FTS_TRIGGERS",
    "MEMORY_DB",
    "SCHEMA_VERSION",
    "DatabaseError",
    "bootstrap",
    "connect",
    "init_schema",
    "schema_version",
    "verify_chinese_tokenizer",
]


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
