"""``prism export`` CLI。

把全库（或按 category / status 过滤的子集）导出为 jsonl，不含向量 blob。
流式输出，纯 SQLite SELECT，不需要起完整 runtime。
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from prism.cli.interface import single
from prism.config import (
    DEFAULT_PROFILE,
    DEFAULT_USER_ID,
    default_config,
    discover_config_path,
    load_config,
    resolve_db_path_for_user,
)

log = logging.getLogger(__name__)

__all__ = ["export_facts", "iter_fact_rows", "main"]


# 导出的列（不含向量 blob — 见模块 docstring）
_EXPORT_COLUMNS = (
    "fact_id, content, category, tags, status, "
    "trust_score, helpful_count, retrieval_count, ttl_days, "
    "created_at, last_retrieved_at, archived_at, archive_reason, "
    "supersedes_id, embedding_model, vector_store, "
    "mirror_source, mirror_target, enrichment_status"
)

_VALID_STATUS = frozenset({"active", "archived", "all"})


def iter_fact_rows(
    db_path: Path,
    *,
    status: str = "active",
    category: str | None = None,
) -> Iterator[dict[str, Any]]:
    """流式迭代 facts — 不一次 load 全表。

    Args:
        db_path: 目标 DB（只读 URI）
        status: ``'active'`` / ``'archived'`` / ``'all'``
        category: 限定 category；None 不过滤

    Yields:
        每行 dict，含 _EXPORT_COLUMNS 字段 + ``entities: list[str]``
    """
    if status not in _VALID_STATUS:
        raise ValueError(
            f"status={status!r} 非法（合法集合：{sorted(_VALID_STATUS)}）"
        )

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        clauses: list[str] = []
        params: list[Any] = []
        if status != "all":
            clauses.append("status = ?")
            params.append(status)
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT {_EXPORT_COLUMNS} FROM facts{where} ORDER BY fact_id ASC"

        # 预编译一条按 fact_id 查 entities 的语句（避免每行重复 prepare）
        entity_sql = (
            "SELECT e.name FROM fact_entities fe "
            "JOIN entities e ON e.entity_id = fe.entity_id "
            "WHERE fe.fact_id = ? ORDER BY e.name"
        )

        for row in conn.execute(sql, params):
            d = dict(row)
            d["entities"] = [
                str(r["name"]) for r in conn.execute(entity_sql, (int(row["fact_id"]),))
            ]
            yield d
    finally:
        conn.close()


def export_facts(
    db_path: Path,
    output: Path | None,
    *,
    status: str = "active",
    category: str | None = None,
) -> int:
    """把 facts 写到 ``output`` jsonl（None 写 stdout）。返回写出行数。"""
    count = 0
    fh: Any
    if output is None:
        fh = sys.stdout
        close = False
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        fh = output.open("w", encoding="utf-8")
        close = True
    try:
        for row in iter_fact_rows(db_path, status=status, category=category):
            fh.write(json.dumps(row, ensure_ascii=False, default=str))
            fh.write("\n")
            count += 1
    finally:
        if close:
            fh.close()
    return count


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m prism export",
        description="把 Prism facts 导出为 jsonl（不含向量 blob）。",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help="目标 DB 路径；未指定时按 cfg.db.path_template + user_id/profile 解析",
    )
    p.add_argument(
        "--user-id",
        default=DEFAULT_USER_ID,
        help=f"目标 user_id（默认 {DEFAULT_USER_ID!r}）— 仅 --db 未指定时生效",
    )
    p.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help=f"目标 profile（默认 {DEFAULT_PROFILE!r}）— 仅 --db 未指定时生效",
    )
    p.add_argument(
        "--data-home",
        default=None,
        help="数据根目录覆盖（默认 ~/.prism）— 仅 --db 未指定时生效",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="可选 prism YAML 配置；未指定走 default_config()",
    )
    p.add_argument(
        "--status",
        choices=sorted(_VALID_STATUS),
        default="active",
        help="状态过滤（默认 active）",
    )
    p.add_argument(
        "--category",
        default=None,
        help="category 过滤（默认全部）",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 jsonl 路径；未指定写 stdout",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.db is not None:
        db_path = args.db
    else:
        cfg_path = args.config or discover_config_path()
        cfg = load_config(cfg_path) if cfg_path else default_config()
        db_path = resolve_db_path_for_user(
            cfg.db, user_id=args.user_id, profile=args.profile,
            data_home=args.data_home,
        )

    if not db_path.exists():
        print(f"错误：DB 不存在 {db_path}", file=sys.stderr)
        return 2

    count = export_facts(
        db_path, args.output, status=args.status, category=args.category
    )
    if args.output is not None:
        print(
            f"已导出 {count} 行到 {args.output}（status={args.status} "
            f"category={args.category or '*'}）",
            file=sys.stderr,
        )
    return 0


MANIFEST = single(
    "export",
    lambda argv: main(argv),
    "export              导出全库为 jsonl（不含向量）",
)

if __name__ == "__main__":
    raise SystemExit(main())