"""``prism memory`` 人工维护 CLI。

提供 mirror / list / show / search / add / edit / remove / archive / restore /
helpful / unhelpful / stats / enrichment 子命令，全部委托 Service 层。

退出码：0 成功 / 1 部分失败 / 2 路径或 fact_id 不存在。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prism.cli.interface import single
from prism.config import (
    DEFAULT_PROFILE,
    DEFAULT_USER_ID,
    default_config,
    discover_config_path,
    load_config,
    resolve_db_path_for_user,
)
from prism.mcp.wire import RuntimeOptions, build_runtime
from prism.semantic import SemanticUnavailable

if TYPE_CHECKING:
    from prism.mcp.wire import PrismRuntime

log = logging.getLogger(__name__)

__all__ = [
    "main",
]


_CONTENT_SUMMARY_WIDTH = 80


def _format_list_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "（无匹配 fact）"
    headers = ("fact_id", "category", "status", "mirror_source", "content")
    widths = {
        "fact_id": max(len(headers[0]), max(len(str(r["fact_id"])) for r in rows)),
        "category": max(len(headers[1]), max(len(r["category"]) for r in rows)),
        "status": max(len(headers[2]), max(len(r["status"]) for r in rows)),
        "mirror_source": max(
            len(headers[3]), max(len(r["mirror_source"]) for r in rows)
        ),
    }
    lines = []
    header_line = (
        f"{headers[0]:>{widths['fact_id']}}  "
        f"{headers[1]:<{widths['category']}}  "
        f"{headers[2]:<{widths['status']}}  "
        f"{headers[3]:<{widths['mirror_source']}}  "
        f"{headers[4]}"
    )
    lines.append(header_line)
    lines.append("-" * len(header_line))
    for r in rows:
        lines.append(
            f"{r['fact_id']:>{widths['fact_id']}}  "
            f"{r['category']:<{widths['category']}}  "
            f"{r['status']:<{widths['status']}}  "
            f"{r['mirror_source']:<{widths['mirror_source']}}  "
            f"{r['content']}"
        )
    return "\n".join(lines)


def _format_show(fact: dict[str, Any]) -> str:
    lines = []
    for key in (
        "fact_id", "content", "category", "status", "mirror_source",
        "mirror_target", "supersedes_id", "trust_score", "helpful_count",
        "retrieval_count", "archived_at", "archive_reason",
        "enrichment_status", "created_at",
    ):
        lines.append(f"{key:<18}: {fact[key]}")
    lines.append(f"{'entities':<18}: {', '.join(fact['entities']) or '（无）'}")
    return "\n".join(lines)


# ─── argparse 构建 ───────────────────────────────────────────────────────────


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help="目标 DB 路径；未指定时按 --user-id / --profile + cfg.db.path_template 解析",
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
        help="可选 prism YAML 配置；未指定自动探测 <agent_home>/prism/ 或 ~/.prism/",
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m prism memory",
        description="Prism 记忆人工维护 CLI：MEMORY.md 镜像 + 查 / 改 / 删。",
    )
    sub = p.add_subparsers(dest="sub", required=True)

    pm = sub.add_parser(
        "mirror",
        help="从 MEMORY.md 文件级镜像（§ 分隔）",
    )
    pm.add_argument("--md", type=Path, required=True, help="MEMORY.md 路径")
    pm.add_argument(
        "--prune",
        action="store_true",
        help="跑完 add 后调 archive_ghost_facts 清孤儿（builtin_memory 来源）",
    )
    _add_common_args(pm)

    pl = sub.add_parser("list", help="浏览 facts 摘要（按 mirror_source / category / status 过滤）")
    pl.add_argument("--mirror-source", default=None, help="过滤 mirror_source（如 builtin_memory）")
    pl.add_argument("--category", default=None, help="过滤 category")
    pl.add_argument("--status", default="active", choices=["active", "archived"], help="过滤 status（默认 active）")
    pl.add_argument("--limit", type=int, default=50, help="单页条数（默认 50）")
    pl.add_argument("--offset", type=int, default=0, help="分页偏移（默认 0）")
    _add_common_args(pl)

    ps = sub.add_parser("show", help="按 fact_id 显示完整字段")
    ps.add_argument("fact_id", type=int, help="fact_id（整数主键）")
    _add_common_args(ps)

    psr = sub.add_parser("search", help="语义查询（PrismRecall.search 结构化返回）")
    psr.add_argument("query", type=str, help="查询文本")
    psr.add_argument("--limit", type=int, default=10, help="返回条数（默认 10）")
    psr.add_argument("--category", default=None, help="过滤 category")
    _add_common_args(psr)

    pe = sub.add_parser("edit", help="按 fact_id 软替换内容（旧归档 + 新建 supersedes 链）")
    pe.add_argument("fact_id", type=int, help="待替换的 fact_id")
    pe.add_argument("--content", required=True, help="新内容")
    pe.add_argument("--category", default=None, help="可选；不传则沿用 mirror 默认 category")
    _add_common_args(pe)

    pr = sub.add_parser("remove", help="按 fact_id 软删除（status='archived'）")
    pr.add_argument("fact_id", type=int, help="待软删的 fact_id")
    pr.add_argument("--reason", default="manual", help="归档原因（默认 manual）")
    _add_common_args(pr)

    pa = sub.add_parser("add", help="新增一条 fact")
    pa.add_argument("content", type=str, help="fact 内容")
    pa.add_argument("--category", default=None, help="可选 category")
    _add_common_args(pa)

    par = sub.add_parser("archive", help="按 fact_id 归档（与 remove 等价，支持自定义 reason）")
    par.add_argument("fact_id", type=int, help="待归档的 fact_id")
    par.add_argument("--reason", default="manual", help="归档原因（默认 manual）")
    _add_common_args(par)

    prs = sub.add_parser("restore", help="恢复已归档的 fact 为 active")
    prs.add_argument("fact_id", type=int, help="待恢复的 fact_id")
    _add_common_args(prs)

    ph = sub.add_parser("helpful", help="标记 fact 为 helpful（提升 trust_score）")
    ph.add_argument("fact_id", type=int, help="fact_id")
    _add_common_args(ph)

    pu = sub.add_parser("unhelpful", help="标记 fact 为 unhelpful（降低 trust_score）")
    pu.add_argument("fact_id", type=int, help="fact_id")
    _add_common_args(pu)

    pst = sub.add_parser("stats", help="显示 Prism 运行统计")
    pst.add_argument("--category", default=None, help="按 category 过滤")
    pst.add_argument("--json", action="store_true", dest="as_json", help="输出 JSON 格式")
    _add_common_args(pst)

    pen = sub.add_parser("enrichment", help="enrichment 诊断 / 修复")
    pen_group = pen.add_mutually_exclusive_group(required=True)
    pen_group.add_argument("--diagnose", action="store_true", help="诊断 enrichment 状态")
    pen_group.add_argument("--fix", action="store_true", help="修复缺失 embedding + 清理队列")
    pen.add_argument("--dry-run", action="store_true", help="仅预览（--fix 时有效）")
    pen.add_argument("--json", action="store_true", dest="as_json", help="输出 JSON 格式")
    _add_common_args(pen)


    return p


# ─── 子命令处理 ───────────────────────────────────────────────────────────────


def _cmd_mirror(args: argparse.Namespace, runtime: PrismRuntime) -> int:
    md: Path = args.md
    if not md.exists():
        print(f"错误：MEMORY.md 不存在 {md}", file=sys.stderr)
        return 2
    stats = runtime.admin_service.mirror_memory_md(md, prune=args.prune)
    print(
        f"完成：total={stats.total} added={stats.added} "
        f"skipped_duplicate={stats.skipped_duplicate} "
        f"failed={stats.failed} archived={stats.archived}",
        file=sys.stderr,
    )
    return 0 if stats.failed == 0 else 1


def _cmd_list(args: argparse.Namespace, runtime: PrismRuntime) -> int:
    result = runtime.fact_service.list(
        mirror_source=args.mirror_source,
        category=args.category,
        status=args.status,
        limit=args.limit,
        offset=args.offset,
    )
    rows = [
        {
            "fact_id": f.fact_id,
            "category": f.category,
            "status": f.status,
            "mirror_source": f.mirror_source or "",
            "content": f.content[:_CONTENT_SUMMARY_WIDTH],
        }
        for f in result.facts
    ]
    print(_format_list_table(rows))
    return 0


def _cmd_show(args: argparse.Namespace, runtime: PrismRuntime) -> int:
    detail = runtime.fact_service.show(args.fact_id)
    if detail is None:
        print(f"错误：fact_id={args.fact_id} 不存在", file=sys.stderr)
        return 2
    fact = {
        "fact_id": detail.fact_id,
        "content": detail.content,
        "category": detail.category,
        "status": detail.status,
        "mirror_source": detail.mirror_source,
        "mirror_target": detail.mirror_target,
        "supersedes_id": detail.supersedes_id,
        "trust_score": detail.trust_score,
        "helpful_count": detail.helpful_count,
        "retrieval_count": detail.retrieval_count,
        "archived_at": detail.archived_at,
        "archive_reason": detail.archive_reason,
        "enrichment_status": detail.enrichment_status,
        "created_at": detail.created_at,
        "entities": list(detail.entities),
    }
    print(_format_show(fact))
    return 0


def _cmd_search(args: argparse.Namespace, runtime: PrismRuntime) -> int:
    try:
        hits = runtime.search_service.search(
            args.query,
            limit=args.limit,
            category=args.category,
        )
    except SemanticUnavailable as e:
        print(f"警告：semantic backend 不可用，search 降级跳过：{e}", file=sys.stderr)
        return 1
    if not hits:
        print("（无匹配 fact）")
        return 0
    for h in hits:
        print(
            f"fact_id={h.fact_id} score={h.score:.3f}\n"
            f"  content: {h.content}\n"
            f"  path_scores: {h.path_scores}"
        )
    return 0


def _cmd_edit(args: argparse.Namespace, runtime: PrismRuntime) -> int:
    try:
        result = runtime.fact_service.edit(
            args.fact_id,
            content=args.content,
            category=args.category,
        )
    except LookupError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 2

    if not result.is_new:
        print(
            f"警告：新旧 content 相同，no-op（fact_id={result.fact_id}）",
            file=sys.stderr,
        )
        return 1

    print(
        f"完成：旧 fact_id={args.fact_id} 已归档 → 新 fact_id={result.fact_id} "
        f"category={result.category} entities={list(result.entities)}",
        file=sys.stderr,
    )
    return 0


def _cmd_remove(args: argparse.Namespace, runtime: PrismRuntime) -> int:
    r = runtime.fact_service.remove(args.fact_id, reason=args.reason)
    if not r.archived:
        print(f"错误：fact_id={args.fact_id} 不存在或已归档", file=sys.stderr)
        return 2
    print(
        f"完成：fact_id={r.fact_id} 已软删（archive_reason={args.reason}）",
        file=sys.stderr,
    )
    return 0


def _cmd_add(args: argparse.Namespace, runtime: PrismRuntime) -> int:
    try:
        result = runtime.fact_service.add(args.content, category=args.category)
    except ValueError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 2

    if not result.is_new:
        print(
            f"警告：content 已存在，no-op（fact_id={result.fact_id}）",
            file=sys.stderr,
        )
        return 1

    print(
        f"完成：fact_id={result.fact_id} category={result.category} "
        f"entities={list(result.entities)}",
        file=sys.stderr,
    )
    return 0


def _cmd_archive(args: argparse.Namespace, runtime: PrismRuntime) -> int:
    try:
        r = runtime.fact_service.archive(args.fact_id, reason=args.reason)
    except ValueError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 2

    if not r.archived:
        print(f"错误：fact_id={args.fact_id} 不存在", file=sys.stderr)
        return 2
    print(
        f"完成：fact_id={r.fact_id} 已归档（archive_reason={r.reason}）",
        file=sys.stderr,
    )
    return 0


def _cmd_restore(args: argparse.Namespace, runtime: PrismRuntime) -> int:
    try:
        r = runtime.fact_service.restore(args.fact_id)
    except ValueError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 2

    if not r.restored:
        print(
            f"错误：fact_id={args.fact_id} 不存在或已是 active",
            file=sys.stderr,
        )
        return 2
    print(
        f"完成：fact_id={r.fact_id} 已恢复为 active（category={r.category}）",
        file=sys.stderr,
    )
    return 0


def _cmd_helpful(args: argparse.Namespace, runtime: PrismRuntime) -> int:
    r = runtime.fact_service.helpful(args.fact_id)
    if not r.applied:
        print(f"错误：fact_id={args.fact_id} 不存在或不可标记", file=sys.stderr)
        return 2
    print(
        f"完成：fact_id={r.fact_id} helpful_count={r.new_helpful_count} "
        f"trust_score={r.new_trust_score:.3f}",
        file=sys.stderr,
    )
    return 0


def _cmd_unhelpful(args: argparse.Namespace, runtime: PrismRuntime) -> int:
    r = runtime.fact_service.unhelpful(args.fact_id)
    if not r.applied:
        print(f"错误：fact_id={args.fact_id} 不存在或不可标记", file=sys.stderr)
        return 2
    print(
        f"完成：fact_id={r.fact_id} helpful_count={r.new_helpful_count} "
        f"trust_score={r.new_trust_score:.3f}",
        file=sys.stderr,
    )
    return 0


def _cmd_stats(args: argparse.Namespace, runtime: PrismRuntime) -> int:
    data = runtime.admin_service.stats(category=args.category)
    if args.as_json:
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    else:
        for section, value in data.items():
            if isinstance(value, dict):
                print(f"\n[{section}]")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{section}: {value}")
    return 0


def _cmd_enrichment(args: argparse.Namespace, runtime: PrismRuntime) -> int:
    if args.diagnose:
        diag = runtime.admin_service.enrichment_diagnose()
        if args.as_json:
            from dataclasses import asdict

            print(json.dumps(asdict(diag), ensure_ascii=False, indent=2, default=str))
        else:
            print(f"queue_count: {diag.queue_count}")
            print(f"missing_vector_count: {diag.missing_vector_count}")
            print("\n[status_distribution]")
            for s in diag.status_distribution:
                print(f"  {s.enrichment_status}: count={s.count} null_vector={s.null_vector}")
            if diag.queue_items:
                print(f"\n[queue_items] (showing {len(diag.queue_items)})")
                for q in diag.queue_items:
                    print(
                        f"  fact_id={q.fact_id} attempts={q.attempts} "
                        f"last_error={q.last_error}"
                    )
            if diag.missing_vectors:
                print(f"\n[missing_vectors] (showing {len(diag.missing_vectors)})")
                for m in diag.missing_vectors:
                    print(f"  fact_id={m.fact_id} content={m.content}")
        return 0

    result = runtime.admin_service.enrichment_fix(dry_run=args.dry_run)
    if args.as_json:
        from dataclasses import asdict

        print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    else:
        print(f"dry_run: {result.dry_run}")
        print(f"semantic_available: {result.semantic_available}")
        if result.dry_run:
            print(f"to_embed: {result.to_embed}")
            print(f"pending_to_clear: {result.pending_to_clear}")
            print(f"queue_to_clear: {result.queue_to_clear}")
        else:
            print(f"embedded: {result.embedded}")
            print(f"embed_failed: {result.embed_failed}")
            print(f"pending_cleared: {result.pending_cleared}")
            print(f"queue_cleared: {result.queue_cleared}")
            print(f"vstore_rebuilt: {result.vstore_rebuilt}")
            print(f"vstore_vectors: {result.vstore_vectors}")
    return 0


# ─── CLI 入口 ────────────────────────────────────────────────────────────────


_DISPATCH = {
    "mirror": _cmd_mirror,
    "list": _cmd_list,
    "show": _cmd_show,
    "search": _cmd_search,
    "edit": _cmd_edit,
    "remove": _cmd_remove,
    "add": _cmd_add,
    "archive": _cmd_archive,
    "restore": _cmd_restore,
    "helpful": _cmd_helpful,
    "unhelpful": _cmd_unhelpful,
    "stats": _cmd_stats,
    "enrichment": _cmd_enrichment,
}


def main(argv: list[str] | None = None) -> int:
    """``python -m prism memory ...`` 入口；返 0 成功 / 1 部分失败 / 2 路径或 fact_id 不存在。"""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg_path = args.config or discover_config_path()
    cfg = load_config(cfg_path) if cfg_path else default_config()

    if args.db is not None:
        db_path = Path(args.db)
    else:
        db_path = resolve_db_path_for_user(
            cfg.db, user_id=args.user_id, profile=args.profile,
            data_home=args.data_home,
        )

    runtime = build_runtime(
        RuntimeOptions(
            db_path_override=str(db_path),
            start_worker=False,
            warmup_prefetch=False,
            call_source="cli",
        )
    )
    try:
        handler = _DISPATCH[args.sub]
        return handler(args, runtime)
    finally:
        runtime.shutdown()


MANIFEST = single(
    "memory",
    lambda argv: main(argv),
    (
        "memory              记忆人工维护：\n"
        "    mirror              从 MEMORY.md 文件级批量镜像\n"
        "    list                按 source / category / status 过滤浏览\n"
        "    show <fact_id>      显示完整字段 + 实体\n"
        "    search <query>      语义查询\n"
        "    add <content>       新增 fact\n"
        "    edit <fact_id>      软替换内容（旧归档 + 新建 supersedes）\n"
        "    remove <fact_id>    软删除\n"
        "    archive <fact_id>   归档（同 remove，支持自定义 reason）\n"
        "    restore <fact_id>   恢复已归档 fact 为 active\n"
        "    helpful <fact_id>   标记 helpful（提升 trust_score）\n"
        "    unhelpful <fact_id> 标记 unhelpful（降低 trust_score）\n"
        "    stats               运行统计（--json 输出 JSON）\n"
        "    enrichment          enrichment 诊断 / 修复（--diagnose / --fix）"
    ),
)

if __name__ == "__main__":
    raise SystemExit(main())