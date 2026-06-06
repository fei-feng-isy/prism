"""``prism eval`` CLI。

跑评估集（jsonl）输出 P@k / R@k / MRR 与 must_include / must_exclude 通过率。

支持 ``naive``（substring baseline）和 ``prism``（端到端 recall.search）两种
retriever。``prism`` 模式每条 case 起独立 ``:memory:`` runtime 以避免 fact 集
互相污染。可选 ``--output`` 把 JSON 报告落盘。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from prism.cli.interface import single
from prism.eval import (
    AggregateMetrics,
    EvalCase,
    EvalReport,
    EvalSetError,
    RetrieveFn,
    evaluate_cases,
    load_cases,
    naive_substring_retriever,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from prism.eval.loader import EvalFact

log = logging.getLogger(__name__)

__all__ = ["build_prism_retriever", "main", "report_to_json"]


def report_to_json(report: EvalReport, *, dataset_path: Path | None = None) -> dict:
    """把 :class:`EvalReport` 序列化成 JSON dict — schema 与 ``stats().eval_baseline`` 对齐。"""
    s: AggregateMetrics = report.summary
    return {
        "dataset": str(dataset_path) if dataset_path else None,
        "n_cases": len(report.cases),
        "n_queries": s.n_queries,
        "mean_precision_at_k": s.mean_precision_at_k,
        "mean_recall_at_k": s.mean_recall_at_k,
        "mrr": s.mrr,
        "empty_rate": s.empty_rate,
        "must_include_pass_rate": s.must_include_pass_rate,
        "must_exclude_pass_rate": s.must_exclude_pass_rate,
    }


def build_prism_retriever() -> RetrieveFn:
    """构造一个 retriever：每条 query 起一个 ``:memory:`` runtime、灌 setup_facts
    再走 ``recall.search``。

    迟绑定 import：让仅用 ``naive`` 的调用者不付 prism runtime 启动成本（jieba
    + numpy + sqlite + bge 加载）。
    """
    from prism.mcp.wire import RuntimeOptions, build_runtime

    def retrieve(
        query: str, k: int, facts: Sequence[EvalFact]
    ) -> list[int]:
        runtime = build_runtime(
            RuntimeOptions(
                db_path_override=":memory:",
                start_worker=False,
                warmup_prefetch=False,
            )
        )
        try:
            # 灌 setup_facts，记录 fact_id → setup 下标的回译表
            fact_id_to_idx: dict[int, int] = {}
            for idx, f in enumerate(facts):
                category = f.category if f.category is not None else "general"
                res = runtime.mirror.mirror_add(
                    f.content,
                    metadata={"category": category},
                )
                if res is None:
                    log.warning("eval prism retriever: 空 content idx=%s", idx)
                    continue
                fact_id_to_idx[res.fact_id] = idx
            # 走 recall.search 结构化路径（as_markdown=False 直返 list[dict]）
            hits = runtime.recall.search(query, limit=k, as_markdown=False)
            assert isinstance(hits, list)  # narrow union for mypy
            out: list[int] = []
            for h in hits:
                fid = h.get("fact_id")
                if fid in fact_id_to_idx:
                    out.append(fact_id_to_idx[fid])
            return out
        finally:
            runtime.shutdown()

    return retrieve


def _format_summary(report: EvalReport) -> str:
    s = report.summary
    return (
        f"n_cases={len(report.cases)} n_queries={s.n_queries}\n"
        f"  P@k    = {s.mean_precision_at_k:.4f}\n"
        f"  R@k    = {s.mean_recall_at_k:.4f}\n"
        f"  MRR    = {s.mrr:.4f}\n"
        f"  empty_rate              = {s.empty_rate:.4f}\n"
        f"  must_include_pass_rate  = {s.must_include_pass_rate:.4f}\n"
        f"  must_exclude_pass_rate  = {s.must_exclude_pass_rate:.4f}"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m prism eval",
        description="跑评估集（jsonl）输出 P@k / R@k / MRR。",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="评估集 jsonl 路径（如 tests/fixtures/eval_set_zh.jsonl）",
    )
    p.add_argument(
        "--retriever",
        choices=["naive", "prism"],
        default="naive",
        help="检索器：naive（默认 substring baseline）或 prism（端到端 recall.search）",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选；把汇总 JSON 落盘到该路径（与 stats().eval_baseline 同 schema）",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="打印 per-query 明细（默认只汇总）",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not args.dataset.exists():
        print(f"错误：评估集不存在 {args.dataset}", file=sys.stderr)
        return 2

    try:
        cases: list[EvalCase] = list(load_cases(args.dataset))
    except EvalSetError as e:
        # 空文件 / schema 错都走这里：load_cases 对空文件抛 "不包含任何用例"
        print(f"错误：评估集为空或格式无效 {args.dataset}：{e}", file=sys.stderr)
        return 2

    retriever: RetrieveFn
    if args.retriever == "naive":
        retriever = naive_substring_retriever
    else:
        retriever = build_prism_retriever()

    print(
        f"评估：dataset={args.dataset} retriever={args.retriever} n_cases={len(cases)}",
        file=sys.stderr,
    )
    report = evaluate_cases(cases, retriever)

    if args.verbose:
        for pq in report.per_query:
            print(
                f"  [{pq.query}] P@{pq.k}={pq.precision_at_k:.3f} "
                f"R@{pq.k}={pq.recall_at_k:.3f} RR={pq.reciprocal_rank:.3f} "
                f"actual={list(pq.actual_ids)} expected={list(pq.expected_ids)}",
                file=sys.stderr,
            )

    print(_format_summary(report), file=sys.stderr)

    if args.output is not None:
        payload = report_to_json(report, dataset_path=args.dataset)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"汇总 JSON 已写入 {args.output}", file=sys.stderr)

    return 0


MANIFEST = single(
    "eval",
    lambda argv: main(argv),
    "eval                跑评估集输出 P@k / R@k / MRR",
)

if __name__ == "__main__":
    raise SystemExit(main())