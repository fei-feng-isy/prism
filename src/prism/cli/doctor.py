"""``prism doctor`` 健康自检 CLI。

逐项验证环境、依赖、Hermes 集成、Config / DB、语义后端、end-to-end smoke，
输出 ✅/⚠️/❌/ℹ️ 报告 + 退出码。

退出码：0 全部 ok / 1 至少一项 warn / 2 至少一项 fail。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import platform
import sqlite3
import sys
import tempfile
import time
import traceback
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from prism import __version__
from prism.cli.interface import single
from prism.config import (
    ConfigError,
    DEFAULT_PROFILE,
    DEFAULT_USER_ID,
    default_config,
    discover_config_path,
    load_config,
    resolve_db_path_for_user,
)
from prism.semantic.backend import check_sentence_transformers_available

log = logging.getLogger(__name__)

__all__ = ["CheckResult", "DoctorReport", "main", "run_doctor"]


# ─── 数据结构 ────────────────────────────────────────────────────────────────


_STATUS_OK = "ok"
_STATUS_WARN = "warn"
_STATUS_FAIL = "fail"
_STATUS_INFO = "info"
_STATUS_SKIP = "skip"

_STATUS_GLYPH = {
    _STATUS_OK: "✅",
    _STATUS_WARN: "⚠️ ",
    _STATUS_FAIL: "❌",
    _STATUS_INFO: "──",
    _STATUS_SKIP: "·",
}


@dataclass(frozen=True, slots=True)
class CheckResult:
    """单项检查结果。"""

    name: str
    status: str  # _STATUS_*
    detail: str
    hint: str = ""


@dataclass
class DoctorReport:
    """一次 doctor run 的全部结果，按组聚合。"""

    groups: list[tuple[str, list[CheckResult]]] = field(default_factory=list)

    def add(self, group: str, result: CheckResult) -> None:
        for name, items in self.groups:
            if name == group:
                items.append(result)
                return
        self.groups.append((group, [result]))

    def exit_code(self) -> int:
        has_fail = False
        has_warn = False
        for _, items in self.groups:
            for r in items:
                if r.status == _STATUS_FAIL:
                    has_fail = True
                elif r.status == _STATUS_WARN:
                    has_warn = True
        if has_fail:
            return 2
        if has_warn:
            return 1
        return 0

    def summary(self) -> str:
        counts = {_STATUS_OK: 0, _STATUS_WARN: 0, _STATUS_FAIL: 0, _STATUS_INFO: 0, _STATUS_SKIP: 0}
        for _, items in self.groups:
            for r in items:
                counts[r.status] = counts.get(r.status, 0) + 1
        if counts[_STATUS_FAIL]:
            verdict = f"无法工作 — {counts[_STATUS_FAIL]} 处核心错误"
        elif counts[_STATUS_WARN]:
            verdict = f"降级运行 — {counts[_STATUS_WARN]} 处需关注"
        else:
            verdict = "全部正常"
        return (
            f"{verdict}（ok={counts[_STATUS_OK]} warn={counts[_STATUS_WARN]} "
            f"fail={counts[_STATUS_FAIL]} info={counts[_STATUS_INFO]}）"
        )

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "version": __version__,
            "exit_code": self.exit_code(),
            "summary": self.summary(),
            "groups": [
                {
                    "name": name,
                    "checks": [asdict(r) for r in items],
                }
                for name, items in self.groups
            ],
        }


# ─── 通用工具 ────────────────────────────────────────────────────────────────


def _module_version(name: str) -> str | None:
    """返回模块版本号；缺包或无 __version__ 返 None。**不副作用 import** 大包。

    用 ``importlib.metadata.version`` 避免触发 sentence_transformers 全套 80 包加载
    （参见 backend.py:117 同款离线优先思路）。
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _has_module(name: str) -> bool:
    """``importlib.util.find_spec`` — 探查包是否存在，不执行 import。"""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _hermes_home() -> Path:
    """Hermes 诊断专用 — ``$HERMES_HOME`` 或回退 ``~/.hermes``。"""
    return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()


# ─── 检查组 1: 环境 ──────────────────────────────────────────────────────────


def _check_env(report: DoctorReport) -> None:
    group = "环境"

    # Python 解释器
    py = f"Python {platform.python_version()} @ {sys.executable}"
    report.add(group, CheckResult("Python 解释器", _STATUS_OK, py))

    # 平台
    report.add(group, CheckResult("操作系统", _STATUS_INFO, platform.platform()))

    # HERMES_HOME
    hermes_home = _hermes_home()
    if hermes_home.exists():
        report.add(group, CheckResult(
            "HERMES_HOME", _STATUS_OK,
            f"{hermes_home} (存在)",
        ))
    else:
        report.add(group, CheckResult(
            "HERMES_HOME", _STATUS_WARN,
            f"{hermes_home} 不存在",
            hint=f"mkdir -p {hermes_home}（或 export HERMES_HOME=/path/to/your/hermes）",
        ))

    # 是否在 hermes venv
    venv_path = hermes_home / "hermes-agent" / "venv"
    in_hermes_venv = (
        venv_path.exists()
        and Path(sys.prefix).resolve() == venv_path.resolve()
    )
    if venv_path.exists():
        if in_hermes_venv:
            report.add(group, CheckResult(
                "hermes venv", _STATUS_OK,
                f"当前 Python 来自 {venv_path}",
            ))
        else:
            report.add(group, CheckResult(
                "hermes venv", _STATUS_WARN,
                f"当前 sys.prefix={sys.prefix} 不在 {venv_path}",
                hint=(
                    "hermes 启动 prism 时用的是 hermes-agent venv —"
                    f"请 `source {venv_path}/bin/activate` 后再 `pip install -e .[semantic]`，"
                    "否则在宿主 Python 装的依赖 hermes 看不到"
                ),
            ))
    else:
        report.add(group, CheckResult(
            "hermes venv", _STATUS_INFO,
            f"{venv_path} 不存在（独立使用 prism 时可忽略）",
        ))


# ─── 检查组 2: 必备依赖 ──────────────────────────────────────────────────────


_REQUIRED_DEPS: list[tuple[str, str, str]] = [
    # (import_name, distribution_name, pip_hint)
    ("numpy", "numpy", "pip install numpy"),
    ("yaml", "PyYAML", "pip install pyyaml"),
    ("jieba", "jieba", "pip install jieba"),
]


def _check_required_deps(report: DoctorReport) -> None:
    group = "必备依赖"

    # sqlite3 是 stdlib，但万一被裁剪
    try:
        report.add(group, CheckResult(
            "sqlite3", _STATUS_OK, f"{sqlite3.sqlite_version}（stdlib）",
        ))
    except Exception as e:  # pragma: no cover
        report.add(group, CheckResult(
            "sqlite3", _STATUS_FAIL, f"stdlib sqlite3 异常：{e}",
            hint="Python 安装缺少 sqlite3 支持，需重装 Python（apt install python3-dev libsqlite3-dev）",
        ))

    for import_name, dist_name, hint in _REQUIRED_DEPS:
        if _has_module(import_name):
            ver = _module_version(dist_name) or "未知版本"
            report.add(group, CheckResult(import_name, _STATUS_OK, ver))
        else:
            report.add(group, CheckResult(
                import_name, _STATUS_FAIL,
                f"未安装",
                hint=hint,
            ))


# ─── 检查组 3: 可选依赖 ──────────────────────────────────────────────────────


def _check_optional_deps(report: DoctorReport, cfg_vector_backend: str | None) -> None:
    group = "可选依赖"

    # sentence-transformers
    if check_sentence_transformers_available():
        ver = _module_version("sentence-transformers") or "未知版本"
        report.add(group, CheckResult(
            "sentence-transformers", _STATUS_OK, ver,
        ))
    else:
        report.add(group, CheckResult(
            "sentence-transformers", _STATUS_WARN,
            "未安装（语义检索降级为 FTS+Jaccard，权重 0.65/0.35）",
            hint="pip install 'prism-memory[semantic]'",
        ))

    # vstore backends — 配置选了哪个就 fail 哪个；其他只 info
    vstore_packages = [
        ("hnswlib", "hnswlib", "hnswlib"),
        ("faiss", "faiss-cpu", "faiss"),
        ("qdrant_client", "qdrant-client", "qdrant"),
        ("psycopg", "psycopg", "pgvector"),
    ]
    for import_name, dist_name, backend_alias in vstore_packages:
        has = _has_module(import_name)
        required = cfg_vector_backend == backend_alias
        if has:
            ver = _module_version(dist_name) or "未知版本"
            report.add(group, CheckResult(
                import_name, _STATUS_OK, f"{ver} (可作 {backend_alias} backend)",
            ))
        elif required:
            report.add(group, CheckResult(
                import_name, _STATUS_FAIL,
                f"未安装但 cfg.vector_store.backend = {backend_alias!r}",
                hint=f"pip install 'prism-memory[{backend_alias}]'",
            ))
        else:
            report.add(group, CheckResult(
                import_name, _STATUS_INFO, f"未安装（cfg 未选择 {backend_alias}，可忽略）",
            ))

    # mcp
    if _has_module("mcp"):
        ver = _module_version("mcp") or "未知版本"
        report.add(group, CheckResult("mcp", _STATUS_OK, ver))
    else:
        report.add(group, CheckResult(
            "mcp", _STATUS_INFO,
            "未安装（仅 MCP server 模式需要）",
        ))


# ─── 检查组 4: Hermes 集成（纯文件级） ───────────────────────────────────────


def _check_hermes_integration(report: DoctorReport) -> None:
    group = "Hermes 集成"

    # 插件是否已安装 — 检查 $HERMES_HOME/plugins/prism/plugin.yaml 是否可达
    plugin_link = _hermes_home() / "plugins" / "prism"
    link_yaml = plugin_link / "plugin.yaml"

    if not plugin_link.exists() and not plugin_link.is_symlink():
        report.add(group, CheckResult(
            "插件安装", _STATUS_FAIL,
            "未安装",
            hint="运行 `hermes memory setup prism` 安装",
        ))
    elif not link_yaml.exists():
        report.add(group, CheckResult(
            "插件安装", _STATUS_WARN,
            f"{plugin_link} 存在但 plugin.yaml 不可见",
            hint="插件目录布局异常，尝试重新安装：`hermes memory setup prism`",
        ))
    else:
        try:
            import yaml
            meta = yaml.safe_load(link_yaml.read_text(encoding="utf-8")) or {}
            ver = meta.get("version", "未知")
        except Exception:
            ver = "未知"
        report.add(group, CheckResult(
            "插件安装", _STATUS_OK, f"已安装（version={ver}）",
        ))

    # memory.provider — 直接读 hermes config.yaml 判断
    hermes_cfg_path = _hermes_home() / "config.yaml"
    if hermes_cfg_path.exists():
        try:
            import yaml
            hermes_cfg = yaml.safe_load(hermes_cfg_path.read_text(encoding="utf-8")) or {}
            provider = (hermes_cfg.get("memory") or {}).get("provider", "")
            if provider == "prism":
                report.add(group, CheckResult(
                    "memory.provider", _STATUS_OK,
                    f"memory.provider = prism（{hermes_cfg_path}）",
                ))
            elif provider:
                report.add(group, CheckResult(
                    "memory.provider", _STATUS_WARN,
                    f"memory.provider = {provider!r}（期望 prism）",
                    hint="运行 `hermes memory setup prism` 切换",
                ))
            else:
                report.add(group, CheckResult(
                    "memory.provider", _STATUS_WARN,
                    "memory.provider 未设置",
                    hint="运行 `hermes memory setup prism` 激活",
                ))
        except Exception as e:
            report.add(group, CheckResult(
                "memory.provider", _STATUS_WARN,
                f"读取 {hermes_cfg_path} 失败：{e}",
            ))
    else:
        report.add(group, CheckResult(
            "memory.provider", _STATUS_INFO,
            f"{hermes_cfg_path} 不存在（独立使用 prism 时可忽略）",
        ))


# ─── 检查组 5: Config / DB ───────────────────────────────────────────────────


def _check_config_db(
    report: DoctorReport,
    config_path: Path | None,
    data_home: str | None = None,
) -> tuple[Any, Path | None]:
    """返回 (cfg or None, db_path or None)；后续 smoke 用。"""
    group = "Config / DB"
    cfg = None
    db_path = None

    # 1. config.yaml
    if config_path is not None and config_path.exists():
        try:
            cfg = load_config(config_path)
            report.add(group, CheckResult(
                "config.yaml", _STATUS_OK,
                f"{config_path}（解析成功）",
            ))
        except ConfigError as e:
            report.add(group, CheckResult(
                "config.yaml", _STATUS_FAIL,
                f"{config_path} 解析失败：{e}",
                hint=f"修复 YAML 语法或删除 {config_path} 让插件重新生成默认模板",
            ))
            cfg = default_config()
    else:
        cfg = default_config()
        if config_path is not None:
            report.add(group, CheckResult(
                "config.yaml", _STATUS_INFO,
                f"{config_path} 不存在 — 走 default_config()",
            ))
        else:
            report.add(group, CheckResult(
                "config.yaml", _STATUS_INFO,
                "未指定 --config — 走 default_config()",
            ))

    # 2. DB 路径解析（不连接）
    try:
        db_path = resolve_db_path_for_user(
            cfg.db, user_id=DEFAULT_USER_ID, profile=DEFAULT_PROFILE,
            data_home=data_home,
        )
        report.add(group, CheckResult(
            "DB 路径", _STATUS_INFO,
            f"{db_path}（profile=default user_id={DEFAULT_USER_ID}）",
        ))
    except Exception as e:
        report.add(group, CheckResult(
            "DB 路径", _STATUS_FAIL,
            f"解析失败：{type(e).__name__}: {e}",
            hint="检查 cfg.db.path_template / data_home_default",
        ))
        return cfg, None

    # 3. DB 文件可访问性
    if not db_path.exists():
        report.add(group, CheckResult(
            "DB 文件", _STATUS_INFO,
            f"{db_path} 尚未创建（首次启动正常，hermes 拉起 prism 后自动 bootstrap）",
        ))
    else:
        try:
            # 只读 URI 探测，避免锁库
            uri = f"file:{db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            try:
                row = conn.execute("PRAGMA user_version").fetchone()
                facts_count = conn.execute(
                    "SELECT COUNT(*) FROM facts WHERE status='active'"
                ).fetchone()[0]
            finally:
                conn.close()
            report.add(group, CheckResult(
                "DB 文件", _STATUS_OK,
                f"{db_path} (user_version={row[0]} active_facts={facts_count})",
            ))
        except sqlite3.OperationalError as e:
            report.add(group, CheckResult(
                "DB 文件", _STATUS_FAIL,
                f"{db_path} 无法打开：{e}",
                hint="检查文件权限 / 路径 / 是否被其它进程独占",
            ))

    # 4. vstore.npz
    vstore_path = db_path.with_suffix(".vstore.npz")
    if vstore_path.exists():
        size = vstore_path.stat().st_size
        report.add(group, CheckResult(
            "vstore 持久化", _STATUS_OK,
            f"{vstore_path} ({size:,} bytes)",
        ))
    else:
        report.add(group, CheckResult(
            "vstore 持久化", _STATUS_INFO,
            f"{vstore_path} 尚未生成（首次写入后自动落盘）",
        ))

    return cfg, db_path


# ─── 检查组 6: 语义后端 ──────────────────────────────────────────────────────


def _check_semantic(report: DoctorReport, cfg: Any) -> None:
    group = "语义后端"

    backend_name = cfg.semantic.backend
    model_name = cfg.semantic.local_model
    report.add(group, CheckResult(
        "配置", _STATUS_INFO,
        f"backend={backend_name} model={model_name}",
    ))

    # HF cache 探测（复用 local_bge._has_hf_cache）
    if not check_sentence_transformers_available():
        report.add(group, CheckResult(
            "BGE HF cache", _STATUS_SKIP,
            "sentence-transformers 缺失，已走降级，无需探测",
        ))
    else:
        try:
            from prism.semantic.local_bge import _has_hf_cache
            cached = _has_hf_cache(model_name)
        except Exception as e:
            report.add(group, CheckResult(
                "BGE HF cache", _STATUS_WARN,
                f"探测异常：{type(e).__name__}: {e}",
                hint="huggingface_hub 未装；pip install huggingface_hub",
            ))
        else:
            if cached:
                report.add(group, CheckResult(
                    "BGE HF cache", _STATUS_OK,
                    f"已下载 {model_name} 到本地 cache",
                ))
            else:
                report.add(group, CheckResult(
                    "BGE HF cache", _STATUS_INFO,
                    f"{model_name} 未在 HF cache（首次 encode 时自动下载约 95MB）",
                ))

    # HF endpoint 策略
    strategy = cfg.semantic.hf_endpoint_strategy
    env_endpoint = os.environ.get("HF_ENDPOINT")
    detail = f"strategy={strategy}"
    if env_endpoint:
        detail += f" HF_ENDPOINT={env_endpoint}（用户已显式设置，策略不覆盖）"
    elif strategy in ("mirror_first", "mirror_only"):
        detail += f" mirror_url={cfg.semantic.hf_mirror_url}"
    report.add(group, CheckResult("HF endpoint", _STATUS_INFO, detail))


# ─── 检查组 7: End-to-end smoke ──────────────────────────────────────────────


def _check_smoke(report: DoctorReport) -> None:
    """临时 DB 跑一遍 add → search，验证完整链路。零侵入用户库。"""
    group = "End-to-end smoke"

    # 导入放函数内，避免 doctor 启动期就拉起 wire（重）
    try:
        from prism.mcp.wire import RuntimeOptions, build_runtime
    except Exception as e:
        report.add(group, CheckResult(
            "wire import", _STATUS_FAIL,
            f"{type(e).__name__}: {e}",
            hint="prism 包导入异常 — 检查 pip install -e . 是否成功",
        ))
        return

    with tempfile.TemporaryDirectory(prefix="prism-doctor-") as tmp:
        tmp_db = Path(tmp) / "smoke.db"
        runtime = None
        t0 = time.monotonic()
        try:
            runtime = build_runtime(
                RuntimeOptions(
                    db_path_override=str(tmp_db),
                    start_worker=False,
                    warmup_prefetch=False,
                )
            )
            content = "prism doctor smoke fact"
            add_result = runtime.remember(
                "add", content=content, category="general",
            )
            elapsed_add = (time.monotonic() - t0) * 1000

            if not isinstance(add_result, dict) or "fact_id" not in add_result:
                report.add(group, CheckResult(
                    "smoke add", _STATUS_FAIL,
                    f"remember(add) 返回异常 schema：{add_result!r}",
                ))
                return

            fact_id = add_result["fact_id"]
            report.add(group, CheckResult(
                "smoke add", _STATUS_OK,
                f"fact_id={fact_id} 写入成功（耗时 {elapsed_add:.0f}ms）",
            ))

            # search 走结构化路径（不依赖 markdown 渲染）
            t1 = time.monotonic()
            try:
                results = runtime.recall(
                    "search",
                    query="doctor smoke",
                    limit=3,
                    as_markdown=False,
                )
            except Exception as e:
                # 语义不可用时 search 应该走降级（fts+jaccard）而不抛；抛了说明降级路径也坏了
                report.add(group, CheckResult(
                    "smoke search", _STATUS_FAIL,
                    f"recall(search) 异常：{type(e).__name__}: {e}",
                    hint=(
                        "降级路径（fts+jaccard）也无法工作；"
                        "查 logs 看 mirror_add 是否真把 fact 写进 facts_fts 表"
                    ),
                ))
                return
            elapsed_search = (time.monotonic() - t1) * 1000

            hit = any(
                isinstance(r, dict) and r.get("fact_id") == fact_id
                for r in (results or [])
            )
            if hit:
                report.add(group, CheckResult(
                    "smoke search", _STATUS_OK,
                    f"召回命中 fact_id={fact_id}（耗时 {elapsed_search:.0f}ms，"
                    f"共 {len(results)} 条结果）",
                ))
            else:
                report.add(group, CheckResult(
                    "smoke search", _STATUS_WARN,
                    f"add 成功但 search 未命中（{len(results or [])} 条无关结果，耗时 {elapsed_search:.0f}ms）",
                    hint=(
                        "可能是降级权重 + 短 query 命中率低；检查 logs 中 RetrievalPipeline 的 path_scores"
                    ),
                ))
        except Exception as e:
            report.add(group, CheckResult(
                "smoke run", _STATUS_FAIL,
                f"{type(e).__name__}: {e}",
                hint=(
                    "build_runtime 或 remember/recall 抛异常 — "
                    "用 `python -m prism doctor --no-smoke` 跳过此项排查其它问题，"
                    f"完整 trace: {traceback.format_exc().splitlines()[-1] if e else ''}"
                ),
            ))
        finally:
            if runtime is not None:
                with suppress(Exception):
                    runtime.shutdown()


# ─── 渲染 ────────────────────────────────────────────────────────────────────


def _render_text(report: DoctorReport) -> str:
    """人类可读的分组报告。"""
    lines: list[str] = []
    lines.append(f"Prism Doctor v{__version__}")
    lines.append("")
    for group_name, items in report.groups:
        lines.append(f"[{group_name}]")
        # 计算 name 列宽以对齐 detail
        max_name = max(len(r.name) for r in items) if items else 0
        for r in items:
            glyph = _STATUS_GLYPH.get(r.status, "?")
            lines.append(f"  {glyph} {r.name:<{max_name}}  {r.detail}")
            if r.hint and r.status in (_STATUS_WARN, _STATUS_FAIL):
                lines.append(f"     → {r.hint}")
        lines.append("")
    lines.append("─" * 60)
    lines.append(f"总结: {report.summary()}")
    lines.append(f"退出码: {report.exit_code()}")
    return "\n".join(lines)


# ─── 入口 ────────────────────────────────────────────────────────────────────


def run_doctor(
    *,
    config_path: Path | None = None,
    skip_smoke: bool = False,
) -> DoctorReport:
    """跑完整自检并返回 DoctorReport。供测试 / 外部代码复用。"""
    report = DoctorReport()
    _check_env(report)
    _check_required_deps(report)
    cfg, _db_path = _check_config_db(report, config_path)
    _check_optional_deps(report, cfg.vector_store.backend if cfg else None)
    _check_hermes_integration(report)
    if cfg is not None:
        _check_semantic(report, cfg)
    if skip_smoke:
        report.add(
            "End-to-end smoke",
            CheckResult("smoke", _STATUS_SKIP, "已通过 --no-smoke 跳过"),
        )
    else:
        _check_smoke(report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m prism doctor",
        description="Prism 健康自检 — 依赖 / 配置 / DB / 语义 / Hermes 集成 / 端到端",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="可选 prism YAML 配置；未指定自动探测 <agent_home>/prism/ 或 ~/.prism/",
    )
    p.add_argument(
        "--no-smoke",
        action="store_true",
        help="跳过 end-to-end 临时 DB 写入+召回测试（CI 或不想加载 BGE 时用）",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="输出 JSON 而非人类可读文本（脚本 / LLM agent 消费）",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # doctor 自己也是诊断工具 — 默认静默 INFO 噪音，让输出干净
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    config_path: Path | None = args.config
    if config_path is None:
        config_path = discover_config_path()

    report = run_doctor(config_path=config_path, skip_smoke=args.no_smoke)

    if args.json:
        print(json.dumps(report.to_json_obj(), ensure_ascii=False, indent=2))
    else:
        print(_render_text(report))

    return report.exit_code()


MANIFEST = single(
    "doctor",
    lambda argv: main(argv),
    "doctor              健康自检（依赖 / 配置 / DB / 语义后端 / 端到端 smoke）",
)

if __name__ == "__main__":
    raise SystemExit(main())