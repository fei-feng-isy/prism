"""Prism 配置文件热补丁 — 已有 YAML 缺少新版字段时自动追加默认值 + 注释。"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path

import yaml
from dataclasses import asdict

from .schema import CallTrackingConfig

__all__ = ["patch_config_file"]


# 每个条目：(YAML 路径元组, 注释, 默认值 YAML 片段生成函数)
# 路径长度 1 = 顶层 key；长度 2 = 二级 key（父 key.子 key）

_SECTION_PATCHES: list[tuple[tuple[str, ...], str, Callable[[], str]]] = [
    (
        ("logging", "call_tracking"),
        "# call_tracking — API 调用追踪（频次/耗时/来源），可关闭以节省 IO\n"
        "#   enabled: true/false        — 总开关\n"
        "#   file_logging: true/false    — 文件日志开关（false 仅内存统计）\n"
        "#   env: PRISM_CALL_TRACKING_ENABLED / PRISM_CALL_TRACKING_FILE\n",
        lambda: yaml.safe_dump(
            asdict(CallTrackingConfig()),
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ),
    ),
]


def patch_config_file(path: str | os.PathLike[str]) -> list[str]:
    """检查已有 YAML 配置文件，补全缺失的配置段（带注释）。

    不修改用户已有的值，仅追加缺失段。返回本次新增的段名列表。

    Args:
        path: 配置文件路径

    Returns:
        新增段名列表（如 ``["logging.call_tracking"]``）；无需补全返空列表
    """
    p = Path(path)
    if not p.exists():
        return []

    try:
        text = p.read_text(encoding="utf-8")
        raw = yaml.safe_load(text) or {}
    except (OSError, yaml.YAMLError):
        return []

    if not isinstance(raw, dict):
        return []

    # 兼容 `prism:` 嵌套
    if "prism" in raw and len(raw) == 1 and isinstance(raw["prism"], dict):
        root = raw["prism"]
    else:
        root = raw

    added: list[str] = []
    appended = text

    for key_path, comment, default_fn in _SECTION_PATCHES:
        if len(key_path) == 1:
            if key_path[0] not in root:
                snippet = default_fn()
                block = f"\n{comment}{key_path[0]}:\n"
                for line in snippet.strip().splitlines():
                    block += f"  {line}\n"
                appended += block
                added.append(key_path[0])
        elif len(key_path) == 2:
            parent_key, child_key = key_path
            parent = root.get(parent_key)
            if parent is None:
                # 父 key 也不存在 → 追加完整段
                snippet = default_fn()
                block = f"\n{comment}{parent_key}:\n"
                indent_snippet = yaml.safe_dump(
                    {child_key: yaml.safe_load(snippet)},
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                )
                for line in indent_snippet.strip().splitlines():
                    block += f"  {line}\n"
                appended += block
                added.append(f"{parent_key}.{child_key}")
            elif isinstance(parent, dict) and child_key not in parent:
                snippet = default_fn()
                child_yaml = yaml.safe_dump(
                    {child_key: yaml.safe_load(snippet)},
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                )
                indented = "".join(
                    f"  {line}\n" for line in child_yaml.strip().splitlines()
                )
                comment_indented = "".join(
                    f"  {line}\n" for line in comment.strip().splitlines()
                )
                insert = _find_parent_block_end(appended, parent_key)
                if insert is not None:
                    appended = (
                        appended[:insert]
                        + comment_indented
                        + indented
                        + appended[insert:]
                    )
                else:
                    appended += f"\n{comment_indented}{indented}"
                added.append(f"{parent_key}.{child_key}")

    if added:
        p.write_text(appended, encoding="utf-8")

    return added


def _find_parent_block_end(text: str, parent_key: str) -> int | None:
    """找到 YAML 文本中 ``parent_key:`` 块的结束位置（插入点）。

    返回该块最后一个缩进行之后的字符偏移量；找不到返回 None。
    """
    pattern = re.compile(rf"^{re.escape(parent_key)}:", re.MULTILINE)
    m = pattern.search(text)
    if m is None:
        return None

    lines = text.split("\n")
    line_start = text[:m.start()].count("\n")

    end_offset = m.end()
    if text[m.end():m.end() + 1] == "\n":
        end_offset = m.end() + 1

    for i in range(line_start + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            end_offset += len(line) + 1
            continue
        if line[0] in (" ", "\t"):
            end_offset += len(line) + 1
        else:
            break

    return min(end_offset, len(text))
