"""Markdown 文件导入业务逻辑（Service 层）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final

from .types import MirrorMdStats

if TYPE_CHECKING:
    from ..mirror import PrismMirror

log = logging.getLogger(__name__)

__all__ = ["ImportService"]

_MIRROR_TARGET: Final[str] = "memory"


class ImportService:
    """Markdown 文件级镜像导入。

    Args:
        mirror: 已构造好的 :class:`PrismMirror`
    """

    def __init__(self, mirror: PrismMirror) -> None:
        self._mirror = mirror

    def mirror_memory_md(
        self,
        md_path: Path,
        *,
        prune: bool = False,
    ) -> MirrorMdStats:
        """把 ``MEMORY.md`` 文件级镜像到 Prism。

        Args:
            md_path: MEMORY.md 路径
            prune: True 时跑完 add 后归档孤儿 fact

        Returns:
            :class:`MirrorMdStats`

        Raises:
            FileNotFoundError: md_path 不存在
        """
        from ..mirror import MIRROR_SOURCE_BUILTIN

        records = self._parse_md_records(md_path)
        total = len(records)
        added = 0
        skipped = 0
        failed = 0

        for content in records:
            try:
                result = self._mirror.mirror_add(
                    content,
                    source=MIRROR_SOURCE_BUILTIN,
                    target=_MIRROR_TARGET,
                )
            except Exception as e:
                failed += 1
                log.warning("mirror_add 失败：content=%r 异常=%s", content[:40], e)
                continue

            if result is None:
                failed += 1
                log.warning("mirror_add 返 None：content=%r", content[:40])
                continue

            if result.is_new:
                added += 1
            else:
                skipped += 1

        archived = 0
        if prune:
            record_set = set(records)
            archived = self._mirror.archive_ghost_facts(
                is_alive=lambda c: c in record_set,
                mirror_source=MIRROR_SOURCE_BUILTIN,
            )

        log.info(
            "memory mirror 完成：total=%d added=%d skipped=%d failed=%d archived=%d",
            total, added, skipped, failed, archived,
        )
        return MirrorMdStats(
            total=total,
            added=added,
            skipped_duplicate=skipped,
            failed=failed,
            archived=archived,
        )

    @staticmethod
    def _parse_md_records(md_path: Path) -> list[str]:
        if not md_path.exists():
            raise FileNotFoundError(str(md_path))
        text = md_path.read_text(encoding="utf-8")
        parts = text.split("\n§\n")
        return [p.strip() for p in parts if p.strip()]
