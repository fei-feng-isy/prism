"""Prism Service 层 — 唯一业务逻辑层。

API 层和 CLI 层退化为薄壳（参数解析 + 格式化输出），
所有业务逻辑统一通过此包访问。
"""

from .admin_service import AdminService
from .fact_service import FactService
from .import_service import ImportService
from .repair_service import RepairService
from .search_service import SearchService
from .stats_service import StatsService

__all__ = [
    "AdminService",
    "FactService",
    "ImportService",
    "RepairService",
    "SearchService",
    "StatsService",
]
