"""Prism 插件命名空间包。

Hermes 端通过 ``importlib.util.spec_from_file_location`` 直接加载
``plugins/hermes/__init__.py`` 不需要本包；本 ``__init__.py`` 仅用于
让 prism 仓库本身的单元测试能 ``from plugins.hermes import ...``。
"""
