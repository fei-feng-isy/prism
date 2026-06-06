---
name: prism-deps
description: "Prism 依赖安装与恢复——将 numpy、jieba、PyYAML、sentence-transformers 装到共享 venv。触发词：prism 报 ImportError、prism 依赖缺失、prism 首次搭建依赖、prism deps。"
---

# Prism Deps — 依赖安装

Prism 记忆系统运行时需要 4 个 Python 包。为避免 `hermes upgrade` 清除手动安装的包，将它们装入独立的共享 venv，通过 `.pth` 桥接文件注入 Hermes Agent 环境。

## 依赖清单

| 包 | 用途 | 备注 |
|----|------|------|
| `numpy` | 向量存储（NPZ 文件读写）、HRR 编码 | 核心依赖 |
| `jieba` | 中文分词（FTS 索引） | 中文环境必需 |
| `PyYAML` | Prism config.yaml 解析 | 核心依赖 |
| `sentence-transformers` | 语义嵌入生成（BAAI/bge-small-zh-v1.5） | 含 torch/transformers，~5GB |

## 架构

```
~/.hermes/
├── shared-venv/                  ← 共享 venv（跨升级保留）
│   └── lib/python3.XX/site-packages/
│       └── <四个包装在这里>
└── hermes-agent/
    └── venv/
        └── lib/python3.XX/site-packages/
            └── hermes-shared.pth  → 指向 shared-venv site-packages
```

`.pth` 是 Python 的 site-packages 路径注入机制——Python 启动时自动加载目录下所有 `.pth` 文件，将其中的路径追加到 `sys.path`。pip 不管理 `.pth`，所以 `hermes upgrade` 重建 venv 时不会删除它。

## 首次搭建

适用于 shared-venv 不存在或 `.pth` 桥接丢失的场景。

```bash
# 1. 创建共享 venv（一次性）。用 Hermes Agent venv 的 python3 保证版本一致
~/.hermes/hermes-agent/venv/bin/python3 -m venv ~/.hermes/shared-venv

# 2. 升级 pip（可选，避免旧版 pip 安装 sentence-transformers 失败）
~/.hermes/shared-venv/bin/pip install --upgrade pip

# 3. 安装 Prism 依赖到共享 venv
~/.hermes/shared-venv/bin/pip install numpy jieba PyYAML sentence-transformers

# 4. 创建 .pth 桥接文件（动态路径，不硬编码 Python 版本）
SITE_PKGS=$(~/.hermes/shared-venv/bin/python3 -c "import site; print(site.getsitepackages()[0])")
HERMES_SITE=$(~/.hermes/hermes-agent/venv/bin/python3 -c "import site; print(site.getsitepackages()[0])")
echo "$SITE_PKGS" > "$HERMES_SITE/hermes-shared.pth"
```

## 增量安装

shared-venv 已存在，只缺部分包时：

```bash
~/.hermes/shared-venv/bin/pip install <missing-package>
```

无需修改 `.pth`——桥接指向整个 site-packages 目录，新增包自动可见。

## 验证

```bash
# 用 Hermes Agent venv 的 python3 验证所有依赖可导入
~/.hermes/hermes-agent/venv/bin/python3 -c "
import numpy; print('numpy', numpy.__version__)
import jieba; print('jieba', jieba.__version__)
import yaml; print('yaml', yaml.__version__)
from sentence_transformers import SentenceTransformer
print('sentence-transformers OK')
m = SentenceTransformer('BAAI/bge-small-zh-v1.5')
print('embed dim:', m.encode(['测试']).shape)
"
```

期望输出：四个包导入成功，最后一行 `embed dim: (1, 512)`。

## 升级后恢复

`hermes upgrade` 重建 Hermes Agent venv 时 `.pth` 文件不受影响——无需任何操作。

若 `.pth` 意外丢失：

```bash
SITE_PKGS=$(~/.hermes/shared-venv/bin/python3 -c "import site; print(site.getsitepackages()[0])")
HERMES_SITE=$(~/.hermes/hermes-agent/venv/bin/python3 -c "import site; print(site.getsitepackages()[0])")
echo "$SITE_PKGS" > "$HERMES_SITE/hermes-shared.pth"
```

## 陷阱

- **始终用 `~/.hermes/hermes-agent/venv/bin/python3` 而非系统 `python3`**。系统 Python 版本可能与 Hermes Agent venv 不一致，导致共享 venv 创建失败或 .pth 桥接无效。
- **不硬编码路径**。`/home/<user>/` 和 `python3.XX` 均通过 `python3 -c "import site; ..."` 动态获取。
- **sentence-transformers 安装体积大（~5GB）**。网络慢时设镜像：`export HF_ENDPOINT=https://hf-mirror.com`。首次运行 `SentenceTransformer()` 还会下载模型文件（~100MB），同样走镜像。
- **`.pth` 文件路径独占一行**。`echo` 自动加换行，Python 可接受。手动编辑时确保路径后无多余字符。
