# Prism — Hermes 插件集成指南

Prism 作为 Hermes 的 MemoryProvider 插件运行，注册 3 个 LLM tool + 1 个 slash 命令 + 1 个 memory 写入钩子。

---

## 安装与部署

### 方式一：使用Hermes命令安装（推荐）

```bash
# GitHub
hermes plugins install https://github.com/fei-feng-isy/prism.git
# Gitee（国内）
hermes plugins install https://gitee.com/ffeng86/prism.git
```

> 安装完成后 hermes 会询问是否切换 `memory.provider`，按 `y` 或直接回车即可。

### 方式二：软链方式安装

```bash
# 1. 软链到 Hermes 插件目录
ln -sfn /path/to/prism ~/.hermes/plugins/prism

# 2. 在 Hermes 的 venv 中安装依赖
source ~/.hermes/hermes-agent/venv/bin/activate
pip install -e ".[semantic]"
```
> 必须链到仓库根：hermes 需要读到 `plugin.yaml`（在仓库根），否则 `kind: standalone` 声明不生效，`/prism` slash 命令注册不上。

### 提示

第一次安装后，第一次使用需要从 HuggingFace 下载语义小模型（`BAAI/bge-small-zh-v1.5`，约 95MB）。国内用户建议先设置镜像加速：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 验证

作为插件安装后，在 Hermes 对话框输入 `/prism doctor` 查看启用情况。如提示缺失依赖等问题，可手动解决或直接让 hermes 代为处理。

---

## 生命周期

Hermes 按以下顺序调用 PrismMemoryProvider：

```
__init__           # 零 IO，只存 config_path
  ↓
is_available()     # 检查必备依赖（jieba + numpy + sqlite3），始终返回 True
  ↓
initialize()       # 构造完整 pipeline：
                   #   1. 解析 data_home + config.yaml（自动生成默认模板）
                   #   2. build_runtime（DB / bank / semantic / vstore / mirror / pipeline）
                   #   3. 后台线程预热 jieba + BGE warmup
  ↓
system_prompt_block()  # 返回系统 prompt 块，注入 fact 数量 + 可用操作提示
  ↓
prefetch(query)    # 每轮会话前自动预取相关 fact，返回 markdown 注入上下文
  ↓
handle_tool_call() # LLM 调用 tool 时 dispatch（见下方 Tool Schemas）
  ↓
on_memory_write()  # 内置 memory 写入钩子（见下方 Memory 镜像）
  ↓
shutdown()         # persist vstore → close DB → join warmup 线程
```

---

## Tool Schemas（OpenAI Function Calling 格式）

Hermes 把以下 3 个 schema 注册为 LLM 可调用的 tool。

### prism_remember

写入或修改 fact。

```json
{
  "name": "prism_remember",
  "parameters": {
    "type": "object",
    "properties": {
      "action": {
        "type": "string",
        "enum": ["add", "update", "remove", "helpful", "unhelpful"]
      },
      "content": {
        "type": "string",
        "description": "Fact text (required for 'add')"
      },
      "category": {
        "type": "string",
        "enum": ["user_pref", "user_env", "project", "tool", "general"]
      }
    },
    "required": ["action"]
  }
}
```

**使用时机：** 用户分享持久偏好、人物信息、项目决策、环境配置等需要长期记忆的内容。推荐优先使用 `prism_remember(add)` 而非内置 memory tool，因为 Prism 支持实体 probe 和多实体推理。

### prism_recall

查询记忆。

```json
{
  "name": "prism_recall",
  "parameters": {
    "type": "object",
    "properties": {
      "action": {
        "type": "string",
        "enum": ["search", "probe", "reason", "related", "contradict"]
      },
      "query": { "type": "string" },
      "entity": { "type": "string" },
      "entities": { "type": "array", "items": { "type": "string" } },
      "category": { "type": "string" },
      "limit": { "type": "integer" },
      "min_trust": { "type": "number" },
      "as_markdown": { "type": "boolean" }
    },
    "required": ["action"]
  }
}
```

**Action 选择：**

| Action | 参数 | 说明 | 示例 |
|--------|------|------|------|
| `search` | `query` | 自由文本语义搜索 | `query="用户的沟通风格"` |
| `probe` | `entity` | 单实体的所有相关 fact | `entity="张三"` |
| `reason` | `entities` | 多实体 AND 联合查找 | `entities=["ffeng", "PostgreSQL"]` |
| `related` | `entity` | 与该实体共现的其他实体 | `entity="张三"` |
| `contradict` | — | 查找相互矛盾的 fact | — |

**使用时机：** 回答关于用户/项目/历史决策的问题前，**必须**先调用 prism_recall。预取上下文会自动注入，但显式查询可获得更精确的结果。

### prism_admin

管理和诊断。

```json
{
  "name": "prism_admin",
  "parameters": {
    "type": "object",
    "properties": {
      "action": {
        "type": "string",
        "enum": ["stats", "list", "archive", "restore", "enrichment_diagnose", "enrichment_fix"]
      },
      "category": { "type": "string" },
      "dry_run": { "type": "boolean" }
    },
    "required": ["action"]
  }
}
```

**使用时机：** 用户问"记忆库里有多少条"、"你还记得什么"、或需要排查检索问题时。

---

## /prism Slash 命令

在 Hermes 对话中输入 `/prism <子命令>` 可直接调用 Prism CLI，等价于 `python -m prism <子命令>`。

```
/prism doctor
/prism memory list
/prism memory stats --json
/prism memory search "PostgreSQL"
/prism reindex --model BAAI/bge-base-zh-v1.5 --dry-run
/prism export --output facts.jsonl
```

可用子命令：`doctor`、`migrate`、`reindex`、`vstore-migrate`、`memory`（含所有子命令）、`eval`、`export`。

输出被 buffered 后一次性返回到对话窗口。长时间运行的命令（如大库 reindex）会阻塞 REPL。

---

## Memory 镜像（on_memory_write）

当 Hermes 的内置 memory 工具写入时（如 `memory_tool(add, ...)`），Prism 会通过 `on_memory_write` 钩子自动镜像：

- **add** → `mirror.on_memory_write("add", target, content)` → 写入 Prism DB
- **replace** → 旧 fact 归档 + 新 fact 创建（supersedes 链）
- **remove** → 对应 fact 软删除

异常被吞掉并 WARN 日志，不影响 Hermes 主流程。

这确保了用户通过内置 memory 工具存储的信息也会同步到 Prism，可通过 prism_recall 检索。

---

## 数据目录

Prism 在 Hermes 下的数据放在 `$HERMES_HOME/prism/`：

```
$HERMES_HOME/prism/
├── config.yaml          # Prism 配置（首次自动生成）
├── default/             # profile 目录
│   └── <user_hash>.db   # SQLite DB（per user）
│   └── <user_hash>.vstore.npz  # 向量索引持久化
└── ...
```

`user_hash` = `sha256(user_id)[:16]`，明文 user_id 不出现在路径中。

---

## 配置

`config.yaml` 首次启动自动生成。关键配置段：

- **semantic** — 嵌入模型、backend（local_bge / cloud / hybrid_rerank）、HF 镜像策略
- **retriever** — 三路融合权重（semantic / fts / jaccard）
- **vector_store** — backend 选择（auto / local_numpy / hnswlib / faiss / pgvector / qdrant）
- **hrr** — HRR 维度和 bundle 配置
- **entities** — 实体抽取配置
- **lifecycle** — trust 衰减、TTL、矛盾检测阈值
- **db** — DB 路径模板

详细配置说明见 `CONFIGURATION.md`。

---

## 降级行为

| 缺失依赖 | 行为 |
|----------|------|
| sentence-transformers | 语义路径权重→0，自动切 FTS+Jaccard（0.65/0.35） |
| 外置 vstore backend | 用 local_numpy（暴力精确） |
| LLM endpoint (rerank) | rerank 自动降级到本地排序，其他功能不受影响 |

所有降级在 `/prism doctor` 中有检测和提示。

---

## plugin.yaml

```yaml
name: prism
kind: standalone
hooks:
  - on_memory_write
```

`kind: standalone` 是必须的——绕过 hermes PluginManager 的 auto-coerce-to-exclusive，让 `/prism` slash 命令正确注册。
