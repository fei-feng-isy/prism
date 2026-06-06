# Prism MCP Server 接入指南

Prism 提供标准 [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server，通过 stdio transport 暴露 3 个工具，供任何支持 MCP 的 LLM 客户端（Claude Desktop、Cursor、Continue 等）调用。

---

## 安装

```bash
pip install -e ".[semantic,mcp]"
```

`mcp` extra 安装 `mcp >= 1.0`（MCP SDK）；`semantic` 安装 sentence-transformers（可选，缺失走降级路径）。

---

## 启动

```bash
python -m prism.mcp
```

启动后阻塞读 stdio，按 Ctrl-C 或 stdin EOF 退出。所有日志写 stderr（stdout 是 MCP 协议帧）。

---

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PRISM_CONFIG` | YAML 配置路径 | 自动探测 `$PRISM_DATA_HOME/config.yaml` |
| `PRISM_PROFILE` | DB 路径的 `{profile}` 段 | `default` |
| `PRISM_USER_ID` | 用户隔离 ID（sha256 哈希后嵌入 DB 路径） | `local_default` |
| `PRISM_DATA_HOME` | 数据根目录 | `~/.prism` |
| `PRISM_DB_PATH` | 直传 DB 路径（绕过 path_template，测试用） | 无 |
| `PRISM_LOG_LEVEL` | 日志级别 | `INFO` |

---

## Claude Desktop 配置示例

在 `claude_desktop_config.json` 中添加：

```json
{
  "mcpServers": {
    "prism": {
      "command": "python",
      "args": ["-m", "prism.mcp"],
      "env": {
        "PRISM_DATA_HOME": "/home/user/.prism",
        "PRISM_USER_ID": "my_user"
      }
    }
  }
}
```

---

## 提供的工具

### prism_remember — 写入/修改记忆

写入或修改 Prism 记忆库中的 fact。

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `action` | string | 是 | `add` / `update` / `remove` / `helpful` / `unhelpful` |
| `content` | string | add 时必填 | fact 内容，应简洁自含 |
| `category` | string | 否 | `user_pref` / `user_env` / `project` / `tool` / `general`（默认 `general`） |

**示例调用：**

```json
{
  "action": "add",
  "content": "Alice 喜欢 PostgreSQL 14",
  "category": "user_pref"
}
```

**返回：**

```json
{
  "fact_id": 42,
  "is_new": true,
  "category": "user_pref",
  "entities": ["Alice", "PostgreSQL"]
}
```

---

### prism_recall — 查询记忆

查询 Prism 记忆库（语义 + FTS + 实体混合检索）。

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `action` | string | 是 | `search` / `probe` / `reason` / `related` / `contradict` |
| `query` | string | search 时必填 | 自由文本查询 |
| `entity` | string | probe/related 时必填 | 单个实体名 |
| `entities` | string[] | reason 时必填 | 多个实体名（AND 联合） |
| `category` | string | 否 | category 过滤 |
| `limit` | integer | 否 | 最大返回条数 |
| `min_trust` | number | 否 | 最低 trust 分数过滤（search + as_markdown=false 时有效） |
| `as_markdown` | boolean | 否 | search 专用：true（默认）返 markdown，false 返结构化 list |

**Action 说明：**

- **search** — 自由文本语义搜索，返回 top-k 最相关 fact
- **probe** — 查找所有提及指定实体的 fact
- **reason** — 查找同时提及所有指定实体的 fact（AND 联合）
- **related** — 查找与指定实体在同一 fact 中共现的其他实体
- **contradict** — 查找相互矛盾的 fact

**示例调用：**

```json
{"action": "search", "query": "用户的沟通风格", "limit": 5}
```

```json
{"action": "probe", "entity": "张三"}
```

```json
{"action": "reason", "entities": ["ffeng", "PostgreSQL"]}
```

```json
{"action": "related", "entity": "张三"}
```

---

### prism_admin — 管理/诊断

检查和管理 Prism 记忆库状态（运维向）。

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `action` | string | 是 | `stats` / `list` / `archive` / `restore` / `enrichment_diagnose` / `enrichment_fix` |
| `category` | string | 否 | stats 的 category 过滤 |
| `dry_run` | boolean | 否 | enrichment_fix 专用：true 只预览不修改 |

**Action 说明：**

- **stats** — 返回健康面板 JSON：DB 计数、vstore 容量、semantic backend 状态、prefetch 预热状态、retriever 权重
- **list** — 浏览已存储的 fact
- **archive** — 归档 fact
- **restore** — 恢复已归档的 fact
- **enrichment_diagnose** — enrichment 深度诊断：队列项、状态分布、缺失向量
- **enrichment_fix** — 修复缺失 embedding + 清 pending 队列 + 重建 vstore

**示例调用：**

```json
{"action": "stats"}
```

```json
{"action": "enrichment_fix", "dry_run": true}
```

---

## 架构

```
LLM Client (Claude Desktop / Cursor / ...)
    │
    │  stdio (MCP protocol)
    ▼
python -m prism.mcp
    │
    ├── create_server(runtime)        # 注册 3 个 tool
    ├── call_prism_tool(runtime, name, args)  # 统一 dispatch
    │       ├── prism_remember → PrismRemember
    │       ├── prism_recall   → PrismRecall
    │       └── prism_admin    → PrismAdmin
    │
    └── PrismRuntime
            ├── SQLite DB (facts, entities, ...)
            ├── VectorStore (local_numpy / hnswlib / ...)
            ├── SemanticBackend (BGE / cloud / ...)
            ├── HRR IncrementalBank
            └── RetrievalPipeline
```

---

## 降级行为

| 条件 | 行为 |
|------|------|
| 未安装 sentence-transformers | 语义检索路径权重降为 0，自动切到 FTS + Jaccard 模式（权重 0.65 / 0.35） |
| 未安装外置 vstore backend | 默认使用 local_numpy（暴力精确） |
| config.yaml 不存在 | 使用 default_config() |
