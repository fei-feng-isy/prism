# Prism

一个给 LLM agent 用的长期记忆系统，中文优先。核心思路是用 HRR（Holographic Reduced Representations）做结构化推理，配合语义嵌入和全文检索做混合召回。支持 5 种 vector store backend，可以作为 Hermes 插件或者 MCP server 跑。

---

## 做了什么

**混合检索** — 语义向量 + FTS5 trigram 全文检索 + 实体 Jaccard 相似度，三条路径融合打分。默认权重 0.55 / 0.30 / 0.15。没装 sentence-transformers 就自动切到纯全文+实体模式，权重 0 / 0.65 / 0.35，不报错也不卡住。

**HRR 增量编码** — 每个 category 维护一个复数 bundle，写入时增量叠加（不是每次全量重算），删除时直接从累加器扣掉对应向量，防止已删数据影响召回。50ms debounce 校准，避免高频写入时重复计算。

**生命周期管理** — 写入支持 add / replace / remove 三种操作，trust 分值按天衰减，超过阈值的 90 天后物理删除。矛盾检测是增量版，P@3 实测 1.0（阈值 0.6）。

**异步实体抽取** — 先走 jieba + regex 快速抽，少于 2 个实体时扔进 SQLite 队列，后台用 LLM 补抽。队列是 crash-safe 的，重启不丢任务。

**多用户隔离** — DB 文件按 `{hermes_home}/{profile}/{sha256(user_id)[:16]}.db` 生成路径，不同用户互不干扰。

**5 种 vector store** — local_numpy（默认，暴力精确）、hnswlib（推荐生产）、faiss、pgvector、qdrant。统一接口，`auto` 模式根据数据量自动升级 backend，切换不丢召回。

**3 种语义 backend** — 本地 BGE 模型、云端 OpenAI/DeepSeek embedding、混合 rerank（本地编码 + LLM 对 top-50 重排）。

**运维命令** — `migrate` 从旧 Holographic 迁移、`reindex` 换嵌入模型、`vstore-migrate` 跨 backend 迁数据，都是幂等的，中断可以重跑。

**离线能用** — 没有 sentence-transformers、没有外置 vector store 都不影响基本功能，自动走降级路径。

**性能** — 100 万条数据下 hnswlib topk 查询 P95 不到 1ms，10 万条跨 backend 迁移大约 18 秒。

---

## 安装

源码安装：

```bash
# GitHub
git clone https://github.com/fei-feng-isy/prism.git
# Gitee（国内）
git clone https://gitee.com/ffeng86/prism.git

cd prism

# 基础安装（HRR + 全文检索 + 实体抽取，不含语义）
pip install -e .

# 含本地语义（推荐）
pip install -e ".[semantic]"

# MCP server
pip install -e ".[semantic,mcp]"

# 按需选 vector store backend
pip install -e ".[semantic,hnswlib]"    # 推荐，十万级以上
pip install -e ".[semantic,faiss]"
pip install -e ".[semantic,pgvector]"
pip install -e ".[semantic,qdrant]"

# 全部装上（开发用）
pip install -e ".[semantic,mcp,hnswlib,faiss,qdrant,dev]"
```

Debian/Ubuntu 用户如果遇到 PEP 668 拦截，先建虚拟环境：

```bash
python -m venv .venv && source .venv/bin/activate
```

---

## 中文嵌入模型

装 `[semantic]` 后首次调用会自动从 HuggingFace 下载 `BAAI/bge-small-zh-v1.5`（约 95MB），缓存到 `~/.cache/huggingface/hub/`，之后不再联网。

国内访问 HuggingFace 不稳定的话，设镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或者在 `$HERMES_HOME/prism/config.yaml` 里配置：

```yaml
semantic:
  hf_endpoint_strategy: mirror_first   # 先走镜像，失败回退官方
```

三种策略：`respect_env`（默认，不动环境变量）、`mirror_first`（镜像优先）、`mirror_only`（只走镜像）。只在首次下载时生效，缓存命中直接走本地。

---

## 三种用法

### Hermes 插件

```bash
hermes plugins install https://github.com/fei-feng-isy/prism.git
# 国内：hermes plugins install https://gitee.com/ffeng86/prism.git

hermes memory setup prism
hermes
```

#### 如果是第一次安装，则有些依赖需要安装，按如下执行：

```
# 与hermes 对话，输入：
加载.hermes/prism/目录下的skill
# hermes加载完成后，输入：
首次安装prism
```

激活后 LLM 自动获得 `prism_remember` / `prism_recall` / `prism_admin` 三个工具，Hermes 内置的 memory write 也会同步到 Prism。

详见 [HERMES.md](HERMES.md)。

### MCP server

```bash
python -m prism.mcp
```

Claude Desktop 配置：

```json
{
  "mcpServers": {
    "prism": {
      "command": "python",
      "args": ["-m", "prism.mcp"],
      "env": { "PRISM_PROFILE": "default" }
    }
  }
}
```

详见 [MCP.md](MCP.md)。

### Python 库

```python
from prism.mcp.wire import RuntimeOptions, build_runtime

runtime = build_runtime(RuntimeOptions(user_id="alice", profile="work"))
try:
    runtime.remember(action="add", content="项目使用 PostgreSQL 14", category="tech")
    md = runtime.recall(action="search", query="数据库")
    stats = runtime.admin(action="stats")
finally:
    runtime.shutdown()
```

---

## 运维

```bash
# 自检（依赖 / 配置 / DB / 语义后端 / 端到端 smoke）
prism doctor
prism doctor --json           # 机器可读输出
```

其他运维命令：`migrate`（从 Holographic 迁移）、`reindex`（换嵌入模型）、`vstore-migrate`（跨 backend 迁数据）、`memory`（记忆库人工维护，13 个子命令）。

详见 [CLI.md](CLI.md)。

---

## Vector store 选型

| Backend | <10万 | >100万 | 持久化 | 额外依赖 | 备注 |
|---------|-------|--------|--------|----------|------|
| `local_numpy` | 合适 | 慢（~1s/query） | `.npz` | 无 | 默认，暴力精确，适合开发 |
| `hnswlib` | 合适 | 合适（<1ms P95） | `.hnsw` | `pip install hnswlib` | 推荐生产用 |
| `faiss` (FlatIP) | ~22ms | 慢 | `.faiss` | `pip install faiss-cpu` | 精确 IP baseline |
| `pgvector` | 合适 | 合适 | 服务端 | `pip install psycopg pgvector` | 多用户共享，需要 PG 14+ |
| `qdrant` | 合适 | 合适 | 服务端/内存 | `pip install qdrant-client` | 多租户，支持内嵌模式 |

---

## 配置

```yaml
prism:
  db:
    hermes_home: ~/.hermes
    profile: default
    user_id: local_default
  semantic:
    backend: local_bge
    model_name: BAAI/bge-small-zh-v1.5
    dim: 512
  vector_store:
    backend: local_numpy
    auto_thresholds:
      hnswlib: 2000
      faiss: 100000
  retriever:
    weight_semantic: 0.55
    weight_fts: 0.30
    weight_jaccard: 0.15
  lifecycle:
    decay_by_category:
      user_pref: { decay_per_day: 1.0,   floor: 0.4 }
      project:   { decay_per_day: 0.999, floor: 0.2 }
      general:   { decay_per_day: 0.995, floor: 0.0 }
    contradiction_threshold: 0.6
    ttl_days: 365
```

全部字段默认值及环境变量覆盖方式详见 [CONFIGURATION.md](CONFIGURATION.md)。

---

## 文档

- [CLI 命令参考](CLI.md)
- [MCP Server](MCP.md)
- [Hermes 插件](HERMES.md)
- [配置参考](CONFIGURATION.md)

---

## 许可

MIT，见 [LICENSE](LICENSE)。
