# Prism 配置参考

Prism 的所有行为都可以通过一个 YAML 配置文件来调整。

**配置文件位置**取决于启动方式：
- 通过 Hermes 插件启动：`~/.hermes/prism/config.yaml`
- 通过 MCP 服务启动：`~/.prism/config.yaml`（可通过 `PRISM_DATA_HOME` 环境变量修改）
- 通过命令行启动：`~/.prism/config.yaml`（可通过 `--config` 参数指定）

**你不需要手动创建这个文件** —— 第一次启动时会自动生成一份包含全部默认值的模板。如果后续版本新增了配置项，启动时也会自动把缺失的部分追加到文件末尾，并附带说明注释，不会影响你已有的设置。

### 配置的优先级

同一个选项如果在多处设置了值，按以下顺序决定谁生效（右边的覆盖左边的）：

**程序内置默认值 → YAML 文件 → 环境变量**

也就是说：环境变量的优先级最高，可以临时覆盖 YAML 里的任何设置，方便在不改文件的情况下调试或部署。

---

## semantic — 语义理解

这一段控制 Prism 如何「理解」文本的含义。Prism 需要把文字转换成数学向量（一串数字），这样才能计算两段文字在语义上有多相似。这个过程叫做「嵌入」（embedding）。

有三种方式可选：

- **`local_bge`**（默认）：在你自己的电脑上运行一个小型 AI 模型来做嵌入，不需要联网，隐私性最好。首次使用时会自动下载模型（约 100MB）。
- **`cloud_embedding`**：调用 OpenAI 等云服务的 API 来做嵌入，需要联网和 API 密钥，速度快但有费用。
- **`hybrid_rerank`**：混合模式 —— 先用本地模型快速筛选候选结果，再调用 LLM（大语言模型）对候选结果重新排序，准确度最高但最慢。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `backend` | string | `local_bge` | 使用哪种嵌入方式，可选 `local_bge` / `cloud_embedding` / `hybrid_rerank` |
| `local_model` | string | `BAAI/bge-small-zh-v1.5` | 本地模型的名称。BGE-small-zh 是一个专为中文优化的小型嵌入模型 |
| `hf_endpoint_strategy` | string | `respect_env` | 模型下载策略（仅首次下载时生效，之后使用本地缓存）。`respect_env`：按系统默认方式下载；`mirror_first`：优先从国内镜像下载（推荐国内用户）；`mirror_only`：只从镜像下载 |
| `hf_mirror_url` | string | `https://hf-mirror.com` | 国内镜像地址。如果你在国内访问 HuggingFace 很慢，可以把 `hf_endpoint_strategy` 改成 `mirror_first` |

### semantic.cloud — 云端嵌入服务

当 `backend` 设为 `cloud_embedding` 时，用这里的配置连接云端 API。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `provider` | string | `openai` | API 提供商名称 |
| `model` | string | `text-embedding-3-small` | 要调用的嵌入模型名称 |
| `api_key_env` | string | `OPENAI_API_KEY` | 存放 API 密钥的环境变量名。Prism 不会在配置文件里直接存密钥，而是从这个环境变量中读取，避免密钥泄露 |
| `timeout_ms` | int | `3000` | API 请求的超时时间，单位毫秒（1秒=1000毫秒）。超过这个时间没有响应就放弃 |

### semantic.rerank — LLM 重排序

当 `backend` 设为 `hybrid_rerank` 时生效。工作流程：先用本地模型从所有记忆中粗选出 `candidate_n` 条候选，然后发给 LLM 做精细排序，最终只保留 `final_k` 条最相关的结果。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `llm_model` | string | `deepseek-v4-flash` | 用于重排序的大语言模型名称 |
| `llm_endpoint` | string | `https://api.deepseek.com/v1` | LLM API 的访问地址 |
| `api_key_env` | string | `DEEPSEEK_API_KEY` | 存放 LLM API 密钥的环境变量名 |
| `candidate_n` | int | `50` | 粗选阶段保留多少条候选记忆，发给 LLM 重排。数字越大越准但越慢 |
| `final_k` | int | `5` | 重排后最终返回多少条结果 |
| `apply_to` | string | `recall_tool` | 在哪些场景使用重排序：`recall_tool` 仅在显式调用搜索时重排；`prefetch` 仅在自动预取时重排；`both` 两者都用 |
| `timeout_ms` | int | `2000` | LLM API 请求超时时间（毫秒） |
| `fallback_on_error` | string | `local` | LLM 调用失败时的应对：`local` 回退到本地排序结果（推荐）；`error` 直接报错 |

---

## vector_store — 向量存储引擎

嵌入后的向量需要存储起来，检索时再从中找到最相似的。不同的存储引擎适合不同的数据规模：

- **`local_numpy`**：用 Python NumPy 库做暴力精确搜索。适合少量数据（几千条以内），无需安装额外依赖，精度最高。
- **`hnswlib`**：一种高性能的近似最近邻搜索算法。适合中等规模数据（几千到十万条），推荐生产环境使用。需要额外安装 `hnswlib` 包。
- **`faiss`**：Facebook 开发的向量搜索库，适合大规模数据（十万条以上）。需要额外安装 `faiss` 包。
- **`pgvector`**：PostgreSQL 数据库的向量搜索扩展，适合已有 PostgreSQL 基础设施的团队。
- **`qdrant`**：独立的向量数据库服务，适合分布式部署。
- **`auto`**（默认）：Prism 根据当前数据量自动选择合适的引擎，数据增长时自动升级，无需手动切换。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `backend` | string | `auto` | 使用哪种存储引擎，可选值见上方说明 |

### vector_store.auto_thresholds — 自动切换阈值

`backend` 为 `auto` 时，Prism 按数据条数自动升级引擎。这里配置升级的阈值。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hnswlib` | int | `2000` | 数据超过 2000 条时，从 local_numpy 升级到 hnswlib |
| `faiss` | int | `100000` | 数据超过 10 万条时，从 hnswlib 升级到 faiss |

### vector_store.pgvector — PostgreSQL 向量搜索

`backend` 为 `pgvector` 时需要配置。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dsn_env` | string | `PRISM_PGVECTOR_DSN` | 存放 PostgreSQL 连接字符串的环境变量名。连接字符串格式示例：`postgresql://user:pass@localhost/dbname` |
| `table_name` | string | `prism_vectors` | 在数据库中创建的向量表的名称 |

### vector_store.qdrant — Qdrant 向量数据库

`backend` 为 `qdrant` 时需要配置。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `url_env` | string | `PRISM_QDRANT_URL` | 存放 Qdrant 服务地址的环境变量名，如 `http://localhost:6333` |
| `api_key_env` | string | `PRISM_QDRANT_API_KEY` | 存放 Qdrant API 密钥的环境变量名（如果 Qdrant 服务不需要认证，可以不设） |
| `collection_name` | string | `prism` | Qdrant 中的集合（collection）名称 |

---

## retriever — 混合检索

Prism 搜索记忆时会同时走三条路径，然后把结果融合在一起排序。这里配置三条路径各自的权重。

三条路径分别是：
- **语义路径**（semantic）：计算查询和记忆的「语义相似度」，适合模糊搜索、同义词匹配
- **全文检索路径**（FTS，Full-Text Search）：基于关键词精确匹配，使用 SQLite 内置的 FTS5 引擎和 trigram（三字符切分）分词器，对中文友好
- **实体路径**（Jaccard）：提取查询和记忆中的实体（人名、工具名等），计算实体集合的 Jaccard 相似度（两个集合交集大小除以并集大小）

**三个权重的总和必须等于 1.0**。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `weight_semantic` | float | `0.55` | 语义路径的权重（0.55 = 55%），占比最大表示语义匹配最重要 |
| `weight_fts` | float | `0.30` | 全文检索路径的权重（0.30 = 30%），关键词精确匹配 |
| `weight_jaccard` | float | `0.15` | 实体路径的权重（0.15 = 15%），实体关联匹配 |

> 如果没有安装语义模型（sentence-transformers），系统会自动切换到降级模式：语义权重归零，全文检索提升到 0.65，实体路径提升到 0.35，功能不受影响只是准确度略有下降。

### retriever.prefetch — 自动预取

Prism 可以在 LLM 回答用户问题之前，自动把相关的记忆注入到 LLM 的上下文中（称为「预取」）。这里配置预取的行为。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `top_k` | int | `5` | 最多注入多少条记忆。太多会浪费 LLM 的上下文窗口 |
| `p95_target_ms` | int | `100` | 性能目标：95% 的预取请求应在此时间内完成（毫秒）。目前仅用于监控参考 |
| `min_trust` | float | `0.3` | 最低可信度阈值（0~1 之间）。trust_score 低于此值的记忆不会被注入，避免注入过时或低质量的信息 |

---

## hrr — 全息向量表示

HRR（Holographic Reduced Representation，全息缩减表示）是 Prism 的核心技术之一。它把每条记忆编码成一个固定长度的复数向量，然后把同一类目下所有记忆的向量叠加成一个「类目向量」。这样做的好处是：添加和删除记忆时只需要增量计算，不用每次都重新扫描所有数据。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dim` | int | `1024` | HRR 向量的维度（就是那串数字有多长）。必须是正偶数。越大表达能力越强但内存占用也越大。1024 是在精度和资源之间的平衡点 |

### hrr.bank — HRR 增量计算参数

bank 是管理 HRR 向量累加器的组件。当记忆被添加或删除时，bank 会增量更新类目向量，并定期做一次完整校准（calibration），确保累加误差不会积累过大。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `remove_debounce_ms` | int | `50` | 去抖窗口（毫秒）。短时间内连续删除多条记忆时，等最后一次删除完成后再统一校准，避免重复计算。50 毫秒意味着 50ms 内的连续删除会合并成一次校准 |
| `calibration_threshold_min` | int | `10` | 触发校准所需的最少记忆条数。如果一个类目下的记忆少于这个数，就暂不校准（数据太少校准意义不大） |
| `calibration_threshold_pct` | float | `0.15` | 脏数据占比阈值。当类目内被修改/删除但未校准的记忆占比超过 15% 时，触发一次完整校准。取值范围 (0, 1]，0.15 表示 15% |
| `snr_warn_factor` | float | `1.2` | SNR（Signal-to-Noise Ratio，信噪比）告警因子。当信噪比低于阈值时输出告警日志，提示可能需要降维或清理数据。1.2 是一个宽松的预警阈值 |

---

## entities — 实体抽取

Prism 会从每条记忆中自动提取「实体」（比如人名、工具名、项目名等）。提取分两个阶段：

1. **Stage 1（快速抽取）**：用 jieba 分词 + 正则表达式快速提取实体，通常能在 1 毫秒内完成
2. **Stage 2（LLM 补充抽取）**：如果 Stage 1 抽出的实体太少，就把这条记忆放入后台队列，由 LLM 做更精细的抽取

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `auto_enrich` | bool | `true` | 是否启用 Stage 2 的 LLM 异步补充抽取。设为 `false` 则只用 Stage 1 的快速抽取 |
| `stage1_min_entities` | int | `2` | Stage 1 抽取结果低于此数量时，才会触发 Stage 2。设为 `2` 表示：如果 jieba+regex 只抽出了 0 或 1 个实体，就交给 LLM 再抽一次 |

---

## lifecycle — 生命周期管理

控制记忆的「保质期」。每条记忆都有一个 trust_score（可信度分数，范围 0~1），表示这条记忆有多可靠/新鲜。随着时间推移，trust_score 会按配置的速率自动衰减。当分数低到一定程度且超过保留天数时，记忆会被自动归档（不删除，只是不再参与检索）。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `archive_after_days` | int | `90` | 超过多少天后，低 trust 的记忆会被自动归档。设为 90 表示最多保留 3 个月 |

### lifecycle.decay_by_category — 按类目配置衰减速率

不同类型的记忆适合不同的衰减策略。比如用户的偏好（「我喜欢简洁的回复」）通常长期有效，不应该衰减；而项目信息（「本周目标是完成 API 重构」）可能很快过时，应该较快衰减。

每个类目（category）有两个参数：

- **`decay_per_day`**：每天的衰减系数。trust_score 每天乘以这个数。`1.0` 表示不衰减，`0.995` 表示每天衰减 0.5%（100 天后约剩 60%）
- **`min_trust_floor`**：衰减下限。trust_score 不会低于这个值。设为 `0.4` 表示即使记忆很旧，trust 也不会低于 0.4，仍可被检索到

| 类目 | `decay_per_day` | `min_trust_floor` | 说明 |
|------|-----------------|-------------------|------|
| `user_pref` | `1.0`（不衰减） | `0.4` | 用户偏好，如「喜欢简洁回复」「习惯用 vim」。长期稳定，不衰减 |
| `user_env` | `1.0`（不衰减） | `0.4` | 用户环境信息，如「用的是 macOS」「Python 3.11」。相对稳定 |
| `project` | `0.999` | `0.2` | 项目相关信息，如「当前在做 v2 重构」。缓慢衰减，每天降 0.1% |
| `general` | `0.995` | `0.0` | 通用信息。衰减最快，每天降 0.5%，且可以衰减到 0（完全失效） |

你可以自定义类目或修改已有类目的衰减参数。例如让 `project` 类目衰减更快：

```yaml
lifecycle:
  decay_by_category:
    project:
      decay_per_day: 0.99   # 每天降 1%，比默认的 0.1% 快 10 倍
      min_trust_floor: 0.1
```

---

## db — 数据库

Prism 使用 SQLite 作为本地数据库，每个用户/场景各自独立一个 `.db` 文件。

数据根目录（`data_home`）由启动方式决定：
- **通过 Hermes 插件启动**：默认 `~/.hermes/prism`（跟随 Hermes 的 `$HERMES_HOME`）
- **通过 MCP 服务启动**：默认 `~/.prism`（可通过 `PRISM_DATA_HOME` 环境变量覆盖）
- **通过命令行启动**：默认 `~/.prism`（可通过 `--data-home` 参数覆盖）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path_template` | string | `{data_home}/{profile}/{user_hash}.db` | 数据库文件路径的模板。包含三个占位符（见下方说明），**三个占位符都必须保留** |
| `data_home_default` | string | `~/.prism` | 当调用方未显式指定 `data_home` 时使用的默认路径 |

**路径模板占位符说明**：

- `{data_home}`：数据根目录。不同启动方式有不同的默认值（见上方说明）
- `{profile}`：配置档名称，用于隔离不同的使用场景（如 "default"、"coder"）。由调用方传入
- `{user_hash}`：用户 ID 的 SHA-256 哈希值的前 16 位，用于多用户隔离。确保不同用户的数据互不干扰，同时不在文件名中暴露原始用户 ID

---

## logging — 日志

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `level` | string | `INFO` | 日志输出级别。级别从低到高：`DEBUG`（最详细）> `INFO`（一般信息）> `WARNING`（告警）> `ERROR`（错误）> `CRITICAL`（严重错误）。设为某个级别后，只输出该级别及更高级别的日志 |

### logging.call_tracking — API 调用追踪

Prism 可以自动记录每次 API 调用的详细信息，包括：哪个接口被调用了、耗时多少、是从哪里调用的（Hermes 插件 / MCP 协议 / 命令行）、是否成功。这些数据用于性能分析和问题排查。

追踪数据保存在两个地方：
1. **内存缓冲区**：最近的调用记录保存在内存中，用于实时统计（如 `prism admin stats` 显示的 P50/P95 延迟数据）
2. **日志文件**（可选）：持久化到磁盘上的 JSON Lines 格式文件，支持自动轮转（文件满了自动切换到新文件，旧文件保留指定数量）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `true` | 追踪总开关。设为 `false` 则完全关闭追踪功能，不记录任何调用数据，对性能零影响 |
| `file_logging` | bool | `true` | 文件日志开关。设为 `false` 时仅保留内存中的实时统计，不往磁盘写日志文件。适合不需要历史记录但想看实时性能数据的场景 |
| `max_bytes` | int | `5000000` | 单个日志文件的最大大小（字节）。默认 5,000,000 字节 = 约 5MB。文件达到此大小后自动轮转 |
| `backup_count` | int | `3` | 保留多少个历史日志文件。设为 3 表示最多保留 `api_calls.jsonl.1`、`.2`、`.3` 三个旧文件，加上当前文件共 4 个，总占用最多约 20MB |
| `buffer_size` | int | `10000` | 内存缓冲区最多保留多少条调用记录。超出后最旧的记录会被丢弃。这些记录用于计算实时的 P50/P95 延迟等统计数据 |

**日志文件位置**：`{数据库所在目录}/logs/api_calls.jsonl`

例如数据库路径是 `~/.prism/default/a1b2c3d4.db`，则日志文件位于 `~/.prism/default/logs/api_calls.jsonl`。

**日志内容示例**（每行一个 JSON 对象）：

```json
{"timestamp":"2026-06-05T14:30:01","service":"search","action":"search","source":"hermes","latency_ms":12.345,"success":true,"detail":{"hit_count":3}}
{"timestamp":"2026-06-05T14:30:02","service":"fact","action":"add","source":"cli","latency_ms":5.678,"success":true,"detail":null}
```

字段含义：
- `timestamp`：调用发生的时间
- `service`：被调用的服务（`fact` = 记忆增删改、`search` = 检索、`admin` = 运维）
- `action`：具体操作（如 `add`、`search`、`remove`、`stats` 等）
- `source`：调用来源（`hermes` = Hermes 插件、`mcp` = MCP 协议、`cli` = 命令行工具）
- `latency_ms`：本次调用耗时（毫秒）
- `success`：是否成功
- `detail`：附加信息（如搜索命中了多少条）

---

## 环境变量速查表

以下环境变量可以覆盖 YAML 文件中的对应配置。适用于容器化部署、CI/CD 等不方便修改配置文件的场景。

**MCP 服务专用环境变量**（不在 YAML 配置中，仅在 MCP 启动时使用）：

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `PRISM_DATA_HOME` | 数据根目录（配置文件、DB、日志都在此目录下） | `~/.prism` |
| `PRISM_CONFIG` | YAML 配置文件路径（优先级高于 data_home/config.yaml） | — |
| `PRISM_DB_PATH` | 直传 DB 路径，绕过 path_template（测试用） | — |
| `PRISM_PROFILE` | 对应 path_template 中的 `{profile}` | `default` |
| `PRISM_USER_ID` | 用户隔离 ID（SHA-256 → user_hash） | `local_default` |

**配置项覆盖环境变量**：

| 环境变量 | 覆盖的配置项 | 类型 | 示例 |
|----------|-------------|------|------|
| `PRISM_SEMANTIC_BACKEND` | `semantic.backend` | string | `cloud_embedding` |
| `PRISM_SEMANTIC_LOCAL_MODEL` | `semantic.local_model` | string | `BAAI/bge-base-zh-v1.5` |
| `PRISM_VECTOR_STORE_BACKEND` | `vector_store.backend` | string | `hnswlib` |
| `PRISM_HRR_DIM` | `hrr.dim` | int | `2048` |
| `PRISM_ENTITIES_AUTO_ENRICH` | `entities.auto_enrich` | bool | `false` |
| `PRISM_RETRIEVER_WEIGHT_SEMANTIC` | `retriever.weight_semantic` | float | `0.6` |
| `PRISM_RETRIEVER_WEIGHT_FTS` | `retriever.weight_fts` | float | `0.25` |
| `PRISM_RETRIEVER_WEIGHT_JACCARD` | `retriever.weight_jaccard` | float | `0.15` |
| `PRISM_LOG_LEVEL` | `logging.level` | string | `DEBUG` |
| `PRISM_DB_PATH_TEMPLATE` | `db.path_template` | string | — |
| `PRISM_CALL_TRACKING_ENABLED` | `logging.call_tracking.enabled` | bool | `0` |
| `PRISM_CALL_TRACKING_FILE` | `logging.call_tracking.file_logging` | bool | `off` |

**bool 类型的值**：`1` / `true` / `yes` / `on` 表示开启，`0` / `false` / `no` / `off` 表示关闭，不区分大小写。

---

## 完整默认配置

以下是所有配置项及其默认值，可以直接复制到 `config.yaml` 中按需修改：

```yaml
semantic:
  backend: local_bge
  local_model: BAAI/bge-small-zh-v1.5
  hf_endpoint_strategy: respect_env
  hf_mirror_url: https://hf-mirror.com
  cloud:
    provider: openai
    model: text-embedding-3-small
    api_key_env: OPENAI_API_KEY
    timeout_ms: 3000
  rerank:
    llm_model: deepseek-v4-flash
    llm_endpoint: https://api.deepseek.com/v1
    api_key_env: DEEPSEEK_API_KEY
    candidate_n: 50
    final_k: 5
    apply_to: recall_tool
    timeout_ms: 2000
    fallback_on_error: local

vector_store:
  backend: auto
  auto_thresholds:
    hnswlib: 2000
    faiss: 100000
  pgvector:
    dsn_env: PRISM_PGVECTOR_DSN
    table_name: prism_vectors
  qdrant:
    url_env: PRISM_QDRANT_URL
    api_key_env: PRISM_QDRANT_API_KEY
    collection_name: prism

retriever:
  weight_semantic: 0.55
  weight_fts: 0.30
  weight_jaccard: 0.15
  prefetch:
    top_k: 5
    p95_target_ms: 100
    min_trust: 0.3

hrr:
  dim: 1024
  bank:
    remove_debounce_ms: 50
    calibration_threshold_min: 10
    calibration_threshold_pct: 0.15
    snr_warn_factor: 1.2

entities:
  auto_enrich: true
  stage1_min_entities: 2

lifecycle:
  archive_after_days: 90
  decay_by_category:
    user_pref:
      decay_per_day: 1.0
      min_trust_floor: 0.4
    user_env:
      decay_per_day: 1.0
      min_trust_floor: 0.4
    project:
      decay_per_day: 0.999
      min_trust_floor: 0.2
    general:
      decay_per_day: 0.995
      min_trust_floor: 0.0

db:
  path_template: '{data_home}/{profile}/{user_hash}.db'
  data_home_default: ~/.prism

logging:
  level: INFO
  call_tracking:
    enabled: true
    file_logging: true
    max_bytes: 5000000
    backup_count: 3
    buffer_size: 10000
```

---

## 常见问题

**Q：我只想关掉日志文件，不想完全关追踪**
A：设置 `logging.call_tracking.file_logging: false`，或环境变量 `PRISM_CALL_TRACKING_FILE=0`。这样 `prism admin stats` 仍然能看到实时性能数据，只是不写磁盘。

**Q：升级后配置文件多了一段不认识的内容**
A：这是自动补全功能。新版本新增的配置段会自动追加到文件末尾并附带注释说明。你原有的配置不会被修改。如果不需要这个新功能，保持默认值即可。

**Q：想完全重置配置**
A：删除 `~/.hermes/prism/config.yaml` 后重启，会自动重新生成包含所有默认值的模板。

**Q：国内下载模型很慢**
A：在配置文件中设置 `semantic.hf_endpoint_strategy: mirror_first`，会优先从国内镜像 `hf-mirror.com` 下载。
