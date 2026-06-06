# Prism CLI 命令参考

Prism 提供一组运维 CLI 工具，通过 `python -m prism <子命令>` 或安装后直接 `prism <子命令>` 调用。

```bash
# 查看所有子命令
prism --help
```

退出码约定（所有子命令通用）：`0` 成功 / `1` 部分失败或警告 / `2` 路径不存在或参数错误。

---

## 通用参数

以下参数在 `memory`、`migrate`、`reindex`、`vstore-migrate`、`export` 中通用：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--db` | 直接指定 DB 路径（跳过 path_template 解析） | 无 |
| `--user-id` | 用户 ID（sha256 哈希后嵌入 DB 路径） | `local_default` |
| `--profile` | profile 名（嵌入 DB 路径） | `default` |
| `--data-home` | 数据根目录覆盖 | `~/.prism` |
| `--config` | YAML 配置文件路径 | `default_config()` |

`--user-id` / `--profile` / `--data-home` 仅在 `--db` 未指定时生效。

---

## doctor — 健康自检

一条命令逐项验证环境、依赖、配置、DB、语义后端、Hermes 集成、端到端 smoke test。

```bash
prism doctor
prism doctor --no-smoke          # 跳过端到端写入+召回测试
prism doctor --config config.yaml
prism doctor --json              # JSON 输出（供脚本 / LLM agent 消费）
```

**检查组：**

1. **环境** — Python 版本、HERMES_HOME、hermes venv 检测
2. **必备依赖** — sqlite3、numpy、pyyaml、jieba
3. **可选依赖** — sentence-transformers、hnswlib、faiss、psycopg、qdrant-client、mcp
4. **Hermes 集成** — plugin.yaml、$HERMES_HOME/plugins/prism 软链、memory.provider
5. **Config / DB** — config.yaml 解析、DB 路径解析、DB 可访问性、vstore.npz
6. **语义后端** — backend 配置、BGE HF cache、HF endpoint 策略
7. **End-to-end smoke** — 临时 DB add → search 完整链路验证

**退出码：** `0` 全部正常 / `1` 至少一项 warn（降级可用） / `2` 至少一项 fail（核心不可用）。

---

## memory — 记忆人工维护

13 个子命令，覆盖 fact 的完整生命周期管理。

```bash
prism memory <子命令> [选项]
```

### mirror — MEMORY.md 镜像

从 Hermes `MEMORY.md` 文件批量镜像 fact 到 DB。

```bash
prism memory mirror --md ~/.hermes/memories/MEMORY.md
prism memory mirror --md MEMORY.md --prune  # 完成后清理孤儿 fact
```

### list — 浏览 fact

```bash
prism memory list                              # 默认显示 active fact
prism memory list --status archived            # 显示已归档的
prism memory list --category user_pref         # 按 category 过滤
prism memory list --mirror-source builtin_memory
prism memory list --limit 100 --offset 50      # 分页
```

### show — 显示 fact 详情

```bash
prism memory show 42    # 按 fact_id 显示完整字段 + 关联实体
```

### search — 语义查询

```bash
prism memory search "用户的数据库偏好"
prism memory search "PostgreSQL" --limit 5 --category tech
```

### add — 新增 fact

```bash
prism memory add "用户偏好 PostgreSQL 14"
prism memory add "项目截止日期是 2026-07-01" --category project
```

### edit — 软替换

旧 fact 归档，新建一条 supersedes 链。

```bash
prism memory edit 42 --content "用户偏好 PostgreSQL 16"
prism memory edit 42 --content "新内容" --category user_env
```

### remove / archive — 软删除

```bash
prism memory remove 42
prism memory remove 42 --reason "已过期"
prism memory archive 42 --reason "用户要求删除"
```

### restore — 恢复归档

```bash
prism memory restore 42    # 已归档 → active
```

### helpful / unhelpful — 反馈标记

```bash
prism memory helpful 42      # trust_score 上升
prism memory unhelpful 42    # trust_score 下降
```

### stats — 运行统计

```bash
prism memory stats
prism memory stats --category user_pref
prism memory stats --json    # JSON 输出
```

### enrichment — enrichment 诊断/修复

```bash
prism memory enrichment --diagnose            # 查看队列、状态分布、缺失向量
prism memory enrichment --diagnose --json
prism memory enrichment --fix                 # 修复缺失 embedding + 清队列 + 重建 vstore
prism memory enrichment --fix --dry-run       # 预览不修改
```

---

## migrate — 数据迁移

从 Holographic memory_store.db 迁移到 Prism DB。保留 trust_score、created_at、helpful_count、retrieval_count。按 content 去重，幂等可重跑。

```bash
prism migrate --from holographic --src ~/.hermes/holographic/memory_store.db
prism migrate --from holographic --src old.db --dst new_prism.db
prism migrate --from holographic --src old.db --user-id alice --profile work
```

| 参数 | 说明 |
|------|------|
| `--from` | 源类型（目前仅 `holographic`），必填 |
| `--src` | 源 DB 路径，必填 |
| `--dst` | 目标 DB 路径（未指定走 path_template 解析） |

---

## reindex — 嵌入模型升级

全量重编码 facts.semantic_vector 到指定 embedding 模型，并重建 vstore.npz。幂等，中断后重跑只补未升级的。

```bash
prism reindex --model BAAI/bge-base-zh-v1.5
prism reindex --model BAAI/bge-base-zh-v1.5 --dim 768 --batch-size 32
prism reindex --model BAAI/bge-base-zh-v1.5 --dry-run   # 仅报告待升量
prism reindex --model BAAI/bge-base-zh-v1.5 --db /path/to/prism.db
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 目标 embedding 模型名，必填 | — |
| `--dim` | 向量维度（不传自动检测） | 自动 |
| `--batch-size` | 单批 encode 大小 | 64 |
| `--dry-run` | 仅扫描报告 | false |

---

## vstore-migrate — Vector Store Backend 切换

把 facts.semantic_vector 全量重建到指定 backend。不改 SQLite 中的向量内容，只改 ANN 索引。

```bash
prism vstore-migrate --to hnswlib
prism vstore-migrate --to faiss --dim 512
prism vstore-migrate --to pgvector     # 需设 PRISM_PGVECTOR_DSN 环境变量
prism vstore-migrate --to qdrant       # 需设 PRISM_QDRANT_URL 环境变量
prism vstore-migrate --to local_numpy --dry-run
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--to` | 目标 backend（local_numpy / hnswlib / faiss / pgvector / qdrant），必填 | — |
| `--vstore-path` | 持久化路径 | `<db>.vstore.{npz\|hnsw\|faiss}` |
| `--dim` | 目标 dim（不传从首行推断） | 自动 |
| `--dry-run` | 仅扫描报告 | false |

---

## eval — 评估集测试

跑评估集（jsonl 格式）输出 P@k / R@k / MRR 和 must_include / must_exclude 通过率。

```bash
prism eval --dataset tests/fixtures/eval_set_zh.jsonl
prism eval --dataset eval.jsonl --retriever prism   # 用完整 Prism 检索器
prism eval --dataset eval.jsonl --verbose           # 打印 per-query 明细
prism eval --dataset eval.jsonl --output report.json
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 评估集 jsonl 路径，必填 | — |
| `--retriever` | `naive`（substring baseline）或 `prism`（端到端） | naive |
| `--output` | 汇总 JSON 输出路径 | 无（仅 stderr） |
| `--verbose` | 打印 per-query 明细 | false |

---

## export — 导出

把全库（或子集）导出为 jsonl，不含向量 blob。

```bash
prism export                                 # 导出 active fact 到 stdout
prism export --output facts.jsonl
prism export --status all                    # 包含已归档的
prism export --status archived --category user_pref
prism export --db /path/to/prism.db --output out.jsonl
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--status` | `active` / `archived` / `all` | active |
| `--category` | 过滤 category | 全部 |
| `--output` | 输出 jsonl 路径（不传写 stdout） | stdout |

导出字段：`fact_id`、`content`、`category`、`tags`、`status`、`trust_score`、`helpful_count`、`retrieval_count`、`ttl_days`、`created_at`、`last_retrieved_at`、`archived_at`、`archive_reason`、`supersedes_id`、`embedding_model`、`vector_store`、`mirror_source`、`mirror_target`、`enrichment_status`、`entities`。
