# 实验结果与数据产物

## 总览

| 周次 | 结果目录 / 主要产物 | 状态 |
|---|---|---|
| Week 1 | `week1-data-stats/`；预处理数据位于服务器 `/data/zby/data/ml-1m/processed/` | ✅ 完成 |
| Week 2 | `week2-sasrec-baseline/`；checkpoint 位于服务器 `~/genrec/saved/` | ✅ 完成 |
| Week 3–4 | `item_emb.npy` 已完成；语义 ID、码本报告与冻结接口待生成 | 🚧 进行中 |
| Week 5–6 | 生成式模型训练日志与对比表 | ⏳ 未开始 |
| Week 7 | 冷启动 / 长尾实验、消融与最终报告 | ⏳ 未开始 |

约定：

- **入库**：体积小、适合审阅的统计、干净日志、表格和图片。
- **服务器保留**：原始数据、可重建的中间数组、checkpoint、TensorBoard 事件和带进度条的原始日志。
- RecBole 日志中的用户 / 物品数量会包含 1 个 PAD token；真实规模仍是 **6,040 用户、3,416 物品**。

## Week 1：数据准备与基础分析

### 数据结论

| 指标 | 结果 |
|---|---:|
| 原始评分 | 1,000,209 |
| 5-core 后交互 | 999,611 |
| 用户 / 物品 | 6,040 / 3,416 |
| train / valid / test 交互 | 987,531 / 6,040 / 6,040 |
| 平均 / 中位序列长度 | 165.5 / 96 |
| 序列长度 P90 / 最大值 | 400 / 2,277 |
| 物品流行度 Gini | 0.6034 |
| Top 10% 物品覆盖的交互 | 42.31% |
| 时间跨度 | 2000-04-25 ～ 2003-02-28 |

### 仓库内文件

| 文件 | 生成者 | 内容 |
|---|---|---|
| [`preprocess_log.json`](week1-data-stats/preprocess_log.json) | `scripts/preprocess.py` | raw → 5-core → split 的规模记录 |
| [`stats.md`](week1-data-stats/stats.md) | `scripts/stats.py` | 人类可读的数据统计与抽样序列 |
| [`stats.json`](week1-data-stats/stats.json) | `scripts/stats.py` | 机器可读统计 |
| [`item_freq.json`](week1-data-stats/item_freq.json) | `scripts/export_item_freq.py` | 3,416 个物品的交互频次，供长尾评测使用 |
| [`item_freq_hist.png`](week1-data-stats/figs/item_freq_hist.png) | `scripts/stats.py` | 物品热度直方图 |
| [`item_freq_loglog.png`](week1-data-stats/figs/item_freq_loglog.png) | `scripts/stats.py` | 物品热度双对数图 |
| [`seq_len_hist.png`](week1-data-stats/figs/seq_len_hist.png) | `scripts/stats.py` | 用户序列长度分布 |

### 服务器数据文件

原始数据位于 `/data/zby/data/ml-1m/raw/`：

| 文件 | 规模 | 用途 |
|---|---:|---|
| `ml-1m.zip` | 5.9 MB | GroupLens 原始压缩包 |
| `README` | 5.6 KB | 数据集官方字段说明 |
| `ratings.dat` | 24.6 MB；1,000,209 行 | `UserID::MovieID::Rating::Timestamp`，唯一行为数据源 |
| `movies.dat` | 171 KB；3,883 行 | `MovieID::Title::Genres`，唯一物品文本来源 |
| `users.dat` | 134 KB；6,040 行 | 用户画像；当前协议不使用 |

`scripts/preprocess.py` 生成 `/data/zby/data/ml-1m/processed/`：

| 文件 | 规模 / 形状 | 下游用途 |
|---|---|---|
| `ml-1m.inter` | 21.5 MB；999,611 条交互 + 表头 | RecBole 的 SASRec / GRU4Rec 输入 |
| `train.jsonl` | 5.7 MB；6,040 行 | 每用户一条训练序列，共 987,531 次训练交互 |
| `valid.jsonl` | 5.8 MB；6,040 行 | `history + target`，用于模型选择 |
| `test.jsonl` | 5.9 MB；6,040 行 | `history + target`，用于最终评测 |
| `user_id_map.json` | 94 KB；6,040 个映射 | 原始用户 ID → 连续 ID `0..6039` |
| `item_id_map.json` | 53 KB；3,416 个映射 | 原始物品 ID → 连续 ID `0..3415` |
| `item_text.json` | 179 KB；3,416 条文本 | Week 3 文本编码输入 |
| `preprocess_log.json` | 299 B | 数据版本与规模校验 |
| `stats/stats.md` | 4.4 KB | 服务器端统计报告 |
| `stats/stats.json` | 568 B | 服务器端机器可读统计 |
| `stats/figs/*.png` | 3 张图 | 与仓库 `week1-data-stats/figs/` 对应 |

RecBole 通过软链接读取同一份数据：

```text
/data/zby/data/recbole/ml-1m -> /data/zby/data/ml-1m/processed
```

### 最小真实样例

```text
# ratings.dat
1::1193::5::978300760

# ml-1m.inter（TAB 分隔；ID 已重映射）
userID:token    itemID:token    timestamp:float
0               2757            978300019.0
```

```json
// train.jsonl
{"user": 0, "items": [2757, 861, 1068, "..."]}

// valid/test.jsonl
{"user": 0, "history": [2757, 861, 1068, "..."], "target": 1578}

// item_text.json
{"0": "Toy Story (1995) | Animation|Children's|Comedy"}
```

`train.items = valid.history`，而 `test.history = valid.history + valid.target`；`scripts/verify_split.py`
已逐用户检查时间单调、目标位置与历史前缀，结果为 **PASS**。

## Week 2：SASRec / GRU4Rec 基线

### 测试集结果

评测为 3,416 物品全量排序。SASRec 的标准差使用 `numpy.std(ddof=0)`。

| 模型 | Seed | 总 epoch | Best epoch | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SASRec | 42 | 43 | 31 | 0.1409 | 0.2204 | 0.0884 | 0.1141 |
| SASRec | 43 | 109 | 97 | 0.1523 | 0.2339 | 0.0968 | 0.1230 |
| SASRec | 44 | 59 | 47 | 0.1526 | 0.2250 | 0.0958 | 0.1191 |
| **SASRec** | **均值 ± std** | — | — | **0.1486 ± 0.0054** | **0.2264 ± 0.0056** | **0.0937 ± 0.0037** | **0.1187 ± 0.0036** |
| **GRU4Rec** | **42** | **67** | **55** | **0.1861** | **0.2657** | **0.1261** | **0.1517** |

独立汇总表见 [`baseline_table.md`](week2-sasrec-baseline/baseline_table.md)。

### 仓库内文件

| 文件 | 来源 | 记录内容 |
|---|---|---|
| [`baseline_table.md`](week2-sasrec-baseline/baseline_table.md) | 从下列日志逐项汇总 | 测试指标、epoch 与协议 |
| `sasrec_s42.log` | `log/SASRec/SASRec-ml-1m-Sep-28-2026_19-17-25-dbfc42.log` | seed 42 完整训练日志 |
| `sasrec_s42_patch_test.log` | `log/SASRec/SASRec-ml-1m-Oct-10-2026_15-55-25-dbfc42.log` | 从 seed 42 best checkpoint 补跑 valid / test |
| `sasrec_s43.log` | `log/SASRec/SASRec-ml-1m-Oct-15-2026_13-00-11-6e006f.log` | seed 43 完整训练与测试日志 |
| `sasrec_s44.log` | `log/SASRec/SASRec-ml-1m-Oct-15-2026_17-48-15-37672d.log` | seed 44 完整训练与测试日志 |
| `gru4rec_s42.log` | `log/GRU4Rec/GRU4Rec-ml-1m-Oct-15-2026_09-39-08-c5973e.log` | GRU4Rec seed 42 完整训练与测试日志 |

seed 42 的 `eval_only.py` 结果与 best epoch 的 valid 指标逐位一致，验证了 checkpoint 读取与评测链路。

### 服务器保留、不进入 Git 的文件

| 位置 | 当前文件 | 原因 |
|---|---|---|
| `~/genrec/saved/` | 3 个 SASRec `.pth` + 1 个 GRU4Rec `.pth` | 二进制 checkpoint；用于复评与断点恢复 |
| `~/genrec/results/` | `sasrec_s42.log`、`sasrec_s42_patch_test.log`、`sasrec_s43.log`、`sasrec_s44.log`、`gru4rec_s42.log` | `tee` 捕获的原始 stdout / stderr，含 tqdm 控制字符 |
| `~/genrec/log_tensorboard/` | 4 个训练 run + 1 个 eval-only run(seed = 42时补跑的test) | TensorBoard 二进制事件 |

仓库只保留 RecBole 自写的干净文本日志；服务器路径和确切文件名见根目录 [`README.md`](../README.md) 的服务器目录树。

## Week 3–4：物品语义标识符

### 已完成：文本编码

| 项 | 结果 |
|---|---|
| 输入 | `item_text.json`，3,416 条 `title (year) \| genres` |
| 编码器 | `/data/LLMs/all-MiniLM-L6-v2` |
| 输出 | 服务器 `~/genrec/item_emb.npy` |
| 形状 / dtype | `(3416, 384)` / `float32` |
| L2 范数 min / mean / max | 0.99999982 / 1.00000000 / 1.00000012 |
| NaN / Inf / 全零行 | 0 / 0 / 0 |
| SHA-256 | `162d6d2a9fb89e6b9e096d97a79a46338d83507ad5cbc07581a730d43e2b77a4` |
| 语义 sanity check | 同组余弦 0.5491；随机配对 0.2774；差值 +0.2717 |
| 案例 | Toy Story (1995) / Toy Story 2 (1999) 余弦 0.8027 |

`item_emb.npy` 是可由 `scripts/encode_items.py` 重建的中间数组，因此不进入 Git。

### 待生成：量化、分析与冻结接口

以下文件尚未生成，不用计划值冒充结果：

| 计划产物 | 作用 | 状态 |
|---|---|---|
| `semantic_ids_kmeans.json` / `semantic_ids_rqvae.json` | RQ-Kmeans baseline 与 RQ-VAE 主方案的 4-token 语义 ID | ⬜ |
| `saved/rq_kmeans_ml1m.npz` / `saved/rqvae_ml1m.pt` | 量化器参数与训练配置 | ⬜ |
| `results/week3-4-semantic-id/codebook_stats.json` | 逐层利用率、困惑度和冲突率 | ⬜ |
| `results/week3-4-semantic-id/figs/*.png` | 两种方法 × 三层的码字分布图 | ⬜ |
| `grid_search.md` | `(L,K)` 网格对比 | ⬜ |
| `prefix_consistency.md` / `prefix_samples.csv` | 前缀语义一致性与人工正反例 | ⬜ |
| `codebook_report.md` | 阶段二汇总报告 | ⬜ |
| `frozen/semantic_ids.json` | 第 5 周使用的最终物品语义 ID | ⬜ |
| `frozen/train.semantic.jsonl` | 语义 token 训练集 | ⬜ |
| `frozen/valid.semantic.jsonl` | 语义 token 验证集 | ⬜ |
| `frozen/test.semantic.jsonl` | 语义 token 测试集 | ⬜ |
| `frozen/token_vocab.json` / `frozen/meta.json` | token 映射、协议与输入产物哈希 | ⬜ |

## Week 5–7

尚无实验文件。后续结果将继续按周放入：

```text
results/
├── week5-6-generative-model/
└── week7-ablation-and-final/
```

## 复现入口

| 目标 | 命令 |
|---|---|
| 生成 Week 1 数据 | `python scripts/preprocess.py --dataset ml-1m` |
| 生成统计与图片 | `python scripts/stats.py --dataset ml-1m` |
| 校验数据划分 | `python scripts/verify_split.py --data-dir /data/zby/data/ml-1m/processed` |
| 导出物品频次 | `python scripts/export_item_freq.py --out item_freq.json` |
| 训练基线 | `python scripts/run_baseline.py --model SASRec --config config/sasrec_ml1m.yaml --seed 42` |
| checkpoint 复评 | `python scripts/eval_only.py --model_file saved/<checkpoint>.pth` |
| 编码物品文本 | `python scripts/encode_items.py --model /data/LLMs/all-MiniLM-L6-v2 --out item_emb.npy` |
