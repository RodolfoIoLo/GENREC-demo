# results —— 产出索引与说明

这里放的是**脚本跑出来的东西**（不是数据、也不是代码）。每个文件都能用下面这条命令重新生成。

## 1. 入库的产出

| 文件 | 生成命令 | 说明 |
|---|---|---|
| `week1-data-stats/stats.md` | `python scripts/stats.py --dataset ml-1m` | 数据规模 / 序列长度 / 物品长尾 / 时间跨度 |
| `week1-data-stats/stats.json` | 同上 | 机器可读版本 |
| `week1-data-stats/figs/item_freq_hist.png` | 同上 | 物品流行度分布 |
| `week1-data-stats/figs/item_freq_loglog.png` | 同上 | 双对数坐标（看长尾是不是幂律） |
| `week1-data-stats/figs/seq_len_hist.png` | 同上 | 用户序列长度分布 |
| `week1-data-stats/preprocess_log.json` | `python scripts/preprocess.py --dataset ml-1m` | 预处理各阶段计数（raw → 5-core → split） |
| `week2-sasrec-baseline/train_0928.log` | `python scripts/run_baseline.py` | SASRec 训练完整日志，339 行 |
| `week2-sasrec-baseline/test_1010.log` | `python scripts/eval_only.py --model_file saved/*.pth` | 从 best checkpoint 补跑的 test 评测 |
| `samples.md` | — | **没入库的大文件的头部样本**（`.inter` / `*.jsonl` / `.pth` …） |

## 2. 服务器上运行一次会生成什么

`~/genrec/` 下跑一次 `run_baseline.py` 会产生 **4 个目录**：3 个是 RecBole 自动建的，第 4 个是我们自己 `tee` 的。

| 目录 | 谁生成的 | 命名规则 | 要不要留 |
|---|---|---|---|
| `saved/` | RecBole（`trainer.py:132` 定名，`:308` 写盘） | `{模型}-{时间}.pth` | **留** —— 复现 test 指标靠它 |
| `log/{模型}/` | RecBole（`utils/logger.py:81`） | `{模型}-{数据集}-{时间}-{配置指纹}.log` | 留要复现的那次 |
| `log_tensorboard/{同名}/` | RecBole（`utils/utils.py:208`） | 与日志同名 | 想看曲线才留 |
| `results/` | **我们自己**（命令行 `\| tee`） | 手动命名 | 可删（干净版在 `log/` 里） |

> **三个 RecBole 目录的名字里都带时间戳 → 每跑一次新建一套，跑 N 次累积 N 套。**

### 2.1 `saved/` —— 模型权重

```
saved/SASRec-Sep-28-2026_19-18-04.pth          3.9 MB
```

⚠️ 名字里**没有数据集名、也没有配置指纹**，换数据集跑会撞名，只能靠时间戳区分。

⚠️ 名字里的时间戳**不是保存时刻**，是 `Trainer` 被构造出来的时刻（训练刚开始）。
实测：文件名写 `19:18:04`，文件真实修改时间是 `20:17` —— 差 59 分钟，正好是 epoch 0 → 31 的耗时。

**里面装了什么**（`torch.load` 出来的 7 个顶层 key）：

| key | 值 | 说明 |
|---|---|---|
| `config` | 一个完整的 `Config` 对象 | **checkpoint 是自描述的** —— 模型名、数据集、数据路径、seed、全部超参都能从它恢复。`eval_only.py` 就靠它重建模型和数据集 |
| `epoch` | `31` | 存盘时是第几个 epoch（= best epoch） |
| `cur_step` | `0` | 早停计数器，存的是 best，所以是 0 |
| `best_valid_score` | `0.1283` | 该 epoch 的 valid NDCG@10，与结果表对上 |
| `state_dict` | 36 个张量 | 真正的权重：`item_embedding.weight (3417,64)`、`position_embedding.weight (200,64)`、2 层 Transformer 的 Q/K/V/dense/FFN/LayerNorm，合计 **331,584** 个参数 |
| `other_parameter` | `{}` | SASRec 没有额外参数表 |
| `optimizer` | `{'state', 'param_groups'}` | **Adam 的动量状态**，参数量 × 2 × 4 B ≈ 2.65 MB |

**为什么有 3.9 MB**：权重本身只有 331,584 × 4 B ≈ 1.33 MB，剩下约 2/3 全是 Adam 动量。
那部分只为断点续训，**对「拿模型做预测」完全没用**。

存盘用 `torch.save(..., pickle_protocol=4)`（`trainer.py:308`），这就是加载时那条
`Detected pickle protocol 4 in the checkpoint` 警告的来源。

**怎么用**：`python scripts/eval_only.py --model_file saved/SASRec-Sep-28-2026_19-18-04.pth`

### 2.2 `log/{模型}/` —— 干净文本日志

```
log/SASRec/SASRec-ml-1m-Sep-28-2026_19-17-25-dbfc42.log
           └模型┘ └数据集┘ └──时间──┘ └指纹┘
```

**那 6 位十六进制是配置指纹**（`utils/logger.py:81`）：

```python
config_str = "".join([str(key) for key in config.final_config_dict.values()])
md5 = hashlib.md5(config_str.encode("utf-8")).hexdigest()[:6]
```

配置一字不差 → 指纹相同。实测我们 3 次运行的日志后缀全是 `dbfc42`。
**改任何一个超参，这 6 位就会变** —— 这是 RecBole 送的「同配置识别码」。

**339 行里写了什么**（按顺序）：

| 段 | 内容 |
|---|---|
| 1 行 | 启动命令本身，例如 `['scripts/run_baseline.py', '--model', 'SASRec']` |
| 约 112 行 | **`Final args`** —— 完整配置，分 General / Training / Evaluation / Dataset / Other Hyper Parameters 五段。**这就是实验配置记录本身，写报告直接贴，别手抄** |
| 约 6 行 | 数据集统计：`users: 6041`、`items: 3417`、`inters: 999611`、`sparsity: 95.1574%` |
| 2 行 | `[Training]` / `[Evaluation]` 的关键参数摘要 |
| 约 27 行 | **模型结构打印** —— 逐层列出 `item_embedding` / `position_embedding` / 2×TransformerLayer 的子模块，末行 `Trainable parameters: 331584` |
| 1 行 | `FLOPs: 20293664.0`（约 20.3 MFLOPs） |
| 43 × 4 行 | 每个 epoch 固定 4 行：`epoch N training [time, train loss]` → `epoch N evaluating [time, valid_score]` → `valid result: ` → 一行 4 个指标 |
| +16 行 | 刷新最优时多一行 `Saving current: saved/….pth`（整个训练共刷新 16 次） |
| 最后 1 行 | `Finished training, best eval result in epoch 31` |

### 2.3 `log_tensorboard/` —— 训练曲线数据

目录名与日志同名，里面是 `events.out.tfevents.<时间>.<主机名>.<pid>.<序号>` —— TensorBoard 的二进制事件文件。

实测 Sep-28 那份记了两条曲线：

```
Loss/Train    43 个点    3362.31 → 2721.19
Vaild_score   43 个点    0.0649  → 0.1274
```

43 个点 = 43 个 epoch。

**怎么看**：`tensorboard --logdir log_tensorboard --port 6006`，再开 SSH 端口转发到本地，浏览器访问 `localhost:6006`。

只跑 `eval_only`（不训练）也会新建一个目录，但因为没训练，只是个 88 字节的空文件。

### 2.4 `results/` —— 我们自己 tee 的

**这个不是 RecBole 生成的** —— RecBole 不知道这个目录存在。它是命令行里的重定向：

```bash
python scripts/run_baseline.py ... 2>&1 | tee results/sasrec_0928_1916.log
```

所以它抓的是**含 tqdm 进度条的原始 stdout + stderr**。tqdm 用 `\r` 刷新，被 `tee` 收进文件后编辑器把 `\r` 当换行，
5 MB 的日志看起来有「上万行」，其实只有 43 个 epoch。

### 2.5 清理建议

| 目标 | 判断 |
|---|---|
| `saved/*.pth` | **必留** |
| `log/{模型}/*.log` | 留要复现的那次，其余可删 |
| `log_tensorboard/` | 40 KB，想看曲线就留 |
| `results/*.log` | 干净版已入库 → 可删 |
| `~/genrec/data` | **死链**（`-> /data/zby/ml-1m`，目标不存在），可删 |

## 3. 关于没入库的产出

数据、checkpoint、被刷过的原始日志都不进 Git —— 逐条原因和它们的**真实内容样本**见 [`samples.md`](samples.md)。
