# 基于语义标识符的生成式推荐方法研究与实现

把物品编码成「词」（语义标识符），把推荐变成「造句」（自回归生成）。

这是 **7 周科研轮转课题**的项目仓库，当前覆盖 **第 1–2 周**：数据管线 + SASRec 判别式基线。

| 阶段 | 内容 | 状态 |
|---|---|---|
| 第 1 周 | 数据准备与统计 | ✅ 完成 |
| 第 2 周 | RecBole 基线复现 | ✅ 完成 |
| 第 3–4 周 | 物品语义标识符构建 | ⏳ 待开始 |
| 第 5–6 周 | 生成式模型训练与评测 | ⏳ 待开始 |
| 第 7 周 | 改进探索、消融与结项 | ⏳ 待开始 |

---

## 快速开始

```bash
# 1. 环境（requirements.txt 里有三条硬 pin，装错会 import 失败，见 §1）
pip install torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 2. 数据（原始数据从 https://grouplens.org/datasets/movielens/1m/ 下载到 data/ml-1m/raw/）
python scripts/preprocess.py   --dataset ml-1m
python scripts/stats.py        --dataset ml-1m
python scripts/verify_split.py --data-dir /data/zby/data/ml-1m/processed    # 应输出 PASS

# 3. 基线（约 80 分钟，务必在 tmux 里跑）
python scripts/run_baseline.py --model SASRec --config config/sasrec_ml1m.yaml
```

---

## 仓库结构

```
repository/
├── README.md                  # 本文件
├── requirements.txt           # 环境快照（含三条硬 pin）
├── .gitignore
│
├── config/
│   └── sasrec_ml1m.yaml       # 第 2 周冻结的实验配置（复现的唯一入口）
│
├── scripts/
│   ├── preprocess.py          # 第 1 周：原始数据 → .inter + jsonl + 映射表 + item_text
│   ├── stats.py               # 第 1 周：数据统计 + 3 张分布图
│   ├── verify_split.py        # 第 1 周：逐用户校验划分（时间单调 / 无泄漏）→ PASS
│   ├── run_baseline.py        # 第 2 周：一键训练 + 评测
│   └── eval_only.py           # 第 2 周：从 checkpoint 只跑评测
│
├── data/
│   └── README.md              # 数据规格说明（仓库不放真实数据）
│
├── results/
│   ├── README.md              # 结果索引：每个文件 ← 哪条命令生成的
│   ├── samples.md             # 没入库的大文件长什么样（头部样本）
│   ├── week1-data-stats/      # 第 1 周产出：stats.md / stats.json / figs/ / preprocess_log.json
│   └── week2-sasrec-baseline/ # 第 2 周产出：train_0928.log / test_1010.log
│
└── note/                      # 学习笔记与周记
    ├── week0-生成式推荐_三阶段全链路.md
    ├── week1-真实数据实例说明.md
    ├── week2-前置知识-RecBole与基线模型.md
    └── 日志...
```

**文档导航**

| 我想知道… | 看哪里 |
|---|---|
| 数据每个文件是什么、字段什么意思、被谁读谁写 | [`data/README.md`](data/README.md) |
| 结果文件是哪个脚本跑出来的 | [`results/README.md`](results/README.md) |
| 那些没进仓库的文件长什么样 | [`results/samples.md`](results/samples.md) |
| 学习笔记 | [`note/`](note/) |
| 服务器上到底有哪些目录和文件 | 本文件 §6 |

---

## 1. 环境

- Python 3.11 + PyTorch 2.14.0 (cu126) + RecBole 1.2.1，服务器 4×RTX 4090
- 服务器 `amax510`，conda 环境名 `genrec`
- 完整依赖快照：`requirements.txt`

> ⚠️ `requirements.txt` 里有**三条硬 pin 必须保留**：`setuptools<82`、`numpy<2`、`scipy<1.18`。
> 少任意一条，`import recbole` 都会直接崩。已验证的可用组合是
> **numpy 1.26.4 + scipy 1.17.1 + setuptools 81.0.0**，原因写在 `requirements.txt` 的注释里。

装完自检：

```bash
python -c "import numpy, scipy, torch; print(numpy.__version__, scipy.__version__, torch.__version__, torch.cuda.is_available())"
python -c "import recbole; print('recbole OK', recbole.__version__)"
pip check
```

---

## 2. 数据

- 数据集：**MovieLens ml-1m**（原始：6040 用户 / 3706 物品 / 1,000,209 条评分）
- 5-core 迭代过滤后：**6040 用户 / 3416 物品 / 999,611 条交互**
- 与 SASRec 论文（arXiv:1808.09781）在 ML-1M 上的预处理结果**完全一致**（论文同样丢弃交互数 < 5 的用户和物品）

划分（leave-one-out，按时间）：

```
每个用户的交互按 timestamp 升序排列
  ├── 最后 1 条      → test
  ├── 倒数第 2 条    → valid
  └── 其余           → train

交互层：train 987,531 / valid 6,040 / test 6,040  （合计 999,611）
样本层：train 981,491 / valid 6,040 / test 6,040
        （每用户 N 条 train 交互展开成 N−1 条序列样本，第一条交互不产生样本）
```

> 样本之间高度重叠：同一用户的 N−1 条样本前缀逐层嵌套，一条交互会同时出现在
> 「1 次当目标」和「多次当历史」的位置上。

**每个文件是什么、里面长什么样、被谁读、被谁写** → 见 [`data/README.md`](data/README.md)。
数据本身不进仓库（体积 + 可复现性，理由见该文档 §4）。

---

## 3. 实验协议（v2，已冻结）

**红线：协议冻结后不再改。换数据集或换划分 = 前面所有结果作废。对比表的列就是协议的一部分。**

### 3.1 数据层（第 1 周确定）

| 项 | 取值 |
|---|---|
| 数据集 | ml-1m |
| 反馈类型 | 隐式，有交互即正样本（1–5 分不设阈值） |
| k-core | k=5，迭代至收敛 |
| 划分 | leave-one-out，按时间：最后 1 条 = test，倒数第 2 条 = valid |
| 评价方式 | **全量排序**（候选集 = 全部 3416 个物品，不是采样） |
| 指标 | Recall@5/10、NDCG@5/10 |
| 随机种子 | 42 |
| ID 处理 | 重映射为 0..N−1 连续整数 |

### 3.2 训练 / 评测层（第 2 周确定）

| 项 | 取值 |
|---|---|
| 基线实现 | **RecBole 1.2.1**（锁定版本，不换实现） |
| 最大序列长度 | `MAX_ITEM_LIST_LENGTH = 200`（RecBole 默认 50 太短） |
| 评测屏蔽 | **屏蔽训练集中出现过的物品**（分数置 −inf，RecBole 官方行为；自研评测脚本必须照做） |
| 损失口径 | CE（交叉熵，不做负采样） |
| 早停 | `valid_metric = NDCG@10`，`stopping_step = 10`；报告中写实际跑到的 epoch 数 |
| 随机种子 | 先跑单种子 42；有余力补 3 种子均值 ± 标准差 |

---

## 4. 结果

| 模型 | 实现 | 评测集 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 随机排序（理论） | — | — | 0.0015 | 0.0029 | 0.0009 | 0.0013 |
| **SASRec** | RecBole 1.2.1 | valid | 0.1601 | 0.2505 | 0.0993 | 0.1283 |
| **SASRec** | RecBole 1.2.1 | **test** | **0.1409** | **0.2204** | **0.0884** | **0.1141** |

> 数字看起来比 SASRec 论文低很多（论文 ML-1M 上 Hit@10 = 0.8245），这是**口径不同**：
> 论文用 1 正 + 100 随机负采样，我们用全量 3416 候选，候选集大 34 倍。两者不可比。

随机基线 = `K / 3416`（1 个正样本在全部物品里随机排）。test Recall@10 约为随机的 **75 倍**。

**指标口径**（leave-one-out 下每用户只有 1 个正确答案）：

- `Recall@K` 与 `Hit@K` **完全等价**（命中非 0 即 1），不是简写，是定义上的相等
- `NDCG@K` 的 IDCG 恒为 1（最理想就是排第 1 得 1.0），所以它等于**命中位置折扣的平均**：
  排第 1 → 1.000，第 2 → 0.631，第 3 → 0.500，第 5 → 0.387，第 10 → 0.289
- 自洽校验：`NDCG@10 / Recall@10 = 0.1141 / 0.2204 ≈ 0.518`，说明命中时平均排在**第 2–3 名**

### 训练细节

| 项 | 值 |
|---|---|
| 日志 | `results/week2-sasrec-baseline/train_0928.log`（训练）、`test_1010.log`（test 评测） |
| 实际训练轮数 | **43 个 epoch（0–42）**，`stopping_step=10` 触发早停（`epochs=300` 是上限，没用满） |
| 最优 epoch | **31** |
| 单 epoch 耗时 | 约 112 秒（480 个 batch，batch_size=2048） |
| 参数量 | **331,584**（其中 item embedding 占 218,688） |
| 网络结构 | 2 层 Transformer，2 头，hidden 64，inner 256，GELU，post-LN |
| 训练 loss | epoch 0 = 3362.31 → epoch 42 = 2721.19（日志是**全 epoch 累加**口径，除以 480 得单条样本平均 ≈ 7.0 → 5.67） |

### 复现性证据

用 `eval_only.py` 从最优 checkpoint 重新评测 valid，得到
`0.1601 / 0.2505 / 0.0993 / 0.1283`，与训练日志里 epoch 31 的记录**逐位相同**。
说明 `seed=42 + reproducibility=True` 的固定种子链路是可靠的。

---

## 5. 复现步骤

### 5.1 数据（第 1 周）

```bash
python scripts/preprocess.py --dataset ml-1m    # 原始 → processed/
python scripts/stats.py      --dataset ml-1m    # 统计表 + 3 张图
python scripts/verify_split.py --data-dir /data/zby/data/ml-1m/processed   # 应输出 PASS
```

### 5.2 基线（第 2 周）

```bash
cd ~/genrec
python scripts/run_baseline.py --model SASRec --config config/sasrec_ml1m.yaml
```

> ⚠️ **必须用 tmux 跑**（约 80 分钟）。SSH 断开会杀掉前台进程：
> `tmux new -s sasrec` → 跑命令 → `Ctrl+B` `D` 脱离。

从已保存的 checkpoint 只补跑评测（约 1 分钟，不重新训练）：

```bash
python scripts/eval_only.py --model_file saved/SASRec-Sep-28-2026_19-18-04.pth
```

### 5.3 目录约定

数据放 `/data/zby/`，不放家目录（服务器约定）。RecBole 要求 `{data_path}/{dataset}/{dataset}.inter`，
所以做了一层软链接：

```bash
ln -s /data/zby/data/ml-1m/processed /data/zby/data/recbole/ml-1m
```

---

## 6. 服务器上的真实目录

主机 `amax510`（Tailscale `100.100.97.78`），用户 `zby`，conda 环境 `genrec`。

### 6.1 代码仓库 `~/genrec/`

```
/home/zby/genrec/
├── config/
│   └── sasrec_ml1m.yaml                                    4.0K
├── scripts/
│   ├── preprocess.py                                         12K
│   ├── stats.py                                              12K
│   ├── verify_split.py                                      8.0K
│   ├── run_baseline.py                                      4.0K
│   └── eval_only.py                                         4.0K
├── requirements.txt                                         4.0K
├── log/SASRec/                       ← RecBole 自写的干净日志（入库的就是这里面的）
│   ├── SASRec-ml-1m-Sep-28-2026_19-17-25-dbfc42.log          20K  ← 已入库
│   ├── SASRec-ml-1m-Oct-10-2026_15-53-50-dbfc42.log         4.0K
│   └── SASRec-ml-1m-Oct-10-2026_15-55-25-dbfc42.log         8.0K  ← 已入库
├── results/                          ← tee 出来的原始日志（不入库）
│   ├── sasrec_0928_1916.log                                 5.0M  ← 被 tqdm 的 \r 刷过
│   └── eval_only_1010.log                                   8.0K
├── saved/
│   └── SASRec-Sep-28-2026_19-18-04.pth                      3.9M  ← best epoch 31 的权重
└── log_tensorboard/                  ← 空（未启用）
```

### 6.2 数据 `/data/zby/data/`

```
/data/zby/data/
├── ml-1m/
│   ├── raw/                          ← 原始数据集（从 GroupLens 下载）
│   │   ├── ml-1m.zip                                        5.7M
│   │   ├── ratings.dat                                       24M
│   │   ├── movies.dat                                       168K
│   │   ├── users.dat                                        132K
│   │   └── README                                           8.0K
│   └── processed/                    ← preprocess.py 的产出
│       ├── ml-1m.inter                                       21M  ← RecBole 读的就是这个
│       ├── train.jsonl                                      5.5M
│       ├── valid.jsonl                                      5.6M
│       ├── test.jsonl                                       5.7M
│       ├── item_id_map.json                                  52K
│       ├── user_id_map.json                                  96K
│       ├── item_text.json                                   176K  ← 第 3 周的文本输入
│       ├── preprocess_log.json                              4.0K  ← 已入库
│       └── stats/
│           ├── stats.md                                     8.0K  ← 已入库
│           ├── stats.json                                   4.0K  ← 已入库
│           └── figs/                                                ← 已入库
│               ├── item_freq_hist.png                        32K
│               ├── item_freq_loglog.png                      40K
│               └── seq_len_hist.png                          32K
└── recbole/
    └── ml-1m -> /data/zby/data/ml-1m/processed             软链接（RecBole 路径要求）
```

> 每个文件里到底是什么内容，看 [`results/samples.md`](results/samples.md)；
> 跑一次会自动生成哪些文件、它们的命名规则和内部结构，看 [`results/README.md`](results/README.md) §2。

---

## 7. 已知问题与踩坑

1. **PyTorch 2.6+ 改了 `torch.load` 的 `weights_only` 默认值**（`False` → `True`），
   而 RecBole 1.2.1 的 `trainer/trainer.py:583` 是裸调用 → 训练结束加载最优模型做 test 时抛
   `_pickle.UnpicklingError`。**训练本身不受影响，但 test 指标会整段丢失。**
   → `run_baseline.py` 与 `eval_only.py` 在 `import recbole` 之前把默认值改回 `False`（checkpoint 是自产的可信文件）。

2. **`load_data_and_model` 不跑 `Config.__init__`**，因此 `compatibility_settings()` 也没跑，
   numpy 的旧别名（`np.float` / `np.bool` / `np.object` …）没被补回来，
   而 `evaluator/metrics.py:190` 的 NDCG 正要用 `np.float`。
   → `eval_only.py` 里手动补一句 `config.compatibility_settings()`。

3. **RecBole 的日志「上万行」是显示假象**。每个 epoch 其实只写 4 行；多出来的是 tqdm 往 stderr
   写的 `\r` 进度条被 `tee` 收进日志，编辑器把 `\r` 当成换行。
   → 用 RecBole 自写的 `log/<模型>/…` 那一份（无 ANSI 颜色码、无 `\r`）。

4. **为什么按类型分目录、而不是按 `week1/` `week2/` 分**：第 3–7 周的脚本会横跨多个周次
   （例如 `run_baseline.py` 在后续周里还要反复改超参跑），按类型分不会随周次增长而需要搬迁。
   「哪一周做了什么」由上面的进度表和 `results/weekN-*/` 子目录名承载。

---
