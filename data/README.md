# data/ -- 数据说明

本目录**不存放真实数据文件**,只保留本说明.

数据全流程由 `scripts/preprocess.py` 一次生成,产出到 `processed/`.下面按"**文件 → 内容 → 作用**"逐项说明.

> **本文件回答「数据是什么」**（字段含义、数量、谁读谁写）。
> **想知道「文件里实际长什么样」** → 看 [`../results/samples.md`](../results/samples.md)（真实头部样本）。
> 两者分工：这里是规格,那里是实物.

---

## 0. 目录结构

```
data/                                  ← 本目录(仓库内,只有这份说明)
└── README.md

/data/zby/data/ml-1m/                  ← 服务器上真实数据的位置(不进仓库)
├── raw/                               ← [输入]原始下载解压的文件
│   ├── ratings.dat
│   ├── movies.dat
│   └── users.dat
├── processed/                         ← [输出]脚本产物,下游全部读这里
│   ├── ml-1m.inter
│   ├── train.jsonl
│   ├── valid.jsonl
│   ├── test.jsonl
│   ├── user_id_map.json
│   ├── item_id_map.json
│   ├── item_text.json
│   ├── preprocess_log.json
│   └── stats/                         ← 统计产物(已入库到 results/week1-data-stats/)
│       ├── stats.md
│       ├── stats.json
│       └── figs/*.png
└── recbole/
    └── ml-1m -> ../ml-1m/processed     ← 软链接(RecBole 的路径要求)
```

**获取路径**:下载 `ml-1m.zip` → 解压到 `raw/` → 跑 `scripts/preprocess.py` → 得到 `processed/`.

**下载地址**:https://files.grouplens.org/datasets/movielens/ml-1m.zip

**生成命令**:

```bash
python scripts/preprocess.py --dataset ml-1m --data-root /data/zby/data --k 5 --seed 42
```

**校验命令**:

```bash
python scripts/verify_split.py --data-dir /data/zby/data/ml-1m/processed   # 应输出 PASS
```

---

## 1. 输入文件(`raw/`,原始下载)

### `ratings.dat`

- **内容**:用户对电影的评分记录,`::` 分隔,无表头.

  ```
  1::1193::5::978300760
  1::661::3::978302109
  ```

  字段依次为 `user_id :: movie_id :: rating :: timestamp`.评分为 1–5 整数,timestamp 是 Unix 秒.
  **1,000,209 条**记录(6,040 用户 × 3,883 部电影的非零组合).

- **作用**:**整个项目的唯一行为数据源**.所有用户序列,所有训练样本都从这里来.
  经过 `preprocess.py` 后被拆成 `.inter` 和三个 `.jsonl`.

### `movies.dat`

- **内容**:电影元信息,`::` 分隔,`latin-1` 编码(不是 UTF-8,有重音字符).

  ```
  1::Toy Story (1995)::Animation|Children's|Comedy
  2::Jumanji (1995)::Adventure|Children's|Fantasy
  ```

  字段依次为 `movie_id :: 标题(年份) :: 类别串(|分隔)`.**3,883 条**.

- **作用**:**项目里唯一的文本来源**.课题叫"基于**语义**标识符",但 `ratings.dat` 只有整数,
  没有语义,语义只能从文本来.第 3 周构造语义 ID 时,这里的"标题 + 类别"会被文本编码器
  转成向量,再进 RQ-VAE.**这份文件的质量直接决定语义标识符的质量上限.**

  > ⚠️ 两个实测到的坑:
  > ① `movie_id` 有 **69 个空号**(不连续,最大 3952),不能当数组下标用;
  > ② 表格对 `::` 转义处理不严,**标题里带 `::` 的少数行需要按"最后一个 `::` 之后是类别"解析**.

### `users.dat`

- **内容**:用户画像,`::` 分隔.

  ```
  1::F::1::10::48067
  ```

  字段依次为 `user_id :: 性别 :: 年龄区间码 :: 职业码 :: 邮编`.**6,040 条**.

- **作用**:**本项目未使用**(协议已冻结为"只用行为序列").保留说明是为了完整性 ----
  如果你想在第 7 周做"加入用户画像"的消融实验,数据在这里现成可用.

---

## 2. 输出文件(`processed/`,脚本产物)

### `ml-1m.inter`

- **内容**:全量交互日志,**Tab 分隔**,第 1 行是列名声明.

  ```
  userID:token	itemID:token	timestamp:float
  0	2757	978300019.0
  0	861	978300055.0
  ```

  共 **999,612 行** = 1 行表头 + **999,611 条交互**.
  冒号后的 `token` / `float` 是 **RecBole 的类型声明语法**(`token`=离散 ID 处理,`float`=数值处理).
  ID 已经是重映射后的连续值(`0..6039` / `0..3415`),timestamp 是 Unix 秒.
  按 `(user, timestamp, item)` 升序排列 ---- **这个行顺序就是 leave-one-out 的切分依据**.

- **作用**:**RecBole 的唯一输入口**.第 2 周的判别式基线(GRU4Rec,SASRec)全部通过它加载数据.
  RecBole 内部会按 `eval_args.split` 自己切分,所以配置必须与我们的 leave-one-out 对齐
  (`split: {'LS': 'valid_and_test'}` + `order: 'TO'`),否则跑出来的不是我们的协议.

### `train.jsonl` / `valid.jsonl` / `test.jsonl`

- **内容**:划分后的序列数据,JSON Lines,**每行一个用户**,各 **6,040 行**.

  `train.jsonl` 字段是 `user` + `items`(整条训练序列的关键字是 `items`,**注意不是 `history`**):

  ```json
  {"user": 0, "items": [2757, 861, 1068, 1444, 1978, 1519, 2925, 1007, 2411, 241, ...]}
  ```

  `valid.jsonl` / `test.jsonl` 字段是 `user` + `history` + `target`(`{int, list[int], int}`):

  ```json
  {"user": 0, "history": [2757, 861, 1068, ..., 1578], "target": 1578}   // valid
  {"user": 0, "history": [2757, 861, 1068, ..., 1578], "target": 46}     // test
  ```

  用户 0 实例:`train` 51 个物品 / `valid` history 51 个 + 目标 `Mulan (1998)` /
  `test` history 52 个 + 目标 `Pocahontas (1995)`.

  各集合条数:**train 987,531 条交互 / valid 6,040 / test 6,040**.

- **作用**:**生成式模型(第 5–6 周)的数据入口**.分工如下:

  | 文件 | 作用 |
  |---|---|
  | `train.jsonl` | 训练.每行整条序列,训练时内部展开成 `n−1` 条"前缀 → 下一个"样本 |
  | `valid.jsonl` | **调超参的靶子**.选层数,beam size,训几轮,只看 valid,不碰 test |
  | `test.jsonl` | **最终评测,只跑一次**.跑完直接写进报告的对比表 |

  > 与 `.inter` 的分工:`.inter` 给 **RecBole**(判别式基线)读,三个 `.jsonl` 给
  > **自研生成式模型**读.两者必须切分一致,否则基线和你自己的模型评测的不是同一个任务.
  > `verify_split.py` 就是逐用户校验这件事的(当前 PASS).

### `user_id_map.json`

- **内容**:用户 ID 重映射表,JSON 对象,**6,040 个键**.

  ```json
  {"1": 0, "2": 1, "3": 2, ...}
  ```

  key 是 ml-1m **原始 user_id**(字符串),value 是重映射后的**连续 ID**(`0..6039`).

- **作用**:**双向翻译**.原始 ID 稀疏不连续,不能当数组下标,所以统一重映射成 `0..N-1`.
  写报告做案例分析时,要把模型输出的 `user 0` 翻回"ml-1m 用户 1"才说得清;
  拿到外部数据想并入时,也靠这张表对齐.

### `item_id_map.json`

- **内容**:物品 ID 重映射表,JSON 对象,**3,416 个键**(5-core 过滤后的留存物品).

  ```json
  {"1": 0, "2": 1, "3": 2, ...}
  ```

  key 是 ml-1m **原始 movie_id**,value 是重映射后的连续 ID(`0..3415`).

- **作用**:**项目里所有文本,向量,语义 ID 的编号基准**.
  `movies.dat` 用的是原始 ID,训练数据用的是新 ID,两者对不上 ----
  这张表就是那个"翻译字典",`item_text.json` 就是靠它生成的.
  **重建语义标识符(改 L 或 K)时必须用它保证编号一致.**

### `item_text.json`

- **内容**:物品文本表,JSON 对象,**3,416 个键**.

  ```json
  {
    "0": "Toy Story (1995) | Animation|Children's|Comedy",
    "1": "Jumanji (1995) | Adventure|Children's|Fantasy",
    "2": "Grumpier Old Men (1995) | Comedy|Romance"
  }
  ```

  key 是**重映射后**的 item_id,value 是 `"标题 (年份) | 类别串"`(用 ` | ` 把标题和类别拼起来).

- **作用**:**语义链的起点,全项目唯一出现原始文本的地方**.
  第 3 周构造语义标识符时,这份文本会被喂给预训练文本编码器(如 Sentence-T5)→ 编码成
  768 维向量存成 `item_emb.npy` → 再进 RQ-VAE 量化成 3 个离散码.

  **它只被使用一次**:编码器跑完,向量存盘之后,`item_text.json` 就退场了 ----
  之后 RQ-VAE 读向量矩阵,生成式模型读语义 ID 表,都不再回看原始文本.
  写报告,做案例分析,把 ID 显示成片名时才再需要它.

### `preprocess_log.json`

- **内容**:预处理运行记录,JSON 对象.

  ```json
  {
    "dataset": "ml-1m",
    "k_core": 5,
    "seed": 42,
    "raw":         {"users": 6040, "items": 3706, "interactions": 1000209},
    "after_kcore": {"users": 6040, "items": 3416, "interactions":  999611},
    "split":       {"train": 987531, "valid": 6040, "test": 6040}
  }
  ```

- **作用**:**实验可复现的凭证**.记录了本次运行的参数(`k_core` / `seed`)和每一步的规模变化,
  任何人(包括三个月后的你自己)对着它就能确认"这份数据是用什么口径生成的".
  写报告时数据规模那一栏直接从这抄,不要凭印象.

  > 注意 `raw.items` 是 **3706** 而不是 3883 ---- 这是 `ratings.dat` 里实际出现过的电影数;
  > 3883 是 `movies.dat` 的总条数,其中 177 部**从未被评分**,预处理第一步就丢掉了.

---

## 3. 各文件谁读谁写

| 文件 | 谁写 | 谁读 |
|---|---|---|
| `raw/*.dat` | 你(手动下载解压) | `preprocess.py` |
| `ml-1m.inter` | `preprocess.py` | **RecBole**(第 2 周判别式基线) |
| `train/valid/test.jsonl` | `preprocess.py` | **自研生成式模型**(第 5–6 周) |
| `user_id_map.json` | `preprocess.py` | 报告案例分析,外部数据对齐 |
| `item_id_map.json` | `preprocess.py` | 文本编码,语义 ID 构建(第 3–4 周) |
| `item_text.json` | `preprocess.py` | **文本编码器**(第 3 周,一次性)→ 产出 `item_emb.npy` |
| `preprocess_log.json` | `preprocess.py` | 报告"数据规模"栏,复现自查 |
| `verify_split.py` | -- | 读全部 jsonl + `.inter` 做一致性校验 |

---
