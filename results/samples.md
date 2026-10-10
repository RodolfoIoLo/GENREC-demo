# 服务器产出 · 头部样本

数据文件不适合进 Git（体积大、可从原始数据集重新生成），但**它们长什么样**应该看得见。
本文件逐个展示真实内容的前几行。字段语义和数量说明见 [`../data/README.md`](../data/README.md)。

> 下面所有内容都是从服务器上 `head` 出来的**真实输出**，不是编的。

---

## 1. `/data/zby/data/ml-1m/raw/`

### `ratings.dat` — 24 MB / 1,000,209 行

```
1::1193::5::978300760
1::661::3::978302109
```

`UserID::MovieID::Rating::Timestamp`，`::` 分隔，原始 id 从 1 开始。

### `movies.dat` — 168 KB / 3,883 行

```
1::Toy Story (1995)::Animation|Children's|Comedy
2::Jumanji (1995)::Adventure|Children's|Fantasy
```

`MovieID::Title::Genres`。

### `users.dat` — 132 KB / 6,040 行

```
1::F::1::10::48067
2::M::56::16::70072
```

`UserID::Gender::Age::Occupation::Zip-code`。（这三个字段我们不用，只取 id。）

---

## 2. `/data/zby/data/ml-1m/processed/`

### `ml-1m.inter` — 21 MB / 999,612 行（1 行表头 + 999,611 条交互）

```
userID:token	itemID:token	timestamp:float
0	2757	978300019.0
0	861	978300055.0
0	1068	978300055.0
0	1444	978300055.0
```

- 表头声明字段类型，之后是 TAB 分隔的数据行
- `userID` / `itemID` 是**重映射后**的 0 基 id（原始 id 见 `*_id_map.json`）
- **已按 `(user, timestamp, item)` 排序**，RecBole 的 leave-one-out 切分依赖这个顺序
- 这个文件**包含全部 999,611 条交互**，train/valid/test 是 RecBole 在内存里切出来的

### `train.jsonl` — 5.5 MB / 6,040 行（每用户 1 行）

```json
{"user": 0, "items": [2757, 861, 1068, 1444, 1978, 1519, 2925, 1007, 2411, 241, 633, 994, 767, 561, 2302, 1630, 1697, 2685, 786, 874, 1631, 134, 867, 1687, 926, 762, 1085, 2404, 1044, 2369, 595, 2513, 496, 2694, 2398, 868, 1959, 550, 997, 2034, 1301, 493, 0, 545, 551, 648, 2297, 673, 1936, 1992, 1314]}
```

user 0 的 train 序列 = **51 个物品**（他总共有 53 条交互：53 − 2 = 51）。

### `valid.jsonl` — 5.6 MB / 6,040 行

```json
{"user": 0, "history": [2757, 861, 1068, 1444, 1978, 1519, 2925, 1007, 2411, 241, 633, 994, 767, 561, 2302, 1630, 1697, 2685, 786, 874, 1631, 134, 867, 1687, 926, 762, 1085, 2404, 1044, 2369, 595, 2513, 496, 2694, 2398, 868, 1959, 550, 997, 2034, 1301, 493, 0, 545, 551, 648, 2297, 673, 1936, 1992, 1314], "target": 1578}
```

`history` 与 train 完全相同（51 个），`target` = **1578 → Mulan (1998)**（倒数第 2 条交互）。

### `test.jsonl` — 5.7 MB / 6,040 行

```json
{"user": 0, "history": [2757, 861, 1068, 1444, 1978, 1519, 2925, 1007, 2411, 241, 633, 994, 767, 561, 2302, 1630, 1697, 2685, 786, 874, 1631, 134, 867, 1687, 926, 762, 1085, 2404, 1044, 2369, 595, 2513, 496, 2694, 2398, 868, 1959, 550, 997, 2034, 1301, 493, 0, 545, 551, 648, 2297, 673, 1936, 1992, 1314, 1578], "target": 46}
```

`history` = train 的 51 个 **+ valid 的 target 1578** = **52 个**，`target` = **46 → Pocahontas (1995)**。

> 注意这三行：train ⊂ valid.history ⊂ test.history，每一层只多一个物品。
> 这就是「样本高度重叠」的实际样子 —— 同一个物品会被反复当作历史输入。

### `item_text.json` — 176 KB / 3,416 个键

```json
{
  "0": "Toy Story (1995) | Animation|Children's|Comedy",
  "1": "Jumanji (1995) | Adventure|Children's|Fantasy",
  "2": "Grumpier Old Men (1995) | Comedy|Romance",
  "3": "Waiting to Exhale (1995) | Comedy|Drama",
  "4": "Father of the Bride Part II (1995) | Comedy",
  "5": "Heat (1995) | Action|Crime|Thriller",
  ...
}
```

键是**重映射后**的物品 id，值是 `标题 (年份) | 类型1|类型2`。
这是第 3 周构建语义标识符时的**文本输入**（喂给预训练文本编码器）。

### `user_id_map.json` / `item_id_map.json` — 96 KB / 52 KB

```json
{
  "1": 0,
  "2": 1,
  "3": 2,
  ...
}
```

`原始 id（字符串） → 重映射 id`。两个文件都从 1 开始映射到 0。

### `preprocess_log.json` — 299 B（全文，已入库）

```json
{
  "dataset": "ml-1m",
  "k_core": 5,
  "seed": 42,
  "raw":         { "users": 6040, "items": 3706, "interactions": 1000209 },
  "after_kcore": { "users": 6040, "items": 3416, "interactions": 999611 },
  "split":       { "train": 987531, "valid": 6040, "test": 6040 }
}
```

---

## 3. `~/genrec/` 里没入库的产出

### `saved/SASRec-Sep-28-2026_19-18-04.pth` — 3.9 MB

二进制 PyTorch checkpoint，存的是 **best epoch 31** 的模型。`torch.load` 出来的结构：

```
top-level keys : ['config', 'epoch', 'cur_step', 'best_valid_score',
                  'state_dict', 'other_parameter', 'optimizer']
epoch          : 31
state_dict     : 36 个条目
                 item_embedding.weight                     (3417, 64)
                 position_embedding.weight                 (200, 64)
                 trm_encoder.layer.0.multi_head_attention.query.weight (64, 64)
                 trm_encoder.layer.0.multi_head_attention.query.bias   (64,)
                 ...
```

**没入库的原因**：① 二进制，Git diff 无意义；② 3.9 MB 里约 2/3 是 Adam 的优化器状态
（`optimizer` 的 moment 是参数量的 2 倍，331,584 × 2 × 4 B ≈ 2.65 MB），对读者没有价值；
③ 训练脚本 + 配置都在，重跑 80 分钟就能再得到一份。

> 想拿它做第 3 周的超参对照实验时，用 `scripts/eval_only.py --model_file <路径>` 直接加载评测。

### `results/sasrec_0928_1916.log` — 5 MB

这是 `python ... | tee` 出来的原始日志。**它被 tqdm 的 `\r` 刷成了「上万行」**，
不是真的有一万多个 epoch。入库的是 RecBole 自己写的干净日志（`log/SASRec/…`，339 行、20 KB）。

### `log_tensorboard/`

RecBole 默认会写 TensorBoard 事件文件，当前实验没启用（`log_wandb: False`、未跑 `get_tensorboard`），目录为空。
