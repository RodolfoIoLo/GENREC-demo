# 序列推荐基线

> MovieLens 1M，3,416 个物品全量排序；Recall@5/10 与 NDCG@5/10。  
> SASRec 汇总使用总体标准差 `numpy.std(ddof=0)`。

| 模型 | Seed | 总 epoch | Best epoch | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SASRec | 42 | 43 | 31 | 0.1409 | 0.2204 | 0.0884 | 0.1141 |
| SASRec | 43 | 109 | 97 | 0.1523 | 0.2339 | 0.0968 | 0.1230 |
| SASRec | 44 | 59 | 47 | 0.1526 | 0.2250 | 0.0958 | 0.1191 |
| **SASRec** | **均值 ± std** | — | — | **0.1486 ± 0.0054** | **0.2264 ± 0.0056** | **0.0937 ± 0.0037** | **0.1187 ± 0.0036** |
| **GRU4Rec** | **42** | **67** | **55** | **0.1861** | **0.2657** | **0.1261** | **0.1517** |

## 固定协议

- RecBole 1.2.1，隐式反馈，CE loss。
- 每个用户按时间 leave-one-out：最后一条 test，倒数第二条 valid。
- 最大序列长度 200；valid NDCG@10 早停，`stopping_step=10`。
- 评测时屏蔽训练期已见物品，候选集为全部 3,416 个真实物品。

## 证据文件

| 文件 | 内容 |
|---|---|
| `sasrec_s42.log` | seed 42 训练；best epoch 31 |
| `sasrec_s42_patch_test.log` | seed 42 checkpoint 的独立 valid / test 复评 |
| `sasrec_s43.log` | seed 43 训练与测试；best epoch 97 |
| `sasrec_s44.log` | seed 44 训练与测试；best epoch 47 |
| `gru4rec_s42.log` | GRU4Rec seed 42 训练与测试；best epoch 55 |
