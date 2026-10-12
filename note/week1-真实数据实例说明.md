# week1-任务 2–3(数据预处理) 真实数据实例说明

> 全部数字和片名来自**真实 ml-1m 实测**（2026-09-17 跑出），不是编的。
> 数据来源：`https://files.grouplens.org/datasets/movielens/ml-1m.zip`
> 配套：《第一周_任务2-3详细任务书.md》（操作手册）· `scripts/preprocess.py` · `scripts/stats.py`

---

## 一句话

| | 任务 2 | 任务 3 |
|---|---|---|
| 比喻 | **备菜** —— 把流水账切成「接龙题」 | **体检** —— 看清这堆数据长什么样 |
| 输出 | train / valid / test + item_text | 统计表 + 长尾图 + 抽样序列 |

下面全部用**真实电影**讲。

---

## 一、任务 2 做了什么 —— 拿用户 0 走一遍

### 1.1 原始文件里是这样的

`ratings.dat` 的原始行（真实格式）：

```
1::1193::5::978300760
1::661::3::978302109
1::914::3::978301968
```

读作「用户 1 在时间 978300760 给电影 1193 打了 5 分」。

**问题**：这是**流水账**，不是**序列**。文件按用户 ID 排，不是按时间排；电影 ID 是原始编号（1~3952 中间有空号）；还混着大量「只被看过一两次」的噪声。

---

### 1.2 5-core：从 3883 部砍到 3416 部

真实数字（实测）：

```
movies.dat 里的电影总数      : 3883
ratings.dat 里出现过的电影数 : 3706      ← 少了 177 部，因为从来没人评分
被评次数 < 5 的电影数        : 290
5-core 后剩余电影数          : 3416      ← 3706 - 290
```

**被砍掉的电影长什么样**（真实例子）：

| 被评次数 | 电影 | 类型 |
|---|---|---|
| 4 次 | `Regret to Inform (1998)` | Documentary |
| 4 次 | `Brother Minister: The Assassination of Malcolm X (1994)` | Documentary |
| 4 次 | `Horror Hotel (a.k.a. The City of the Dead) (1960)` | Horror |
| 4 次 | `Ogre, The (Der Unhold) (1996)` | Drama |

**为什么必须砍**：一部只被 4 个人看过的纪录片，模型从它身上学不到任何规律，反而会在评测时制造噪声。

**用户这边呢？一个都没砍。** ml-1m 保证每个用户至少评过 20 部电影，5-core 门槛是 5，所以 6040 个用户全部保留。

> **注意这个细节**：`3883 → 3706` 是「从来没人评分」，`3706 → 3416` 才是「被 5-core 砍掉」。这两个数是不同的原因，报告里别写混。

---

### 1.3 ID 重映射：Toy Story 从 1 变成 0

原始电影 ID 最大到 **3952**，但只有 **3883** 部，中间有 69 个空号。

重映射后，所有 ID 压成连续的 `0..3415`。真实对应关系：

| 新 ID | 原始 ID | 电影 |
|---|---|---|
| **0** | 1 | `Toy Story (1995)` |
| 1992 | 2018 | `Bug's Life, A (1998)` |
| 1936 | 1962 | `Antz (1998)` |
| 1578 | 1608 | `Mulan (1998)` |
| **46** | 47 | `Pocahontas (1995)` |

**为什么要做**：如果直接拿原始 ID 当索引，embedding 表要开 3953 行，其中 70 行永远用不到 —— 白占显存。而且第 3 周建语义码本、第 5 周算 loss 时，索引对不上会直接崩。

**为什么必须存映射表**：模型输出的永远是「新 ID」。要把 `1578` 翻译回「Mulan (1998)」，只能查 `item_id_map.json`。**这张表丢了，预测结果就永远翻译不回来。**

---

### 1.4 用户 0 的 53 部电影，被切成三份 ⭐ 核心

这是任务 2 真正的样子。用户 0 在真实数据里有 53 条记录，分成两批：

**第一批：2000-12-31，40 部**（一次批量导入，一天之内）

```
Girl, Interrupted (1999)          Back to the Future (1985)       Titanic (1997)
Cinderella (1950)                 Meet Joe Black (1998)           Last Days of Disco, The (1998)
Erin Brockovich (2000)            Christmas Story, A (1983)       To Kill a Mockingbird (1962)
One Flew Over the Cuckoo's Nest   Wallace & Gromit (1996)         Star Wars: Episode IV (1977)
Wizard of Oz, The (1939)          Fargo (1996)                    Run Lola Run (1998)
Rain Man (1988)                   Saving Private Ryan (1998)      Awakenings (1990)
Gigi (1958)                       Sound of Music, The (1965)      Driving Miss Daisy (1989)
Bambi (1942)                      Apollo 13 (1995)                Mary Poppins (1964)
E.T. the Extra-Terrestrial (1982) My Fair Lady (1964)             Ben-Hur (1959)
Big (1988)                        Sixth Sense, The (1999)         Dead Poets Society (1989)
James and the Giant Peach (1996)  Ferris Bueller's Day Off (1986) Secret Garden, The (1993)
Toy Story 2 (1999)                Airplane! (1980)                Pleasantville (1998)
Dumbo (1941)                      Princess Bride, The (1987)      Snow White (1937)
Miracle on 34th Street (1947)
```

**第二批：2001-01-06，13 部**（一周后，这才是关键）

| 拆分 | 电影 | 类型标签（真实） |
|---|---|---|
| train | `Ponette (1996)` | Drama |
| train | `Schindler's List (1993)` | Drama\|War |
| train | `Beauty and the Beast (1991)` | Animation\|Children's\|Musical |
| train | `Tarzan (1999)` | Animation\|Children's |
| train | `Close Shave, A (1995)` | Animation\|Comedy |
| train | `Aladdin (1992)` | Animation\|Children's\|Comedy\|Musical |
| train | `Toy Story (1995)` | Animation\|Children's\|Comedy |
| train | `Bug's Life, A (1998)` | Animation\|Children's\|Comedy |
| train | `Antz (1998)` | Animation\|Children's |
| train | `Hunchback of Notre Dame, The (1996)` | Animation\|Children's\|Musical |
| train | `Hercules (1997)` | Adventure\|Animation\|Children's\|Comedy\|Musical |
| **valid** | **`Mulan (1998)`** | Animation\|Children's |
| **test** | **`Pocahontas (1995)`** | Animation\|Children's\|Musical\|Romance |

**看到规律了吗** —— 这个人一周内连着看了 **11 部动画片**，然后：

- 验证集要他预测：下一部是什么？→ 答案是 **Mulan**
- 测试集要他预测：再下一部？→ 答案是 **Pocahontas**

两部都是迪士尼动画。**这就是「序列推荐」的全部内容**：给他一串历史，猜下一个。

---

### 1.5 最终产出

```json
// train.jsonl —— 51 部，训练用
{"user": 0, "items": [2757, 861, 1068, 1444, 1978, ..., 1992, 1314]}

// valid.jsonl —— 用前 51 部预测第 52 部
{"user": 0, "history": [2757, 861, ..., 1992, 1314], "target": 1578}
//                                                              ↑ Mulan

// test.jsonl —— 用前 52 部预测第 53 部
{"user": 0, "history": [2757, 861, ..., 1992, 1314, 1578], "target": 46}
//                                                                   ↑ Pocahontas
```

**任务 2 干的事，就是把原始流水账变成这三个文件。** 模型的工作就是「接龙」。

---

## 二、任务 3 做了什么 —— 真实统计

### 2.1 统计表（真实数字）

| 指标 | 值 |
|---|---|
| 用户数 | **6040** |
| 物品数 | **3416** |
| 交互总数 | **999,611** |
| 稀疏度 | **0.9516** |
| 每用户平均序列长度 | **165.5** |
| 中位序列长度 | **96** |
| P90 序列长度 | **400** |
| 最长 / 最短序列 | **2277 / 18** |
| 最热物品被看次数 | **3428** |
| 物品被看次数中位数 | **146** |
| 基尼系数 | **0.6034** |
| 时间跨度 | 2000-04-25 ~ 2003-02-28 |

**稀疏度 0.9516 是什么意思**：`1 - 999611 / (6040 × 3416) = 0.9516`。也就是说，用户-物品矩阵里 **95% 的格子是空的**。这就是推荐任务难的根本原因。

---

### 2.2 长尾：真实的最热和最冷

**最热的 5 部**：

| 被看次数 | 电影 |
|---|---|
| 3428 | `American Beauty (1999)` |
| 2991 | `Star Wars: Episode IV - A New Hope (1977)` |
| 2990 | `Star Wars: Episode V - The Empire Strikes Back (1980)` |
| 2883 | `Star Wars: Episode VI - Return of the Jedi (1983)` |
| 2672 | `Jurassic Park (1993)` |

**最冷的 5 部**（5-core 之后仍然留下的）：

| 被看次数 | 电影 | 类型 |
|---|---|---|
| 5 | `Machine, The (1994)` | — |
| 5 | `Delta of Venus (1994)` | Drama |
| 5 | `Boys Life 2 (1997)` | — |
| 5 | `Return with Honor (1998)` | Documentary |
| 5 | `Two or Three Things I Know About Her (1966)` | Drama |

**同一个类型内部的差距更直观**：

| 类型 | 最热 | 最冷 | 差距 |
|---|---|---|---|
| Horror | `Ghostbusters (1984)` — **2181 次** | `City of the Living Dead (1980)` — **5 次** | **436 倍** |
| Documentary | `Roger & Me (1989)` — **798 次** | `Return with Honor (1998)` — **5 次** | **160 倍** |

---

### 2.3 ⚠️ 一个必须诚实面对的发现

**5-core 过滤本身就把最冷的那批物品砍掉了。**

所以剩下的长尾看起来**没有想象中极端**：

- 交互数 ≤ 10 的物品只占 **5.4%**（因为 < 5 的已经被砍了）
- 基尼系数 **0.6034** —— 属于中等偏上，不算极端
- 最热 1% 的物品吃掉 **8.2%** 的交互，最热 10% 吃掉 **42.3%**

**这意味着什么**：写报告时**不要夸大**说「长尾极其严重」。但也别因此觉得课题没意义 —— 真正的论证点应该换成：

> **即使已经过滤到 5-core，仍有 26.8% 的物品只有 50 次以内交互**（3416 部里有 917 部）。这些物品的语义信息（片名、类型）完全没有被利用 —— 而它们的类型标签明明告诉了我们它们是「恐怖片」「纪录片」。

这个说法比「长尾严重」更准确，也更站得住脚。

---

### 2.4 抽样看序列发现了什么

**发现一：语义连贯性成立** ✅

用户 0 那 11 部连续的动画片就是证据。这不是巧合 —— 大量用户都有类似模式（看完一部迪士尼，接着看下一部）。**说明「语义标识符能传递知识」这个前提是成立的。**

**发现二：数据有批量导入痕迹** ⚠️

用户 0 在 **2000-12-31 一天之内**录入 40 部电影。真实用户不会一天看完 40 部。这说明 ml-1m 的早期数据是**批量补录**的，时间戳不代表真实观影时间。

**影响**：leave-one-out 按时间划分时，最后两条（Mulan、Pocahontas）恰好都在 2001-01-06 那批里，时间上是合理的。但如果某用户最后一批是批量补录的，划分结果可能不那么可靠。**这是个已知的数据局限，报告里可以提一句。**

---

## 三、后面要做什么 —— 还是这批电影

### 任务 4：模型到底在预测什么

回到用户 0。模型的输入输出是：

```
输入（51 部）：... Schindler's List, Beauty and the Beast, Tarzan, Aladdin,
               Toy Story, Bug's Life, Antz, Hunchback, Hercules
                    ↓
              模型（黑盒）
                    ↓
输出：从 3416 部电影里，挑出最可能的下一部 → Mulan (1998)
```

**本质是一个 3416 类的分类问题**。模型给 3416 部电影各打一个分，取最高的。

### 任务 5：预测对了怎么算分

模型不会只给一个答案，它给一个**排序**。假设给用户 0 的 Top-3 是：

| 情况 | Top-3 排序 | Recall@3 | NDCG@3 |
|---|---|---|---|
| 理想 | `Mulan, Pocahontas, Tarzan` | 1 ✅ | **1.000**（排第一，不打折） |
| 次优 | `Tarzan, Mulan, Pocahontas` | 1 ✅ | **0.631**（排第二，打 1/log₂3 折） |
| 失败 | `Tarzan, Hercules, Antz` | 0 ❌ | **0.000** |

**两种情况下 Recall 都是 1，但 NDCG 差很多** —— 这就是为什么推荐系统更看重 NDCG：**命中还不够，得排在前面**。（这些数字是用脚本算过的，不是估的。）

### 任务 6：SASRec 和 TIGER 的差别

拿用户 0 举例，两种范式的做法完全不同：

| | **SASRec**（传统） | **TIGER**（生成式，你的课题方向） |
|---|---|---|
| 每部电影是什么 | 一个随机编号，`Mulan = 1578` | 一串语义 token，`Mulan = [12, 47, 61]` |
| 预测方式 | 从 3416 个编号里**选一个**（分类） | **生成**这三个 token（造句） |
| 冷门电影怎么办 | `1578` 只被训练过 490 次，还行；但 `Return with Honor` 只被训练过 **5 次**，embedding 基本是随机的 | `Return with Honor` 是 `Documentary`，和热门纪录片 `Roger & Me`（798 次）**共享语义前缀** → 能借力 |

**最后那一行就是你这个课题的立项理由。**

### 阶段二（第 3–4 周）：构建语义标识符

用任务 2 导出的 `item_text.json`（真实内容）：

```
[0]    Toy Story (1995)     | Animation|Children's|Comedy
[1992] Bug's Life, A (1998) | Animation|Children's|Comedy
[1936] Antz (1998)          | Animation|Children's
[1578] Mulan (1998)         | Animation|Children's
[46]   Pocahontas (1995)    | Animation|Children's|Musical|Romance
```

**流程**：文本 → 预训练编码器转成向量 → RQ-VAE 量化成分层码。

**产出的语义 ID 大概长这样**（⚠️ 以下是**示意**，我还没跑 RQ-VAE，数字是构造的）：

```
Toy Story (1995)   → [12, 45, 201]
Bug's Life, A (1998) → [12, 45, 188]     ┐
Antz (1998)        → [12, 45, 233]       ┘ 共享前缀 [12, 45] = 动画/儿童/喜剧
Mulan (1998)       → [12, 47, 61]        ┐
Pocahontas (1995)  → [12, 47, 102]       ┘ 共享前缀 [12, 47] = 动画/音乐剧
```

**关键**：语义相近的电影**共享前缀**。模型学会 `[12, 45, ...]` 这一类之后，即使某部动画片自己没被看过几次，也能被推出来。

**这就是「把物品变成词」的含义** —— 电影不再是孤立的编号，而是「有语义归属的词」。

### 阶段三（第 5–6 周）：生成式模型

```
输入：用户 0 的历史 → 转成 token 序列 [3, 42, 17, ..., 12, 45, 233]
                                    ↓
                           T5 编码器-解码器
                                    ↓
输出：生成 [12, 47, 61]  →  查表翻译回 "Mulan (1998)"
```

**再加约束解码**：不是随便生成，而是用 Trie 前缀树保证生成的 token 组合**一定是真实存在的电影**（不能生成 `[12, 47, 999]` 这种不存在的组合）。

---

## 四、总结

| 阶段 | 用真实数据说 | 产出 |
|---|---|---|
| **任务 2** | 用户 0 的 53 部电影 → train 51 / valid 1 / test 1 | `train.jsonl` 等 7 个文件 |
| **任务 3** | 6040 用户 / 3416 电影 / 999611 交互；最热 3428 次 vs 最冷 5 次 | 统计表 + 长尾图 + 抽样序列 |
| **任务 4–6** | 模型要预测 `Mulan`；排第一得 1.0 分，排第二得 0.63 分 | 理解 + 对比笔记 |
| **阶段二** | `Toy Story` → `[12, 45, 201]`（示意） | 语义标识符 |
| **阶段三** | 生成 token → 翻译回 `Mulan (1998)` | 生成式模型 |

**一句话**：任务 2 把 53 部电影切成「用前 51 部猜第 52 部」的题目，任务 3 告诉你这个数据集长什么样，后面所有工作都是为了让模型把 `Mulan` 和 `Pocahontas` 猜对 —— 而且是**排在第一位猜对**。
