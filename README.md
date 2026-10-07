# joinplan

无数据库依赖的 JSON 连接顺序规划器（纯 Python 标准库）。给定 2～9 张表、
表间谓词及其有理选择率，枚举所有合法二叉连接树，返回总代价最小的树。

## 模型

- 允许任意二叉连接树，但每次合并的两侧之间必须至少有一条谓词。
- 子树估计行数 = 子树内所有基表行数之积 × 子树内部全部谓词选择率之积
  （精确有理数运算）。
- 总代价 = 所有非叶节点估计行数之和。
- 并列时：每个内部节点的左右子树按树串字节序排列，取全括号树串字典序
  最小者。
- 谓词图不连通时，返回各连通分量（各自给出最优子计划）。

## 输入（stdin 或文件参数）

```json
{
  "tables": [{"name": "A", "rows": 100}, {"name": "B", "rows": 200}],
  "predicates": [{"left": "A", "right": "B", "selectivity": "1/10"}]
}
```

- `tables`：2～9 张，表名为唯一可打印 ASCII（不含括号），`rows` 为正整数。
- `predicates`：可省略；`left`/`right` 必须是不同的已声明表；同一表对
  可有多条谓词。`selectivity` 为 `[0,1]` 内的有理数，写作 `"p/q"`
  （也接受整数 `0`/`1`；未约分的分数会自动约分）。

## 第二情形选择率（可选）

上线前若已掌握同一批谓词在第二种数据分布下的选择率，可提供与
`predicates` 按下标一一对应的 `selectivities2` 数组。启用后表数限制收紧
为 2～6，连接合法性仍只由原谓词图决定：

```json
{
  "tables": [{"name": "A", "rows": 100}, {"name": "B", "rows": 100}],
  "predicates": [{"left": "A", "right": "B", "selectivity": "1/1000"}],
  "selectivities2": ["1/5"]
}
```

- 每棵树用精确有理数分别计算两种情形的节点估计行数与总代价。
- 目标依次为：最小化 `max(cost1, cost2)`（任一情形都不会代价过高）→
  最小化 `cost1 + cost2` → 取全括号树串字典序最小者。
- 子集 DP 为每个表集合保留**全部非支配 `(cost1, cost2)` 对**（及见证
  树串），而不是只留单一情形下最便宜的一棵子树——折中树可能在任一
  单情形下都不是最优（见 `examples/dual4.json`）。
- 输出在每个 `cost` 旁新增 `cost2`，每个连接节点新增 `rows2`，每个
  谓词项新增 `selectivity2`；省略 `selectivities2` 时输入输出与单情形
  完全逐项一致（不会出现多余键）。

## 物化连接结果（可选：直接扫缓存，还是按谓词图重算）

报表工程师若已持有某组表的物化连接结果，可再提供一个 `materialized`
对象，描述这张"缓存叶节点"：

```json
{
  "tables": [
    {"name": "A", "rows": 100}, {"name": "B", "rows": 200},
    {"name": "C", "rows": 50}
  ],
  "predicates": [
    {"left": "A", "right": "B", "selectivity": "1/10"},
    {"left": "B", "right": "C", "selectivity": "1/4"}
  ],
  "selectivities2": ["1/5", "1"],
  "materialized": {
    "tables": ["A", "B"],
    "rows": 2000,
    "read_cost": "100",
    "rows2": 4000,
    "read_cost2": "100"
  }
}
```

- `tables`：原表的**非空真子集**，至少两张、至多 n−1 张（不允许重复、
  不允许引用未声明表），且其诱导子图必须能按原谓词图合法连接
  （诱导图连通）。启用 `materialized` 后总表数收紧为 **3～6**。
- `rows`：已观测行数，非负整数（允许 0）；`read_cost`：读取这份缓存
  的代价，非负有理数（允许 0，写作 `"p/q"` 或整数）。
- 双情形（提供 `selectivities2`）时还必须提供与第一种情形并列的
  `rows2` 与 `read_cost2`，缺字段即整次拒绝；单情形下出现这两个键
  同样拒绝。任何字段非法（负代价、零分母、子集不连通等）都会在规划
  开始前一次性报错，不会产生部分计划。

规划时对包含该子集的连通分量同时求两份计划：

1. **重算（recompute）**：普通规划，忽略物化描述；
2. **缓存（cached）**：把这组表收缩成**一个不可拆叶节点**——叶节点行数
   直接采用观测值（而不是基表行数之积），读取代价计入总代价且只计一次；
   子集内部谓词视为已生效，**不再重复乘选择率**，也不会出现在缓存树里；
   跨子集谓词仍在两侧首次合并时照常生效（允许同一切口上有多条）。

两份计划沿用现有规则比较：单情形比最低总代价；双情形先比
`max(cost1, cost2)` 再比 `cost1 + cost2`。**完全并列时优先重算。**
不连通图中其他分量两份计划相同，比较的是全图代价之和。

输出在原文档末尾新增 `materialized` 块：

- `chosen`：`"cached"` 或 `"recompute"`；
- `materialized_leaf`：被选中时嵌入树中的同一份物化叶节点
  `{"type": "materialized", "tables": [...排序...], "rows": <int>,
  "read_cost": "p/q"}`（双情形另有 `rows2`/`read_cost2`）；
- `covered_predicates`：被缓存覆盖（视为已生效）的谓词，按输入顺序；
- `recompute_cost` / `cached_cost`（双情形另有 `*_cost2`）：逐情形
  成本。顶部的 `cost`/`tree` 始终对应被选中的方案。

省略 `materialized` 时输出与此前**逐项一致**，不会出现上述任何键。

```sh
python3 joinplan.py examples/materialized3.json
```

## 输出

连通：

```json
{"status": "ok", "cost": "27000/1", "tree_string": "((AB)C)", "tree": {...}}
```

不连通：

```json
{"status": "disconnected", "components": [{"tables": [...], "cost": "p/q",
  "tree_string": "...", "tree": {...}}, ...]}
```

输入非法时输出 `{"status": "error", "error": "..."}` 并以退出码 2 结束。

树节点：叶子为 `{"type": "table", "name": ..., "rows": <int>}`；连接节点为
`{"type": "join", "rows": "p/q", "tables": [...], "predicates": [...],
"children": [left, right]}`，其中 `predicates` 是在该次合并首次生效的谓词
（按输入顺序），`children` 按子树串字节序排列。所有有理数（`cost`、连接
节点 `rows`）以约分后的 `"p/q"` 字符串表示。双情形模式下还会出现 `cost2`、
`rows2` 与谓词项内的 `selectivity2`。启用物化结果时，被选中的缓存树在子集
位置上是 `{"type": "materialized", "tables": [...], "rows": <int>,
"read_cost": "p/q"}` 叶节点（双情形另有 `rows2`/`read_cost2`）。

## 运行

本地：

```sh
python3 joinplan.py < examples/chain3.json
python3 joinplan.py examples/bushy4.json
python3 joinplan.py examples/dual4.json      # 双情形：min max 折中树
```

Compose（`joinplan` 服务）：

```sh
docker compose build joinplan
docker compose run -T joinplan < examples/chain3.json
```

## 测试

pytest 对小图枚举所有合法二叉树（不做子集剪枝），与规划器对拍总代价和
并列裁决，并逐节点校验估计行数、首次生效谓词、子节点顺序与合并合法性；
双情形模式同样对拍 `(max, sum, 树串)` 目标顺序（含各层并列），并校验
`rows2`/`selectivity2`/`cost2`。物化结果另有一套对拍：把子集独立收缩成
一个带观测行数与读取代价的不可拆叶节点，穷举"重算或读取缓存"的全部
合法树，逐节点校验组内谓词不重复乘、跨组谓词照常首次生效、物化叶节点
与逐情形成本一致，覆盖零行、零代价、跨组多谓词与并列裁决：

```sh
python3 -m pytest -q                      # 本地
docker compose run --rm joinplan-tests    # 或经 Compose
```
