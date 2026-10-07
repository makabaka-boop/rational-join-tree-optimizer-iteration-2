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
`rows2` 与谓词项内的 `selectivity2`。

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
`rows2`/`selectivity2`/`cost2`：

```sh
python3 -m pytest -q                      # 本地
docker compose run --rm joinplan-tests    # 或经 Compose
```
