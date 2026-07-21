# 日频因子模板

> **布局迁移（2026-07）**：手写 python 因子与挖掘因子统一为「每因子三件套」，
> 落在 `workspace/factor_store/<market>/<name>/`：
>
> ```
> workspace/factor_store/ashare/<name>/
> ├── meta.json      # 元信息 + ledger_snapshot（裁决真相仍是 factor_library jsonl）
> ├── factor.py      # 可执行因子代码（DailyFactor 类 或 expression 生成包装）
> └── factor.parquet # 物化面板（仅 active/probation）
> ```
>
> 旧路径 `workspace/factors/daily/{factor_name}.py` **仍兼容扫描**，但会打
> `DeprecationWarning`；请把新因子写到 factor_store，旧文件由迁移流程收尾删除。
>
> 裁决真相不变：`workspace/factor_library/<market>.jsonl`（status/lift/admission）。
> 资产库是载体；用 `fz factor-library store sync` / `verify` 维护与校验。
>
> **物化口径（parquet）** 与裁决评估窗分离：store sync 统一按
> `all_a` × `2016-01-01` ~ 最新已完结交易日写 `factor.parquet` /
> `meta.materialization`；jsonl 的 `eval_start`/`eval_end`/`universe`
> （多为 csi300 评估窗）保持不动。

## 编写约定（python 类）

把下面代码保存为 `workspace/factor_store/ashare/<name>/factor.py`，并配一份 `meta.json`
（`kind: "python"`）。复制后改 `name`、类名、`description`、`lookback_days` 和公式。

- 继承 `DailyFactor`。
- `category = "daily"`，`frequency` 可以省略或显式设为 `"daily"`。
- `required_data` 按实际使用声明，常见值是 `["daily"]` 或 `["daily", "daily_basic"]`。
- `compute(ctx)` 返回 Polars `DataFrame`，至少包含 `trade_date`、`ts_code`、`factor_value`。
- 有复权价格需求时优先使用 `close_adj`、`open_adj`、`high_adj`、`low_adj`。
- 所有 `shift`、`rolling_*` 必须 `.over("ts_code")`，避免跨股票串号。
- 用 `ctx.start` 过滤预热期，只输出请求区间。
- 行业、市值中性化放在 YAML 的 `preprocessing` 配置里，不要写进因子本身。

## meta.json 最小字段

```json
{
  "name": "my_daily_alpha",
  "kind": "python",
  "expression": "py::my_daily_alpha",
  "frequency": "daily",
  "description": "20 日复权动量",
  "source_run_id": null,
  "created_at": "2026-07-21",
  "ledger_snapshot": {
    "status": null,
    "lift": null,
    "admission_ic": null,
    "ic_train": null,
    "holdout_ic": null,
    "truth": "workspace/factor_library/ashare.jsonl"
  },
  "materialization": null
}
```

入库后 `ledger_snapshot` / `materialization` 由 `store sync` 与 lift 准入路径自动刷新。
`materialization` 窗口/universe 固定为 store 口径（`all_a` / `2016-01-01`~最新），
不跟记录上的评估 `universe`（如 csi300）走。

## 可复制代码（factor.py）

```python
"""日频示例因子：20 日复权动量。"""

import polars as pl

from factorzen.daily.data.context import FactorDataContext
from factorzen.daily.factors.base import DailyFactor


class MyDailyAlpha(DailyFactor):
    name = "my_daily_alpha"
    category = "daily"
    frequency = "daily"
    required_data = ["daily"]
    lookback_days = 30
    description = "20 日复权动量：close_adj[t] / close_adj[t-20] - 1"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        return (
            ctx.daily.sort(["ts_code", "trade_date"])
            .with_columns(
                (pl.col("close_adj") / pl.col("close_adj").shift(20).over("ts_code") - 1.0)
                .alias("factor_value")
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .filter(pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())
            .collect()
        )


MyDailyAlpha()
```

## 验证

```bash
pixi run fz factor list
pixi run fz factor run my_daily_alpha --start 20230101 --end 20241231 --universe csi500
# 资产库同步 / 一致性
pixi run fz factor-library store sync --market ashare --only my_daily_alpha --no-materialize
pixi run fz factor-library store verify --market ashare
```

## 检查点

- `lookback_days` 大于最长回看窗口，给停牌和节假日留余量。
- 因子值只使用当前日期及以前可获得的数据。
- 输出列名固定为 `factor_value`，不要输出多个因子列。
- 如果公式依赖估值、市值或换手字段，先把对应数据类型加入 `required_data`。
- 手写因子进库仍走 lift 准入：`fz factor-library lift-test --factor <name>`。

---

## 相关文档

- [因子编写指南](../../../docs/guides/factor-authoring.md) —— 完整接口说明、如何让手写因子进因子库
- [因子库与增量准入](../../../docs/concepts/factor-library.md) —— 因子入库的裁决机制
- [CLI 参考](../../../docs/reference/cli.md) —— `fz factor` 全部参数
