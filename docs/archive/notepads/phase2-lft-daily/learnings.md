# Learnings - scripts/ 入口脚本

## 2026-05-13: 创建 run_lft_single.py 和 run_lft_compare.py

### 关键发现
- `compute_fwd_returns()` 默认 `ret_col="ret_1d"`，但脚本中计算的每日收益率列名为 `"ret"`，必须显式传递 `ret_col="ret"`
- `scripts/` 子目录下运行时，需要 `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` 才能导入 `config/common/lft`
- `get_factor()` 返回的是因子**类**（Type[LFTFactor]），不是实例，需要手动实例化 `factor_cls()`

### API 契约确认
| 模块 | 函数/类 | 关键参数 |
|------|---------|---------|
| common.logger | `setup_logging()`, `get_logger(name)` | 幂等初始化 |
| common.loader | `fetch_daily(start, end)` | YYYYMMDD 格式字符串 |
| common.calendar | `get_trade_dates(start, end) -> list[date]` | YYYYMMDD 格式字符串 |
| common.universe | `get_universe(date_str, name) -> pl.DataFrame` | 含 ts_code, name 等列 |
| lft.factors.registry | `get_factor(name) -> Type[LFTFactor]` | 返回类，需实例化 |
| lft.preprocessing.pipeline | `quick_preprocess(df, col) -> pl.DataFrame` | 输出列: factor_clean |
| lft.evaluation.ic_analysis | `compute_fwd_returns(df, ret_col)` | 默认 ret_col="ret_1d" |
| lft.evaluation.ic_analysis | `compute_rank_ic(factor_df, daily_ret, factor_col)` | 默认 factor_col="factor_clean" |
| lft.evaluation.backtest | `run_stratified_backtest(factor_df, daily_ret)` | daily_ret 需含 ret 列 |
| lft.evaluation.correlation | `compute_factor_correlation(factor_dict)` | 输入 {name: df} 字典 |

### 已注册因子
- momentum_20d, reversal_5d, turnover_5d, volatility_20d
