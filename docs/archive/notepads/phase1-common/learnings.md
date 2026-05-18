# Learnings: phase1-common

## 2026-05-13: calendar.py 创建

- 实现了 `common/calendar.py`，基于 Tushare `trade_cal` 接口（SSE 交易所）的交易日历查询模块。
- 缓存机制：首次调用拉取全量数据（1990-2030），存为 `data/cache/trade_cal.parquet`，7 天过期（`CACHE_EXPIRE_DAYS`）后自动刷新。
- Tushare 仅缓存失效时才初始化（`_fetch_from_tushare` 内部 `import tushare as ts`），避免模块级初始化造成循环依赖。
- 6 个公开函数：
  - `get_trade_calendar(start, end)` → 返回 `pl.DataFrame`，支持按日期区间筛选
  - `is_trade_date(d)` → 支持 `date` 和 `str('YYYYMMDD')` 两种输入
  - `prev_trade_date(d, n)` / `next_trade_date(d, n)` → 前/后第 n 个交易日
  - `get_trade_dates(start, end)` → 区间内交易日列表
  - `get_trading_sessions()` → 固定返回 A 股交易时段
- 日期转换统一用 `datetime.strptime(s, "%Y%m%d").date()`
- 依赖导入：`from config.settings import DATA_CACHE` / `from config.tushare_config import TUSHARE_TOKEN, CACHE_EXPIRE_DAYS`
