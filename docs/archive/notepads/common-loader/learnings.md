# loader.py 实现记录

## 设计决策

### 分段策略
- **日线 (daily/daily_basic)**: 按年分段，缓存检查用季度首月 (1, 4, 7, 10) 作为代理检查点
- **分钟线 (minute)**: 按月分段，精确匹配 partition_exists(year, month)
- **财务 (finance)**: 按季度分段，缓存检查用季度第一个月
- **股票基础信息 (stock_basic)**: 全量缓存 7 天，过期自动刷新

### 缓存检查策略
- 日线数据量大，分片后使用 Q1/Q2/Q3/Q4 首月 partition 存在性作为"年份已缓存"的代理判断
- 分钟数据和季度数据直接检查 partition
- stock_basic 用文件修改时间判断缓存是否过期

### 关键约定
- `_retry` 返回原始 Tushare 结果（pandas DataFrame），由各 fetch 函数自行 `pl.from_pandas()` 转换
- `_rate_limit` 使用全局 `_last_call` 时间戳，确保多函数调用间也限流
- 日期字符串统一使用 `_str_to_date()` 转换为 pl.Date

### 无数据/失败处理
- 参数/权限错误 (`token`, `param`, `积分`): 立即 raise，不重试
- 网络/超时错误: 指数退避重试 MAX_RETRIES 次
- 某分段拉取失败: 记录错误日志后 `continue` 到下一分段，不阻断整体流程
- stock_basic 拉取失败且有过期缓存: 降级返回过期数据
