# 性能工程

> 让「快」有数字、有方法论、有回归门。本文记录基准口径与**诚实的**实测结果(含"并行没起效"这样的负面发现)。均可复现。

## 遗传搜索并行评分 + 瓶颈定位

遗传搜索每代对种群逐个体评分(表达式 → 求值 → IC 打分)。已加入线程池批量预热分数缓存的
`score_many` 钩子(`--workers`),**确定性保证**:缓存键为表达式串、值只依赖表达式,
填充顺序与 worker 数无关——`tests/test_discovery_genetic_parallel.py` 断言同 seed 下
`workers=1` 与 `workers=4` 的 leaderboard(表达式 + 分数)逐项相等。

### 实测(80 股 × 250 交易日合成集, `--trials 120 --seed 11`)

| workers | 耗时 | 加速比 |
|---------|------|--------|
| 1 | 42.65s | 1.00× |
| 4 | 38.53s | 1.11× |
| 8 | 38.29s | 1.11× |

**加速比仅 1.11× 并快速饱和。** cProfile(100 trials,60×200)显示:总耗时 86.9s 中评分
(`_factor_values` + `score_candidate`)占约 92%,理应是并行主战场——但线程池几乎没提速。

### 结论:评分是 Python/GIL-bound,不是 polars-bound

评分的热点在 IC 打分的**逐日 `group_by` 迭代 + 每日 `np.corrcoef`**(Python 循环,持有 GIL),
而非 polars 惰性表达式(那才释放 GIL)。因此线程池无法并行——这是一个**诚实的负面结果**:
不做 profile 就上线程池,会误以为"并行了"。

`score_many` 钩子本身是正确的可扩展基础设施(确定性已验证),真正的加速需先消除 GIL 瓶颈:

1. **向量化 IC 计算**(首选):把逐日 `group_by`+`corrcoef` 改为 polars 批量截面 rank + 相关,
   评分释放 GIL 后线程池收益才会显现。
2. **多进程**:`ProcessPoolExecutor` 绕过 GIL,但需权衡 `daily`/`bundle` 的序列化成本
   (大盘数据可能得不偿失)。
3. **Rust 内核**(见下):把滚动/截面算子下沉到原生代码。

## 基准回归(规划)

后续接入 `pytest-benchmark`(算子求值 / 回测快路径 / 风险模型构建 / 遗传单代)固定规模基线,
CI 加性能回归监控 job(阈值告警不阻塞,避免 runner 抖动误伤)。

## 原生算子内核(规划)

热点滚动算子(分组 rolling corr/std/rank)以 Rust + pyo3 重写,特性开关接入 + 与 polars 路径
逐元素数值一致性测试。

## 复现

```bash
# 加速比
fz mine search --method genetic --trials 120 --seed 11 --workers 1   # vs --workers 4
# profile
python -m cProfile -s cumulative -m factorzen.cli.main mine search --method genetic ...
```
