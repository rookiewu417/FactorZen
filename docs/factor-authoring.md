# 因子编写

创建日频因子：

```bash
pixi run fz factor new my_alpha --frequency daily
```

生成文件：

```text
workspace/factors/daily/my_alpha.py
```

运行测试：

```bash
pixi run fz factor run my_alpha --start 20250101 --end 20260513 --universe csi500
```

因子发现只扫描 `workspace/factors/{daily,weekly,monthly,intraday}` 和 `workspace/factors/qlib` 中的因子实现。

`src/factorzen/daily/factors` 与 `src/factorzen/intraday/factors` 只保留框架基类和注册中心；日常研究不要在 `src` 里新增因子实现。
