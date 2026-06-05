# qlib 因子

本包把 qlib 内置的 Alpha158 与 Alpha360 特征集，暴露为 FactorZen 的日频因子。

每个 qlib 特征注册为**一个**独立的 FactorZen 因子，因为现有评估管线期望单列 `factor_value`。

## 数据源

运行期计算依赖 `pyqlib` 与 qlib 数据包。默认值：

- `QLIB_PROVIDER_URI=~/.qlib/qlib_data/cn_data`
- `QLIB_INSTRUMENTS=csi500`

运行因子前可通过环境变量覆盖其中任意一项。

## 因子命名

- **Alpha158**：`qlib_alpha158_<feature>`，例如 `qlib_alpha158_kmid`、`qlib_alpha158_ma20`、`qlib_alpha158_vsumd60`。
- **Alpha360**：`qlib_alpha360_<feature>`，例如 `qlib_alpha360_close0`、`qlib_alpha360_open59`、`qlib_alpha360_volume0`。

Alpha158 含 158 个特征：9 个 K 线形态特征、4 个当前价格特征，以及 29 个滚动算子族，窗口为 `5`、`10`、`20`、`30`、`60`。

Alpha360 含 360 个特征：`CLOSE`、`OPEN`、`HIGH`、`LOW`、`VWAP`、`VOLUME` 六类，每类带 `59` 到 `0` 的偏移。
