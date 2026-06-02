# qlib factors

This package exposes qlib's built-in Alpha158 and Alpha360 feature sets as FactorZen daily factors.

Each qlib feature is registered as one FactorZen factor because the existing evaluation pipeline expects a single `factor_value` column.

## Data source

Runtime computation uses `pyqlib` and a qlib data bundle. Defaults:

- `QLIB_PROVIDER_URI=~/.qlib/qlib_data/cn_data`
- `QLIB_INSTRUMENTS=csi500`

Override either environment variable before running a factor.

## Factor names

- Alpha158: `qlib_alpha158_<feature>`, for example `qlib_alpha158_kmid`, `qlib_alpha158_ma20`, `qlib_alpha158_vsumd60`.
- Alpha360: `qlib_alpha360_<feature>`, for example `qlib_alpha360_close0`, `qlib_alpha360_open59`, `qlib_alpha360_volume0`.

Alpha158 includes 158 features: 9 K-line shape features, 4 current-price features, and 29 rolling operator families over windows `5`, `10`, `20`, `30`, and `60`.

Alpha360 includes 360 features: `CLOSE`, `OPEN`, `HIGH`, `LOW`, `VWAP`, and `VOLUME`, each with offsets `59` down to `0`.
