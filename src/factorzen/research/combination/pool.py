# src/factorzen/research/combination/pool.py
"""库池容器：单骨架宽面板（Compact）与增量混合（Hybrid）。

**放在 research/combination 的原因（架构方向）**：discovery→research 是既有单向依赖
（lift_test → combination.models/cv）；容器类被两侧消费，必须落在**被依赖侧**才不成环
（research/combination 的 models/oos 反向 import discovery 会闭环，架构守卫拒绝）。
构建入口仍在 ``discovery.factor_library.build_library_pool``（从此处 import 并 re-export，
旧调用方 ``from factorzen.discovery.factor_library import CompactLibraryPool`` 不变）。

本模块只依赖 polars——绝不 import lightgbm/discovery/agents。
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping

import polars as pl

# 库池内存：utf8 date+code 键约 41B/行；dict-of-frames 会 × 因子数复制键。
# 估算 n_factors × n_rows × 该常数 ≥ 阈值时自动走单骨架宽面板（compact）。
# 阈值 4 GiB：2026-07-17 实测 csi800×2020-2026(87 因子×2.3M 行,键副本估算 7.6G)
# 在 23G 机器上逐轮护栏 OOM——该量级必须省;compact 有逐值 parity 测试背书,
# 提前切换无数值代价。小帧/测试(<4G)仍走 legacy 零回归。
POOL_KEY_BYTES_PER_ROW = 41
POOL_COMPACT_BYTES_THRESHOLD = 4 * (1024**3)  # 4 GiB
# 值列 f32 阈值:估算 n_factors×n_rows×8B ≥ 此值时 compact 池值列存 f32
# (仅存储层;__getitem__/numpy 边界升回 f64)。csi800 级 ~1.25G 不触发,f64 零回归。
POOL_VALUE_F32_BYTES_THRESHOLD = 2 * (1024**3)  # 2 GiB
# ts_code 读回 Categorical 的行数阈值(P4c):与 discovery.preparation.
# KEYS_CATEGORICAL_ROWS_THRESHOLD 同值对齐;本模块禁止 import discovery,
# 故本地定义,一致性由 tests/test_pool_prebuild.py 断言防漂移。
POOL_KEYS_CATEGORICAL_ROWS_THRESHOLD = 4_000_000


class CompactLibraryPool(Mapping[str, pl.DataFrame]):
    """单骨架宽面板库池：键列一份 + 每因子一列 f64（null 保留非有限）。

    作为 ``Mapping[str, DataFrame]`` 兼容壳：``pool[expr]`` 惰性拼出长表并
    ``filter(null/inf)``，与旧 ``build_library_pool`` 单因子帧语义对齐。

    **热路径禁止**对全池 ``list(values())`` / ``dict(pool)``——会再次复制键。
    应走 ``wide`` 或 ``build_library_panel`` / ``build_library_corr_panel`` 的
    compact 分支（一次散射到矩阵）。
    """

    __slots__ = ("_names", "_wide")

    def __init__(
        self,
        wide: pl.DataFrame,
        names: tuple[str, ...] | None = None,
    ) -> None:
        if names is None:
            names = tuple(
                c for c in wide.columns if c not in ("trade_date", "ts_code")
            )
        self._wide = wide
        self._names = names

    @property
    def wide(self) -> pl.DataFrame:
        return self._wide

    @property
    def factor_names(self) -> tuple[str, ...]:
        return self._names

    def __getitem__(self, key: str) -> pl.DataFrame:
        if key not in self._names:
            raise KeyError(key)
        # factor_value 出口恒 f64:内部可为 f32 存储(大池省内存),API dtype 契约不变
        return (
            self._wide.select(
                ["trade_date", "ts_code",
                 pl.col(key).cast(pl.Float64).alias("factor_value")]
            )
            .filter(
                pl.col("factor_value").is_not_null()
                & pl.col("factor_value").is_finite()
            )
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._names

    def __bool__(self) -> bool:
        return len(self._names) > 0

    def __repr__(self) -> str:
        return f"CompactLibraryPool(n_factors={len(self._names)}, n_rows={self._wide.height})"

    def filter_dates(self, dates) -> CompactLibraryPool:
        """按 trade_date 集合过滤（M1 mining 段切片）；丢弃全 null/非有限列。"""
        date_list = list(dates) if not isinstance(dates, list) else dates
        if not date_list:
            return CompactLibraryPool(
                self._wide.head(0), (),
            )
        filtered = self._wide.filter(pl.col("trade_date").is_in(date_list))
        return self._drop_empty_factor_cols(filtered, self._names)

    def with_safe_feature_names(self) -> CompactLibraryPool:
        """键 → f000, f001, …（lift / LGBM 安全名），列序=插入序。"""
        rename = {n: f"f{i:03d}" for i, n in enumerate(self._names)}
        new_names = tuple(rename[n] for n in self._names)
        return CompactLibraryPool(self._wide.rename(rename), new_names)

    def drop_degenerate(self) -> CompactLibraryPool:
        """剔除全 null / 无有限值的因子列。"""
        return self._drop_empty_factor_cols(self._wide, self._names)

    def to_feature_wide(self, *, cast_date_utf8: bool = False) -> pl.DataFrame:
        """宽表 [trade_date, ts_code, *factor_names]（lift build_panel 用）。

        行集 = **至少一因子有限**——与 legacy「逐因子 filter 后 outer join」的并集
        逐行对齐;不滤则预热期全 null 行(带 ret)会混进 LGBM 面板与 fold 日期轴,
        造成 compact/legacy 静默数值漂移。
        """
        cols = ["trade_date", "ts_code", *self._names]
        out = self._wide.select(cols)
        if self._names:
            any_finite = pl.any_horizontal(
                [pl.col(n).is_not_null() & pl.col(n).is_finite() for n in self._names]
            )
            out = out.filter(any_finite)
        if cast_date_utf8:
            out = out.with_columns(pl.col("trade_date").cast(pl.Utf8))
        return out

    def with_extra_factors(
        self, extras: Mapping[str, pl.DataFrame],
    ) -> HybridLibraryPool:
        """基线 compact + 新增长表因子（lift 候选增量）。"""
        return HybridLibraryPool(self, dict(extras))

    def write_parquet(self, path) -> None:
        """磁盘契约:ts_code 落盘转 Utf8;trade_date 保持 Date;值列 dtype 原样。"""
        self._wide.with_columns(pl.col("ts_code").cast(pl.Utf8)).write_parquet(path)

    @classmethod
    def from_parquet(
        cls,
        path,
        factor_names=None,
        *,
        categorical_keys=None,
    ) -> CompactLibraryPool:
        """从 parquet 装载宽面板池。

        - ``factor_names``:给定时按该顺序恢复因子列序,缺列抛 ``ValueError``;
          ``None`` 则从列自动推(与构造器一致)。
        - ``categorical_keys``:``None``(默认)→ 行数 ≥ 阈值时 ts_code 转 Categorical;
          ``True``/``False`` 显式开关。
        """
        wide = pl.read_parquet(path)
        if factor_names is not None:
            names = tuple(factor_names)
            missing = [n for n in names if n not in wide.columns]
            if missing:
                raise ValueError(
                    f"parquet 缺因子列: {missing}"
                )
        else:
            names = None

        if categorical_keys is None:
            use_cat = wide.height >= POOL_KEYS_CATEGORICAL_ROWS_THRESHOLD
        else:
            use_cat = bool(categorical_keys)
        if use_cat and "ts_code" in wide.columns:
            wide = wide.with_columns(pl.col("ts_code").cast(pl.Categorical))

        return cls(wide, names)

    @staticmethod
    def _drop_empty_factor_cols(
        wide: pl.DataFrame, names: tuple[str, ...],
    ) -> CompactLibraryPool:
        if wide.is_empty() or not names:
            return CompactLibraryPool(
                wide.select(["trade_date", "ts_code"]) if "trade_date" in wide.columns
                else wide,
                (),
            )
        kept: list[str] = []
        for n in names:
            if n not in wide.columns:
                continue
            col = wide[n]
            # 有限非空才保留（对齐旧 filter is_finite 后非空）
            finite = col.is_not_null() & col.is_finite()
            if bool(finite.any()):
                kept.append(n)
        if not kept:
            return CompactLibraryPool(wide.select(["trade_date", "ts_code"]), ())
        return CompactLibraryPool(
            wide.select(["trade_date", "ts_code", *kept]), tuple(kept),
        )


class HybridLibraryPool(Mapping[str, pl.DataFrame]):
    """compact 基线 + 少量新增长表因子（lift 增量 combine）。

    基线不物化为 dict；新因子以长表持有。``dict(pool)`` 仍会炸——combine 须识别本类型。
    """

    __slots__ = ("base", "extras")

    def __init__(
        self,
        base: CompactLibraryPool,
        extras: dict[str, pl.DataFrame],
    ) -> None:
        self.base = base
        self.extras = extras

    def __getitem__(self, key: str) -> pl.DataFrame:
        if key in self.extras:
            return self.extras[key]
        return self.base[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.base
        yield from self.extras

    def __len__(self) -> int:
        return len(self.base) + len(self.extras)

    def __contains__(self, key: object) -> bool:
        return key in self.base or key in self.extras

    def __bool__(self) -> bool:
        return bool(self.base) or bool(self.extras)


def estimate_library_pool_key_bytes(n_factors: int, n_rows: int) -> int:
    """估算 dict-of-frames 键副本峰值字节数。"""
    return int(n_factors) * int(n_rows) * POOL_KEY_BYTES_PER_ROW


def should_use_compact_pool(
    n_factors: int,
    n_rows: int,
    *,
    threshold: int = POOL_COMPACT_BYTES_THRESHOLD,
) -> bool:
    return estimate_library_pool_key_bytes(n_factors, n_rows) >= threshold


def is_compact_pool(obj: object) -> bool:
    return isinstance(obj, (CompactLibraryPool, HybridLibraryPool))
