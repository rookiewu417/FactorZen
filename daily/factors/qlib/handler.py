"""Qlib Alpha158/Alpha360 feature wrappers for FactorZen."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
import polars as pl

from daily.factors.base import DailyFactor

if TYPE_CHECKING:
    from daily.data.context import FactorDataContext


ALPHA158_ROLLING_OPS = [
    "ROC",
    "MA",
    "STD",
    "BETA",
    "RSQR",
    "RESI",
    "MAX",
    "MIN",
    "QTLU",
    "QTLD",
    "RANK",
    "RSV",
    "IMAX",
    "IMIN",
    "IMXD",
    "CORR",
    "CORD",
    "CNTP",
    "CNTN",
    "CNTD",
    "SUMP",
    "SUMN",
    "SUMD",
    "VMA",
    "VSTD",
    "WVMA",
    "VSUMP",
    "VSUMN",
    "VSUMD",
]
ALPHA158_WINDOWS = [5, 10, 20, 30, 60]
ALPHA158_FEATURES = (
    ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]
    + ["OPEN0", "HIGH0", "LOW0", "VWAP0"]
    + [f"{op}{window}" for op in ALPHA158_ROLLING_OPS for window in ALPHA158_WINDOWS]
)
ALPHA360_FEATURES = [
    f"{field}{offset}"
    for field in ("CLOSE", "OPEN", "HIGH", "LOW", "VWAP", "VOLUME")
    for offset in range(59, -1, -1)
]

_QLIB_INITIALIZED = False
_QLIB_FRAME_CACHE: dict[tuple[str, str, str, str, str], pl.DataFrame] = {}


def alpha158_factor_names() -> list[str]:
    return [f"qlib_alpha158_{feature.lower()}" for feature in ALPHA158_FEATURES]


def alpha360_factor_names() -> list[str]:
    return [f"qlib_alpha360_{feature.lower()}" for feature in ALPHA360_FEATURES]


def _class_name(prefix: str, feature: str) -> str:
    parts = re.findall(r"[A-Za-z]+|\d+", feature.lower())
    return prefix + "".join(part.capitalize() for part in parts)


def _to_qlib_date(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def _to_tushare_code(instrument: Any) -> str:
    code = str(instrument)
    if len(code) == 8 and code[:2] in {"SH", "SZ"}:
        return f"{code[2:]}.{code[:2]}"
    return code


def _default_provider_uri() -> str:
    return os.getenv("QLIB_PROVIDER_URI", str(Path.home() / ".qlib" / "qlib_data" / "cn_data"))


def _default_instruments() -> str:
    return os.getenv("QLIB_INSTRUMENTS", "csi500")


def _init_qlib(provider_uri: str) -> None:
    global _QLIB_INITIALIZED
    if _QLIB_INITIALIZED:
        return

    import qlib
    from qlib.constant import REG_CN

    qlib.init(provider_uri=provider_uri, region=REG_CN)
    _QLIB_INITIALIZED = True


def _normalize_qlib_frame(df: pd.DataFrame) -> pl.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(col[-1]) for col in df.columns]

    pdf = df.reset_index()
    rename_map = {
        "datetime": "trade_date",
        "date": "trade_date",
        "instrument": "ts_code",
        "symbol": "ts_code",
    }
    pdf = pdf.rename(columns={k: v for k, v in rename_map.items() if k in pdf.columns})

    required = {"trade_date", "ts_code"}
    missing = required.difference(pdf.columns)
    if missing:
        raise ValueError(f"qlib output missing required columns: {sorted(missing)}")

    pdf["trade_date"] = pd.to_datetime(pdf["trade_date"]).dt.date
    pdf["ts_code"] = pdf["ts_code"].map(_to_tushare_code)

    return pl.from_pandas(pdf)


def load_qlib_feature_frame(
    handler_name: str,
    feature_name: str,
    ctx: FactorDataContext,
) -> pl.DataFrame:
    provider_uri = os.getenv("QLIB_PROVIDER_URI", _default_provider_uri())
    instruments = os.getenv("QLIB_INSTRUMENTS", _default_instruments())
    cache_key = (handler_name, ctx.start, ctx.end, provider_uri, instruments)

    if cache_key in _QLIB_FRAME_CACHE:
        frame = _QLIB_FRAME_CACHE[cache_key]
        if feature_name not in frame.columns:
            raise KeyError(
                f"qlib feature '{feature_name}' not found. Available: {frame.columns}"
            )
        return frame.select(["trade_date", "ts_code", feature_name])

    _init_qlib(provider_uri)

    from qlib.contrib.data.handler import Alpha158, Alpha360

    handler_cls = {"alpha158": Alpha158, "alpha360": Alpha360}[handler_name]
    handler = handler_cls(
        instruments=instruments,
        start_time=_to_qlib_date(ctx.start),
        end_time=_to_qlib_date(ctx.end),
        fit_start_time=_to_qlib_date(ctx.start),
        fit_end_time=_to_qlib_date(ctx.end),
        infer_processors=[],
        learn_processors=[],
    )
    frame = _normalize_qlib_frame(handler.fetch(col_set="feature"))
    _QLIB_FRAME_CACHE[cache_key] = frame
    if feature_name not in frame.columns:
        raise KeyError(f"qlib feature '{feature_name}' not found. Available: {frame.columns}")
    return frame.select(["trade_date", "ts_code", feature_name])


class QlibFeatureFactor(DailyFactor):
    category = "qlib"
    required_data: ClassVar[list[str]] = ["daily"]
    lookback_days = 60
    qlib_handler_name: ClassVar[str]
    qlib_feature_name: ClassVar[str]

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        frame = load_qlib_feature_frame(self.qlib_handler_name, self.qlib_feature_name, ctx)
        return frame.select(
            [
                "trade_date",
                "ts_code",
                pl.col(self.qlib_feature_name).cast(pl.Float64).alias("factor_value"),
            ]
        )


def _make_feature_class(handler_name: str, feature_name: str) -> type[QlibFeatureFactor]:
    prefix = "QlibAlpha158" if handler_name == "alpha158" else "QlibAlpha360"
    factor_name = f"qlib_{handler_name}_{feature_name.lower()}"
    return type(
        _class_name(prefix, feature_name),
        (QlibFeatureFactor,),
        {
            "name": factor_name,
            "description": f"qlib {handler_name.upper()} feature {feature_name}",
            "qlib_handler_name": handler_name,
            "qlib_feature_name": feature_name,
            "__module__": __name__,
        },
    )


for _feature in ALPHA158_FEATURES:
    globals()[_class_name("QlibAlpha158", _feature)] = _make_feature_class("alpha158", _feature)

for _feature in ALPHA360_FEATURES:
    globals()[_class_name("QlibAlpha360", _feature)] = _make_feature_class("alpha360", _feature)
