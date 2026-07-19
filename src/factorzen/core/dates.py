"""trade_date 形态规范化：单一真源。

**为什么需要它**（2026-07-18 实证的 live P0）：同一条 lift 链路上，
候选面板 ``trade_date`` 常是 ``pl.Date``（prepped 帧原生），收益面板被
``_build_ret_panel`` 显式 ``cast(pl.Utf8)`` 成 ISO ``YYYY-MM-DD``。
若一侧转成紧凑 ``YYYYMMDD``、另一侧是 ISO，则 join **零命中**且不报错——
IC 序列静默变空、``_mean_ic`` 返回哨兵 0.0，一路写进因子库。

同理，窗界字符串（``cli._lift_admission_str`` 产出 ISO）与紧凑日期直接比大小
也会静默错行：``"20260405" > "2026-04-10"`` 逐字符比较为真（``'0' > '-'``）。

因此**凡是把 trade_date 变成字符串参与 join / 比较的地方，一律过本模块**，
统一到 ISO ``YYYY-MM-DD``（与库内既有 ``scored_start`` / ``admission_*`` 形态一致）。
"""
from __future__ import annotations

from typing import Any

import polars as pl

ISO_FMT = "%Y-%m-%d"

_TEMPORAL = (pl.Date, pl.Datetime)


def iso_date_str(v: Any) -> str | None:
    """标量 → ISO ``YYYY-MM-DD``；``None`` 透传。

    接受 ``date`` / ``datetime`` / ``"YYYYMMDD"`` / ``"YYYY-MM-DD"`` /
    ``"YYYY/MM/DD"``；无法识别的串原样返回（调用方比较时仍是自洽的）。
    """
    if v is None:
        return None
    if hasattr(v, "strftime"):
        try:
            return v.strftime(ISO_FMT)
        except (TypeError, ValueError):
            pass
    s = str(v).strip().replace("/", "-")
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    if len(s) >= 10 and s[4] == "-":
        return s[:10]
    return s


def iso_date_expr(dtype: Any, col: str = "trade_date") -> pl.Expr:
    """列 → ISO ``YYYY-MM-DD`` 字符串表达式（``dtype`` 为该列当前类型）。

    时间类型走 ``strftime``；字符串类型把紧凑 8 位数字补横杠、其余截前 10 字符
    （``Datetime`` cast 出的 ``"2026-04-05 00:00:00"`` 亦落到正确前缀）。
    """
    if dtype in _TEMPORAL:
        return pl.col(col).dt.strftime(ISO_FMT).alias(col)
    s = pl.col(col).cast(pl.Utf8)
    compact = (s.str.len_chars() == 8) & s.str.contains(r"^\d{8}$")
    return (
        pl.when(compact)
        .then(
            pl.concat_str([
                s.str.slice(0, 4), pl.lit("-"),
                s.str.slice(4, 2), pl.lit("-"),
                s.str.slice(6, 2),
            ])
        )
        .otherwise(s.str.slice(0, 10))
        .alias(col)
    )


def with_iso_date(df: pl.DataFrame, col: str = "trade_date") -> pl.DataFrame:
    """就地把 ``col`` 规范成 ISO 字符串列；无该列则原样返回。"""
    if df is None or col not in df.columns:
        return df
    return df.with_columns(iso_date_expr(df.schema[col], col))
