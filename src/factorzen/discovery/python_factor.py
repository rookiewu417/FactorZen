# src/factorzen/discovery/python_factor.py
"""python 型因子物化：registry DailyFactor → [trade_date, ts_code, factor_value] 面板。

一期约束：仅 market=ashare、仅 daily 频率。扩窗预热复用 ``FactorDataContext.lookback_days``
（与 ``pipelines/daily_single`` 构建 ctx 同路径：ctx 内 ``expanded_start`` 往前推交易日）。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl

from factorzen.core.logger import get_logger

_LOG = get_logger(__name__)


def _load_universe_codes(start: str, end: str, universe: str) -> list[str]:
    """PIT membership 并集 ts_codes（与 fz factor run 的 load_pit_membership 同 core 入口）。

    直连 ``core.universe``（discovery 不许依赖 pipelines——架构分层）；错误语义与
    ``pipelines.daily_single.load_pit_membership`` 对齐：构造失败拒绝回退期末快照
    （look-ahead+幸存偏差），非 all_a 空成分 fail-loudly。模块级函数便于测试 monkeypatch。
    """
    from factorzen.core.universe import get_universe_membership

    try:
        membership = get_universe_membership(start, end, universe)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"universe={universe!r} 的逐日 PIT membership 构造失败"
            f"（{type(exc).__name__}: {exc}）；拒绝回退期末快照。"
        ) from exc

    ts_codes = (
        membership["ts_code"].unique().to_list() if not membership.is_empty() else []
    )
    if not ts_codes and universe != "all_a":
        raise ValueError(
            f"universe={universe!r} 在 {start}~{end} 无 membership（成分未回补？）"
        )
    return list(ts_codes)


def _to_date(v: Any) -> Any:
    """YYYYMMDD str / date → 与 polars Date 列可比的 date。"""
    if hasattr(v, "year"):
        return v
    s = str(v).replace("-", "")[:8]
    return datetime.strptime(s, "%Y%m%d").date()


def materialize_python_panel(
    name: str,
    start: str,
    end: str,
    universe: str,
    *,
    market: str = "ashare",
) -> pl.DataFrame:
    """registry 中的 python 因子 → 过滤后的三列面板。

    Parameters
    ----------
    name
        registry 因子名（与 ``DailyFactor.name`` / 库记录 ``name`` 一致）。
    start, end
        评估窗 ``YYYYMMDD``（闭区间）。lookback 预热在 ctx 内扩，返回值裁回此窗。
    universe
        股票池名（如 ``csi300``），经 PIT membership 取并集。
    market
        一期仅 ``ashare``；其它 → ``ValueError``。

    Returns
    -------
    pl.DataFrame
        列 ``[trade_date, ts_code, factor_value]``；null/非有限已滤。
        ``trade_date`` dtype 与因子 compute 输出一致（通常 Date，对齐库池网格再 join）。

    Raises
    ------
    ValueError
        非 ashare / 未注册 / 计算失败等（对外统一 ``ValueError``）。
    """
    if market != "ashare":
        raise ValueError(
            f"python 型因子一期仅支持 market='ashare'（A股），收到 market={market!r}"
        )

    from factorzen.daily.data.context import FactorDataContext
    from factorzen.daily.factors.registry import get_factor

    try:
        factor_cls = get_factor(name)
    except KeyError as exc:
        raise ValueError(f"python 因子未注册: {name!r}") from exc
    except Exception as exc:
        raise ValueError(f"python 因子查找失败: {name!r}: {exc}") from exc

    try:
        factor = factor_cls()
    except Exception as exc:
        raise ValueError(f"python 因子实例化失败: {name!r}: {exc}") from exc

    try:
        ts_codes = _load_universe_codes(start, end, universe)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"universe={universe!r} 加载失败: {type(exc).__name__}: {exc}"
        ) from exc

    # 与 daily_single.main 同款：lookback 交给 FactorDataContext.expanded_start 扩窗
    # （prev_trade_date(start, lookback_days)），不在此内联复制日历逻辑。
    try:
        ctx = FactorDataContext(
            start=start,
            end=end,
            required_data=list(getattr(factor, "required_data", ["daily"])),
            lookback_days=int(getattr(factor, "lookback_days", 20) or 20),
            universe=ts_codes if ts_codes else None,
            snapshot_mode="daily",  # 一期仅 daily
        )
        raw = factor.compute(ctx)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"python 因子 compute 失败: {name!r}: {type(exc).__name__}: {exc}"
        ) from exc

    if raw is None or raw.is_empty():
        return pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "ts_code": pl.Utf8,
                "factor_value": pl.Float64,
            }
        )

    need = {"trade_date", "ts_code", "factor_value"}
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(
            f"python 因子 {name!r} compute 缺列 {sorted(missing)}；"
            f"契约=[trade_date, ts_code, factor_value]"
        )

    start_d = _to_date(start)
    end_d = _to_date(end)
    td = pl.col("trade_date")
    # Date 列直接比；Utf8 列先 parse，避免 dtype 漂移静默漏滤
    if raw["trade_date"].dtype == pl.Date:
        in_window = (td >= start_d) & (td <= end_d)
    else:
        td_d = td.cast(pl.Utf8).str.replace_all("-", "").str.slice(0, 8)
        in_window = (td_d >= start) & (td_d <= end)

    panel = (
        raw.select(["trade_date", "ts_code", "factor_value"])
        .filter(in_window)
        .filter(
            pl.col("factor_value").is_not_null()
            & pl.col("factor_value").is_finite()
        )
    )
    return panel
