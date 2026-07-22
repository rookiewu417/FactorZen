# src/factorzen/discovery/python_factor.py
"""python 型因子物化：registry DailyFactor → [trade_date, ts_code, factor_value] 面板。

一期约束：仅 market=ashare、仅 daily 频率。扩窗预热复用 ``FactorDataContext.lookback_days``
（与 ``pipelines/daily_single`` 构建 ctx 同路径：ctx 内 ``expanded_start`` 往前推交易日）。

磁盘缓存：仅 python 面板（``materialize_python_panel``）。表达式因子不缓存——物化便宜且
流经多种 prep 帧，缓存易毒化（与库池 expression 路径同裁决）。
"""
from __future__ import annotations

import contextlib
import hashlib
import inspect
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from factorzen.core.logger import get_logger

_LOG = get_logger(__name__)


def _load_universe_codes(start: str, end: str, universe: str) -> list[str]:
    """PIT membership 并集 ts_codes（与 fz factor eval/backtest 的 load_pit_membership 同 core 入口）。

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


def _impl_source_sha(factor_cls: type) -> str | None:
    """实现源文件 sha1[:16]；动态类 / 取不到文件 / 读失败 → None（None = 不缓存）。

    动态类不可指纹——缓存键缺 impl 维度会毒化，必须整条路径跳过缓存。
    ``type()`` 生成的类有时 ``getsourcefile`` 会回落到调用方模块路径，但
    ``getsource`` 必失败；以 ``getsource`` 能否取到类体作为可指纹门槛。
    """
    try:
        src = inspect.getsourcefile(factor_cls)
        if not src:
            return None
        # 无源码体（动态 type() 等）→ 不可指纹
        try:
            inspect.getsource(factor_cls)
        except (OSError, TypeError):
            return None
        return hashlib.sha1(Path(src).read_bytes()).hexdigest()[:16]
    except Exception:
        return None


def _panel_cache_path(market: str, name: str, key: str) -> Path:
    """面板缓存路径：``DATA_CACHE/python_factor_panels/{market}/{name}/{key}.parquet``。"""
    from factorzen.config.settings import DATA_CACHE

    return Path(DATA_CACHE) / "python_factor_panels" / market / name / f"{key}.parquet"


def _panel_cache_key(
    market: str,
    name: str,
    start: str,
    end: str,
    universe: str,
    impl_sha: str,
    lookback_days: int,
) -> str:
    """缓存键 sha1 hexdigest[:24]。

    维度必须全部在键里（缓存键完整性是本仓 P1 教训）：
    - market：市场隔离（不同日历/数据源）
    - name：因子身份（registry 名）
    - start/end：评估窗（裁窗后面板不同）
    - universe：PIT membership 并集（票池变 → 面板变）
    - impl_sha：实现源码指纹（改源必须失效；None 时调用方全程跳过缓存）
    - lookback_days：预热窗口影响区间头部取值，必须入键
      （基类 dataclass 修复曾使同 impl_sha 下有效 lookback 发生变化）
    """
    payload = f"{market}|{name}|{start}|{end}|{universe}|{impl_sha}|lb{lookback_days}"
    return hashlib.sha1(payload.encode()).hexdigest()[:24]


def _try_read_panel_cache(path: Path) -> pl.DataFrame | None:
    """命中且三列齐全 → 面板；损坏/读失败 → 删文件并返回 None（不崩）。"""
    need = {"trade_date", "ts_code", "factor_value"}
    try:
        if not path.exists():
            return None
        cached = pl.read_parquet(path)
        if not need.issubset(set(cached.columns)):
            path.unlink(missing_ok=True)
            return None
        return cached.select(["trade_date", "ts_code", "factor_value"])
    except Exception as exc:
        _LOG.warning("python panel 缓存读失败 %s: %s；将重算", path, exc)
        with contextlib.suppress(Exception):
            path.unlink(missing_ok=True)
        return None


def _try_write_panel_cache(path: Path, panel: pl.DataFrame) -> None:
    """原子落盘（tmp → rename）；失败只 warning 不崩。ts_code Categorical → Utf8 磁盘契约。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        to_write = panel
        if "ts_code" in to_write.columns and to_write["ts_code"].dtype == pl.Categorical:
            to_write = to_write.with_columns(pl.col("ts_code").cast(pl.Utf8))
        tmp = path.with_suffix(path.suffix + ".tmp")
        to_write.write_parquet(tmp)
        tmp.replace(path)
    except Exception as exc:
        _LOG.warning("python panel 缓存写失败 %s: %s", path, exc)
        with contextlib.suppress(Exception):
            tmp = path.with_suffix(path.suffix + ".tmp")
            if tmp.exists():
                tmp.unlink(missing_ok=True)


def materialize_python_panel(
    name: str,
    start: str,
    end: str,
    universe: str,
    *,
    market: str = "ashare",
    use_cache: bool = True,
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
    use_cache
        磁盘缓存开关（默认 True）。``impl_sha is None``（动态类等）时强制跳过缓存。
        表达式因子不缓存的裁决：物化便宜且流经多种 prep 帧，缓存易毒化——本函数
        仅覆盖 python 面板。

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

    # 先实例化：缓存键与 FactorDataContext 共用同一 factor.lookback_days（防双路径漂移）
    try:
        factor = factor_cls()
    except Exception as exc:
        raise ValueError(f"python 因子实例化失败: {name!r}: {exc}") from exc

    lookback_days = int(getattr(factor, "lookback_days", 20) or 20)

    # 缓存：impl_sha None → 动态类不可指纹 → 全程跳过（不读不写）
    impl_sha = _impl_source_sha(factor_cls) if use_cache else None
    cache_path: Path | None = None
    if use_cache and impl_sha is not None:
        key = _panel_cache_key(
            market, name, start, end, universe, impl_sha, lookback_days,
        )
        try:
            cache_path = _panel_cache_path(market, name, key)
            hit = _try_read_panel_cache(cache_path)
            if hit is not None:
                return hit
        except Exception as exc:
            # 缓存层任何异常不许影响计算结果
            _LOG.warning("python panel 缓存探测失败: %s", exc)
            cache_path = None

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
            lookback_days=lookback_days,
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
        panel = pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "ts_code": pl.Utf8,
                "factor_value": pl.Float64,
            }
        )
    else:
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

    # 写缓存：失败只 warning；use_cache 关 / impl_sha 无 → 不写。
    # 空面板不写——「文件存在 ≠ 数据完整」：数据未回补时的空结果一旦落盘，
    # 回补后会持续命中空缓存直到 impl/窗口变更，属静默数据缺失。
    if use_cache and impl_sha is not None and cache_path is not None and not panel.is_empty():
        _try_write_panel_cache(cache_path, panel)

    return panel
