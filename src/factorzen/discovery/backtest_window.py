# src/factorzen/discovery/backtest_window.py
"""默认回测窗口——因子库 rebuild 与挖掘共用的枢纽。

「最近约 10 年滚动到数据最新端」（约 2016–2026 当 end 在 2026）：
``end = min(该市场缓存最新可用交易日, today?)``，``start = end − years``。
A 股再与数据湖左缘 ``2016-01-01`` 取 max（与 store 物化口径对齐）；
crypto 因数据成熟度另设起点下限 20210101。返回 ``YYYYMMDD``。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from factorzen.config.settings import (
    CRYPTO_LAKE,
    DATA_RAW_DAILY,
    DATA_RAW_FUTURES,
    DATA_RAW_US,
)

# 各市场缓存根（探测最新可用交易日用）。改这里即改探测源。
_ASHARE_DAILY_ROOT = DATA_RAW_DAILY
_FUTURES_DAILY_ROOT = DATA_RAW_FUTURES
_CRYPTO_LAKE_ROOT = CRYPTO_LAKE
# 美股缓存：provider 按 symbol 落 ``data/raw/us_daily/{sym}.parquet``（非 Hive 年月分区）。
_US_DAILY_ROOT = DATA_RAW_US

# 默认回看年数：10 → end≈2026 时约 2016–2026（与全 A 挖掘/store 主口径一致）。
DEFAULT_WINDOW_YEARS = 10
# A 股数据湖左缘 / store 物化起点（``factor_store.STORE_MATERIALIZE_START``）。
_ASHARE_START_FLOOR = date(2016, 1, 1)
# crypto 数据成熟度起点下限（早于此的 K 线覆盖不足，不作训练窗口）。
_CRYPTO_START_FLOOR = date(2021, 1, 1)


def _coerce_date(v) -> date:
    """把 parquet/JSON 里的 trade_date 值（date/datetime/str/int）统一成 ``date``。"""
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v).strip().replace("-", "").replace("/", "")
    if len(s) >= 8 and s[:8].isdigit():
        return datetime.strptime(s[:8], "%Y%m%d").date()
    raise ValueError(f"无法解析交易日: {v!r}")


def _max_trade_date_partitioned(root: str | Path) -> date | None:
    """扫 Hive 分区 ``root/year=YYYY/month=MM/*.parquet``，读最新分区 parquet 的 trade_date 最大值。

    只读最深（最新 year→month）的那个分区文件，避免全量扫描。目录/文件缺失或无 trade_date 列
    → None（调用方回退）。
    """
    base = Path(root)
    if not base.is_dir():
        return None
    parquets = sorted(base.glob("year=*/month=*/*.parquet"))
    if not parquets:
        return None
    import polars as pl

    # 从最新分区往前找，直到取到有效 trade_date（防末尾空/损坏分区）。
    for pq in reversed(parquets):
        try:
            mx = (pl.scan_parquet(pq).select(pl.col("trade_date").max())
                  .collect().item())
        except Exception:
            continue
        if mx is not None:
            try:
                return _coerce_date(mx)
            except ValueError:
                continue
    return None


def _crypto_lake_end() -> date | None:
    """crypto 数据湖最新覆盖日：优先读 ``manifest.json`` 的 ``end``，缺失回退扫 klines 月文件名。"""
    manifest = Path(_CRYPTO_LAKE_ROOT) / "manifest.json"
    if manifest.is_file():
        try:
            meta = json.loads(manifest.read_text())
            end = meta.get("end")
            if end:
                return _coerce_date(end)
        except Exception:
            pass
    # 回退：klines_1m/symbol=*/YYYY-MM.parquet 的最大月份 → 该月月末（保守）。
    kl = Path(_CRYPTO_LAKE_ROOT) / "klines_1m"
    if kl.is_dir():
        months = sorted(p.stem for p in kl.glob("symbol=*/*.parquet") if p.stem[:7].replace("-", "").isdigit())
        if months:
            y, m = months[-1][:4], months[-1][5:7]
            try:
                nm = date(int(y) + (m == "12"), (int(m) % 12) + 1, 1)
                from datetime import timedelta
                return nm - timedelta(days=1)
            except Exception:
                return None
    return None


def _us_latest_date(root: str | Path) -> date | None:
    """美股缓存最新交易日：扫 ``root/*.parquet``（每 symbol 一文件）取 trade_date 最大值。"""
    base = Path(root)
    if not base.is_dir():
        return None
    if not any(base.glob("*.parquet")):
        return None
    import polars as pl

    try:
        mx = (pl.scan_parquet(str(base / "*.parquet"))
              .select(pl.col("trade_date").max()).collect().item())
    except Exception:
        return None
    if mx is None:
        return None
    try:
        return _coerce_date(mx)
    except ValueError:
        return None


def latest_data_date(market: str) -> date | None:
    """探测该市场缓存的最大可用 ``trade_date``。缓存缺失 → None（调用方须回退/报错）。

    A股扫 ``data/raw/daily``；期货扫 ``data/raw/fut_daily``；crypto 读 ``data/crypto_lake``；
    美股扫 ``data/raw/us_daily``（每 symbol 一 parquet）。
    """
    if market == "ashare":
        return _max_trade_date_partitioned(_ASHARE_DAILY_ROOT)
    if market == "futures":
        return _max_trade_date_partitioned(_FUTURES_DAILY_ROOT)
    if market == "crypto":
        return _crypto_lake_end()
    if market == "us":
        return _us_latest_date(_US_DAILY_ROOT)
    raise ValueError(f"未知 market={market!r}，应为 ashare/crypto/futures/us")


def _minus_years(d: date, years: int) -> date:
    """d 回退 years 年；2/29 落到无闰年时退到 2/28。"""
    try:
        return d.replace(year=d.year - years)
    except ValueError:
        return d.replace(year=d.year - years, day=28)


def default_window(
    market: str,
    *,
    years: int = DEFAULT_WINDOW_YEARS,
    today: date | None = None,
) -> tuple[str, str]:
    """(start, end) as ``YYYYMMDD``。``end = min(latest_data_date(market), today?)``，
    ``start = end − years 年``（默认 ``DEFAULT_WINDOW_YEARS=10`` ≈ 2016–2026）。

    - ashare：``start`` 再与 ``2016-01-01`` 取 max（数据湖左缘 / store 物化口径）
    - crypto：``start`` 封到 ≥ 20210101

    缓存探测不到最新交易日时抛 ``ValueError``（诚实报错，不静默取 today——那会把无数据的
    未来窗口伪造成有数据）。
    """
    latest = latest_data_date(market)
    if latest is None:
        raise ValueError(
            f"探测不到 {market} 缓存的最新交易日（缓存可能未回补）；"
            f"请先补数据或显式传 --start/--end 覆盖默认窗口。"
        )
    end = latest if today is None else min(latest, today)
    start = _minus_years(end, years)
    if market == "ashare" and start < _ASHARE_START_FLOOR:
        start = _ASHARE_START_FLOOR
    if market == "crypto" and start < _CRYPTO_START_FLOOR:
        start = _CRYPTO_START_FLOOR
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


# ── 评估窗下界的真实预热预留 ────────────────────────────────────────────────

@dataclass(frozen=True)
class WarmupClamp:
    """``clamp_eval_start_for_warmup`` 的结果（全部字段进 manifest，可复现）。"""

    requested_start: str        # 调用方要的下界（YYYYMMDD）
    effective_start: str        # 实际该用的下界（YYYYMMDD）；未前推时 == requested_start
    reserved_bars: int          # 要求预留的预热 bar 数
    warm_bars_requested: int    # requested_start 之前**帧里真有**的 bar 数
    warm_bars_effective: int    # effective_start 之前**帧里真有**的 bar 数
    clamped: bool               # 是否发生了前推
    degenerate: bool            # 帧总长不足 reserved_bars，无从前推（如实报告，不前推）


def clamp_eval_start_for_warmup(
    frame_dates, requested_start: str, reserved_bars: int,
) -> WarmupClamp:
    """把评估窗下界前推到「它之前真有 ``reserved_bars`` 根 bar」的那个交易日。

    动机（2026-07-28 实测缺陷）：扩窗预热是**请求**，不是保证。
    `FactorDataContext.expanded_start` 会向前要 ``lookback_days`` 个自然日，但盘上没有
    的数据变不出来——A股 ``data/raw/daily`` 最早 20160104，用 ``--start 20160104`` 跑
    `factor-library rebuild` 时装配帧的第一行就是 20160104，
    ``prepped.filter(trade_date < eval_start)`` **为空** → `warmup_bars_by_leaf` 对
    **每个**叶子都算出 ``have=0`` → 任何带 ts_*/delay 的表达式一律被
    `warmup_shortfall` 判「预热不足」→ ``ic_train=None`` 被丢 → ``n_evaluated=0``
    而进程 exit=0。整轮 rebuild 静默空转，库里 90 条 active/probation 里 80 条带时序
    算子，一条都刷不上。

    修法不是放松预热门（那会把 `operators._MIN=3` 造出的截断窗口噪声灌进 IC 序列和
    DSR 池），而是**让评估窗自己让开预热段**：下界前推到帧里第 ``reserved_bars``
    根 bar 之后的第一天。前推后 t 日信号仍只用 ≤t 的信息（铁律#1 不受影响，前推只是
    把更早、更脏的截面移出训练统计），且窗口口径与调用方装配帧时声明的预热预算一致。

    Args:
        frame_dates: 装配帧里出现过的交易日（``date`` / ``YYYYMMDD`` 字符串混排均可，
            无需排序去重，本函数自己做）。
        requested_start: 调用方要的下界（``YYYYMMDD``）。
        reserved_bars: 要预留的预热 bar 数（rebuild 用 ``AGENT_WARMUP_LOOKBACK``，
            与 `_prepare_agent_mining_data` 装帧时声明的预算同一个数）。

    Returns:
        `WarmupClamp`。预热已足（``warm_bars_requested >= reserved_bars``）→
        ``clamped=False`` 且 ``effective_start == requested_start``（零回归）。
        帧总长 ``<= reserved_bars`` → ``degenerate=True`` 且**不前推**（前推到帧外
        等于把评估窗清空；如实报告让调用方大声抱怨，比悄悄产出空结果诚实）。
    """
    dates = sorted({_coerce_date(d) for d in frame_dates})
    req = _coerce_date(requested_start)
    req_s = req.strftime("%Y%m%d")
    n_reserved = max(0, int(reserved_bars))
    warm_req = sum(1 for d in dates if d < req)
    if warm_req >= n_reserved:
        return WarmupClamp(req_s, req_s, n_reserved, warm_req, warm_req, False, False)
    if len(dates) <= n_reserved:
        # 整帧都不够预留（空帧 / 极短窗口 / 测试桩）→ 无从前推
        return WarmupClamp(req_s, req_s, n_reserved, warm_req, warm_req, False, True)
    eff = dates[n_reserved]
    return WarmupClamp(req_s, eff.strftime("%Y%m%d"), n_reserved,
                       warm_req, n_reserved, True, False)
