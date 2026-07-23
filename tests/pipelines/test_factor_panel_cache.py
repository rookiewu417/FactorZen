"""ensure_factor_store_panel：单一 store parquet 复用 / 补行 / expression 失效。"""

from __future__ import annotations

import json
from datetime import date, timedelta

import polars as pl
import pytest


def _panel(dates: list[date], codes: list[str], seed: float = 1.0) -> pl.DataFrame:
    rows = []
    for d in dates:
        for i, code in enumerate(codes):
            v = seed + float(i) + d.toordinal() * 0.0001
            rows.append(
                {
                    "trade_date": d,
                    "ts_code": code,
                    "factor_value": v,
                    "factor_clean": v * 0.5,
                }
            )
    return pl.DataFrame(rows)


def _stub_factor(name: str = "stub_f", expression: str | None = "rank(close)"):
    calls: list[tuple[str, str]] = []

    class Stub:
        def __init__(self):
            self.name = name
            self.expression = expression
            self.required_data = ["daily"]
            self.lookback_days = 0
            self.category = "test"
            self.calls = calls

        def compute(self, ctx):
            calls.append((ctx.start, ctx.end))
            # ctx 用 YYYYMMDD
            s = date(
                int(str(ctx.start)[:4]),
                int(str(ctx.start)[4:6]),
                int(str(ctx.start)[6:8]),
            )
            e = date(
                int(str(ctx.end)[:4]),
                int(str(ctx.end)[4:6]),
                int(str(ctx.end)[6:8]),
            )
            # 覆盖 start..end 的日历日（测试不依赖交易日历）
            n = (e - s).days + 1
            dates = [s + timedelta(days=i) for i in range(max(n, 1))]
            return _panel(dates, ["000001.SZ", "000002.SZ"], seed=len(calls))

    return Stub()


@pytest.fixture
def patched_ensure(monkeypatch, tmp_path):
    """mock ensure_data / finalize，离线。"""
    from factorzen.pipelines import factor_panel_cache as fpc

    store = tmp_path / "factors"
    store.mkdir()

    monkeypatch.setattr(
        "factorzen.discovery.factor_store.DEFAULT_ROOT",
        str(store),
    )
    monkeypatch.setattr(
        fpc,
        "_compute_segment",
        lambda factor, seg_start, seg_end, *, benchmark=None: _cast_from_stub(
            factor, seg_start, seg_end
        ),
    )
    monkeypatch.setattr(
        "factorzen.discovery.factor_store.store_materialize_end",
        lambda: "2020-01-20",
    )
    monkeypatch.setattr(
        "factorzen.discovery.factor_store.STORE_MATERIALIZE_START",
        "2020-01-01",
    )
    return store


def _cast_from_stub(factor, seg_start, seg_end):
    from factorzen.pipelines.factor_panel_cache import _cast_panel, _filter_window

    # 记录调用
    if hasattr(factor, "calls"):
        factor.calls.append((seg_start, seg_end))
    sk = str(seg_start).replace("-", "")[:8]
    ek = str(seg_end).replace("-", "")[:8]
    s = date(int(sk[:4]), int(sk[4:6]), int(sk[6:8]))
    e = date(int(ek[:4]), int(ek[4:6]), int(ek[6:8]))
    n = (e - s).days + 1
    dates = [s + timedelta(days=i) for i in range(max(n, 1))]
    seed = float(len(getattr(factor, "calls", [])) or 1)
    panel = _panel(dates, ["000001.SZ", "000002.SZ"], seed=seed)
    return _filter_window(_cast_panel(panel), seg_start, seg_end)


def test_ensure_full_window_when_missing(patched_ensure):
    from factorzen.pipelines.factor_panel_cache import ensure_factor_store_panel

    store = patched_ensure
    factor = _stub_factor()
    out = ensure_factor_store_panel(
        factor, "20200105", "20200110", root=str(store)
    )
    assert out is not None
    pq = store / "ashare" / factor.name / "factor.parquet"
    assert pq.exists()
    df = pl.read_parquet(pq)
    assert df.columns == ["trade_date", "ts_code", "factor_value", "factor_clean"]
    meta = json.loads((pq.parent / "meta.json").read_text(encoding="utf-8"))
    mat = meta["materialization"]
    assert mat["start"] == "2020-01-01"
    assert mat["end"] == "2020-01-20"
    assert mat["universe"] == "all_a"
    assert len(factor.calls) == 1
    # 全窗 target_start=min(req, STORE_START)=2016? wait we patched STORE to 2020-01-01
    assert factor.calls[0][0].replace("-", "")[:8] == "20200101"


def test_ensure_hit_no_compute(patched_ensure):
    from factorzen.discovery.factor_store import finalize_factor_panel
    from factorzen.pipelines.factor_panel_cache import ensure_factor_store_panel

    store = patched_ensure
    factor = _stub_factor()
    asset = store / "ashare" / factor.name
    asset.mkdir(parents=True)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(20)]
    panel = finalize_factor_panel(
        _panel(dates, ["000001.SZ", "000002.SZ"]).drop("factor_clean")
    )
    panel.write_parquet(asset / "factor.parquet")
    (asset / "meta.json").write_text(
        json.dumps(
            {
                "name": factor.name,
                "expression": "rank(close)",
                "materialization": {
                    "start": "2020-01-01",
                    "end": "2020-01-20",
                    "universe": "all_a",
                },
            }
        ),
        encoding="utf-8",
    )
    mtime_before = (asset / "factor.parquet").stat().st_mtime_ns
    content_before = (asset / "factor.parquet").read_bytes()

    out = ensure_factor_store_panel(
        factor, "20200105", "20200110", root=str(store)
    )
    assert out is not None
    assert factor.calls == []
    assert (asset / "factor.parquet").stat().st_mtime_ns == mtime_before
    assert (asset / "factor.parquet").read_bytes() == content_before


def test_ensure_extends_tail(patched_ensure):
    from factorzen.discovery.factor_store import finalize_factor_panel
    from factorzen.pipelines.factor_panel_cache import ensure_factor_store_panel

    store = patched_ensure
    factor = _stub_factor()
    asset = store / "ashare" / factor.name
    asset.mkdir(parents=True)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(10)]  # ~01-01..01-10
    panel = finalize_factor_panel(
        _panel(dates, ["000001.SZ", "000002.SZ"]).drop("factor_clean")
    )
    old_n = panel.height
    panel.write_parquet(asset / "factor.parquet")
    (asset / "meta.json").write_text(
        json.dumps(
            {
                "name": factor.name,
                "expression": "rank(close)",
                "materialization": {
                    "start": "2020-01-01",
                    "end": "2020-01-10",
                    "universe": "all_a",
                },
            }
        ),
        encoding="utf-8",
    )

    out = ensure_factor_store_panel(
        factor, "20200105", "20200115", root=str(store)
    )
    assert out is not None
    assert len(factor.calls) == 1
    # 只算尾段
    assert "20200111" in factor.calls[0][0].replace("-", "") or factor.calls[0][
        0
    ].startswith("2020-01-11")
    df = pl.read_parquet(asset / "factor.parquet")
    assert df.height > old_n
    assert df.unique(subset=["trade_date", "ts_code"]).height == df.height
    assert list(asset.glob("*.parquet")) == [asset / "factor.parquet"]
    assert df.columns == ["trade_date", "ts_code", "factor_value", "factor_clean"]


def test_ensure_extends_head(patched_ensure):
    from factorzen.discovery.factor_store import finalize_factor_panel
    from factorzen.pipelines.factor_panel_cache import ensure_factor_store_panel

    store = patched_ensure
    factor = _stub_factor()
    asset = store / "ashare" / factor.name
    asset.mkdir(parents=True)
    dates = [date(2020, 1, 10) + timedelta(days=i) for i in range(5)]
    panel = finalize_factor_panel(
        _panel(dates, ["000001.SZ", "000002.SZ"]).drop("factor_clean")
    )
    old_n = panel.height
    panel.write_parquet(asset / "factor.parquet")
    (asset / "meta.json").write_text(
        json.dumps(
            {
                "name": factor.name,
                "expression": "rank(close)",
                "materialization": {
                    "start": "2020-01-10",
                    "end": "2020-01-20",
                    "universe": "all_a",
                },
            }
        ),
        encoding="utf-8",
    )

    out = ensure_factor_store_panel(
        factor, "20200105", "20200112", root=str(store)
    )
    assert out is not None
    assert len(factor.calls) == 1
    df = pl.read_parquet(asset / "factor.parquet")
    assert df.height > old_n
    assert df.unique(subset=["trade_date", "ts_code"]).height == df.height


def test_ensure_expression_stale_full_recompute(patched_ensure):
    from factorzen.discovery.factor_store import finalize_factor_panel
    from factorzen.pipelines.factor_panel_cache import ensure_factor_store_panel

    store = patched_ensure
    factor = _stub_factor(expression="rank(open)")
    asset = store / "ashare" / factor.name
    asset.mkdir(parents=True)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(10)]
    panel = finalize_factor_panel(
        _panel(dates, ["000001.SZ", "000002.SZ"]).drop("factor_clean")
    )
    panel.write_parquet(asset / "factor.parquet")
    (asset / "meta.json").write_text(
        json.dumps(
            {
                "name": factor.name,
                "expression": "rank(close)",  # 与 factor 不一致
                "materialization": {
                    "start": "2020-01-01",
                    "end": "2020-01-20",
                    "universe": "all_a",
                },
            }
        ),
        encoding="utf-8",
    )

    out = ensure_factor_store_panel(
        factor, "20200105", "20200110", root=str(store)
    )
    assert out is not None
    assert len(factor.calls) == 1
    meta = json.loads((asset / "meta.json").read_text(encoding="utf-8"))
    assert meta["expression"] == "rank(open)"


def test_save_results_does_not_clobber_store_panel(tmp_path, monkeypatch):
    """回归：评估 _save_results 不得用评估窗子集覆盖 store parquet。"""
    from factorzen.daily.evaluation.backtest import StrategyBacktestResult
    from factorzen.daily.evaluation.ic_analysis import ICAnalysisResult
    from factorzen.daily.evaluation.turnover import TurnoverResult
    from factorzen.pipelines import _report_persistence as persist

    store = tmp_path / "store"
    panel_path = store / "ashare" / "momentum_20d" / "factor.parquet"
    panel_path.parent.mkdir(parents=True)
    full = _panel(
        [date(2020, 1, 1) + timedelta(days=i) for i in range(30)],
        ["000001.SZ", "000002.SZ", "000003.SZ"],
    )
    full.write_parquet(panel_path)
    before = panel_path.read_bytes()
    monkeypatch.setattr(
        "factorzen.discovery.factor_store.DEFAULT_ROOT", str(store)
    )

    # 禁止 write_factor_panel 被调用
    def _boom(*_a, **_k):
        raise AssertionError("write_factor_panel must not be called from _save_results")

    monkeypatch.setattr(
        "factorzen.discovery.factor_store.write_factor_panel",
        _boom,
    )

    run_dir = tmp_path / "run"
    subset = full.filter(pl.col("trade_date") <= date(2020, 1, 5))
    ic = ICAnalysisResult(
        factor_name="momentum_20d",
        ic_mean=0.01,
        ic_std=0.1,
        ir=0.1,
        ic_positive_ratio=0.5,
        n_periods=1,
        ic_series=pl.DataFrame({"trade_date": [date(2020, 1, 2)], "ic": [0.01]}),
    )
    returns = pl.DataFrame(
        {
            "trade_date": [date(2020, 1, 2)],
            "gross_return": [0.0],
            "cost": [0.0],
            "borrow_cost": [0.0],
            "net_return": [0.0],
            "nav": [1.0],
            "cash_weight": [1.0],
            "turnover": [0.0],
        }
    )
    bt = StrategyBacktestResult(
        factor_name="momentum_20d",
        strategy_name="quantile_long_short",
        n_groups=5,
        returns=returns,
        nav=returns.select(
            [
                "trade_date",
                "gross_return",
                "cost",
                "borrow_cost",
                "net_return",
                "nav",
                "cash_weight",
            ]
        ),
        positions=pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "ts_code": pl.Utf8,
                "weight": pl.Float64,
                "market_value": pl.Float64,
            }
        ),
        trades=pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "ts_code": pl.Utf8,
                "prev_weight": pl.Float64,
                "target_weight": pl.Float64,
                "filled_delta_weight": pl.Float64,
                "turnover": pl.Float64,
                "cost": pl.Float64,
                "block_reason": pl.Utf8,
            }
        ),
        summary_stats={"portfolio": {"sharpe": 0.0}},
        config={},
    )
    to = TurnoverResult(
        factor_name="momentum_20d",
        avg_turnover=0.1,
        daily_turnover=pl.DataFrame(
            {"trade_date": [date(2020, 1, 2)], "turnover": [0.1]}
        ),
        migration_matrix=pl.DataFrame({"from": [0], "to": [1], "count": [1]}),
    )
    persist._save_results(
        run_dir,
        "momentum_20d",
        "20200101",
        "20200105",
        subset,
        ic,
        bt,
        to,
    )
    assert panel_path.read_bytes() == before
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["store_panel"] == str(panel_path)


def test_head_extend_computes_down_to_target_start(patched_ensure):
    """回归 BUG-1a：补头下界=target_start（=min(start, STORE 起点)），meta 声称须真实。"""
    from factorzen.discovery.factor_store import finalize_factor_panel
    from factorzen.pipelines.factor_panel_cache import ensure_factor_store_panel

    store = patched_ensure
    factor = _stub_factor()
    asset = store / "ashare" / factor.name
    asset.mkdir(parents=True)
    dates = [date(2020, 1, 10) + timedelta(days=i) for i in range(5)]
    finalize_factor_panel(
        _panel(dates, ["000001.SZ", "000002.SZ"]).drop("factor_clean")
    ).write_parquet(asset / "factor.parquet")
    (asset / "meta.json").write_text(
        json.dumps(
            {
                "name": factor.name,
                "expression": "rank(close)",
                "materialization": {
                    "start": "2020-01-10",
                    "end": "2020-01-20",
                    "universe": "all_a",
                },
            }
        ),
        encoding="utf-8",
    )

    out = ensure_factor_store_panel(factor, "20200105", "20200112", root=str(store))
    assert out is not None
    # 头段从 target_start（STORE 起点 2020-01-01）算起，而非请求 start
    assert factor.calls[0][0].replace("-", "")[:8] == "20200101"
    meta = json.loads((asset / "meta.json").read_text(encoding="utf-8"))
    assert meta["materialization"]["start"] == "2020-01-01"
    # 声称即事实：再请求更早窗直接命中且含 01-05 行
    n_calls = len(factor.calls)
    out2 = ensure_factor_store_panel(factor, "20200101", "20200112", root=str(store))
    assert len(factor.calls) == n_calls, "已覆盖窗不得重算"
    assert out2.filter(pl.col("trade_date") == date(2020, 1, 5)).height > 0


def test_tail_only_extend_does_not_overclaim_start(patched_ensure):
    """回归 BUG-1b：只补尾时 meta.start 保持 cover_start，不得虚标到 2020-01-01。"""
    from factorzen.discovery.factor_store import finalize_factor_panel
    from factorzen.pipelines.factor_panel_cache import ensure_factor_store_panel

    store = patched_ensure
    factor = _stub_factor()
    asset = store / "ashare" / factor.name
    asset.mkdir(parents=True)
    dates = [date(2020, 1, 10) + timedelta(days=i) for i in range(5)]
    finalize_factor_panel(
        _panel(dates, ["000001.SZ", "000002.SZ"]).drop("factor_clean")
    ).write_parquet(asset / "factor.parquet")
    (asset / "meta.json").write_text(
        json.dumps(
            {
                "name": factor.name,
                "expression": "rank(close)",
                "materialization": {
                    "start": "2020-01-10",
                    "end": "2020-01-14",
                    "universe": "all_a",
                },
            }
        ),
        encoding="utf-8",
    )

    out = ensure_factor_store_panel(factor, "20200110", "20200118", root=str(store))
    assert out is not None
    assert len(factor.calls) == 1  # 只有尾段
    meta = json.loads((asset / "meta.json").read_text(encoding="utf-8"))
    assert meta["materialization"]["start"] == "2020-01-10", "未算头段不得声称更早覆盖"
    assert meta["materialization"]["end"] == "2020-01-18"


def test_python_factor_source_hash_invalidation(patched_ensure):
    """回归 BUG-3：meta 记过 source_hash 且与当前实现不一致 → 整窗重算。"""
    from factorzen.discovery.factor_store import finalize_factor_panel
    from factorzen.pipelines.factor_panel_cache import ensure_factor_store_panel

    store = patched_ensure
    factor = _stub_factor(expression=None)  # python 型：无 expression
    asset = store / "ashare" / factor.name
    asset.mkdir(parents=True)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(20)]
    finalize_factor_panel(
        _panel(dates, ["000001.SZ", "000002.SZ"]).drop("factor_clean")
    ).write_parquet(asset / "factor.parquet")

    def _meta(source_hash):
        mat = {"start": "2020-01-01", "end": "2020-01-20", "universe": "all_a"}
        if source_hash is not None:
            mat["source_hash"] = source_hash
        return {"name": factor.name, "kind": "python", "materialization": mat}

    # 未记 hash：不失效（向后兼容旧 meta）
    (asset / "meta.json").write_text(json.dumps(_meta(None)), encoding="utf-8")
    ensure_factor_store_panel(factor, "20200105", "20200110", root=str(store))
    assert factor.calls == [], "无 source_hash 记录不应触发重算"

    # 记过不同 hash：判过期 → 整窗重算，且新 meta 记录当前 hash
    (asset / "meta.json").write_text(json.dumps(_meta("deadbeef")), encoding="utf-8")
    out = ensure_factor_store_panel(factor, "20200105", "20200110", root=str(store))
    assert out is not None
    assert len(factor.calls) == 1, "source_hash 不一致必须整窗重算"
    meta = json.loads((asset / "meta.json").read_text(encoding="utf-8"))
    new_hash = meta["materialization"]["source_hash"]
    assert new_hash and new_hash != "deadbeef"


def test_is_daily_frequency_gate():
    """回归 BUG-2：weekly/monthly（args 或因子）一律不走面板缓存。"""
    from types import SimpleNamespace

    from factorzen.pipelines.factor_panel_cache import is_daily_frequency

    daily_f = SimpleNamespace(frequency="daily")
    weekly_f = SimpleNamespace(frequency="weekly")
    assert is_daily_frequency(SimpleNamespace(frequency="daily"), daily_f)
    assert not is_daily_frequency(SimpleNamespace(frequency="weekly"), daily_f)
    assert not is_daily_frequency(SimpleNamespace(frequency="daily"), weekly_f)
    assert not is_daily_frequency(SimpleNamespace(frequency="monthly"), daily_f)
    # 缺省视为 daily
    assert is_daily_frequency(SimpleNamespace(), SimpleNamespace())
