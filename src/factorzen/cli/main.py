"""Unified FactorZen command line interface."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import sys
from typing import TYPE_CHECKING, Any

from factorzen.config.settings import (
    FACTOR_EVALUATIONS_DIR,
    FACTOR_LIBRARY_DIR,
    MINE_TEAM_DIR,
    PORTFOLIOS_DIR,
    REPORTS_DIR,
    ROOT,
    SIM_DIR,
)
from factorzen.experiments.run_paths import run_dir

if TYPE_CHECKING:  # 本模块 Path 一律函数内导入（保持 CLI 启动开销）；此处仅供注解
    from pathlib import Path


# ── mine / lift-test 的 --set 通配覆盖 ────────────────────────────────────────
# 被砍 CLI 旗标经 --set KEY=VALUE 仍可达；未知 KEY fail-loudly 列出合法键。

_BOOL_TRUE = frozenset({"1", "true", "yes", "on"})
_BOOL_FALSE = frozenset({"0", "false", "no", "off", ""})


def _coerce_set_value(raw: str, typ: type, choices: tuple[str, ...] | None = None) -> Any:
    s = raw.strip()
    if typ is bool:
        low = s.lower()
        if low in _BOOL_TRUE:
            return True
        if low in _BOOL_FALSE:
            return False
        raise ValueError(f"期望 bool，收到 {raw!r}（可用 true/false/1/0）")
    if typ is int:
        if s.lower() in ("none", "null"):
            return None
        return int(s)
    if typ is float:
        if s.lower() in ("none", "null"):
            return None
        return float(s)
    # str
    if s.lower() in ("none", "null") and choices is None:
        return None
    if choices is not None and s not in choices:
        raise ValueError(f"期望 {{{', '.join(choices)}}}，收到 {raw!r}")
    return s


# key → (type, default, choices|None)
_MINE_SEARCH_SET: dict[str, tuple[type, Any, tuple[str, ...] | None]] = {
    "workers": (int, 1, None),
    "holdout_ratio": (float, 0.2, None),
    "train_ratio": (float, 0.7, None),
    "decorr_threshold": (float, 0.7, None),
    "min_n_train": (int, 5, None),
    "dsr_alpha": (float, 0.1, None),
    "no_library": (bool, False, None),
    "no_library_orthogonal": (bool, False, None),
    "objective": (str, "residual", ("raw", "residual")),
    "intraday_leaves": (bool, False, None),
    "intraday_freq": (str, "5min", None),
}

_MINE_AGENT_SET: dict[str, tuple[type, Any, tuple[str, ...] | None]] = {
    "patience": (int, None, None),
    "heal_rounds": (int, 2, None),
    "no_library_orthogonal": (bool, False, None),
    "objective": (str, "residual", ("raw", "residual")),
    "intraday_leaves": (bool, False, None),
    "intraday_freq": (str, "5min", None),
    "intraday_scout": (bool, False, None),
    "scout_k": (int, 4, None),
    "scout_max_leaves": (int, 12, None),
}

_MINE_TEAM_SET: dict[str, tuple[type, Any, tuple[str, ...] | None]] = {
    "index_path": (str, str(MINE_TEAM_DIR / "experiment_index.jsonl"), None),
    "patience": (int, None, None),
    "heal_rounds": (int, 2, None),
    "hypotheses_per_round": (int, 1, None),
    "no_library": (bool, False, None),
    "no_library_orthogonal": (bool, False, None),
    "objective": (str, "residual", ("raw", "residual")),
    "no_campaign_prior": (bool, False, None),
    "llm_workers": (int, 4, None),
    "no_auto_lift": (bool, False, None),
    "no_sleeve_gate": (bool, False, None),
    "lift_se_mult": (float, 1.0, None),
    "lift_workers": (int, None, None),
    "intraday_leaves": (bool, False, None),
    "intraday_freq": (str, "5min", None),
    "intraday_scout": (bool, False, None),
    "scout_k": (int, 4, None),
    "scout_max_leaves": (int, 12, None),
}

_LIFT_TEST_SET: dict[str, tuple[type, Any, tuple[str, ...] | None]] = {
    "top_m": (int, 20, None),
    "queue_ic_floor": (float, None, None),
    "include_sub_floor": (bool, False, None),
    "threshold": (float, None, None),
    "library_root": (str, None, None),
    "se_mult": (float, 1.0, None),
    "allow_active": (bool, False, None),
    "horizon": (int, None, None),
    "lift_workers": (int, None, None),
    "intraday_leaves": (bool, False, None),
    "intraday_freq": (str, "5min", None),
}

# strategies run：策略参数经 --set 通配；缺省在 schema 默认 / handler 按 name 补齐。
_STRATEGIES_RUN_SET: dict[str, tuple[type, Any, tuple[str, ...] | None]] = {
    "ma_window": (int, 200, None),
    "top_n": (int, None, None),  # trend/momentum 默认 50；sleeve 默认 200
    "index_code": (str, "000300.SH", None),
    "timing": (bool, True, None),
    "lookback": (int, 126, None),
    "index_codes": (str, "000300.SH,000905.SH,000852.SH", None),
    "rebalance": (str, "monthly", ("monthly", "weekly", "daily")),
    "score_col": (str, None, None),
    "scores": (str, None, None),
    "holding_days": (int, 10, None),
    "direction": (str, "top", ("top", "bottom")),
    "n_groups": (int, 5, None),
    "group": (int, 1, None),
}


def _apply_set_overrides(
    args: argparse.Namespace,
    schema: dict[str, tuple[type, Any, tuple[str, ...] | None]],
) -> int | None:
    """把 ``--set KEY=VALUE`` 注入 args；未知键 fail-loudly。成功返回 None，失败返回 exit code。"""
    for key, (_typ, default, _choices) in schema.items():
        if not hasattr(args, key):
            setattr(args, key, default)

    overrides = getattr(args, "set_overrides", None) or []
    for item in overrides:
        if "=" not in item:
            print(f"--set 需要 KEY=VALUE 形式，收到: {item!r}", file=sys.stderr)
            return 2
        key, raw = item.split("=", 1)
        key = key.strip().replace("-", "_")
        if key not in schema:
            legal = ", ".join(sorted(schema))
            print(f"--set 未知键 {key!r}；合法键: {legal}", file=sys.stderr)
            return 2
        typ, _default, choices = schema[key]
        try:
            val = _coerce_set_value(raw, typ, choices)
        except ValueError as exc:
            print(f"--set {key}: {exc}", file=sys.stderr)
            return 2
        # patience 须 >=1（与旧 _positive_patience 同口径）
        if key == "patience" and val is not None and int(val) < 1:
            print(f"--set patience: 须 >=1，收到 {val!r}", file=sys.stderr)
            return 2
        setattr(args, key, val)
    return None


def _factor_template(class_name: str, factor_name: str, frequency: str) -> str:
    base = "DailyFactor" if frequency != "intraday" else "IntradayFactor"
    import_path = (
        "factorzen.daily.factors.base"
        if frequency != "intraday"
        else "factorzen.intraday.factors.base"
    )
    context_type = "FactorDataContext" if frequency != "intraday" else "IntradayDataContext"
    context_import = (
        "factorzen.daily.data.context"
        if frequency != "intraday"
        else "factorzen.intraday.data.context"
    )
    time_col = "trade_date" if frequency != "intraday" else "trade_time"
    source = "ctx.daily" if frequency != "intraday" else "ctx.minute"
    return f'''"""User factor: {factor_name}."""

import polars as pl

from {context_import} import {context_type}
from {import_path} import {base}


class {class_name}({base}):
    name = "{factor_name}"
    frequency = "{frequency}"
    description = "{factor_name}"

    def compute(self, ctx: {context_type}) -> pl.DataFrame:
        frame = {source}
        return (
            frame.select(["{time_col}", "ts_code"])
            .with_columns(pl.lit(None, dtype=pl.Float64).alias("factor_value"))
            .collect()
        )
'''


def _class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.replace("-", "_").split("_")) + "Factor"


def _cmd_factor_new(args: argparse.Namespace) -> int:
    """脚手架：写 factor_store 三件套中的 factor.py + 最小 meta.json。"""
    import json
    from datetime import date

    market = getattr(args, "market", None) or "ashare"
    asset_dir = ROOT / "workspace" / "factor_store" / market / args.name
    target = asset_dir / "factor.py"
    meta_path = asset_dir / "meta.json"
    if target.exists() and not args.force:
        print(f"Factor already exists: {target}", file=sys.stderr)
        return 2
    asset_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(
        _factor_template(_class_name(args.name), args.name, args.freq),
        encoding="utf-8",
    )
    if not meta_path.exists() or args.force:
        meta = {
            "name": args.name,
            "kind": "python",
            "expression": f"py::{args.name}",
            "frequency": args.freq,
            "description": args.name,
            "source_run_id": None,
            "created_at": date.today().isoformat(),
            "ledger_snapshot": {
                "status": None,
                "lift": None,
                "admission_ic": None,
                "ic_train": None,
                "holdout_ic": None,
                "truth": f"workspace/factor_library/{market}.jsonl",
            },
            "materialization": None,
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(target)
    return 0


def _cmd_factor_list(args: argparse.Namespace) -> int:
    if args.freq == "intraday":
        from factorzen.intraday.factors.registry import list_factors
    else:
        from factorzen.daily.factors.registry import list_factors
        from factorzen.discovery.library_provider import load_library_factors

        # 注入 factor_library expression 型；库损坏/缺失不崩 list
        try:
            load_library_factors()
        except ValueError as e:
            print(f"[factor] load_library_factors 跳过: {e}", file=sys.stderr)

    for name in list_factors():
        print(name)
    return 0


def _forward_factor_track(args: argparse.Namespace, *, track: str) -> int:
    """将 CLI 参数转发到 ``daily_single.main(track=...)``。"""
    from factorzen.pipelines import daily_single

    forwarded = [f"fz factor {args.factor_command}"]
    if args.name:
        forwarded.extend(["--factor", args.name])
    if args.start:
        forwarded.extend(["--start", args.start])
    if args.end:
        forwarded.extend(["--end", args.end])
    if args.universe:
        forwarded.extend(["--universe", args.universe])
    forwarded.extend(["--frequency", args.frequency])
    if args.config:
        forwarded.extend(["--config", args.config])
    if args.seed is not None:
        forwarded.extend(["--seed", str(args.seed)])
    if args.benchmark:
        forwarded.extend(["--benchmark", args.benchmark])
    if args.dry_run:
        forwarded.append("--dry-run")
    for override in getattr(args, "set_overrides", None) or []:
        forwarded.extend(["--set", override])
    # 成交口径：CLI 默认可实现 (1 / open_adj)；显式 --exec-lag 0 回旧口径对照
    if getattr(args, "exec_lag", None) is not None:
        forwarded.extend(["--exec-lag", str(int(args.exec_lag))])
    if getattr(args, "exec_price_col", None) is not None:
        forwarded.extend(["--exec-price-col", str(args.exec_price_col)])
    # 信号轨专属旋钮(仅 eval 子命令定义,backtest 轨 args 上不存在)。
    # 注意信号轨刻意**不提供成本参数**——它是纯毛口径,成本走交易轨。
    if getattr(args, "n_groups", None) is not None:
        forwarded.extend(["--n-groups", str(int(args.n_groups))])

    old_argv = sys.argv
    try:
        sys.argv = forwarded
        daily_single.main(track=track)
    finally:
        sys.argv = old_argv
    return 0


def _cmd_factor_eval(args: argparse.Namespace) -> int:
    """因子研究评估（信号层，毛口径）。"""
    return _forward_factor_track(args, track="eval")


def _cmd_factor_backtest(args: argparse.Namespace) -> int:
    """模拟交易回测（日环撮合，净口径）。"""
    return _forward_factor_track(args, track="backtest")


def _cmd_factor_sweep(args: argparse.Namespace) -> int:
    from datetime import datetime

    from factorzen.config.settings import FACTOR_EVALUATIONS_DIR
    from factorzen.pipelines.factor_sweep import (
        format_sweep_csv,
        format_sweep_table,
        pipeline_runner,
        run_sweep,
    )

    factor = args.name
    start, end, universe = args.start, args.end, args.universe
    if args.config:
        from factorzen.config.research import load_run_config

        cfg = load_run_config(args.config)
        factor = factor or cfg.factor
        start = start or cfg.start
        end = end or cfg.end
        universe = universe or cfg.universe

    if not (factor and start and end):
        print("sweep 需要 factor 与 start/end（经位置参数/--config/CLI 提供）", file=sys.stderr)
        return 2
    if not args.grid:
        print("sweep 需要至少一个 --grid key=v1,v2,...", file=sys.stderr)
        return 2

    runner = pipeline_runner(
        factor=factor,
        start=start,
        end=end,
        config_path=args.config,
        universe=universe,
    )
    rows = run_sweep(
        args.grid,
        runner,
        sort_by=args.sort_by,
        extra_overrides=args.set_overrides,
    )
    print(format_sweep_table(rows))

    out_dir = FACTOR_EVALUATIONS_DIR / f"sweep_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep_results.csv"
    csv_path.write_text(format_sweep_csv(rows), encoding="utf-8")
    print(f"\n结果已保存: {csv_path}")
    return 0


def _cmd_runs_path(args: argparse.Namespace) -> int:
    report = run_dir(args.run_id) / "report.html"
    if not report.exists():
        print(f"Report not found: {report}", file=sys.stderr)
        return 2
    print(report)
    return 0


def _cmd_data_fetch(args: argparse.Namespace) -> int:
    from factorzen.core import loader

    if args.data_type == "daily":
        frame = loader.fetch_daily(args.start, args.end)
    elif args.data_type == "fundamentals":
        # fina_indicator 全套质量/成长字段 → finance_fina_indicator 分区（按公告日 PIT 对齐用）
        from factorzen.discovery.operators import FUNDAMENTAL_FEATURES

        fields = "ts_code,ann_date,end_date," + ",".join(sorted(FUNDAMENTAL_FEATURES))
        frame = loader.fetch_finance("fina_indicator", args.start, args.end, fields=fields)
    elif args.data_type == "flows":
        # 资金流(moneyflow) + 北向持股(hk_hold)，日频 point-in-time，供 net_mf_amount/north_ratio 叶子
        mf = loader.fetch_moneyflow(args.start, args.end)
        hk = loader.fetch_hk_hold(args.start, args.end)
        print(f"moneyflow: {len(mf)} rows | hk_hold: {len(hk)} rows")
        return 0
    elif args.data_type == "margin_detail":
        # 两融明细(margin_detail)，日频；T+1 披露 lag 在 attach 层完成
        frame = loader.fetch_margin_detail(args.start, args.end)
    elif args.data_type == "stk_holdernumber":
        # 股东户数，低频；ann_date PIT 对齐在 attach_holders
        frame = loader.fetch_stk_holdernumber(args.start, args.end)
    elif args.data_type == "top_list":
        # 龙虎榜，日频事件；盘后披露 lag + 已知日未上榜 fill 0（未拉取=null）在 attach 层完成
        frame = loader.fetch_top_list(args.start, args.end)
    else:
        frame = loader.fetch_daily_basic(args.start, args.end)
    rows = len(frame) if hasattr(frame, "__len__") else "unknown"
    print(f"{args.data_type}: {rows} rows")
    return 0


def _cmd_data_crypto_backfill(args: argparse.Namespace) -> int:
    from factorzen.markets.crypto import vision
    from factorzen.markets.crypto.lake import CryptoLake, month_range

    lake = CryptoLake(args.lake_root)
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        all_syms = vision.list_um_symbols()
        rank_month = vision._prev_month(month_range(args.end, args.end)[0])
        symbols = vision.rank_symbols_by_amount(all_syms, rank_month, args.top_n)
        print(f"[backfill] Top-{args.top_n} by {rank_month} 成交额: {symbols[:5]}...")
    manifest = vision.backfill(lake, symbols, args.start, args.end)
    gaps = manifest["gaps"]
    n_gaps = len(gaps) if isinstance(gaps, list) else 0
    print(f"[backfill] 完成: {len(symbols)} 标的 → {lake.root} (gaps={n_gaps})")
    return 0


def _cmd_data_intraday_features_build(args: argparse.Namespace) -> int:
    """物化日内特征面板：``fz data intraday-features build``。"""
    from factorzen.intraday.features.engine import build_intraday_features

    codes = None
    if getattr(args, "codes", None):
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    try:
        report = build_intraday_features(
            args.start,
            args.end,
            freq=args.freq,
            version=args.version,
            codes=codes,
            overwrite=bool(getattr(args, "overwrite", False)),
            force=bool(getattr(args, "force", False)),
            workers=int(getattr(args, "workers", 1) or 1),
        )
    except Exception as exc:
        print(f"[intraday-features] build 失败: {exc}", file=sys.stderr)
        return 1
    print(
        f"[intraday-features] build 完成: months={report.months} "
        f"rows={report.rows} n_stocks={report.n_stocks} "
        f"manifest={report.manifest_path}"
    )
    return 0


def _cmd_data_intraday_features_status(args: argparse.Namespace) -> int:
    """查看日内特征 manifest 与分区：``fz data intraday-features status``。"""
    from factorzen.config.settings import INTRADAY_FEATURES_DIR
    from factorzen.core.storage import partition_exists
    from factorzen.intraday.features.engine import read_manifest
    from factorzen.intraday.sessions import normalize_freq

    freq = normalize_freq(args.freq)
    version = args.version
    manifest = read_manifest(version=version, freq=freq, base_dir=INTRADAY_FEATURES_DIR)
    if manifest is None:
        print(
            f"[intraday-features] 无 manifest（version={version} freq={freq}），"
            "请先运行: fz data intraday-features build --start ... --end ...",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    cov = manifest.get("coverage") or {}
    months = list(cov.get("months") or [])
    mld = cov.get("month_last_date") or {}
    data_type = f"{version}/{freq}"
    # last_date 必须显示：只有它能区分「整月已算」与「算了前 10 天」，
    # 光看 partition_exists=True 会把部分月读成完整月（2026-07-19 实际踩过）
    print("\nmonth\tpartition_exists\tlast_date")
    for ym in months:
        y_str, m_str = ym.split("-")
        y, m = int(y_str), int(m_str)
        ok = partition_exists(data_type, y, m, base_dir=INTRADAY_FEATURES_DIR)
        print(f"{ym}\t{ok}\t{mld.get(ym) or '-'}")
    return 0


def _cmd_config_validate(args: argparse.Namespace) -> int:
    from factorzen.config.research import default_benchmark_for_universe, load_run_config

    config = load_run_config(args.path)
    benchmark = config.benchmark or default_benchmark_for_universe(config.universe)
    effective = config.model_copy(update={"benchmark": benchmark})
    payload = {
        "config": effective.model_dump(),
        "output_dir": (ROOT / "workspace" / "factor_evaluations" / "<run_id>").as_posix(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_runs_list(args: argparse.Namespace) -> int:
    index_path = FACTOR_EVALUATIONS_DIR / "experiment_index.jsonl"
    if not index_path.exists():
        print(f"No runs index found: {index_path}", file=sys.stderr)
        return 2

    rows: list[dict[str, object]] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    if args.limit:
        rows = rows[-args.limit :]

    print("run_id\tstatus\tfactor\tuniverse\ttimestamp")
    for row in rows:
        print(
            "\t".join(
                str(row.get(key, ""))
                for key in ("run_id", "status", "factor", "universe", "timestamp")
            )
        )
    return 0


def _mine_search_crypto(args: argparse.Namespace) -> int:
    """crypto perps 挖掘（live CCXT）：universe 快照 → run_crypto_mining。"""
    from factorzen.markets.crypto.mining import run_crypto_mining
    from factorzen.markets.crypto.profile import build_crypto_profile

    profile = build_crypto_profile(top_n=args.top_n)
    symbols = profile.universe.snapshot(args.end)
    if not symbols:
        print("[mine] crypto universe 为空（检查网络/交易所可用性）", file=sys.stderr)
        return 1
    res = run_crypto_mining(
        profile,
        symbols,
        args.start,
        args.end,
        n_trials=args.trials,
        top_k=args.top_k,
        seed=args.seed,
        method=args.method,
        freq=args.freq,
        # 六个护栏/并行参数经 **session_kw 透传到 run_session，否则用户设的
        # --dsr-alpha/--holdout-ratio/--workers 等被静默丢弃、按默认执行。
        holdout_ratio=args.holdout_ratio,
        train_ratio=args.train_ratio,
        decorr_threshold=args.decorr_threshold,
        min_n_train=args.min_n_train,
        dsr_alpha=args.dsr_alpha,
        workers=args.workers,
        library_orthogonal=not getattr(args, "no_library_orthogonal", False),
        objective=getattr(args, "objective", "residual"),
    )
    sd = res["session_dir"]
    print(f"[mine] crypto 完成：{len(res['candidates'])} 个候选 / {len(symbols)} 标的 → {sd}")
    return 0


def _mine_search_futures(args: argparse.Namespace) -> int:
    """商品期货挖掘（M1，Tushare fut_daily 主力连续后复权）：universe 快照 → run_futures_mining。"""
    from factorzen.markets.futures.mining import run_futures_mining
    from factorzen.markets.futures.profile import build_futures_profile

    profile = build_futures_profile(top_n=args.top_n)
    symbols = profile.universe.snapshot(args.end)
    if not symbols:
        print("[mine] futures universe 为空（检查 Tushare 权限/数据覆盖）", file=sys.stderr)
        return 1
    res = run_futures_mining(
        profile,
        symbols,
        args.start,
        args.end,
        n_trials=args.trials,
        top_k=args.top_k,
        seed=args.seed,
        method=args.method,
        holdout_ratio=args.holdout_ratio,
        train_ratio=args.train_ratio,
        decorr_threshold=args.decorr_threshold,
        min_n_train=args.min_n_train,
        dsr_alpha=args.dsr_alpha,
        workers=args.workers,
        library_orthogonal=not getattr(args, "no_library_orthogonal", False),
        objective=getattr(args, "objective", "residual"),
    )
    sd = res["session_dir"]
    print(f"[mine] futures 完成：{len(res['candidates'])} 个候选 / {len(symbols)} 品种 → {sd}")
    return 0


def _mine_search_us(args: argparse.Namespace) -> int:
    """美股挖掘（M1，Yahoo chart 后复权日线）：静态 S&P500 快照 → run_us_mining。"""
    from factorzen.markets.us.mining import run_us_mining
    from factorzen.markets.us.profile import build_us_profile

    profile = build_us_profile(top_n=args.top_n)
    symbols = profile.universe.snapshot(args.end)
    if not symbols:
        print("[mine] us universe 为空（检查 sp500 快照）", file=sys.stderr)
        return 1
    res = run_us_mining(
        profile,
        symbols,
        args.start,
        args.end,
        n_trials=args.trials,
        top_k=args.top_k,
        seed=args.seed,
        method=args.method,
        holdout_ratio=args.holdout_ratio,
        train_ratio=args.train_ratio,
        decorr_threshold=args.decorr_threshold,
        min_n_train=args.min_n_train,
        dsr_alpha=args.dsr_alpha,
        workers=args.workers,
        update_library=not getattr(args, "no_library", False),
        library_orthogonal=not getattr(args, "no_library_orthogonal", False),
        objective=getattr(args, "objective", "residual"),
    )
    sd = res["session_dir"]
    print(f"[mine] us 完成：{len(res['candidates'])} 个候选 / {len(symbols)} 标的 → {sd}")
    return 0


def _cmd_mine_search(args: argparse.Namespace) -> int:
    err = _apply_set_overrides(args, _MINE_SEARCH_SET)
    if err is not None:
        return err
    if (
        getattr(args, "market", "ashare") not in ("crypto",)
        and getattr(args, "freq", "daily") != "daily"
    ):
        print("[mine] --freq 仅 crypto 支持;ashare/futures/us 只有 daily", file=sys.stderr)
        return 2
    # --intraday-leaves 仅 ashare（在 market 分流前拦截）
    if getattr(args, "intraday_leaves", False) and getattr(args, "market", "ashare") != "ashare":
        print("[mine] --intraday-leaves 仅 ashare 支持", file=sys.stderr)
        return 2
    if getattr(args, "market", "ashare") == "crypto":
        return _mine_search_crypto(args)
    if getattr(args, "market", "ashare") == "futures":
        return _mine_search_futures(args)
    if getattr(args, "market", "ashare") == "us":
        return _mine_search_us(args)
    from factorzen.pipelines.factor_mine import run_mine

    res = run_mine(
        start=args.start,
        end=args.end,
        universe=args.universe,
        n_trials=args.trials,
        top_k=args.top_k,
        seed=args.seed,
        method=args.method,
        holdout_ratio=args.holdout_ratio,
        train_ratio=args.train_ratio,
        decorr_threshold=args.decorr_threshold,
        min_n_train=args.min_n_train,
        dsr_alpha=args.dsr_alpha,
        workers=args.workers,
        update_library=not getattr(args, "no_library", False),
        library_orthogonal=not getattr(args, "no_library_orthogonal", False),
        objective=getattr(args, "objective", "residual"),
        intraday=bool(getattr(args, "intraday_leaves", False)),
        intraday_freq=getattr(args, "intraday_freq", "5min") or "5min",
        intraday_expr_leaves=getattr(args, "intraday_expr_leaves", None),
        exec_lag=int(
            getattr(args, "exec_lag", 1) if getattr(args, "exec_lag", None) is not None else 1
        ),
        exec_price_col=getattr(args, "exec_price_col", "open_adj"),
    )
    sd = res["session_dir"]
    print(f"[mine] 完成：{len(res['candidates'])} 个候选 → {sd}")
    print(
        "[mine] 复现：入库候选 fz factor-library list 查 name 后 "
        "fz factor eval <name> --set preprocessing.neutralize=false；"
        "未入库候选：表达式在 candidates.csv"
    )
    print(
        "[mine] 注：candidates.csv 的 IC 为挖掘内估计(plain zscore)；"
        "fz factor eval 默认带中性化，IC parity 需 neutralize=false"
    )
    return 0


def _cmd_research_run(args: argparse.Namespace) -> int:
    from factorzen.pipelines.research_run import run_research

    res = run_research(
        start=args.start,
        end=args.end,
        universe=args.universe,
        n_trials=args.trials,
        method=args.method,
        seed=args.seed,
        top_k=args.top_k,
        rebalance_days=args.rebalance_days,
        warmup=args.warmup,
        risk_aversion=args.lam,
        w_max=args.w_max,
        turnover=args.turnover,
        industry_neutral=args.industry_neutral,
        lookback=args.lookback,
        run_id=args.run_id,
        command=["research", "run"],
        intraday=bool(getattr(args, "intraday_leaves", False)),
        intraday_freq=getattr(args, "intraday_freq", "5min") or "5min",
        exec_lag=int(args.exec_lag) if getattr(args, "exec_lag", None) is not None else 1,
        exec_price_col=getattr(args, "exec_price_col", "open_adj"),
    )
    print(f"[research] 完成 run_id={res['run_id']} 因子={res['expression']!r}")
    print(
        f"[research] 调仓 {res['n_rebalances']} 次 · sharpe={res['sharpe']} · ann_ret={res['ann_ret']}"
    )
    print(f"[research] mining={res['mining_session_dir']}")
    print(f"[research] portfolios={res['portfolios_root']}  sim={res['sim_dir']}")
    print(f"[research] dashboard → {res['report_html']}")
    return 0


def _positive_patience(raw: str) -> int:
    """`--patience` 必须 >= 1。

    早停判据是 `no_improve >= patience`；patience=0 时它在第 2 轮开头恒成立——**即使刚产出
    新候选**——于是静默变成「只跑 1 轮」，无视 `--iterations`。而 help 文案说的是
    「连续 N 轮无新候选则早停」，用户传 0 期望「不早停/更激进」，得到的却相反。
    不早停请省略该参数（默认 None）。
    """
    n = int(raw)
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"patience 必须 >= 1（实得 {n}）；0/负数会让循环在第 2 轮无条件早停。"
            "不早停请省略 --patience。"
        )
    return n


def _data_window(args: argparse.Namespace) -> dict:
    """挖掘产物的数据窗口指纹，落进 manifest 的 params（铁律#3：可复现）。"""
    return {
        "start": args.start,
        "end": args.end,
        "universe": args.universe,
        "market": getattr(args, "market", "ashare"),
    }


def _command_line(args: argparse.Namespace) -> str:
    """触发本次运行的命令行（由 main() 从实际 argv 组装，非 sys.argv）。"""
    return getattr(args, "command_line", "")


def _membership_prep_meta_empty(universe: str | None = None) -> dict:
    """非 A 股 / 未走 prepare_mining_daily 时的 membership 溯源占位（mode=None）。"""
    return {
        "membership_mode": None,
        "membership_hash": None,
        "membership_n_rows": None,
        "universe": universe,
    }


def _data_window_with_membership(args: argparse.Namespace, prep_meta: dict) -> dict:
    """data_window + membership_* 三字段（与 start/end/universe 平级并入 params）。

    若 prep_meta 含 ``intraday_panel`` 溯源，一并写入（--intraday-leaves 路径）。
    """
    out = {
        **_data_window(args),
        "membership_mode": prep_meta.get("membership_mode"),
        "membership_hash": prep_meta.get("membership_hash"),
        "membership_n_rows": prep_meta.get("membership_n_rows"),
    }
    if "intraday_panel" in prep_meta:
        out["intraday_panel"] = prep_meta["intraday_panel"]
    return out


def _prepare_agent_mining_data(args: argparse.Namespace):
    """按 market 装配含预热前缀的挖掘帧，返回 ``(daily, profile, prep_meta)``。

    - ashare：`prepare_mining_daily`（复权价 + daily_basic + 全叶子），profile=None（零回归）；
      ``prep_meta`` 含 ``membership_mode`` / ``membership_hash`` / ``membership_n_rows`` / ``universe``。
    - crypto：`build_crypto_daily`（Vision 湖），向前多拉 `AGENT_WARMUP_LOOKBACK` 自然日作预热前缀
      （crypto 24/7，1 bar≈1 自然日，与 A 股口径一致）；symbols 取 --symbols 或 universe Top-N；
      membership 不适用 → mode=None。

    daily 为空（crypto 湖无对应 symbol 数据）→ 返回 ``(None, profile, prep_meta)``，调用方报错退出。
    """

    from factorzen.config.constants import AGENT_WARMUP_LOOKBACK
    from factorzen.pipelines.factor_mine import prepare_mining_daily

    market = getattr(args, "market", "ashare")
    if market == "crypto":
        import datetime as _dt

        from factorzen.markets.crypto.mining import build_crypto_daily
        from factorzen.markets.crypto.profile import build_crypto_profile

        profile = build_crypto_profile(top_n=getattr(args, "top_n", 50))
        if getattr(args, "symbols", None):
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        else:
            symbols = profile.universe.snapshot(args.end)
        prep_meta = _membership_prep_meta_empty(getattr(args, "universe", None))
        if not symbols:
            return None, profile, prep_meta
        warmup_start = (
            _dt.datetime.strptime(args.start, "%Y%m%d").date()
            - _dt.timedelta(days=AGENT_WARMUP_LOOKBACK)
        ).strftime("%Y%m%d")
        freq = getattr(args, "freq", None) or profile.base_freq
        daily = build_crypto_daily(profile.provider, symbols, warmup_start, args.end, freq)
        return (None if daily.is_empty() else daily), profile, prep_meta
    if market == "futures":
        import datetime as _dt

        from factorzen.markets.futures.mining import build_futures_daily
        from factorzen.markets.futures.profile import build_futures_profile

        profile = build_futures_profile(top_n=getattr(args, "top_n", 40))
        if getattr(args, "symbols", None):
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        else:
            symbols = profile.universe.snapshot(args.end)
        prep_meta = _membership_prep_meta_empty(getattr(args, "universe", None))
        if not symbols:
            return None, profile, prep_meta
        # 预热前缀：AGENT_WARMUP_LOOKBACK 交易日 → 自然日近似（243 交易日/年，×1.55 覆盖节假日）。
        warmup_start = (
            _dt.datetime.strptime(args.start, "%Y%m%d").date()
            - _dt.timedelta(days=int(AGENT_WARMUP_LOOKBACK * 1.55))
        ).strftime("%Y%m%d")
        daily = build_futures_daily(profile.provider, symbols, warmup_start, args.end)
        return (None if daily.is_empty() else daily), profile, prep_meta
    if market == "us":
        import datetime as _dt

        from factorzen.markets.us.mining import build_us_daily
        from factorzen.markets.us.profile import build_us_profile

        profile = build_us_profile(top_n=getattr(args, "top_n", 50))
        if getattr(args, "symbols", None):
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        else:
            symbols = profile.universe.snapshot(args.end)
        prep_meta = _membership_prep_meta_empty(getattr(args, "universe", None))
        if not symbols:
            return None, profile, prep_meta
        # 预热前缀：AGENT_WARMUP_LOOKBACK 交易日 → 自然日近似（252 交易日/年，×1.5 覆盖周末/假日）。
        warmup_start = (
            _dt.datetime.strptime(args.start, "%Y%m%d").date()
            - _dt.timedelta(days=int(AGENT_WARMUP_LOOKBACK * 1.5))
        ).strftime("%Y%m%d")
        daily = build_us_daily(profile.provider, symbols, warmup_start, args.end)
        return (None if daily.is_empty() else daily), profile, prep_meta
    # A 股：预热前缀用 agent 专用加长值（LLM 窗口无搜索空间上界，长窗因子用 180 会被误判欠预热）。
    prep_meta = {}
    daily = prepare_mining_daily(
        args.start,
        args.end,
        args.universe,
        lookback_days=AGENT_WARMUP_LOOKBACK,
        out_meta=prep_meta,
        intraday=bool(getattr(args, "intraday_leaves", False)),
        intraday_freq=getattr(args, "intraday_freq", "5min") or "5min",
        intraday_expr_leaves=getattr(args, "intraday_expr_leaves", None),
    )
    if not prep_meta:
        # 替身实现可能不填 out_meta：补占位，调用方仍能稳定解包
        prep_meta = _membership_prep_meta_empty(getattr(args, "universe", None))
    return daily, None, prep_meta


def _cmd_mine_agent(args: argparse.Namespace) -> int:
    err = _apply_set_overrides(args, _MINE_AGENT_SET)
    if err is not None:
        return err
    if getattr(args, "market", "ashare") != "crypto" and getattr(args, "freq", "daily") != "daily":
        print("[mine] --freq 仅 crypto 支持;ashare/futures/us 只有 daily", file=sys.stderr)
        return 2
    # --intraday-scout 仅 ashare；隐含 --intraday-leaves（reference 需要 i_*）
    if getattr(args, "intraday_scout", False):
        if getattr(args, "market", "ashare") != "ashare":
            print("[mine] --intraday-scout 仅 ashare 支持", file=sys.stderr)
            return 2
        args.intraday_leaves = True
    from factorzen.pipelines.factor_mine_agent import run_agent_mine

    daily, profile, prep_meta = _prepare_agent_mining_data(args)
    if daily is None:
        print("[mine-agent] crypto 挖掘帧为空（检查 --symbols 或数据湖覆盖）", file=sys.stderr)
        return 1
    # eval_start = 挖掘窗口 start（预热前缀边界），与 M1 `run_mine(eval_start=start)` 同口径：
    # 缺了它预热前缀会被 split_holdout 当训练数据。
    # membership_* 并入 data_window → agent params（与 start/end/universe 平级，铁律#3）。
    res = run_agent_mine(
        daily,
        n_rounds=args.iterations,
        seed=args.seed,
        top_k=args.top_k,
        human_review=args.human_review,
        patience=args.patience,
        heal_rounds=args.heal_rounds,
        data_window=_data_window_with_membership(args, prep_meta),
        command=_command_line(args),
        eval_start=args.start,
        profile=profile,
        library_orthogonal=not getattr(args, "no_library_orthogonal", False),
        objective=getattr(args, "objective", "residual"),
        intraday_scout=bool(getattr(args, "intraday_scout", False)),
        scout_k=int(getattr(args, "scout_k", 4) or 4),
        scout_max_leaves=int(getattr(args, "scout_max_leaves", 12) or 12),
        scout_freq=getattr(args, "intraday_freq", "5min") or "5min",
        exec_lag=int(args.exec_lag) if getattr(args, "exec_lag", None) is not None else 1,
        exec_price_col=getattr(args, "exec_price_col", "open_adj"),
    )
    print(f"[mine-agent] 候选 {res['n_candidates']} 个 / N={res['n_trials']} → {res['run_dir']}")
    return 0


def _cmd_pool_prebuild(args: argparse.Namespace) -> int:
    """mine team 库池预构建（子进程入口）：prep → 剪叶 → build_library_pool → parquet。

    与 ``team_orchestrator.run_team_agent`` 的池前序列同源——改一侧必查另一侧。
    """
    from datetime import datetime
    from pathlib import Path

    from factorzen.agents.nodes import AgentContext
    from factorzen.core.experiment import get_git_sha
    from factorzen.discovery.evaluation import _preprocess_daily
    from factorzen.discovery.factor_library import (
        build_library_pool,
        library_file_hash,
        python_pool_cache_key,
        write_pool_cache,
    )
    from factorzen.discovery.leaf_health import (
        apply_leaf_exclusion,
        filter_leaves_by_holdout_coverage,
        log_excluded_leaves,
    )
    from factorzen.validation.holdout import holdout_boundary

    def _to_date(s: str):
        import datetime as _dt

        return _dt.datetime.strptime(s, "%Y%m%d").date()

    try:
        # 同源铁律：与 mine team / agent 同走 _prepare_agent_mining_data
        daily, profile, prep_meta = _prepare_agent_mining_data(args)
        if daily is None:
            print(
                "[pool-prebuild] 挖掘帧为空（检查 --symbols 或数据湖覆盖）",
                file=sys.stderr,
            )
            return 1

        # 与 run_team_agent 池前序列同源——改一侧必查另一侧
        session_prepped = _preprocess_daily(daily, profile)
        del daily  # 释放 raw（内存预算关键）

        # holdout 边界（与 run_team_agent 同一口径：先裁 eval_start 再 holdout_boundary）
        eval_start_date = _to_date(args.start)
        _dates_split = session_prepped["trade_date"]
        _dates_split = _dates_split.filter(_dates_split >= eval_start_date)
        holdout_start = holdout_boundary(
            sorted(_dates_split.unique().to_list()),
            float(getattr(args, "holdout_ratio", 0.2)),
        )
        del _dates_split

        # 剪叶（同 team 序）：AgentContext → filter → apply_leaf_exclusion
        ctx = AgentContext.from_profile(profile)
        _kept, excluded_leaves = filter_leaves_by_holdout_coverage(
            session_prepped,
            list(ctx.leaf_names),
            holdout_start,
            leaf_map=ctx.leaf_map,
        )
        log_excluded_leaves(excluded_leaves, prefix="pool-prebuild")
        ctx.leaf_names, ctx.leaf_map = apply_leaf_exclusion(
            list(ctx.leaf_names),
            ctx.leaf_map,
            excluded_leaves,
        )

        market = getattr(profile, "name", None) or getattr(args, "market", "ashare") or "ashare"
        lib_root = args.library_root or str(Path(args.index_path).parent / "factor_library")
        # 强制 compact（与父进程装载 CompactLibraryPool 契约一致）
        # universe：库含 python 记录时物化必需（与 run_team_agent 池调用同口径）
        _pool_universe = getattr(args, "universe", None)
        lib_pool = build_library_pool(
            market,
            session_prepped,
            ctx.leaf_map,
            root=lib_root,
            eval_start=eval_start_date,
            compact=True,
            universe=_pool_universe,
        )

        # meta 真实填：date 字段必须 str(date)=ISO（"2021-01-04"），与 load_pool_cache 校验同源
        out_dir = Path(args.out)
        write_pool_cache(
            lib_pool,
            out_dir,
            meta={
                "market": market,
                "statuses": ["active"],
                "eval_start": str(eval_start_date),
                "library_hash": library_file_hash(market, lib_root),
                "python_pool_key": python_pool_cache_key(
                    market,
                    root=lib_root,
                    statuses=("active",),
                    universe=_pool_universe,
                ),
                "prepped_height": session_prepped.height,
                "prepped_date_min": str(session_prepped["trade_date"].min()),
                "prepped_date_max": str(session_prepped["trade_date"].max()),
                "data_window": {
                    "start": args.start,
                    "end": args.end,
                    "universe": getattr(args, "universe", None),
                    "market": market,
                    "membership_hash": prep_meta.get("membership_hash"),
                },
                "git_sha": get_git_sha(),
                "created_at": datetime.now().isoformat(),
            },
        )
        n_factors = 0 if not lib_pool else len(lib_pool)
        print(
            f"[pool-prebuild] 完成 n_factors={n_factors} → {out_dir}",
            flush=True,
        )
        return 0
    except ValueError as exc:
        print(f"[pool-prebuild] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"[pool-prebuild] 失败 {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


def _cmd_mine_team(args: argparse.Namespace) -> int:
    err = _apply_set_overrides(args, _MINE_TEAM_SET)
    if err is not None:
        return err
    if getattr(args, "market", "ashare") != "crypto" and getattr(args, "freq", "daily") != "daily":
        print("[mine] --freq 仅 crypto 支持;ashare/futures/us 只有 daily", file=sys.stderr)
        return 2
    # --intraday-scout 仅 ashare；隐含 --intraday-leaves（reference 需要 i_*）
    if getattr(args, "intraday_scout", False):
        if getattr(args, "market", "ashare") != "ashare":
            print("[mine] --intraday-scout 仅 ashare 支持", file=sys.stderr)
            return 2
        args.intraday_leaves = True
    # ── 库池子进程预构建（父进程此刻未载大帧）────────────────────────────────
    # subprocess.run 全新解释器 exec；绝不用 multiprocessing/fork（polars 死锁风险）。
    import hashlib
    import os
    import subprocess
    from pathlib import Path

    import factorzen.pipelines.factor_mine_team as pmt
    from factorzen.config.settings import MINE_TEAM_DIR as _mine_team_dir
    from factorzen.discovery.factor_library import library_file_hash

    pool_cache_dir = None
    use_subproc = (
        bool(getattr(args, "pool_subproc", False))
        or os.environ.get("FACTORZEN_POOL_SUBPROC") == "1"
    )
    if use_subproc and not getattr(args, "no_library_orthogonal", False):
        # 与 run_team_mine 硬编码 library_root 同源（out_dir 默认 MINE_TEAM_DIR 的同级
        # factor_library=workspace/factor_library）。若用 index_path.parent 会落到
        # mine_team/factor_library，装载侧 hash 永不命中。
        # 函数内 from-import：测试可 monkeypatch settings.MINE_TEAM_DIR
        lib_root = str(Path(_mine_team_dir).parent / "factor_library")
        market = getattr(args, "market", "ashare") or "ashare"
        lib_hash = library_file_hash(market, lib_root) or "nolib"
        # holdout_ratio：CLI 当前不透传 → 常量 0.2，与 run_team_agent 默认参数同源
        _holdout_ratio_key = "0.2"
        key_src = "|".join(
            [
                lib_hash,
                args.start,
                args.end,
                str(getattr(args, "universe", None)),
                market,
                _holdout_ratio_key,
                str(bool(getattr(args, "intraday_leaves", False))),
                str(getattr(args, "intraday_freq", "5min") or "5min"),
            ]
        )
        key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
        cache_dir = _mine_team_dir / "_pool_cache" / key
        if (cache_dir / "pool_meta.json").exists():
            print(f"[mine-team] 复用现有池缓存 {cache_dir}", flush=True)
            pool_cache_dir = str(cache_dir)
        else:
            # 路径归位：顶层 pool-prebuild → mine pool-prebuild
            cmd = [
                sys.executable,
                "-m",
                "factorzen.cli.main",
                "mine",
                "pool-prebuild",
                "--start",
                args.start,
                "--end",
                args.end,
                "--market",
                market,
                "--index-path",
                str(args.index_path),
                "--library-root",
                lib_root,
                "--holdout-ratio",
                _holdout_ratio_key,
                "--out",
                str(cache_dir),
            ]
            # universe/symbols/top_n/intraday 旗标逐一透传（None 不传）
            if getattr(args, "universe", None) is not None:
                cmd.extend(["--universe", str(args.universe)])
            if getattr(args, "symbols", None):
                cmd.extend(["--symbols", str(args.symbols)])
            if getattr(args, "top_n", None) is not None:
                cmd.extend(["--top-n", str(args.top_n)])
            if bool(getattr(args, "intraday_leaves", False)):
                cmd.append("--intraday-leaves")
            if getattr(args, "intraday_freq", None):
                cmd.extend(["--intraday-freq", str(args.intraday_freq)])
            print(f"[mine-team] 池预构建子进程启动:{' '.join(cmd)}", flush=True)
            proc = subprocess.run(cmd)  # stdout/stderr 直通；不设 timeout
            if proc.returncode == 0 and (cache_dir / "pool_meta.json").exists():
                pool_cache_dir = str(cache_dir)
            else:
                print(
                    f"[mine-team] 警告:池预构建子进程失败(exit={proc.returncode})→ 回退进程内构建",
                    file=sys.stderr,
                )
    elif use_subproc:
        print(
            "[mine-team] --pool-subproc 与 --no-library-orthogonal 同开:池不会被使用,跳过子进程",
            flush=True,
        )

    # 数据装配与 agent 路径共用 `_prepare_agent_mining_data`（ashare=A 股 loader，
    # crypto=Vision 湖 + 预热前缀）。消除双路径漂移。
    daily, profile, prep_meta = _prepare_agent_mining_data(args)
    if daily is None:
        print("[mine-team] crypto 挖掘帧为空（检查 --symbols 或数据湖覆盖）", file=sys.stderr)
        return 1
    # eval_start = 挖掘窗口 start（预热前缀边界），同 M1/agent 口径，见 _cmd_mine_agent。
    # 所有权交接(P5):CLI 层不钉住 raw daily,使深层的释放真实生效(全 A ~3.5G)。
    _daily_holder = [daily]
    del daily
    res = pmt.run_team_mine(
        _daily_holder.pop(),
        n_rounds=args.iterations,
        seed=args.seed,
        top_k=args.top_k,
        index_path=args.index_path,
        structured=args.structured,
        patience=args.patience,
        heal_rounds=args.heal_rounds,
        hypotheses_per_round=args.hypotheses_per_round,
        data_window=_data_window_with_membership(args, prep_meta),
        command=_command_line(args),
        eval_start=args.start,
        profile=profile,
        update_library=not getattr(args, "no_library", False),
        library_orthogonal=not getattr(args, "no_library_orthogonal", False),
        objective=getattr(args, "objective", "residual"),
        llm_workers=getattr(args, "llm_workers", 4),
        auto_lift=not bool(getattr(args, "no_auto_lift", False)),
        lift_se_mult=float(getattr(args, "lift_se_mult", 1.0)),
        lift_workers=getattr(args, "lift_workers", None),  # None→自适应(按可用内存)
        campaign_prior_enabled=not bool(getattr(args, "no_campaign_prior", False)),
        intraday_scout=bool(getattr(args, "intraday_scout", False)),
        scout_k=int(getattr(args, "scout_k", 4) or 4),
        scout_max_leaves=int(getattr(args, "scout_max_leaves", 12) or 12),
        scout_freq=getattr(args, "intraday_freq", "5min") or "5min",
        pool_cache_dir=pool_cache_dir,
        # 成交口径：CLI 默认可实现 (1 / open_adj)；--exec-lag 0 回旧口径（不可实现，仅对照）
        exec_lag=int(args.exec_lag) if getattr(args, "exec_lag", None) is not None else 1,
        exec_price_col=getattr(args, "exec_price_col", "open_adj"),
        sleeve_gate=not bool(getattr(args, "no_sleeve_gate", False)),
    )
    print(f"[mine-team] 候选 {res['n_candidates']} 个 / N={res['n_trials']} → {res['run_dir']}")
    return 0


def _cmd_factor_library_rebuild(args: argparse.Namespace) -> int:
    from datetime import date
    from datetime import datetime as _dt
    from pathlib import Path

    import polars as pl

    from factorzen.core.experiment import get_git_sha
    from factorzen.discovery import factor_library as fl
    from factorzen.discovery.backtest_window import default_window
    from factorzen.validation.holdout import split_holdout

    market = args.market
    # ── 定向重估目标（--only / --only-file 并集；均缺省 → None = 全量 rebuild）──
    only: list[str] | None = None
    only_raw: list[str] = list(getattr(args, "only", None) or [])
    only_file = getattr(args, "only_file", None)
    if only_file:
        try:
            for line in Path(only_file).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    only_raw.append(line)
        except OSError as exc:
            print(f"[factor-library] 读取 --only-file 失败：{exc}", file=sys.stderr)
            return 1
    if only_raw:
        only = only_raw
    elif getattr(args, "only", None) is not None or only_file:
        # 显式给了定向旗标却解析出空集：静默降级成全量重估会重排全库 status，
        # 是「以为只动几条、实际动全库」的最坏结果 → fail-loudly
        print(
            "[factor-library] --only/--only-file 解析出空目标集；如需全量重估请不带这两个旗标重跑",
            file=sys.stderr,
        )
        return 1

    # A 股不带 --universe = 全 A 5000+ 只拉取，多年窗口必 OOM（实测 ~22GB 被杀）；
    # 库的评估口径历史上一直是命名池（csi300），无池 rebuild 几乎必为误操作。
    if market == "ashare" and not getattr(args, "universe", None):
        print(
            "[factor-library] 警告：未指定 --universe，将拉取全 A 股（内存开销极大，"
            "多年窗口可能 OOM）；库的历史口径为 --universe csi300",
            file=sys.stderr,
        )
    # 窗口：显式 --start/--end 覆盖，否则默认窗口（最近约 6 年滚动到数据最新端）
    if args.start and args.end:
        start, end = args.start, args.end
    else:
        try:
            start, end = default_window(market)
        except ValueError as exc:
            print(f"[factor-library] {exc}", file=sys.stderr)
            return 1
    # 装配数据（复用挖掘装配 `_prepare_agent_mining_data`，含预热前缀）：窗口写回 args
    args.start, args.end = start, end
    # 源收集提到 prep 之前：分钟叶子自动置位要先知道会碰哪些表达式（见下）
    sources = fl.collect_source_expressions(market)
    # 自动置位（与 `factor-library lift-test` 同款）：库内任一记录或本批源引用 i_*/ix_*
    # → 装日内面板。**必须自动，不能只靠旗标**——lift 复审覆盖库内全部 lift 轨记录，
    # 忘了加旗标就会让它们物化失败；而复审把「算不出来」当「无增量」处理过（已修，
    # 现在会保持原状 + 非零退出），根子上还是这里不该缺列。
    from factorzen.discovery.preparation import (
        expressions_need_intraday,
        intraday_expr_leaf_names,
    )

    _scan_exprs: list[str] = [str(e) for e in sources if e and not fl.is_python_identity(str(e))]
    # root 在调用点取（`load_library` 的默认参数在 def 时求值，绑死的是导入时的
    # DEFAULT_ROOT；显式传才跟得上 patch/配置，也与 lift-test 的取法一致）
    _lib_root = getattr(args, "library_root", None) or fl.DEFAULT_ROOT
    # 库读不出来不该挡住 rebuild；真缺列会在复审的求值失败守卫里被点名
    with contextlib.suppress(Exception):
        _scan_exprs += [
            str(r.expression)
            for r in fl.load_library(market, root=_lib_root)
            if r.expression and not fl.is_python_identity(str(r.expression))
        ]
    if getattr(args, "intraday_leaves", False) or expressions_need_intraday(_scan_exprs):
        args.intraday_leaves = True
    _ix = intraday_expr_leaf_names(_scan_exprs)
    if _ix:
        args.intraday_expr_leaves = _ix
        args.intraday_leaves = True
    daily, profile, _prep_meta = _prepare_agent_mining_data(args)
    if daily is None:
        print(
            "[factor-library] 挖掘帧为空（检查 --symbols / 数据湖覆盖 / 缓存回补）", file=sys.stderr
        )
        return 1
    leaf_map = profile.factors.leaf_features() if profile is not None else None
    if not sources:
        print(f"[factor-library] 提示：未从历史产物收集到 {market} 候选（将产出空库文件）")
    evaluate, compact_materialize = fl.build_library_evaluator(
        daily,
        holdout_ratio=args.holdout_ratio,
        eval_start=start,
        leaf_map=leaf_map,
        profile=profile,
    )
    # lift 复审评分窗 = single 轨 evaluator 的 holdout 尾段（同 split_holdout 口径）
    # build_library_evaluator 内部：sample = prepped[trade_date>=eval_start] 再 split；
    # prep 不改 trade_date 集合，这里对 daily 同边界切分即可复用同一 holdout 起止。
    es_date = _dt.strptime(start, "%Y%m%d").date() if start else None
    sample = daily if es_date is None else daily.filter(pl.col("trade_date") >= es_date)
    _, holdout_df, holdout_start = split_holdout(
        sample,
        holdout_ratio=float(args.holdout_ratio),
    )
    lift_adm_start = _lift_admission_str(holdout_start)
    lift_adm_end = _lift_admission_str(holdout_df["trade_date"].max())

    # lift 复审 active 池物化：透传 python_universe（expression + py:: 统一入口）。
    # 惰性构建：只有 rebuild 真调物化（存在 lift 轨复审 + active 记录）才 prep——
    # ①无复审的 rebuild 不多付一次 prep；②退化帧（缺列的小样本/测试）prep 失败
    # 降级为 materialize 恒 None（旧行为），不许在 CLI 层直接崩。
    def _make_lazy_rebuild_materializer():
        state: dict = {"mat": None, "failed": False}

        def _mat(expr: str):
            if state["failed"]:
                return None
            if state["mat"] is None:
                try:
                    from factorzen.discovery.evaluation import _preprocess_daily
                    from factorzen.discovery.lift_test import _materializer_from_prepped

                    prepped_mat = _preprocess_daily(daily, profile).sort(["ts_code", "trade_date"])
                    state["mat"] = _materializer_from_prepped(
                        prepped_mat,
                        leaf_map,
                        python_universe=args.universe,
                        python_market=market,
                    )
                except Exception as exc:
                    print(
                        f"[factor-library] rebuild 复审物化器构建失败"
                        f"（{type(exc).__name__}: {exc}），active 基线池回退为空",
                        file=sys.stderr,
                    )
                    state["failed"] = True
                    return None
            return state["mat"](expr)

        return _mat

    materialize = _make_lazy_rebuild_materializer()
    res = fl.rebuild(
        market,
        sources=sources,
        eval_window=(start, end),
        universe=args.universe,
        horizon=args.horizon,
        evaluate=evaluate,
        compact_materialize=compact_materialize,
        materialize=materialize,
        git_sha=get_git_sha(),
        now=date.today().strftime("%Y-%m-%d"),
        leaf_map=leaf_map,
        decorr_threshold=args.decorr_threshold,
        daily=daily,
        profile=profile,
        admission_start=lift_adm_start,
        admission_end=lift_adm_end,
        only=only,
    )
    # lift 轨复审失败时 rebuild 已恢复旧记录；CLI 必须 fail-loudly，禁止「表面成功」
    if res.lift_review_error is not None:
        print(
            f"[factor-library] lift 轨复审失败：{res.lift_review_error}"
            f"（旧 lift 记录已恢复，本次 rebuild 不完整）",
            file=sys.stderr,
        )
        return 1
    mode = f"定向重估 {len(only)} 条" if only else "全量 rebuild"
    print(
        f"[factor-library] {market} {mode}：新增 {res.added} / 更新 {res.updated} / "
        f"标记 correlated {res.correlated} / 跳过 {res.skipped}（窗口 {start}–{end}）"
    )
    if only:
        # 定向语义必须每次说清楚：操作者最容易误以为「重估完 correlated 会自动升回来」
        print(
            "[factor-library] 定向重估：只刷新指标 + 只降不升"
            "（correlated/no_lift 升回 active 需跑全量 rebuild）"
        )
    if res.gate_failed:
        # 指标已按真值刷新，但这些记录已不满足 library gate 却仍留在库里 → 必须大声说
        print(
            f"[factor-library] 警告：{len(res.gate_failed)} 条定向目标重估后不再满足 "
            f"library gate（指标已刷新，status 未裁决，需人工复核或跑全量 rebuild）："
            f"{', '.join(res.gate_failed[:10])}"
            f"{' …' if len(res.gate_failed) > 10 else ''}",
            file=sys.stderr,
        )
    print(f"[factor-library] → {FACTOR_LIBRARY_DIR}/{market}.jsonl + {market}.md")
    if res.lift_eval_failed:
        # 求值失败的记录已保持原状（未降级），但本次 lift 轨结论不完整 → 非零退出。
        # 最常见成因：记录带分钟派生叶子（i_*），而本次 rebuild 未装配分钟面板。
        n = len(res.lift_eval_failed)
        print(
            f"[factor-library] 错误：{n} 条 lift 记录复审时求值失败，已保持原状未降级"
            f"（本次 lift 轨结论不完整）："
            f"{', '.join(res.lift_eval_failed[:10])}"
            f"{' …' if n > 10 else ''}",
            file=sys.stderr,
        )
        print(
            "[factor-library] 若这些表达式含分钟派生叶子（i_*），需带 --intraday-leaves "
            "重跑，否则叶子不在挖掘帧里必然物化失败",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_factor_library_list(args: argparse.Namespace) -> int:
    from factorzen.discovery import factor_library as fl

    lib = sorted(fl.load_library(args.market), key=fl._sort_key)
    if not lib:
        print(f"[factor-library] {args.market} 库为空")
        return 0
    print(f"[factor-library] {args.market}: {len(lib)} 个因子（holdout_ic 降序）")
    for i, r in enumerate(lib, 1):
        print(
            f"  {i:>3}. holdout_ic={fl._fmt(r.holdout_ic)} ic_train={fl._fmt(r.ic_train)} "
            f"[{r.status}] {r.expression}"
        )
    return 0


def _cmd_factor_library_show(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from factorzen.discovery import factor_library as fl

    lib = sorted(fl.load_library(args.market), key=fl._sort_key)
    rec = None
    if args.expression:
        norm = fl._normalize(args.expression)
        rec = next((r for r in lib if r.expression in (norm, args.expression)), None)
    elif args.rank is not None:
        if 1 <= args.rank <= len(lib):
            rec = lib[args.rank - 1]
    else:
        print("[factor-library] 需指定 --expression 或 --rank", file=sys.stderr)
        return 2
    if rec is None:
        print(f"[factor-library] 未找到该因子（market={args.market}）", file=sys.stderr)
        return 1
    for k, v in asdict(rec).items():
        print(f"  {k}: {v}")
    return 0


def _factor_store_panel_loader(
    *,
    start: str,
    end: str,
    universe: str,
    market: str,
    intraday_leaves: bool = False,
):
    """factor_store 物化的数据装配（CLI 层注入，依赖倒置防 discovery→cli 环）。

    返回 preprocess 后的 polars 挖掘帧。

    经生产通道拉挖掘帧并 preprocess，与 mine team/agent 同源
    （``_prepare_agent_mining_data`` + ``_preprocess_daily``）。
    """
    import polars as pl

    from factorzen.discovery.evaluation import _preprocess_daily

    ns = argparse.Namespace(
        market=market,
        start=start,
        end=end,
        universe=universe,
        horizon=1,
        intraday_leaves=bool(intraday_leaves),
        intraday_freq="5min",
        intraday_expr_leaves=None,
        top_n=50,
        symbols=None,
        freq="daily",
    )
    daily, profile, _prep_meta = _prepare_agent_mining_data(ns)
    if daily is None or daily.is_empty():
        raise RuntimeError(f"挖掘帧为空 market={market} {start}–{end} universe={universe}")
    prepped = _preprocess_daily(daily, profile).sort(["ts_code", "trade_date"])
    float_cols = [
        c
        for c, dt in zip(prepped.columns, prepped.dtypes, strict=False)
        if dt in (pl.Float32, pl.Float64)
    ]
    if float_cols:
        prepped = prepped.with_columns([pl.col(c).fill_nan(None) for c in float_cols])
    return prepped


def _cmd_factor_library_store_sync(args: argparse.Namespace) -> int:
    """从 jsonl 同步 factor_store 三件套（meta + factor.py + 可选 parquet）。"""
    import time

    from factorzen.discovery import factor_library as fl
    from factorzen.discovery import factor_store as fs

    market = args.market
    lib_root = getattr(args, "lib_root", None) or fl.DEFAULT_ROOT
    store_root = getattr(args, "root", None) or fs.DEFAULT_ROOT
    only_raw = getattr(args, "only", None)
    only = [x.strip() for x in only_raw.split(",") if x.strip()] if only_raw else None
    materialize = not bool(getattr(args, "no_materialize", False))
    mat_start = fs.STORE_MATERIALIZE_START
    mat_end = fs.store_materialize_end()
    mat_univ = fs.STORE_MATERIALIZE_UNIVERSE
    print(
        f"[factor-library store sync] market={market} root={store_root} "
        f"lib_root={lib_root} materialize={materialize} "
        f"window={mat_start}..{mat_end} universe={mat_univ}"
        + (f" only={only}" if only else " only=ALL"),
        flush=True,
    )
    t0 = time.perf_counter()
    stats = fs.sync_store(
        market,
        root=store_root,
        only=only,
        materialize=materialize,
        lib_root=lib_root,
        default_universe=mat_univ,
        panel_loader=_factor_store_panel_loader,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"[factor-library store sync] done in {elapsed:.1f}s: "
        f"written={stats.get('written')} materialized={stats.get('materialized')} "
        f"skipped_materialize={stats.get('skipped_materialize')} "
        f"errors={len(stats.get('errors') or [])} total={stats.get('total')}",
        flush=True,
    )
    for err in (stats.get("errors") or [])[:20]:
        print(f"  ! {err}", flush=True)
    return 1 if stats.get("errors") else 0


def _cmd_factor_library_store_verify(args: argparse.Namespace) -> int:
    """校验 meta.expression 与 jsonl 一致。"""
    from factorzen.discovery import factor_library as fl
    from factorzen.discovery import factor_store as fs

    market = args.market
    lib_root = getattr(args, "lib_root", None) or fl.DEFAULT_ROOT
    store_root = getattr(args, "root", None) or fs.DEFAULT_ROOT
    report = fs.verify_store(market, root=store_root, lib_root=lib_root)
    n_drift = len(report.get("drifts") or [])
    n_miss = len(report.get("missing_in_store") or [])
    n_extra = len(report.get("missing_in_ledger") or [])
    print(
        f"[factor-library store verify] market={market} ok={report.get('ok')} "
        f"checked={report.get('n_checked')} drifts={n_drift} "
        f"missing_in_store={n_miss} missing_in_ledger={n_extra}",
        flush=True,
    )
    for d in (report.get("drifts") or [])[:50]:
        print(f"  DRIFT {d}", flush=True)
    for n in (report.get("missing_in_store") or [])[:20]:
        print(f"  MISSING_STORE {n}", flush=True)
    for n in (report.get("missing_in_ledger") or [])[:20]:
        print(f"  EXTRA_STORE {n}", flush=True)
    return 0 if report.get("ok") else 1


def _cmd_factor_library_forward_track(args: argparse.Namespace) -> int:
    """记录 as_of 日库内因子的 paper forward RankIC。

    forward 确认窗口随真实时间累积；ops 每日链路接线为后续工作。
    非 ashare fail closed（return 2）；全部 failed → return 1；
    历史回灌/未来日拒（return 2，--allow-backfill 逃生口）。
    """
    from factorzen.discovery.backtest_window import latest_data_date
    from factorzen.discovery.factor_library import DEFAULT_ROOT
    from factorzen.discovery.forward_track import record_forward_ics

    market = args.market
    root = getattr(args, "root", None) or DEFAULT_ROOT
    as_of = getattr(args, "date", None)

    # S5/P8：非 A 股入口 fail closed（尚未接入 profile/provider/leaf-map）
    if market != "ashare":
        print(
            f"[factor-library forward-track] 非 A 股入口 fail closed："
            f"market={market} 暂未接入 profile/provider/leaf-map；"
            f"勿用 A 股数据求值非 A 股因子。",
            file=sys.stderr,
        )
        return 2

    if not as_of:
        latest = latest_data_date(market)
        if latest is None:
            print(
                f"[factor-library forward-track] 探测不到 {market} 最新交易日；"
                f"请显式传 --date YYYYMMDD",
                file=sys.stderr,
            )
            return 1
        as_of = latest.strftime("%Y%m%d")
    try:
        out = record_forward_ics(
            market,
            as_of,
            root=root,
            universe=getattr(args, "universe", None),
            allow_backfill=bool(getattr(args, "allow_backfill", False)),
            max_backfill_days=int(getattr(args, "max_backfill_days", 10)),
        )
    except ValueError as exc:
        print(
            f"[factor-library forward-track] 失败：{exc}",
            file=sys.stderr,
        )
        return 2
    recorded = int(out.get("recorded", 0) or 0)
    failed = int(out.get("failed", 0) or 0)
    print(
        f"[factor-library forward-track] {market} as_of={as_of}："
        f"recorded={recorded} "
        f"skipped_existing={out.get('skipped_existing', 0)} "
        f"failed={failed}"
    )
    if recorded > 0 and failed == recorded:
        print(
            f"[factor-library forward-track] 全部 failed（recorded={recorded}），退出码 1",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_factor_library_forward_review(args: argparse.Namespace) -> int:
    """裁决 probation 因子的 paper forward 证据；默认 dry-run，--apply 才写库。

    forward 确认窗口随真实时间累积；ops 每日链路接线为后续工作。
    """
    from factorzen.discovery.factor_library import DEFAULT_ROOT
    from factorzen.discovery.forward_track import forward_review

    market = args.market
    root = getattr(args, "root", None) or DEFAULT_ROOT
    apply = bool(getattr(args, "apply", False))
    rows = forward_review(
        market,
        root=root,
        min_days=int(getattr(args, "min_days", 60)),
        se_mult=float(getattr(args, "se_mult", 1.645)),
        block_days=int(getattr(args, "block_days", 20)),
        apply=apply,
    )
    print(
        f"[factor-library forward-review] {market}：{len(rows)} 个 probation 裁决（apply={apply}）"
    )
    if rows:
        print(f"{'expression':<42} {'decision':<10} {'n':>5} {'mean':>10} {'ci_low':>10}")
        for r in rows:
            expr = r.get("expression") or ""
            if len(expr) > 40:
                expr = expr[:37] + "..."
            mean = r.get("mean")
            ci = r.get("ci_low")
            mean_s = f"{mean:.4f}" if isinstance(mean, (int, float)) and mean == mean else "-"
            ci_s = f"{ci:.4f}" if isinstance(ci, (int, float)) and ci == ci else "-"
            print(
                f"{expr:<42} {r.get('decision', '-'):<10} "
                f"{r.get('n_days', 0):>5} {mean_s:>10} {ci_s:>10}"
            )
    if apply:
        n_promote = sum(1 for r in rows if r.get("decision") == "promote")
        n_demote = sum(1 for r in rows if r.get("decision") == "demote")
        n_hold = sum(1 for r in rows if r.get("decision") == "hold")
        print(
            f"[factor-library forward-review] 状态转换："
            f"promote={n_promote} demote={n_demote} hold={n_hold}"
        )
    else:
        print("[factor-library forward-review] dry-run（加 --apply 写库并更新 markdown）")
    return 0


def _cmd_factor_library_lift_null(args: argparse.Namespace) -> int:
    """lift 统计层 null 校准：扫 se_mult×min_blocks，打印误准入率校准表。"""
    from factorzen.discovery.lift_null import (
        calibration_table,
        format_calibration_markdown,
    )

    se_mults = tuple(float(x) for x in args.se_mults.split(",") if x.strip())
    min_blocks = tuple(int(x) for x in args.min_blocks.split(",") if x.strip())
    rows = calibration_table(
        n_days=args.n_days,
        daily_sigma=args.daily_sigma,
        ar1=args.ar1,
        se_mults=se_mults,
        min_blocks_options=min_blocks,
        n_sims=args.n_sims,
        seed=args.seed,
    )
    print(
        f"[lift-null] H0=无真实 lift；n_days={args.n_days} σ={args.daily_sigma} "
        f"ar1={args.ar1} n_sims={args.n_sims} seed={args.seed}"
    )
    print("[lift-null] 统计层下界：真实链路含选择偏差，误准入只会更高")
    print(format_calibration_markdown(rows))
    return 0


def _timestamped_sibling(path: Path) -> Path:
    """``x.json`` → ``x_{YYYYmmddTHHMMSS}.json``，**保证不覆写已存在文件**。

    同秒重跑（或时钟回拨）撞名时追加 ``_2``/``_3``…… 后缀——审计归档一旦落盘
    就不可变，静默覆盖等于丢证据。
    """
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    cand = path.with_name(f"{path.stem}_{ts}{path.suffix}")
    n = 2
    while cand.exists():
        cand = path.with_name(f"{path.stem}_{ts}_{n}{path.suffix}")
        n += 1
    return cand


def _lift_admission_str(v) -> str | None:
    """边界日期 → admission 窗字符串（对齐 polars Date→Utf8 的 YYYY-MM-DD）。"""
    if v is None:
        return None
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d")
    s = str(v).strip().replace("/", "-")
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    if len(s) >= 10 and s[4] == "-":
        return s[:10]
    return s


def _holdout_bounds_from_manifest(man: dict) -> tuple[str | None, str | None]:
    """从 session manifest 抽 holdout 评分窗边界。

    侦察结论：
    - mining_session：顶层 ``holdout_start``（``str(date)``，常为 YYYY-MM-DD）
    - mine_team / mine-agent：当前**不落** holdout_start，仅 params 有
      holdout_ratio / start / end / eval_start——无交易日历时无法反推切点，
      故只认显式 holdout 字段（顶层或 params.holdout_start）。
    - 回退：session 末 auto-lift 已写入的 ``lift_group.admission_start/end``
      （team_1002 等历史 team manifest 的实操评分窗）。
    - end：params.end / end / mining_end（多 session 取最晚）
    """
    start = man.get("holdout_start")
    if start is None:
        params = man.get("params") or {}
        start = params.get("holdout_start")
    end = man.get("holdout_end") or man.get("mining_end") or man.get("end")
    if end is None:
        params = man.get("params") or {}
        end = params.get("end") or params.get("eval_end")
    # team auto-lift 回写的评分窗（无顶层 holdout_start 时）
    if start is None or end is None:
        lg = man.get("lift_group") or {}
        if isinstance(lg, dict):
            if start is None:
                start = lg.get("admission_start")
            if end is None:
                end = lg.get("admission_end")
    return _lift_admission_str(start), _lift_admission_str(end)


def _horizon_from_manifest(man: dict) -> int | None:
    """从 session manifest 抽 mining horizon。

    键名（对照写盘代码）：
    - team：``write_team_manifest`` 把调用方 ``params`` 原样落盘 → ``params.horizon``
      （``run_team_mine`` 当前 params 未必写 horizon，缺则返回 None）
    - mining_session：顶层字段可选 ``horizon``（``run_session`` 有入参但历史 manifest
      多数未落盘）
    - 顶层 ``horizon`` 优先于 ``params.horizon``
    - 回退：``lift_group.horizon``（auto-lift 实跑评分窗）
    """
    raw = man.get("horizon")
    if raw is None:
        params = man.get("params") or {}
        raw = params.get("horizon")
    if raw is None:
        lg = man.get("lift_group") or {}
        if isinstance(lg, dict):
            raw = lg.get("horizon")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _group_lift_candidates_by_admission(
    session_items: list[dict],
) -> list[tuple[str | None, str | None, list[dict]]]:
    """跨 session 按 expression 去重（首次出现胜出），再按 admission 窗分组。

    输入每项 ``{"session", "candidates", "adm_start", "adm_end"}``。
    返回 ``(adm_start, adm_end, candidates)`` 列表，顺序按窗首次出现稳定。
    纯函数：无 IO、不建帧、不调 run_lift_tests。
    """
    seen: set[str] = set()
    group_order: list[tuple[str | None, str | None]] = []
    groups: dict[tuple[str | None, str | None], list[dict]] = {}

    for item in session_items:
        key = (item.get("adm_start"), item.get("adm_end"))
        for cand in item.get("candidates") or []:
            expr = cand.get("expression")
            if not expr or expr in seen:
                continue
            seen.add(expr)
            if key not in groups:
                groups[key] = []
                group_order.append(key)
            groups[key].append(cand)

    return [(s, e, groups[(s, e)]) for s, e in group_order]


def _resolve_session_index_path(session_dir: str, man: dict):
    """优先 manifest ``params.index_path``（存在才用）；否则回退 session 父目录下
    ``experiment_index.jsonl``（常见：manifest 记的是临时 worktree 绝对路径）。"""
    from pathlib import Path as _P

    params = man.get("params") or {}
    ip = params.get("index_path")
    if ip:
        p = _P(ip)
        if p.exists():
            return p
    return _P(session_dir).parent / "experiment_index.jsonl"


def _data_window_from_session_manifest(man: dict) -> dict:
    """从 session manifest params 取 data_window（分族召回用）。"""
    params = man.get("params") or {}
    return {
        "start": params.get("start"),
        "end": params.get("end"),
        "universe": params.get("universe"),
        "market": params.get("market") or man.get("market"),
    }


def _session_lift_queue_norm_set(man: dict) -> set[str]:
    """session manifest 的 lift 队列表达式集合（归一化，供候选归属）。"""
    from factorzen.agents.experiment_index import _normalize
    from factorzen.discovery.lift_test import extract_gray_candidates_from_manifest

    out: set[str] = set()
    for c in extract_gray_candidates_from_manifest(man):
        expr = c.get("expression")
        if expr:
            out.add(_normalize(str(expr)))
    return out


def _write_cli_lift_rejects_to_index(
    *,
    results: list[dict],
    session_items: list[dict],
    session_manifests: dict[str, dict],
    threshold: float,
    se_mult: float,
) -> int:
    """--apply 时把本批 lift 拒绝写回各来源 session 的 experiment_index。

    含 group_gate_fail 行与 lift_admission==reject 行。返回写入条数。
    """
    from factorzen.agents.experiment_index import (
        ExperimentIndex,
        _normalize,
        build_lift_reject_record,
    )
    from factorzen.discovery.lift_test import lift_admission

    # expression(norm) → 首个归属 session
    expr_to_session: dict[str, str] = {}
    session_cand_meta: dict[str, dict[str, dict]] = {}  # sess → norm_expr → cand
    for item in session_items:
        sess = str(item.get("session") or "")
        man = session_manifests.get(sess) or {}
        queue_set = _session_lift_queue_norm_set(man)
        for c in item.get("candidates") or []:
            expr = c.get("expression")
            if not expr:
                continue
            ne = _normalize(str(expr))
            # 归属：优先本 session 队列命中；否则首次见到的 session
            if ne not in expr_to_session and (not queue_set or ne in queue_set):
                expr_to_session[ne] = sess
            session_cand_meta.setdefault(sess, {})[ne] = c

    # 按 session 聚合待写记录
    by_session: dict[str, list[dict]] = {}
    for row in results:
        expr = row.get("expression")
        if not expr:
            continue
        err = str(row.get("error") or "")
        is_gg = err.startswith("group_gate_fail")
        if is_gg:
            reason = "group_gate_fail"
        elif lift_admission(row, threshold=float(threshold), se_mult=float(se_mult)) == "reject":
            reason = "below_bar"
        else:
            continue  # active/probation 不写回
        ne = _normalize(str(expr))
        owner: str | None = expr_to_session.get(ne)
        if owner is None:
            # 回退：扫各 session 队列
            for item in session_items:
                s = str(item.get("session") or "")
                man = session_manifests.get(s) or {}
                if ne in _session_lift_queue_norm_set(man):
                    owner = s
                    break
        if owner is None:
            continue
        src = (session_cand_meta.get(owner) or {}).get(ne) or {}
        by_session.setdefault(owner, []).append(
            build_lift_reject_record(
                expression=str(expr),
                data_window=_data_window_from_session_manifest(
                    session_manifests.get(owner) or {},
                ),
                lift=row.get("lift"),
                lift_se=row.get("lift_se"),
                lift_reason=reason,
                source="cli_lift_test",
                ic_train=src.get("ic_train") if isinstance(src, dict) else None,
                residual_ic_train=(src.get("residual_ic_train") if isinstance(src, dict) else None),
                baseline_rank_ic=row.get("baseline"),
                admission_start=row.get("admission_start"),
                admission_end=row.get("admission_end"),
            )
        )

    n_written = 0
    for sess, recs in by_session.items():
        man = session_manifests.get(sess) or {}
        ip = _resolve_session_index_path(sess, man)
        ExperimentIndex(str(ip)).append(recs)
        n_written += len(recs)
    return n_written


def _cmd_factor_library_lift_test(args: argparse.Namespace) -> int:
    """灰区/lift 队列候选 → 组合 OOS lift 实验；默认 dry-run，--apply 才入库。"""
    err = _apply_set_overrides(args, _LIFT_TEST_SET)
    if err is not None:
        return err
    import json
    from datetime import date
    from pathlib import Path

    from factorzen.core.experiment import get_git_sha
    from factorzen.discovery import factor_library as fl
    from factorzen.discovery.guardrails import (
        DEFAULT_GRAY_IC_FLOOR,
        DEFAULT_LIFT_THRESHOLD,
        DEFAULT_RAW_GRAY_IC_FLOOR,
        is_sub_floor_candidate,
    )
    from factorzen.discovery.lift_test import (
        DEFAULT_HORIZON,
        LiftEvalContext,
        _rank_ic_key,
        extract_gray_candidates_from_manifest,
        filter_candidates_by_coverage,
        group_gate_ok,
        make_lift_context,
        partition_lift_queue_by_sleeve,
        resolve_lift_workers,
        run_group_lift,
        run_lift_tests,
    )

    sessions = list(args.session or [])
    factor_names = list(getattr(args, "factor", None) or [])
    if not sessions and not factor_names:
        print(
            "[factor-library lift-test] 需至少一个 --session 目录或 --factor 因子名",
            file=sys.stderr,
        )
        return 2

    market = args.market

    # --factor：一期仅 ashare + 必填 universe；registry 存在性 fail-loudly
    py_cands: list[dict] = []
    if factor_names:
        if market != "ashare":
            print(
                f"[factor-library lift-test] --factor 一期仅支持 market=ashare，"
                f"收到 market={market!r}",
                file=sys.stderr,
            )
            return 2
        if not getattr(args, "universe", None):
            print(
                "[factor-library lift-test] --factor 时 --universe 必填（如 csi300）",
                file=sys.stderr,
            )
            return 2
        from factorzen.daily.factors.registry import get_factor

        for name in factor_names:
            try:
                get_factor(name)
            except KeyError:
                print(
                    f"[factor-library lift-test] 未注册因子: {name!r}",
                    file=sys.stderr,
                )
                return 2
            py_cands.append(
                {
                    "expression": fl.python_identity(name),
                    "kind": "python",
                    "name": name,
                    "impl": name,
                }
            )

    # 旗标覆盖优先：任一非 None → 所有候选归同一旗标窗（escape hatch）
    flag_start = getattr(args, "admission_start", None)
    flag_end = getattr(args, "admission_end", None)
    use_flag_window = flag_start is not None or flag_end is not None
    flag_adm_start = _lift_admission_str(flag_start) if flag_start is not None else None
    flag_adm_end = _lift_admission_str(flag_end) if flag_end is not None else None
    args_end = _lift_admission_str(getattr(args, "end", None))

    session_items: list[dict] = []
    session_manifests: dict[str, dict] = {}  # session_dir → manifest（--apply 写回 index 用）
    manifest_horizons: list[int] = []
    for s in sessions:
        man_path = Path(s) / "manifest.json"
        if not man_path.is_file():
            print(f"[factor-library lift-test] 跳过（无 manifest）: {s}", file=sys.stderr)
            continue
        try:
            man = json.loads(man_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[factor-library lift-test] 读 manifest 失败 {s}: {exc}", file=sys.stderr)
            continue
        session_manifests[str(s)] = man
        gray_s = extract_gray_candidates_from_manifest(man)
        hs, he = _holdout_bounds_from_manifest(man)
        man_h = _horizon_from_manifest(man)
        if man_h is not None:
            manifest_horizons.append(man_h)
        if use_flag_window:
            s_start, s_end = flag_adm_start, flag_adm_end
        else:
            s_start = hs
            s_end = he or args_end
        session_items.append(
            {
                "session": str(s),
                "candidates": gray_s,
                "adm_start": s_start,
                "adm_end": s_end,
            }
        )

    # horizon：--horizon 旗标 > 首个 session manifest mining horizon > DEFAULT_HORIZON
    flag_horizon = getattr(args, "horizon", None)
    if flag_horizon is not None:
        resolved_horizon = int(flag_horizon)
    elif manifest_horizons:
        resolved_horizon = manifest_horizons[0]
        if len(set(manifest_horizons)) > 1:
            print(
                f"[factor-library lift-test] 警告：多 session mining horizon 不一致 "
                f"{manifest_horizons}，统一使用第一个 session 的 {resolved_horizon}",
                file=sys.stderr,
            )
    else:
        resolved_horizon = DEFAULT_HORIZON

    groups = _group_lift_candidates_by_admission(session_items)
    # python 候选单独成组：admission 窗取旗标；未给旗标 → g_start=None（沿用无独立性保证警告）
    if py_cands:
        groups.append((flag_adm_start, flag_adm_end, py_cands))

    n_gray = sum(len(cands) for _, _, cands in groups)
    if n_gray == 0:
        print("[factor-library lift-test] 未从 session/--factor 抽到候选")
        return 0

    print(
        f"[factor-library lift-test] 候选 {n_gray} 个（去重后），"
        f"admission 分组 {len(groups)} 组" + (f"（含 python {len(py_cands)}）" if py_cands else "")
    )
    for gi, (g_start, g_end, cands) in enumerate(groups, start=1):
        print(
            f"[factor-library lift-test]   组{gi}: "
            f"{g_start or '—'} ~ {g_end or '—'}  候选 {len(cands)} 个"
        )
        if g_start is None:
            print(
                "[factor-library lift-test] 警告：lift 评分未裁剪到 holdout 窗（无独立性保证）",
                file=sys.stderr,
            )

    # 装配日频帧一次（各 session 共享；跨 universe 分帧另任务）——与 mine agent/team
    # **同源** `_prepare_agent_mining_data`。禁止另起一套 loader，否则事件叶子缺列/fill
    # 语义漂移 → 候选近乎全空 → build_panel「行因子齐全」暴跌、lift 成噪声。
    # 自动置位：lift 队列 ∪ 库内 active 任一表达式引用 i_* → 装日内面板（堵死缺列静默失败）。
    lib_root = getattr(args, "library_root", None) or fl.DEFAULT_ROOT
    from factorzen.discovery.preparation import (
        expressions_need_intraday,
        intraday_expr_leaf_names,
    )

    all_exprs: list[str] = []
    for _gs, _ge, cands in groups:
        for c in cands:
            e = c.get("expression") if isinstance(c, dict) else None
            if not e:
                continue
            es = str(e)
            # py:: 哨兵不是可 parse 表达式；跳过以免误触 ix_ 词法或无意义 parse 尝试
            if fl.is_python_identity(es):
                continue
            all_exprs.append(es)
    try:
        for rec in fl.load_library(market, root=lib_root):
            if getattr(rec, "status", None) == "active" and rec.expression:
                es = str(rec.expression)
                if fl.is_python_identity(es):
                    continue
                all_exprs.append(es)
    except Exception:
        pass
    need_intraday = bool(getattr(args, "intraday_leaves", False)) or expressions_need_intraday(
        all_exprs
    )
    if need_intraday:
        args.intraday_leaves = True
    # ix_* 表达式叶子透传 prepare → attach_intraday
    ix_leaves = intraday_expr_leaf_names(all_exprs)
    if ix_leaves:
        args.intraday_expr_leaves = ix_leaves
        args.intraday_leaves = True

    daily, profile, _prep_meta = _prepare_agent_mining_data(args)
    if daily is None:
        print("[factor-library lift-test] 挖掘帧为空", file=sys.stderr)
        return 1
    leaf_map = profile.factors.leaf_features() if profile is not None else None
    threshold = getattr(args, "threshold", None)
    if threshold is None:
        threshold = DEFAULT_LIFT_THRESHOLD
    # 默认 top_m=20；--top-m 0 → 全测逃生口（no silent caps：截断必 stderr + manifest 记账）
    top_m_raw = getattr(args, "top_m", 20)
    if top_m_raw is None or int(top_m_raw) == 0:
        top_m: int | None = None  # 全测
    else:
        top_m = int(top_m_raw)
    seed = getattr(args, "seed", 0) or 0
    se_mult = float(getattr(args, "se_mult", 1.0) or 1.0)
    python_universe = getattr(args, "universe", None)

    # base_ctx：prep 一次；admission 窗 per-group replace（不改 horizon）
    try:
        base_ctx = make_lift_context(
            market,
            daily,
            profile=profile,
            leaf_map=leaf_map,
            horizon=resolved_horizon,
            admission_start=None,
            admission_end=None,
            library_root=lib_root,
            python_universe=python_universe,
            python_market=market,
        )
    except Exception:
        # 回退=raw 帧当 prepped:派生叶子(ret_1d 等)将全空——真实数据不应走到这
        # (2026-07-14 事故根因:候选全空面板→lift 全噪声)。仅容极简 mock 帧。
        print(
            "[factor-library lift-test] 警告：预处理失败,回退 raw 帧(派生叶子将缺失,"
            "真实数据下结果不可信)",
            file=sys.stderr,
        )
        base_ctx = LiftEvalContext(
            market=market,
            prepped=daily.sort(["ts_code", "trade_date"]) if daily.height else daily,
            leaf_map=leaf_map,
            horizon=resolved_horizon,
            admission_start=None,
            admission_end=None,
            library_root=lib_root,
            profile_name=getattr(profile, "name", None) if profile is not None else None,
            python_universe=python_universe,
            python_market=market,
        )

    lift_workers_arg = getattr(args, "lift_workers", None)  # None→自适应(按可用内存)
    workers_resolved = resolve_lift_workers(lift_workers_arg)
    print(
        f"[factor-library lift-test] lift_workers={workers_resolved}"
        + (
            "（自适应）"
            if lift_workers_arg is None
            else f"（显式 --lift-workers={lift_workers_arg}）"
        ),
        flush=True,
    )
    if lift_workers_arg == 1 and n_gray > 10:
        print(
            "[factor-library lift-test] 警告：--lift-workers 1 且候选 "
            f">{n_gray} 个，串行将极慢，建议留空走自适应",
            file=sys.stderr,
        )

    # 物化 memo：filter 与 run_lift_tests 共用，避免二次物化
    from factorzen.discovery.lift_test import _materializer_from_prepped

    mat_base = _materializer_from_prepped(
        base_ctx.prepped,
        leaf_map,
        python_universe=python_universe,
        python_market=market,
    )
    mat_cache: dict[str, object] = {}

    def memo_mat(expr: str):
        if expr in mat_cache:
            return mat_cache[expr]
        out = mat_base(expr)
        mat_cache[expr] = out
        return out

    results: list[dict] = []
    all_dropped: list[dict] = []
    all_sub_floor: list[dict] = []
    lift_groups_meta: list[dict] = []
    truncated_from: int | None = None
    n_lift_evaluated = 0

    # 组门连坐防呆：组门是「整批候选等权残差组合」的短路门，噪声占多数时会
    # 稀释真信号、把整组连坐拒掉（2026-07-17 事故：130/150 sub-floor →
    # 组 lift=-0.0007 → 150 条全拒 + 写回 lift_rejected）。sub-floor 候选按
    # 当前噪声地板本就不该在 lift 队列里（地板收紧前入队的历史积压），
    # 故默认剔出组门 = 恢复不变量；--include-sub-floor 为逃生口。
    queue_ic_floor = getattr(args, "queue_ic_floor", None)
    include_sub_floor = bool(getattr(args, "include_sub_floor", False))

    for g_start, g_end, cands in groups:
        if include_sub_floor:
            in_floor = list(cands)
            sub_floor = [
                c
                for c in cands
                if (
                    not c.get("sleeve_candidate")
                    and is_sub_floor_candidate(c, floor=queue_ic_floor)
                )
            ]
        else:
            in_floor, sub_floor = [], []
            for c in cands:
                # sleeve 旁路进队列的依据是子集 IC，残差/裸 IC 本就弱——
                # residual 噪声地板不得把 sleeve 剔出（否则 overlay 通道永远空）
                if c.get("sleeve_candidate"):
                    in_floor.append(c)
                elif is_sub_floor_candidate(c, floor=queue_ic_floor):
                    sub_floor.append(c)
                else:
                    in_floor.append(c)
        if sub_floor:
            floor_desc = (
                f"{queue_ic_floor}"
                if queue_ic_floor is not None
                else f"{DEFAULT_GRAY_IC_FLOOR}(残差)/{DEFAULT_RAW_GRAY_IC_FLOOR}(裸IC)"
            )
            share = len(sub_floor) / max(1, len(cands))
            action = "保留进组门（--include-sub-floor）" if include_sub_floor else "剔出组门"
            print(
                f"[factor-library lift-test] 警告：{len(sub_floor)}/{len(cands)} 个候选低于"
                f"噪声地板 sub-floor（floor={floor_desc}）→ {action}"
                f"（组 {g_start or '—'}~{g_end or '—'}）",
                file=sys.stderr,
            )
            if share >= 0.5:
                # 事故形态：噪声占多数。组门等权组合会被噪声主导。
                print(
                    f"[factor-library lift-test] 警告：sub-floor 占比 {share:.0%} ≥50%，"
                    "属「历史积压全量复测」形态——组门是小队列短路语义，"
                    "噪声占主时整组连坐拒会误杀真信号"
                    + (
                        "（当前 --include-sub-floor 已关闭防呆，风险自负）"
                        if include_sub_floor
                        else "（已按默认防呆剔除）"
                    ),
                    file=sys.stderr,
                )
            # 记账：被剔者不产生 results 行，故不会被写回 lift_rejected
            all_sub_floor.extend(
                {
                    "expression": c.get("expression"),
                    "ic_train": c.get("ic_train"),
                    "residual_ic_train": c.get("residual_ic_train"),
                    "admission_start": g_start,
                    "admission_end": g_end,
                    "filtered": not include_sub_floor,
                }
                for c in sub_floor
            )
        if not in_floor:
            lift_groups_meta.append(
                {
                    "admission_start": g_start,
                    "admission_end": g_end,
                    "skipped": "empty_after_sub_floor",
                }
            )
            continue

        n_in = len(in_floor)
        ordered = sorted(in_floor, key=_rank_ic_key, reverse=True)
        if top_m is not None and n_in > top_m:
            selected = ordered[:top_m]
            truncated_from = (truncated_from or 0) + n_in
            print(
                f"[factor-library lift-test] 警告：--top-m={top_m} 将截断候选 "
                f"（输入 {n_in} 个,按 |residual_ic_train| 排序截前 top_m={top_m}, "
                f"被截 truncated_from={n_in}）",
                file=sys.stderr,
            )
        else:
            selected = ordered
            if top_m is not None:
                # 未截断：不累加 truncated_from（顶层可省略或 =n）
                pass

        # holdout_start：admission 起点字符串可与 Date 比较时用 g_start；None 不裁
        holdout_start = g_start
        kept, dropped = filter_candidates_by_coverage(
            selected,
            materialize_candidate=memo_mat,
            holdout_start=holdout_start,
        )
        all_dropped.extend(dropped)
        if dropped:
            print(
                f"[factor-library lift-test] 覆盖剔除 {len(dropped)} 个"
                f"（组 {g_start or '—'}~{g_end or '—'}）",
                file=sys.stderr,
            )
        if not kept:
            lift_groups_meta.append(
                {
                    "admission_start": g_start,
                    "admission_end": g_end,
                    "skipped": "empty_after_coverage",
                }
            )
            continue

        grp_ctx = dataclasses.replace(
            base_ctx,
            admission_start=g_start,
            admission_end=g_end,
        )
        # sleeve 不与稠密混 residual 组门（07-19 overlay 口径；见 sleeve3-progress）
        dense_kept, sleeve_kept = partition_lift_queue_by_sleeve(kept)
        if sleeve_kept:
            print(
                f"[factor-library lift-test] sleeve 队列 {len(sleeve_kept)} 条"
                f"（跳过 residual 组门 → overlay 个体）"
                f"（组 {g_start or '—'}~{g_end or '—'}）",
                flush=True,
            )

        if dense_kept:
            group = run_group_lift(
                dense_kept,
                market=market,
                daily=daily,
                leaf_map=leaf_map,
                library_root=lib_root,
                seed=seed,
                threshold=threshold,
                materialize_candidate=memo_mat,
                ctx=grp_ctx,
            )
            # 防御性剥离：组结果本无 base_daily；若旧 mock 注入帧则不进 JSON manifest
            group_view = {k: v for k, v in group.items() if k != "base_daily"}
            lift_groups_meta.append(group_view)
            n_lift_evaluated += 1  # 组门计 1 次

            group_ok, bar = group_gate_ok(
                group,
                threshold=float(threshold),
                lift_se_mult=se_mult,
            )
            g_lift, g_se = group.get("lift"), group.get("lift_se")
            print(
                f"[factor-library lift-test] 组门(residual) lift={g_lift!r} se={g_se!r} "
                f"bar={bar:.4f} → {'过' if group_ok else '拒'}"
                f"（dense={len(dense_kept)}）",
                flush=True,
            )
            if not group_ok:
                # 组门不过：仅稠密 skip 逐候选；sleeve 仍走 overlay
                reason = f"group_gate_fail(lift={g_lift!r},se={g_se!r},bar={bar:.4f})"
                for c in dense_kept:
                    results.append(
                        {
                            "expression": c.get("expression"),
                            "lift": g_lift,
                            "lift_se": g_se,
                            "baseline": group.get("baseline"),
                            "passed": False,
                            "error": reason,
                            "admission_start": g_start,
                            "admission_end": g_end,
                            "ic_train": c.get("ic_train"),
                            "residual_ic_train": c.get("residual_ic_train"),
                        }
                    )
            else:
                rows = run_lift_tests(
                    dense_kept,
                    market=market,
                    daily=daily,
                    leaf_map=leaf_map,
                    library_root=lib_root,
                    top_m=None,  # CLI 已截断；此处全测 kept
                    threshold=threshold,
                    seed=seed,
                    ctx=grp_ctx,
                    lift_workers=lift_workers_arg,
                    materialize_candidate=memo_mat,
                )
                n_lift_evaluated += len(rows)
                for r in rows:
                    r = dict(r)
                    r.setdefault("admission_start", g_start)
                    r.setdefault("admission_end", g_end)
                    results.append(r)
        else:
            lift_groups_meta.append(
                {
                    "admission_start": g_start,
                    "admission_end": g_end,
                    "skipped": "no_dense_after_sleeve_split",
                    "n_sleeve": len(sleeve_kept),
                }
            )

        if sleeve_kept:
            sleeve_rows = run_lift_tests(
                sleeve_kept,
                market=market,
                daily=daily,
                leaf_map=leaf_map,
                library_root=lib_root,
                top_m=None,
                threshold=threshold,
                seed=seed,
                ctx=grp_ctx,
                lift_workers=lift_workers_arg,
                materialize_candidate=memo_mat,
            )
            n_lift_evaluated += len(sleeve_rows)
            for r in sleeve_rows:
                r = dict(r)
                r.setdefault("admission_start", g_start)
                r.setdefault("admission_end", g_end)
                results.append(r)

    # 打印表（含 lift_se / second_half）
    print(
        f"[factor-library lift-test] 评分完成："
        f"{len(groups)} 组 / {len(results)} 行（horizon={base_ctx.horizon}）"
    )
    print(
        f"{'expression':40s}  {'lift':>8s}  {'lift_se':>8s}  {'2nd_half':>8s}  "
        f"{'baseline':>8s}  passed"
    )
    for r in results:
        raw_expr = r.get("expression") or ""
        # python 候选打印用 name（可读性），避免 py:: 哨兵裸奔
        if fl.is_python_identity(str(raw_expr)):
            label = r.get("name") or fl._python_name_from_expression(str(raw_expr)) or raw_expr
        else:
            label = raw_expr
        expr = str(label)[:40]
        lift = r.get("lift")
        se = r.get("lift_se")
        sh = r.get("lift_second_half")
        base = r.get("baseline")
        ls = f"{lift:+.4f}" if lift is not None else "  n/a "
        ses = f"{se:.4f}" if se is not None else "  n/a "
        shs = f"{sh:+.4f}" if sh is not None else "  n/a "
        bs = f"{base:.4f}" if base is not None else "  n/a "
        print(f"{expr:40s}  {ls:>8s}  {ses:>8s}  {shs:>8s}  {bs:>8s}  {r.get('passed')}")

    # 默认 dry-run；仅 --apply 才写库 + 写回 lift 拒绝到 experiment_index
    # （--dry-run 为兼容旗标，与 --apply 互斥；dry-run 保持纯只读）
    dry_run = not bool(getattr(args, "apply", False))
    admissions = None
    # 仅对真正跑过 lift 且非 group_gate_fail 的行入库
    scored = [r for r in results if not str(r.get("error") or "").startswith("group_gate")]
    if scored and not dry_run:
        # apply 路径：lift_admission + upsert_lift_admissions（延迟导入，契约同任务 D）
        from factorzen.discovery.factor_library import upsert_lift_admissions

        admissions = upsert_lift_admissions(
            scored,
            market=market,
            root=lib_root,
            meta={
                "eval_start": args.start,
                "eval_end": args.end,
                "universe": getattr(args, "universe", None),
                "horizon": base_ctx.horizon,
                "run_id": f"lift_{date.today().isoformat()}",
                "session_dir": ",".join(sessions),
                "git_sha": get_git_sha(),
                "now": date.today().strftime("%Y-%m-%d"),
                "leaf_map": leaf_map,
            },
            threshold=threshold,
            se_mult=se_mult,
            allow_active=bool(getattr(args, "allow_active", False)),
            # W1 相关性门：复用 lift 已用的物化 memo（库侧 active 面板此前已物化过，
            # memo 命中不重复算；只有本批准入的少数候选需要新物化）。
            # 不传 = 静默漏掉去重，故此处必须接通。
            materialize=memo_mat,
        )
        print(
            f"[factor-library lift-test] 入库：added_active={admissions.get('added_active', 0)} "
            f"added_probation={admissions.get('added_probation', 0)} "
            f"rejected={admissions.get('rejected', 0)}"
            + (
                f" capped_active={admissions.get('capped_active', 0)}"
                if admissions.get("capped_active")
                else ""
            )
        )
    elif dry_run:
        n_pass = sum(1 for r in results if r.get("passed"))
        print(f"[factor-library lift-test] dry-run：通过 {n_pass} 个，不写库（加 --apply 写库）")
    else:
        print("[factor-library lift-test] 无结果行")

    # --apply：lift 拒绝写回 experiment_index（group_gate_fail + below_bar；dry-run 零写入）
    if not dry_run and results:
        try:
            n_idx = _write_cli_lift_rejects_to_index(
                results=results,
                session_items=session_items,
                session_manifests=session_manifests,
                threshold=float(threshold),
                se_mult=float(se_mult),
            )
            if n_idx:
                print(f"[factor-library lift-test] experiment_index 写回 lift_rejected {n_idx} 条")
        except Exception as exc:
            print(
                f"[factor-library lift-test] 警告：lift 拒绝写回 index 失败: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    # 落 lift manifest 到第一个 session（可审计）
    admission_groups = [
        {
            "admission_start": gs,
            "admission_end": ge,
            "n_candidates": len(cs),
        }
        for gs, ge, cs in groups
    ]
    # 单组时顶层 admission_* 与组一致（单 session 零回归）；多组不写并集
    top_adm_start = groups[0][0] if len(groups) == 1 else None
    top_adm_end = groups[0][1] if len(groups) == 1 else None
    # 单组 lift_group 顶层；多组放 list
    lift_group_field: dict | list | None
    if len(lift_groups_meta) == 1:
        lift_group_field = lift_groups_meta[0]
    elif lift_groups_meta:
        lift_group_field = lift_groups_meta
    else:
        lift_group_field = None
    lift_manifest = {
        "market": market,
        "start": args.start,
        "end": args.end,
        "universe": getattr(args, "universe", None),
        "threshold": threshold,
        "top_m": top_m if top_m is not None else 0,
        "seed": seed,
        "admission_start": top_adm_start,
        "admission_end": top_adm_end,
        "admission_groups": admission_groups,
        "horizon": base_ctx.horizon,
        "n_gray_input": n_gray,
        "n_tested": len(results),
        "n_passed": sum(1 for r in results if r.get("passed")),
        "n_lift_evaluated": n_lift_evaluated,
        "dry_run": dry_run,
        "baseline": results[0].get("baseline") if results else None,
        "results": results,
        "sessions": sessions,
        "git_sha": get_git_sha(),
        "admissions": admissions,
        "lift_dropped_coverage": all_dropped,
        "lift_group": lift_group_field,
        # 组门连坐防呆记账（sub_floor_filtered=False 即逃生口开着，如实记但未剔）
        "queue_ic_floor": queue_ic_floor,
        "n_sub_floor": len(all_sub_floor),
        "sub_floor_filtered": not include_sub_floor,
        "lift_dropped_sub_floor": all_sub_floor,
    }
    if truncated_from is not None:
        lift_manifest["truncated_from"] = truncated_from
    if sessions:
        out_man = Path(sessions[0]) / "lift_test_manifest.json"
    else:
        # 仅 --factor：落库根目录可审计
        Path(lib_root).mkdir(parents=True, exist_ok=True)
        out_man = Path(lib_root) / f"lift_test_{market}_manifest.json"
    payload = json.dumps(lift_manifest, ensure_ascii=False, indent=2)
    # 审计保全：先落**不可变时间戳归档**，再覆写稳定名做 latest 指针。
    # 2026-07-17 事故：失败的全量复测覆写掉了此前 top-20 成功的 manifest
    # （n_passed=2 的证据文件丢失）。归档永不覆写 → 成功证据不可被后续 run 抹掉；
    # 稳定名保持原路径原语义 → 下游（文档/人工/测试）零回归。
    archive = _timestamped_sibling(out_man)
    archive.write_text(payload, encoding="utf-8")
    out_man.write_text(payload, encoding="utf-8")
    print(f"[factor-library lift-test] → {out_man}（归档 {archive.name}）")
    return 0


def _cmd_mine_leaderboard(args: argparse.Namespace) -> int:
    from pathlib import Path

    import polars as pl

    csv = Path(args.session_dir) / "candidates.csv"
    if not csv.exists():
        print(f"[mine] 找不到 {csv}", file=sys.stderr)
        return 2
    df = pl.read_csv(csv)
    # 默认只列通过防过拟合护栏的候选；--all 显示全部（老 session 无 passed 列时显示全部）
    if not getattr(args, "all", False) and "passed" in df.columns:
        kept = df.filter(pl.col("passed").cast(pl.Utf8).str.to_lowercase() == "true")
        if kept.height == 0:
            print(
                f"[mine] {csv}: 无候选通过防过拟合护栏；用 --all 查看全部 {df.height} 个候选",
                file=sys.stderr,
            )
            return 0
        df = kept
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=80, tbl_width_chars=200):
        print(df)
    return 0


def _mine_export_alpha_crypto(args: argparse.Namespace) -> int:
    """crypto export-alpha（live CCXT）：读候选表达式 → 当日截面 α → parquet。"""
    from datetime import datetime, timedelta
    from pathlib import Path

    from factorzen.discovery.export import read_candidate_expression
    from factorzen.markets.crypto.mining import export_crypto_alpha
    from factorzen.markets.crypto.profile import build_crypto_profile

    expr = read_candidate_expression(args.session, args.rank, require_passed=not args.all)
    profile = build_crypto_profile(top_n=args.top_n)
    symbols = profile.universe.snapshot(args.date)
    start = (datetime.strptime(args.date, "%Y%m%d") - timedelta(days=args.lookback)).strftime(
        "%Y%m%d"
    )
    cross = export_crypto_alpha(
        profile, expr, symbols, start, args.date, date=args.date, freq=args.freq
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cross.write_parquet(args.out)
    print(
        f"[mine] export-alpha(crypto): rank={args.rank} expr={expr!r} date={args.date} "
        f"→ {args.out} ({cross.height} 个标的)"
    )
    return 0


def _mine_export_alpha_futures(args: argparse.Namespace) -> int:
    """futures export-alpha（Tushare 主力连续）：读候选表达式 → 当日截面 α → parquet。"""
    from datetime import datetime, timedelta
    from pathlib import Path

    from factorzen.discovery.export import read_candidate_expression
    from factorzen.markets.futures.mining import export_futures_alpha
    from factorzen.markets.futures.profile import build_futures_profile

    expr = read_candidate_expression(args.session, args.rank, require_passed=not args.all)
    profile = build_futures_profile(top_n=args.top_n)
    symbols = profile.universe.snapshot(args.date)
    start = (datetime.strptime(args.date, "%Y%m%d") - timedelta(days=args.lookback)).strftime(
        "%Y%m%d"
    )
    cross = export_futures_alpha(profile, expr, symbols, start, args.date, date=args.date)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cross.write_parquet(args.out)
    print(
        f"[mine] export-alpha(futures): rank={args.rank} expr={expr!r} date={args.date} "
        f"→ {args.out} ({cross.height} 个品种)"
    )
    return 0


def _mine_export_alpha_us(args: argparse.Namespace) -> int:
    """us export-alpha（Yahoo 后复权）：读候选表达式 → 当日截面 α → parquet。"""
    from datetime import datetime, timedelta
    from pathlib import Path

    from factorzen.discovery.export import read_candidate_expression
    from factorzen.markets.us.mining import export_us_alpha
    from factorzen.markets.us.profile import build_us_profile

    expr = read_candidate_expression(args.session, args.rank, require_passed=not args.all)
    profile = build_us_profile(top_n=args.top_n)
    symbols = profile.universe.snapshot(args.date)
    start = (datetime.strptime(args.date, "%Y%m%d") - timedelta(days=args.lookback)).strftime(
        "%Y%m%d"
    )
    cross = export_us_alpha(profile, expr, symbols, start, args.date, date=args.date)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cross.write_parquet(args.out)
    print(
        f"[mine] export-alpha(us): rank={args.rank} expr={expr!r} date={args.date} "
        f"→ {args.out} ({cross.height} 个标的)"
    )
    return 0


def _cmd_mine_export_alpha(args: argparse.Namespace) -> int:
    if getattr(args, "market", "ashare") != "crypto" and getattr(args, "freq", "daily") != "daily":
        print("[mine] --freq 仅 crypto 支持;ashare/futures/us 只有 daily", file=sys.stderr)
        return 2
    if getattr(args, "market", "ashare") == "crypto":
        return _mine_export_alpha_crypto(args)
    if getattr(args, "market", "ashare") == "futures":
        return _mine_export_alpha_futures(args)
    if getattr(args, "market", "ashare") == "us":
        return _mine_export_alpha_us(args)
    from factorzen.core.universe import get_universe
    from factorzen.daily.data.context import FactorDataContext
    from factorzen.discovery.export import (
        export_alpha_cross_section,
        read_candidate_expression,
    )

    expr = read_candidate_expression(args.session, args.rank, require_passed=not args.all)
    uni = get_universe(args.date, args.universe)["ts_code"].to_list()
    ctx = FactorDataContext(
        start=args.date,
        end=args.date,
        required_data=["daily", "daily_basic"],
        lookback_days=args.lookback,
        universe=uni,
    )
    out = export_alpha_cross_section(expr, ctx, args.date, args.out)
    import polars as pl

    n = pl.read_parquet(out).height
    print(
        f"[mine] export-alpha: rank={args.rank} expr={expr!r} date={args.date} → {out} ({n} 只股票)"
    )
    return 0


def _validate_overfit_crypto(args: argparse.Namespace) -> int:
    """crypto 单表达式防过拟合验证（live CCXT）。"""
    if not getattr(args, "expression", None):
        print('[validate] crypto 需 --expression "<表达式>"', file=sys.stderr)
        return 1
    from factorzen.markets.crypto.mining import validate_crypto_expression
    from factorzen.markets.crypto.profile import build_crypto_profile

    profile = build_crypto_profile(top_n=args.top_n)
    symbols = profile.universe.snapshot(args.end)
    rep = validate_crypto_expression(
        profile, args.expression, symbols, args.start, args.end, freq=args.freq
    )
    print(
        f"[validate] {args.expression}: IC={rep['ic_mean']:.4f} IR={rep['ir']:.4f} "
        f"DSR_p={rep['dsr_p']:.4f} IC_95%CI=[{rep['ci_lo']:.4f},{rep['ci_hi']:.4f}]"
    )
    print("[validate] 注：单因子 N=1（无多重检验扣减）；PBO 仅适用候选池，此处略。")
    return 0


def _validate_overfit_futures(args: argparse.Namespace) -> int:
    """futures 单表达式防过拟合验证（Tushare 主力连续）。"""
    if not getattr(args, "expression", None):
        print('[validate] futures 需 --expression "<表达式>"', file=sys.stderr)
        return 1
    from factorzen.markets.futures.mining import validate_futures_expression
    from factorzen.markets.futures.profile import build_futures_profile

    profile = build_futures_profile(top_n=args.top_n)
    symbols = profile.universe.snapshot(args.end)
    rep = validate_futures_expression(profile, args.expression, symbols, args.start, args.end)
    print(
        f"[validate] {args.expression}: IC={rep['ic_mean']:.4f} IR={rep['ir']:.4f} "
        f"DSR_p={rep['dsr_p']:.4f} IC_95%CI=[{rep['ci_lo']:.4f},{rep['ci_hi']:.4f}]"
    )
    print("[validate] 注：单因子 N=1（无多重检验扣减）；PBO 仅适用候选池，此处略。")
    return 0


def _validate_overfit_us(args: argparse.Namespace) -> int:
    """us 单表达式防过拟合验证（Yahoo 后复权）。"""
    if not getattr(args, "expression", None):
        print('[validate] us 需 --expression "<表达式>"', file=sys.stderr)
        return 1
    from factorzen.markets.us.mining import validate_us_expression
    from factorzen.markets.us.profile import build_us_profile

    profile = build_us_profile(top_n=args.top_n)
    symbols = profile.universe.snapshot(args.end)
    rep = validate_us_expression(profile, args.expression, symbols, args.start, args.end)
    print(
        f"[validate] {args.expression}: IC={rep['ic_mean']:.4f} IR={rep['ir']:.4f} "
        f"DSR_p={rep['dsr_p']:.4f} IC_95%CI=[{rep['ci_lo']:.4f},{rep['ci_hi']:.4f}]"
    )
    print("[validate] 注：单因子 N=1（无多重检验扣减）；PBO 仅适用候选池，此处略。")
    return 0


def _cmd_validate_overfit(args: argparse.Namespace) -> int:
    if getattr(args, "market", "ashare") != "crypto" and getattr(args, "freq", "daily") != "daily":
        print("[validate] --freq 仅 crypto 支持;ashare/futures/us 只有 daily", file=sys.stderr)
        return 2
    if getattr(args, "market", "ashare") == "crypto":
        return _validate_overfit_crypto(args)
    if getattr(args, "market", "ashare") == "futures":
        return _validate_overfit_futures(args)
    if getattr(args, "market", "ashare") == "us":
        return _validate_overfit_us(args)
    from factorzen.daily.data.context import FactorDataContext
    from factorzen.daily.factors.registry import get_factor
    from factorzen.discovery.library_provider import load_library_factors
    from factorzen.discovery.scoring import ic_overfit_report

    # factor 位置参数 nargs='?' 可缺省；缺省时给友好用法提示，而非 get_factor(None) 裸 KeyError
    if not getattr(args, "factor", None):
        print(
            "[validate] 缺少因子名：用法 fz validate overfit <factor> --start ... --end ...",
            file=sys.stderr,
        )
        return 2
    # ashare daily：注入 library expression 因子（库损坏不崩）
    try:
        load_library_factors()
    except ValueError as e:
        print(f"[validate] load_library_factors 跳过: {e}", file=sys.stderr)
    factor = get_factor(args.factor)()
    uni = None
    if getattr(args, "universe", None):
        from factorzen.core.universe import get_universe

        uni = get_universe(args.end, args.universe)["ts_code"].to_list()
    ctx = FactorDataContext(
        start=args.start,
        end=args.end,
        required_data=["daily", "daily_basic"],
        lookback_days=getattr(factor, "lookback_days", 60),
        universe=uni,
    )
    fdf = factor.compute(ctx)
    rep = ic_overfit_report(fdf, ctx.daily.collect(), train_ratio=1.0)
    print(
        f"[validate] {args.factor}: IC={rep['ic_mean']:.4f} IR={rep['ir']:.4f} "
        f"DSR_p={rep['dsr_p']:.4f} IC_95%CI=[{rep['ci_lo']:.4f},{rep['ci_hi']:.4f}]"
    )
    print("[validate] 注：单因子 N=1（无多重检验扣减）；PBO 仅适用候选池，此处略。")
    return 0


def _portfolio_build_crypto(args: argparse.Namespace) -> int:
    """crypto 市场中性做空组合（live CCXT）。"""
    import polars as pl

    from factorzen.markets.crypto.portfolio import build_crypto_portfolio
    from factorzen.markets.crypto.profile import build_crypto_profile

    profile = build_crypto_profile(top_n=args.top_n)
    symbols = profile.universe.snapshot(args.end)
    adf = (
        pl.read_parquet(args.alpha_file)
        if args.alpha_file.endswith(".parquet")
        else pl.read_csv(args.alpha_file)
    )
    _end = args.end or ""
    signal_date = (
        f"{_end[:4]}-{_end[4:6]}-{_end[6:]}" if len(_end) == 8 and _end.isdigit() else _end
    )
    res = build_crypto_portfolio(
        profile,
        adf,
        symbols,
        args.start,
        args.end,
        market_neutral=True,
        w_max=args.w_max,
        gross_limit=args.gross_limit,
        risk_aversion=args.lam,
        signal_date=signal_date,
        freq=args.freq,
    )
    print(
        f"[portfolio] crypto status={res['status']} holdings={res['n_holdings']} → {res['run_dir']}"
    )
    return 0


def _cmd_portfolio_build(args: argparse.Namespace) -> int:
    if getattr(args, "market", "ashare") != "crypto" and getattr(args, "freq", "daily") != "daily":
        print("[portfolio] --freq 仅 crypto 支持;ashare 只有 daily", file=sys.stderr)
        return 2
    if getattr(args, "market", "ashare") == "crypto" and getattr(args, "risk_dir", None):
        print("[portfolio] --risk-dir 仅 ashare 支持;crypto 不支持", file=sys.stderr)
        return 2
    if getattr(args, "market", "ashare") == "crypto":
        return _portfolio_build_crypto(args)
    import numpy as np
    import polars as pl

    from factorzen.core import loader
    from factorzen.core.universe import get_universe
    from factorzen.pipelines.portfolio_build import run_portfolio
    from factorzen.pipelines.risk_build import load_risk_inputs, load_risk_model_result
    from factorzen.risk.model import RiskModel

    stocks = get_universe(args.end, args.universe)
    uni = stocks["ts_code"].to_list()
    if getattr(args, "risk_dir", None):
        # 复用 fz risk build 产物，跳过 load_risk_inputs / RiskModel.build（保留 get_universe 供 sectors）
        try:
            risk_result = load_risk_model_result(args.risk_dir)
        except ValueError as e:
            print(f"[portfolio] --risk-dir 加载失败: {e}", file=sys.stderr)
            return 2
    else:
        # 补 lookback 历史预热滚动风格因子（否则 build 静默退化为少数因子，见 load_risk_inputs）
        daily, daily_basic = load_risk_inputs(loader, args.start, args.end, uni)
        risk_result = RiskModel().build(daily, daily_basic, stocks, args.start, args.end)
    codes = risk_result.factor_exposures.codes
    # α：从 --alpha-file 读取截面信号(ts_code + alpha)，对齐 codes 顺序(缺失填 0)
    adf = (
        pl.read_parquet(args.alpha_file)
        if args.alpha_file.endswith(".parquet")
        else pl.read_csv(args.alpha_file)
    )
    amap = dict(zip(adf["ts_code"].to_list(), adf["alpha"].to_list(), strict=False))
    alpha = np.array([float(amap.get(c, 0.0)) for c in codes])
    neutral = (
        [n for n in risk_result.factor_names if n.startswith("ind_")]
        if args.industry_neutral
        else None
    )
    # --industry-neutral 使用 universe 等权基准：target = X_s.T @ w_bench（等权行业暴露）
    # 而非绝对 0；raw one-hot 列下 target=0 + long_only + Σw=1 必然 infeasible。
    # MVP：等权基准（真实指数基准权重留后续扩展）。
    bench_weights = np.full(len(codes), 1.0 / len(codes)) if args.industry_neutral else None
    _ind_map = dict(zip(stocks["ts_code"].to_list(), stocks["industry"].to_list(), strict=False))
    sectors = [(_ind_map.get(c) or "") for c in codes]
    # 将 args.end (YYYYMMDD) 转成 ISO 格式 YYYY-MM-DD，供 sim 的 date.fromisoformat() 解析
    _end: str = args.end or ""
    if len(_end) == 8 and _end.isdigit():
        _signal_date: str | None = f"{_end[:4]}-{_end[4:6]}-{_end[6:]}"
    else:
        _signal_date = _end or None
    res = run_portfolio(
        alpha,
        risk_result,
        codes=codes,
        stock_returns=np.zeros(len(codes)),
        sectors=sectors,
        factor_returns_latest={},
        risk_aversion=args.lam,
        w_max=args.w_max,
        neutral_factors=neutral,
        turnover_budget=args.turnover,
        bench_weights=bench_weights,
        signal_date=_signal_date,
        out_dir=getattr(args, "out_dir", str(PORTFOLIOS_DIR)),
        run_id=getattr(args, "run_id", None) or args.end,  # 默认按 end 日期分目录，多期不覆盖
    )
    print(f"[portfolio] status={res['status']} holdings={res['n_holdings']} → {res['run_dir']}")
    return 0


def _cmd_risk_build(args: argparse.Namespace) -> int:
    from factorzen.core import loader
    from factorzen.core.universe import get_universe
    from factorzen.pipelines.risk_build import load_risk_inputs, run_risk_build

    stocks = get_universe(args.end, args.universe)  # 含 industry 列
    uni = stocks["ts_code"].to_list()
    # 补 lookback 历史预热滚动风格因子（否则 build 静默退化为少数因子，见 load_risk_inputs）
    daily, daily_basic = load_risk_inputs(loader, args.start, args.end, uni)
    res = run_risk_build(
        daily,
        daily_basic,
        stocks,
        args.start,
        args.end,
        cov_half_life=args.cov_half_life,
        nw_lags=args.nw_lags,
        spec_half_life=args.spec_half_life,
        spec_shrinkage=args.spec_shrinkage,
    )
    n_valid = res.get("n_valid_dates", "?")
    n_mis = res.get("n_factor_mismatch", res.get("n_dropped_dates", "?"))
    print(
        f"[risk] factors={len(res['factor_names'])} R2={res['r_squared']:.4f} "
        f"valid_days={n_valid} n_factor_mismatch={n_mis} → {res['run_dir']}"
    )
    return 0


def _strategies_rebalance_dates(trade_dates: list, freq: str) -> list:
    """从交易日列表抽调仓日：daily=全日；weekly/monthly=该周/月首个交易日。"""
    if freq == "daily":
        return list(trade_dates)
    out: list = []
    prev = None
    for d in trade_dates:
        if freq == "weekly":
            key = d.isocalendar()[:2]
        else:  # monthly
            key = (d.year, d.month)
        if key != prev:
            out.append(d)
            prev = key
    return out


def _cmd_strategies_run(args: argparse.Namespace) -> int:
    """``fz strategies run <name>``：生成 weights 产物 → sim，打印 run_dir 与关键指标。"""
    from datetime import datetime, timedelta
    from pathlib import Path

    import polars as pl

    from factorzen.core import loader
    from factorzen.strategies.runner import run_strategy_simulation

    name = args.name
    rid = args.run_id or name
    root = Path(args.out_dir) / rid
    products_dir = root / "products"
    products_dir.mkdir(parents=True, exist_ok=True)

    def _extend_start(start: str, cal_days: int) -> str:
        d = datetime.strptime(start, "%Y%m%d") - timedelta(days=cal_days)
        return d.strftime("%Y%m%d")

    # ── 生成 run_dirs ──
    if name == "trend_timing":
        top_n = args.top_n if args.top_n is not None else 50
        ma_window = int(args.ma_window)
        daily = loader.fetch_daily(args.start, args.end)
        if daily.is_empty():
            print("[strategies] daily 面板为空", file=sys.stderr)
            return 2
        idx_start = _extend_start(args.start, max(ma_window * 3, 400))
        index_daily = loader.fetch_index_daily(args.index_code, idx_start, args.end)
        trade_dates = sorted(daily.select("trade_date").unique()["trade_date"].to_list())
        max_td = trade_dates[-1]
        rebalance = [
            d
            for d in _strategies_rebalance_dates(trade_dates, args.rebalance)
            if d < max_td
        ]
        if not rebalance:
            print("[strategies] 调仓日为空（窗口过短？）", file=sys.stderr)
            return 2
        from factorzen.strategies.trend_timing import generate_trend_timing_products

        run_dirs = generate_trend_timing_products(
            str(products_dir),
            index_daily,
            daily,
            rebalance,
            index_code=args.index_code,
            ma_window=ma_window,
            top_n=top_n,
            timing=bool(args.timing),
        )
    elif name == "momentum_rotation":
        top_n = args.top_n if args.top_n is not None else 50
        lookback = int(args.lookback)
        daily = loader.fetch_daily(args.start, args.end)
        if daily.is_empty():
            print("[strategies] daily 面板为空", file=sys.stderr)
            return 2
        codes = [c.strip() for c in str(args.index_codes).split(",") if c.strip()]
        if not codes:
            print("[strategies] index_codes 为空", file=sys.stderr)
            return 2
        idx_start = _extend_start(args.start, max(lookback * 3, 400))
        index_dailies = {
            c: loader.fetch_index_daily(c, idx_start, args.end) for c in codes
        }
        trade_dates = sorted(daily.select("trade_date").unique()["trade_date"].to_list())
        max_td = trade_dates[-1]
        rebalance = [
            d
            for d in _strategies_rebalance_dates(trade_dates, args.rebalance)
            if d < max_td
        ]
        if not rebalance:
            print("[strategies] 调仓日为空（窗口过短？）", file=sys.stderr)
            return 2
        from factorzen.strategies.momentum_rotation import (
            generate_momentum_rotation_products,
        )

        run_dirs = generate_momentum_rotation_products(
            str(products_dir),
            index_dailies,
            daily,
            rebalance,
            lookback=lookback,
            top_n=top_n,
        )
    elif name in ("sleeve", "quantile_group"):
        if not args.scores or not args.score_col:
            print(
                f"[strategies] {name} 需要 --set scores=<parquet> --set score_col=<列名>",
                file=sys.stderr,
            )
            return 2
        from factorzen.pipelines.combine_backtest import load_market_panel
        from factorzen.pipelines.daily_single import filter_frame_by_membership

        market = load_market_panel(
            start=args.start, end=args.end, universe=args.universe, market="ashare"
        )
        daily = market["price_df"]
        trade_dates = sorted(daily.select("trade_date").unique()["trade_date"].to_list())
        if not trade_dates:
            print("[strategies] 行情交易日为空", file=sys.stderr)
            return 2
        scores = pl.read_parquet(
            args.scores, columns=["trade_date", "ts_code", args.score_col]
        )
        scores = scores.filter(
            pl.col("trade_date").is_between(trade_dates[0], trade_dates[-1])
        )
        scores = filter_frame_by_membership(scores, market["membership"])
        if scores.is_empty():
            print("[strategies] PIT membership 过滤后分数截面为空", file=sys.stderr)
            return 2
        if name == "sleeve":
            top_n = args.top_n if args.top_n is not None else 200
            from factorzen.strategies.sleeve import generate_sleeve_products

            run_dirs = generate_sleeve_products(
                str(products_dir),
                scores,
                score_col=args.score_col,
                top_n=top_n,
                holding_days=int(args.holding_days),
                trade_dates=trade_dates,
                direction=args.direction,
            )
        else:
            from factorzen.strategies.quantile_group import (
                generate_quantile_group_products,
            )

            run_dirs = generate_quantile_group_products(
                str(products_dir),
                scores,
                score_col=args.score_col,
                n_groups=int(args.n_groups),
                group=int(args.group),
                trade_dates=trade_dates,
            )
    else:
        print(f"[strategies] 未知策略 {name!r}", file=sys.stderr)
        return 2

    if not run_dirs:
        print("[strategies] 未生成任何 weights 产物", file=sys.stderr)
        return 2

    res = run_strategy_simulation(
        run_dirs,
        daily,
        out_dir=str(root),
        run_id="sim",
    )
    sharpe = res.get("sharpe")
    max_dd = res.get("max_dd")
    ann_ret = res.get("ann_ret")

    def _fmt(v: object) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return f"{float(v):.4f}"
        try:
            return f"{float(str(v)):.4f}"
        except (TypeError, ValueError):
            return str(v)

    print(
        f"[strategies] name={name} products={len(run_dirs)} "
        f"run_dir={res['run_dir']} "
        f"sharpe={_fmt(sharpe)} max_dd={_fmt(max_dd)} ann_ret={_fmt(ann_ret)}"
    )
    return 0


def _cmd_sim_run(args: argparse.Namespace) -> int:
    if getattr(args, "market", "ashare") != "crypto" and getattr(args, "freq", "daily") != "daily":
        print("[sim] --freq 仅 crypto 支持;ashare 只有 daily", file=sys.stderr)
        return 2
    from pathlib import Path

    portfolio_root = Path(args.portfolio_dir)
    if not portfolio_root.exists():
        print(f"[sim] portfolio-dir not found: {portfolio_root}", file=sys.stderr)
        return 2

    run_dirs = sorted(
        p
        for p in portfolio_root.iterdir()
        # 同时要求 manifest.json：portfolio_build 先写 weights 再写 manifest，中途崩溃会
        # 留下含 weights 无 manifest 的半成品目录，_load_weights_by_date 无条件读 manifest
        # 会 FileNotFoundError 炸掉整批 sim。
        if p.is_dir() and (p / "weights.parquet").exists() and (p / "manifest.json").exists()
    )
    if not run_dirs:
        print(f"[sim] no portfolio run dirs found under {portfolio_root}", file=sys.stderr)
        return 2

    if getattr(args, "market", "ashare") == "crypto":
        from factorzen.markets.crypto.backtest import run_crypto_simulation
        from factorzen.markets.crypto.profile import build_crypto_profile

        profile = build_crypto_profile(top_n=getattr(args, "top_n", 50))
        res = run_crypto_simulation(
            [str(p) for p in run_dirs],
            profile,
            args.start,
            args.end,
            out_dir=str(SIM_DIR),
            run_id=args.run_id,
            freq=args.freq,
        )
    else:
        from factorzen.core import loader
        from factorzen.sim.engine import run_portfolio_simulation

        daily = loader.fetch_daily(args.start, args.end)
        res = run_portfolio_simulation(
            [str(p) for p in run_dirs],
            daily,
            out_dir=str(SIM_DIR),
            run_id=args.run_id,
        )
    print(
        f"[sim] run_dir={res['run_dir']} "
        f"sharpe={res['sharpe']:.4f} "
        f"max_dd={res['max_dd']:.4f} "
        f"ann_ret={res['ann_ret']:.4f}"
    )
    return 0


def _cmd_report_portfolio(args: argparse.Namespace) -> int:
    import json as _json
    from pathlib import Path

    import polars as pl

    from factorzen.reports.portfolio_report import generate_portfolio_report

    sim_dir = Path(args.sim_dir) if args.sim_dir else None

    # 读 metrics.json + sim manifest（含 market）
    metrics: dict = {}
    run_id = "portfolio"
    market = getattr(args, "market", None) or "ashare"
    if sim_dir is not None:
        metrics_path = sim_dir / "metrics.json"
        if metrics_path.exists():
            metrics = _json.loads(metrics_path.read_text(encoding="utf-8"))
            run_id = sim_dir.name
        sim_mf = sim_dir / "manifest.json"
        if sim_mf.exists() and not getattr(args, "market", None):
            # 未显式指定 --market 时，从 sim manifest 自动识别
            market = _json.loads(sim_mf.read_text(encoding="utf-8")).get("market", market)

    # 读 portfolio_dir 产物
    attribution_df: pl.DataFrame | None = None
    risk_summary_df: pl.DataFrame | None = None
    portfolio_manifest: dict | None = None
    if args.portfolio_dir:
        pdir = Path(args.portfolio_dir)
        att_path = pdir / "attribution.csv"
        if att_path.exists():
            attribution_df = pl.read_csv(att_path)
        risk_path = pdir / "risk_summary.csv"
        if risk_path.exists():
            risk_summary_df = pl.read_csv(risk_path)
        mf_path = pdir / "manifest.json"
        if mf_path.exists():
            portfolio_manifest = _json.loads(mf_path.read_text(encoding="utf-8"))

    # 尝试从 sim_dir/nav.parquet 重建轻量 sim_result 对象，供两个图表函数使用：
    # _make_returns_chart 只访问 .nav 渲染净值曲线；_make_monthly_return_heatmap
    # 只访问 .returns 渲染月度收益热力图（用 _safe_attr 安全取值，缺失该属性时
    # 返回 None、函数静默跳过不渲染）。nav.parquet 本身已含计算热力图所需的
    # net_return 列，故 .returns 直接复用同一份 nav_df 即可——
    # 早期版本只设置了 .nav，导致热力图在这条唯一的生产路径下恒为死代码。
    sim_result = None
    if sim_dir is not None:
        nav_path = sim_dir / "nav.parquet"
        if nav_path.exists():
            from types import SimpleNamespace

            _nav_df = pl.read_parquet(nav_path)
            if not _nav_df.is_empty():
                # returns=nav_df（含 net_return）供月度收益热力图渲染
                sim_result = SimpleNamespace(nav=_nav_df, returns=_nav_df)

    html = generate_portfolio_report(
        sim_result=sim_result,
        metrics=metrics,
        attribution_df=attribution_df,
        risk_summary_df=risk_summary_df,
        portfolio_manifest=portfolio_manifest,
        market=market,
    )

    # 输出路径
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = REPORTS_DIR / f"portfolio_{run_id}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(out_path)
    return 0


def _cmd_sim_show(args: argparse.Namespace) -> int:
    from pathlib import Path

    metrics_path = Path(args.sim_dir) / "metrics.json"
    if not metrics_path.exists():
        print(f"[sim] metrics.json not found: {metrics_path}", file=sys.stderr)
        return 2

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    keys = ["ann_ret", "sharpe", "max_dd", "ann_turnover", "total_cost"]
    for k in keys:
        if k in metrics:
            print(f"{k}: {metrics[k]}")
    extras = {k: v for k, v in metrics.items() if k not in keys}
    if extras:
        print(json.dumps(extras, ensure_ascii=False, indent=2))
    return 0


def _cmd_live_replay(args: argparse.Namespace) -> int:
    from datetime import date as _date

    import polars as pl

    from factorzen.core import loader
    from factorzen.core.universe import get_universe
    from factorzen.execution.drivers import run_replay

    stocks = get_universe(args.end, args.universe) if args.universe else None
    daily = loader.fetch_daily(args.start, args.end)
    if stocks is not None:
        daily = daily.filter(pl.col("ts_code").is_in(stocks["ts_code"].to_list()))
    out = run_replay(
        session_dir=args.session_dir,
        portfolio_run_dirs=args.portfolio_run_dirs,
        daily=daily,
        initial_cash=args.initial_cash,
        from_date=_date.fromisoformat(args.from_date) if args.from_date else None,
        to_date=_date.fromisoformat(args.to_date) if args.to_date else None,
        seed=args.seed,
    )
    print(
        f"replay 完成: {out['n_steps']} 步, 终值 NAV={out['final_nav']:.2f} → {out['session_dir']}"
    )
    return 0


def _cmd_live_init(args: argparse.Namespace) -> int:
    from factorzen.execution.store import SessionStore

    SessionStore(args.session_dir).init(
        {
            "broker": args.broker,
            "command": ["fz", "live", "init"],
            "initial_cash": args.initial_cash,
            "slippage_bps": args.slippage_bps,
        }
    )
    print(f"[live] init 会话 → {args.session_dir}")
    return 0


def _cmd_live_step(args: argparse.Namespace) -> int:
    import json as _json
    from datetime import date as _date
    from pathlib import Path

    import polars as pl

    from factorzen.core import loader
    from factorzen.core.universe import get_universe
    from factorzen.execution.drivers import run_daily_step

    stocks = get_universe(args.end, args.universe) if args.universe else None
    daily = loader.fetch_daily(args.start, args.end)
    if stocks is not None:
        daily = daily.filter(pl.col("ts_code").is_in(stocks["ts_code"].to_list()))
    cfg = _json.loads((Path(args.session_dir) / "manifest.json").read_text()).get("config", {})
    cfg.setdefault("initial_cash", 1_000_000.0)
    cfg.setdefault("slippage_bps", 0.0)
    d = _date.fromisoformat(f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}")
    out = run_daily_step(args.session_dir, d, args.portfolio_run_dirs, daily, config=cfg)
    status = "跳过(已记录)" if out["skipped"] else f"{out['n_fills']}成交 NAV={out['nav_after']}"
    print(f"[live] step {out['as_of']}: {status}")
    return 0


def _cmd_live_status(args: argparse.Namespace) -> int:
    from factorzen.execution.store import SessionStore

    s = SessionStore(args.session_dir)
    st = s.load_state()
    nav = s.nav_frame()
    last = nav["as_of_date"][-1] if nav.height else "(无)"
    # state.json 有两种形状：可续跑态（run_daily_step 落的 broker.state()=
    # {cash: float, pos, order_seq}）或显示视图（run_replay 留的 step() 返回=
    # {positions, cash: {available,total_asset,market_value}}）。两者都要兼容，
    # 不能假设只有前者，否则对 replay session 会打印整个 cash dict、且持仓数
    # 因取错键（pos vs positions）恒报 0。
    if st is None:
        cash: float | str = "N/A"
        n_pos = 0
    else:
        cash_raw = st.get("cash")
        if isinstance(cash_raw, dict):
            avail = cash_raw.get("available")
            total = cash_raw.get("total_asset")
            val = avail if avail is not None else total
            cash = float(val) if isinstance(val, int | float) else "N/A"
        elif isinstance(cash_raw, int | float):
            cash = float(cash_raw)
        else:
            cash = "N/A"
        positions = st.get("pos")
        if positions is None:
            positions = st.get("positions", {})
        n_pos = len(positions)
    print(f"[live] 末记录日={last} 现金={cash} 持仓数={n_pos}")
    return 0


def _cmd_live_report(args: argparse.Namespace) -> int:
    import json as _json
    from pathlib import Path

    import polars as pl

    from factorzen.core import loader
    from factorzen.core.universe import get_universe
    from factorzen.execution.attribution import build_attribution_report

    stocks = get_universe(args.end, args.universe) if args.universe else None
    daily = loader.fetch_daily(args.start, args.end)
    if stocks is not None:
        daily = daily.filter(pl.col("ts_code").is_in(stocks["ts_code"].to_list()))
    cfg = _json.loads((Path(args.session_dir) / "manifest.json").read_text()).get("config", {})
    rep = build_attribution_report(
        args.session_dir,
        args.portfolio_run_dirs,
        daily,
        initial_cash=float(cfg.get("initial_cash", 1_000_000.0)),
    )
    print(
        f"[live] 归因: 总缺口={rep['total_gap_ann_ret'] * 1e4:.1f}bps/年 "
        f"成本={rep['cost_bps']:.1f} 滑点={rep['slippage_bps']:.1f} residual={rep['residual_bps']:.1f} "
        f"| 年化换手(双边)={rep.get('ann_turnover', 0.0):.2f} 成交={rep.get('n_fills', 0)}笔"
    )
    for r, v in rep["missed_by_reason"].items():
        print(f"        未成交[{r}]: {v['count']}次 名义额={v['notional']:.0f}")
    return 0


def _cmd_combine_run(args: argparse.Namespace) -> int:
    from factorzen.pipelines.factor_combine import run_factor_combination

    methods = None if args.methods == "all" else args.methods.split(",")
    res = run_factor_combination(
        factor_files=args.factors,
        ret_file=args.ret,
        train_days=args.train_days,
        test_days=args.test_days,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
        methods=methods,
        seed=args.seed,
        out_dir=args.out_dir,
        run_id=args.run_id,
        command=["combine", "run"],
    )
    print(f"[combine] 完成 → {res['run_dir']}")
    print(res["comparison"])
    return 0


def _cmd_combine_from_session(args: argparse.Namespace) -> int:
    from factorzen.pipelines.factor_combine import combine_from_session

    methods = None if args.methods == "all" else args.methods.split(",")
    res = combine_from_session(
        session_dirs=args.session,
        start=args.start,
        end=args.end,
        universe=args.universe,
        horizon=args.horizon,
        passed_only=not args.all,
        top_n=args.top_n,
        decorr_threshold=args.decorr_threshold,
        methods=methods,
        seed=args.seed,
        out_dir=args.out_dir,
        run_id=args.run_id,
        train_days=args.train_days,
        test_days=args.test_days,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
    )
    print(f"[combine] 因子库组合完成 → {res['run_dir']}")
    print(
        f"[combine] 纳入 {len(res['factors_used'])} 个因子；"
        f"去相关剔除 {len(res['dropped_correlated'])} 个近亲"
    )
    for d in res["dropped_correlated"]:
        ident = d.get("identity", d.get("expression"))
        print(f"[combine]   ✗ {ident} → 与 {d['corr_with']} 相关 {d['corr']:.2f}")
    print(res["comparison"])
    return 0


def _cmd_combine_from_library(args: argparse.Namespace) -> int:
    """因子库选品 → 物化 → 四方法 OOS；ValueError → stderr + exit 2。"""
    from factorzen.pipelines.factor_combine import combine_from_library

    methods = None if args.methods == "all" else args.methods.split(",")
    statuses = args.statuses
    if isinstance(statuses, str):
        statuses = tuple(p.strip() for p in statuses.split(",") if p.strip())
    try:
        res = combine_from_library(
            market=args.market,
            statuses=tuple(statuses),
            library_root=args.library_root,
            start=args.start,
            end=args.end,
            universe=args.universe,
            horizon=args.horizon,
            top_n=args.top_n,
            decorr_threshold=args.decorr_threshold,
            methods=methods,
            seed=args.seed,
            out_dir=args.out_dir,
            run_id=args.run_id,
            train_days=args.train_days,
            test_days=args.test_days,
            purge_days=args.purge_days,
            embargo_days=args.embargo_days,
            no_store=bool(getattr(args, "no_store", False)),
        )
    except ValueError as exc:
        print(f"[combine] {exc}", file=sys.stderr)
        return 2

    n_selected = len(res.get("factors_status") or {})
    n_skipped = len(res.get("skipped_materialize") or [])
    n_mat = n_selected - n_skipped
    n_drop = len(res.get("dropped_correlated") or [])
    print(f"[combine] 库→组合完成 → {res['run_dir']}")
    print(
        f"[combine] 选品 {n_selected}、物化成功 {n_mat}、"
        f"去相关剔除 {n_drop}、纳入 {len(res['factors_used'])}"
    )
    if res.get("truncated_from") is not None:
        print(f"[combine] top_n 截断自 {res['truncated_from']}")
    for d in res.get("dropped_correlated") or []:
        ident = d.get("identity", d.get("expression"))
        print(f"[combine]   ✗ {ident} → 与 {d['corr_with']} 相关 {d['corr']:.2f}")
    print(res["comparison"])
    return 0


def _cmd_combine_backtest(args: argparse.Namespace) -> int:
    """组合 OOS 分数 → 日环策略回测桥；ValueError → stderr + exit 2。"""
    from factorzen.pipelines.combine_backtest import cmd_combine_backtest

    return cmd_combine_backtest(args)


def _ops_as_of(date_arg: str | None):
    from datetime import date as _date

    if date_arg:
        return _date.fromisoformat(f"{date_arg[:4]}-{date_arg[4:6]}-{date_arg[6:]}")
    return _date.today()


def _cmd_ops_daily(args: argparse.Namespace) -> int:
    from factorzen.ops.config import load_ops_config
    from factorzen.ops.runner import run_ops_daily

    cfg = load_ops_config(args.config)
    return run_ops_daily(cfg, _ops_as_of(args.date))


def _cmd_ops_status(args: argparse.Namespace) -> int:
    import json as _json

    from factorzen.ops.config import load_ops_config
    from factorzen.ops.state import OpsState

    cfg = load_ops_config(args.config)
    summary = OpsState(cfg.state_dir, _ops_as_of(args.date)).summary()
    print(_json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    from factorzen.cli.parser import build_parser as assemble_parser

    return assemble_parser(sys.modules[__name__])


def _schema_for_parsed_args(
    args: argparse.Namespace,
) -> dict[str, tuple[type, Any, tuple[str, ...] | None]] | None:
    """按已解析命令选 --set schema（parse 后 / handler 前统一注入）。"""
    cmd = getattr(args, "command", None)
    if cmd == "mine":
        sub = getattr(args, "mine_command", None)
        if sub == "search":
            return _MINE_SEARCH_SET
        if sub == "agent":
            return _MINE_AGENT_SET
        if sub == "team":
            return _MINE_TEAM_SET
    if cmd == "factor-library" and getattr(args, "factor_library_command", None) == "lift-test":
        return _LIFT_TEST_SET
    if cmd == "strategies" and getattr(args, "strategies_command", None) == "run":
        return _STRATEGIES_RUN_SET
    return None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    effective = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(effective)
    # 落 manifest 用（铁律#3）。记「实际传入的 argv」而非 sys.argv——main() 被程序化调用时
    # （如 research run 编排器）sys.argv 是外层进程的命令行，会记错。
    args.command_line = "fz " + " ".join(effective)
    schema = _schema_for_parsed_args(args)
    if schema is not None:
        err = _apply_set_overrides(args, schema)
        if err is not None:
            return err
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
