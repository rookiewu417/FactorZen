"""Unified FactorZen command line interface."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from factorzen.config.settings import (
    DAILY_FACTORS_DIR,
    FACTOR_EVALUATIONS_DIR,
    FACTOR_LIBRARY_DIR,
    PORTFOLIOS_DIR,
    REPORTS_DIR,
    ROOT,
    SIM_DIR,
)
from factorzen.experiments.run_paths import run_dir


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
    target = ROOT / "workspace" / "factors" / args.freq / f"{args.name}.py"
    if target.exists() and not args.force:
        print(f"Factor already exists: {target}", file=sys.stderr)
        return 2
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        _factor_template(_class_name(args.name), args.name, args.freq),
        encoding="utf-8",
    )
    print(target)
    return 0


def _cmd_factor_list(args: argparse.Namespace) -> int:
    if args.freq == "intraday":
        from factorzen.intraday.factors.registry import list_factors
    else:
        from factorzen.daily.factors.registry import list_factors

    for name in list_factors():
        print(name)
    return 0


def _cmd_factor_test(args: argparse.Namespace) -> int:
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

    old_argv = sys.argv
    try:
        sys.argv = forwarded
        daily_single.main()
    finally:
        sys.argv = old_argv
    return 0


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


def _cmd_report_build(args: argparse.Namespace) -> int:
    from factorzen.pipelines import generate_report

    factor_name = args.name or args.factor
    forwarded = [f"fz report {args.report_command}"]
    if factor_name:
        forwarded.extend(["--factor", factor_name])
    if args.start:
        forwarded.extend(["--start", args.start])
    if args.end:
        forwarded.extend(["--end", args.end])
    if args.universe:
        forwarded.extend(["--universe", args.universe])
    forwarded.extend(["--frequency", args.frequency])
    if args.benchmark:
        forwarded.extend(["--benchmark", args.benchmark])
    if args.config:
        forwarded.extend(["--config", args.config])
    if args.reuse:
        forwarded.append("--reuse")

    old_argv = sys.argv
    try:
        sys.argv = forwarded
        generate_report.main()
    finally:
        sys.argv = old_argv
    return 0


def _cmd_report_open(args: argparse.Namespace) -> int:
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
    data_type = f"{version}/{freq}"
    print("\nmonth\tpartition_exists")
    for ym in months:
        y_str, m_str = ym.split("-")
        y, m = int(y_str), int(m_str)
        ok = partition_exists(data_type, y, m, base_dir=INTRADAY_FEATURES_DIR)
        print(f"{ym}\t{ok}")
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


def _cmd_runs_show(args: argparse.Namespace) -> int:
    manifest_path = FACTOR_EVALUATIONS_DIR / args.run_id / "manifest.json"
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
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
        profile, symbols, args.start, args.end,
        n_trials=args.trials, top_k=args.top_k, seed=args.seed, method=args.method,
        freq=args.freq,
        # 六个护栏/并行参数经 **session_kw 透传到 run_session，否则用户设的
        # --dsr-alpha/--holdout-ratio/--workers 等被静默丢弃、按默认执行。
        holdout_ratio=args.holdout_ratio, train_ratio=args.train_ratio,
        decorr_threshold=args.decorr_threshold, min_n_train=args.min_n_train,
        dsr_alpha=args.dsr_alpha, workers=args.workers,
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
        profile, symbols, args.start, args.end,
        n_trials=args.trials, top_k=args.top_k, seed=args.seed, method=args.method,
        holdout_ratio=args.holdout_ratio, train_ratio=args.train_ratio,
        decorr_threshold=args.decorr_threshold, min_n_train=args.min_n_train,
        dsr_alpha=args.dsr_alpha, workers=args.workers,
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
        profile, symbols, args.start, args.end,
        n_trials=args.trials, top_k=args.top_k, seed=args.seed, method=args.method,
        holdout_ratio=args.holdout_ratio, train_ratio=args.train_ratio,
        decorr_threshold=args.decorr_threshold, min_n_train=args.min_n_train,
        dsr_alpha=args.dsr_alpha, workers=args.workers,
        update_library=not getattr(args, "no_library", False),
        library_orthogonal=not getattr(args, "no_library_orthogonal", False),
        objective=getattr(args, "objective", "residual"),
    )
    sd = res["session_dir"]
    print(f"[mine] us 完成：{len(res['candidates'])} 个候选 / {len(symbols)} 标的 → {sd}")
    return 0


def _cmd_mine_search(args: argparse.Namespace) -> int:
    if getattr(args, "market", "ashare") not in ("crypto",) and getattr(args, "freq", "daily") != "daily":
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
    )
    sd = res["session_dir"]
    print(f"[mine] 完成：{len(res['candidates'])} 个候选 → {sd}")
    print(f"[mine] 复现：cp {sd}/exported/*.py {DAILY_FACTORS_DIR}/ && "
          f"fz factor run <name> --set preprocessing.neutralize=false")
    print("[mine] 注：candidates.csv 的 IC 为挖掘内估计(plain zscore)；"
          "fz factor run 默认带中性化，IC parity 需 neutralize=false")
    return 0


def _cmd_research_run(args: argparse.Namespace) -> int:
    from factorzen.pipelines.research_run import run_research

    res = run_research(
        start=args.start, end=args.end, universe=args.universe,
        n_trials=args.trials, method=args.method, seed=args.seed, top_k=args.top_k,
        rebalance_days=args.rebalance_days, warmup=args.warmup,
        risk_aversion=args.lam, w_max=args.w_max, turnover=args.turnover,
        industry_neutral=args.industry_neutral, lookback=args.lookback,
        run_id=args.run_id, command=["research", "run"],
        intraday=bool(getattr(args, "intraday_leaves", False)),
        intraday_freq=getattr(args, "intraday_freq", "5min") or "5min",
    )
    print(f"[research] 完成 run_id={res['run_id']} 因子={res['expression']!r}")
    print(f"[research] 调仓 {res['n_rebalances']} 次 · sharpe={res['sharpe']} · ann_ret={res['ann_ret']}")
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
        warmup_start = (_dt.datetime.strptime(args.start, "%Y%m%d").date()
                        - _dt.timedelta(days=AGENT_WARMUP_LOOKBACK)).strftime("%Y%m%d")
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
        warmup_start = (_dt.datetime.strptime(args.start, "%Y%m%d").date()
                        - _dt.timedelta(days=int(AGENT_WARMUP_LOOKBACK * 1.55))).strftime("%Y%m%d")
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
        warmup_start = (_dt.datetime.strptime(args.start, "%Y%m%d").date()
                        - _dt.timedelta(days=int(AGENT_WARMUP_LOOKBACK * 1.5))).strftime("%Y%m%d")
        daily = build_us_daily(profile.provider, symbols, warmup_start, args.end)
        return (None if daily.is_empty() else daily), profile, prep_meta
    # A 股：预热前缀用 agent 专用加长值（LLM 窗口无搜索空间上界，长窗因子用 180 会被误判欠预热）。
    prep_meta = {}
    daily = prepare_mining_daily(
        args.start, args.end, args.universe,
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
    res = run_agent_mine(daily, n_rounds=args.iterations, seed=args.seed,
                         top_k=args.top_k, human_review=args.human_review,
                         patience=args.patience, heal_rounds=args.heal_rounds,
                         data_window=_data_window_with_membership(args, prep_meta),
                         command=_command_line(args),
                         eval_start=args.start, profile=profile,
                         library_orthogonal=not getattr(args, "no_library_orthogonal", False),
                         objective=getattr(args, "objective", "residual"),
                         intraday_scout=bool(getattr(args, "intraday_scout", False)),
                         scout_k=int(getattr(args, "scout_k", 4) or 4),
                         scout_max_leaves=int(getattr(args, "scout_max_leaves", 12) or 12),
                         scout_freq=getattr(args, "intraday_freq", "5min") or "5min")
    print(f"[mine-agent] 候选 {res['n_candidates']} 个 / N={res['n_trials']} → {res['run_dir']}")
    return 0


def _cmd_mine_team(args: argparse.Namespace) -> int:
    if getattr(args, "market", "ashare") != "crypto" and getattr(args, "freq", "daily") != "daily":
        print("[mine] --freq 仅 crypto 支持;ashare/futures/us 只有 daily", file=sys.stderr)
        return 2
    # --intraday-scout 仅 ashare；隐含 --intraday-leaves（reference 需要 i_*）
    if getattr(args, "intraday_scout", False):
        if getattr(args, "market", "ashare") != "ashare":
            print("[mine] --intraday-scout 仅 ashare 支持", file=sys.stderr)
            return 2
        args.intraday_leaves = True
    import factorzen.pipelines.factor_mine_team as pmt

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
        _daily_holder.pop(), n_rounds=args.iterations, seed=args.seed,
        top_k=args.top_k, index_path=args.index_path,
        structured=args.structured, patience=args.patience,
        heal_rounds=args.heal_rounds,
        hypotheses_per_round=args.hypotheses_per_round,
        data_window=_data_window_with_membership(args, prep_meta),
        command=_command_line(args),
        eval_start=args.start, profile=profile,
        update_library=not getattr(args, "no_library", False),
        library_orthogonal=not getattr(args, "no_library_orthogonal", False),
        objective=getattr(args, "objective", "residual"),
        llm_workers=getattr(args, "llm_workers", 1),
        auto_lift=not bool(getattr(args, "no_auto_lift", False)),
        lift_se_mult=float(getattr(args, "lift_se_mult", 1.0)),
        lift_workers=getattr(args, "lift_workers", None),  # None→自适应(按可用内存)
        campaign_prior_enabled=not bool(getattr(args, "no_campaign_prior", False)),
        intraday_scout=bool(getattr(args, "intraday_scout", False)),
        scout_k=int(getattr(args, "scout_k", 4) or 4),
        scout_max_leaves=int(getattr(args, "scout_max_leaves", 12) or 12),
        scout_freq=getattr(args, "intraday_freq", "5min") or "5min",
    )
    print(f"[mine-team] 候选 {res['n_candidates']} 个 / N={res['n_trials']} → {res['run_dir']}")
    return 0


def _cmd_factor_library_rebuild(args: argparse.Namespace) -> int:
    from datetime import date
    from datetime import datetime as _dt

    import polars as pl

    from factorzen.core.experiment import get_git_sha
    from factorzen.discovery import factor_library as fl
    from factorzen.discovery.backtest_window import default_window
    from factorzen.validation.holdout import split_holdout

    market = args.market
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
    daily, profile, _prep_meta = _prepare_agent_mining_data(args)
    if daily is None:
        print("[factor-library] 挖掘帧为空（检查 --symbols / 数据湖覆盖 / 缓存回补）",
              file=sys.stderr)
        return 1
    leaf_map = profile.factors.leaf_features() if profile is not None else None
    sources = fl.collect_source_expressions(market)
    if not sources:
        print(f"[factor-library] 提示：未从历史产物收集到 {market} 候选（将产出空库文件）")
    evaluate, compact_materialize = fl.build_library_evaluator(
        daily, holdout_ratio=args.holdout_ratio, eval_start=start, leaf_map=leaf_map,
        profile=profile)
    # lift 复审评分窗 = single 轨 evaluator 的 holdout 尾段（同 split_holdout 口径）
    # build_library_evaluator 内部：sample = prepped[trade_date>=eval_start] 再 split；
    # prep 不改 trade_date 集合，这里对 daily 同边界切分即可复用同一 holdout 起止。
    es_date = _dt.strptime(start, "%Y%m%d").date() if start else None
    sample = daily if es_date is None else daily.filter(pl.col("trade_date") >= es_date)
    _, holdout_df, holdout_start = split_holdout(
        sample, holdout_ratio=float(args.holdout_ratio),
    )
    lift_adm_start = _lift_admission_str(holdout_start)
    lift_adm_end = _lift_admission_str(holdout_df["trade_date"].max())
    res = fl.rebuild(
        market, sources=sources, eval_window=(start, end), universe=args.universe,
        horizon=args.horizon, evaluate=evaluate,
        compact_materialize=compact_materialize,
        git_sha=get_git_sha(), now=date.today().strftime("%Y-%m-%d"),
        leaf_map=leaf_map, decorr_threshold=args.decorr_threshold,
        daily=daily, profile=profile,
        admission_start=lift_adm_start, admission_end=lift_adm_end,
    )
    # lift 轨复审失败时 rebuild 已恢复旧记录；CLI 必须 fail-loudly，禁止「表面成功」
    if res.lift_review_error is not None:
        print(
            f"[factor-library] lift 轨复审失败：{res.lift_review_error}"
            f"（旧 lift 记录已恢复，本次 rebuild 不完整）",
            file=sys.stderr,
        )
        return 1
    print(f"[factor-library] {market} rebuild：新增 {res.added} / 更新 {res.updated} / "
          f"标记 correlated {res.correlated} / 跳过 {res.skipped}（窗口 {start}–{end}）")
    print(f"[factor-library] → {FACTOR_LIBRARY_DIR}/{market}.jsonl + {market}.md")
    return 0


def _cmd_factor_library_list(args: argparse.Namespace) -> int:
    from factorzen.discovery import factor_library as fl

    lib = sorted(fl.load_library(args.market), key=fl._sort_key)
    if not lib:
        print(f"[factor-library] {args.market} 库为空")
        return 0
    print(f"[factor-library] {args.market}: {len(lib)} 个因子（holdout_ic 降序）")
    for i, r in enumerate(lib, 1):
        print(f"  {i:>3}. holdout_ic={fl._fmt(r.holdout_ic)} ic_train={fl._fmt(r.ic_train)} "
              f"[{r.status}] {r.expression}")
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


def _cmd_factor_library_render(args: argparse.Namespace) -> int:
    from factorzen.discovery import factor_library as fl

    fl.render_markdown(args.market)
    print(f"[factor-library] 已重生 {FACTOR_LIBRARY_DIR}/{args.market}.md")
    return 0


def _cmd_factor_library_tag_legacy(args: argparse.Namespace) -> int:
    """把 evidence_tier 为 None 的记录落盘标 legacy（幂等，不改 status）。"""
    from factorzen.discovery import factor_library as fl

    market = args.market
    root = getattr(args, "root", None) or fl.DEFAULT_ROOT
    out = fl.tag_legacy_records(market, root=root)
    print(
        f"[factor-library tag-legacy] {market}：标记 legacy {out['tagged']} 条"
        f"（库合计 {out['total']}，已有 tier 不动；不改 status）"
    )
    return 0


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
            f"[factor-library forward-track] 全部 failed"
            f"（recorded={recorded}），退出码 1",
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
        f"[factor-library forward-review] {market}："
        f"{len(rows)} 个 probation 裁决（apply={apply}）"
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
        print(
            "[factor-library forward-review] dry-run（加 --apply 写库并更新 markdown）"
        )
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
        n_days=args.n_days, daily_sigma=args.daily_sigma, ar1=args.ar1,
        se_mults=se_mults, min_blocks_options=min_blocks,
        n_sims=args.n_sims, seed=args.seed,
    )
    print(f"[lift-null] H0=无真实 lift；n_days={args.n_days} σ={args.daily_sigma} "
          f"ar1={args.ar1} n_sims={args.n_sims} seed={args.seed}")
    print("[lift-null] 统计层下界：真实链路含选择偏差，误准入只会更高")
    print(format_calibration_markdown(rows))
    return 0


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
    return _lift_admission_str(start), _lift_admission_str(end)


def _horizon_from_manifest(man: dict) -> int | None:
    """从 session manifest 抽 mining horizon。

    键名（对照写盘代码）：
    - team：``write_team_manifest`` 把调用方 ``params`` 原样落盘 → ``params.horizon``
      （``run_team_mine`` 当前 params 未必写 horizon，缺则返回 None）
    - mining_session：顶层字段可选 ``horizon``（``run_session`` 有入参但历史 manifest
      多数未落盘）
    - 顶层 ``horizon`` 优先于 ``params.horizon``
    """
    raw = man.get("horizon")
    if raw is None:
        params = man.get("params") or {}
        raw = params.get("horizon")
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
                residual_ic_train=(
                    src.get("residual_ic_train") if isinstance(src, dict) else None
                ),
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
    import json
    from datetime import date
    from pathlib import Path

    from factorzen.core.experiment import get_git_sha
    from factorzen.discovery import factor_library as fl
    from factorzen.discovery.guardrails import DEFAULT_LIFT_THRESHOLD
    from factorzen.discovery.lift_test import (
        DEFAULT_HORIZON,
        LiftEvalContext,
        _rank_ic_key,
        extract_gray_candidates_from_manifest,
        filter_candidates_by_coverage,
        group_gate_ok,
        make_lift_context,
        resolve_lift_workers,
        run_group_lift,
        run_lift_tests,
    )

    sessions = list(args.session or [])
    if not sessions:
        print("[factor-library lift-test] 需至少一个 --session 目录", file=sys.stderr)
        return 2

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
    n_gray = sum(len(cands) for _, _, cands in groups)
    if n_gray == 0:
        print("[factor-library lift-test] 未从 session 抽到 gray_zone 候选")
        return 0

    print(
        f"[factor-library lift-test] gray_zone 候选 {n_gray} 个（去重后），"
        f"admission 分组 {len(groups)} 组"
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

    market = args.market
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
            if e:
                all_exprs.append(str(e))
    try:
        for rec in fl.load_library(market, root=lib_root):
            if getattr(rec, "status", None) == "active" and rec.expression:
                all_exprs.append(str(rec.expression))
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

    # base_ctx：prep 一次；admission 窗 per-group replace（不改 horizon）
    try:
        base_ctx = make_lift_context(
            market, daily,
            profile=profile,
            leaf_map=leaf_map,
            horizon=resolved_horizon,
            admission_start=None,
            admission_end=None,
            library_root=lib_root,
        )
    except Exception:
        # 回退=raw 帧当 prepped:派生叶子(ret_1d 等)将全空——真实数据不应走到这
        # (2026-07-14 事故根因:候选全空面板→lift 全噪声)。仅容极简 mock 帧。
        print("[factor-library lift-test] 警告：预处理失败,回退 raw 帧(派生叶子将缺失,"
              "真实数据下结果不可信)", file=sys.stderr)
        base_ctx = LiftEvalContext(
            market=market,
            prepped=daily.sort(["ts_code", "trade_date"]) if daily.height else daily,
            leaf_map=leaf_map,
            horizon=resolved_horizon,
            admission_start=None,
            admission_end=None,
            library_root=lib_root,
            profile_name=getattr(profile, "name", None) if profile is not None else None,
        )

    lift_workers_arg = getattr(args, "lift_workers", None)  # None→自适应(按可用内存)
    workers_resolved = resolve_lift_workers(lift_workers_arg)
    print(
        f"[factor-library lift-test] lift_workers={workers_resolved}"
        + ("（自适应）" if lift_workers_arg is None else f"（显式 --lift-workers={lift_workers_arg}）"),
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

    mat_base = _materializer_from_prepped(base_ctx.prepped, leaf_map)
    mat_cache: dict[str, object] = {}

    def memo_mat(expr: str):
        if expr in mat_cache:
            return mat_cache[expr]
        out = mat_base(expr)
        mat_cache[expr] = out
        return out

    results: list[dict] = []
    all_dropped: list[dict] = []
    lift_groups_meta: list[dict] = []
    truncated_from: int | None = None
    n_lift_evaluated = 0

    for g_start, g_end, cands in groups:
        n_in = len(cands)
        ordered = sorted(cands, key=_rank_ic_key, reverse=True)
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
            lift_groups_meta.append({
                "admission_start": g_start,
                "admission_end": g_end,
                "skipped": "empty_after_coverage",
            })
            continue

        grp_ctx = dataclasses.replace(
            base_ctx, admission_start=g_start, admission_end=g_end,
        )
        group = run_group_lift(
            kept,
            market=market,
            daily=daily,
            leaf_map=leaf_map,
            library_root=lib_root,
            seed=seed,
            threshold=threshold,
            materialize_candidate=memo_mat,
            ctx=grp_ctx,
        )
        shared_base_daily = group.get("base_daily")
        group_view = {k: v for k, v in group.items() if k != "base_daily"}
        lift_groups_meta.append(group_view)
        n_lift_evaluated += 1  # 组门计 1 次

        group_ok, bar = group_gate_ok(
            group, threshold=float(threshold), lift_se_mult=se_mult,
        )
        g_lift, g_se = group.get("lift"), group.get("lift_se")
        print(
            f"[factor-library lift-test] 组门 lift={g_lift!r} se={g_se!r} "
            f"bar={bar:.4f} → {'过' if group_ok else '拒'}",
            flush=True,
        )
        if not group_ok:
            # 组门不过：全体 skip 逐候选；结果行记 reject 原因（lift/se 取组门值供 index 写回）
            reason = (
                f"group_gate_fail(lift={g_lift!r},se={g_se!r},bar={bar:.4f})"
            )
            for c in kept:
                results.append({
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
                })
            continue

        rows = run_lift_tests(
            kept,
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
            base_daily=shared_base_daily,
        )
        n_lift_evaluated += len(rows)
        for r in rows:
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
        expr = (r.get("expression") or "")[:40]
        lift = r.get("lift")
        se = r.get("lift_se")
        sh = r.get("lift_second_half")
        base = r.get("baseline")
        ls = f"{lift:+.4f}" if lift is not None else "  n/a "
        ses = f"{se:.4f}" if se is not None else "  n/a "
        shs = f"{sh:+.4f}" if sh is not None else "  n/a "
        bs = f"{base:.4f}" if base is not None else "  n/a "
        print(
            f"{expr:40s}  {ls:>8s}  {ses:>8s}  {shs:>8s}  {bs:>8s}  {r.get('passed')}"
        )

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
        print(
            f"[factor-library lift-test] dry-run：通过 {n_pass} 个，不写库（加 --apply 写库）"
        )
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
                print(
                    f"[factor-library lift-test] experiment_index 写回 lift_rejected {n_idx} 条"
                )
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
    }
    if truncated_from is not None:
        lift_manifest["truncated_from"] = truncated_from
    out_man = Path(sessions[0]) / "lift_test_manifest.json"
    out_man.write_text(
        json.dumps(lift_manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[factor-library lift-test] → {out_man}")
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
            print(f"[mine] {csv}: 无候选通过防过拟合护栏；用 --all 查看全部 {df.height} 个候选",
                  file=sys.stderr)
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
    cross = export_crypto_alpha(profile, expr, symbols, start, args.date, date=args.date,
                                freq=args.freq)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cross.write_parquet(args.out)
    print(f"[mine] export-alpha(crypto): rank={args.rank} expr={expr!r} date={args.date} "
          f"→ {args.out} ({cross.height} 个标的)")
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
    start = (datetime.strptime(args.date, "%Y%m%d") - timedelta(days=args.lookback)).strftime("%Y%m%d")
    cross = export_futures_alpha(profile, expr, symbols, start, args.date, date=args.date)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cross.write_parquet(args.out)
    print(f"[mine] export-alpha(futures): rank={args.rank} expr={expr!r} date={args.date} "
          f"→ {args.out} ({cross.height} 个品种)")
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
    start = (datetime.strptime(args.date, "%Y%m%d") - timedelta(days=args.lookback)).strftime("%Y%m%d")
    cross = export_us_alpha(profile, expr, symbols, start, args.date, date=args.date)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cross.write_parquet(args.out)
    print(f"[mine] export-alpha(us): rank={args.rank} expr={expr!r} date={args.date} "
          f"→ {args.out} ({cross.height} 个标的)")
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
    print(f"[mine] export-alpha: rank={args.rank} expr={expr!r} date={args.date} "
          f"→ {out} ({n} 只股票)")
    return 0


def _validate_overfit_crypto(args: argparse.Namespace) -> int:
    """crypto 单表达式防过拟合验证（live CCXT）。"""
    if not getattr(args, "expression", None):
        print("[validate] crypto 需 --expression \"<表达式>\"", file=sys.stderr)
        return 1
    from factorzen.markets.crypto.mining import validate_crypto_expression
    from factorzen.markets.crypto.profile import build_crypto_profile

    profile = build_crypto_profile(top_n=args.top_n)
    symbols = profile.universe.snapshot(args.end)
    rep = validate_crypto_expression(profile, args.expression, symbols, args.start, args.end,
                                     freq=args.freq)
    print(
        f"[validate] {args.expression}: IC={rep['ic_mean']:.4f} IR={rep['ir']:.4f} "
        f"DSR_p={rep['dsr_p']:.4f} IC_95%CI=[{rep['ci_lo']:.4f},{rep['ci_hi']:.4f}]"
    )
    print("[validate] 注：单因子 N=1（无多重检验扣减）；PBO 仅适用候选池，此处略。")
    return 0


def _validate_overfit_futures(args: argparse.Namespace) -> int:
    """futures 单表达式防过拟合验证（Tushare 主力连续）。"""
    if not getattr(args, "expression", None):
        print("[validate] futures 需 --expression \"<表达式>\"", file=sys.stderr)
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
        print("[validate] us 需 --expression \"<表达式>\"", file=sys.stderr)
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
    from factorzen.discovery.scoring import ic_overfit_report

    # factor 位置参数 nargs='?' 可缺省；缺省时给友好用法提示，而非 get_factor(None) 裸 KeyError
    if not getattr(args, "factor", None):
        print("[validate] 缺少因子名：用法 fz validate overfit <factor> --start ... --end ...",
              file=sys.stderr)
        return 2
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
    signal_date = f"{_end[:4]}-{_end[4:6]}-{_end[6:]}" if len(_end) == 8 and _end.isdigit() else _end
    res = build_crypto_portfolio(
        profile, adf, symbols, args.start, args.end,
        market_neutral=True, w_max=args.w_max, gross_limit=args.gross_limit,
        risk_aversion=args.lam, signal_date=signal_date, freq=args.freq,
    )
    print(f"[portfolio] crypto status={res['status']} holdings={res['n_holdings']} → {res['run_dir']}")
    return 0


def _cmd_portfolio_build(args: argparse.Namespace) -> int:
    if getattr(args, "market", "ashare") != "crypto" and getattr(args, "freq", "daily") != "daily":
        print("[portfolio] --freq 仅 crypto 支持;ashare 只有 daily", file=sys.stderr)
        return 2
    if getattr(args, "market", "ashare") == "crypto":
        return _portfolio_build_crypto(args)
    import numpy as np
    import polars as pl

    from factorzen.core import loader
    from factorzen.core.universe import get_universe
    from factorzen.pipelines.portfolio_build import run_portfolio
    from factorzen.pipelines.risk_build import load_risk_inputs
    from factorzen.risk.model import RiskModel

    stocks = get_universe(args.end, args.universe)
    uni = stocks["ts_code"].to_list()
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
    bench_weights = (
        np.full(len(codes), 1.0 / len(codes)) if args.industry_neutral else None
    )
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
        p for p in portfolio_root.iterdir()
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
            [str(p) for p in run_dirs], profile, args.start, args.end,
            out_dir=str(SIM_DIR), run_id=args.run_id, freq=args.freq,
        )
    else:
        from factorzen.core import loader
        from factorzen.sim.engine import run_portfolio_simulation

        daily = loader.fetch_daily(args.start, args.end)
        res = run_portfolio_simulation(
            [str(p) for p in run_dirs], daily, out_dir=str(SIM_DIR), run_id=args.run_id,
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
    print(f"replay 完成: {out['n_steps']} 步, 终值 NAV={out['final_nav']:.2f} → {out['session_dir']}")
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
        session_dirs=args.session, start=args.start, end=args.end, universe=args.universe,
        horizon=args.horizon, passed_only=not args.all, top_n=args.top_n,
        decorr_threshold=args.decorr_threshold,
        methods=methods, seed=args.seed, out_dir=args.out_dir, run_id=args.run_id,
        train_days=args.train_days, test_days=args.test_days,
        purge_days=args.purge_days, embargo_days=args.embargo_days,
    )
    print(f"[combine] 因子库组合完成 → {res['run_dir']}")
    print(f"[combine] 纳入 {len(res['factors_used'])} 个因子；"
          f"去相关剔除 {len(res['dropped_correlated'])} 个近亲")
    for d in res["dropped_correlated"]:
        print(f"[combine]   ✗ {d['expression']} → 与 {d['corr_with']} 相关 {d['corr']:.2f}")
    print(res["comparison"])
    return 0


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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    effective = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(effective)
    # 落 manifest 用（铁律#3）。记「实际传入的 argv」而非 sys.argv——main() 被程序化调用时
    # （如 research run 编排器）sys.argv 是外层进程的命令行，会记错。
    args.command_line = "fz " + " ".join(effective)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
