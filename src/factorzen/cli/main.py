"""Unified FactorZen command line interface."""

from __future__ import annotations

import argparse
import json
import sys

from factorzen.config.settings import FACTOR_EVALUATIONS_DIR, ROOT
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
    if args.ic_method:
        forwarded.extend(["--ic-method", args.ic_method])
    if args.neutralized_ic:
        forwarded.append("--neutralized-ic")
    if args.event_study:
        forwarded.append("--event-study")
    if args.llm_explain:
        forwarded.append("--llm-explain")
    if args.llm_refresh:
        forwarded.append("--llm-refresh")
    if args.all:
        forwarded.append("--all")
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
        from factorzen.core.config_loader import load_run_config

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
    if args.ic_method:
        forwarded.extend(["--ic-method", args.ic_method])
    if args.neutralized_ic:
        forwarded.append("--neutralized-ic")
    if args.event_study:
        forwarded.append("--event-study")
    if args.llm_explain:
        forwarded.append("--llm-explain")
    if args.llm_refresh:
        forwarded.append("--llm-refresh")
    if args.all:
        forwarded.append("--all")

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


def _cmd_config_validate(args: argparse.Namespace) -> int:
    from factorzen.core.config_loader import default_benchmark_for_universe, load_run_config

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
    )
    sd = res["session_dir"]
    print(f"[mine] us 完成：{len(res['candidates'])} 个候选 / {len(symbols)} 标的 → {sd}")
    return 0


def _cmd_mine_search(args: argparse.Namespace) -> int:
    if getattr(args, "market", "ashare") not in ("crypto",) and getattr(args, "freq", "daily") != "daily":
        print("[mine] --freq 仅 crypto 支持;ashare/futures/us 只有 daily", file=sys.stderr)
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
    )
    sd = res["session_dir"]
    print(f"[mine] 完成：{len(res['candidates'])} 个候选 → {sd}")
    print(f"[mine] 复现：cp {sd}/exported/*.py workspace/factors/daily/ && "
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


def _prepare_agent_mining_data(args: argparse.Namespace):
    """按 market 装配含预热前缀的挖掘帧，返回 ``(daily, profile)``（profile=None → A 股）。

    - ashare：`prepare_mining_daily`（复权价 + daily_basic + 全叶子），profile=None（零回归）。
    - crypto：`build_crypto_daily`（Vision 湖），向前多拉 `AGENT_WARMUP_LOOKBACK` 自然日作预热前缀
      （crypto 24/7，1 bar≈1 自然日，与 A 股口径一致）；symbols 取 --symbols 或 universe Top-N。

    daily 为空（crypto 湖无对应 symbol 数据）→ 返回 ``(None, profile)``，调用方报错退出。
    """
    from factorzen.pipelines.factor_mine import AGENT_WARMUP_LOOKBACK, prepare_mining_daily

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
        if not symbols:
            return None, profile
        warmup_start = (_dt.datetime.strptime(args.start, "%Y%m%d").date()
                        - _dt.timedelta(days=AGENT_WARMUP_LOOKBACK)).strftime("%Y%m%d")
        freq = getattr(args, "freq", None) or profile.base_freq
        daily = build_crypto_daily(profile.provider, symbols, warmup_start, args.end, freq)
        return (None if daily.is_empty() else daily), profile
    if market == "futures":
        import datetime as _dt

        from factorzen.markets.futures.mining import build_futures_daily
        from factorzen.markets.futures.profile import build_futures_profile

        profile = build_futures_profile(top_n=getattr(args, "top_n", 40))
        if getattr(args, "symbols", None):
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        else:
            symbols = profile.universe.snapshot(args.end)
        if not symbols:
            return None, profile
        # 预热前缀：AGENT_WARMUP_LOOKBACK 交易日 → 自然日近似（243 交易日/年，×1.55 覆盖节假日）。
        warmup_start = (_dt.datetime.strptime(args.start, "%Y%m%d").date()
                        - _dt.timedelta(days=int(AGENT_WARMUP_LOOKBACK * 1.55))).strftime("%Y%m%d")
        daily = build_futures_daily(profile.provider, symbols, warmup_start, args.end)
        return (None if daily.is_empty() else daily), profile
    if market == "us":
        import datetime as _dt

        from factorzen.markets.us.mining import build_us_daily
        from factorzen.markets.us.profile import build_us_profile

        profile = build_us_profile(top_n=getattr(args, "top_n", 50))
        if getattr(args, "symbols", None):
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        else:
            symbols = profile.universe.snapshot(args.end)
        if not symbols:
            return None, profile
        # 预热前缀：AGENT_WARMUP_LOOKBACK 交易日 → 自然日近似（252 交易日/年，×1.5 覆盖周末/假日）。
        warmup_start = (_dt.datetime.strptime(args.start, "%Y%m%d").date()
                        - _dt.timedelta(days=int(AGENT_WARMUP_LOOKBACK * 1.5))).strftime("%Y%m%d")
        daily = build_us_daily(profile.provider, symbols, warmup_start, args.end)
        return (None if daily.is_empty() else daily), profile
    # A 股：预热前缀用 agent 专用加长值（LLM 窗口无搜索空间上界，长窗因子用 180 会被误判欠预热）。
    daily = prepare_mining_daily(args.start, args.end, args.universe,
                                 lookback_days=AGENT_WARMUP_LOOKBACK)
    return daily, None


def _cmd_mine_agent(args: argparse.Namespace) -> int:
    if getattr(args, "market", "ashare") != "crypto" and getattr(args, "freq", "daily") != "daily":
        print("[mine] --freq 仅 crypto 支持;ashare/futures/us 只有 daily", file=sys.stderr)
        return 2
    from factorzen.pipelines.factor_mine_agent import run_agent_mine

    daily, profile = _prepare_agent_mining_data(args)
    if daily is None:
        print("[mine-agent] crypto 挖掘帧为空（检查 --symbols 或数据湖覆盖）", file=sys.stderr)
        return 1
    # eval_start = 挖掘窗口 start（预热前缀边界），与 M1 `run_mine(eval_start=start)` 同口径：
    # 缺了它预热前缀会被 split_holdout 当训练数据。
    res = run_agent_mine(daily, n_rounds=args.iterations, seed=args.seed,
                         top_k=args.top_k, human_review=args.human_review,
                         patience=args.patience, heal_rounds=args.heal_rounds,
                         data_window=_data_window(args), command=_command_line(args),
                         eval_start=args.start, profile=profile,
                         library_orthogonal=not getattr(args, "no_library_orthogonal", False))
    print(f"[mine-agent] 候选 {res['n_candidates']} 个 / N={res['n_trials']} → {res['run_dir']}")
    return 0


def _cmd_mine_team(args: argparse.Namespace) -> int:
    if getattr(args, "market", "ashare") != "crypto" and getattr(args, "freq", "daily") != "daily":
        print("[mine] --freq 仅 crypto 支持;ashare/futures/us 只有 daily", file=sys.stderr)
        return 2
    from factorzen.pipelines.factor_mine_team import run_team_mine

    # 数据装配与 agent 路径共用 `_prepare_agent_mining_data`（ashare=A 股 loader，
    # crypto=Vision 湖 + 预热前缀）。消除双路径漂移。
    daily, profile = _prepare_agent_mining_data(args)
    if daily is None:
        print("[mine-team] crypto 挖掘帧为空（检查 --symbols 或数据湖覆盖）", file=sys.stderr)
        return 1
    # eval_start = 挖掘窗口 start（预热前缀边界），同 M1/agent 口径，见 _cmd_mine_agent。
    res = run_team_mine(daily, n_rounds=args.iterations, seed=args.seed,
                        top_k=args.top_k, index_path=args.index_path,
                        structured=args.structured, patience=args.patience,
                        heal_rounds=args.heal_rounds,
                        hypotheses_per_round=args.hypotheses_per_round,
                        data_window=_data_window(args), command=_command_line(args),
                        eval_start=args.start, profile=profile,
                        update_library=not getattr(args, "no_library", False),
                        library_orthogonal=not getattr(args, "no_library_orthogonal", False))
    print(f"[mine-team] 候选 {res['n_candidates']} 个 / N={res['n_trials']} → {res['run_dir']}")
    return 0


def _cmd_factor_library_rebuild(args: argparse.Namespace) -> int:
    from datetime import date

    from factorzen.core.experiment import get_git_sha
    from factorzen.discovery import factor_library as fl
    from factorzen.discovery.backtest_window import default_window

    market = args.market
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
    daily, profile = _prepare_agent_mining_data(args)
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
    res = fl.rebuild(market, sources=sources, eval_window=(start, end), universe=args.universe,
                     horizon=args.horizon, evaluate=evaluate,
                     compact_materialize=compact_materialize,
                     git_sha=get_git_sha(), now=date.today().strftime("%Y-%m-%d"),
                     leaf_map=leaf_map, decorr_threshold=args.decorr_threshold)
    print(f"[factor-library] {market} rebuild：新增 {res.added} / 更新 {res.updated} / "
          f"标记 correlated {res.correlated} / 跳过 {res.skipped}（窗口 {start}–{end}）")
    print(f"[factor-library] → workspace/factor_library/{market}.jsonl + {market}.md")
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
    print(f"[factor-library] 已重生 workspace/factor_library/{args.market}.md")
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
        out_dir=getattr(args, "out_dir", "workspace/portfolios"),
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
    print(f"[risk] factors={len(res['factor_names'])} R2={res['r_squared']:.4f} → {res['run_dir']}")
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
            out_dir="workspace/sim", run_id=args.run_id, freq=args.freq,
        )
    else:
        from factorzen.core import loader
        from factorzen.sim.engine import run_portfolio_simulation

        daily = loader.fetch_daily(args.start, args.end)
        res = run_portfolio_simulation(
            [str(p) for p in run_dirs], daily, out_dir="workspace/sim", run_id=args.run_id,
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
        out_path = Path("workspace/reports") / f"portfolio_{run_id}.html"
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


def _add_factor_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("name", nargs="?", help="Factor name")
    parser.add_argument("--start", default=None, help="Start date YYYYMMDD")
    parser.add_argument("--end", default=None, help="End date YYYYMMDD")
    parser.add_argument("--universe", default=None, help="Universe name")
    parser.add_argument(
        "--frequency",
        "--freq",
        dest="frequency",
        choices=["daily", "weekly", "monthly"],
        default="daily",
        help="Factor frequency",
    )
    parser.add_argument("--benchmark", default=None, help="Benchmark index code")
    parser.add_argument("--config", default=None, help="YAML run config path")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed")
    parser.add_argument(
        "--set",
        action="append",
        default=None,
        dest="set_overrides",
        metavar="KEY=VALUE",
        help="Override any config field, repeatable: --set backtest.top_n=30",
    )
    parser.add_argument("--all", action="store_true", help="Enable deep evaluation preset")
    parser.add_argument("--dry-run", action="store_true", help="Print effective config without running")
    parser.add_argument(
        "--ic-method",
        default=None,
        choices=["rank", "pearson", "both"],
        dest="ic_method",
        help="IC method",
    )
    parser.add_argument("--neutralized-ic", action="store_true", dest="neutralized_ic")
    parser.add_argument("--event-study", action="store_true", dest="event_study")
    parser.add_argument(
        "--llm-explain",
        action="store_true",
        help="Enable LLM explanation; no-config daily runs enable this by default",
    )
    parser.add_argument("--llm-refresh", action="store_true")


def _add_report_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("name", nargs="?", help="Factor name")
    parser.add_argument("--factor", default=None, help="Factor name")
    parser.add_argument("--start", default=None, help="Start date YYYYMMDD")
    parser.add_argument("--end", default=None, help="End date YYYYMMDD")
    parser.add_argument("--universe", default=None, help="Universe name")
    parser.add_argument(
        "--frequency",
        "--freq",
        dest="frequency",
        choices=["daily", "weekly", "monthly"],
        default="daily",
        help="Factor frequency",
    )
    parser.add_argument("--reuse", action="store_true", help="Reuse existing artifacts")
    parser.add_argument("--all", action="store_true", help="Enable deep report preset")
    parser.add_argument("--benchmark", default=None, help="Benchmark index code")
    parser.add_argument("--config", default=None, help="YAML run config path")
    parser.add_argument(
        "--ic-method",
        default=None,
        choices=["rank", "pearson", "both"],
        dest="ic_method",
        help="IC method",
    )
    parser.add_argument("--neutralized-ic", action="store_true", dest="neutralized_ic")
    parser.add_argument("--event-study", action="store_true", dest="event_study")
    parser.add_argument("--llm-explain", action="store_true")
    parser.add_argument("--llm-refresh", action="store_true")


def _add_freq_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--freq", choices=["1m", "5m", "15m", "1h", "daily"], default="daily",
                   help="bar 粒度(仅 crypto;ashare 只支持 daily)")


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
    from factorzen.discovery.guardrails import DEFAULT_DSR_ALPHA  # 护栏阈值单一真源，防漂移

    parser = argparse.ArgumentParser(prog="fz", description="FactorZen research CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    factor = sub.add_parser("factor", help="Factor workflows")
    factor_sub = factor.add_subparsers(dest="factor_command", required=True)

    new = factor_sub.add_parser("new", help="Create a user factor template")
    new.add_argument("name")
    new.add_argument(
        "--frequency",
        "--freq",
        dest="freq",
        choices=["daily", "weekly", "monthly", "intraday"],
        default="daily",
    )
    new.add_argument("--force", action="store_true")
    new.set_defaults(func=_cmd_factor_new)

    list_cmd = factor_sub.add_parser("list", help="List registered factors")
    list_cmd.add_argument(
        "--frequency",
        "--freq",
        dest="freq",
        choices=["daily", "weekly", "monthly", "intraday"],
        default="daily",
    )
    list_cmd.set_defaults(func=_cmd_factor_list)

    run = factor_sub.add_parser("run", help="Run a single factor evaluation")
    _add_factor_run_arguments(run)
    run.set_defaults(func=_cmd_factor_test)

    test = factor_sub.add_parser("test", help="Deprecated alias for 'factor run'")
    _add_factor_run_arguments(test)
    test.set_defaults(func=_cmd_factor_test)

    sweep = factor_sub.add_parser("sweep", help="Parameter grid sweep over --set overrides")
    sweep.add_argument("name", nargs="?", help="Factor name (or supply via --config)")
    sweep.add_argument("--config", default=None, help="Base YAML run config path")
    sweep.add_argument(
        "--grid",
        action="append",
        default=None,
        metavar="KEY=V1,V2,...",
        help="Grid dimension, repeatable: --grid backtest.top_n=30,50,100",
    )
    sweep.add_argument(
        "--set",
        action="append",
        default=None,
        dest="set_overrides",
        metavar="KEY=VALUE",
        help="Fixed override applied to every combo",
    )
    sweep.add_argument("--start", default=None, help="Start date YYYYMMDD")
    sweep.add_argument("--end", default=None, help="End date YYYYMMDD")
    sweep.add_argument("--universe", default=None, help="Universe name")
    sweep.add_argument(
        "--sort-by",
        default="ir",
        dest="sort_by",
        help="Metric to rank rows by (ir/ic_mean/ic_pos/t)",
    )
    sweep.set_defaults(func=_cmd_factor_sweep)

    report = sub.add_parser("report", help="Report workflows")
    report_sub = report.add_subparsers(dest="report_command", required=True)

    build_cmd = report_sub.add_parser("build", help="Build a factor report")
    _add_report_build_arguments(build_cmd)
    build_cmd.set_defaults(func=_cmd_report_build)

    path_cmd = report_sub.add_parser("path", help="Print report path for a run")
    path_cmd.add_argument("run_id")
    path_cmd.set_defaults(func=_cmd_report_open)

    open_cmd = report_sub.add_parser("open", help="Deprecated alias for 'report path'")
    open_cmd.add_argument("run_id")
    open_cmd.set_defaults(func=_cmd_report_open)

    pf_report = report_sub.add_parser("portfolio", help="Generate portfolio dashboard HTML report")
    pf_report.add_argument(
        "--sim-dir",
        default=None,
        dest="sim_dir",
        help="模拟产物目录（含 metrics.json）",
    )
    pf_report.add_argument(
        "--portfolio-dir",
        default=None,
        dest="portfolio_dir",
        help="组合构建产物目录（含 attribution.csv / risk_summary.csv / manifest.json）",
    )
    pf_report.add_argument(
        "--out",
        default=None,
        dest="out",
        help="HTML 输出路径；默认 workspace/reports/portfolio_<run_id>.html",
    )
    pf_report.add_argument(
        "--market",
        choices=["ashare", "crypto"],
        default=None,
        help="市场语境(默认从 sim manifest 自动识别；crypto=USDT/365/资金费/sector)",
    )
    pf_report.set_defaults(func=_cmd_report_portfolio)

    data = sub.add_parser("data", help="Data workflows")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    fetch = data_sub.add_parser("fetch", help="Fetch raw data into cache")
    fetch.add_argument("data_type", choices=["daily", "daily-basic", "fundamentals", "flows"])
    fetch.add_argument("--start", required=True, help="Start date YYYYMMDD")
    fetch.add_argument("--end", required=True, help="End date YYYYMMDD")
    fetch.set_defaults(func=_cmd_data_fetch)

    crypto_p = data_sub.add_parser("crypto", help="Crypto data lake workflows")
    crypto_sub = crypto_p.add_subparsers(dest="crypto_command", required=True)
    bf = crypto_sub.add_parser("backfill", help="Backfill 1m klines/funding/OI from Binance Vision")
    bf.add_argument("--start", required=True)
    bf.add_argument("--end", required=True)
    bf.add_argument("--symbols", default=None, help="逗号分隔;缺省=按上月成交额 Top-N 自动选池")
    bf.add_argument("--top-n", dest="top_n", type=int, default=50)
    bf.add_argument("--lake-root", dest="lake_root", default="workspace/crypto_lake")
    bf.set_defaults(func=_cmd_data_crypto_backfill)

    config = sub.add_parser("config", help="Config workflows")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    validate = config_sub.add_parser("validate", help="Validate a YAML run config")
    validate.add_argument("path", help="YAML run config path")
    validate.set_defaults(func=_cmd_config_validate)

    runs = sub.add_parser("runs", help="Run history workflows")
    runs_sub = runs.add_subparsers(dest="runs_command", required=True)
    list_cmd = runs_sub.add_parser("list", help="List recorded runs")
    list_cmd.add_argument("--limit", type=int, default=20, help="Maximum rows to print")
    list_cmd.set_defaults(func=_cmd_runs_list)
    show_cmd = runs_sub.add_parser("show", help="Show one run manifest")
    show_cmd.add_argument("run_id")
    show_cmd.set_defaults(func=_cmd_runs_show)

    # ── fz mine ──（与 fz factor 并列的顶层命令组）
    mine = sub.add_parser("mine", help="Factor mining workflows")
    mine_sub = mine.add_subparsers(dest="mine_command", required=True)

    m_search = mine_sub.add_parser("search", help="Search candidate factor expressions")
    m_search.add_argument("--start", required=True, help="Start date YYYYMMDD")
    m_search.add_argument("--end", required=True, help="End date YYYYMMDD")
    m_search.add_argument("--universe", default=None, help="Universe name (e.g. csi500)")
    m_search.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare",
                          help="Market profile (default ashare; crypto=USDT-M perps; "
                               "futures=国内商品期货主力连续; us=S&P500 Yahoo 后复权)")
    m_search.add_argument("--top-n", dest="top_n", type=int, default=50,
                          help="crypto/futures universe size (Top-N by turnover); us=S&P500 静态池截断 (default 50)")
    m_search.add_argument("--method", choices=["random", "genetic"], default="random")
    m_search.add_argument("--trials", type=int, default=200)
    m_search.add_argument("--top-k", dest="top_k", type=int, default=10)
    m_search.add_argument("--seed", type=int, default=42)
    m_search.add_argument("--workers", type=int, default=1,
                          help="遗传搜索并行评分线程数(默认 1;同 seed 结果与串行等价)")
    m_search.add_argument("--holdout-ratio", dest="holdout_ratio", type=float, default=0.2,
                          help="永久隔离的 OOS holdout 占比（默认 0.2）")
    m_search.add_argument("--train-ratio", dest="train_ratio", type=float, default=0.7,
                          help="mining 段内 train/valid 切分比例（默认 0.7）")
    m_search.add_argument("--decorr-threshold", dest="decorr_threshold", type=float, default=0.7,
                          help="top-K 贪心去相关的 |corr| 门槛，≥该值视为近重复剔除（默认 0.7）")
    m_search.add_argument("--min-n-train", dest="min_n_train", type=int, default=5,
                          help="候选 train 段最少有效 IC 天数，不足则丢弃（默认 5）")
    m_search.add_argument("--dsr-alpha", dest="dsr_alpha", type=float, default=DEFAULT_DSR_ALPHA,
                          help="护栏 passed 标记的 DSR 显著性阈值（默认 0.10，2026-07 松一档）")
    m_search.add_argument("--no-library", dest="no_library", action="store_true",
                          help="关闭收尾自动 upsert 因子库（默认开，passed 候选进 workspace/factor_library）")
    m_search.add_argument("--no-library-orthogonal", dest="no_library_orthogonal",
                          action="store_true",
                          help="关闭搜索期库级正交过滤（默认开：top-K 贪心去相关时避开库内 active 方向；"
                               "与 --no-library 无关，后者只关收尾 upsert）")
    _add_freq_arg(m_search)
    m_search.set_defaults(func=_cmd_mine_search)

    m_lb = mine_sub.add_parser("leaderboard", help="Print a mining session leaderboard")
    m_lb.add_argument("session_dir", help="Path to a mining session directory")
    m_lb.add_argument("--all", action="store_true",
                      help="Show all candidates, including those failing the overfitting guardrails")
    m_lb.set_defaults(func=_cmd_mine_leaderboard)

    m_exp = mine_sub.add_parser(
        "export-alpha",
        help="Compute one candidate's cross-sectional alpha → (ts_code,alpha) parquet",
    )
    m_exp.add_argument("--session", required=True,
                       help="Mining session dir (contains candidates.csv)")
    m_exp.add_argument("--rank", type=int, default=1,
                       help="Candidate rank in candidates.csv (1-based, default 1)")
    m_exp.add_argument("--date", required=True, help="Cross-section date YYYYMMDD")
    m_exp.add_argument("--universe", default="all_a", help="Universe name (default all_a)")
    m_exp.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare",
                       help="Market profile (default ashare; crypto=USDT-M perps; futures=商品期货; us=S&P500)")
    m_exp.add_argument("--top-n", dest="top_n", type=int, default=50,
                       help="crypto/futures universe size (Top-N by turnover); us=S&P500 静态池截断 (default 50)")
    m_exp.add_argument("--lookback", type=int, default=60,
                       help="Trade-day lookback for time-series operators (default 60)")
    m_exp.add_argument("--out", required=True,
                       help="Output parquet path (columns: ts_code, alpha)")
    m_exp.add_argument("--all", action="store_true",
                       help="Allow exporting a candidate that failed the overfitting guardrails "
                            "(default: only passed candidates)")
    _add_freq_arg(m_exp)
    m_exp.set_defaults(func=_cmd_mine_export_alpha)

    m_agent = mine_sub.add_parser("agent", help="LLM-guided agent factor mining")
    m_agent.add_argument("--start", required=True)
    m_agent.add_argument("--end", required=True)
    m_agent.add_argument("--universe", default=None)
    m_agent.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare",
                         help="Market profile (default ashare; crypto=USDT-M perps via Vision lake; "
                              "futures=国内商品期货主力连续 via Tushare; us=S&P500 via Yahoo chart)")
    m_agent.add_argument("--symbols", default=None,
                         help="crypto/futures/us only: 逗号分隔 symbols；缺省=universe Top-N 快照")
    m_agent.add_argument("--top-n", dest="top_n", type=int, default=50,
                         help="crypto/futures universe size (Top-N by turnover); us=S&P500 静态池截断 (default 50)")
    m_agent.add_argument("--iterations", type=int, default=5)
    m_agent.add_argument("--top-k", dest="top_k", type=int, default=5)
    m_agent.add_argument("--seed", type=int, default=42)
    m_agent.add_argument("--human-review", action="store_true", dest="human_review")
    m_agent.add_argument("--patience", type=_positive_patience, default=None,
                         help="连续 N 轮无新候选则早停（N>=1；默认不早停，跑满 --iterations）")
    m_agent.add_argument("--heal-rounds", dest="heal_rounds", type=int, default=2,
                         help="表达式解析失败时回灌 LLM 修正的最大轮数（0=关闭）")
    m_agent.add_argument("--no-library-orthogonal", dest="no_library_orthogonal",
                         action="store_true",
                         help="关闭搜索期库级正交过滤（默认开：护栏阶段避开库内 active 方向）")
    _add_freq_arg(m_agent)
    m_agent.set_defaults(func=_cmd_mine_agent)

    m_team = mine_sub.add_parser("team", help="Multi-agent team factor mining")
    m_team.add_argument("--start", required=True)
    m_team.add_argument("--end", required=True)
    m_team.add_argument("--universe", default=None)
    m_team.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare",
                        help="Market profile (default ashare; crypto=USDT-M perps via Vision lake; "
                             "futures=国内商品期货主力连续 via Tushare; us=S&P500 via Yahoo chart)")
    m_team.add_argument("--symbols", default=None,
                        help="crypto/futures/us only: 逗号分隔 symbols；缺省=universe Top-N 快照")
    m_team.add_argument("--top-n", dest="top_n", type=int, default=50,
                        help="crypto/futures universe size (Top-N by turnover); us=S&P500 静态池截断 (default 50)")
    m_team.add_argument("--iterations", type=int, default=5)
    m_team.add_argument("--top-k", dest="top_k", type=int, default=5)
    m_team.add_argument("--seed", type=int, default=42)
    m_team.add_argument("--index-path", dest="index_path",
                        default="workspace/mine_team/experiment_index.jsonl")
    m_team.add_argument("--structured", action="store_true",
                        help="结构化假设(机制/预期符号/证伪判据) + 任务分解后逐任务翻译")
    m_team.add_argument("--patience", type=_positive_patience, default=None,
                        help="连续 N 轮无新候选则早停（N>=1；默认不早停，跑满 --iterations）")
    m_team.add_argument("--heal-rounds", dest="heal_rounds", type=int, default=2,
                        help="表达式解析失败时回灌 LLM 修正的最大轮数（0=关闭）")
    m_team.add_argument("--hypotheses-per-round", dest="hypotheses_per_round",
                        type=int, default=1,
                        help="每轮提多少个假设（默认1；>1 提升单轮产能，护栏/Critic 仍每轮一次）")
    m_team.add_argument("--no-library", dest="no_library", action="store_true",
                        help="关闭收尾自动 upsert 因子库（默认开，最终候选进 workspace/factor_library）")
    m_team.add_argument("--no-library-orthogonal", dest="no_library_orthogonal",
                        action="store_true",
                        help="关闭搜索期库级正交过滤（默认开：护栏阶段避开库内 active 方向；"
                             "与 --no-library 无关，后者只关收尾 upsert）")
    _add_freq_arg(m_team)
    m_team.set_defaults(func=_cmd_mine_team)

    # ── fz factor-library ──（分市场因子登记簿：rebuild / list / show / render）
    fl = sub.add_parser("factor-library",
                        help="因子库登记簿（分市场·全信息·自动维护）：rebuild/list/show/render")
    fl_sub = fl.add_subparsers(dest="factor_library_command", required=True)

    fl_rb = fl_sub.add_parser("rebuild",
                              help="从历史产物在统一默认窗口重算并重建某市场的因子库")
    fl_rb.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare")
    fl_rb.add_argument("--start", default=None, help="覆盖默认窗口起点 YYYYMMDD（缺省=最近6年滚动）")
    fl_rb.add_argument("--end", default=None, help="覆盖默认窗口终点 YYYYMMDD（缺省=数据最新端）")
    fl_rb.add_argument("--universe", default=None, help="A股 universe 名（如 csi300）")
    fl_rb.add_argument("--horizon", type=int, default=1, help="前向收益持有期（默认1）")
    fl_rb.add_argument("--top-n", dest="top_n", type=int, default=50,
                       help="crypto/futures universe size (Top-N by turnover); us=S&P500 静态池截断 (default 50)")
    fl_rb.add_argument("--symbols", default=None,
                       help="crypto/futures/us only: 逗号分隔 symbols；缺省=universe Top-N 快照")
    fl_rb.add_argument("--decorr-threshold", dest="decorr_threshold", type=float, default=0.7,
                       help="去相关 |corr| 门槛，超此仍收录但标 correlated（默认0.7）")
    fl_rb.add_argument("--holdout-ratio", dest="holdout_ratio", type=float, default=0.2)
    _add_freq_arg(fl_rb)
    fl_rb.set_defaults(func=_cmd_factor_library_rebuild)

    fl_ls = fl_sub.add_parser("list", help="列出库内因子（rank/expression/holdout_ic/status）")
    fl_ls.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare")
    fl_ls.set_defaults(func=_cmd_factor_library_list)

    fl_sh = fl_sub.add_parser("show", help="单因子全字段")
    fl_sh.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare")
    fl_sh.add_argument("--expression", default=None, help="按表达式（规范形）查")
    fl_sh.add_argument("--rank", type=int, default=None, help="按库内排名查（1-based，holdout_ic 降序）")
    fl_sh.set_defaults(func=_cmd_factor_library_show)

    fl_rd = fl_sub.add_parser("render", help="重生 {market}.md（不重算）")
    fl_rd.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare")
    fl_rd.set_defaults(func=_cmd_factor_library_render)

    # ── fz validate ──（与 fz mine 并列的顶层命令组）
    # ── fz research ──（端到端编排：mine → 头部 passed 因子 → 循环 build → sim → report）
    research = sub.add_parser("research", help="End-to-end research orchestration")
    research_sub = research.add_subparsers(dest="research_command", required=True)
    r_run = research_sub.add_parser(
        "run", help="mine → 头部 passed 因子 → 按调仓日循环 build → sim → report（同一 run_id）")
    r_run.add_argument("--start", required=True, help="Start date YYYYMMDD")
    r_run.add_argument("--end", required=True, help="End date YYYYMMDD")
    r_run.add_argument("--universe", default=None, help="Universe name (default all_a)")
    r_run.add_argument("--method", choices=["random", "genetic"], default="random")
    r_run.add_argument("--trials", type=int, default=200)
    r_run.add_argument("--top-k", dest="top_k", type=int, default=10)
    r_run.add_argument("--seed", type=int, default=42)
    r_run.add_argument("--rebalance-days", dest="rebalance_days", type=int, default=20,
                       help="调仓间隔（交易日数，默认 20≈月频）")
    r_run.add_argument("--warmup", type=int, default=60,
                       help="起始跳过的交易日数，留给时序算子 lookback（默认 60）")
    r_run.add_argument("--lookback", type=int, default=60,
                       help="因子计算 lookback 交易日数（默认 60）")
    r_run.add_argument("--lam", type=float, default=1.0, help="风险厌恶系数（默认 1.0）")
    r_run.add_argument("--w-max", dest="w_max", type=float, default=0.05,
                       help="单票权重上限（默认 0.05）")
    r_run.add_argument("--turnover", type=float, default=None, help="换手预算（默认无约束）")
    r_run.add_argument("--industry-neutral", dest="industry_neutral", action="store_true",
                       help="行业中性到 universe 等权基准")
    r_run.add_argument("--run-id", dest="run_id", default=None,
                       help="贯穿全链路的 run_id（默认 research_<seed>_<method>）")
    r_run.set_defaults(func=_cmd_research_run)

    validate = sub.add_parser("validate", help="Overfitting / robustness checks")
    validate_sub = validate.add_subparsers(dest="validate_command", required=True)
    vo = validate_sub.add_parser("overfit", help="Deflated Sharpe + bootstrap CI for one factor")
    vo.add_argument("factor", nargs="?", help="Registered factor name (ashare)")
    vo.add_argument("--start", required=True)
    vo.add_argument("--end", required=True)
    vo.add_argument("--universe", default=None)
    vo.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare",
                    help="Market profile (default ashare; crypto/futures/us 需 --expression)")
    vo.add_argument("--expression", default=None,
                    help="Factor expression to validate (required for --market crypto/futures/us)")
    vo.add_argument("--top-n", dest="top_n", type=int, default=50,
                    help="crypto/futures/us universe size (default 50)")
    _add_freq_arg(vo)
    vo.set_defaults(func=_cmd_validate_overfit)

    # ── fz risk ──（顶层命令组）
    risk = sub.add_parser("risk", help="Risk model workflows")
    risk_sub = risk.add_subparsers(dest="risk_command", required=True)
    r_build = risk_sub.add_parser("build", help="Build Barra risk model")
    r_build.add_argument("--start", required=True, help="Start date YYYYMMDD")
    r_build.add_argument("--end", required=True, help="End date YYYYMMDD")
    r_build.add_argument("--universe", default="all_a", help="Universe name")
    r_build.add_argument("--cov-half-life", type=int, default=90, dest="cov_half_life")
    r_build.add_argument("--nw-lags", type=int, default=2, dest="nw_lags")
    r_build.add_argument("--spec-half-life", type=int, default=90, dest="spec_half_life")
    r_build.add_argument("--spec-shrinkage", type=float, default=0.3, dest="spec_shrinkage")
    r_build.set_defaults(func=_cmd_risk_build)

    # ── fz portfolio ──（顶层命令组）
    portfolio = sub.add_parser("portfolio", help="Portfolio construction & attribution")
    pf_sub = portfolio.add_subparsers(dest="portfolio_command", required=True)
    p_build = pf_sub.add_parser("build", help="Build optimized portfolio + attribution")
    p_build.add_argument("--start", required=True)
    p_build.add_argument("--end", required=True)
    p_build.add_argument("--universe", default="all_a")
    p_build.add_argument(
        "--alpha-file",
        required=True,
        dest="alpha_file",
        help="α 信号文件(parquet/csv: 列 ts_code + alpha)",
    )
    p_build.add_argument("--lam", type=float, default=1.0, dest="lam", help="风险厌恶系数")
    p_build.add_argument("--w-max", type=float, default=0.05, dest="w_max")
    p_build.add_argument("--turnover", type=float, default=None)
    p_build.add_argument("--industry-neutral", action="store_true", dest="industry_neutral")
    p_build.add_argument("--market", choices=["ashare", "crypto"], default="ashare",
                         help="Market profile (default ashare; crypto=市场中性做空)")
    p_build.add_argument("--top-n", dest="top_n", type=int, default=50,
                         help="crypto universe size (default 50)")
    p_build.add_argument("--gross-limit", dest="gross_limit", type=float, default=1.0,
                         help="crypto 毛敞口上限 Σ|w| (default 1.0)")
    p_build.add_argument("--run-id", dest="run_id", default=None,
                         help="产物子目录名(默认=end 日期串)；多期构建须用不同 run_id 避免覆盖")
    p_build.add_argument("--out-dir", dest="out_dir", default="workspace/portfolios",
                         help="组合产物根目录(默认 workspace/portfolios)")
    _add_freq_arg(p_build)
    p_build.set_defaults(func=_cmd_portfolio_build)

    # ── fz sim ──（顶层命令组）
    sim = sub.add_parser("sim", help="Portfolio simulation workflows")
    sim_sub = sim.add_subparsers(dest="sim_command", required=True)

    s_run = sim_sub.add_parser("run", help="Run portfolio simulation")
    s_run.add_argument(
        "--portfolio-dir",
        required=True,
        dest="portfolio_dir",
        help="组合产物根目录，其下各 {run_id}/ 含 weights.parquet + manifest.json",
    )
    s_run.add_argument("--start", required=True, help="Start date YYYYMMDD")
    s_run.add_argument("--end", required=True, help="End date YYYYMMDD")
    s_run.add_argument("--run-id", default=None, dest="run_id", help="可选输出 run_id")
    s_run.add_argument("--market", choices=["ashare", "crypto"], default="ashare",
                       help="Market profile (default ashare; crypto=funding+做空 NAV 回测)")
    s_run.add_argument("--top-n", dest="top_n", type=int, default=50,
                       help="crypto universe size (default 50)")
    _add_freq_arg(s_run)
    s_run.set_defaults(func=_cmd_sim_run)

    s_show = sim_sub.add_parser("show", help="Show simulation metrics")
    s_show.add_argument(
        "--sim-dir",
        required=True,
        dest="sim_dir",
        help="模拟输出目录（含 metrics.json）",
    )
    s_show.set_defaults(func=_cmd_sim_show)

    # ── fz live ──（顶层命令组）
    live = sub.add_parser("live", help="向前执行(纸面/实盘)工作流")
    live_sub = live.add_subparsers(dest="live_command", required=True)
    lp = live_sub.add_parser("replay", help="历史窗口 replay 出向前 NAV(A类)")
    lp.add_argument("--session-dir", required=True, dest="session_dir")
    lp.add_argument("--portfolio-run-dir", action="append", required=True, dest="portfolio_run_dirs")
    lp.add_argument("--start", required=True)   # 行情窗口起(YYYYMMDD)
    lp.add_argument("--end", required=True)      # 行情窗口止
    lp.add_argument("--universe", default=None)
    lp.add_argument("--initial-cash", type=float, default=1_000_000.0, dest="initial_cash")
    lp.add_argument("--broker", choices=["paper"], default="paper")
    lp.add_argument("--from-date", default=None, dest="from_date")  # 可选:窗口内进一步裁剪(YYYY-MM-DD)
    lp.add_argument("--to-date", default=None, dest="to_date")
    lp.add_argument("--seed", type=int, default=0)
    lp.set_defaults(func=_cmd_live_replay)

    li = live_sub.add_parser("init", help="初始化向前会话")
    li.add_argument("--session-dir", required=True, dest="session_dir")
    li.add_argument("--initial-cash", type=float, default=1_000_000.0, dest="initial_cash")
    li.add_argument("--slippage-bps", type=float, default=0.0, dest="slippage_bps")
    li.add_argument("--broker", choices=["paper"], default="paper")
    li.set_defaults(func=_cmd_live_init)

    ls = live_sub.add_parser("step", help="推进一个交易日(可续跑)")
    ls.add_argument("--session-dir", required=True, dest="session_dir")
    ls.add_argument("--date", required=True)  # YYYYMMDD
    ls.add_argument(
        "--portfolio-run-dir", action="append", required=True, dest="portfolio_run_dirs"
    )
    ls.add_argument("--start", required=True)  # 行情窗口(含ADV回看)
    ls.add_argument("--end", required=True)
    ls.add_argument("--universe", default=None)
    ls.set_defaults(func=_cmd_live_step)

    lst = live_sub.add_parser("status", help="打印会话当前状态")
    lst.add_argument("--session-dir", required=True, dest="session_dir")
    lst.set_defaults(func=_cmd_live_status)

    lr = live_sub.add_parser("report", help="生成A类分歧归因报告")
    lr.add_argument("--session-dir", required=True, dest="session_dir")
    lr.add_argument(
        "--portfolio-run-dir", action="append", required=True, dest="portfolio_run_dirs"
    )
    lr.add_argument("--start", required=True)
    lr.add_argument("--end", required=True)
    lr.add_argument("--universe", default=None)
    lr.set_defaults(func=_cmd_live_report)

    # ── combine:多因子组合 OOS 对比 ──
    combine = sub.add_parser("combine", help="多因子组合 OOS 对比实验")
    combine_sub = combine.add_subparsers(dest="combine_command", required=True)
    cr = combine_sub.add_parser("run", help="四方法(等权/IC加权/max_ir/lgbm)OOS 对比")
    cr.add_argument(
        "--factor", action="append", required=True, dest="factors",
        help="因子 parquet[trade_date,ts_code,factor_value](可多次)",
    )
    cr.add_argument("--ret", required=True, help="前向收益 parquet[trade_date,ts_code,ret]")
    cr.add_argument("--train-days", type=int, default=120, dest="train_days")
    cr.add_argument("--test-days", type=int, default=20, dest="test_days")
    cr.add_argument("--purge-days", type=int, default=5, dest="purge_days")
    cr.add_argument("--embargo-days", type=int, default=0, dest="embargo_days")
    cr.add_argument("--methods", default="all", help="逗号分隔(equal_weight,ic_weighted,max_ir,lgbm)或 all")
    cr.add_argument("--seed", type=int, default=0)
    cr.add_argument("--run-id", default=None, dest="run_id")
    cr.add_argument("--out-dir", default="workspace/combinations", dest="out_dir")
    cr.set_defaults(func=_cmd_combine_run)

    # combine from-session:挖掘因子库 → 物化 → 组合 OOS(端到端接线)
    cfs = combine_sub.add_parser("from-session",
                                 help="从挖掘 session 的因子库直接跑组合 OOS(物化+收益面板自动生成)")
    cfs.add_argument("--session", required=True, nargs="+",
                     help="挖掘 session 目录(含 candidates.csv)，可传多个跨 run 合并去重")
    cfs.add_argument("--start", required=True, help="物化窗口起 YYYYMMDD")
    cfs.add_argument("--end", required=True, help="物化窗口止 YYYYMMDD")
    cfs.add_argument("--universe", default=None, help="票池(默认全A)")
    cfs.add_argument("--horizon", type=int, default=5, help="前向收益持有期(交易日,默认5)")
    cfs.add_argument("--top-n", dest="top_n", type=int, default=None, help="只取库前 N 个因子")
    cfs.add_argument("--decorr-threshold", dest="decorr_threshold", type=float, default=0.7,
                     help="贪心去相关阈值(|corr|>阈值剔除近亲；1.0 关闭，默认0.7)")
    cfs.add_argument("--all", action="store_true", help="含未过护栏的因子(默认只用 passed 库因子)")
    cfs.add_argument("--train-days", type=int, default=120, dest="train_days")
    cfs.add_argument("--test-days", type=int, default=20, dest="test_days")
    cfs.add_argument("--purge-days", type=int, default=5, dest="purge_days")
    cfs.add_argument("--embargo-days", type=int, default=0, dest="embargo_days")
    cfs.add_argument("--methods", default="all", help="逗号分隔或 all")
    cfs.add_argument("--seed", type=int, default=0)
    cfs.add_argument("--run-id", default=None, dest="run_id")
    cfs.add_argument("--out-dir", default="workspace/combinations", dest="out_dir")
    cfs.set_defaults(func=_cmd_combine_from_session)

    # ── ops:无人值守运营 ──
    ops = sub.add_parser("ops", help="无人值守运营(每日链路)")
    ops_sub = ops.add_subparsers(dest="ops_command", required=True)

    od = ops_sub.add_parser("daily", help="执行一个交易日的无人值守链路")
    od.add_argument("--config", required=True, help="ops.yaml 配置路径")
    od.add_argument("--date", default=None, help="YYYYMMDD,缺省今天")
    od.set_defaults(func=_cmd_ops_daily)

    ost = ops_sub.add_parser("status", help="打印某日各阶段状态")
    ost.add_argument("--config", required=True, help="ops.yaml 配置路径")
    ost.add_argument("--date", default=None, help="YYYYMMDD,缺省今天")
    ost.set_defaults(func=_cmd_ops_status)

    return parser


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
