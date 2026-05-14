"""Intraday single-factor IC evaluation.

Usage:
    python scripts/run_intraday_single.py --factor momentum_1min --start 20260401 --end 20260430
    python scripts/run_intraday_single.py --factor momentum_1min --start 20260401 --end 20260430 --demo

The --demo flag generates synthetic minute-bar data so the script can run
without real data loaded in data/raw/minute/.
"""

import argparse
import base64
import io
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl

from config.settings import OUTPUT_INTRADAY_FACTORS, OUTPUT_INTRADAY_RESULTS, OUTPUT_INTRADAY_REPORTS
from common.logger import setup_logging, get_logger
from intraday.evaluation.ic_analysis import compute_intraday_rank_ic

setup_logging()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Synthetic data generator (demo / offline mode)
# ---------------------------------------------------------------------------

def _make_demo_data(
    start: str,
    end: str,
    n_stocks: int = 50,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate synthetic minute-bar factor + return data for demo purposes."""
    rng = np.random.default_rng(seed)

    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")
    n_days = (end_dt - start_dt).days + 1
    trade_days = [
        start_dt + timedelta(days=d)
        for d in range(n_days)
        if (start_dt + timedelta(days=d)).weekday() < 5
    ]

    # 09:30 to 14:55 every 5 minutes = (5*60+25)/5 + 1 = 67 bars/day
    minute_offsets = list(range(0, 325, 5))  # 0..320 min from 09:30

    factor_rows = []
    ret_rows = []
    for day in trade_days:
        for offset in minute_offsets:
            ts = day + timedelta(hours=9, minutes=30 + offset)
            for i in range(n_stocks):
                code = f"{i:06d}.SH"
                factor_rows.append({
                    "trade_time": ts,
                    "ts_code": code,
                    "factor_value": float(rng.standard_normal()),
                })
                ret_rows.append({
                    "trade_time": ts,
                    "ts_code": code,
                    "fwd_ret_1bar": float(rng.standard_normal() * 0.002),
                })

    return pl.DataFrame(factor_rows), pl.DataFrame(ret_rows)


# ---------------------------------------------------------------------------
# Real data loader (requires data/raw/minute/ parquet files)
# ---------------------------------------------------------------------------

def _load_real_data(
    factor_name: str,
    start: str,
    end: str,
    universe: list[str] | None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load minute-bar factor data and compute next-bar returns from storage."""
    from common.storage import load_parquet
    from intraday.data.context import MFTDataContext
    from intraday.factors.registry import get_factor as get_intraday_factor

    factor_cls = get_intraday_factor(factor_name)
    ctx = MFTDataContext(start=start, end=end, universe=universe)
    factor_df = factor_cls().compute(ctx)

    # Build 1-bar forward return from minute close prices
    minute_df = ctx.minute.collect().sort(["ts_code", "trade_time"])
    ret_df = minute_df.with_columns(
        pl.col("close").shift(-1).over("ts_code").alias("_next_close")
    ).with_columns(
        ((pl.col("_next_close") / pl.col("close")) - 1.0).alias("fwd_ret_1bar")
    ).select(["trade_time", "ts_code", "fwd_ret_1bar"]).filter(
        pl.col("fwd_ret_1bar").is_not_null()
    )

    return factor_df, ret_df


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _chart_daily_ic(daily_ic: pl.DataFrame) -> str | None:
    """Render daily IC series as base64 PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if daily_ic.is_empty():
            return None

        dates = daily_ic["trade_date"].to_list()
        ics = daily_ic["ic"].to_list()
        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in ics]
        cumulative_ic = np.cumsum([v if v is not None else 0 for v in ics])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

        ax1.bar(range(len(dates)), ics, color=colors, alpha=0.7, width=0.6)
        ax1.axhline(0, color="black", linewidth=0.5)
        ax1.set_ylabel("Daily IC", fontsize=10)
        ax1.set_title("Intraday Factor Daily IC", fontsize=12)

        ax2.plot(range(len(dates)), cumulative_ic, color="#3498db", linewidth=1.5)
        ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax2.set_ylabel("Cumulative IC", fontsize=10)
        ax2.set_xticks(range(len(dates)))
        ax2.set_xticklabels(
            [str(d) for d in dates],
            rotation=45, ha="right", fontsize=8
        )

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Chart render failed: {e}")
        return None


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _render_html(result, factor_name: str, bar_size: str, universe_label: str, date_range: str) -> str:
    """Render intraday IC result to HTML using Jinja2 template."""
    from datetime import datetime as dt
    from jinja2 import Environment, FileSystemLoader

    template_dir = Path(__file__).resolve().parent.parent / "reporting" / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
    template = env.get_template("intraday_ic.html")

    daily_ic_chart = _chart_daily_ic(result.daily_ic)

    daily_ic_table = None
    if not result.daily_ic.is_empty():
        daily_ic_table = result.daily_ic.to_dicts()

    segment_table = None
    if not result.segment_ic.is_empty():
        segment_table = result.segment_ic.to_dicts()

    metrics = {
        "ic_mean": result.ic_mean,
        "ic_std": result.ic_std,
        "ir": result.ir,
        "ic_positive_ratio": result.ic_positive_ratio,
        "n_periods": result.n_periods,
        "daily_ic_table": daily_ic_table,
        "segment_table": segment_table,
    }

    return template.render(
        factor_name=factor_name,
        bar_size=bar_size,
        universe=universe_label,
        date_range=date_range,
        generated_at=dt.now().strftime("%Y-%m-%d %H:%M"),
        metrics=metrics,
        charts={"daily_ic_chart": daily_ic_chart},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Intraday single-factor IC evaluation")
    parser.add_argument("--factor", default="momentum_1min", help="Factor name")
    parser.add_argument("--start", required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date YYYYMMDD")
    parser.add_argument("--universe", default=None, help="Comma-separated ts_codes or preset name")
    parser.add_argument("--demo", action="store_true", help="Use synthetic data (no real data needed)")
    args = parser.parse_args()

    logger.info(f"==== Intraday IC Evaluation: {args.factor} | {args.start}~{args.end} ====")

    if args.demo:
        logger.info("Demo mode: generating synthetic minute-bar data")
        factor_df, ret_df = _make_demo_data(args.start, args.end)
        universe_label = f"synthetic (50 stocks)"
    else:
        universe = args.universe.split(",") if args.universe else None
        try:
            factor_df, ret_df = _load_real_data(args.factor, args.start, args.end, universe)
        except Exception as e:
            logger.error(f"Real data load failed: {e}. Try --demo flag for synthetic data.")
            sys.exit(1)
        universe_label = args.universe or "all"

    logger.info(f"Factor rows: {len(factor_df):,} | Return rows: {len(ret_df):,}")

    result = compute_intraday_rank_ic(factor_df, ret_df)
    result_with_name = result.__class__(
        factor_name=args.factor,
        ic_mean=result.ic_mean,
        ic_std=result.ic_std,
        ir=result.ir,
        ic_positive_ratio=result.ic_positive_ratio,
        n_periods=result.n_periods,
        daily_ic=result.daily_ic,
        segment_ic=result.segment_ic,
    )

    logger.info(f"\n{result_with_name.summary()}")

    # Save outputs
    OUTPUT_INTRADAY_FACTORS.mkdir(parents=True, exist_ok=True)
    OUTPUT_INTRADAY_RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUT_INTRADAY_REPORTS.mkdir(parents=True, exist_ok=True)

    factor_df.write_parquet(
        str(OUTPUT_INTRADAY_FACTORS / f"{args.factor}_{args.start}_{args.end}.parquet")
    )

    if not result.daily_ic.is_empty():
        result.daily_ic.write_parquet(
            str(OUTPUT_INTRADAY_RESULTS / f"{args.factor}_daily_ic.parquet")
        )

    # Render HTML
    date_range = f"{args.start} ~ {args.end}"
    html = _render_html(result_with_name, args.factor, "1min", universe_label, date_range)
    report_path = OUTPUT_INTRADAY_REPORTS / f"{args.factor}_{args.start}_{args.end}.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info(f"Report saved: {report_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
