"""批量拉取 CSI300（或自定义）股票分钟线数据。

用法:
  # 2000积分账号（stk_mins限速 2次/分钟），必须加 --call-delay 62
  pixi run python scripts/fetch_minute_data.py --start 20260301 --end 20260516 --call-delay 62
  # 测试时先用 --n-stocks 5 验证，再扩大 universe
  pixi run python scripts/fetch_minute_data.py --start 20260301 --end 20260516 --call-delay 62 --n-stocks 5

注意:
  stk_mins 接口 2000积分账号限 2次/分钟（约1次/30秒）。
  不加 --call-delay 会因频率超限触发临时封禁（可能持续 10-30 分钟）。
"""

import argparse
import sys
import time

from common.loader import fetch_minute
from common.logger import get_logger, setup_logging
from common.universe import get_universe

setup_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="批量拉取分钟线数据")
    parser.add_argument("--start", required=True, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", required=True, help="截止日期 YYYYMMDD")
    parser.add_argument("--universe", default="csi300", help="股票池名称（csi300 / csi800）")
    parser.add_argument(
        "--freq", default="1min", help="分钟频率：1min / 5min / 15min / 30min / 60min"
    )
    parser.add_argument("--delay", type=float, default=0.5, help="每只股票间隔秒数（防限流）")
    parser.add_argument(
        "--call-delay",
        type=float,
        default=62.0,
        help="stk_mins 每次 API 调用后最小间隔秒数（2000积分账号限 2次/分钟，默认 62.0 跨越固定窗口）",
    )
    parser.add_argument(
        "--n-stocks",
        type=int,
        default=None,
        help="仅抓取股票池前 N 只（用于测试或分批拉取）",
    )
    args = parser.parse_args()

    universe = get_universe(args.end, args.universe)
    if universe.is_empty():
        logger.error(f"股票池为空: {args.universe}")
        sys.exit(1)
    ts_codes = universe["ts_code"].to_list()
    if args.n_stocks is not None:
        ts_codes = ts_codes[: args.n_stocks]
        logger.info(f"--n-stocks 限制，仅取前 {len(ts_codes)} 只")
    logger.info(
        f"股票池 {args.universe}: {len(ts_codes)} 只，准备拉取 {args.freq} 分钟线 {args.start}~{args.end}"
    )

    ok, fail = 0, 0
    for i, ts_code in enumerate(ts_codes, 1):
        logger.info(f"[{i}/{len(ts_codes)}] {ts_code}")
        try:
            fetch_minute(ts_code, args.freq, args.start, args.end, call_delay=args.call_delay)
            ok += 1
        except Exception as e:
            logger.error(f"  {ts_code} 失败: {e}")
            fail += 1
        if args.delay > 0 and i < len(ts_codes):
            time.sleep(args.delay)

    logger.info(f"完成: 成功 {ok} 只，失败 {fail} 只")


if __name__ == "__main__":
    main()
