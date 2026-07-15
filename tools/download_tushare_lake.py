#!/usr/bin/env python
"""可续跑的 Tushare 数据湖下载器（支持官方或兼容代理端点）。

设计要点：
  - 全局限速（滚动 60s 窗口，默认 ≤140/min，留 150 硬限余量）。
  - 断点续传：覆盖账本（_ledger/*.done），跳过已完成逻辑单元，不靠"文件存在"。
  - 原子写 parquet（.tmp → rename），中断不产生半截文件。
  - 瞬时错误退避重试；鉴权/参数类错误不重试并记 errors.jsonl。
  - 优先级：参考骨架 → 必需日线(daily/adj/basic) → 1min 分钟 → 其余日线 → 基本面/股东 → 指数/板块。

凭据只从项目 ``.env`` 的 ``TUSHARE_TOKEN`` 读取，不写入数据湖或 manifest。
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import polars as pl
import requests

from factorzen.config.settings import DATA_DIR
from factorzen.config.tushare_config import ensure_token
from factorzen.core.experiment import get_git_sha

URL = os.environ.get("TUSHARE_API_URL", "http://api.tushare.pro")
TOKEN = ""
MAX_PER_MIN = int(os.environ.get("TUSHARE_LAKE_MAX_PER_MIN", "140"))
MIN_INTERVAL = 60.0 / MAX_PER_MIN
WORKERS = int(os.environ.get("TUSHARE_LAKE_WORKERS", "6"))
ROW_CAP = 8000  # 单次返回上限
MINUTE_CHUNK_DAYS = 30  # 30 交易日 × 241 = 7230 < 8000

# 鉴权/参数类错误码：不重试，直接记账跳过
HARD_CODES = {40001, 40101, 40102, 40103, 50101, 50102}
# 明确无权限的 msg 关键词
PERM_MSG = ("不在积分档", "没有权限", "权限", "单独开通", "积分")


class RateLimiter:
    """线程安全预约式限速器：锁内预约时间槽，锁外 sleep，保证并发下聚合 ≤max/min。"""

    def __init__(self, max_per_min: int):
        self.max = max_per_min
        self.interval = 60.0 / max_per_min
        self.lock = threading.Lock()
        self.calls: deque[float] = deque()  # 已预约槽时间戳
        self.next_slot = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_slot)  # 最小间隔约束
            # 滚动 60s 窗口：任一 60s 内不超过 max
            while self.calls and self.calls[0] <= t - 60.0:
                self.calls.popleft()
            if len(self.calls) >= self.max:
                t = max(t, self.calls[0] + 60.0)
                while self.calls and self.calls[0] <= t - 60.0:
                    self.calls.popleft()
            self.next_slot = t + self.interval
            self.calls.append(t)
            sleep_for = t - now
        if sleep_for > 0:
            time.sleep(sleep_for)


RL = RateLimiter(MAX_PER_MIN)


class HardError(Exception):
    """鉴权/参数类错误——不重试。"""


class FatalAuth(Exception):
    """全局鉴权失效（token 过期/封禁）——应停止整个任务。"""


class DailyCap(Exception):
    """某接口今日调用配额用尽（如 stk_mins 20000/天，code=-2001）——今日停该接口，明日续。"""


def api_call(
    api_name: str, params: dict, fields: str = "", max_tries: int = 6
) -> tuple[list[str], list[list]]:
    """返回 items(list of rows)。瞬时错误退避重试；硬错误抛 HardError。

    返回 (cols, items)。
    """
    backoff = 2.0
    last_err = ""
    for _attempt in range(1, max_tries + 1):
        RL.wait()
        try:
            resp = requests.post(
                URL,
                json={"api_name": api_name, "token": TOKEN, "params": params, "fields": fields},
                headers={"Accept-Encoding": "gzip"},
                timeout=60,
            )
        except Exception as e:  # 网络抖动/超时 → 重试
            last_err = f"{type(e).__name__}:{str(e)[:80]}"
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
        try:
            j = resp.json()
        except Exception:  # 非 JSON（代理偶发）→ 重试
            last_err = f"non-json http={resp.status_code} body={resp.text[:80]!r}"
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
        code = j.get("code")
        if code == 0:
            data = j.get("data") or {}
            return data.get("fields") or [], (data.get("items") or [])
        msg = (j.get("msg") or "")
        # 注意 code=-2001 被代理复用：既表"日配额用尽"，也表"并发请求过多"(瞬时)。
        # 仅按文案区分：真·日配额 → DailyCap(不重试)；并发过多/其它 → 落到下方通用重试分支。
        if "已达上限" in msg or "明日再试" in msg or "调用已达上限" in msg:
            raise DailyCap(f"code={code} msg={msg}")
        # 全局鉴权失效
        if code in (40001,) or any(k in msg for k in ("token", "Token", "过期", "expired")):
            raise FatalAuth(f"code={code} msg={msg}")
        # 无权限 / 参数错误 → 硬错误，不重试
        if code in HARD_CODES or any(k in msg for k in PERM_MSG):
            raise HardError(f"code={code} msg={msg}")
        # 其它（限流/瞬时 code=-1 等）→ 重试
        last_err = f"code={code} msg={msg[:80]}"
        time.sleep(backoff)
        backoff = min(backoff * 2, 60)
    raise TimeoutError(f"{api_name} 重试 {max_tries} 次仍失败: {last_err}")


def _to_df(cols: list[str], items: list[list]) -> pl.DataFrame:
    """按列构造（全列扫描推断类型）；混合/异常回退全字符串，保证任何接口都能落盘。"""
    data = {c: [row[i] if i < len(row) else None for row in items] for i, c in enumerate(cols)}
    try:
        # strict=False：把同列的 int/float 混合提升为 Float64（数值字段常见），保持数值类型
        return pl.DataFrame(data, strict=False)
    except Exception:
        # 数值 vs 字符串真混合 → 回退全字符串，保证任何接口都能落盘
        sdata = {c: [None if v is None else str(v) for v in vals] for c, vals in data.items()}
        return pl.DataFrame(sdata, schema={c: pl.Utf8 for c in cols})


class Lake:
    def __init__(self, root: Path):
        self.root = root
        self.ledger_dir = root / "_ledger"
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self._done_cache: dict[str, set] = {}
        self.err_path = self.ledger_dir / "errors.jsonl"
        self.lock = threading.Lock()

    def done_set(self, ledger: str) -> set:
        if ledger not in self._done_cache:
            p = self.ledger_dir / f"{ledger}.done"
            s = set()
            if p.exists():
                s = {ln.strip() for ln in p.read_text().splitlines() if ln.strip()}
            self._done_cache[ledger] = s
        return self._done_cache[ledger]

    def mark(self, ledger: str, key: str) -> None:
        with self.lock:
            self.done_set(ledger).add(key)
            with (self.ledger_dir / f"{ledger}.done").open("a", encoding="utf-8") as f:
                f.write(key + "\n")

    def log_err(self, api: str, key: str, err: str) -> None:
        rec = {"api": api, "key": key, "err": str(err)[:300]}
        with self.lock, self.err_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def write_parquet(self, rel: str, cols: list[str], items: list[list]) -> int:
        if not items:
            return 0
        out = self.root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        df = _to_df(cols, items)
        tmp = out.with_suffix(out.suffix + ".tmp")
        df.write_parquet(tmp)
        tmp.replace(out)
        return df.height


def load_trade_dates(lake: Lake, start: str, end: str) -> list[str]:
    """SSE 开市日列表（升序），并落 reference/trade_cal.parquet。"""
    cols, items = api_call("trade_cal", {"exchange": "SSE", "start_date": start, "end_date": end},
                           "exchange,cal_date,is_open,pretrade_date")
    lake.write_parquet("reference/trade_cal.parquet", cols, items)
    ci = cols.index("cal_date")
    oi = cols.index("is_open")
    dates = sorted({r[ci] for r in items if r[oi] in (1, "1")})
    return dates


def load_universe(lake: Lake) -> list[dict]:
    """全市场股票池（含退市），落 reference/stock_basic.parquet。返回 dict 列表。"""
    fields = ("ts_code,symbol,name,area,industry,fullname,market,exchange,"
              "list_status,list_date,delist_date,is_hs,curr_type")
    all_items = []
    cols = None
    for st in ("L", "D", "P"):
        try:
            c, items = api_call("stock_basic", {"list_status": st}, fields)
            cols = c or cols
            all_items.extend(items)
        except HardError:
            pass
    lake.write_parquet("reference/stock_basic.parquet", cols, all_items)
    idx = {name: i for i, name in enumerate(cols)}
    uni = []
    for r in all_items:
        uni.append({
            "ts_code": r[idx["ts_code"]],
            "list_date": r[idx["list_date"]],
            "delist_date": r[idx.get("delist_date", idx["ts_code"])] if "delist_date" in idx else None,
        })
    return uni


# ────────────────────────── 各阶段 ──────────────────────────

def phase_reference(lake: Lake) -> None:
    log("=== Phase 0: 参考骨架 ===")
    refs = [
        ("index_basic", {"market": "SSE"}, "index_basic_SSE"),
        ("index_basic", {"market": "SZSE"}, "index_basic_SZSE"),
        ("index_basic", {"market": "CSI"}, "index_basic_CSI"),
        ("index_basic", {"market": "SW"}, "index_basic_SW"),
        ("index_classify", {"src": "SW2021"}, "index_classify_SW2021"),
        ("ths_index", {}, "ths_index"),
        ("namechange", {}, "namechange"),
    ]
    for api, params, name in refs:
        if name in lake.done_set("reference"):
            continue
        try:
            cols, items = api_call(api, params)
            n = lake.write_parquet(f"reference/{name}.parquet", cols, items)
            lake.mark("reference", name)
            log(f"  {name}: {n} 行")
        except (HardError, TimeoutError) as e:
            lake.log_err(api, name, e)
            log(f"  {name}: 跳过 ({e})")


STOP = threading.Event()       # 全局停止（token 失效等 FatalAuth 触发）
MINUTE_CAP = threading.Event()  # 分钟接口今日配额用尽（DailyCap 触发），本轮停分钟


def _parallel(items: list, worker_fn, label: str, every: int) -> int:
    """并发执行 worker_fn(item)（返回 True=成功）。FatalAuth 置 STOP 并向上抛。"""
    total = len(items)
    if not total:
        return 0
    t0 = time.time()
    n_ok = 0
    n_done = 0
    fatal = None
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(worker_fn, it) for it in items]
        for fut in as_completed(futs):
            n_done += 1
            try:
                if fut.result():
                    n_ok += 1
            except FatalAuth as e:
                fatal = e
                STOP.set()
            except Exception as e:
                log(f"  [{label}] worker 异常: {type(e).__name__}: {str(e)[:120]}")
            if n_done % every == 0:
                el = time.time() - t0
                rate = n_done / el if el else 0
                eta = (total - n_done) / rate / 60 if rate else 0
                log(f"  [{label}] {n_done}/{total}  {rate*60:.0f}/min  ETA {eta:.0f}min")
    log(f"  [{label}] 完成 {n_ok}/{total}  用时 {(time.time()-t0)/60:.1f}min")
    if fatal:
        raise fatal
    return n_ok


def run_bydate(lake: Lake, api: str, dates: list[str], extra: dict | None = None,
               date_key: str = "trade_date", fields: str = "") -> None:
    ledger = f"bydate_{api}"
    done = lake.done_set(ledger)
    todo = [d for d in dates if d not in done]
    if not todo:
        log(f"  [{api}] 已完成 {len(dates)} 日")
        return
    log(f"  [{api}] 待抓 {len(todo)}/{len(dates)} 交易日（并发 {WORKERS}）")
    cap = threading.Event()

    def work(d: str) -> bool:
        if STOP.is_set() or cap.is_set():
            return False
        params = {date_key: d}
        if extra:
            params.update(extra)
        try:
            cols, items = api_call(api, params, fields)
            lake.write_parquet(f"daily/{api}/date={d}.parquet", cols, items)
            lake.mark(ledger, d)
            return True
        except DailyCap as e:
            lake.log_err(api, d, e)
            cap.set()
            return False  # 该接口今日配额尽
        except HardError as e:
            lake.log_err(api, d, e)
            lake.mark(ledger, d)
            return False
        except TimeoutError as e:
            lake.log_err(api, d, e)
            return False

    # 预探一枪：整表无权限则快速跳过（避免 1600+ 次无谓调用）
    try:
        p0 = {date_key: todo[0], **(extra or {})}
        cols, items = api_call(api, p0, fields)
        lake.write_parquet(f"daily/{api}/date={todo[0]}.parquet", cols, items)
        lake.mark(ledger, todo[0])
        rest = todo[1:]
    except DailyCap as e:
        log(f"  [{api}] 今日配额已尽，跳过（明日续）: {e}")
        return
    except HardError as e:
        log(f"  [{api}] 首枪硬错误，整表跳过: {e}")
        lake.log_err(api, "ALL", e)
        for d in todo:
            lake.mark(ledger, d)
        return
    except TimeoutError:
        rest = todo  # 瞬时失败，交给并行重试
    _parallel(rest, work, api, 200)


def run_perstock(lake: Lake, api: str, uni: list[dict], start: str, end: str,
                 use_dates: bool = True, fields: str = "") -> None:
    ledger = f"perstock_{api}"
    done = lake.done_set(ledger)
    todo = [u for u in uni if u["ts_code"] not in done]
    if not todo:
        log(f"  [{api}] 已完成 {len(uni)} 只")
        return
    log(f"  [{api}] 待抓 {len(todo)}/{len(uni)} 只（并发 {WORKERS}）")
    cap = threading.Event()

    def work(u: dict) -> bool:
        if STOP.is_set() or cap.is_set():
            return False
        code = u["ts_code"]
        params = {"ts_code": code}
        if use_dates:
            params["start_date"] = start
            params["end_date"] = end
        try:
            cols, items = api_call(api, params, fields)
            lake.write_parquet(f"perstock/{api}/{code}.parquet", cols, items)
            lake.mark(ledger, code)
            return True
        except DailyCap as e:
            lake.log_err(api, code, e)
            cap.set()
            return False
        except HardError as e:
            lake.log_err(api, code, e)
            lake.mark(ledger, code)
            return False
        except TimeoutError as e:
            lake.log_err(api, code, e)
            return False

    # 预探一枪
    try:
        params = {"ts_code": todo[0]["ts_code"]}
        if use_dates:
            params["start_date"] = start
            params["end_date"] = end
        cols, items = api_call(api, params, fields)
        lake.write_parquet(f"perstock/{api}/{todo[0]['ts_code']}.parquet", cols, items)
        lake.mark(ledger, todo[0]["ts_code"])
        rest = todo[1:]
    except DailyCap as e:
        log(f"  [{api}] 今日配额已尽，跳过（明日续）: {e}")
        return
    except HardError as e:
        log(f"  [{api}] 首枪硬错误，整表跳过: {e}")
        lake.log_err(api, "ALL", e)
        for u in todo:
            lake.mark(ledger, u["ts_code"])
        return
    except TimeoutError:
        rest = todo
    _parallel(rest, work, api, 400)


def run_minute(lake: Lake, uni: list[dict], dates: list[str], start: str) -> None:
    log("=== Phase 1: 1min 分钟数据（长杆）===")
    ledger = "minute_1min"
    done = lake.done_set(ledger)
    date_start_ok = [d for d in dates if d >= start]
    todo = [u for u in uni if u["ts_code"] not in done]
    log(f"  待抓 {len(todo)}/{len(uni)} 只；交易日 {len(date_start_ok)}（{start}~）；并发 {WORKERS}")
    calls = [0]
    clock = threading.Lock()

    def work(u: dict) -> bool:
        if STOP.is_set() or MINUTE_CAP.is_set():
            return False
        code = u["ts_code"]
        lo = max(start, (u["list_date"] or start))
        hi = u["delist_date"] or "99999999"
        sub = [d for d in date_start_ok if lo <= d <= hi]
        if not sub:
            lake.mark(ledger, code)
            return True
        chunks = [sub[k:k + MINUTE_CHUNK_DAYS] for k in range(0, len(sub), MINUTE_CHUNK_DAYS)]
        rows: list[list] = []
        cols_seen: list[str] = []
        ok = True
        for ch in chunks:
            if STOP.is_set() or MINUTE_CAP.is_set():
                return False  # 配额用尽/停止：不标记，保留续传
            sd = f"{ch[0][:4]}-{ch[0][4:6]}-{ch[0][6:]} 09:00:00"
            ed = f"{ch[-1][:4]}-{ch[-1][4:6]}-{ch[-1][6:]} 15:30:00"
            try:
                cols, items = api_call("stk_mins", {"ts_code": code, "freq": "1min",
                                                    "start_date": sd, "end_date": ed})
                with clock:
                    calls[0] += 1
                if cols:
                    cols_seen = cols
                rows.extend(items)
            except DailyCap:
                MINUTE_CAP.set()
                return False  # 今日 20000 用尽，立即停分钟
            except HardError as e:
                lake.log_err("stk_mins", code, e)
                ok = False
                break
            except TimeoutError as e:
                lake.log_err("stk_mins", f"{code}:{ch[0]}-{ch[-1]}", e)
                ok = False
        if not ok:
            return False  # 不标记，下次重试（parquet 若已写将被覆盖）
        if rows and cols_seen:
            ti = cols_seen.index("trade_time")
            rows = list({r[ti]: r for r in rows}.values())
            rows.sort(key=lambda r: r[ti])
            lake.write_parquet(f"minute/1min/{code}.parquet", cols_seen, rows)
        lake.mark(ledger, code)
        return True

    _parallel(todo, work, "minute", 50)
    if MINUTE_CAP.is_set():
        log(f"  ⚠ 分钟今日配额(20000)用尽，共 {calls[0]} 次调用；明日重跑续传")
    else:
        log(f"  分钟阶段结束（本轮全部完成），共 {calls[0]} 次调用")


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATA_DIR / "_tushare_lake"))
    ap.add_argument("--min-start", default="20200101")
    ap.add_argument("--phases", default="0,1,2,3,4")
    ap.add_argument("--smoke", action="store_true", help="仅前 N 只/少量日期，验证端到端")
    ap.add_argument("--smoke-n", type=int, default=3)
    ap.add_argument("--workers", type=int, default=None, help="并发 worker 数（覆盖 LAKE_WORKERS）")
    ap.add_argument("--codes", default="", help="定向模式：逗号分隔 ts_code，只对这些股票跑（校对/查缺补漏）")
    ap.add_argument("--codes-file", default="", help="定向模式：每行一个 ts_code 的文件")
    args = ap.parse_args()

    global TOKEN, WORKERS
    if args.workers:
        WORKERS = args.workers

    TOKEN = ensure_token()

    root = Path(args.root)
    lake = Lake(root)
    phases = set(args.phases.split(","))
    today = datetime.now().strftime("%Y%m%d")

    manifest = {
        "endpoint": URL, "min_start": args.min_start, "end": today,
        "started": datetime.now().isoformat(timespec="seconds"),
        "max_per_min": MAX_PER_MIN, "smoke": args.smoke, "phases": args.phases,
        "command": shlex.join(sys.argv), "git_sha": get_git_sha(),
        "universe": "all_a_history", "root": str(root.resolve()),
    }
    (root).mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"数据湖根: {root}")
    log(f"分钟区间: {args.min_start} ~ {today} | 限速 {MAX_PER_MIN}/min")

    # 骨架：交易日 + 股票池（始终需要）
    cal_start = "20191201"
    dates = load_trade_dates(lake, cal_start, today)
    uni = load_universe(lake)
    # 只保留在分钟区间内有交易可能的股票（退市早于 min_start 的丢弃）
    uni = [u for u in uni if not (u["delist_date"] and u["delist_date"] < args.min_start)]
    log(f"股票池 {len(uni)} 只（含退市，已剔除 {args.min_start} 前退市）；交易日 {len(dates)}")

    # 定向模式：只对指定 ts_code 跑（分钟校对/查缺补漏）。建议配单独 --root 避免与主账本冲突。
    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    if args.codes_file:
        codes += [ln.strip() for ln in Path(args.codes_file).read_text().splitlines() if ln.strip()]
    if codes:
        by = {u["ts_code"]: u for u in uni}
        uni = [by.get(c, {"ts_code": c, "list_date": None, "delist_date": None}) for c in codes]
        log(f"定向模式: {len(uni)} 只指定股票 {codes[:5]}{'...' if len(codes) > 5 else ''}")

    if args.smoke:
        uni = uni[: args.smoke_n]
        dates = dates[-8:]  # 最近 8 个交易日
        log(f"SMOKE: 股票 {len(uni)} 只，日期 {dates}")

    if "0" in phases:
        phase_reference(lake)

    # Phase 2a: 必需日线（先于分钟，使分钟可用：复权因子）
    if "2" in phases:
        log("=== Phase 2a: 必需日线 daily/adj_factor/daily_basic ===")
        for api in ("daily", "adj_factor", "daily_basic"):
            run_bydate(lake, api, dates)

    if "1" in phases:
        run_minute(lake, uni, dates, args.min_start)

    if "2" in phases:
        log("=== Phase 2b: 其余全市场日线 ===")
        # 注：cyq_chips 按 trade_date 返回空（需 ts_code），不走 bydate；如需筹码分布另按逐股拉。
        # 有价值的快接口在前先落地；bak_daily(≈daily+daily_basic 冗余)/stk_factor_pro(~11s/次,可由OHLCV导出) 放最后。
        bydate_more = [
            "moneyflow", "limit_list_d", "top_list", "top_inst", "block_trade",
            "margin_detail", "moneyflow_hsgt", "hsgt_top10",
            "stk_limit", "suspend_d", "index_dailybasic",
            "stk_factor_pro", "bak_daily",
        ]
        for api in bydate_more:
            run_bydate(lake, api, dates)

    if "3" in phases:
        log("=== Phase 3: 基本面 / 股东（逐股全历史）===")
        fin_start = "20100101"
        perstock_dated = [
            "income", "balancesheet", "cashflow", "fina_indicator",
            "forecast", "express", "dividend", "cyq_perf",
            "stk_holdernumber", "stk_holdertrade",
        ]
        for api in perstock_dated:
            run_perstock(lake, api, uni, fin_start, today, use_dates=True)
        for api in ("stk_managers",):
            run_perstock(lake, api, uni, fin_start, today, use_dates=False)

    if "4" in phases:
        log("=== Phase 4: 指数 / 板块 ===")
        # 指数日线：主要指数（沪深300/500/1000/上证/深证/创业/科创/中证全指等）
        idx_codes = ["000001.SH", "000300.SH", "000905.SH", "000852.SH", "000016.SH",
                     "399001.SZ", "399006.SZ", "000688.SH", "000985.CSI", "399905.SZ"]
        for code in idx_codes:
            u = [{"ts_code": code, "list_date": None, "delist_date": None}]
            run_perstock(lake, "index_daily", u, args.min_start, today, use_dates=True)
        # 申万板块日线
        try:
            sw = pl.read_parquet(root / "reference/index_classify_SW2021.parquet")
            sw_codes = [{"ts_code": c, "list_date": None, "delist_date": None}
                        for c in sw["index_code"].to_list()] if "index_code" in sw.columns else []
            for u in sw_codes:
                run_perstock(lake, "sw_daily", [u], args.min_start, today, use_dates=True)
        except Exception as e:
            log(f"  sw_daily 跳过: {e}")

    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    manifest["status"] = "complete"
    manifest["n_symbols"] = len(uni)
    manifest["n_trade_dates"] = len(dates)
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 收尾 sentinel
    (root / "_DONE").write_text(
        datetime.now().isoformat(timespec="seconds"), encoding="utf-8"
    )
    log("=== 全部阶段结束 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
