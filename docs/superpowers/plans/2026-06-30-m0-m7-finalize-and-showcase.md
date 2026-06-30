# M0 收口 + M7 模拟交易与展示页 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 收口 M0 微观结构(停牌阻断/快照/性能基准入库)+ 建 M7 模拟交易闭环(M4 组合 → 回测净值)+ 成果展示页(HTML dashboard),完成 FactorZen 平台升级全链路。

**Architecture:** M0 是轻收口——修 1 个 GEM 涨停浮点 bug + 提交 4 个已散落的文件(代码已完整)。M7 高复用:模拟交易复用 `backtest.py` 的 `PrecomputedWeightsStrategy` + `run_strategy_backtest`(只加 20-30 行胶水把 M4 `weights.parquet`→`weights_by_date`);展示页复用 `reports/` 的 Jinja2 + matplotlib 图表引擎(新建 `portfolio_dashboard.html` 模板 + 渲染函数)。

**Tech Stack:** polars · `backtest.py`(已有回测引擎) · Jinja2 + matplotlib(reports 引擎) · `fz` CLI。

## Global Constraints

- **测试精简**:每 task 1 个核心行为测试即可,不追求全覆盖(用户要求加快)。M0 复用现成 `test_microstructure.py`(29 测试)。
- **复用优先**:M7 绝不重写回测/报告引擎,只加胶水层 + 新模板。implementer 必须先 Read 现有 `backtest.py` / `reports/_charts.py` 确认精确签名,按实际接口写。
- **提交**:conventional commits;作者 `rookiewu417 <1007372080@qq.com>`。M0 提交它的 4 个散落文件;M7 各 task 只 `git add` 自己的文件,**绝不 `-A`**。
- **环境**:`pixi run pytest` / `pixi run ruff check`。polars 1.41.2。
- 落盘约定仿 M3/M4:`workspace/sim/{run_id}/` 产物 + `manifest.json`(含 `git_sha`)。

---

## File Structure

| 文件 | 职责 | Task |
|---|---|---|
| `src/factorzen/core/universe.py`(改) + `backtest.py`(改) + `core/benchmark.py` + `tests/test_microstructure.py` | M0 散落收口(修 bug + 提交) | 1 |
| `src/factorzen/sim/__init__.py` + `sim/engine.py` | 模拟交易胶水(M4 weights → 回测净值 → 落盘) | 2 |
| `src/factorzen/cli/main.py`(改) | `fz sim run` / `fz sim show` | 3 |
| `src/factorzen/reports/portfolio_report.py` + `reports/templates/portfolio_dashboard.html` | 成果展示页(Jinja2 渲染) | 4 |
| `src/factorzen/cli/main.py`(改) + `README.md` | `fz report portfolio` + README | 5 |

---

## Task 1: M0 收口(修浮点 bug + 提交散落文件)

**Files:**
- Modify: `src/factorzen/daily/evaluation/backtest.py`(GEM 涨停容差)
- Modify: `src/factorzen/core/universe.py`(已有 `get_universe_snapshot`,仅可能需导出)
- Test: `tests/test_microstructure.py`(已存在,29 测试,当前挂 1)

**Interfaces:**
- Produces: M0 微观结构层入库(`get_universe_snapshot`/停牌阻断/`benchmark.py` 工具)

- [ ] **Step 1: 跑现有测试确认 1 个失败**

Run: `pixi run pytest tests/test_microstructure.py -q`
Expected: `28 passed, 1 failed` — `test_gem_limit_up_uses_20pct_threshold`(`AssertionError: 0.5 == 0.0`)

- [ ] **Step 2: 修 GEM 涨停浮点精度 bug**

根因:`open=11.98, pre_close=10.0` → `(11.98/10.0-1.0)*100 = 19.7999...`,与 `effective_limit_up=19.8` 比 `>=` 恰好不触发。在 `backtest.py` 的 `_apply_trade_constraints` 里,涨停判断加浮点容差(找到 `opening_pct >= effective_limit_up` 这一比较,改为带 `- 1e-9` 容差):

```python
# 涨停阻断:浮点容差防止 19.7999... >= 19.8 漏判
if opening_pct >= effective_limit_up - 1e-9:
    return 0.0, "limit_up"
```
(同理若有跌停 `<=` 比较,对称加 `+ 1e-9`。只改比较容差,不动其它逻辑。)

- [ ] **Step 3: 确认 `get_universe_snapshot` 可导入**

确认下游能 `from factorzen.core.universe import get_universe_snapshot`(它已定义在 `universe.py`)。若项目用 `core/__init__.py` 统一导出(检查现有 `get_universe` 是否在 `__init__.py` 导出),则按相同约定补上 `get_universe_snapshot`;若现有代码都直接从 `universe` 模块 import,则无需改 `__init__.py`。

- [ ] **Step 4: 跑全部 M0 测试通过**

Run: `pixi run pytest tests/test_microstructure.py -q`
Expected: `29 passed`

- [ ] **Step 5: ruff + 提交 M0 4 文件**

```bash
pixi run ruff check src/factorzen/core/benchmark.py src/factorzen/core/universe.py src/factorzen/daily/evaluation/backtest.py tests/test_microstructure.py
git add src/factorzen/core/benchmark.py src/factorzen/core/universe.py src/factorzen/daily/evaluation/backtest.py tests/test_microstructure.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(microstructure): M0 收口(停牌阻断+快照+性能基准入库, GEM涨停容差修复)"
```

---

## Task 2: 模拟交易引擎(sim/engine.py)

**Files:**
- Create: `src/factorzen/sim/__init__.py`(空), `src/factorzen/sim/engine.py`
- Test: `tests/test_sim_engine.py`

**Interfaces:**
- Consumes: `backtest.py` 的 `PrecomputedWeightsStrategy(weights_by_date: dict[date, pl.DataFrame[ts_code, target_weight]])` + `run_strategy_backtest(...)` → `StrategyBacktestResult`(`.summary_stats` 含 `ann_ret`/`ann_vol`/`sharpe`/`max_dd`/`avg_turnover`/`ann_turnover`/`total_cost`;`.nav`/`.returns`/`.positions`/`.trades`)
- Produces: `run_portfolio_simulation(portfolio_run_dirs, daily, *, out_dir="workspace/sim", run_id=None) -> dict`(落 `nav.parquet`/`metrics.json`/`manifest.json`,返回 `{run_dir, sharpe, max_dd, ann_ret}`)

- [ ] **Step 1: 先读 backtest.py 确认精确签名**

Read `src/factorzen/daily/evaluation/backtest.py`:确认 `PrecomputedWeightsStrategy.__init__` 入参、`run_strategy_backtest(strategy, price_df, start, end, ...)` 的完整签名与返回 `StrategyBacktestResult` 的字段/`summary_stats` 键名。**按实际签名写**(下面骨架以探索结论为准,若不符以实际为准)。

- [ ] **Step 2: 写核心 smoke 测试**

```python
# tests/test_sim_engine.py
from datetime import date
from pathlib import Path
import json
import numpy as np
import polars as pl

from factorzen.sim.engine import run_portfolio_simulation


def _write_portfolio_dir(tmp_path, run_id, codes, weights, sig_date):
    d = tmp_path / run_id
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"ts_code": codes, "target_weight": weights,
                  "prev_weight": [0.0] * len(codes)}).write_parquet(d / "weights.parquet")
    (d / "manifest.json").write_text(json.dumps({"run_id": run_id, "signal_date": sig_date}))
    return str(d)


def _fake_daily(codes, start="20230101", end="20230228"):
    # 构造 2 只股票的日线(含 pct_chg),区间覆盖信号日
    dates = pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 2, 28), "1d", eager=True)
    rng = np.random.default_rng(0)
    rows = []
    for c in codes:
        for dt in dates:
            rows.append({"trade_date": dt, "ts_code": c, "open": 10.0, "high": 10.5,
                         "low": 9.5, "close": 10.0, "pre_close": 10.0, "change": 0.0,
                         "pct_chg": float(rng.normal(0, 1)), "vol": 1e6, "amount": 1e7})
    return pl.DataFrame(rows)


def test_run_portfolio_simulation_produces_metrics(tmp_path: Path):
    codes = ["000001.SZ", "000002.SZ"]
    p1 = _write_portfolio_dir(tmp_path, "p1", codes, [0.5, 0.5], "2023-01-10")
    daily = _fake_daily(codes)
    res = run_portfolio_simulation([p1], daily, out_dir=str(tmp_path / "sim"), run_id="s1")
    run_dir = Path(res["run_dir"])
    assert (run_dir / "nav.parquet").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "manifest.json").exists()
    m = json.loads((run_dir / "metrics.json").read_text())
    # 绩效指标齐全(键名以 backtest summary_stats 为准)
    for k in ["ann_ret", "sharpe", "max_dd"]:
        assert k in m
    assert "sharpe" in res
```

- [ ] **Step 3: 实现 engine.py**(骨架,以 Step 1 确认的真实签名为准)

```python
# src/factorzen/sim/engine.py
"""模拟交易闭环:M4 目标组合 → backtest 回测 → 净值/绩效落盘。"""
from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

import polars as pl

from factorzen.daily.evaluation.backtest import (
    PrecomputedWeightsStrategy,
    run_strategy_backtest,
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _load_weights_by_date(portfolio_run_dirs: list[str]) -> dict[date, pl.DataFrame]:
    """各 run_dir 的 weights.parquet → {signal_date: DataFrame[ts_code, target_weight]}。"""
    out: dict[date, pl.DataFrame] = {}
    for rd in portfolio_run_dirs:
        rd_p = Path(rd)
        manifest = json.loads((rd_p / "manifest.json").read_text())
        sig = manifest.get("signal_date")
        if sig is None:
            continue
        sig_date = date.fromisoformat(sig)
        w = pl.read_parquet(rd_p / "weights.parquet").select(["ts_code", "target_weight"])
        out[sig_date] = w
    return out


def run_portfolio_simulation(portfolio_run_dirs, daily: pl.DataFrame, *,
                             out_dir="workspace/sim", run_id=None) -> dict:
    weights_by_date = _load_weights_by_date(portfolio_run_dirs)
    if not weights_by_date:
        raise ValueError("no portfolio weights with signal_date found")
    start = min(weights_by_date).strftime("%Y%m%d")
    end = daily["trade_date"].max()
    end = end.strftime("%Y%m%d") if hasattr(end, "strftime") else str(end)

    strategy = PrecomputedWeightsStrategy(weights_by_date)
    bt = run_strategy_backtest(strategy, daily, start, end)  # 签名以 Step1 为准

    rid = run_id or "sim"
    run_dir = Path(out_dir) / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    bt.nav.write_parquet(run_dir / "nav.parquet")
    metrics = dict(bt.summary_stats)  # ann_ret/ann_vol/sharpe/max_dd/avg_turnover/...
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    (run_dir / "manifest.json").write_text(json.dumps(
        {"run_id": rid, "n_signals": len(weights_by_date), "start": start, "end": end,
         "git_sha": _git_sha()}, ensure_ascii=False, indent=2))
    return {"run_dir": str(run_dir), "sharpe": metrics.get("sharpe"),
            "max_dd": metrics.get("max_dd"), "ann_ret": metrics.get("ann_ret")}
```

- [ ] **Step 4: 跑测试通过 + ruff + 提交**

```bash
pixi run pytest tests/test_sim_engine.py -v
pixi run ruff check src/factorzen/sim/ tests/test_sim_engine.py
git add src/factorzen/sim/__init__.py src/factorzen/sim/engine.py tests/test_sim_engine.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(sim): 模拟交易闭环(M4 组合→backtest 回测→净值/绩效落盘)"
```

---

## Task 3: CLI `fz sim run` / `fz sim show`

**Files:**
- Modify: `src/factorzen/cli/main.py`
- Test: `tests/test_sim_cli.py`

**Interfaces:**
- Consumes: `run_portfolio_simulation`(Task 2);`loader.fetch_daily`;仿 `_cmd_risk_build` 数据加载

- [ ] **Step 1: 写 parser 测试**

```python
# tests/test_sim_cli.py
def test_parser_has_sim_run():
    from factorzen.cli.main import build_parser
    p = build_parser()
    args = p.parse_args(["sim", "run", "--portfolio-dir", "workspace/portfolios",
                         "--start", "20230101", "--end", "20241231"])
    assert args.command == "sim"
    assert args.sim_command == "run"
    assert callable(args.func)
```

- [ ] **Step 2: 接入 CLI**(仿 `fz risk build` 注册 + handler 延迟 import)

`build_parser()` 加顶层 `sim` 组:`sim run`(`--portfolio-dir`(可多值或目录) `--start` `--end` `--run-id`)+ `sim show`(`--sim-dir`)。handler `_cmd_sim_run`:`loader.fetch_daily(start,end)` → 收集 `portfolio-dir` 下各 `{run_id}/` → `run_portfolio_simulation(dirs, daily, run_id=...)` → print 绩效摘要。`_cmd_sim_show` 读 `metrics.json` 打印。

- [ ] **Step 3: 跑测试通过 + 提交**

```bash
pixi run pytest tests/test_sim_cli.py -v
pixi run ruff check src/factorzen/cli/main.py tests/test_sim_cli.py
git add src/factorzen/cli/main.py tests/test_sim_cli.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(sim): fz sim run/show CLI"
```

---

## Task 4: 成果展示页(portfolio_report.py + dashboard 模板)

**Files:**
- Create: `src/factorzen/reports/portfolio_report.py`, `src/factorzen/reports/templates/portfolio_dashboard.html`
- Test: `tests/test_portfolio_report.py`

**Interfaces:**
- Consumes: `reports/_charts.py` 的图表函数(`_make_returns_chart`/`_make_monthly_return_heatmap`/`_make_attribution_chart`,base64 PNG);Jinja2 env(`reports/` 现有 `FileSystemLoader` + `metric_number`/`metric_percent` 过滤器);M7 sim 的 `StrategyBacktestResult`/metrics + M4 `attribution.csv`/`risk_summary.csv`/`manifest.json`
- Produces: `generate_portfolio_report(sim_result, *, metrics, attribution_df=None, risk_summary_df=None, portfolio_manifest=None) -> str`(HTML 字符串)

- [ ] **Step 1: 先读 reports 现有引擎确认接口**

Read `src/factorzen/reports/tear_sheet.py`(Jinja2 env 怎么建/过滤器名)、`reports/_charts.py`(`_make_returns_chart` 等签名 + `_fig_to_base64`)、`reports/templates/tear_sheet.html`(CSS 骨架)。**按实际签名写**。

- [ ] **Step 2: 写渲染测试**

```python
# tests/test_portfolio_report.py
def test_generate_portfolio_report_html_has_sections(tmp_path):
    import polars as pl
    from factorzen.reports.portfolio_report import generate_portfolio_report
    metrics = {"ann_ret": 0.12, "ann_vol": 0.18, "sharpe": 0.67, "max_dd": -0.15,
               "ann_turnover": 3.2, "total_cost": 0.01}
    attribution = pl.DataFrame({"type": ["brinson_allocation", "factor_return"],
                                "key": ["银行", "size"], "value": [0.01, 0.005]})
    risk = pl.DataFrame({"metric": ["total_risk", "factor_risk", "specific_risk"],
                         "value": [0.18, 0.15, 0.10]})
    html = generate_portfolio_report(None, metrics=metrics, attribution_df=attribution,
                                     risk_summary_df=risk,
                                     portfolio_manifest={"n_holdings": 87, "status": "optimal"})
    assert isinstance(html, str) and len(html) > 500
    # 关键 section 存在
    assert "sharpe" in html.lower() or "夏普" in html
    assert "0.67" in html or "67" in html       # 绩效数值渲染进去
    assert "总风险" in html or "total_risk" in html or "0.18" in html
```

- [ ] **Step 3: 实现 portfolio_report.py + 模板**

`portfolio_report.py`:用现有 Jinja2 env 加载 `portfolio_dashboard.html`,渲染 context(绩效卡 metrics / 风险卡 risk_summary / 归因表 attribution / manifest meta)。若传了 `sim_result`(StrategyBacktestResult),用 `_make_returns_chart`/`_make_monthly_return_heatmap` 生成 base64 图嵌入;无则跳过图(测试不依赖真实 bt)。`portfolio_dashboard.html` copy `tear_sheet.html` 的 `<style>` CSS 骨架,sections:综合绩效卡 → 净值曲线(有 sim 时)→ 风险归因 → 当前持仓 meta → M1-M6 模块状态卡(静态超链接占位)。条件渲染:`portfolio_manifest.return_attribution_available` 为 false 时归因区标注"建仓时点占位"。

- [ ] **Step 4: 跑测试通过 + 提交**

```bash
pixi run pytest tests/test_portfolio_report.py -v
pixi run ruff check src/factorzen/reports/portfolio_report.py tests/test_portfolio_report.py
git add src/factorzen/reports/portfolio_report.py src/factorzen/reports/templates/portfolio_dashboard.html tests/test_portfolio_report.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(reports): 组合成果展示页(复用 Jinja2+matplotlib 引擎)"
```

---

## Task 5: CLI `fz report portfolio` + README

**Files:**
- Modify: `src/factorzen/cli/main.py`, `README.md`
- Test: `tests/test_report_cli.py`

**Interfaces:**
- Consumes: `generate_portfolio_report`(Task 4);sim 落盘产物

- [ ] **Step 1: 写 parser 测试**

```python
# tests/test_report_cli.py
def test_parser_has_report_portfolio():
    from factorzen.cli.main import build_parser
    p = build_parser()
    args = p.parse_args(["report", "portfolio", "--sim-dir", "workspace/sim/s1",
                         "--portfolio-dir", "workspace/portfolios/p1"])
    assert args.command == "report"
    assert args.report_command == "portfolio"
    assert callable(args.func)
```

- [ ] **Step 2: 接入 CLI**

`build_parser()` 加 `report` 组(或扩展现有)的 `portfolio` 子命令:`--sim-dir`(读 metrics.json/nav.parquet)+ `--portfolio-dir`(读 attribution.csv/risk_summary.csv/manifest.json)+ `--out`(HTML 输出路径,默认 `workspace/reports/portfolio_<ts>.html`)。handler `_cmd_report_portfolio`:读产物 → `generate_portfolio_report(...)` → 写 HTML → print 路径。

- [ ] **Step 3: 跑测试通过**

Run: `pixi run pytest tests/test_report_cli.py -v` → PASS

- [ ] **Step 4: README + 提交**

README「核心能力」表补两行:模拟交易(`fz sim run` M4 组合→回测净值/绩效)、成果展示页(`fz report portfolio` HTML dashboard)。
```bash
git add src/factorzen/cli/main.py tests/test_report_cli.py README.md
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(reports): fz report portfolio CLI + README; M0/M7 收官"
```

---

## 收尾验收(全部 task 完成后)

- [ ] `pixi run pytest tests/test_microstructure.py tests/test_sim_engine.py tests/test_sim_cli.py tests/test_portfolio_report.py tests/test_report_cli.py -q` 全绿
- [ ] M0 4 文件入库,工作区不再有散落未提交的 M0 改动
- [ ] (可选)真实数据 smoke:`fz portfolio build` → `fz sim run` → `fz report portfolio` 全链路产 HTML
- [ ] `git status --short` 干净
- [ ] 更新 memory roadmap(M0/M7 完成,M0-M7 全部收官)

---

*M0/M7 完成后,FactorZen 实现「因子挖掘 → 防过拟合 → 风险模型 → 智能挖掘 → 组合优化 → 模拟交易 → 成果展示」完整买方级研究平台闭环。*
