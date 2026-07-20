"""test_combine_from_library.py：combine from-library：因子库选品 → 物化 → 四方法 OOS。
test_combine_cli_smoke.py：fz combine run CLI 冒烟。
test_library_provider.py：registry library provider：load_library_factors 注入 expression 型（Batch 2）。
"""


from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from factorzen.cli.main import main
from factorzen.discovery.library_provider import load_library_factors
from factorzen.pipelines import factor_combine

# ==== 来自 test_combine_from_library.py ====

def _daily(n_stocks=40, n_days=200, seed=1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2023, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for i in range(n_stocks):
        c, px = f"{i:06d}.SZ", 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px, "close_adj": px,
                "open": px * 0.99, "high": px * 1.01, "low": px * 0.98,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
            })
    return pl.DataFrame(rows)


def _write_lib__combine_from_library(root: Path, market: str, records: list[dict]) -> None:
    path = root / f"{market}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


def _expr_rec(
    expression: str,
    *,
    name: str | None = None,
    status: str = "active",
    ic_train: float = 0.05,
    **extra,
) -> dict:
    d = {
        "expression": expression,
        "market": "ashare",
        "status": status,
        "kind": "expression",
        "ic_train": ic_train,
        "name": name,
    }
    d.update(extra)
    return d


def test_combine_from_library_end_to_end(tmp_path, monkeypatch):
    """3 条 expression active → 跑通；factors_used 是 name；manifest/返回字段齐全。"""
    # 先 import 再 patch，避免 string-target 首次导入陷阱
    import factorzen.pipelines.factor_mine as fm

    monkeypatch.setattr(fm, "prepare_mining_daily", lambda *a, **k: _daily())

    lib = tmp_path / "lib"
    _write_lib__combine_from_library(lib, "ashare", [
        _expr_rec("rank(close)", name="f_close", ic_train=0.08),
        _expr_rec("ts_mean(vol,5)", name="f_vol", ic_train=0.06),
        _expr_rec("neg(rank(ts_std(close,10)))", name="f_vol_neg", ic_train=0.04),
    ])
    res = factor_combine.combine_from_library(
        market="ashare",
        library_root=str(lib),
        start="20230103",
        end="20231231",
        universe=None,
        horizon=5,
        train_days=60,
        test_days=15,
        decorr_threshold=1.0,
        out_dir=str(tmp_path / "out"),
    )
    comp = res["comparison"]
    methods = set(comp["method"].to_list())
    assert {"equal_weight", "ic_weighted", "max_ir"} <= methods
    assert comp.height >= 3
    # 可读 name，不是 factor_{i} / 表达式原文
    assert set(res["factors_used"]) == {"f_close", "f_vol", "f_vol_neg"}
    assert res["factors_status"] == {
        "f_close": "active", "f_vol": "active", "f_vol_neg": "active",
    }
    assert res["skipped_materialize"] == []
    assert res["dropped_correlated"] == []
    assert res["market"] == "ashare"
    assert res["statuses"] == ["active"]
    assert res["library_hash"] is not None
    assert "run_dir" in res
    assert res.get("truncated_from") is None


def test_combine_from_library_statuses_filter(tmp_path, monkeypatch):
    """probation 默认不入选；statuses 含 probation 则入选。"""
    import factorzen.pipelines.factor_mine as fm

    monkeypatch.setattr(fm, "prepare_mining_daily", lambda *a, **k: _daily())

    lib = tmp_path / "lib"
    _write_lib__combine_from_library(lib, "ashare", [
        _expr_rec("rank(close)", name="a_active", status="active", ic_train=0.08),
        _expr_rec("rank(vol)", name="a_prob", status="probation", ic_train=0.07),
        _expr_rec("rank(high)", name="b_active", status="active", ic_train=0.05),
    ])
    # 默认 statuses=active → 只有 2 条 active；应能跑
    res = factor_combine.combine_from_library(
        market="ashare", library_root=str(lib),
        start="20230103", end="20231231",
        train_days=60, test_days=15, decorr_threshold=1.0,
        out_dir=str(tmp_path / "o1"),
    )
    assert "a_prob" not in res["factors_used"]
    assert set(res["factors_used"]) == {"a_active", "b_active"}

    # 含 probation → 3 条入选
    res2 = factor_combine.combine_from_library(
        market="ashare", library_root=str(lib),
        statuses=("active", "probation"),
        start="20230103", end="20231231",
        train_days=60, test_days=15, decorr_threshold=1.0,
        out_dir=str(tmp_path / "o2"),
    )
    assert "a_prob" in res2["factors_used"]
    assert len(res2["factors_used"]) == 3


def test_combine_from_library_python_with_expression(tmp_path, monkeypatch):
    """python 面板注入 + expression 同台进组合；universe=None + python → ValueError。"""
    import factorzen.discovery.factor_library as fl
    import factorzen.pipelines.factor_mine as fm

    monkeypatch.setattr(fm, "prepare_mining_daily", lambda *a, **k: _daily())

    daily = _daily()
    # 与网格同口径的假 python 面板
    py_panel = daily.select([
        "trade_date", "ts_code",
        (pl.col("close_adj") * 0.01).alias("factor_value"),
    ])

    def _fake_mat(r, df, *, market, universe, python_materializer, start, end):
        # 只服务我们的 fake_py；其它走原路径（本测无）
        if (r.name or "") == "fake_py" or fl.is_python_identity(r.expression):
            return (
                df.select(["trade_date", "ts_code"])
                .join(py_panel, on=["trade_date", "ts_code"], how="inner")
            )
        return None

    monkeypatch.setattr(fl, "_materialize_python_on_grid", _fake_mat)

    lib = tmp_path / "lib"
    py_key = fl.python_identity("fake_py")
    _write_lib__combine_from_library(lib, "ashare", [
        _expr_rec("rank(close)", name="e1", ic_train=0.08),
        {
            "expression": py_key, "market": "ashare", "status": "active",
            "kind": "python", "name": "fake_py", "impl": "fake_py",
            "ic_train": 0.06,
        },
        _expr_rec("ts_mean(vol,5)", name="e2", ic_train=0.04),
    ])

    res = factor_combine.combine_from_library(
        market="ashare", library_root=str(lib),
        start="20230103", end="20231231", universe="csi300",
        train_days=60, test_days=15, decorr_threshold=1.0,
        out_dir=str(tmp_path / "o"),
    )
    assert "fake_py" in res["factors_used"]
    assert "e1" in res["factors_used"] and "e2" in res["factors_used"]

    with pytest.raises(ValueError, match=r"universe|python"):
        factor_combine.combine_from_library(
            market="ashare", library_root=str(lib),
            start="20230103", end="20231231", universe=None,
            train_days=60, test_days=15, out_dir=str(tmp_path / "o2"),
        )


def test_combine_from_library_skip_bad_expression(tmp_path, monkeypatch):
    """坏表达式跳过并记入 skipped_materialize；剩余 ≥2 仍跑。"""
    import factorzen.pipelines.factor_mine as fm

    monkeypatch.setattr(fm, "prepare_mining_daily", lambda *a, **k: _daily())

    lib = tmp_path / "lib"
    bad = "this_is_not_a_valid_expr_zzz()"
    _write_lib__combine_from_library(lib, "ashare", [
        _expr_rec("rank(close)", name="ok1", ic_train=0.08),
        _expr_rec(bad, name="bad_one", ic_train=0.07),
        _expr_rec("ts_mean(vol,5)", name="ok2", ic_train=0.05),
    ])
    res = factor_combine.combine_from_library(
        market="ashare", library_root=str(lib),
        start="20230103", end="20231231",
        train_days=60, test_days=15, decorr_threshold=1.0,
        out_dir=str(tmp_path / "o"),
    )
    assert bad in res["skipped_materialize"]
    assert "bad_one" not in res["factors_used"]
    assert set(res["factors_used"]) == {"ok1", "ok2"}


def test_combine_from_library_needs_two_and_top_n(tmp_path, monkeypatch):
    """<2 记录 → ValueError；top_n 截断记 truncated_from。"""
    import factorzen.pipelines.factor_mine as fm

    monkeypatch.setattr(fm, "prepare_mining_daily", lambda *a, **k: _daily())

    lib = tmp_path / "lib"
    _write_lib__combine_from_library(lib, "ashare", [
        _expr_rec("rank(close)", name="only_one", ic_train=0.08),
    ])
    with pytest.raises(ValueError, match="不足 2 个"):
        factor_combine.combine_from_library(
            market="ashare", library_root=str(lib),
            start="20230103", end="20231231",
            out_dir=str(tmp_path / "o"),
        )

    _write_lib__combine_from_library(lib, "ashare", [
        _expr_rec("rank(close)", name="n1", ic_train=0.09),
        _expr_rec("rank(vol)", name="n2", ic_train=0.07),
        _expr_rec("rank(high)", name="n3", ic_train=0.05),
        _expr_rec("ts_mean(vol,5)", name="n4", ic_train=0.03),
    ])
    res = factor_combine.combine_from_library(
        market="ashare", library_root=str(lib),
        start="20230103", end="20231231",
        top_n=2, train_days=60, test_days=15, decorr_threshold=1.0,
        out_dir=str(tmp_path / "o2"),
    )
    assert res["truncated_from"] == 4
    assert len(res["factors_used"]) == 2
    # |ic_train| 降序：n1, n2
    assert res["factors_used"] == ["n1", "n2"]


def test_combine_from_library_cli_parser_smoke():
    """CLI 参数解析冒烟：不炸、statuses 逗号解析正确。"""
    from factorzen.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "combine", "from-library",
        "--start", "20230103",
        "--end", "20231231",
        "--market", "ashare",
        "--statuses", "active,probation",
        "--top-n", "10",
        "--universe", "csi300",
        "--library-root", "/tmp/lib",
    ])
    assert args.combine_command == "from-library"
    assert args.market == "ashare"
    assert args.statuses == ("active", "probation")
    assert args.top_n == 10
    assert args.universe == "csi300"
    assert args.library_root == "/tmp/lib"
    assert args.start == "20230103"
    assert callable(args.func)

    # 非法 status
    with pytest.raises(SystemExit):
        parser.parse_args([
            "combine", "from-library",
            "--start", "20230103", "--end", "20231231",
            "--statuses", "active,bogus",
        ])


def test_manifest_records_full_provenance(tmp_path, monkeypatch):
    """manifest 必须记全窗口/票池/选品参数——否则事后无法判断一次 run 覆盖了什么。

    **2026-07-19 实际吃亏**：追查一个疑似数据污染时，需要判断历史 combine run 的
    窗口是否覆盖脏日，却发现 manifest 只有 ``command=['combine','from-library']``、
    ``config={'seed': 0}``，start/end/universe 全无——只能去读 combined parquet
    的 trade_date 反推。CLAUDE.md 要求「manifest 记全命令/窗口，漏了=假复现」。

    （反推的结论是没被污染，但那是运气；缺这些字段本身就让 manifest 失去复现价值。）
    """
    import factorzen.pipelines.factor_mine as fm

    monkeypatch.setattr(fm, "prepare_mining_daily", lambda *a, **k: _daily())

    lib = tmp_path / "lib"
    _write_lib__combine_from_library(lib, "ashare", [
        _expr_rec("rank(close)", name="f_close", ic_train=0.08),
        _expr_rec("ts_mean(vol,5)", name="f_vol", ic_train=0.06),
    ])
    res = factor_combine.combine_from_library(
        market="ashare",
        library_root=str(lib),
        start="20230103",
        end="20231231",
        universe="csi300",
        horizon=5,
        train_days=60,
        test_days=15,
        decorr_threshold=1.0,
        methods=["equal_weight"],
        out_dir=str(tmp_path / "out"),
    )
    manifest = json.loads((Path(res["run_dir"]) / "manifest.json").read_text())
    cfg = manifest.get("config") or {}

    # 窗口与票池：判断「这次 run 覆盖了哪段数据」的最小充分集
    assert cfg.get("start") == "20230103", cfg
    assert cfg.get("end") == "20231231", cfg
    assert cfg.get("universe") == "csi300", cfg
    assert cfg.get("market") == "ashare", cfg
    assert cfg.get("horizon") == 5, cfg
    # 选品参数：决定纳入哪些因子
    assert cfg.get("statuses") == ["active"], cfg
    assert cfg.get("decorr_threshold") == 1.0, cfg
    assert cfg.get("seed") is not None, cfg
    # 库指纹：同窗口不同库版本结果不同
    assert cfg.get("library_hash") is not None, cfg


# ==== 来自 test_combine_cli_smoke.py ====

def _write_inputs(tmp_path, n_days=120, n_stocks=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = [f"2025{1 + i // 28:02d}{1 + i % 28:02d}" for i in range(n_days)]
    ra, rb, rr = [], [], []
    for d in dates:
        fa = rng.standard_normal(n_stocks)
        fb = rng.standard_normal(n_stocks)
        ret = 0.8 * fa - 0.4 * fb + rng.standard_normal(n_stocks) * 0.3
        for s in range(n_stocks):
            c = f"{s:04d}.SZ"
            ra.append({"trade_date": d, "ts_code": c, "factor_value": float(fa[s])})
            rb.append({"trade_date": d, "ts_code": c, "factor_value": float(fb[s])})
            rr.append({"trade_date": d, "ts_code": c, "ret": float(ret[s])})
    fa_p = tmp_path / "fa.parquet"
    fb_p = tmp_path / "fb.parquet"
    ret_p = tmp_path / "ret.parquet"
    pl.DataFrame(ra).write_parquet(fa_p)
    pl.DataFrame(rb).write_parquet(fb_p)
    pl.DataFrame(rr).write_parquet(ret_p)
    return fa_p, fb_p, ret_p


def test_fz_combine_run_smoke(tmp_path):
    fa_p, fb_p, ret_p = _write_inputs(tmp_path)
    out = tmp_path / "out"
    rc = main(
        [
            "combine", "run",
            "--factor", str(fa_p),
            "--factor", str(fb_p),
            "--ret", str(ret_p),
            "--train-days", "60",
            "--test-days", "20",
            "--purge-days", "5",
            "--methods", "equal_weight,lgbm",
            "--seed", "0",
            "--run-id", "cli1",
            "--out-dir", str(out),
        ]
    )
    assert rc == 0
    run_dir = out / "cli1"
    assert (run_dir / "comparison.csv").exists()
    assert (run_dir / "report.md").exists()
    comp = pl.read_csv(run_dir / "comparison.csv")
    assert set(comp["method"].to_list()) == {"equal_weight", "lgbm"}


# ==== 来自 test_library_provider.py ====

def _write_lib__library_provider(root: Path, market: str, records: list[dict]) -> None:
    path = root / f"{market}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


def _default_name(expr: str) -> str:
    return f"mined_{hashlib.sha1(expr.encode()).hexdigest()[:8]}"


@pytest.fixture
def reg_mod():
    """daily registry 模块；teardown reset 全局单例——LibFactor 注入若滞留会
    污染同进程后续测试文件（全量跑实锤过 test_daily_factors 次序失败）。"""
    import factorzen.daily.factors.registry as reg

    yield reg
    reg._registry.reset()


# ── 1. 基本注入 ──────────────────────────────────────────────────────────────


def test_load_library_factors_registers_expression_records(tmp_path, reg_mod):
    named = "lib_prov_named_alpha"
    expr_named = "rank(close)"
    expr_anon = "neg(rank(close))"
    anon_name = _default_name(expr_anon)
    _write_lib__library_provider(
        tmp_path,
        "ashare",
        [
            {
                "expression": expr_named,
                "market": "ashare",
                "kind": "expression",
                "name": named,
                "status": "active",
            },
            {
                "expression": expr_anon,
                "market": "ashare",
                "kind": "expression",
                "status": "probation",
                # name 缺省 → default_name_for_expression
            },
        ],
    )
    n = load_library_factors(market="ashare", root=str(tmp_path))
    assert n == 2

    cls_named = reg_mod.get_factor(named)
    inst = cls_named()
    assert inst.name == named
    assert inst.expression == expr_named
    assert inst.lookback_days >= 60
    assert "[active]" in inst.description

    cls_anon = reg_mod.get_factor(anon_name)
    inst_anon = cls_anon()
    assert inst_anon.expression == expr_anon
    assert "[probation]" in inst_anon.description


# ── 2. 冲突让位 ──────────────────────────────────────────────────────────────


def test_load_library_factors_yields_to_existing(tmp_path, reg_mod, caplog):
    from factorzen.daily.factors.base import DailyFactor

    conflict = "lib_prov_conflict_builtin"
    # 先注册同名假因子（模拟 workspace/builtin 占用）
    fake = type(
        "FakeConflict",
        (DailyFactor,),
        {
            "name": conflict,
            "frequency": "daily",
            "description": "fake occupant",
            "lookback_days": 20,
            "compute": lambda self, ctx: None,
        },
    )
    assert reg_mod._registry.register(fake, override=True) is True

    _write_lib__library_provider(
        tmp_path,
        "ashare",
        [
            {
                "expression": "rank(vol)",
                "market": "ashare",
                "kind": "expression",
                "name": conflict,
                "status": "active",
            }
        ],
    )
    with caplog.at_level("WARNING"):
        n = load_library_factors(market="ashare", root=str(tmp_path))
    assert n == 0
    assert any("让位" in r.message or conflict in r.message for r in caplog.records)
    # 仍是假因子，非 LibFactor
    assert reg_mod.get_factor(conflict) is fake


# ── 3. python 型跳过 ─────────────────────────────────────────────────────────


def test_load_library_factors_skips_python_kind(tmp_path, reg_mod):
    py_name = "lib_prov_python_skip_xyz"
    _write_lib__library_provider(
        tmp_path,
        "ashare",
        [
            {
                "expression": f"py::{py_name}",
                "market": "ashare",
                "kind": "python",
                "name": py_name,
                "impl": py_name,
                "status": "active",
            },
            {
                "expression": "rank(amount)",
                "market": "ashare",
                "kind": "expression",
                "name": "lib_prov_expr_ok",
                "status": "active",
            },
        ],
    )
    n = load_library_factors(market="ashare", root=str(tmp_path))
    assert n == 1
    reg_mod.get_factor("lib_prov_expr_ok")
    with pytest.raises(KeyError):
        reg_mod.get_factor(py_name)


# ── 4. 幂等 ──────────────────────────────────────────────────────────────────


def test_load_library_factors_idempotent(tmp_path, reg_mod, caplog):
    name = "lib_prov_idempotent_once"
    _write_lib__library_provider(
        tmp_path,
        "ashare",
        [
            {
                "expression": "rank(high)",
                "market": "ashare",
                "kind": "expression",
                "name": name,
                "status": "active",
            }
        ],
    )
    n1 = load_library_factors(market="ashare", root=str(tmp_path))
    assert n1 == 1
    names_after_1 = reg_mod.list_factors()
    assert names_after_1.count(name) == 1

    with caplog.at_level("WARNING"):
        n2 = load_library_factors(market="ashare", root=str(tmp_path))
    assert n2 == 0
    names_after_2 = reg_mod.list_factors()
    assert names_after_2.count(name) == 1
    # 二次 load 不因自身已注入再刷「让位」warning
    assert not any("让位" in r.message and name in r.message for r in caplog.records)


# ── 5. 损坏库文件 ────────────────────────────────────────────────────────────


def test_load_library_factors_tolerates_corrupt_jsonl(tmp_path, reg_mod):
    path = tmp_path / "ashare.jsonl"
    path.write_text(
        '{"expression":"rank(low)","market":"ashare","kind":"expression","name":"lib_prov_ok_corrupt","status":"active"}\n'
        "NOT_JSON_LINE\n"
        '{"expression":"neg(rank(low))","market":"ashare","kind":"expression","name":"lib_prov_ok2_corrupt","status":"correlated"}\n',
        encoding="utf-8",
    )
    n = load_library_factors(market="ashare", root=str(tmp_path))
    assert n == 2
    reg_mod.get_factor("lib_prov_ok_corrupt")
    reg_mod.get_factor("lib_prov_ok2_corrupt")


# ── 6. CLI 冒烟 ──────────────────────────────────────────────────────────────


def test_cmd_factor_list_includes_library_factor(tmp_path, reg_mod, monkeypatch, capsys):
    import argparse

    from factorzen.cli import main as cli

    name = "lib_prov_cli_list_visible"
    _write_lib__library_provider(
        tmp_path,
        "ashare",
        [
            {
                "expression": "rank(open)",
                "market": "ashare",
                "kind": "expression",
                "name": name,
                "status": "active",
            }
        ],
    )
    # 把默认库根指到 tmp（load_library_factors 无 root 参数时用 DEFAULT_ROOT）
    monkeypatch.setattr(
        "factorzen.discovery.factor_library.DEFAULT_ROOT",
        str(tmp_path),
    )
    # 同时 patch daily registry 里 load 用的默认（若已 import DEFAULT_ROOT 为值则走函数内再 import）
    args = argparse.Namespace(freq="daily")
    rc = cli._cmd_factor_list(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert name in out
