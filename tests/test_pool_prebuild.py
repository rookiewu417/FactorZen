"""库池 parquet 磁盘交接 + cache_dir 装载校验（全离线）。"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tests.daily.test_library_pool_cache import _mk_daily, _seed_lib, _write_lib

# ── helpers ─────────────────────────────────────────────────────────────────


def _assert_pool_wide_equal(a: pl.DataFrame, b: pl.DataFrame) -> None:
    """wide 逐值相等;ts_code dtype 按读回契约对齐(小帧 Utf8)。"""
    left = a
    right = b
    if left.schema.get("ts_code") != right.schema.get("ts_code"):
        # 对齐到 Utf8 再比(原池可能 Categorical/Utf8,读回小帧 Utf8)
        left = left.with_columns(pl.col("ts_code").cast(pl.Utf8))
        right = right.with_columns(pl.col("ts_code").cast(pl.Utf8))
    assert_frame_equal(left, right, check_dtypes=True)


def _hand_meta(
    *,
    market: str,
    root: str,
    statuses,
    daily: pl.DataFrame,
    eval_start=None,
) -> dict:
    from factorzen.discovery.factor_library import library_file_hash

    return {
        "market": market,
        "statuses": list(statuses),
        "eval_start": str(eval_start) if eval_start is not None else None,
        "library_hash": library_file_hash(market, root),
        "prepped_height": daily.height,
        "prepped_date_min": str(daily["trade_date"].min()),
        "prepped_date_max": str(daily["trade_date"].max()),
        "data_window": None,
        "git_sha": None,
        "created_at": "test",
    }


# ── 1. round-trip f64 ───────────────────────────────────────────────────────


def test_parquet_roundtrip_f64(tmp_path):
    from factorzen.discovery.factor_library import CompactLibraryPool, build_library_pool

    daily = _mk_daily(n_days=30, n_stocks=8)
    root = str(_seed_lib(tmp_path / "lib"))
    pool = build_library_pool("ashare", daily, root=root, compact=True)
    assert isinstance(pool, CompactLibraryPool)
    assert len(pool) > 0

    path = tmp_path / "pool_wide.parquet"
    pool.write_parquet(path)
    loaded = CompactLibraryPool.from_parquet(path, list(pool.factor_names))

    assert loaded.factor_names == pool.factor_names
    # 小帧读回 ts_code 为 Utf8
    assert loaded.wide.schema["ts_code"] == pl.Utf8
    _assert_pool_wide_equal(
        pool.wide.with_columns(pl.col("ts_code").cast(pl.Utf8)),
        loaded.wide,
    )
    name = pool.factor_names[0]
    assert_frame_equal(
        pool[name].with_columns(pl.col("ts_code").cast(pl.Utf8)),
        loaded[name],
        check_dtypes=True,
    )


# ── 2. round-trip f32 ───────────────────────────────────────────────────────


def test_parquet_roundtrip_f32(tmp_path, monkeypatch):
    # factor_library 从 pool 名绑定导入;patch 生效点是 factor_library 模块内名字
    import factorzen.discovery.factor_library as fl

    monkeypatch.setattr(fl, "POOL_VALUE_F32_BYTES_THRESHOLD", 1)

    daily = _mk_daily(n_days=30, n_stocks=8)
    root = str(_seed_lib(tmp_path / "lib"))
    pool = fl.build_library_pool("ashare", daily, root=root, compact=True)
    assert isinstance(pool, fl.CompactLibraryPool)
    assert len(pool) > 0
    name0 = pool.factor_names[0]
    assert pool.wide.schema[name0] == pl.Float32

    path = tmp_path / "pool_f32.parquet"
    pool.write_parquet(path)
    loaded = fl.CompactLibraryPool.from_parquet(path, list(pool.factor_names))
    assert loaded.factor_names == pool.factor_names
    assert loaded.wide.schema[name0] == pl.Float32
    _assert_pool_wide_equal(
        pool.wide.with_columns(pl.col("ts_code").cast(pl.Utf8)),
        loaded.wide,
    )
    assert_frame_equal(
        pool[name0].with_columns(pl.col("ts_code").cast(pl.Utf8)),
        loaded[name0],
        check_dtypes=True,
    )


# ── 3. Categorical 阈值 ─────────────────────────────────────────────────────


def test_from_parquet_categorical_keys(tmp_path):
    from factorzen.discovery.factor_library import CompactLibraryPool, build_library_pool
    from factorzen.research.combination import pool as pool_mod

    daily = _mk_daily(n_days=20, n_stocks=5)
    root = str(_seed_lib(tmp_path / "lib"))
    built = build_library_pool("ashare", daily, root=root, compact=True)
    assert isinstance(built, CompactLibraryPool)
    path = tmp_path / "wide.parquet"
    built.write_parquet(path)
    names = list(built.factor_names)

    # 默认阈值(4M)下小帧 → Utf8
    default_loaded = CompactLibraryPool.from_parquet(path, names)
    assert default_loaded.wide.schema["ts_code"] == pl.Utf8

    # 显式 True/False
    cat_on = CompactLibraryPool.from_parquet(path, names, categorical_keys=True)
    assert cat_on.wide.schema["ts_code"] == pl.Categorical
    cat_off = CompactLibraryPool.from_parquet(path, names, categorical_keys=False)
    assert cat_off.wide.schema["ts_code"] == pl.Utf8

    # 阈值降到 1 → 自动 Categorical
    old = pool_mod.POOL_KEYS_CATEGORICAL_ROWS_THRESHOLD
    try:
        pool_mod.POOL_KEYS_CATEGORICAL_ROWS_THRESHOLD = 1
        auto_cat = CompactLibraryPool.from_parquet(path, names)
        assert auto_cat.wide.schema["ts_code"] == pl.Categorical
    finally:
        pool_mod.POOL_KEYS_CATEGORICAL_ROWS_THRESHOLD = old


# ── 4. 常量对齐 ─────────────────────────────────────────────────────────────


def test_keys_categorical_threshold_aligned():
    from factorzen.discovery.preparation import KEYS_CATEGORICAL_ROWS_THRESHOLD
    from factorzen.research.combination.pool import (
        POOL_KEYS_CATEGORICAL_ROWS_THRESHOLD,
    )

    assert (
        POOL_KEYS_CATEGORICAL_ROWS_THRESHOLD == KEYS_CATEGORICAL_ROWS_THRESHOLD
    )


# ── 5. 缓存命中 ─────────────────────────────────────────────────────────────


def test_pool_cache_hit(tmp_path):
    from factorzen.discovery.factor_library import (
        CompactLibraryPool,
        build_library_pool,
        write_pool_cache,
    )

    daily = _mk_daily(n_days=30, n_stocks=8)
    root = str(_seed_lib(tmp_path / "lib"))
    statuses = ("active",)
    pool = build_library_pool(
        "ashare", daily, root=root, compact=True, statuses=statuses,
    )
    assert isinstance(pool, CompactLibraryPool)

    cache_dir = tmp_path / "cache"
    write_pool_cache(
        pool,
        cache_dir,
        meta=_hand_meta(
            market="ashare", root=root, statuses=statuses, daily=daily,
        ),
    )
    assert (cache_dir / "pool_meta.json").exists()
    assert (cache_dir / "pool_wide.parquet").exists()

    loaded = build_library_pool(
        "ashare", daily, root=root, compact=True, statuses=statuses,
        cache_dir=cache_dir,
    )
    assert isinstance(loaded, CompactLibraryPool)
    assert loaded.factor_names == pool.factor_names
    _assert_pool_wide_equal(
        pool.wide.with_columns(pl.col("ts_code").cast(pl.Utf8)),
        loaded.wide.with_columns(pl.col("ts_code").cast(pl.Utf8)),
    )
    name = pool.factor_names[0]
    assert_frame_equal(
        pool[name].with_columns(pl.col("ts_code").cast(pl.Utf8)),
        loaded[name].with_columns(pl.col("ts_code").cast(pl.Utf8)),
        check_dtypes=True,
    )


# ── 6. 缓存失效各路 ─────────────────────────────────────────────────────────


def test_pool_cache_invalidation_paths(tmp_path, capsys):
    from factorzen.discovery.factor_library import (
        CompactLibraryPool,
        build_library_pool,
        load_pool_cache,
        write_pool_cache,
    )

    daily = _mk_daily(n_days=30, n_stocks=8)
    lib_root = tmp_path / "lib"
    root = str(_seed_lib(lib_root))
    statuses = ("active",)
    pool = build_library_pool(
        "ashare", daily, root=root, compact=True, statuses=statuses,
    )
    assert isinstance(pool, CompactLibraryPool)

    cache_dir = tmp_path / "cache"
    meta = _hand_meta(
        market="ashare", root=root, statuses=statuses, daily=daily,
    )
    write_pool_cache(pool, cache_dir, meta=meta)

    # 6a. 库文件追加 → hash 变 → 重建(返回仍正确)
    lib_path = lib_root / "ashare.jsonl"
    with lib_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "expression": "rank(open)",
            "market": "ashare",
            "status": "active",
            "ic_train": 0.02,
        }, ensure_ascii=False) + "\n")
    rebuilt = build_library_pool(
        "ashare", daily, root=root, compact=True, statuses=statuses,
        cache_dir=cache_dir,
    )
    assert isinstance(rebuilt, CompactLibraryPool)
    # 多了一个因子,不是缓存旧池
    assert "rank(open)" in rebuilt.factor_names
    assert set(pool.factor_names).issubset(set(rebuilt.factor_names))
    out = capsys.readouterr().out
    assert "池缓存失效" in out

    # 恢复库并重写缓存,测其余失效路径
    _write_lib(lib_root, "ashare", [
        {"expression": "rank(close)", "market": "ashare", "status": "active",
         "ic_train": 0.05},
        {"expression": "rank(vol)", "market": "ashare", "status": "active",
         "ic_train": 0.04},
        {"expression": "rank(amount)", "market": "ashare", "status": "active",
         "ic_train": 0.03},
    ])
    pool2 = build_library_pool(
        "ashare", daily, root=root, compact=True, statuses=statuses,
    )
    write_pool_cache(
        pool2, cache_dir,
        meta=_hand_meta(
            market="ashare", root=root, statuses=statuses, daily=daily,
        ),
    )

    # 6b. expect_height 不匹配(帧裁一行)
    daily_short = daily.head(daily.height - 1)
    miss_h = load_pool_cache(
        cache_dir,
        market="ashare",
        root=root,
        statuses=statuses,
        eval_start=None,
        expect_height=daily_short.height,
        expect_date_min=daily_short["trade_date"].min(),
        expect_date_max=daily_short["trade_date"].max(),
    )
    assert miss_h is None
    assert "池缓存失效" in capsys.readouterr().out

    # 6c. meta 缺失
    meta_path = cache_dir / "pool_meta.json"
    meta_bak = meta_path.read_text(encoding="utf-8")
    meta_path.unlink()
    miss_meta = load_pool_cache(
        cache_dir,
        market="ashare",
        root=root,
        statuses=statuses,
        eval_start=None,
        expect_height=daily.height,
        expect_date_min=daily["trade_date"].min(),
        expect_date_max=daily["trade_date"].max(),
    )
    assert miss_meta is None
    meta_path.write_text(meta_bak, encoding="utf-8")

    # 6d. statuses 不匹配
    miss_st = load_pool_cache(
        cache_dir,
        market="ashare",
        root=root,
        statuses=("probation",),
        eval_start=None,
        expect_height=daily.height,
        expect_date_min=daily["trade_date"].min(),
        expect_date_max=daily["trade_date"].max(),
    )
    assert miss_st is None
    assert "池缓存失效" in capsys.readouterr().out


# ── 7. 空池 ─────────────────────────────────────────────────────────────────


def test_empty_pool_cache(tmp_path):
    from factorzen.discovery.factor_library import (
        build_library_pool,
        load_pool_cache,
        write_pool_cache,
    )

    daily = _mk_daily(n_days=20, n_stocks=5)
    # 空库:无 jsonl
    root = str(tmp_path / "empty_lib")
    Path(root).mkdir(parents=True, exist_ok=True)
    empty = build_library_pool("ashare", daily, root=root, compact=True)
    assert empty == {}

    cache_dir = tmp_path / "cache_empty"
    write_pool_cache(
        empty,
        cache_dir,
        meta=_hand_meta(
            market="ashare", root=root, statuses=("active",), daily=daily,
        ),
    )
    assert (cache_dir / "pool_meta.json").exists()
    assert not (cache_dir / "pool_wide.parquet").exists()

    loaded = load_pool_cache(
        cache_dir,
        market="ashare",
        root=root,
        statuses=("active",),
        eval_start=None,
        expect_height=daily.height,
        expect_date_min=daily["trade_date"].min(),
        expect_date_max=daily["trade_date"].max(),
    )
    assert loaded == {}


# ── 8. factor_names 缺列 ────────────────────────────────────────────────────


def test_from_parquet_missing_factor_names_invalidates(tmp_path, capsys):
    from factorzen.discovery.factor_library import (
        CompactLibraryPool,
        build_library_pool,
        load_pool_cache,
        write_pool_cache,
    )

    daily = _mk_daily(n_days=20, n_stocks=5)
    root = str(_seed_lib(tmp_path / "lib"))
    pool = build_library_pool("ashare", daily, root=root, compact=True)
    assert isinstance(pool, CompactLibraryPool)

    cache_dir = tmp_path / "cache_bad"
    write_pool_cache(
        pool,
        cache_dir,
        meta=_hand_meta(
            market="ashare", root=root, statuses=("active",), daily=daily,
        ),
    )
    # 篡改 meta:多写一个不存在的因子名
    meta_path = cache_dir / "pool_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["factor_names"] = [*list(meta["factor_names"]), "not_a_real_factor"]
    meta["n_factors"] = len(meta["factor_names"])
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # from_parquet 直接抛
    with pytest.raises(ValueError, match="not_a_real_factor"):
        CompactLibraryPool.from_parquet(
            cache_dir / "pool_wide.parquet", meta["factor_names"],
        )

    # load_pool_cache 按失效处理,不崩
    result = load_pool_cache(
        cache_dir,
        market="ashare",
        root=root,
        statuses=("active",),
        eval_start=None,
        expect_height=daily.height,
        expect_date_min=daily["trade_date"].min(),
        expect_date_max=daily["trade_date"].max(),
    )
    assert result is None
    assert "池缓存失效" in capsys.readouterr().out


# ── 9+. CLI: pool-prebuild + mine team --pool-subproc ───────────────────────


def _iso_start_end(daily: pl.DataFrame) -> tuple[str, str]:
    dmin = daily["trade_date"].min()
    dmax = daily["trade_date"].max()
    return dmin.strftime("%Y%m%d"), dmax.strftime("%Y%m%d")


def test_cmd_pool_prebuild_e2e_and_dual_path(tmp_path, monkeypatch):
    """_cmd_pool_prebuild 端到端 + 双路径一致性(cache 装载 vs 进程内 compact)。"""
    from factorzen.cli import main as cli
    from factorzen.discovery.evaluation import _preprocess_daily
    from factorzen.discovery.factor_library import (
        CompactLibraryPool,
        build_library_pool,
        library_file_hash,
    )

    daily = _mk_daily(n_days=30, n_stocks=8)
    root = str(_seed_lib(tmp_path / "lib"))
    start, end = _iso_start_end(daily)
    out = tmp_path / "pool_out"
    prepped_holder: dict = {}

    def fake_prepare(args):
        return daily, None, {
            "membership_hash": "mh1",
            "membership_mode": "test",
            "membership_n_rows": daily.height,
            "universe": None,
        }

    real_preprocess = _preprocess_daily

    def capture_preprocess(frame, profile=None):
        out_df = real_preprocess(frame, profile)
        prepped_holder["df"] = out_df
        return out_df

    monkeypatch.setattr(cli, "_prepare_agent_mining_data", fake_prepare)
    monkeypatch.setattr(
        "factorzen.discovery.evaluation._preprocess_daily", capture_preprocess,
    )
    # _cmd_pool_prebuild 从 evaluation 本地 import,需 patch 调用点所在名
    # 因函数内 from-import,patch 源模块即可在 import 时取到
    import factorzen.discovery.evaluation as eval_mod
    monkeypatch.setattr(eval_mod, "_preprocess_daily", capture_preprocess)

    rc = cli.main([
        "pool-prebuild",
        "--start", start,
        "--end", end,
        "--library-root", root,
        "--out", str(out),
    ])
    assert rc == 0
    assert (out / "pool_meta.json").exists()
    assert (out / "pool_wide.parquet").exists()

    meta = json.loads((out / "pool_meta.json").read_text(encoding="utf-8"))
    assert meta["market"] == "ashare"
    assert meta["library_hash"] == library_file_hash("ashare", root)
    assert meta["prepped_height"] == prepped_holder["df"].height
    # ISO date 格式(str(date)),不是 YYYYMMDD
    assert meta["eval_start"] == f"{start[:4]}-{start[4:6]}-{start[6:]}"
    assert "-" in meta["prepped_date_min"]

    prepped = prepped_holder["df"]
    import datetime as _dt
    eval_start = _dt.datetime.strptime(start, "%Y%m%d").date()

    loaded = build_library_pool(
        "ashare", prepped, root=root, compact=True,
        eval_start=eval_start, cache_dir=out,
    )
    built = build_library_pool(
        "ashare", prepped, root=root, compact=True,
        eval_start=eval_start,
    )
    assert isinstance(loaded, CompactLibraryPool)
    assert isinstance(built, CompactLibraryPool)
    assert loaded.factor_names == built.factor_names
    _assert_pool_wide_equal(
        loaded.wide.with_columns(pl.col("ts_code").cast(pl.Utf8)),
        built.wide.with_columns(pl.col("ts_code").cast(pl.Utf8)),
    )


def _mine_team_base_argv(*, start="20210104", end="20210331", extra=None):
    argv = [
        "mine", "team",
        "--start", start,
        "--end", end,
        "--iterations", "1",
        "--seed", "1",
        "--index-path", "workspace/mine_team/custom_index.jsonl",
    ]
    if extra:
        argv.extend(extra)
    return argv


def _patch_mine_team_data_and_run(monkeypatch, cli, *, captured: dict):
    import polars as pl

    fake_daily = pl.DataFrame({"ts_code": ["600000.SH"], "trade_date": ["2021-01-04"]})

    def fake_prepare(args):
        return fake_daily, None, {
            "membership_hash": "mh1",
            "membership_mode": None,
            "membership_n_rows": 1,
            "universe": None,
        }

    def fake_run_team_mine(daily, **kwargs):
        captured["kwargs"] = kwargs
        captured["daily"] = daily
        return {"n_candidates": 0, "n_trials": 0, "run_dir": "workspace/mine_team/x"}

    monkeypatch.setattr(cli, "_prepare_agent_mining_data", fake_prepare)
    monkeypatch.setattr(
        "factorzen.pipelines.factor_mine_team.run_team_mine", fake_run_team_mine,
    )


def test_cmd_mine_team_pool_subproc_success(monkeypatch, tmp_path):
    """--pool-subproc 成功路:subprocess 被调,pool_cache_dir 透传,argv 含 pool-prebuild。"""
    import subprocess as sp

    from factorzen.cli import main as cli
    from factorzen.config import settings as settings_mod

    mine_dir = tmp_path / "mine_team"
    mine_dir.mkdir()
    monkeypatch.setattr(settings_mod, "MINE_TEAM_DIR", mine_dir)
    # MINE_TEAM_DIR.parent / factor_library → tmp_path/factor_library
    (tmp_path / "factor_library").mkdir(exist_ok=True)

    captured: dict = {}
    sub_calls: list = []

    def fake_run(cmd, **kw):
        sub_calls.append(list(cmd))
        # --out 位置
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "pool_meta.json").write_text(
            json.dumps({"n_factors": 0, "market": "ashare"}), encoding="utf-8",
        )
        return sp.CompletedProcess(cmd, 0)

    monkeypatch.setattr(sp, "run", fake_run)
    _patch_mine_team_data_and_run(monkeypatch, cli, captured=captured)

    rc = cli.main(_mine_team_base_argv(extra=["--pool-subproc"]))
    assert rc == 0
    assert len(sub_calls) == 1
    argv = sub_calls[0]
    assert "pool-prebuild" in argv
    assert "--start" in argv and "20210104" in argv
    assert "--end" in argv and "20210331" in argv
    assert captured["kwargs"].get("pool_cache_dir") is not None
    assert Path(captured["kwargs"]["pool_cache_dir"]).name  # non-empty key dir
    assert (Path(captured["kwargs"]["pool_cache_dir"]) / "pool_meta.json").exists()


def test_cmd_mine_team_pool_subproc_failure_fallback(monkeypatch, tmp_path, capsys):
    """子进程失败 → pool_cache_dir=None + stderr 警告。"""
    import subprocess as sp

    from factorzen.cli import main as cli
    from factorzen.config import settings as settings_mod

    mine_dir = tmp_path / "mine_team"
    mine_dir.mkdir()
    monkeypatch.setattr(settings_mod, "MINE_TEAM_DIR", mine_dir)
    (tmp_path / "factor_library").mkdir(exist_ok=True)

    captured: dict = {}
    sub_calls: list = []

    def fake_run(cmd, **kw):
        sub_calls.append(list(cmd))
        return sp.CompletedProcess(cmd, 1)

    monkeypatch.setattr(sp, "run", fake_run)
    _patch_mine_team_data_and_run(monkeypatch, cli, captured=captured)

    rc = cli.main(_mine_team_base_argv(extra=["--pool-subproc"]))
    assert rc == 0
    assert len(sub_calls) == 1
    assert captured["kwargs"].get("pool_cache_dir") is None
    err = capsys.readouterr().err
    assert "警告" in err and "回退进程内构建" in err


def test_cmd_mine_team_pool_subproc_env(monkeypatch, tmp_path):
    """FACTORZEN_POOL_SUBPROC=1 等效 --pool-subproc。"""
    import subprocess as sp

    from factorzen.cli import main as cli
    from factorzen.config import settings as settings_mod

    mine_dir = tmp_path / "mine_team"
    mine_dir.mkdir()
    monkeypatch.setattr(settings_mod, "MINE_TEAM_DIR", mine_dir)
    (tmp_path / "factor_library").mkdir(exist_ok=True)
    monkeypatch.setenv("FACTORZEN_POOL_SUBPROC", "1")

    captured: dict = {}
    sub_calls: list = []

    def fake_run(cmd, **kw):
        sub_calls.append(list(cmd))
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "pool_meta.json").write_text("{}", encoding="utf-8")
        return sp.CompletedProcess(cmd, 0)

    monkeypatch.setattr(sp, "run", fake_run)
    _patch_mine_team_data_and_run(monkeypatch, cli, captured=captured)

    rc = cli.main(_mine_team_base_argv())  # 不带 --pool-subproc
    assert rc == 0
    assert len(sub_calls) == 1
    assert captured["kwargs"].get("pool_cache_dir") is not None


def test_cmd_mine_team_pool_cache_reuse(monkeypatch, tmp_path):
    """预置 cache_dir(meta 存在) → subprocess.run 不被调用。"""
    import hashlib
    import subprocess as sp

    from factorzen.cli import main as cli
    from factorzen.config import settings as settings_mod
    from factorzen.discovery.factor_library import library_file_hash

    mine_dir = tmp_path / "mine_team"
    mine_dir.mkdir()
    monkeypatch.setattr(settings_mod, "MINE_TEAM_DIR", mine_dir)
    (tmp_path / "factor_library").mkdir(exist_ok=True)

    lib_root = str(tmp_path / "factor_library")
    market = "ashare"
    lib_hash = library_file_hash(market, lib_root) or "nolib"
    key_src = "|".join([
        lib_hash, "20210104", "20210331", "None", market, "0.2", "False", "5min",
    ])
    key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    cache_dir = mine_dir / "_pool_cache" / key
    cache_dir.mkdir(parents=True)
    (cache_dir / "pool_meta.json").write_text(
        json.dumps({"n_factors": 0}), encoding="utf-8",
    )

    captured: dict = {}
    sub_calls: list = []

    def fake_run(cmd, **kw):
        sub_calls.append(list(cmd))
        return sp.CompletedProcess(cmd, 0)

    monkeypatch.setattr(sp, "run", fake_run)
    _patch_mine_team_data_and_run(monkeypatch, cli, captured=captured)

    rc = cli.main(_mine_team_base_argv(extra=["--pool-subproc"]))
    assert rc == 0
    assert sub_calls == []
    assert captured["kwargs"].get("pool_cache_dir") == str(cache_dir)


def test_cmd_mine_team_no_pool_subproc_zero_regression(monkeypatch, tmp_path):
    """不带旗标、无 env → subprocess 不调,pool_cache_dir=None。"""
    import subprocess as sp

    from factorzen.cli import main as cli
    from factorzen.config import settings as settings_mod

    mine_dir = tmp_path / "mine_team"
    mine_dir.mkdir()
    monkeypatch.setattr(settings_mod, "MINE_TEAM_DIR", mine_dir)
    monkeypatch.delenv("FACTORZEN_POOL_SUBPROC", raising=False)

    captured: dict = {}
    sub_calls: list = []

    def fake_run(cmd, **kw):
        sub_calls.append(list(cmd))
        return sp.CompletedProcess(cmd, 0)

    monkeypatch.setattr(sp, "run", fake_run)
    _patch_mine_team_data_and_run(monkeypatch, cli, captured=captured)

    rc = cli.main(_mine_team_base_argv())
    assert rc == 0
    assert sub_calls == []
    assert captured["kwargs"].get("pool_cache_dir") is None


def test_parser_pool_prebuild_and_pool_subproc_flag():
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args([
        "pool-prebuild",
        "--start", "20220101",
        "--end", "20231231",
        "--out", "/tmp/pool",
    ])
    assert args.command == "pool-prebuild"
    assert args.start == "20220101"
    assert args.out == "/tmp/pool"
    assert args.holdout_ratio == 0.2
    assert callable(args.func)

    team = p.parse_args([
        "mine", "team",
        "--start", "20220101",
        "--end", "20231231",
        "--pool-subproc",
    ])
    assert team.pool_subproc is True
