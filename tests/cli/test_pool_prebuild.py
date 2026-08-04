"""库池 parquet 磁盘交接 + cache_dir 装载校验（全离线）。"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tests.daily.test_factor_library import _mk_daily, _seed_lib, _write_lib

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


def test_parquet_roundtrip_suite(tmp_path):
    """test_parquet_roundtrip_f64；test_parquet_roundtrip_f32"""
    # -- 原 test_parquet_roundtrip_f64 --
    def _section_0_test_parquet_roundtrip_f64(tmp_path):
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

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_parquet_roundtrip_f64(_tp0)

    # -- 原 test_parquet_roundtrip_f32 --
    def _section_1_test_parquet_roundtrip_f32(tmp_path, mp):
        import factorzen.discovery.factor_library as fl

        mp.setattr(fl, "POOL_VALUE_F32_BYTES_THRESHOLD", 1)

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

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_parquet_roundtrip_f32(_tp1, mp)


# ── 2. round-trip f32 ───────────────────────────────────────────────────────


# ── 3. Categorical 阈值 ─────────────────────────────────────────────────────


def test_from_parquet_keys_suite(tmp_path, capsys):
    """test_from_parquet_categorical_keys；test_keys_categorical_threshold_aligned；test_from_parquet_missing_factor_names_invalidates"""
    # -- 原 test_from_parquet_categorical_keys --
    def _section_0_test_from_parquet_categorical_keys(tmp_path):
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

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_from_parquet_categorical_keys(_tp0)

    # -- 原 test_keys_categorical_threshold_aligned --
    def _section_1_test_keys_categorical_threshold_aligned():
        from factorzen.discovery.preparation import KEYS_CATEGORICAL_ROWS_THRESHOLD
        from factorzen.research.combination.pool import (
            POOL_KEYS_CATEGORICAL_ROWS_THRESHOLD,
        )

        assert (
            POOL_KEYS_CATEGORICAL_ROWS_THRESHOLD == KEYS_CATEGORICAL_ROWS_THRESHOLD
        )

    _section_1_test_keys_categorical_threshold_aligned()

    # -- 原 test_from_parquet_missing_factor_names_invalidates --
    def _section_2_test_from_parquet_missing_factor_names_invalidates(tmp_path, capsys):
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

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_from_parquet_missing_factor_names_invalidates(_tp2, capsys)


# ── 4. 常量对齐 ─────────────────────────────────────────────────────────────


# ── 5. 缓存命中 ─────────────────────────────────────────────────────────────


def test_pool_cache_lifecycle_suite(tmp_path, capsys):
    """test_pool_cache_hit；test_pool_cache_invalidation_paths；test_empty_pool_cache"""
    # -- 原 test_pool_cache_hit --
    def _section_0_test_pool_cache_hit(tmp_path):
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

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_pool_cache_hit(_tp0)

    # -- 原 test_pool_cache_invalidation_paths --
    def _section_1_test_pool_cache_invalidation_paths(tmp_path, capsys):
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

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_pool_cache_invalidation_paths(_tp1, capsys)

    # -- 原 test_empty_pool_cache --
    def _section_2_test_empty_pool_cache(tmp_path):
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

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_empty_pool_cache(_tp2)


# ── 6. 缓存失效各路 ─────────────────────────────────────────────────────────


# ── 7. 空池 ─────────────────────────────────────────────────────────────────


# ── 8. factor_names 缺列 ────────────────────────────────────────────────────


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
        "mine", "pool-prebuild",
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
        "--set", "index_path=workspace/mine_team/custom_index.jsonl",
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


def test_pool_subproc_cli_suite(tmp_path, capsys):
    """--pool-subproc 成功路:subprocess 被调,pool_cache_dir 透传,argv 含 pool-prebuild。；子进程失败 → pool_cache_dir=None + stderr 警告。；FACTORZEN_POOL_SUBPROC=1 等效 --pool-subproc。；预置 cache_dir(meta 存在) → subprocess.run 不被调用。；不带旗标、无 env → subprocess 不调,pool_cache_dir=None。；test_parser_pool_prebuild_and_pool_subproc_flag"""
    # -- 原 test_cmd_mine_team_pool_subproc_success --
    def _section_0_test_cmd_mine_team_pool_subproc_success(mp, tmp_path):
        import subprocess as sp

        from factorzen.cli import main as cli
        from factorzen.config import settings as settings_mod

        mine_dir = tmp_path / "mine_team"
        mine_dir.mkdir()
        mp.setattr(settings_mod, "MINE_TEAM_DIR", mine_dir)
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

        mp.setattr(sp, "run", fake_run)
        _patch_mine_team_data_and_run(mp, cli, captured=captured)

        rc = cli.main(_mine_team_base_argv(extra=["--pool-subproc"]))
        assert rc == 0
        assert len(sub_calls) == 1
        argv = sub_calls[0]
        assert "pool-prebuild" in argv
        # spawn 路径归位：mine pool-prebuild（非顶层）
        assert "mine" in argv
        i_pool = argv.index("pool-prebuild")
        assert argv[i_pool - 1] == "mine"
        assert "--start" in argv and "20210104" in argv
        assert "--end" in argv and "20210331" in argv
        assert captured["kwargs"].get("pool_cache_dir") is not None
        assert Path(captured["kwargs"]["pool_cache_dir"]).name  # non-empty key dir
        assert (Path(captured["kwargs"]["pool_cache_dir"]) / "pool_meta.json").exists()

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_cmd_mine_team_pool_subproc_success(mp, _tp0)

    # -- 原 test_cmd_mine_team_pool_subproc_failure_fallback --
    def _section_1_test_cmd_mine_team_pool_subproc_failure_fallback(mp, tmp_path, capsys):
        import subprocess as sp

        from factorzen.cli import main as cli
        from factorzen.config import settings as settings_mod

        mine_dir = tmp_path / "mine_team"
        mine_dir.mkdir()
        mp.setattr(settings_mod, "MINE_TEAM_DIR", mine_dir)
        (tmp_path / "factor_library").mkdir(exist_ok=True)

        captured: dict = {}
        sub_calls: list = []

        def fake_run(cmd, **kw):
            sub_calls.append(list(cmd))
            return sp.CompletedProcess(cmd, 1)

        mp.setattr(sp, "run", fake_run)
        _patch_mine_team_data_and_run(mp, cli, captured=captured)

        rc = cli.main(_mine_team_base_argv(extra=["--pool-subproc"]))
        assert rc == 0
        assert len(sub_calls) == 1
        assert captured["kwargs"].get("pool_cache_dir") is None
        err = capsys.readouterr().err
        assert "警告" in err and "回退进程内构建" in err

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_cmd_mine_team_pool_subproc_failure_fallback(mp, _tp1, capsys)

    # -- 原 test_cmd_mine_team_pool_subproc_env --
    def _section_2_test_cmd_mine_team_pool_subproc_env(mp, tmp_path):
        import subprocess as sp

        from factorzen.cli import main as cli
        from factorzen.config import settings as settings_mod

        mine_dir = tmp_path / "mine_team"
        mine_dir.mkdir()
        mp.setattr(settings_mod, "MINE_TEAM_DIR", mine_dir)
        (tmp_path / "factor_library").mkdir(exist_ok=True)
        mp.setenv("FACTORZEN_POOL_SUBPROC", "1")

        captured: dict = {}
        sub_calls: list = []

        def fake_run(cmd, **kw):
            sub_calls.append(list(cmd))
            out = Path(cmd[cmd.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "pool_meta.json").write_text("{}", encoding="utf-8")
            return sp.CompletedProcess(cmd, 0)

        mp.setattr(sp, "run", fake_run)
        _patch_mine_team_data_and_run(mp, cli, captured=captured)

        rc = cli.main(_mine_team_base_argv())  # 不带 --pool-subproc
        assert rc == 0
        assert len(sub_calls) == 1
        assert captured["kwargs"].get("pool_cache_dir") is not None

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_cmd_mine_team_pool_subproc_env(mp, _tp2)

    # -- 原 test_cmd_mine_team_pool_cache_reuse --
    def _section_3_test_cmd_mine_team_pool_cache_reuse(mp, tmp_path):
        import hashlib
        import subprocess as sp

        from factorzen.cli import main as cli
        from factorzen.config import settings as settings_mod
        from factorzen.discovery.factor_library import library_file_hash

        mine_dir = tmp_path / "mine_team"
        mine_dir.mkdir()
        mp.setattr(settings_mod, "MINE_TEAM_DIR", mine_dir)
        (tmp_path / "factor_library").mkdir(exist_ok=True)

        lib_root = str(tmp_path / "factor_library")
        market = "ashare"
        lib_hash = library_file_hash(market, lib_root) or "nolib"
        # CLI 默认 universe=all_a（与 key 指纹同源）
        key_src = "|".join([
            lib_hash, "20210104", "20210331", "all_a", market, "0.2", "False", "5min",
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

        mp.setattr(sp, "run", fake_run)
        _patch_mine_team_data_and_run(mp, cli, captured=captured)

        rc = cli.main(_mine_team_base_argv(extra=["--pool-subproc"]))
        assert rc == 0
        assert sub_calls == []
        assert captured["kwargs"].get("pool_cache_dir") == str(cache_dir)

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_cmd_mine_team_pool_cache_reuse(mp, _tp3)

    # -- 原 test_cmd_mine_team_no_pool_subproc_zero_regression --
    def _section_4_test_cmd_mine_team_no_pool_subproc_zero_regression(mp, tmp_path):
        import subprocess as sp

        from factorzen.cli import main as cli
        from factorzen.config import settings as settings_mod

        mine_dir = tmp_path / "mine_team"
        mine_dir.mkdir()
        mp.setattr(settings_mod, "MINE_TEAM_DIR", mine_dir)
        mp.delenv("FACTORZEN_POOL_SUBPROC", raising=False)

        captured: dict = {}
        sub_calls: list = []

        def fake_run(cmd, **kw):
            sub_calls.append(list(cmd))
            return sp.CompletedProcess(cmd, 0)

        mp.setattr(sp, "run", fake_run)
        _patch_mine_team_data_and_run(mp, cli, captured=captured)

        # 自适应可能在 all_a 上开启子进程；显式 --no-pool-subproc 保证零回归关断
        rc = cli.main(_mine_team_base_argv(extra=["--no-pool-subproc"]))
        assert rc == 0
        assert sub_calls == []
        assert captured["kwargs"].get("pool_cache_dir") is None

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_4_test_cmd_mine_team_no_pool_subproc_zero_regression(mp, _tp4)

    # -- 原 test_parser_pool_prebuild_and_pool_subproc_flag --
    def _section_5_test_parser_pool_prebuild_and_pool_subproc_flag():
        from factorzen.cli.main import build_parser

        p = build_parser()
        args = p.parse_args([
            "mine", "pool-prebuild",
            "--start", "20220101",
            "--end", "20231231",
            "--out", "/tmp/pool",
        ])
        assert args.command == "mine"
        assert args.mine_command == "pool-prebuild"
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

    _section_5_test_parser_pool_prebuild_and_pool_subproc_flag()


# ── 10. 值列 spill 落盘（构建期 OOM 修复）──────────────────────────────────


def _spill_lib_records() -> list[dict]:
    """3 好 + 1 全 null(amount 被置空) + 1 解析失败 + 1 python 哨兵。"""
    return [
        {"expression": "rank(close)", "market": "ashare", "status": "active",
         "ic_train": 0.05},
        {"expression": "ts_mean(vol, 3)", "market": "ashare", "status": "active",
         "ic_train": 0.045},
        {"expression": "rank(amount)", "market": "ashare", "status": "active",
         "ic_train": 0.04},          # amount 全 null → 全 null 列跳过
        {"expression": "rank(not_a_leaf)", "market": "ashare", "status": "active",
         "ic_train": 0.03},          # parse_expr 抛 ValueError → 单因子跳过
        {"expression": "py::pyfac", "kind": "python", "name": "pyfac",
         "market": "ashare", "status": "active", "ic_train": 0.02},
        {"expression": "delta(high, 2)", "market": "ashare", "status": "active",
         "ic_train": 0.01},
    ]


def _spill_daily() -> pl.DataFrame:
    """_mk_daily + amount 全 null（构造「全 null 列跳过」分支）。"""
    daily = _mk_daily(n_days=24, n_stocks=6)
    return daily.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("amount"),
    )


def _py_materializer(daily: pl.DataFrame):
    """py 因子面板：对 daily 网格前 2/3 行给确定性值（有覆盖也有缺口）。"""
    def _mat(name: str):
        if name != "pyfac":
            return None
        sub = daily.select(["trade_date", "ts_code"]).head(
            int(daily.height * 2 / 3),
        )
        return sub.with_columns(
            (pl.col("ts_code").str.slice(3, 3).cast(pl.Float64)
             + pl.col("trade_date").dt.day().cast(pl.Float64)).alias("factor_value"),
        )
    return _mat


def test_pool_spill_parity_suite(tmp_path, capsys):
    """spill 落盘路径 vs 内存路径:宽帧逐值/列序/names/dtype 全等 + 落盘副产物契约。"""
    import factorzen.discovery.factor_library as fl

    daily = _spill_daily()
    root = str(tmp_path / "lib")
    _write_lib(Path(root), "ashare", _spill_lib_records())
    mat = _py_materializer(daily)

    def _build(*, spill: bool, eval_start=None, spill_dir=None):
        with pytest.MonkeyPatch.context() as mp:
            # spill 与 f32 同估算口径:两者必须一起开/一起关,否则 dtype 天然不等
            mp.setattr(
                fl, "POOL_SPILL_BYTES_THRESHOLD", 1 if spill else 1 << 60,
            )
            mp.setattr(
                fl, "POOL_VALUE_F32_BYTES_THRESHOLD", 1 if spill else 1 << 60,
            )
            return fl.build_library_pool(
                "ashare", daily, root=root, compact=True,
                eval_start=eval_start, python_materializer=mat,
                spill_dir=spill_dir,
            )

    # -- 10a. 基准:内存路径本身产出非平凡池(判别力前提) --
    mem = _build(spill=False)
    assert isinstance(mem, fl.CompactLibraryPool)
    # 6 条记录 → 4 条应物化(rank(amount) 全 null / rank(not_a_leaf) 解析失败被跳)
    assert mem.factor_names == (
        "rank(close)", "ts_mean(vol, 3)", "py::pyfac", "delta(high, 2)",
    ), mem.factor_names
    assert list(mem.wide.columns) == ["trade_date", "ts_code", *mem.factor_names]
    # 内存路径未触发 f32(阈值被顶到 1<<60)
    assert mem.wide.schema["rank(close)"] == pl.Float64

    # 与 spill 同 dtype 的内存基准(只开 f32,不开 spill)——parity 比对的对照组
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fl, "POOL_SPILL_BYTES_THRESHOLD", 1 << 60)
        mp.setattr(fl, "POOL_VALUE_F32_BYTES_THRESHOLD", 1)
        mem_f32 = fl.build_library_pool(
            "ashare", daily, root=root, compact=True,
            python_materializer=mat,
        )
    assert isinstance(mem_f32, fl.CompactLibraryPool)

    # -- 10b. spill 路径逐值 parity + 落盘副产物 --
    sdir = tmp_path / "spill"
    spilled = _build(spill=True, spill_dir=sdir)
    out = capsys.readouterr().out
    assert "spill 模式" in out
    assert isinstance(spilled, fl.CompactLibraryPool)
    # 落盘契约:每个入池因子恰一个单列 parquet,行高=最终宽帧行高
    files = sorted(sdir.glob("*.parquet"))
    assert len(files) == len(spilled.factor_names) == 4
    one = pl.read_parquet(files[0])
    assert one.columns == ["v"]
    assert one.height == spilled.wide.height
    # 显式 spill_dir 由调用方持有 → 不被删
    assert sdir.exists()

    assert spilled.factor_names == mem_f32.factor_names
    assert list(spilled.wide.columns) == list(mem_f32.wide.columns)
    _assert_pool_wide_equal(spilled.wide, mem_f32.wide)
    # null 位置逐列一致(不是「都非 null」这种恒真断言)
    for n in spilled.factor_names:
        assert spilled.wide[n].null_count() == mem_f32.wide[n].null_count()
        assert (
            spilled.wide[n].is_null().to_list() == mem_f32.wide[n].is_null().to_list()
        )
    # py 因子确实带 null(materializer 只覆盖 2/3 网格)——保证上面 null 比对非平凡
    assert spilled.wide["py::pyfac"].null_count() > 0
    # 值列确实是 f32(spill 与 f32 同阈值同时触发)
    assert spilled.wide.schema["rank(close)"] == pl.Float32

    # -- 10c. eval_start 裁剪语义在 spill 下一致 --
    all_dates = sorted(daily["trade_date"].unique().to_list())
    cut = all_dates[len(all_dates) // 2]
    mem_cut = _build(spill=False, eval_start=cut)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fl, "POOL_SPILL_BYTES_THRESHOLD", 1 << 60)
        mp.setattr(fl, "POOL_VALUE_F32_BYTES_THRESHOLD", 1)
        mem_cut_f32 = fl.build_library_pool(
            "ashare", daily, root=root, compact=True,
            eval_start=cut, python_materializer=mat,
        )
    spill_cut = _build(spill=True, eval_start=cut, spill_dir=tmp_path / "spill_cut")
    assert isinstance(mem_cut, fl.CompactLibraryPool)
    assert isinstance(mem_cut_f32, fl.CompactLibraryPool)
    assert isinstance(spill_cut, fl.CompactLibraryPool)
    # 裁剪真的裁掉了行(否则下面的相等是恒真)
    assert 0 < mem_cut.wide.height < mem.wide.height
    assert spill_cut.wide.height == mem_cut.wide.height
    assert spill_cut.factor_names == mem_cut_f32.factor_names
    _assert_pool_wide_equal(spill_cut.wide, mem_cut_f32.wide)
    assert spill_cut.wide["trade_date"].min() == cut

    # -- 10d. 默认 spill_dir(自建临时目录)构建完即删 --
    made: dict = {}
    real_mkdtemp = fl.tempfile.mkdtemp

    def fake_mkdtemp(*a, **kw):
        d = real_mkdtemp(dir=str(tmp_path))
        made["dir"] = d
        return d

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fl, "POOL_SPILL_BYTES_THRESHOLD", 1)
        mp.setattr(fl, "POOL_VALUE_F32_BYTES_THRESHOLD", 1)
        mp.setattr(fl.tempfile, "mkdtemp", fake_mkdtemp)
        auto = fl.build_library_pool(
            "ashare", daily, root=root, compact=True, python_materializer=mat,
        )
    assert isinstance(auto, fl.CompactLibraryPool)
    assert made["dir"] and not Path(made["dir"]).exists()
    _assert_pool_wide_equal(auto.wide, mem_f32.wide)

    # -- 10e. 小帧默认阈值不触发 spill(零回归护栏) --
    capsys.readouterr()
    plain = fl.build_library_pool(
        "ashare", daily, root=root, compact=True, python_materializer=mat,
    )
    assert isinstance(plain, fl.CompactLibraryPool)
    assert "spill 模式" not in capsys.readouterr().out
    _assert_pool_wide_equal(plain.wide, mem.wide)

    # -- 10f. 空池(全部跳过)也不留临时目录、仍返回 {} --
    empty_root = str(tmp_path / "lib_bad")
    _write_lib(Path(empty_root), "ashare", [
        {"expression": "rank(not_a_leaf)", "market": "ashare",
         "status": "active", "ic_train": 0.1},
    ])
    made.clear()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fl, "POOL_SPILL_BYTES_THRESHOLD", 1)
        mp.setattr(fl, "POOL_VALUE_F32_BYTES_THRESHOLD", 1)
        mp.setattr(fl.tempfile, "mkdtemp", fake_mkdtemp)
        nothing = fl.build_library_pool(
            "ashare", daily, root=empty_root, compact=True,
        )
    assert nothing == {}
    assert made["dir"] and not Path(made["dir"]).exists()


# ── 11. prepped 帧裁列（构建期 OOM 修复）──────────────────────────────────


def test_pool_prebuild_prepped_trim_suite(tmp_path, monkeypatch, capsys):
    """library_pool_required_columns 列集正确 + 裁后求值不变 + prebuild 端到端裁列。"""
    import factorzen.discovery.factor_library as fl
    from factorzen.discovery.expression import LEAF_FEATURES

    # -- 11a. 列集 = 键列 ∪ 表达式叶列;python/坏记录不贡献列 --
    root = str(tmp_path / "lib")
    _write_lib(Path(root), "ashare", [
        {"expression": "rank(close)", "market": "ashare", "status": "active",
         "ic_train": 0.05},
        {"expression": "ts_mean(vol, 3)", "market": "ashare", "status": "active",
         "ic_train": 0.04},
        {"expression": "add(delta(high, 2), low)", "market": "ashare",
         "status": "active", "ic_train": 0.03},
        {"expression": "rank(not_a_leaf)", "market": "ashare", "status": "active",
         "ic_train": 0.02},
        {"expression": "py::pyfac", "kind": "python", "name": "pyfac",
         "market": "ashare", "status": "active", "ic_train": 0.01},
        # 非 active → 不参与 statuses=("active",) 的列集
        {"expression": "rank(amount)", "market": "ashare", "status": "probation",
         "ic_train": 0.06},
    ])
    got = fl.library_pool_required_columns("ashare", None, root=root)
    # 期望值由 LEAF_FEATURES 独立查表拼出(不抄实现)
    expect = {"trade_date", "ts_code"} | {
        LEAF_FEATURES[leaf] for leaf in ("close", "vol", "high", "low")
    }
    assert got == expect
    # 判别力:probation 的 amount、坏记录的叶都不在;close 走的是映射后的 close_adj
    assert LEAF_FEATURES["close"] == "close_adj"
    assert "close_adj" in got and "close" not in got
    assert LEAF_FEATURES["amount"] not in got
    assert "not_a_leaf" not in got
    # 显式 statuses 放开 probation 后 amount 列才进来
    got_pro = fl.library_pool_required_columns(
        "ashare", None, statuses=("active", "probation"), root=root,
    )
    assert got_pro == expect | {LEAF_FEATURES["amount"]}

    # -- 11b. 裁后求值逐值不变 --
    daily = _mk_daily(n_days=24, n_stocks=6)
    from factorzen.discovery.evaluation import _preprocess_daily

    prepped = _preprocess_daily(daily).with_columns(
        pl.lit(1.234).alias("junk_col"),
    )
    keep = fl.library_pool_required_columns("ashare", None, root=root)
    sel = [c for c in prepped.columns if c in keep]
    # 判别力:确实裁掉了列(否则下面相等恒真)
    assert len(sel) < len(prepped.columns)
    assert "junk_col" in prepped.columns and "junk_col" not in sel
    full_pool = fl.build_library_pool("ashare", prepped, root=root, compact=True)
    trim_pool = fl.build_library_pool(
        "ashare", prepped.select(sel), root=root, compact=True,
    )
    assert isinstance(full_pool, fl.CompactLibraryPool)
    assert isinstance(trim_pool, fl.CompactLibraryPool)
    assert full_pool.factor_names == trim_pool.factor_names
    assert len(full_pool.factor_names) == 3
    _assert_pool_wide_equal(trim_pool.wide, full_pool.wide)

    # -- 11c. _cmd_pool_prebuild 端到端:交给 build 的帧已裁列 --
    from factorzen.cli import main as cli

    captured: dict = {}
    real_build = fl.build_library_pool

    def spy_build(market, frame, *a, **kw):
        captured["cols"] = list(frame.columns)
        return real_build(market, frame, *a, **kw)

    def fake_prepare(args):
        return daily.with_columns(pl.lit(9.9).alias("junk_col")), None, {
            "membership_hash": "mh1", "membership_mode": "test",
            "membership_n_rows": daily.height, "universe": None,
        }

    monkeypatch.setattr(cli, "_prepare_agent_mining_data", fake_prepare)
    monkeypatch.setattr(fl, "build_library_pool", spy_build)

    start, end = _iso_start_end(daily)
    out = tmp_path / "pool_out"
    rc = cli.main([
        "mine", "pool-prebuild",
        "--start", start, "--end", end,
        "--library-root", root, "--out", str(out),
    ])
    assert rc == 0
    cols = captured["cols"]
    assert "junk_col" not in cols
    assert set(cols) == keep & set(cols)
    assert expect.issubset(set(cols))
    assert "裁列" in capsys.readouterr().out
    # 裁列后池内容与「不裁列」一致(端到端不改语义)
    loaded = fl.CompactLibraryPool.from_parquet(out / "pool_wide.parquet")
    assert set(loaded.factor_names) == set(full_pool.factor_names)


