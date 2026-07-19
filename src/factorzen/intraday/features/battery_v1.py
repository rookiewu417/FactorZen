"""日内特征电池 v1：17 个日频标量特征的规格定义。

记号（在 ``freq`` 重采样后的桶序列上，按
``session_bar_index(标签) // minutes`` 得 1-based 桶序 ``i=1..N``，
``N=bars_per_day``；``k30=30//minutes``）：

- ``r_i = close_i/close_{i-1} − 1``（组内按时间排序 shift；首桶
  ``r_1 = close_1/open_1 − 1``）。
- ``V = Σvol``、``A = Σamount``（当日全部桶）。
- **有效桶** = ``vol>0 且 close 非空且 r 有限``。
- ``ε = 1e-12``。任何分母 ≤0 或统计无意义 → null。

``pre``/``agg`` 作为声明性载体；实际计算由 ``engine.compute_day_panel`` 实现
（含 smart_money 等难以纯表达式表达的特征）。
"""

from __future__ import annotations

import polars as pl

from factorzen.intraday.features.spec import IntradayFeatureSpec
from factorzen.intraday.sessions import ASHARE_BAR_FREQS, normalize_freq

_EPS = 1e-12


def _placeholder_agg(name: str) -> pl.Expr:
    """引擎侧实现的特征占位 agg（schema 声明用）。"""
    return pl.lit(None, dtype=pl.Float64).alias(name)


def battery_v1(freq: str = "5min") -> list[IntradayFeatureSpec]:
    """构造 v1 电池的 17 个特征规格。

    Args:
        freq: 已规范化或可规范化的 bar 频率。

    Returns:
        长度 17 的 ``IntradayFeatureSpec`` 列表，顺序稳定。
    """
    freq_n = normalize_freq(freq)
    minutes = ASHARE_BAR_FREQS[freq_n].minutes
    n = ASHARE_BAR_FREQS[freq_n].bars_per_day
    k30 = 30 // minutes

    specs: list[IntradayFeatureSpec] = [
        IntradayFeatureSpec(
            name="i_rv",
            freq=freq_n,
            pre=(),
            agg=_placeholder_agg("i_rv"),
            formula="sqrt(Σ r_i²)",
            description="日内已实现波动率（桶收益平方和的平方根）",
        ),
        IntradayFeatureSpec(
            name="i_rskew",
            freq=freq_n,
            pre=(),
            agg=_placeholder_agg("i_rskew"),
            formula="sqrt(N_v)·Σr³/(Σr²)^1.5；N_v=有效桶数；Σr²≤ε→null",
            description="日内收益偏度（有效桶三阶矩标准化）",
        ),
        IntradayFeatureSpec(
            name="i_rkurt",
            freq=freq_n,
            pre=(),
            agg=_placeholder_agg("i_rkurt"),
            formula="N_v·Σr⁴/(Σr²)²；Σr²≤ε→null",
            description="日内收益峰度（有效桶四阶矩标准化）",
        ),
        IntradayFeatureSpec(
            name="i_downvol_ratio",
            freq=freq_n,
            pre=(),
            agg=_placeholder_agg("i_downvol_ratio"),
            formula="Σ_{r<0} r² / Σ r²；Σr²≤ε→null",
            description="下行波动占比（负收益平方和占总已实现方差）",
        ),
        IntradayFeatureSpec(
            name="i_updown_vol",
            freq=freq_n,
            pre=(),
            agg=_placeholder_agg("i_updown_vol"),
            formula="ln((Σ_{r>0}r²+ε)/(Σ_{r<0}r²+ε))",
            description="上下行波动比的对数（正/负收益平方和）",
        ),
        IntradayFeatureSpec(
            name="i_ret_open30",
            freq=freq_n,
            pre=(),
            agg=_placeholder_agg("i_ret_open30"),
            formula=(
                f"c_last(i≤k30)/o_first−1；k30={k30}，"
                "c_last 取 i≤k30 最后一个有 close 的桶"
            ),
            description="开盘约 30 分钟收益（首桶 open → 前 k30 桶末 close）",
        ),
        IntradayFeatureSpec(
            name="i_ret_close30",
            freq=freq_n,
            pre=(),
            agg=_placeholder_agg("i_ret_close30"),
            formula=(
                f"c_last/c_last(i≤N−k30)−1；N={n}，k30={k30}"
            ),
            description="收盘约 30 分钟收益（N−k30 桶末 → 全日末 close）",
        ),
        IntradayFeatureSpec(
            name="i_ret_mid",
            freq=freq_n,
            pre=(),
            agg=_placeholder_agg("i_ret_mid"),
            formula=(
                f"c_last(i≤N−k30)/c_last(i≤k30)−1；N={n}，k30={k30}"
            ),
            description="中间时段收益（开盘 30 分末 → 收盘 30 分前）",
        ),
        IntradayFeatureSpec(
            name="i_vwap_dev",
            freq=freq_n,
            pre=(),
            agg=_placeholder_agg("i_vwap_dev"),
            formula="c_last/(A/V)−1；V≤0→null",
            description="收盘相对全日 VWAP 的偏离",
        ),
        IntradayFeatureSpec(
            name="i_pv_corr",
            freq=freq_n,
            pre=(),
            agg=_placeholder_agg("i_pv_corr"),
            formula="pearson corr(close_i, vol_i) 于有效桶；有效桶<10→null",
            description=(
                "价量 Pearson 相关（有效桶≥10；适用 freq≤15min，"
                "30min 时 N=8 恒 null）"
            ),
        ),
        IntradayFeatureSpec(
            name="i_smart_money",
            freq=freq_n,
            pre=(),
            agg=_placeholder_agg("i_smart_money"),
            formula=(
                "S_i=|r_i|/sqrt(vol_i)（vol>0）；按 S 降序累计 vol，"
                "取累计 vol≤0.2·V 的桶再加跨过阈值的一根为 smart 集；"
                "(Σ_smart amount/Σ_smart vol)/(A/V)；V≤0 或 smart 空→null"
            ),
            description="聪明钱：高 |r|/√vol 桶 VWAP 相对全日 VWAP 的比值",
        ),
        IntradayFeatureSpec(
            name="i_vol_open30_share",
            freq=freq_n,
            pre=(),
            agg=_placeholder_agg("i_vol_open30_share"),
            formula=f"Σ_{{i≤k30}} vol / V；k30={k30}；V≤0→null",
            description="开盘约 30 分钟成交量占比",
        ),
        IntradayFeatureSpec(
            name="i_vol_close30_share",
            freq=freq_n,
            pre=(),
            agg=_placeholder_agg("i_vol_close30_share"),
            formula=f"Σ_{{i>N−k30}} vol / V；N={n}，k30={k30}；V≤0→null",
            description="收盘约 30 分钟成交量占比",
        ),
        IntradayFeatureSpec(
            name="i_vol_entropy",
            freq=freq_n,
            pre=(),
            agg=_placeholder_agg("i_vol_entropy"),
            formula="−Σ p ln p / ln(N')，p=vol_i/Σvol（vol>0）；N'<2→null",
            description="成交量时间分布归一化熵（越均匀越接近 1）",
        ),
        IntradayFeatureSpec(
            name="i_amihud",
            freq=freq_n,
            pre=(),
            agg=_placeholder_agg("i_amihud"),
            formula="mean(|r_i|/amount_i)×1e9（amount>0 桶；amount 单位=元）",
            description="Amihud 非流动性（桶级 |收益|/成交额 均值，×1e9）",
        ),
        IntradayFeatureSpec(
            name="i_path_eff",
            freq=freq_n,
            pre=(),
            agg=_placeholder_agg("i_path_eff"),
            formula=(
                "|c_last−o_first|/Σ|Δ_i|；"
                "Δ_1=c_1−o_1、Δ_i=c_i−c_{i−1}；分母≤ε→null"
            ),
            description="价格路径效率（净位移 / 路径总长）",
        ),
        IntradayFeatureSpec(
            name="i_max_ret_share",
            freq=freq_n,
            pre=(),
            agg=_placeholder_agg("i_max_ret_share"),
            formula="max|r| / Σ|r|；Σ≤ε→null",
            description="最大单桶绝对收益占全日绝对收益之和的比重",
        ),
        # ── 涨跌停邻域（A 股特有的离散状态机）──────────────────────────────
        # 前 17 个都是连续路径统计；这三个测「硬约束下的触/封/开」，机制不同。
        # 需调用方传 daily_ref（pre_close + 当日板块限幅），缺则三叶 null。
        IntradayFeatureSpec(
            name="i_limit_up_seal_share",
            freq=freq_n,
            pre=(),
            agg=_placeholder_agg("i_limit_up_seal_share"),
            formula="Σ1[close_i ≥ P_up−ε] / n_valid；P_up=round(pre_close×(1+L),2)",
            description="封板时长占比；未封→0（0 有信息：今天没封过），非 null",
        ),
        IntradayFeatureSpec(
            name="i_limit_up_open_count",
            freq=freq_n,
            pre=(),
            agg=_placeholder_agg("i_limit_up_open_count"),
            formula="Σ_{i≥2} 1[seal_{i-1}=1 ∧ seal_i=0]（不除以 N；次数截面可比）",
            description="开板次数（封住后又打开）；未封→0",
        ),
        IntradayFeatureSpec(
            name="i_limit_up_first_touch",
            freq=freq_n,
            pre=(),
            agg=_placeholder_agg("i_limit_up_first_touch"),
            formula="min{i: high_i ≥ P_up−ε} / n_bars ∈ (0,1]；全日未触→1.0",
            description=(
                "首次触板的相对时刻（越小越早）；全日未触填 1.0 而非 null"
                "——否则截面 95%+ null，rank/IC 塌成少数触板票的子样本游戏"
            ),
        ),
    ]
    assert len(specs) == 20
    _ = _EPS  # 文档常量，计算在 engine
    return specs
