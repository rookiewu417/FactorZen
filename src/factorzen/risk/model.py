"""统一风险模型接口：构建 Barra 多因子风险模型并提供风险预测与分解。"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from factorzen.core.logger import get_logger
from factorzen.risk.covariance import estimate_factor_covariance, estimate_specific_risk
from factorzen.risk.exposures import (
    ExposureMatrix,
    materialize_industry_panel,
    materialize_style_panel,
)
from factorzen.risk.style_factors import STYLE_FACTOR_NAMES

logger = get_logger(__name__)


@dataclass
class RiskModelResult:
    """风险模型构建结果。

    Attributes:
        factor_exposures: 最新一期暴露矩阵 (n_stocks × n_factors)。
        factor_covariance: 因子协方差矩阵 (n_factors × n_factors)。
        specific_risk: 特质风险向量（标准差），shape (n_stocks,)。
        factor_returns: 因子收益时间序列 DataFrame。
        r_squared: 截面回归平均 R²。
        factor_names: 因子名称列表。
        n_dropped_dates: 因因子集不一致等原因被跳过的交易日数（退化可见性）。
        n_valid_dates: 成功完成截面回归的交易日数。
        n_factor_mismatch: 因子集与全局固定集不一致被跳过的日数（应≈0，W2 后）。
    """

    factor_exposures: ExposureMatrix = field(
        default_factory=ExposureMatrix
    )
    factor_covariance: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    specific_risk: np.ndarray = field(default_factory=lambda: np.empty(0))
    factor_returns: pl.DataFrame = field(default_factory=pl.DataFrame)
    r_squared: float = 0.0
    factor_names: list[str] = field(default_factory=list)
    # 因因子集与固定全集不一致而被跳过的交易日数（>0 须醒目暴露，不许静默）。
    n_dropped_dates: int = 0
    n_valid_dates: int = 0
    n_factor_mismatch: int = 0


class RiskModel:
    """Barra 多因子风险模型。

    通过截面回归估计因子收益，再从因子收益时间序列估计因子协方差和特质风险。

    Usage::

        model = RiskModel(cov_half_life=90, nw_lags=2, spec_half_life=90, spec_shrinkage=0.3)
        result = model.build(daily_data, daily_basic, stocks, "20250101", "20250630")
        total_risk = model.predict_risk(weights, result)
        decomp = model.decompose_risk(weights, result)
    """

    def __init__(
        self,
        cov_half_life: int = 90,
        nw_lags: int = 2,
        spec_half_life: int = 90,
        spec_shrinkage: float = 0.3,
        periods_per_year: int = 252,
    ) -> None:
        """初始化风险模型参数。

        Args:
            cov_half_life: 因子协方差指数加权半衰期。
            nw_lags: Newey-West 自相关修正滞后阶数。
            spec_half_life: 特质风险指数加权半衰期。
            spec_shrinkage: 特质风险贝叶斯收缩强度。
            periods_per_year: 年化周期数（A 股日频 252，crypto 日频 365）。
        """
        self.cov_half_life = cov_half_life
        self.nw_lags = nw_lags
        self.spec_half_life = spec_half_life
        self.spec_shrinkage = spec_shrinkage
        self.periods_per_year = periods_per_year

    def build(
        self,
        daily_data: pl.DataFrame,
        daily_basic: pl.DataFrame,
        stocks: pl.DataFrame,
        start_date: str,
        end_date: str,
        style_registry: dict | None = None,
        style_names: list[str] | None = None,
        ret_col: str = "pct_chg",
        ret_is_pct: bool = True,
        *,
        style_panel: pl.DataFrame | None = None,
        industry_panel: pl.DataFrame | None = None,
        industry_names: list[str] | None = None,
    ) -> RiskModelResult:
        """构建风险模型。

        流程：
        1. 获取日期区间内所有交易日
        2. **一次**物化风格面板 + 行业面板（全窗并集行业列，缺列 0）
        3. 每个交易日切片暴露，运行截面回归: ret_i = X_i @ f + eps_i
        4. 收集因子收益 f_t 序列
        5. 估计因子协方差矩阵
        6. 估计特质风险

        Args:
            daily_data: 日线行情 DataFrame。
            daily_basic: 每日估值指标 DataFrame。
            stocks: 股票基本信息 DataFrame。
            start_date: 起始日期 "YYYYMMDD" 或 "YYYY-MM-DD"。
            end_date: 截止日期 "YYYYMMDD" 或 "YYYY-MM-DD"。
            style_panel: 预物化风格面板（research 跨调仓复用，须已按目标 universe 标准化）。
            industry_panel / industry_names: 预物化行业面板或固定行业全集。

        Returns:
            RiskModelResult
        """
        # ── 1. 解析日期 ──────────────────────────────────────────────────────
        start_dt = _parse_date(start_date)
        end_dt = _parse_date(end_date)
        names = STYLE_FACTOR_NAMES if style_names is None else style_names

        # ── 2. 获取交易日列表 ────────────────────────────────────────────────
        if daily_data["trade_date"].dtype == pl.Date:
            trade_dates = (
                daily_data.filter(
                    (pl.col("trade_date") >= start_dt) & (pl.col("trade_date") <= end_dt)
                )
                .select("trade_date")
                .unique()
                .sort("trade_date")["trade_date"]
                .to_list()
            )
        else:
            trade_dates = (
                daily_data.select("trade_date")
                .unique()
                .sort("trade_date")["trade_date"]
                .to_list()
            )

        if not trade_dates:
            logger.warning(f"日期区间 [{start_date}, {end_date}] 内无交易日")
            return RiskModelResult()

        logger.info(f"风险模型构建：{len(trade_dates)} 个交易日 [{start_date} ~ {end_date}]")

        # ── 3. 一次物化风格 + 行业（W1/W2）──────────────────────────────────
        if style_panel is None:
            style_panel = materialize_style_panel(
                daily_data, daily_basic, style_registry, names, standardize=True
            )
        # 风格列：面板中实际存在的（全程空的因子不会出现）
        style_cols = [n for n in names if n in style_panel.columns]

        if industry_panel is None:
            industry_panel, ind_cols = materialize_industry_panel(
                stocks, trade_dates, industry_names=industry_names
            )
        else:
            ind_cols = (
                _normalize_ind_cols(industry_names)
                if industry_names is not None
                else [c for c in industry_panel.columns if c.startswith("ind_")]
            )

        # 全局固定因子名 = 风格 + 行业并集（W2：任何日不再因 ind 漂移丢弃）
        factor_names: list[str] = style_cols + ind_cols
        k_factors = len(factor_names)

        # ── 4. 合并暴露面板 + 收益，按日分区（避免 484× filter 全窗）────────
        scale = 100.0 if ret_is_pct else 1.0
        ret_df = daily_data.select(["trade_date", "ts_code", ret_col]).with_columns(
            (pl.col(ret_col) / scale).alias("ret")
        )

        # 风格 fill 0 + 对齐行业
        exp_panel = style_panel
        if style_cols:
            exp_panel = exp_panel.with_columns(
                [pl.col(c).fill_null(0.0) for c in style_cols]
            )
        if ind_cols and industry_panel is not None and not industry_panel.is_empty():
            ind_sel = industry_panel
            for c in ind_cols:
                if c not in ind_sel.columns:
                    ind_sel = ind_sel.with_columns(pl.lit(0.0).alias(c))
            exp_panel = exp_panel.join(
                ind_sel.select(["trade_date", "ts_code", *ind_cols]),
                on=["trade_date", "ts_code"],
                how="left",
            ).with_columns([pl.col(c).fill_null(0.0) for c in ind_cols])
        elif ind_cols:
            exp_panel = exp_panel.with_columns([pl.lit(0.0).alias(c) for c in ind_cols])

        # 只保留回归窗 + 内连接收益
        exp_panel = exp_panel.filter(
            (pl.col("trade_date") >= start_dt) & (pl.col("trade_date") <= end_dt)
        )
        # 丢掉首风格全 null 的行（style 已 fill 0，用是否在 style_panel 原始有值不太方便；
        # 至少要求 ts_code 非空且有收益）
        joined = exp_panel.join(
            ret_df.select(["trade_date", "ts_code", "ret"]),
            on=["trade_date", "ts_code"],
            how="inner",
        ).filter(pl.col("ret").is_not_null() & pl.col("ret").is_finite())

        # 按日分区：dict[date] -> DataFrame 切片（一次 partition，O(N)）
        day_tables: dict[dt.date, pl.DataFrame] = {}
        if not joined.is_empty():
            for td, grp in joined.group_by("trade_date", maintain_order=True):
                # group_by key: polars 返回 (date,) 或 date 视版本
                key = td[0] if isinstance(td, tuple) else td
                if isinstance(key, dt.datetime):
                    key = key.date()
                day_tables[key] = grp

        # ── 5. 逐日截面回归（numpy lstsq，避免 statsmodels 开销）────────────
        factor_return_rows: list[dict] = []
        residual_dict: dict[str, list[tuple[dt.date, float]]] = {}
        r_squared_list: list[float] = []
        last_exposure: ExposureMatrix | None = None
        n_factor_mismatch = 0
        n_skipped_other = 0

        for trade_date_val in trade_dates:
            if isinstance(trade_date_val, dt.datetime):
                d_key = trade_date_val.date()
            elif isinstance(trade_date_val, dt.date):
                d_key = trade_date_val
            else:
                d_key = _parse_date(str(trade_date_val))

            day = day_tables.get(d_key)
            if day is None or day.is_empty():
                n_skipped_other += 1
                continue

            codes = day["ts_code"].to_list()
            y = day["ret"].to_numpy().astype(np.float64)
            # 因子矩阵：按固定 factor_names 列序
            X_cols = [day[c].to_numpy().astype(np.float64) for c in factor_names]
            X = np.column_stack(X_cols) if X_cols else np.empty((len(codes), 0))

            if len(codes) < k_factors + 1 or k_factors == 0:
                n_skipped_other += 1
                continue

            try:
                f_t, eps_t, r2 = _ols_numpy(y, X)
            except Exception as e:
                logger.warning(f"截面回归失败 ({trade_date_val}): {e}")
                n_skipped_other += 1
                continue

            exposure = ExposureMatrix(
                codes=list(codes),
                factor_names=list(factor_names),
                matrix=X,
            )

            row_dict: dict[str, object] = {"trade_date": trade_date_val}
            for i, name in enumerate(factor_names):
                row_dict[name] = float(f_t[i])
            factor_return_rows.append(row_dict)

            for i, code in enumerate(codes):
                residual_dict.setdefault(code, []).append((trade_date_val, float(eps_t[i])))

            r_squared_list.append(r2)
            last_exposure = exposure

        n_dropped = n_factor_mismatch  # 兼容字段：因子集不一致丢日
        n_valid = len(factor_return_rows)

        # ── 6. 处理结果 ─────────────────────────────────────────────────────
        if not factor_return_rows or last_exposure is None:
            logger.warning("风险模型构建失败：无有效截面回归结果")
            return RiskModelResult(
                n_dropped_dates=n_dropped,
                n_valid_dates=0,
                n_factor_mismatch=n_factor_mismatch,
            )

        factor_returns_df = pl.DataFrame(factor_return_rows)
        fr_matrix = np.column_stack(
            [factor_returns_df[name].to_numpy().astype(np.float64) for name in factor_names]
        )

        # ── 7. 因子协方差估计 ────────────────────────────────────────────────
        factor_cov = estimate_factor_covariance(
            fr_matrix, half_life=self.cov_half_life, nw_lags=self.nw_lags
        )

        # ── 8. 特质风险估计 ──────────────────────────────────────────────────
        last_codes = last_exposure.codes
        valid_trade_dates = [row["trade_date"] for row in factor_return_rows]
        N_last = len(last_codes)

        residual_matrix = _build_residual_matrix(residual_dict, last_codes, valid_trade_dates)

        col_means = np.nanmean(residual_matrix, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        for j in range(N_last):
            nan_mask = np.isnan(residual_matrix[:, j])
            residual_matrix[nan_mask, j] = col_means[j]

        spec_risk = estimate_specific_risk(
            residual_matrix, half_life=self.spec_half_life, shrinkage=self.spec_shrinkage
        )

        # ── 9. 汇总 + 退化可见性 ─────────────────────────────────────────────
        avg_r2 = float(np.mean(r_squared_list)) if r_squared_list else 0.0

        if n_factor_mismatch > 0:
            logger.warning(
                f"[n_factor_mismatch={n_factor_mismatch}] 风险模型：{n_factor_mismatch}/"
                f"{len(trade_dates)} 个交易日因因子集与全局固定集不一致被跳过——"
                "W2 后这不应发生；请检查 reindex_exposure / 行业并集逻辑。"
            )
        skipped_total = n_factor_mismatch + n_skipped_other
        if skipped_total > 0 and n_valid < len(trade_dates):
            logger.info(
                f"风险模型有效截面 {n_valid}/{len(trade_dates)} 日"
                f"（factor_mismatch={n_factor_mismatch}, other_skip={n_skipped_other}）"
            )

        return RiskModelResult(
            factor_exposures=last_exposure,
            factor_covariance=factor_cov,
            specific_risk=spec_risk,
            factor_returns=factor_returns_df,
            r_squared=avg_r2,
            factor_names=factor_names,
            n_dropped_dates=n_dropped,
            n_valid_dates=n_valid,
            n_factor_mismatch=n_factor_mismatch,
        )

    def predict_risk(
        self,
        weights: np.ndarray,
        result: RiskModelResult,
    ) -> float:
        """预测组合总风险（年化波动率）。

        σ² = w' X F X' w + w' D² w
        """
        X = result.factor_exposures.matrix
        F = result.factor_covariance
        D = result.specific_risk

        Xw = X.T @ weights
        factor_var = float(Xw @ F @ Xw)
        specific_var = float(np.sum((D * weights) ** 2))

        total_var = factor_var + specific_var
        total_std = np.sqrt(max(total_var, 0.0))
        return float(total_std * np.sqrt(self.periods_per_year))

    def decompose_risk(
        self,
        weights: np.ndarray,
        result: RiskModelResult,
    ) -> dict[str, float]:
        """风险分解：将组合风险拆分为各因子贡献与特质贡献。

        返回口径说明见历史 docstring：total/factor/specific 为标准差口径；
        各因子名为 MCR 份额口径，不可与 specific_risk 直接相加。
        """
        X = result.factor_exposures.matrix
        F = result.factor_covariance
        D = result.specific_risk

        Xw = X.T @ weights
        factor_var = float(Xw @ F @ Xw)
        specific_var = float(np.sum((D * weights) ** 2))

        total_var = factor_var + specific_var
        total_std = np.sqrt(max(total_var, 0.0))

        ann = np.sqrt(self.periods_per_year)
        decomp: dict[str, float] = {
            "total_risk": float(total_std * ann),
            "factor_risk": float(np.sqrt(max(factor_var, 0.0)) * ann),
            "specific_risk": float(np.sqrt(max(specific_var, 0.0)) * ann),
        }

        if total_var > 1e-15:
            F_Xw = F @ Xw
            for i, name in enumerate(result.factor_names):
                var_contrib = float(Xw[i] * F_Xw[i])
                risk_contrib = var_contrib / total_var * total_std * ann
                decomp[name] = float(risk_contrib)
        else:
            for name in result.factor_names:
                decomp[name] = 0.0

        return decomp


def _normalize_ind_cols(industry_names: list[str]) -> list[str]:
    return [n if n.startswith("ind_") else f"ind_{n}" for n in industry_names]


def _ols_numpy(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """截面 OLS：numpy lstsq + 残差 R²（与 statsmodels OLS 数值对齐，快一个数量级）。

    不加截距（行业哑变量已吸收）。
    """
    # rcond=None 用默认截断
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    ss_res = float(np.dot(resid, resid))
    y_mean = float(np.mean(y))
    ss_tot = float(np.dot(y - y_mean, y - y_mean))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    # 数值保护
    if not np.isfinite(r2):
        r2 = 0.0
    return beta.astype(np.float64), resid.astype(np.float64), float(r2)


def _parse_date(date_str: str) -> dt.date:
    """将日期字符串解析为 date 对象。"""
    if "-" in date_str:
        return dt.date.fromisoformat(date_str)
    return dt.datetime.strptime(date_str, "%Y%m%d").date()


def _build_residual_matrix(
    residual_dict: dict[str, list[tuple[dt.date, float]]],
    codes: list[str],
    trade_dates: list[dt.date],
) -> np.ndarray:
    """按真实交易日索引对齐重建残差矩阵。

    ``residual_dict[code]`` 按交易日顺序追加 ``(trade_date, residual)``，但仅在该
    股票当天参与截面回归时才有记录——窗口中途停牌/缺数据会造成中间缺口。
    显式按交易日索引定位，缺失日期保持 NaN。
    """
    date_to_row = {d: i for i, d in enumerate(trade_dates)}
    matrix = np.full((len(trade_dates), len(codes)), np.nan)
    for j, code in enumerate(codes):
        for trade_date_val, resid in residual_dict.get(code, []):
            row = date_to_row.get(trade_date_val)
            if row is not None:
                matrix[row, j] = resid
    return matrix
