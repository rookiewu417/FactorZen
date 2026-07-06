"""统一风险模型接口：构建 Barra 多因子风险模型并提供风险预测与分解。"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import statsmodels.api as sm

from factorzen.core.logger import get_logger
from factorzen.risk.covariance import estimate_factor_covariance, estimate_specific_risk
from factorzen.risk.exposures import ExposureMatrix, compute_exposures

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
    """

    factor_exposures: ExposureMatrix = field(
        default_factory=ExposureMatrix
    )
    factor_covariance: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    specific_risk: np.ndarray = field(default_factory=lambda: np.empty(0))
    factor_returns: pl.DataFrame = field(default_factory=pl.DataFrame)
    r_squared: float = 0.0
    factor_names: list[str] = field(default_factory=list)
    # 因因子集与首个有效截面不一致而被跳过的交易日数（>0 通常表示窗口早期滚动风格
    # 因子数据不足、模型退化；见 build 内告警与 pipelines.risk_build.load_risk_inputs）。
    n_dropped_dates: int = 0


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
    ) -> RiskModelResult:
        """构建风险模型。

        流程：
        1. 获取日期区间内所有交易日
        2. 每个交易日计算暴露矩阵，运行截面回归: ret_i = X_i @ f + eps_i
        3. 收集因子收益 f_t 序列
        4. 估计因子协方差矩阵
        5. 估计特质风险

        Args:
            daily_data: 日线行情 DataFrame。
            daily_basic: 每日估值指标 DataFrame。
            stocks: 股票基本信息 DataFrame。
            start_date: 起始日期 "YYYYMMDD" 或 "YYYY-MM-DD"。
            end_date: 截止日期 "YYYYMMDD" 或 "YYYY-MM-DD"。

        Returns:
            RiskModelResult
        """
        # ── 1. 解析日期 ──────────────────────────────────────────────────────
        start_dt = _parse_date(start_date)
        end_dt = _parse_date(end_date)

        # ── 2. 获取交易日列表 ────────────────────────────────────────────────
        # 从 daily_data 中提取在日期区间内的唯一交易日
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

        # ── 3. 准备收益率数据 ────────────────────────────────────────────────
        scale = 100.0 if ret_is_pct else 1.0  # A 股 pct_chg 是百分比；crypto ret_1d 已是小数
        ret_df = daily_data.select(["trade_date", "ts_code", ret_col]).with_columns(
            (pl.col(ret_col) / scale).alias("ret")
        )

        # ── 4. 逐日截面回归 ─────────────────────────────────────────────────
        factor_return_rows: list[dict] = []
        # ts_code -> [(trade_date, residual), ...]：连同交易日一起记录，重建矩阵时
        # 按真实交易日对齐（而非"取最后 N 个右对齐"），避免窗口中途缺口错位。
        residual_dict: dict[str, list[tuple[dt.date, float]]] = {}
        r_squared_list: list[float] = []
        last_exposure: ExposureMatrix | None = None
        factor_names: list[str] | None = None
        n_factor_mismatch = 0  # 因子集与首个有效截面不一致被跳过的交易日数（退化可见性）

        for trade_date_val in trade_dates:
            # 计算暴露
            exposure = compute_exposures(
                daily_data, daily_basic, stocks, trade_date_val, style_registry, style_names
            )

            if exposure.n_stocks == 0 or exposure.n_factors == 0:
                continue

            # 如果是第一次，记住因子名
            if factor_names is None:
                factor_names = exposure.factor_names

            # 获取当日收益率
            if isinstance(trade_date_val, dt.date):
                day_ret = ret_df.filter(pl.col("trade_date") == trade_date_val)
            else:
                day_ret = ret_df.filter(pl.col("trade_date") == pl.lit(trade_date_val))

            if day_ret.is_empty():
                continue

            # 匹配暴露矩阵中的股票
            code_to_idx = {c: i for i, c in enumerate(exposure.codes)}
            matched_codes: list[str] = []
            matched_rets: list[float] = []
            matched_rows: list[int] = []

            for row in day_ret.iter_rows(named=True):
                code = row["ts_code"]
                ret_val = row["ret"]
                if code in code_to_idx and ret_val is not None and np.isfinite(ret_val):
                    matched_codes.append(code)
                    matched_rets.append(ret_val)
                    matched_rows.append(code_to_idx[code])

            if len(matched_codes) < exposure.n_factors + 1:
                # 样本数不足以运行回归
                continue

            y = np.array(matched_rets)
            X = exposure.matrix[matched_rows, :]

            # 确保 factor_names 与当前暴露一致。比较因子**名字序列**而非仅列数——
            # 行业成分漂移会使某日因子集「名字不同但个数相同」（如 ind_B 调出、ind_C
            # 调入），只比列数会让该日回归系数被错标到别的因子名下，污染因子收益→协方差→
            # 归因。因子数不同通常源于窗口早期滚动风格因子数据不足。两种不一致都跳过并计数。
            if exposure.factor_names != factor_names:
                n_factor_mismatch += 1
                continue

            # ── 截面 OLS 回归（不加截距，行业哑变量已包含）──────────────────
            try:
                model = sm.OLS(y, X).fit()
                f_t = model.params  # 因子收益 shape (K,)
                eps_t = model.resid  # 残差 shape (N_matched,)
                r2 = float(model.rsquared)
            except Exception as e:
                logger.warning(f"截面回归失败 ({trade_date_val}): {e}")
                continue

            # 记录因子收益
            row_dict: dict[str, object] = {"trade_date": trade_date_val}
            for i, name in enumerate(factor_names):
                row_dict[name] = float(f_t[i])
            factor_return_rows.append(row_dict)

            # 记录残差（连同交易日一起追加）
            for i, code in enumerate(matched_codes):
                residual_dict.setdefault(code, []).append((trade_date_val, float(eps_t[i])))

            r_squared_list.append(r2)
            last_exposure = exposure

        # ── 5. 处理结果 ─────────────────────────────────────────────────────
        if not factor_return_rows or factor_names is None or last_exposure is None:
            logger.warning("风险模型构建失败：无有效截面回归结果")
            return RiskModelResult()

        # 因子收益 DataFrame
        factor_returns_df = pl.DataFrame(factor_return_rows)

        # 因子收益矩阵 shape (T_valid, K)
        fr_matrix = np.column_stack(
            [factor_returns_df[name].to_numpy().astype(np.float64) for name in factor_names]
        )

        # ── 6. 因子协方差估计 ────────────────────────────────────────────────
        factor_cov = estimate_factor_covariance(
            fr_matrix, half_life=self.cov_half_life, nw_lags=self.nw_lags
        )

        # ── 7. 特质风险估计 ──────────────────────────────────────────────────
        # 构建残差矩阵 (T_valid, N_last)：按真实交易日索引对齐放置，而非
        # "取最后 N 个右对齐"（窗口中途缺口会被错位推后，见 _build_residual_matrix）
        last_codes = last_exposure.codes
        valid_trade_dates = [row["trade_date"] for row in factor_return_rows]
        N_last = len(last_codes)

        residual_matrix = _build_residual_matrix(residual_dict, last_codes, valid_trade_dates)

        # 用列均值填充 NaN
        col_means = np.nanmean(residual_matrix, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        for j in range(N_last):
            nan_mask = np.isnan(residual_matrix[:, j])
            residual_matrix[nan_mask, j] = col_means[j]

        spec_risk = estimate_specific_risk(
            residual_matrix, half_life=self.spec_half_life, shrinkage=self.spec_shrinkage
        )

        # ── 8. 汇总 ─────────────────────────────────────────────────────────
        avg_r2 = float(np.mean(r_squared_list)) if r_squared_list else 0.0

        if n_factor_mismatch > 0:
            logger.warning(
                f"风险模型：{n_factor_mismatch}/{len(trade_dates)} 个交易日因因子集与首个"
                "有效截面不一致被跳过——通常是窗口早期滚动风格因子（momentum/volatility/"
                "growth 等 252/60 日窗）数据不足所致；请确保输入 daily/daily_basic 含足够"
                " lookback 历史，否则因子协方差仅由少数退化截面估计。"
            )

        return RiskModelResult(
            factor_exposures=last_exposure,
            factor_covariance=factor_cov,
            specific_risk=spec_risk,
            factor_returns=factor_returns_df,
            r_squared=avg_r2,
            factor_names=factor_names,
            n_dropped_dates=n_factor_mismatch,
        )

    def predict_risk(
        self,
        weights: np.ndarray,
        result: RiskModelResult,
    ) -> float:
        """预测组合总风险（年化波动率）。

        σ² = w' X F X' w + w' D² w

        其中：
        - X: 因子暴露 (n × k)
        - F: 因子协方差 (k × k)
        - D: 特质风险对角阵 (n × n)

        Args:
            weights: 组合权重向量，shape (n_stocks,)。
            result: build() 返回的 RiskModelResult。

        Returns:
            组合年化波动率（日度 σ × √252）。
        """
        X = result.factor_exposures.matrix
        F = result.factor_covariance
        D = result.specific_risk

        # 因子风险
        Xw = X.T @ weights  # shape (K,)
        factor_var = float(Xw @ F @ Xw)

        # 特质风险
        specific_var = float(np.sum((D * weights) ** 2))

        total_var = factor_var + specific_var
        total_std = np.sqrt(max(total_var, 0.0))

        # 年化（A 股日频 √252，crypto √365）
        return float(total_std * np.sqrt(self.periods_per_year))

    def decompose_risk(
        self,
        weights: np.ndarray,
        result: RiskModelResult,
    ) -> dict[str, float]:
        """风险分解：将组合风险拆分为各因子贡献与特质贡献。

        Args:
            weights: 组合权重向量，shape (n_stocks,)。
            result: build() 返回的 RiskModelResult。

        Returns:
            dict，包含两套**不同口径、不可混用相加**的量：

            - "total_risk"/"factor_risk"/"specific_risk"：各自独立的标准差口径
              （``sqrt(var)*sqrt(252)``）。三者满足方差可加（
              ``factor_var + specific_var = total_var``），但标准差本身不可加——
              ``factor_risk + specific_risk != total_risk`` 是预期行为，不是 bug。
            - 各因子名称（如 "size"）：该因子按边际贡献（MCR）分摊到 total_risk 的
              份额，即 ``Xw_k*(F@Xw)_k / total_var * total_std * sqrt(252)``。这套值
              **彼此可加**，Σ(各因子份额) = ``factor_risk**2 / total_risk``（而非
              ``factor_risk`` 本身）——这是加权 MCR 分解，不是把 factor_risk 欧拉
              分解到各因子（详见 tests/test_risk_model.py 对应测试的注释）。
              "specific_risk" 键**没有**对应的份额口径镜像字段；下游消费方
              （如 attribution/risk_attribution.py）如需把「可加的各因子份额」和
              「不可加的 specific_risk」放进同一个结果里，必须自己注明两者口径不同，
              不能直接相加/相除。
        """
        X = result.factor_exposures.matrix
        F = result.factor_covariance
        D = result.specific_risk

        # 组合因子暴露
        Xw = X.T @ weights  # shape (K,)

        # 因子方差
        factor_var = float(Xw @ F @ Xw)

        # 特质方差
        specific_var = float(np.sum((D * weights) ** 2))

        total_var = factor_var + specific_var
        total_std = np.sqrt(max(total_var, 0.0))

        ann = np.sqrt(self.periods_per_year)  # A 股 √252，crypto √365
        decomp: dict[str, float] = {
            "total_risk": float(total_std * ann),
            "factor_risk": float(np.sqrt(max(factor_var, 0.0)) * ann),
            "specific_risk": float(np.sqrt(max(specific_var, 0.0)) * ann),
        }

        # 边际因子贡献：MCR_k = (F @ Xw)_k * Xw_k / total_var
        if total_var > 1e-15:
            F_Xw = F @ Xw
            for i, name in enumerate(result.factor_names):
                # 因子 k 的方差贡献 = Xw_k * (F @ Xw)_k
                var_contrib = float(Xw[i] * F_Xw[i])
                risk_contrib = var_contrib / total_var * total_std * ann
                decomp[name] = float(risk_contrib)
        else:
            for name in result.factor_names:
                decomp[name] = 0.0

        return decomp


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
    股票当天参与截面回归时才有记录——窗口中途停牌/缺数据会造成中间缺口（非仅
    起点）。若简单地"取最后 N 个右对齐"拼接（历史实现的 bug），缺口前的残差会
    被整体推后一位，错位进入 EWMA 衰减下权重更高的"近因"位置，而真正的缺口
    （应为 NaN）反而被推到矩阵最前面、被列均值填充逻辑悄悄抹平。这里改为显式
    按交易日索引定位，缺失日期保持 NaN。

    Args:
        residual_dict: ts_code -> [(trade_date, residual), ...]。
        codes: 矩阵列顺序对应的股票代码列表。
        trade_dates: 矩阵行顺序对应的交易日列表（升序，长度 = 矩阵行数）。

    Returns:
        残差矩阵，shape (len(trade_dates), len(codes))；该股票当天缺席的位置
        显式为 NaN（按真实交易日对齐，不做位置无关的右对齐拼接）。
    """
    date_to_row = {d: i for i, d in enumerate(trade_dates)}
    matrix = np.full((len(trade_dates), len(codes)), np.nan)
    for j, code in enumerate(codes):
        for trade_date_val, resid in residual_dict.get(code, []):
            row = date_to_row.get(trade_date_val)
            if row is not None:
                matrix[row, j] = resid
    return matrix
