"""市场抽象地基：7 个 Port 抽象接口 + MarketProfile。

设计原则（Ports & Adapters）：
- 共享引擎只依赖每个 Port 的**核心方法**，不感知具体市场。
- 市场特有数据（A 股财报、crypto funding/OI）走各 adapter 的**扩展方法**，
  只被该市场自己的因子/风险代码内部调用，引擎不依赖。
- 列名沿用中性约定：``ts_code``（标的）/ ``trade_date``（日期），跨市场统一。

每个新市场（crypto/美股/商品期货……）实现这 7 个 Port，打包成一个
``MarketProfile``，经 ``registry`` 注册后由 ``markets.get(name)`` 取用。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import polars as pl


class DataProvider(ABC):
    """行情/元数据接入。引擎依赖 ``fetch_bars`` + ``fetch_symbol_meta``；
    市场特有拉数（A 股 daily_basic/finance、crypto funding/OI）作为子类扩展方法。"""

    @abstractmethod
    def fetch_bars(
        self, symbols: list[str] | None, start: str, end: str, freq: str = "daily"
    ) -> pl.DataFrame:
        """返回 OHLCV bar，列至少含 ``ts_code, trade_date, open, high, low, close, vol, amount``。"""

    @abstractmethod
    def fetch_symbol_meta(self) -> pl.DataFrame:
        """返回标的元数据，列至少含 ``ts_code``（可含 ``name, list_date``）。"""


class Calendar(ABC):
    """交易日历。把 A 股硬编的 252/交易日概念抽象掉，crypto 用 24/7 连续日历。"""

    @abstractmethod
    def sessions(self, start: str, end: str) -> list[date]:
        """[start, end]（``YYYYMMDD``）内所有交易日期。"""

    @abstractmethod
    def is_session(self, d: date | str) -> bool:
        """*d* 是否为交易日。"""

    @abstractmethod
    def next_session(self, d: date | str, n: int = 1) -> date:
        """*d* 之后第 *n* 个交易日。"""

    @abstractmethod
    def prev_session(self, d: date | str, n: int = 1) -> date:
        """*d* 之前第 *n* 个交易日。"""

    @abstractmethod
    def periods_per_year(self, freq: str = "daily") -> float:
        """年化周期数（A 股日频 252，crypto 日频 365）。替代硬编 TRADING_DAYS_PER_YEAR。"""


class TradingRules(ABC):
    """交易约束与撮合口径。A 股：涨跌停+停牌+T+1+long-only；crypto：近空约束+T+0+可空。"""

    @property
    @abstractmethod
    def allow_short(self) -> bool:
        """是否允许做空（A 股 False，crypto perps True）。"""

    @property
    @abstractmethod
    def settlement_lag(self) -> int:
        """结算滞后 bar 数（A 股 T+1=1，crypto T+0=0）。"""

    @property
    @abstractmethod
    def execution_price_col(self) -> str:
        """撮合价列名（A 股 t+1 开盘 ``open``，crypto next-bar ``close``）。"""

    @abstractmethod
    def tradable_mask(self, bars: pl.DataFrame, side: str) -> pl.Series:
        """逐行返回该 bar 该方向（``buy``/``sell``）是否可交易的布尔 Series。"""


class CostModel(ABC):
    """交易与持有成本。A 股：佣金+滑点+卖出印花税+融券；crypto：maker/taker+滑点+funding。"""

    @abstractmethod
    def trade_cost(self, side: str, notional: float, is_maker: bool = False) -> float:
        """单笔交易成本（``side`` in {buy, sell}，``notional`` 成交额）。"""

    @abstractmethod
    def carry_cost(
        self, position_value: float, periods: int, funding_rate: float = 0.0
    ) -> float:
        """持有成本。crypto：funding 逐期计提（多头付正、空头收）；A 股：融券利息。

        ``position_value`` 带符号（多正空负）。返回正=成本、负=收入。
        """


class Universe(ABC):
    """标的池与基准。A 股：指数成分+ST/次新过滤；crypto：成交额 Top-N+流动性过滤。"""

    @abstractmethod
    def snapshot(self, d: date | str) -> list[str]:
        """截至 *d* 的可交易标的列表（``ts_code``）。"""

    @abstractmethod
    def benchmark(self, start: str, end: str) -> pl.DataFrame:
        """基准净值序列，列含 ``trade_date, close``。"""


class FactorSet(ABC):
    """因子叶子集与派生列。A 股：价量+daily_basic；crypto：价量+funding+OI。"""

    @abstractmethod
    def leaf_features(self) -> dict[str, str]:
        """discovery 叶子名 → 求值表列名映射。"""

    @abstractmethod
    def basic_features(self) -> set[str]:
        """需 join 的非价量叶子集合（触发额外数据拉取）。"""

    @abstractmethod
    def derived_columns(self, bars: pl.DataFrame) -> pl.DataFrame:
        """给 bars 追加派生列（vwap/log_vol/ret_1d 等）。"""


class RiskModel(ABC):
    """风险因子集与分组。A 股：Barra 风格+申万行业；crypto：BTC-beta 等+sector。

    本抽象在 MC0 定义，crypto/A 股实现延后到 MC3（本期 MarketProfile.risk 传 None）。
    """

    @abstractmethod
    def style_factors(self) -> dict:
        """风格因子注册表。"""

    @abstractmethod
    def sector_classification(self, symbols: list[str], d: date | str) -> pl.DataFrame:
        """标的的行业/板块 one-hot 归属。"""


@dataclass(frozen=True)
class MarketProfile:
    """一个市场的完整能力打包。引擎消费此对象，不直接依赖任何具体市场实现。"""

    name: str
    quote_currency: str
    base_freq: str
    provider: DataProvider
    calendar: Calendar
    rules: TradingRules
    costs: CostModel
    universe: Universe
    factors: FactorSet
    risk: RiskModel | None = None
