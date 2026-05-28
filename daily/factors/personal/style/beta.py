"""Beta 风格因子：重用 daily.factors.personal.daily.beta 的 Beta60D。"""

from daily.factors.personal.daily.beta import Beta60D as BetaStyle  # noqa: F401

# 注册别名，供 get_factor("beta_style") 使用
# 实际使用 Beta60D 实例（已在 daily.beta 模块实例化并注册）
