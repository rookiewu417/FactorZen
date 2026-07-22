"""报告图表基建：CJK 字体、Figure→base64、日期轴。

具体业务图表在 ``signal_report`` / ``trading_report`` 中实现，本模块不放图。
"""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # 非交互后端，不弹窗
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def _register_cjk_fonts() -> None:
    """发现并注册常见 CJK 字体，让未预装中文字体的环境也能渲染报告图表。

    重点是 matplotlib 默认不扫描的 WSL 挂载 Windows 字体目录（``/mnt/*/Windows/Fonts``，
    含 Microsoft YaHei / SimHei），以及用户字体目录。任意一项失败都静默，绝不影响 import。
    """
    import contextlib
    import glob
    from pathlib import Path

    from matplotlib import font_manager

    candidates: list[str] = []
    # WSL：Windows 自带中文字体（任意盘符）
    for pat in ("msyh*.tt?", "simhei.ttf", "simsun.tt?"):
        candidates += glob.glob(f"/mnt/*/Windows/Fonts/{pat}")
    # 用户字体目录（含通过 setup 装入的 Noto Sans CJK）
    for d in (Path.home() / ".local/share/fonts", Path.home() / ".fonts"):
        if d.exists():
            for ext in ("*.ttf", "*.ttc", "*.otf"):
                candidates += [str(p) for p in d.glob(ext)]
    for fp in candidates:
        with contextlib.suppress(Exception):
            font_manager.fontManager.addfont(fp)


_register_cjk_fonts()

# 中文字体支持：按优先级回退（Windows / Linux 常见 CJK 字体），末位回落 DejaVu Sans。
# 必须设置 font.sans-serif 列表（而非 font.family 单值），否则 matplotlib 无法
# 在缺失首选字体时自动尝试后续字体，会把中文渲染成豆腐块并刷屏 glyph 缺失告警。
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "Noto Sans CJK JP",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False


def _fig_to_base64(fig: plt.Figure) -> str:
    """将 matplotlib Figure 转为 base64 PNG 字符串。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _with_plot_dates(frame: Any, date_col: str = "trade_date") -> tuple[Any, str, bool]:
    """Return a frame with a datetime x-axis column when dates can be parsed."""
    if date_col not in frame.columns:
        out = frame.copy()
        out["_plot_index"] = np.arange(len(out))
        return out, "_plot_index", False
    try:
        pandas = __import__("pandas")
        parsed = pandas.to_datetime(frame[date_col], errors="coerce")
    except Exception:
        return frame, date_col, False
    if not parsed.notna().any():
        return frame, date_col, False
    out = frame.copy()
    out["_plot_date"] = parsed
    return out, "_plot_date", True


def _format_sparse_x_axis(ax: plt.Axes, *, is_date_axis: bool, max_ticks: int = 7) -> None:
    """Keep report chart x-axis labels readable on dense daily samples."""
    if is_date_axis:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=max(3, int(max_ticks)))
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    else:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=max(3, int(max_ticks))))
    ax.tick_params(axis="x", labelsize=8)
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment("center")
