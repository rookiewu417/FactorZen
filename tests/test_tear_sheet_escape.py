"""tear_sheet 外部不可信文本须 HTML 转义。

新版模板不再有 `| safe` 旁路，全部变量走 jinja2 autoescape；
本测试守住该不变量：因子名 / 方向判定 reason / 质量警告中的
HTML 注入内容不得原样出现在渲染结果里。
"""
from __future__ import annotations

from factorzen.reports.tear_sheet import generate_tear_sheet


def test_tear_sheet_escapes_untrusted_text():
    html = generate_tear_sheet(
        "<script>alert('xss')</script>",
        None,
        None,
        None,
        date_range="2024-01-01 ~ 2024-06-30",
        universe="csi300",
        backtest_direction={
            "direction": "reversed",
            "reason": "<img src=x onerror=alert(1)>",
        },
        quality_report={"warnings": ["<script>alert('q')</script>"]},
    )
    assert "<script>" not in html, "因子名/警告中的 <script> 应被转义"
    assert "&lt;script&gt;" in html, "转义后应出现 &lt;script&gt;"
    assert "<img src=x" not in html, "方向 reason 中的 <img onerror> 应被转义"
