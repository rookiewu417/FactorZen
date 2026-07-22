"""Web Dashboard 页面(服务端渲染 Jinja2 + ECharts)。

单页总览:各域产物计数 + 最近 run 列表 + live 净值曲线。纯读 ArtifactIndex,零侵入。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from factorzen.server.artifacts import DOMAINS, ArtifactIndex

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _pick_nav(
    idx: ArtifactIndex, summary: dict,
) -> tuple[list[tuple[str, float]], str | None]:
    """按优先序取第一个「有 run 且 nav_series 非空」的域。

    优先：execution → combine_backtests → sim。找不到 → 空列表 + None。
    """
    for domain in ("execution", "combine_backtests", "sim"):
        runs = summary.get(domain) or []
        if not runs:
            continue
        nav = idx.nav_series(domain, runs[-1]["run_id"])
        if nav:
            return nav, domain
    return [], None


def register_views(app: FastAPI, idx: ArtifactIndex) -> None:
    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> Response:
        summary = {d: idx.list_runs(d) for d in DOMAINS}
        counts = {d: len(runs) for d, runs in summary.items()}
        nav, nav_domain = _pick_nav(idx, summary)
        return _TEMPLATES.TemplateResponse(
            request,
            "dashboard.html",
            {
                "counts": counts,
                "summary": summary,
                "nav": nav,
                "nav_domain": nav_domain,
                "domains": DOMAINS,
            },
        )
