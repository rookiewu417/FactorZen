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


def register_views(app: FastAPI, idx: ArtifactIndex) -> None:
    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> Response:
        summary = {d: idx.list_runs(d) for d in DOMAINS}
        counts = {d: len(runs) for d, runs in summary.items()}
        execs = summary.get("execution", [])
        nav = idx.nav_series("execution", execs[-1]["run_id"]) if execs else []
        return _TEMPLATES.TemplateResponse(
            request,
            "dashboard.html",
            {"counts": counts, "summary": summary, "nav": nav, "domains": DOMAINS},
        )
