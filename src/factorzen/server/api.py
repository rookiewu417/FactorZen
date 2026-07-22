"""FactorZen 只读 REST API。

暴露 workspace 产物(runs 列表 / manifest / NAV 序列 / 因子库 / 运营 / 报告)。
纯读层,零侵入现有 pipeline。
`create_app(workspace_dir)` 便于测试注入 tmp 目录;模块级 `app` 供 uvicorn 启动。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from factorzen.config.settings import WORKSPACE_DIR
from factorzen.server.artifacts import DOMAINS, ArtifactIndex
from factorzen.server.library import MARKETS, FactorLibraryIndex
from factorzen.server.opsview import OpsViewIndex


def create_app(workspace_dir: str | Path | None = None) -> FastAPI:
    root = workspace_dir or WORKSPACE_DIR
    idx = ArtifactIndex(root)
    lib = FactorLibraryIndex(root)
    ops = OpsViewIndex(root)
    app = FastAPI(
        title="FactorZen API",
        description="A 股量化研究平台 · 只读产物 API",
        version="0.1.0",
    )

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "domains": DOMAINS}

    @app.get("/api/overview")
    def overview() -> dict:
        """各域产物计数与最新 run 摘要。"""
        return {"domains": idx.overview()}

    @app.get("/api/runs")
    def runs(domain: str) -> dict:
        if domain not in DOMAINS:
            raise HTTPException(status_code=404, detail=f"未知 domain: {domain}")
        return {"domain": domain, "runs": idx.list_runs(domain)}

    @app.get("/api/runs/{domain}/{run_id}")
    def run_detail(domain: str, run_id: str) -> dict:
        try:
            return idx.run_detail(domain, run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/nav/{domain}/{run_id}")
    def nav(domain: str, run_id: str) -> dict:
        if domain not in DOMAINS:
            raise HTTPException(status_code=404, detail=f"未知 domain: {domain}")
        return {
            "domain": domain,
            "run_id": run_id,
            "nav": idx.nav_series(domain, run_id),
        }

    # ---- 因子库 / 因子资产 ----

    @app.get("/api/library/{market}")
    def library_list(market: str) -> dict:
        if market not in MARKETS:
            raise HTTPException(status_code=404, detail=f"未知 market: {market}")
        return lib.list_factors(market)

    @app.get("/api/library/{market}/track")
    def library_track(
        market: str,
        expression: str = Query(..., description="因子表达式"),
    ) -> dict:
        if market not in MARKETS:
            raise HTTPException(status_code=404, detail=f"未知 market: {market}")
        return lib.forward_track(market, expression)

    @app.get("/api/store/{market}")
    def store_list(market: str) -> dict:
        if market not in MARKETS:
            raise HTTPException(status_code=404, detail=f"未知 market: {market}")
        return lib.list_store(market)

    @app.get("/api/store/{market}/{name}")
    def store_detail(market: str, name: str) -> dict:
        if market not in MARKETS:
            raise HTTPException(status_code=404, detail=f"未知 market: {market}")
        try:
            return lib.store_detail(market, name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # ---- 运营 ----

    @app.get("/api/ops/campaigns")
    def ops_campaigns() -> dict:
        return ops.list_campaigns()

    @app.get("/api/ops/campaigns/{name}/log")
    def ops_campaign_log(
        name: str,
        tail: int = Query(200, ge=1, le=2000),
    ) -> dict:
        try:
            return ops.campaign_log(name, tail=tail)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # ---- 报告 ----

    @app.get("/api/reports")
    def reports_list() -> dict:
        return ops.list_reports()

    @app.get("/api/reports/file")
    def reports_file(path: str = Query(..., description="相对 reports/ 的路径")) -> dict:
        """读取报告文本；超限 413；路径遍历 / 非法扩展名 404。"""
        try:
            return ops.read_report(path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc

    from factorzen.server.views import register_views

    register_views(app, idx)

    # SPA 静态资源:仓库根/webui/dist 存在时挂到 /ui;测试环境无 dist 则静默跳过
    repo_root = Path(__file__).resolve().parents[3]
    ui_dist = repo_root / "webui" / "dist"
    if (ui_dist / "index.html").exists():
        app.mount("/ui", StaticFiles(directory=str(ui_dist), html=True), name="ui")

    return app


app = create_app()
