"""FactorZen REST API。

暴露 workspace 产物(runs 列表 / manifest / NAV 序列 / 因子库 / 运营 / 报告)、
受控文件管理(列目录 / 读 / 写文本 / 删 / 浏览器直开)、后台任务中心与 CLI schema。
产物层仍为零侵入只读；files 编辑/删除与 jobs 为 workspace 内受控写接口。
`create_app(workspace_dir)` 便于测试注入 tmp 目录;模块级 `app` 供 uvicorn 启动。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from factorzen.config.settings import ROOT, WORKSPACE_DIR
from factorzen.server.artifacts import DOMAINS, ArtifactIndex
from factorzen.server.cli_schema import get_cli_schema
from factorzen.server.files import FileManager
from factorzen.server.jobs import JobManager
from factorzen.server.library import MARKETS, FactorLibraryIndex
from factorzen.server.opsview import OpsViewIndex


class FileWriteBody(BaseModel):
    """PUT /api/files/content 请求体。"""

    path: str = Field(..., description="相对 workspace 根的路径")
    content: str = Field(..., description="文本内容")


class JobSubmitBody(BaseModel):
    """POST /api/jobs 请求体。"""

    kind: str = Field(..., description="cli | script")
    argv: list[str] = Field(default_factory=list, description="命令参数")
    title: str = Field(..., description="任务标题")


def create_app(workspace_dir: str | Path | None = None) -> FastAPI:
    root = Path(workspace_dir) if workspace_dir is not None else Path(WORKSPACE_DIR)
    idx = ArtifactIndex(root)
    lib = FactorLibraryIndex(root)
    ops = OpsViewIndex(root)
    files = FileManager(root)
    jobs = JobManager(
        root / "_ops" / "webui_jobs",
        workspace_dir=root,
        project_root=ROOT,
    )
    app = FastAPI(
        title="FactorZen API",
        description="A 股量化研究平台 · 产物 API + 受控文件管理 + 任务中心",
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

    # ---- 文件管理（workspace 受控读写） ----

    @app.get("/api/files")
    def files_list(path: str = Query("", description="相对 workspace 根；空=根")) -> dict:
        """列目录：dirs + files。"""
        try:
            return files.list_dir(path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/files/raw")
    def files_raw(
        path: str = Query(..., description="相对 workspace 根的文件路径"),
    ) -> FileResponse:
        """浏览器直开：FileResponse 自动 content-type；html 直接渲染。"""
        try:
            target = files.raw_path(path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        # inline：html/图片在浏览器直接渲染，而非强制下载
        return FileResponse(
            path=str(target),
            filename=target.name,
            content_disposition_type="inline",
        )

    @app.get("/api/files/content")
    def files_content(
        path: str = Query(..., description="相对 workspace 根的文件路径"),
    ) -> dict:
        """读文件：text / parquet / binary。"""
        try:
            return files.read_content(path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.put("/api/files/content")
    def files_write(body: FileWriteBody) -> dict:
        """覆盖写文本；非文本扩展名 400；父目录不存在 404。"""
        try:
            return files.write_content(body.path, body.content)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            # 非法扩展名
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/files")
    def files_delete(
        path: str = Query(..., description="相对 workspace 根"),
        recursive: bool = Query(False, description="非空目录是否递归删除"),
    ) -> dict:
        """删除文件或目录。删根 400；非空目录不带 recursive 409。"""
        try:
            return files.delete(path, recursive=recursive)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    # ---- 任务中心 ----

    @app.post("/api/jobs")
    def jobs_submit(body: JobSubmitBody) -> dict:
        """提交后台任务。校验失败 400。"""
        try:
            return jobs.submit(list(body.argv), body.kind, body.title)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/jobs")
    def jobs_list() -> dict:
        return {"jobs": jobs.list_jobs()}

    @app.get("/api/jobs/{job_id}")
    def jobs_detail(job_id: str) -> dict:
        try:
            return jobs.job_detail(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/log")
    def jobs_log(
        job_id: str,
        tail: int = Query(200, ge=1, le=2000),
    ) -> dict:
        try:
            return jobs.job_log(job_id, tail=tail)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/kill")
    def jobs_kill(job_id: str) -> dict:
        """终止 running 任务；非 running 409。"""
        try:
            return jobs.kill(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    # ---- CLI schema ----

    @app.get("/api/cli/schema")
    def cli_schema() -> dict[str, Any]:
        """整棵 fz 命令树（供表单自动生成）。"""
        return get_cli_schema()

    from factorzen.server.views import register_views

    register_views(app, idx)

    # SPA 静态资源:仓库根/webui/dist 存在时挂到 /ui;测试环境无 dist 则静默跳过
    repo_root = Path(__file__).resolve().parents[3]
    ui_dist = repo_root / "webui" / "dist"
    if (ui_dist / "index.html").exists():
        app.mount("/ui", StaticFiles(directory=str(ui_dist), html=True), name="ui")

    return app


app = create_app()
