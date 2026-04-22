from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import backend.api as api_module
import backend.admin_api as admin_api_module
from backend import db

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


def create_app() -> FastAPI:
    app = FastAPI(title="AICIS API")

    # 启动时确保 SQLite 表已建
    db.init_schema()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # No-cache middleware for development
    @app.middleware("http")
    async def _no_cache(request, call_next):
        response = await call_next(request)
        path = request.url.path or ""
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    app.include_router(api_module.router)
    app.include_router(admin_api_module.router)

    # 主前端静态；frontend/admin/index.html 会通过此挂载以 /admin/ 路径对外暴露，
    # 不会与 /admin/logs 等 API（精确匹配）冲突。
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
