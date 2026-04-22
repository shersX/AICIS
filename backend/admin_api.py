"""后台管理接口（问答日志 / 会话追溯）。

所有路由都要求 ``Authorization: Bearer <ADMIN_TOKEN>``，``ADMIN_TOKEN`` 从 ``.env`` 读取。
未配置 ``ADMIN_TOKEN`` 时，路由依然挂载但所有请求 503，防止裸奔。
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from backend import db

router = APIRouter(prefix="/admin", tags=["admin"])


def _get_admin_token() -> str:
    # 每次调用都重新读取，方便运维热更新（改 .env 后重启也更灵活）
    return (os.getenv("ADMIN_TOKEN") or "").strip()


def require_admin_token(authorization: Optional[str] = Header(default=None)) -> None:
    token = _get_admin_token()
    if not token:
        raise HTTPException(
            status_code=503,
            detail="后台管理未启用：请在 .env 中配置 ADMIN_TOKEN 并重启服务。",
        )
    if not authorization:
        raise HTTPException(status_code=401, detail="缺少 Authorization 头")
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(status_code=401, detail="Authorization 头格式错误")
    if parts[1].strip() != token:
        raise HTTPException(status_code=403, detail="Token 无效")


@router.get("/ping")
async def admin_ping(_: None = Depends(require_admin_token)):
    """心跳：用于前端登录页校验 token 是否正确。"""
    return {"ok": True}


@router.get("/stats")
async def admin_stats(
    days: int = Query(7, ge=1, le=90),
    _: None = Depends(require_admin_token),
):
    return db.admin_stats(days=days)


@router.get("/logs")
async def admin_logs(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    model: Optional[str] = None,
    status: Optional[str] = None,
    role: Optional[str] = None,
    q: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: None = Depends(require_admin_token),
):
    return db.admin_query_logs(
        user_id=user_id,
        session_id=session_id,
        model=model,
        status=status,
        role=role,
        q=q,
        start=start,
        end=end,
        page=page,
        page_size=page_size,
    )


@router.get("/logs/{message_id}")
async def admin_log_detail(
    message_id: int,
    _: None = Depends(require_admin_token),
):
    item = db.admin_get_log_detail(message_id)
    if not item:
        raise HTTPException(status_code=404, detail="消息不存在")
    return item


@router.get("/sessions")
async def admin_sessions(
    user_id: Optional[str] = None,
    q: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: None = Depends(require_admin_token),
):
    return db.admin_query_sessions(
        user_id=user_id,
        q=q,
        start=start,
        end=end,
        page=page,
        page_size=page_size,
    )


@router.get("/sessions/{session_id}")
async def admin_session_detail(
    session_id: str,
    _: None = Depends(require_admin_token),
):
    item = db.admin_get_session_detail(session_id)
    if not item:
        raise HTTPException(status_code=404, detail="会话不存在")
    return item
