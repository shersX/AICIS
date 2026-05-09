import re
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import sys

from backend.auth import resolve_employee
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    SessionListResponse,
    SessionInfo,
    SessionMessagesResponse,
    MessageInfo,
    SessionDeleteResponse,
)
from backend.sso import EmployeeInfo
from backend.agent import chat_with_agent, chat_with_agent_stream, storage
from backend import db

router = APIRouter()


def _ensure_self(emp: Optional[EmployeeInfo], user_id_in_path: str) -> None:
    """若已认证为某员工，禁止越权访问其他人的会话；匿名访问不限制。"""
    if emp is not None and user_id_in_path != emp.employee_no:
        raise HTTPException(
            status_code=403,
            detail="无权访问其他员工的会话",
        )


def _resolve_user_identity(
    emp: Optional[EmployeeInfo],
    fallback_user_id: Optional[str],
) -> tuple[str, Optional[str]]:
    """根据"是否已认证"决定本次请求落库使用的 user_id / employee_name。"""
    if emp is not None:
        return emp.employee_no, (emp.employee_name or None)
    return (fallback_user_id or "default_user"), None


def _guard_anonymous_overwrite_sso_session(
    session_id: Optional[str],
    emp: Optional[EmployeeInfo],
    fallback_user_id: Optional[str],
) -> None:
    """禁止匿名 default_user 继续写入「已绑定员工工号」的会话，避免覆盖追溯信息。"""
    if emp is not None or not session_id:
        return
    uid = (fallback_user_id or "").strip() or "default_user"
    if uid != "default_user":
        return
    if db.session_has_sso_employee_binding(session_id):
        raise HTTPException(
            status_code=403,
            detail=(
                "此会话已绑定登录员工身份，不能使用匿名方式继续写入。"
                "请刷新网页 A 或重新打开助手，以使用当前 OA 登录态。"
            ),
        )


@router.get("/sessions/{user_id}/{session_id}", response_model=SessionMessagesResponse)
async def get_session_messages(
    user_id: str,
    session_id: str,
    emp: Optional[EmployeeInfo] = Depends(resolve_employee),
):
    """获取指定会话的所有消息"""
    _ensure_self(emp, user_id)
    try:
        items = storage.get_session_messages_with_trace(user_id, session_id)
        messages = [
            MessageInfo(
                type=i["type"],
                content=i["content"],
                timestamp=i.get("timestamp") or "",
                rag_trace=i.get("rag_trace"),
            )
            for i in items
        ]
        return SessionMessagesResponse(messages=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{user_id}", response_model=SessionListResponse)
async def list_sessions(
    user_id: str,
    emp: Optional[EmployeeInfo] = Depends(resolve_employee),
):
    """获取用户的所有会话列表"""
    _ensure_self(emp, user_id)
    try:
        rows = storage.list_sessions_with_info(user_id)
        sessions = [
            SessionInfo(
                session_id=r["session_id"],
                updated_at=r.get("updated_at") or "",
                message_count=r.get("message_count") or 0,
                employee_no=r.get("employee_no"),
                employee_name=r.get("employee_name"),
            )
            for r in rows
        ]
        return SessionListResponse(sessions=sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{user_id}/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(
    user_id: str,
    session_id: str,
    emp: Optional[EmployeeInfo] = Depends(resolve_employee),
):
    """删除指定会话"""
    _ensure_self(emp, user_id)
    try:
        deleted = storage.delete_session(user_id, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="会话不存在")
        return SessionDeleteResponse(session_id=session_id, message="成功删除会话")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    emp: Optional[EmployeeInfo] = Depends(resolve_employee),
):
    _guard_anonymous_overwrite_sso_session(request.session_id, emp, request.user_id)
    user_id, employee_name = _resolve_user_identity(emp, request.user_id)
    try:
        resp = chat_with_agent(
            request.message,
            user_id,
            request.session_id,
            employee_name=employee_name,
        )
        if isinstance(resp, dict):
            return ChatResponse(**resp)
        return ChatResponse(response=resp)
    except Exception as e:
        message = str(e)
        match = re.search(r"Error code:\s*(\d{3})", message)
        if match:
            code = int(match.group(1))
            if code == 429:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "上游模型服务触发限流/额度限制（429）。请检查账号额度/模型状态。\n"
                        f"原始错误：{message}"
                    ),
                )
            if code in (401, 403):
                raise HTTPException(status_code=code, detail=message)
            raise HTTPException(status_code=code, detail=message)
        raise HTTPException(status_code=500, detail=message)


@router.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    emp: Optional[EmployeeInfo] = Depends(resolve_employee),
):
    """跟 Agent 对话 (流式)"""
    _guard_anonymous_overwrite_sso_session(request.session_id, emp, request.user_id)
    user_id, employee_name = _resolve_user_identity(emp, request.user_id)

    async def event_generator():
        try:
            async for chunk in chat_with_agent_stream(
                request.message,
                user_id,
                request.session_id,
                employee_name=employee_name,
            ):
                yield chunk
        except Exception as e:
            print(f"[CHAT_STREAM] 生成器异常: {type(e).__name__}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            error_data = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
