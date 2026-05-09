"""嵌入式问答的鉴权层。

链路概述
--------
1. 浏览器在网页 A 中已登录 OA，cookie 中含 ``oassotoken``。
2. 同源反代下的 ``embed.js`` 读到 ``oassotoken``，POST 给 ``/auth/sso_login``。
3. 后端用 :mod:`backend.sso` 调真实 SSO 校验员工身份，颁发短效 ``ticket``
   写入 SQLite ``tickets`` 表。
4. 前端拿到 ticket 后，所有业务请求带 ``Authorization: Bearer <ticket>``。
5. ``require_employee`` 依赖在每个受保护路由上校验 ticket 并返回 :class:`EmployeeInfo`。

为什么不直接在 chat 接口里透传 oassotoken
----------------------------------------
- oassotoken 不应频繁出现在请求中（出现越多，泄露面越大）。
- 每次都打 SSO 性能差且会拖累 OA 系统。
- ticket 后端可控、可撤销、可观测；oassotoken 不行。
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException

from backend import db
from backend.schemas import SSOLoginRequest, SSOLoginResponse
from backend.sso import EmployeeInfo, SSOAuthError, SSONetworkError, default_client

router = APIRouter(prefix="/auth", tags=["auth"])


def _ticket_ttl_hours() -> int:
    raw = os.getenv("TICKET_TTL_HOURS", "8")
    try:
        v = int(raw)
    except ValueError:
        v = 8
    return max(1, min(168, v))  # 限制 1h ~ 7d


@router.post("/sso_login", response_model=SSOLoginResponse)
def sso_login(req: SSOLoginRequest) -> SSOLoginResponse:
    token = (req.oassotoken or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="oassotoken 不能为空")

    try:
        info = default_client().verify(token)
    except SSOAuthError as e:
        raise HTTPException(status_code=401, detail=f"SSO 校验失败：{e}")
    except SSONetworkError as e:
        # 网络层故障：返回 502，前端可决定是否提示用户稍后重试。
        raise HTTPException(status_code=502, detail=f"无法访问 SSO：{e}")
    except Exception as e:  # noqa: BLE001
        print(f"[AUTH] SSO 调用未知异常：{type(e).__name__}: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="SSO 服务异常")

    # 顺手清理一下过期票据，避免无限增长。
    try:
        db.cleanup_expired_tickets()
    except Exception as e:  # noqa: BLE001
        print(f"[AUTH] cleanup_expired_tickets 异常（忽略）：{e}", file=sys.stderr)

    ticket, expires_at = db.create_ticket(
        info.employee_no,
        info.employee_name,
        token,
        ttl_hours=_ticket_ttl_hours(),
    )
    return SSOLoginResponse(
        ticket=ticket,
        employee_no=info.employee_no,
        employee_name=info.employee_name,
        expires_at=expires_at,
    )


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def require_employee(
    authorization: Optional[str] = Header(default=None),
) -> EmployeeInfo:
    """严格鉴权：必须有有效 ticket，否则 401。

    目前路由层默认使用 :func:`resolve_employee`（按场景松紧），此函数预留给
    将来需要强制鉴权的场景（例如对外开放的某些只读接口）。
    """
    ticket = _extract_bearer(authorization)
    if not ticket:
        raise HTTPException(
            status_code=401,
            detail="缺少有效的 Authorization Bearer ticket",
        )
    row = db.get_active_ticket(ticket)
    if not row:
        raise HTTPException(status_code=401, detail="ticket 已失效，请重新登录")
    return EmployeeInfo(
        employee_no=row["employee_no"],
        employee_name=row.get("employee_name") or "",
    )


def resolve_employee(
    authorization: Optional[str] = Header(default=None),
    x_aicis_embed: Optional[str] = Header(default=None),
) -> Optional[EmployeeInfo]:
    """按访问场景解析当前员工身份。

    - **嵌入式访问**（请求头 ``X-AICIS-Embed: 1``）：必须带有效 ticket，
      否则 401。这样保证嵌入网页 A 的对话一定能绑到真实员工身份。
    - **独立访问**（无该 header，例如开发者直接打开 ``localhost:8000``）：
      不强制要求 ticket。带了就解析返回 EmployeeInfo；没带或 ticket 失效
      则返回 ``None``，路由层会回退到匿名 ``user_id``。
    """
    is_embed = (x_aicis_embed or "").strip() == "1"
    ticket = _extract_bearer(authorization)

    if ticket:
        row = db.get_active_ticket(ticket)
        if row:
            return EmployeeInfo(
                employee_no=row["employee_no"],
                employee_name=row.get("employee_name") or "",
            )
        # 带了 ticket 但已失效：embed 模式下必须重新换；独立模式下退化为匿名。
        if is_embed:
            raise HTTPException(status_code=401, detail="ticket 已失效，请重新登录")
        return None

    # 完全没带 ticket
    if is_embed:
        raise HTTPException(
            status_code=401,
            detail="嵌入式访问需要鉴权 ticket",
        )
    return None
