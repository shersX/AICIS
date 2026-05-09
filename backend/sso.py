"""SSO 客户端：基于 oassotoken 校验员工身份。

设计说明
--------
- 通过 ``SSO_MODE`` 环境变量切换 ``mock`` / ``optimus`` 两种实现，方便开发期
  脱离真实 OA 也能走通整条链路。
- ``OptimusSSOClient`` 调用真实接口：
    * 方法：GET
    * Cookie 名：``oassotoken`` (注意不是 ``oassotok``)
    * 成功返回示例::

        {
          "code": "0",
          "success": true,
          "msg": "成功",
          "data": {"Code": "工号", "Name": "姓名"}
        }
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass(frozen=True)
class EmployeeInfo:
    employee_no: str
    employee_name: str = ""


class SSOAuthError(Exception):
    """oassotoken 无效 / 上游拒绝 / 字段缺失等业务错误。"""


class SSONetworkError(Exception):
    """连接/超时等基础设施错误，调用方可决定重试或返回 502。"""


class SSOClient:
    def verify(self, oassotoken: str) -> EmployeeInfo:  # pragma: no cover - 接口
        raise NotImplementedError


class MockSSOClient(SSOClient):
    """开发/演示模式：按 oassotoken 前缀生成可读的假员工，便于联调验证。"""

    def verify(self, oassotoken: str) -> EmployeeInfo:
        token = (oassotoken or "").strip()
        if not token:
            raise SSOAuthError("oassotoken 为空")
        suffix = token[:6] if len(token) >= 6 else token
        return EmployeeInfo(
            employee_no=f"MOCK_{suffix}",
            employee_name=f"测试员工_{suffix}",
        )


class OptimusSSOClient(SSOClient):
    """真实 Optimus SSO 客户端。"""

    def __init__(
        self,
        url: str,
        timeout: float = 5.0,
        cookie_name: str = "oassotoken",
        verify_ssl: bool = True,
    ):
        if not url:
            raise ValueError("OptimusSSOClient 需要非空的 url")
        self.url = url
        self.timeout = timeout
        self.cookie_name = cookie_name
        self.verify_ssl = verify_ssl

    def verify(self, oassotoken: str) -> EmployeeInfo:
        token = (oassotoken or "").strip()
        if not token:
            raise SSOAuthError("oassotoken 为空")
        try:
            resp = requests.get(
                self.url,
                cookies={self.cookie_name: token},
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
        except requests.RequestException as e:
            raise SSONetworkError(f"调用 SSO 失败：{e}") from e

        if resp.status_code != 200:
            raise SSOAuthError(
                f"SSO 返回 HTTP {resp.status_code}：{resp.text[:200]}"
            )
        try:
            body = resp.json()
        except ValueError as e:
            raise SSOAuthError(f"SSO 返回非 JSON：{resp.text[:200]}") from e

        if str(body.get("code")) != "0" or not body.get("success", False):
            raise SSOAuthError(body.get("msg") or f"SSO 校验未通过：{body}")

        data = body.get("data") or {}
        # OA 字段大小写以接口文档为准；当前是 Code / Name。
        code = (data.get("psnCode") or data.get("code") or "").strip()
        name = (data.get("empName") or data.get("name") or "").strip()
        if not code:
            raise SSOAuthError(f"SSO 返回缺少员工工号字段：{data}")
        return EmployeeInfo(employee_no=code, employee_name=name)


_DEFAULT_OPTIMUS_URL = "http://optimusinternal.isimcere.com/optimus/login/check"


def _build_optimus_client() -> OptimusSSOClient:
    url = (os.getenv("SSO_URL") or _DEFAULT_OPTIMUS_URL).strip()
    timeout = float(os.getenv("SSO_TIMEOUT", "5"))
    verify_ssl = (os.getenv("SSO_VERIFY_SSL", "true").lower() != "false")
    return OptimusSSOClient(url=url, timeout=timeout, verify_ssl=verify_ssl)


def get_sso_client() -> SSOClient:
    """工厂：根据 SSO_MODE 决定返回 mock 或真实客户端。

    第一次构造时往 stderr 打一行日志，方便上线时确认当前生效模式。
    """
    mode = (os.getenv("SSO_MODE") or "mock").strip().lower()
    if mode in ("mock", "test", "dev"):
        client: SSOClient = MockSSOClient()
        print(f"[SSO] 使用 MockSSOClient (SSO_MODE={mode})", file=sys.stderr)
        return client
    if mode in ("optimus", "production", "prod", "real"):
        client = _build_optimus_client()
        print(
            f"[SSO] 使用 OptimusSSOClient url={client.url} "
            f"timeout={client.timeout}s verify_ssl={client.verify_ssl}",
            file=sys.stderr,
        )
        return client
    raise RuntimeError(f"未知的 SSO_MODE: {mode!r}（应为 mock 或 optimus）")


# 进程内单例：避免每次请求都打印一遍日志、构造一次对象。
_DEFAULT_CLIENT: Optional[SSOClient] = None


def default_client() -> SSOClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = get_sso_client()
    return _DEFAULT_CLIENT
