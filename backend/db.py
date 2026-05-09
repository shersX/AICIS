"""SQLite 持久化层：会话 / 消息存储与后台日志查询。

设计原则：
- 进程内单例 `sqlite3.Connection`，`check_same_thread=False`；
- 所有写操作通过 `_WRITE_LOCK` 串行化；
- 读操作依赖 WAL 模式下的并发读能力，不加锁；
- 时间格式统一使用 `datetime.now().isoformat()`（与既有代码风格一致，便于迁移）。
"""

from __future__ import annotations

import hashlib
import json as _json
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

_DEFAULT_REL_PATH = os.path.join("data", "aicis.db")

_DB_PATH: Optional[str] = None
_CONN: Optional[sqlite3.Connection] = None
_WRITE_LOCK = threading.Lock()
_INIT_LOCK = threading.Lock()
_INITIALIZED = False


def _resolve_db_path() -> str:
    configured = os.getenv("SQLITE_PATH")
    if configured:
        return os.path.abspath(configured)
    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(package_root, _DEFAULT_REL_PATH)


def get_db_path() -> str:
    global _DB_PATH
    if _DB_PATH is None:
        _DB_PATH = _resolve_db_path()
    return _DB_PATH


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def get_conn() -> sqlite3.Connection:
    """获取进程内单例连接；`isolation_level=None` 让我们手动掌控事务。"""
    global _CONN
    if _CONN is None:
        path = get_db_path()
        _ensure_parent_dir(path)
        conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        _CONN = conn
    return _CONN


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    title         TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    employee_no   TEXT,
    employee_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_user_updated ON sessions(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_updated      ON sessions(updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    model           TEXT,
    rag_trace_json  TEXT,
    latency_ms      INTEGER,
    status          TEXT NOT NULL DEFAULT 'ok',
    error_message   TEXT,
    created_at      TEXT NOT NULL,
    employee_no     TEXT,
    employee_name   TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_user_created    ON messages(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_created         ON messages(created_at DESC);

CREATE TABLE IF NOT EXISTS tickets (
    ticket          TEXT PRIMARY KEY,
    employee_no     TEXT NOT NULL,
    employee_name   TEXT,
    oassotoken_hash TEXT,
    created_at      TEXT NOT NULL,
    expires_at      TEXT NOT NULL,
    last_used_at    TEXT,
    revoked_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_tickets_employee ON tickets(employee_no);
CREATE INDEX IF NOT EXISTS idx_tickets_expires  ON tickets(expires_at);
"""


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl_type: str) -> None:
    """幂等给老表补列。SQLite 支持 ALTER TABLE ADD COLUMN，但不支持 IF NOT EXISTS。"""
    if column not in _table_columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")


def init_schema() -> None:
    """幂等建表 + 老表平滑加列；多次调用安全。"""
    global _INITIALIZED
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        conn = get_conn()
        with _WRITE_LOCK:
            conn.executescript(_SCHEMA_SQL)
            # 老库里没有员工字段时，补列；不影响新库（已包含在 _SCHEMA_SQL 里）。
            _ensure_column(conn, "sessions", "employee_no", "TEXT")
            _ensure_column(conn, "sessions", "employee_name", "TEXT")
            _ensure_column(conn, "messages", "employee_no", "TEXT")
            _ensure_column(conn, "messages", "employee_name", "TEXT")
        _INITIALIZED = True


def _now_iso() -> str:
    return datetime.now().isoformat()


def _compute_title(messages: List[Dict[str, Any]]) -> Optional[str]:
    """取首条 user 消息的前 50 字作为标题。"""
    for m in messages:
        if m.get("role") == "user":
            content = (m.get("content") or "").strip().replace("\n", " ")
            if content:
                return content[:50]
    return None


def _trace_to_json(trace: Any) -> Optional[str]:
    if trace is None:
        return None
    if isinstance(trace, str):
        return trace
    try:
        return _json.dumps(trace, ensure_ascii=False, default=str)
    except Exception:
        return None


def session_has_sso_employee_binding(session_id: str) -> bool:
    """会话是否曾以「非匿名」员工身份落库（employee_no 已填且不是占位 default_user）。

    用于防止后续匿名请求（典型 user_id=default_user）覆盖 SSO 会话的归属字段。
    """
    if not session_id:
        return False
    conn = get_conn()
    row = conn.execute(
        "SELECT employee_no FROM sessions WHERE session_id=?",
        (session_id,),
    ).fetchone()
    if not row:
        return False
    eno = (row["employee_no"] or "").strip()
    return bool(eno) and eno != "default_user"


def save_session(
    user_id: str,
    session_id: str,
    messages: List[Dict[str, Any]],
    *,
    employee_no: Optional[str] = None,
    employee_name: Optional[str] = None,
) -> None:
    """替换式保存整个会话的消息列表（事务内 upsert session + 全量替换 messages）。

    messages 每项字典支持的键：
      role / content / created_at / model / rag_trace / latency_ms / status / error_message

    ``employee_no``/``employee_name``：本次会话归属的真实员工身份，会同时写入
    sessions 与每条 messages 行；老数据没有这两个字段时，传 None 即可（向下兼容）。
    """
    conn = get_conn()
    now = _now_iso()
    title = _compute_title(messages)
    msg_count = len(messages)

    with _WRITE_LOCK:
        conn.execute("BEGIN")
        try:
            row = conn.execute(
                "SELECT created_at FROM sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO sessions(session_id, user_id, title, message_count, "
                    "created_at, updated_at, employee_no, employee_name) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (
                        session_id,
                        user_id,
                        title,
                        msg_count,
                        now,
                        now,
                        employee_no,
                        employee_name,
                    ),
                )
            else:
                # 用 COALESCE 避免传入 None 时把已有员工信息覆盖丢失。
                conn.execute(
                    "UPDATE sessions SET user_id=?, title=COALESCE(?, title), "
                    "message_count=?, updated_at=?, "
                    "employee_no=COALESCE(?, employee_no), "
                    "employee_name=COALESCE(?, employee_name) "
                    "WHERE session_id=?",
                    (
                        user_id,
                        title,
                        msg_count,
                        now,
                        employee_no,
                        employee_name,
                        session_id,
                    ),
                )

            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))

            for m in messages:
                conn.execute(
                    "INSERT INTO messages(session_id, user_id, role, content, model, "
                    "rag_trace_json, latency_ms, status, error_message, created_at, "
                    "employee_no, employee_name) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        session_id,
                        user_id,
                        m.get("role") or "user",
                        m.get("content") or "",
                        m.get("model"),
                        _trace_to_json(m.get("rag_trace")),
                        m.get("latency_ms"),
                        m.get("status") or "ok",
                        m.get("error_message"),
                        m.get("created_at") or now,
                        employee_no,
                        employee_name,
                    ),
                )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise


def load_session_messages(user_id: str, session_id: str) -> List[Dict[str, Any]]:
    """按时间升序返回某会话的所有消息（含 rag_trace 反序列化）。"""
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, role, content, model, rag_trace_json, latency_ms, status, "
        "error_message, created_at FROM messages "
        "WHERE session_id=? AND user_id=? ORDER BY created_at, id",
        (session_id, user_id),
    ).fetchall()
    return [_row_to_message_dict(r) for r in rows]


def _row_to_message_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    rt = d.pop("rag_trace_json", None)
    if rt:
        try:
            d["rag_trace"] = _json.loads(rt)
        except Exception:
            d["rag_trace"] = None
    else:
        d["rag_trace"] = None
    return d


def list_sessions_info(user_id: str) -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT session_id, title, message_count, created_at, updated_at "
        "FROM sessions WHERE user_id=? ORDER BY updated_at DESC",
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_session(user_id: str, session_id: str) -> bool:
    conn = get_conn()
    with _WRITE_LOCK:
        cur = conn.execute(
            "DELETE FROM sessions WHERE session_id=? AND user_id=?",
            (session_id, user_id),
        )
        return cur.rowcount > 0


# ---------------- 后台查询 ----------------

_ALLOWED_ROLES = {"user", "assistant", "system"}
_ALLOWED_STATUSES = {"ok", "error"}


def admin_query_logs(
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    model: Optional[str] = None,
    status: Optional[str] = None,
    role: Optional[str] = None,
    q: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    employee_no: Optional[str] = None,
    employee_name: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> Dict[str, Any]:
    """带筛选/分页的消息列表。返回条目不含完整 rag_trace_json，仅有 has_rag_trace 标记。"""
    conn = get_conn()
    clauses: List[str] = []
    params: List[Any] = []

    if user_id:
        clauses.append("m.user_id = ?")
        params.append(user_id)
    if session_id:
        clauses.append("m.session_id = ?")
        params.append(session_id)
    if model:
        clauses.append("m.model = ?")
        params.append(model)
    if status and status in _ALLOWED_STATUSES:
        clauses.append("m.status = ?")
        params.append(status)
    if role and role in _ALLOWED_ROLES:
        clauses.append("m.role = ?")
        params.append(role)
    if q:
        clauses.append("m.content LIKE ?")
        params.append(f"%{q}%")
    if start:
        clauses.append("m.created_at >= ?")
        params.append(start)
    if end:
        clauses.append("m.created_at <= ?")
        params.append(end)
    if employee_no:
        clauses.append("m.employee_no = ?")
        params.append(employee_no)
    if employee_name:
        clauses.append("m.employee_name LIKE ?")
        params.append(f"%{employee_name}%")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    total_row = conn.execute(
        f"SELECT COUNT(*) AS c FROM messages m {where}", params
    ).fetchone()
    total = total_row["c"] if total_row else 0

    page = max(1, int(page))
    page_size = max(1, min(500, int(page_size)))
    offset = (page - 1) * page_size

    rows = conn.execute(
        f"""
        SELECT m.id, m.session_id, m.user_id, m.role, m.content, m.model,
               m.latency_ms, m.status, m.error_message, m.created_at,
               m.employee_no, m.employee_name,
               CASE WHEN m.rag_trace_json IS NOT NULL THEN 1 ELSE 0 END AS has_rag_trace
        FROM messages m
        {where}
        ORDER BY m.created_at DESC, m.id DESC
        LIMIT ? OFFSET ?
        """,
        (*params, page_size, offset),
    ).fetchall()

    items = []
    for r in rows:
        item = dict(r)
        item["has_rag_trace"] = bool(item.get("has_rag_trace"))
        items.append(item)

    return {"total": total, "page": page, "page_size": page_size, "items": items}


def admin_get_log_detail(message_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    row = conn.execute(
        "SELECT id, session_id, user_id, role, content, model, rag_trace_json, "
        "latency_ms, status, error_message, created_at, employee_no, employee_name "
        "FROM messages WHERE id=?",
        (message_id,),
    ).fetchone()
    if not row:
        return None
    return _row_to_message_dict(row)


def admin_query_sessions(
    *,
    user_id: Optional[str] = None,
    q: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    employee_no: Optional[str] = None,
    employee_name: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> Dict[str, Any]:
    conn = get_conn()
    clauses: List[str] = []
    params: List[Any] = []

    if user_id:
        clauses.append("s.user_id = ?")
        params.append(user_id)
    if q:
        clauses.append("s.title LIKE ?")
        params.append(f"%{q}%")
    if start:
        clauses.append("s.updated_at >= ?")
        params.append(start)
    if end:
        clauses.append("s.updated_at <= ?")
        params.append(end)
    if employee_no:
        clauses.append("s.employee_no = ?")
        params.append(employee_no)
    if employee_name:
        clauses.append("s.employee_name LIKE ?")
        params.append(f"%{employee_name}%")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    total_row = conn.execute(
        f"SELECT COUNT(*) AS c FROM sessions s {where}", params
    ).fetchone()
    total = total_row["c"] if total_row else 0

    page = max(1, int(page))
    page_size = max(1, min(500, int(page_size)))
    offset = (page - 1) * page_size

    rows = conn.execute(
        f"""
        SELECT s.session_id, s.user_id, s.title, s.message_count,
               s.created_at, s.updated_at, s.employee_no, s.employee_name
        FROM sessions s
        {where}
        ORDER BY s.updated_at DESC
        LIMIT ? OFFSET ?
        """,
        (*params, page_size, offset),
    ).fetchall()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [dict(r) for r in rows],
    }


def admin_get_session_detail(session_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    s = conn.execute(
        "SELECT session_id, user_id, title, message_count, created_at, updated_at, "
        "employee_no, employee_name "
        "FROM sessions WHERE session_id=?",
        (session_id,),
    ).fetchone()
    if not s:
        return None
    msgs = conn.execute(
        "SELECT id, session_id, user_id, role, content, model, rag_trace_json, "
        "latency_ms, status, error_message, created_at, employee_no, employee_name "
        "FROM messages "
        "WHERE session_id=? ORDER BY created_at, id",
        (session_id,),
    ).fetchall()
    return {
        "session": dict(s),
        "messages": [_row_to_message_dict(m) for m in msgs],
    }


def admin_stats(days: int = 7) -> Dict[str, Any]:
    """总览统计：会话总数、消息总数、错误总数，以及最近 N 天每日消息量。"""
    conn = get_conn()
    days = max(1, min(90, int(days)))

    total_sessions = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()["c"]
    total_messages = conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"]
    total_errors = conn.execute(
        "SELECT COUNT(*) AS c FROM messages WHERE status='error'"
    ).fetchone()["c"]

    rows = conn.execute(
        """
        SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS c
        FROM messages
        GROUP BY day
        ORDER BY day DESC
        LIMIT ?
        """,
        (days,),
    ).fetchall()

    return {
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "total_errors": total_errors,
        "daily_messages": [{"day": r["day"], "count": r["c"]} for r in rows],
    }


# ---------------- 鉴权票据 (tickets) ----------------


def _hash_oassotoken(oassotoken: str) -> str:
    """只存指纹，不存原文，便于排查"同一员工同一来源"而又不暴露真 cookie。"""
    return hashlib.sha256(oassotoken.encode("utf-8")).hexdigest()


def create_ticket(
    employee_no: str,
    employee_name: Optional[str],
    oassotoken: str,
    *,
    ttl_hours: int = 8,
) -> Tuple[str, str]:
    """颁发一张新票据。返回 (ticket_str, expires_at_iso)。"""
    conn = get_conn()
    ticket = secrets.token_urlsafe(32)
    now_dt = datetime.now()
    now_iso = now_dt.isoformat()
    expires_at_iso = (now_dt + timedelta(hours=max(1, int(ttl_hours)))).isoformat()
    with _WRITE_LOCK:
        conn.execute(
            "INSERT INTO tickets(ticket, employee_no, employee_name, oassotoken_hash, "
            "created_at, expires_at, last_used_at, revoked_at) "
            "VALUES(?,?,?,?,?,?,NULL,NULL)",
            (
                ticket,
                employee_no,
                employee_name,
                _hash_oassotoken(oassotoken),
                now_iso,
                expires_at_iso,
            ),
        )
    return ticket, expires_at_iso


def get_active_ticket(ticket: str) -> Optional[Dict[str, Any]]:
    """返回未撤销且未过期的票据；同时把 last_used_at 顺带刷新。找不到返回 None。"""
    if not ticket:
        return None
    conn = get_conn()
    now_iso = _now_iso()
    row = conn.execute(
        "SELECT ticket, employee_no, employee_name, created_at, expires_at, "
        "last_used_at, revoked_at "
        "FROM tickets WHERE ticket=?",
        (ticket,),
    ).fetchone()
    if not row:
        return None
    if row["revoked_at"] is not None:
        return None
    if (row["expires_at"] or "") <= now_iso:
        return None
    with _WRITE_LOCK:
        conn.execute(
            "UPDATE tickets SET last_used_at=? WHERE ticket=?",
            (now_iso, ticket),
        )
    out = dict(row)
    out["last_used_at"] = now_iso
    return out


def revoke_ticket(ticket: str) -> bool:
    conn = get_conn()
    with _WRITE_LOCK:
        cur = conn.execute(
            "UPDATE tickets SET revoked_at=? WHERE ticket=? AND revoked_at IS NULL",
            (_now_iso(), ticket),
        )
        return cur.rowcount > 0


def cleanup_expired_tickets() -> int:
    """物理删除已过期或已撤销超过 7 天的票据，避免无限膨胀。"""
    conn = get_conn()
    cutoff = (datetime.now() - timedelta(days=7)).isoformat()
    now_iso = _now_iso()
    with _WRITE_LOCK:
        cur = conn.execute(
            "DELETE FROM tickets WHERE expires_at < ? OR (revoked_at IS NOT NULL AND revoked_at < ?)",
            (now_iso, cutoff),
        )
        return cur.rowcount
