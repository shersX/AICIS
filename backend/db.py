"""SQLite 持久化层：会话 / 消息存储与后台日志查询。

设计原则：
- 进程内单例 `sqlite3.Connection`，`check_same_thread=False`；
- 所有写操作通过 `_WRITE_LOCK` 串行化；
- 读操作依赖 WAL 模式下的并发读能力，不加锁；
- 时间格式统一使用 `datetime.now().isoformat()`（与既有代码风格一致，便于迁移）。
"""

from __future__ import annotations

import json as _json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

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
    updated_at    TEXT NOT NULL
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
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_user_created    ON messages(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_created         ON messages(created_at DESC);
"""


def init_schema() -> None:
    """幂等建表；多次调用安全。"""
    global _INITIALIZED
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        conn = get_conn()
        with _WRITE_LOCK:
            conn.executescript(_SCHEMA_SQL)
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


def save_session(user_id: str, session_id: str, messages: List[Dict[str, Any]]) -> None:
    """替换式保存整个会话的消息列表（事务内 upsert session + 全量替换 messages）。

    messages 每项字典支持的键：
      role / content / created_at / model / rag_trace / latency_ms / status / error_message
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
                    "INSERT INTO sessions(session_id, user_id, title, message_count, created_at, updated_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (session_id, user_id, title, msg_count, now, now),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET user_id=?, title=COALESCE(?, title), "
                    "message_count=?, updated_at=? WHERE session_id=?",
                    (user_id, title, msg_count, now, session_id),
                )

            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))

            for m in messages:
                conn.execute(
                    "INSERT INTO messages(session_id, user_id, role, content, model, "
                    "rag_trace_json, latency_ms, status, error_message, created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)",
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
        "latency_ms, status, error_message, created_at FROM messages WHERE id=?",
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
               s.created_at, s.updated_at
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
        "SELECT session_id, user_id, title, message_count, created_at, updated_at "
        "FROM sessions WHERE session_id=?",
        (session_id,),
    ).fetchone()
    if not s:
        return None
    msgs = conn.execute(
        "SELECT id, session_id, user_id, role, content, model, rag_trace_json, "
        "latency_ms, status, error_message, created_at FROM messages "
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
