from dataclasses import dataclass
from typing import Any, Optional
import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool


load_dotenv()

AMAP_WEATHER_API = os.getenv("AMAP_WEATHER_API")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")


@dataclass
class ToolRuntime:
    """单次 Agent 调用（invoke/astream）的运行时状态，通过 RunnableConfig['configurable']['tool_runtime'] 注入，避免全局变量串扰。"""

    knowledge_calls_this_turn: int = 0
    last_rag_context: Optional[dict] = None
    rag_step_queue: Any = None
    rag_step_loop: Any = None

    def reset_guards(self) -> None:
        self.knowledge_calls_this_turn = 0

    def set_last_rag_context(self, context: dict) -> None:
        self.last_rag_context = context

    def get_last_rag_context(self, clear: bool = True) -> Optional[dict]:
        ctx = self.last_rag_context
        if clear:
            self.last_rag_context = None
        return ctx

    def set_rag_step_queue(self, queue: Any) -> None:
        self.rag_step_queue = queue
        if queue:
            import asyncio

            try:
                self.rag_step_loop = asyncio.get_running_loop()
            except RuntimeError:
                self.rag_step_loop = asyncio.get_event_loop()
        else:
            self.rag_step_loop = None

    def emit_rag_step(self, icon: str, label: str, detail: str = "") -> None:
        if self.rag_step_queue is not None and self.rag_step_loop is not None:
            step = {"icon": icon, "label": label, "detail": detail}
            try:
                if not self.rag_step_loop.is_closed():
                    self.rag_step_loop.call_soon_threadsafe(self.rag_step_queue.put_nowait, step)
            except Exception:
                pass


def _format_publish_date(publish_time) -> str:
    """将时间戳（秒/毫秒）格式化为 YYYY-MM-DD。"""
    try:
        ts = int(publish_time)
        if ts <= 0:
            return "未知"
        # 兼容毫秒时间戳
        if ts > 10**12:
            ts = ts // 1000
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return "未知"


def get_current_weather(location: str, extensions: Optional[str] = "base") -> str:
    """获取天气信息"""
    if not location:
        return "location参数不能为空"
    if extensions not in ("base", "all"):
        return "extensions参数错误，请输入base或all"

    if not AMAP_WEATHER_API or not AMAP_API_KEY:
        return "天气服务未配置（缺少 AMAP_WEATHER_API 或 AMAP_API_KEY）"

    params = {
        "key": AMAP_API_KEY,
        "city": location,
        "extensions": extensions,
        "output": "json",
    }

    try:
        resp = requests.get(AMAP_WEATHER_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "1":
            return f"查询失败：{data.get('info', '未知错误')}"

        if extensions == "base":
            lives = data.get("lives", [])
            if not lives:
                return f"未查询到 {location} 的天气数据"
            w = lives[0]
            return (
                f"【{w.get('city', location)} 实时天气】\n"
                f"天气状况：{w.get('weather', '未知')}\n"
                f"温度：{w.get('temperature', '未知')}℃\n"
                f"湿度：{w.get('humidity', '未知')}%\n"
                f"风向：{w.get('winddirection', '未知')}\n"
                f"风力：{w.get('windpower', '未知')}级\n"
                f"更新时间：{w.get('reporttime', '未知')}"
            )

        forecasts = data.get("forecasts", [])
        if not forecasts:
            return f"未查询到 {location} 的天气预报数据"
        f0 = forecasts[0]
        out = [f"【{f0.get('city', location)} 天气预报】", f"更新时间：{f0.get('reporttime', '未知')}", ""]
        today = (f0.get("casts") or [])[0] if f0.get("casts") else {}
        out += [
            "今日天气：",
            f"  白天：{today.get('dayweather','未知')}",
            f"  夜间：{today.get('nightweather','未知')}",
            f"  气温：{today.get('nighttemp','未知')}~{today.get('daytemp','未知')}℃",
        ]
        return "\n".join(out)

    except requests.exceptions.Timeout:
        return "错误：请求天气服务超时"
    except requests.exceptions.RequestException as e:
        return f"错误：天气服务请求失败 - {e}"
    except Exception as e:
        return f"错误：解析天气数据失败 - {e}"


@tool("search_knowledge_base")
def search_knowledge_base(query: str, config: RunnableConfig) -> str:
    """Search for information in the knowledge base using hybrid retrieval (dense + sparse vectors)."""
    configurable = config.get("configurable") or {}
    runtime: ToolRuntime | None = configurable.get("tool_runtime")
    if runtime is None:
        runtime = ToolRuntime()

    if runtime.knowledge_calls_this_turn >= 1:
        return (
            "TOOL_CALL_LIMIT_REACHED: search_knowledge_base has already been called once in this turn. "
            "Use the existing retrieval result and provide the final answer directly."
        )
    runtime.knowledge_calls_this_turn += 1

    from backend.rag_pipeline import run_rag_graph

    rag_result = run_rag_graph(query, tool_runtime=runtime)

    docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
    if rag_trace:
        runtime.set_last_rag_context({"rag_trace": rag_trace})

    if not docs:
        return "No relevant documents found in the knowledge base."

    formatted = []
    for i, result in enumerate(docs, 1):
        title = result.get("title", "无标题")
        source = result.get("origin_name", result.get("filename", "Unknown"))
        text = result.get("text", result.get("summary", ""))
        url = result.get("url", "")
        publish_date = _format_publish_date(result.get("publish_time", 0))

        item = (
            f"[{i}] {title}\n"
            f"来源: {source}\n"
            f"发布时间: {publish_date}\n"
            f"链接: {url}"
        )
        if text:
            item += f"\n摘要: {text}"
        formatted.append(item)

    return "Retrieved News:" + "\n\n-----\n\n".join(formatted)
