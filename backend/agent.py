from dotenv import load_dotenv
import os
import json
import asyncio
import sys
import time
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from backend.tools import get_current_weather, search_knowledge_base, ToolRuntime
from backend import db
from datetime import datetime

load_dotenv()

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


_LC_TYPE_TO_ROLE = {"human": "user", "ai": "assistant", "system": "system"}
_ROLE_TO_LC_TYPE = {"user": "human", "assistant": "ai", "system": "system"}


class ConversationStorage:
    """基于 SQLite 的会话存储；对外保持与原实现相同的方法签名。"""

    def save(
        self,
        user_id: str,
        session_id: str,
        messages: list,
        metadata: dict = None,
        extra_message_data: list = None,
    ):
        """替换式保存整个会话的消息列表。

        ``extra_message_data`` 与 ``messages`` 同长度，每项可包含：
          ``rag_trace`` / ``model`` / ``latency_ms`` / ``status`` / ``error_message``
        """
        now_iso = datetime.now().isoformat()
        serialized = []
        for idx, msg in enumerate(messages):
            extra = (
                extra_message_data[idx]
                if extra_message_data and idx < len(extra_message_data)
                else None
            ) or {}
            lc_type = getattr(msg, "type", None) or ""
            role = _LC_TYPE_TO_ROLE.get(lc_type, "system")
            serialized.append(
                {
                    "role": role,
                    "content": getattr(msg, "content", "") or "",
                    "created_at": now_iso,
                    "model": extra.get("model"),
                    "rag_trace": extra.get("rag_trace"),
                    "latency_ms": extra.get("latency_ms"),
                    "status": extra.get("status"),
                    "error_message": extra.get("error_message"),
                }
            )
        db.save_session(user_id, session_id, serialized)

    def load(self, user_id: str, session_id: str) -> list:
        rows = db.load_session_messages(user_id, session_id)
        messages: list = []
        for r in rows:
            role = r.get("role")
            content = r.get("content") or ""
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
        return messages

    def list_sessions(self, user_id: str) -> list:
        return [r["session_id"] for r in db.list_sessions_info(user_id)]

    def delete_session(self, user_id: str, session_id: str) -> bool:
        return db.delete_session(user_id, session_id)

    def get_session_messages_with_trace(self, user_id: str, session_id: str) -> list:
        """给 HTTP 路由使用：返回对齐前端 MessageInfo 格式的消息列表。"""
        rows = db.load_session_messages(user_id, session_id)
        out = []
        for r in rows:
            role = r.get("role")
            out.append(
                {
                    "type": _ROLE_TO_LC_TYPE.get(role, role or "system"),
                    "content": r.get("content") or "",
                    "timestamp": r.get("created_at") or "",
                    "rag_trace": r.get("rag_trace"),
                }
            )
        return out

    def list_sessions_with_info(self, user_id: str) -> list:
        return db.list_sessions_info(user_id)


def _current_date_system_message() -> SystemMessage:
    """每次请求使用的当前日期（不写入持久化历史，仅注入本轮调用）。"""
    today_zh = f"{datetime.now().year}年{datetime.now().month}月{datetime.now().day}日"
    return SystemMessage(content=f"当前日期是：{today_zh}。")


def create_agent_instance():
    model = init_chat_model(
        model=MODEL,
        model_provider="openai",
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.3,
        stream_usage=True,
    )

    agent = create_agent(
        model=model,
        tools=[get_current_weather, search_knowledge_base],
        system_prompt=(
    "You are AICIS情报助手 that loves to help users. "
    "When responding, you may use tools to assist. "
    "Use search_knowledge_base when users ask questions related to the pharmaceutical industry. "
    "Do not call the same tool repeatedly in one turn. At most one knowledge tool call per turn. "
    "Once you call search_knowledge_base and receive its result, you MUST immediately produce the Final Answer based on that result. "
    "After receiving search_knowledge_base result, you MUST NOT call any tool again (including get_current_weather or search_knowledge_base). "
    "If the retrieved context is insufficient, answer honestly that you don't know instead of making up facts. "
    "If search_knowledge_base returns 'No relevant documents found in the knowledge base.', state clearly that no relevant information was found in the current retrieval scope, do not fabricate facts, and suggest 1-2 concrete next steps (e.g., broaden time range or add specific keywords). "
    "If tool results include a Step-back Question/Answer, use that general principle to reason and answer, "
    "but do not reveal chain-of-thought. "
    "If you don't know the answer, admit it honestly. "
    "Always respond in the same language as the user's latest message unless the user explicitly asks for another language. "
    "When presenting multiple facts, attach the corresponding source link right after each fact (inline citation), instead of listing all sources only at the end,with no omissions allowed. "
    "When providing information from retrieved documents, ALWAYS include the source URL in Markdown format. "
    "Format sources as: [标题](URL) so users can click directly."
),
    )
    return agent, model


_REQUIRED_ENV = {"ARK_API_KEY": API_KEY, "MODEL": MODEL, "BASE_URL": BASE_URL}
_missing = [k for k, v in _REQUIRED_ENV.items() if not v]
if _missing:
    raise RuntimeError(
        f"启动失败：以下环境变量缺失或为空：{', '.join(_missing)}。"
        "请检查项目根目录的 .env 文件或系统环境变量配置。"
    )

agent, model = create_agent_instance()

storage = ConversationStorage()


def _agent_model_label() -> str:
    """主对话模型在 LangChain 中的可读名称（与 rag_pipeline grader 日志风格对齐）。"""
    try:
        return (
            getattr(model, "model_name", None)
            or getattr(model, "model", None)
            or MODEL
            or "(unknown)"
        )
    except Exception:
        return MODEL or "(unknown)"


def _log_agent_model(phase: str) -> None:
    """stderr：Agent / 摘要 环节使用的模型。"""
    print(f"[AGENT] {phase} 当前模型：{_agent_model_label()}", file=sys.stderr)


def summarize_old_messages(model, messages: list) -> str:
    """将旧消息总结为摘要"""
    old_conversation = "\n".join([
        f"{'用户' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in messages
    ])

    summary_prompt = f"""请总结以下对话的关键信息：

{old_conversation}
总结（包含用户信息、重要事实、待办事项）："""

    _log_agent_model("历史摘要")
    summary = model.invoke(summary_prompt).content
    return summary


def _build_extra_for_assistant(
    total_messages: int,
    *,
    rag_trace,
    model_label: str,
    latency_ms: int,
    status: str,
    error_message,
):
    """构造 extra_message_data：仅最后一条 assistant 携带元数据。"""
    return [None] * (total_messages - 1) + [
        {
            "rag_trace": rag_trace,
            "model": model_label,
            "latency_ms": latency_ms,
            "status": status,
            "error_message": error_message,
        }
    ]


def chat_with_agent(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """使用 Agent 处理用户消息并返回响应"""
    messages = storage.load(user_id, session_id)
    runtime = ToolRuntime()

    if len(messages) > 50:
        summary = summarize_old_messages(model, messages[:40])

        messages = [
            SystemMessage(content=f"之前的对话摘要：\n{summary}")
        ] + messages[40:]

    messages.append(HumanMessage(content=user_text))
    invoke_messages = [_current_date_system_message()] + messages
    _log_agent_model("对话生成（非流式）")

    status = "ok"
    error_message = None
    response_content = ""
    caught_exc: BaseException | None = None
    start_ts = time.perf_counter()
    try:
        result = agent.invoke(
            {"messages": invoke_messages},
            config={
                "recursion_limit": 8,
                "configurable": {"tool_runtime": runtime},
            },
        )

        if isinstance(result, dict):
            if "output" in result:
                response_content = result["output"]
            elif "messages" in result and result["messages"]:
                msg = result["messages"][-1]
                response_content = getattr(msg, "content", str(msg))
            else:
                response_content = str(result)
        elif hasattr(result, "content"):
            response_content = result.content
        else:
            response_content = str(result)
    except Exception as e:  # noqa: BLE001
        status = "error"
        error_message = str(e)[:2000]
        response_content = ""
        caught_exc = e
    latency_ms = int((time.perf_counter() - start_ts) * 1000)

    messages.append(AIMessage(content=response_content))

    rag_context = runtime.get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    extra_message_data = _build_extra_for_assistant(
        len(messages),
        rag_trace=rag_trace,
        model_label=_agent_model_label(),
        latency_ms=latency_ms,
        status=status,
        error_message=error_message,
    )
    try:
        storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)
    except Exception as save_exc:  # noqa: BLE001
        # 保存失败不掩盖真正的业务错误，只打日志
        print(f"[AGENT] 保存会话失败：{save_exc}", file=sys.stderr)

    if caught_exc is not None:
        raise caught_exc

    return {
        "response": response_content,
        "rag_trace": rag_trace,
    }


async def chat_with_agent_stream(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """使用 Agent 处理用户消息并流式返回响应。
    
    架构：使用统一输出队列 + 后台任务，确保 RAG 检索步骤在工具执行期间实时推送，
    而非等待工具完成后才显示。
    """
    print(f"[AGENT_STREAM] 开始处理: user_text={user_text[:50]}...", file=sys.stderr)
    messages = storage.load(user_id, session_id)
    runtime = ToolRuntime()

    output_queue = asyncio.Queue()

    class _RagStepProxy:
        """代理对象：将 emit_rag_step 的原始 step dict 包装后放入统一输出队列。"""
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})

    runtime.set_rag_step_queue(_RagStepProxy())

    if len(messages) > 50:
        summary = summarize_old_messages(model, messages[:40])
        messages = [
            SystemMessage(content=f"之前的对话摘要：\n{summary}")
        ] + messages[40:]

    messages.append(HumanMessage(content=user_text))
    invoke_messages = [_current_date_system_message()] + messages

    full_response = ""
    agent_error_msg: str | None = None

    async def _agent_worker():
        nonlocal full_response, agent_error_msg
        try:
            print(f"[AGENT_WORKER] 开始执行 agent", file=sys.stderr)
            _log_agent_model("对话生成（流式）")
            async for msg, metadata in agent.astream(
                {"messages": invoke_messages},
                stream_mode="messages",
                config={
                    "recursion_limit": 8,
                    "configurable": {"tool_runtime": runtime},
                },
            ):
                if not isinstance(msg, AIMessageChunk):
                    continue
                if getattr(msg, "tool_call_chunks", None):
                    continue

                content = ""
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, str):
                            content += block
                        elif isinstance(block, dict) and block.get("type") == "text":
                            content += block.get("text", "")

                if content:
                    full_response += content
                    await output_queue.put({"type": "content", "content": content})
        except Exception as e:
            print(f"[AGENT_WORKER] 异常: {type(e).__name__}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            agent_error_msg = str(e)[:2000]
            await output_queue.put({"type": "error", "content": str(e)})
        finally:
            print(f"[AGENT_WORKER] 完成", file=sys.stderr)
            await output_queue.put(None)

    start_ts = time.perf_counter()
    agent_task = asyncio.create_task(_agent_worker())

    try:
        while True:
            try:
                event = await asyncio.wait_for(output_queue.get(), timeout=60)
            except asyncio.TimeoutError:
                continue
            if event is None:
                break
            try:
                yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                print(f"[AGENT_STREAM] yield 错误: {e}", file=sys.stderr)
                break
    except GeneratorExit:
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
        raise
    finally:
        runtime.set_rag_step_queue(None)
        if not agent_task.done():
             agent_task.cancel()

    latency_ms = int((time.perf_counter() - start_ts) * 1000)

    rag_context = runtime.get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    if rag_trace:
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

    yield "data: [DONE]\n\n"

    status = "error" if agent_error_msg else "ok"

    messages.append(AIMessage(content=full_response))
    extra_message_data = _build_extra_for_assistant(
        len(messages),
        rag_trace=rag_trace,
        model_label=_agent_model_label(),
        latency_ms=latency_ms,
        status=status,
        error_message=agent_error_msg,
    )
    try:
        storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)
    except Exception as save_exc:  # noqa: BLE001
        print(f"[AGENT_STREAM] 保存会话失败：{save_exc}", file=sys.stderr)
