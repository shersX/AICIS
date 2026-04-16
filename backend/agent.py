from dotenv import load_dotenv
import os
import json
import asyncio
import sys
import tempfile
from filelock import FileLock, Timeout
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from backend.tools import get_current_weather, search_knowledge_base, ToolRuntime
from datetime import datetime

load_dotenv()

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")

class ConversationStorage:
    """对话存储（跨进程 filelock + 临时文件原子替换，避免并发丢失更新与半截 JSON）。"""

    def __init__(self, storage_file: str = None):
        if storage_file:
            storage_path = os.path.abspath(storage_file)
        else:
            package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            data_dir = os.path.join(package_root, "data")
            os.makedirs(data_dir, exist_ok=True)
            storage_path = os.path.join(data_dir, "customer_service_history.json")

        self.storage_file = storage_path
        lock_path = f"{storage_path}.lock"
        lock_timeout = float(os.getenv("CONVERSATION_STORAGE_LOCK_TIMEOUT", "60"))
        self._file_lock = FileLock(lock_path, timeout=lock_timeout)

    def _load_unlocked(self) -> dict:
        """加载数据（调用方须已持有 self._file_lock）。"""
        if not os.path.exists(self.storage_file):
            return {}
        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _atomic_write_json_unlocked(self, data: dict) -> None:
        """原子写入 JSON：临时文件 fsync 后 os.replace（须在锁内调用）。"""
        dir_name = os.path.dirname(self.storage_file) or "."
        fd, tmp_path = tempfile.mkstemp(
            dir=dir_name,
            prefix=".customer_service_history.",
            suffix=".tmp.json",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.storage_file)
        except BaseException:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _acquire_lock(self):
        try:
            self._file_lock.acquire()
        except Timeout as e:
            raise RuntimeError(
                f"会话存储文件锁获取超时（{self._file_lock.lock_file}），"
                "可适当增大环境变量 CONVERSATION_STORAGE_LOCK_TIMEOUT（秒）。"
            ) from e

    def save(self, user_id: str, session_id: str, messages: list, metadata: dict = None, extra_message_data: list = None):
        """保存对话"""
        self._acquire_lock()
        try:
            data = self._load_unlocked()

            if user_id not in data:
                data[user_id] = {}

            serialized = []
            for idx, msg in enumerate(messages):
                record = {
                    "type": msg.type,
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat(),
                }
                if extra_message_data and idx < len(extra_message_data):
                    extra = extra_message_data[idx] or {}
                    if "rag_trace" in extra:
                        record["rag_trace"] = extra["rag_trace"]
                serialized.append(record)

            data[user_id][session_id] = {
                "messages": serialized,
                "metadata": metadata or {},
                "updated_at": datetime.now().isoformat(),
            }

            self._atomic_write_json_unlocked(data)
        finally:
            self._file_lock.release()

    def load(self, user_id: str, session_id: str) -> list:
        """加载对话"""
        self._acquire_lock()
        try:
            data = self._load_unlocked()
            if user_id not in data or session_id not in data[user_id]:
                return []

            messages = []
            for msg_data in data[user_id][session_id]["messages"]:
                if msg_data["type"] == "human":
                    messages.append(HumanMessage(content=msg_data["content"]))
                elif msg_data["type"] == "ai":
                    messages.append(AIMessage(content=msg_data["content"]))
                elif msg_data["type"] == "system":
                    messages.append(SystemMessage(content=msg_data["content"]))

            return messages
        finally:
            self._file_lock.release()

    def list_sessions(self, user_id: str) -> list:
        """列出用户的所有会话"""
        self._acquire_lock()
        try:
            data = self._load_unlocked()
            if user_id not in data:
                return []
            return list(data[user_id].keys())
        finally:
            self._file_lock.release()

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除指定用户的会话，返回是否删除成功"""
        self._acquire_lock()
        try:
            data = self._load_unlocked()
            if user_id not in data or session_id not in data[user_id]:
                return False

            del data[user_id][session_id]
            if not data[user_id]:
                del data[user_id]

            self._atomic_write_json_unlocked(data)
            return True
        finally:
            self._file_lock.release()

    def _load(self) -> dict:
        """加载完整存储快照（供 API 等读会话列表；与 save/delete 互斥）。"""
        self._acquire_lock()
        try:
            return self._load_unlocked()
        finally:
            self._file_lock.release()


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
    # 提取旧对话
    old_conversation = "\n".join([
        f"{'用户' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in messages
    ])

    # 生成摘要
    summary_prompt = f"""请总结以下对话的关键信息：

{old_conversation}
总结（包含用户信息、重要事实、待办事项）："""

    _log_agent_model("历史摘要")
    summary = model.invoke(summary_prompt).content
    return summary


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
    result = agent.invoke(
        {"messages": invoke_messages},
        config={
            "recursion_limit": 8,
            "configurable": {"tool_runtime": runtime},
        },
    )

    response_content = ""
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
    
    messages.append(AIMessage(content=response_content))

    rag_context = runtime.get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)

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

    # 统一输出队列：所有事件（content / rag_step）都汇入这里
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

    async def _agent_worker():
        """后台任务：运行 agent 并将内容 chunk 推入输出队列。"""
        nonlocal full_response
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
            await output_queue.put({"type": "error", "content": str(e)})
        finally:
            print(f"[AGENT_WORKER] 完成", file=sys.stderr)
            # 哨兵：通知主循环 agent 已完成
            await output_queue.put(None)

    # 启动后台任务
    agent_task = asyncio.create_task(_agent_worker())

    try:
        # 主循环：持续从统一队列取事件并 yield SSE
        # RAG 步骤在工具执行期间通过 call_soon_threadsafe 实时入队，不需要等 agent 产出 chunk
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
        # 客户端断开连接（AbortController）时，FastAPI 会向此生成器抛出 GeneratorExit
        # 我们必须在此处取消后台任务
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass  # 任务已成功取消
        raise  # 重新抛出 GeneratorExit 以便 FastAPI 正确处理关闭
    finally:
        # 正常结束或异常退出时清理
        runtime.set_rag_step_queue(None)
        if not agent_task.done():
             agent_task.cancel()

    # 获取 RAG trace
    rag_context = runtime.get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    # 发送 trace 信息
    if rag_trace:
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

    # 发送结束信号
    yield "data: [DONE]\n\n"

    # 保存对话
    messages.append(AIMessage(content=full_response))
    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)
