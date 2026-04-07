from typing import Any, Literal, TypedDict, List, Optional
import os,re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import sys
import traceback

from backend.rag_utils import retrieve_documents, step_back_expand, generate_hypothetical_document
load_dotenv()

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
GRADE_MODEL = os.getenv("GRADE_MODEL", "Pro/zai-org/GLM-5")

_grader_model = None
_router_model = None


def _get_grader_model():
    global _grader_model
    if not API_KEY or not GRADE_MODEL:
        return None
    if _grader_model is None:
        _grader_model = init_chat_model(
            model=GRADE_MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
            stream_usage=True,
        )
    return _grader_model


def _get_router_model():
    global _router_model
    if not API_KEY or not MODEL:
        return None
    if _router_model is None:
        _router_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
            stream_usage=True,
        )
    return _router_model


GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Output ONLY 'yes' or 'no' — no extra words, no explanation.\n"
    "If the context has keywords OR semantic meaning related to the question, output yes.\n"
    "If uncertain or not relevant, output no.\n"
    "Here is the user question: {question} \n"    
    "Here is the retrieved document: \n\n {context} \n\n"
)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class RewriteStrategy(BaseModel):
    """Choose a query expansion strategy."""

    strategy: Literal["step_back", "hyde", "complex"]


class RAGState(TypedDict):
    question: str
    query: str
    context: str
    docs: List[dict]
    route: Optional[str]
    expansion_type: Optional[str]
    expanded_query: Optional[str]
    step_back_question: Optional[str]
    step_back_answer: Optional[str]
    hypothetical_doc: Optional[str]
    time_filter_expr: Optional[str]
    time_range_text: Optional[str]
    is_time_sensitive: Optional[bool]
    rag_trace: Optional[dict]
    tool_runtime: Optional[Any]


def _emit_rag_from_state(state: RAGState, icon: str, label: str, detail: str = "") -> None:
    """通过 graph state 中的 tool_runtime 推送步骤，避免依赖模块级全局队列。"""
    rt = state.get("tool_runtime")
    if rt is not None and hasattr(rt, "emit_rag_step"):
        rt.emit_rag_step(icon, label, detail)


def _parse_relative_days(question: str) -> Optional[int]:
    lowered = question.lower()
    if "最近一周" in question or "近一周" in question:
        return 7
    if "最近三天" in question or "近三天" in question:
        return 3
    if "最近两天" in question or "近两天" in question:
        return 2
    if "最近一天" in question or "近一天" in question:
        return 1
    if "本周" in question:
        return 7

    match = re.search(r"(最近|近)\s*(\d+)\s*天", question)
    if match:
        return int(match.group(2))
    match_en = re.search(r"(last|past)\s*(\d+)\s*days?", lowered)
    if match_en:
        return int(match_en.group(2))
    # English common expressions
    if "past week" in lowered or "last week" in lowered:
        return 7
    if "past month" in lowered or "last month" in lowered:
        return 30
    if "past day" in lowered or "last day" in lowered:
        return 1
    if "past three days" in lowered or "last three days" in lowered:
        return 3
    if "past two days" in lowered or "last two days" in lowered:
        return 2
    if "past one day" in lowered or "last one day" in lowered:
        return 1
    match_en_week = re.search(r"(last|past)\s*(\d+)\s*weeks?", lowered)
    if match_en_week:
        return int(match_en_week.group(2)) * 7
    match_en_month = re.search(r"(last|past)\s*(\d+)\s*months?", lowered)
    if match_en_month:
        return int(match_en_month.group(2)) * 30
    return None


def _build_publish_time_filter_expr(question: str) -> tuple[str, Optional[str], bool]:
    days = _parse_relative_days(question)
    if not days or days <= 0:
        return "", None, False

    now_dt = datetime.now()
    start_dt = now_dt - timedelta(days=days)
    start_sec = int(start_dt.timestamp())
    end_sec = int(now_dt.timestamp())
    start_ms = start_sec * 1000
    end_ms = end_sec * 1000
    expr = (
        f"((publish_time >= {start_sec} and publish_time <= {end_sec}) "
        f"or (publish_time >= {start_ms} and publish_time <= {end_ms}))"
    )
    time_range_text = f"{start_dt.strftime('%Y-%m-%d')}..{now_dt.strftime('%Y-%m-%d')}"
    return expr, time_range_text, True


def _format_docs(docs: List[dict]) -> str:
    if not docs:
        return ""
    chunks = []
    for i, doc in enumerate(docs, 1):
        title = doc.get("title", "无标题")
        source = doc.get("origin_name", doc.get("filename", "Unknown"))
        text = doc.get("text", doc.get("summary", ""))
        url = doc.get("url", "")
        
        item = f"[{i}] {title}\n来源: {source}"
        if url:
            item += f"\n链接: {url}"
        if text:
            item += f"\n摘要: {text}"
        chunks.append(item)
    return "\n\n---\n\n".join(chunks)


def retrieve_initial(state: RAGState) -> RAGState:
    print(f"[RAG_PIPELINE] retrieve_initial 开始, question={state['question']}", file=sys.stderr)
    query = state["question"]
    filter_expr = state.get("time_filter_expr") or ""
    time_range_text = state.get("time_range_text")
    is_time_sensitive = bool(state.get("is_time_sensitive"))
    _emit_rag_from_state(state,"🔍", "正在检索新闻知识库...", f"查询: {query[:50]}")
    retrieved = retrieve_documents(query, top_k=8, filter_expr=filter_expr)
    results = retrieved.get("docs", [])
    retrieve_meta = retrieved.get("meta", {})
    context = _format_docs(results)
    _emit_rag_from_state(state,
        "📰",
        "新闻检索",
        f"候选 {retrieve_meta.get('candidate_k', 0)}"
    )
    _emit_rag_from_state(state,"✅", f"检索完成，找到 {len(results)} 条新闻", f"模式: {retrieve_meta.get('retrieval_mode', 'hybrid')}")
    rag_trace = {
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": query,
        "expanded_query": query,
        "retrieved_chunks": results,
        "initial_retrieved_chunks": results,
        "retrieval_stage": "initial",
        "rerank_enabled": retrieve_meta.get("rerank_enabled"),
        "rerank_applied": retrieve_meta.get("rerank_applied"),
        "rerank_model": retrieve_meta.get("rerank_model"),
        "rerank_endpoint": retrieve_meta.get("rerank_endpoint"),
        "rerank_error": retrieve_meta.get("rerank_error"),
        "retrieval_mode": retrieve_meta.get("retrieval_mode"),
        "candidate_k": retrieve_meta.get("candidate_k"),
        "time_filter_expr": filter_expr,
        "time_range_text": time_range_text,
        "is_time_sensitive": is_time_sensitive,
    }
    return {
        "query": query,
        "docs": results,
        "context": context,
        "rag_trace": rag_trace,
    }


def grade_documents_node(state: RAGState) -> RAGState:
    print(f"[RAG_PIPELINE] grade_documents_node 开始", file=sys.stderr)
    grader = _get_grader_model()
    _emit_rag_from_state(state,"📊", "正在评估文档相关性...")
    if not grader:
        print(f"[RAG_PIPELINE] grade_documents_node: grader 为 None", file=sys.stderr)
        grade_update = {
            "grade_score": "unknown",
            "grade_route": "rewrite_question",
            "rewrite_needed": True,
        }
        rag_trace = state.get("rag_trace", {}) or {}
        rag_trace.update(grade_update)
        return {"route": "rewrite_question", "rag_trace": rag_trace}
    question = state["question"]
    context = state.get("context", "")
    prompt = GRADE_PROMPT.format(question=question, context=context)
    print(f"[RAG_PIPELINE] grade_documents_node: 调用 grader, prompt长度={len(prompt)}", file=sys.stderr)
    try:
        raw_response = grader.invoke([{"role": "user", "content": prompt}])
        
        print(f"[RAG_PIPELINE] grader 当前模型：{grader.model_name}， 原始响应 type={type(raw_response)}, content={repr(raw_response.content)}", file=sys.stderr)
        
        # 尝试解析文本响应
        text = raw_response.content.lower().strip()
        if "yes" in text and "no" not in text[:10]:
            score = "yes"
        elif "no" in text and "yes" not in text[:10]:
            score = "no"
        else:
            # 尝试提取 yes/no
            match = re.search(r'\b(yes|no)\b', text)
            score = match.group(1) if match else "no"
        
        print(f"[RAG_PIPELINE] 解析后 score={score}", file=sys.stderr)
    except Exception as e:
        print(f"[RAG_PIPELINE] grade_documents_node 异常: {type(e).__name__}: {e}", file=sys.stderr)
        
        traceback.print_exc(file=sys.stderr)
        grade_update = {
            "grade_score": "error",
            "grade_route": "rewrite_question",
            "rewrite_needed": True,
            "grade_error": str(e),
        }
        rag_trace = state.get("rag_trace", {}) or {}
        rag_trace.update(grade_update)
        return {"route": "rewrite_question", "rag_trace": rag_trace}
    # score 已在上面解析完成
    route = "generate_answer" if score == "yes" else "rewrite_question"
    if route == "generate_answer":
        _emit_rag_from_state(state,"✅", "文档相关性评估通过", f"评分: {score}")
    else:
        _emit_rag_from_state(state,"⚠️", "文档相关性不足，将重写查询", f"评分: {score}")
    grade_update = {
        "grade_score": score,
        "grade_route": route,
        "rewrite_needed": route == "rewrite_question",
    }
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update(grade_update)
    print(f"[RAG_PIPELINE] grade_documents_node 完成, score={score}, route={route}", file=sys.stderr)
    return {"route": route, "rag_trace": rag_trace}


def rewrite_question_node(state: RAGState) -> RAGState:

    question = state["question"]
    is_time_sensitive = bool(state.get("is_time_sensitive"))
    _emit_rag_from_state(state,"✏️", "正在重写查询...")
    router = _get_router_model()
    strategy = "step_back"
    if router:
        prompt = (
            "请根据用户问题选择最合适的查询扩展策略，仅输出策略名。\n"
            "- step_back：包含具体名称、日期、代码等细节，需要先理解通用概念的问题。\n"
            "- hyde：模糊、概念性、需要解释或定义的问题。\n"
            "- complex：多步骤、需要分解或综合多种信息的复杂问题。\n"
            f"用户问题：{question}"
        )
        try:
            # 暂时移除 structured_output，查看原始响应
            raw_response = router.invoke([{"role": "user", "content": prompt}])
            print(f"[RAG_PIPELINE] router 原始响应: {repr(raw_response.content)}", file=sys.stderr)

            # 解析策略
            text = raw_response.content.lower().strip()
            match = re.search(r'\b(step_back|hyde|complex)\b', text)
            strategy = match.group(1) if match else "step_back"
            print(f"[RAG_PIPELINE] 解析后 strategy={strategy}", file=sys.stderr)
        except Exception as e:
            print(f"[RAG_PIPELINE] router 异常: {e}", file=sys.stderr)
            strategy = "step_back"

    if is_time_sensitive and strategy in ("step_back", "complex"):
        strategy = "hyde"
        _emit_rag_from_state(state,"⏱️", "检测到时效查询", "已禁用 step-back，避免偏离时间窗口")

    expanded_query = question
    step_back_question = ""
    step_back_answer = ""
    hypothetical_doc = ""

    if strategy in ("step_back", "complex"):
        _emit_rag_from_state(state,"🧠", f"使用策略: {strategy}", "生成退步问题")
        step_back = step_back_expand(question)
        step_back_question = step_back.get("step_back_question", "")
        step_back_answer = step_back.get("step_back_answer", "")
        expanded_query = step_back.get("expanded_query", question)

    if strategy in ("hyde", "complex"):
        _emit_rag_from_state(state,"📝", "HyDE 假设性文档生成中...")
        hypothetical_doc = generate_hypothetical_document(question)

    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update({
        "rewrite_strategy": strategy,
        "rewrite_query": expanded_query,
    })

    return {
        "expansion_type": strategy,
        "expanded_query": expanded_query,
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "hypothetical_doc": hypothetical_doc,
        "rag_trace": rag_trace,
    }


def retrieve_expanded(state: RAGState) -> RAGState:
    strategy = state.get("expansion_type") or "step_back"
    filter_expr = state.get("time_filter_expr") or ""
    _emit_rag_from_state(state,"🔄", "使用扩展查询重新检索...", f"策略: {strategy}")
    results: List[dict] = []
    rerank_applied_any = False
    rerank_enabled_any = False
    rerank_model = None
    rerank_endpoint = None
    rerank_errors = []
    retrieval_mode = None
    candidate_k = None

    if strategy in ("hyde", "complex"):
        hypothetical_doc = state.get("hypothetical_doc") or generate_hypothetical_document(state["question"])
        retrieved_hyde = retrieve_documents(hypothetical_doc, top_k=8, filter_expr=filter_expr)
        results.extend(retrieved_hyde.get("docs", []))
        hyde_meta = retrieved_hyde.get("meta", {})
        _emit_rag_from_state(state,
            "🧱",
            "HyDE 检索",
            f"候选 {hyde_meta.get('candidate_k', 0)} 条",
        )
        rerank_applied_any = rerank_applied_any or bool(hyde_meta.get("rerank_applied"))
        rerank_enabled_any = rerank_enabled_any or bool(hyde_meta.get("rerank_enabled"))
        rerank_model = rerank_model or hyde_meta.get("rerank_model")
        rerank_endpoint = rerank_endpoint or hyde_meta.get("rerank_endpoint")
        if hyde_meta.get("rerank_error"):
            rerank_errors.append(f"hyde:{hyde_meta.get('rerank_error')}")
        retrieval_mode = retrieval_mode or hyde_meta.get("retrieval_mode")
        candidate_k = candidate_k or hyde_meta.get("candidate_k")

    if strategy in ("step_back", "complex"):
        expanded_query = state.get("expanded_query") or state["question"]
        retrieved_stepback = retrieve_documents(expanded_query, top_k=8, filter_expr=filter_expr)
        results.extend(retrieved_stepback.get("docs", []))
        step_meta = retrieved_stepback.get("meta", {})
        _emit_rag_from_state(state,
            "🧱",
            "Step-back 检索",
            f"候选 {step_meta.get('candidate_k', 0)} 条",
        )
        rerank_applied_any = rerank_applied_any or bool(step_meta.get("rerank_applied"))
        rerank_enabled_any = rerank_enabled_any or bool(step_meta.get("rerank_enabled"))
        rerank_model = rerank_model or step_meta.get("rerank_model")
        rerank_endpoint = rerank_endpoint or step_meta.get("rerank_endpoint")
        if step_meta.get("rerank_error"):
            rerank_errors.append(f"step_back:{step_meta.get('rerank_error')}")
        retrieval_mode = retrieval_mode or step_meta.get("retrieval_mode")
        candidate_k = candidate_k or step_meta.get("candidate_k")

    deduped = []
    seen = set()
    for item in results:
        key = (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    # 扩展阶段可能合并了多路召回（如 hyde + step_back），
    # 这里统一重排展示名次，避免出现 1,2,3,4,5,4,5 这类重复名次。
    for idx, item in enumerate(deduped, 1):
        item["rrf_rank"] = idx

    context = _format_docs(deduped)
    _emit_rag_from_state(state,"✅", f"扩展检索完成，共 {len(deduped)} 个片段")
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update({
        "expanded_query": state.get("expanded_query") or state["question"],
        "step_back_question": state.get("step_back_question", ""),
        "step_back_answer": state.get("step_back_answer", ""),
        "hypothetical_doc": state.get("hypothetical_doc", ""),
        "expansion_type": strategy,
        "retrieved_chunks": deduped,
        "expanded_retrieved_chunks": deduped,
        "retrieval_stage": "expanded",
        "rerank_enabled": rerank_enabled_any,
        "rerank_applied": rerank_applied_any,
        "rerank_model": rerank_model,
        "rerank_endpoint": rerank_endpoint,
        "rerank_error": "; ".join(rerank_errors) if rerank_errors else None,
        "retrieval_mode": retrieval_mode,
        "candidate_k": candidate_k,
        "time_filter_expr": filter_expr,
        "time_range_text": state.get("time_range_text"),
        "is_time_sensitive": bool(state.get("is_time_sensitive")),
    })
    return {"docs": deduped, "context": context, "rag_trace": rag_trace}


def build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve_initial", retrieve_initial)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_question", rewrite_question_node)
    graph.add_node("retrieve_expanded", retrieve_expanded)

    graph.set_entry_point("retrieve_initial")
    graph.add_edge("retrieve_initial", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        lambda state: state.get("route"),
        {
            "generate_answer": END,
            "rewrite_question": "rewrite_question",
        },
    )
    graph.add_edge("rewrite_question", "retrieve_expanded")
    graph.add_edge("retrieve_expanded", END)
    return graph.compile()


rag_graph = build_rag_graph()


def run_rag_graph(question: str, tool_runtime: Optional[Any] = None) -> dict:
    time_filter_expr, time_range_text, is_time_sensitive = _build_publish_time_filter_expr(question)
    return rag_graph.invoke({
        "question": question,
        "query": question,
        "context": "",
        "docs": [],
        "route": None,
        "expansion_type": None,
        "expanded_query": None,
        "step_back_question": None,
        "step_back_answer": None,
        "hypothetical_doc": None,
        "time_filter_expr": time_filter_expr,
        "time_range_text": time_range_text,
        "is_time_sensitive": is_time_sensitive,
        "rag_trace": None,
        "tool_runtime": tool_runtime,
    })
