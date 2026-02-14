from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.core.utils import now_ts


Route = Literal["RAG", "NO_RAG", "TOOL", "CLARIFY"]


class GraphState(TypedDict, total=False):
    conversation_id: str
    request_id: str
    user_id: str | None

    query: str
    history: list[dict[str, Any]]

    route: Route
    retrieved: list[dict[str, Any]]

    answer: str
    citations: list[dict[str, Any]]


@dataclass
class GraphDeps:
    router_mode: str
    retriever: Any
    llm: Any


def react_route(query: str, history: list[dict[str, Any]], llm: Any) -> Route:
    history_str = "\n".join(
        [f"{m.get('role', '')}: {m.get('content', '')}" for m in history[-5:]]
    )

    system_prompt = (
        "你是路由判定器，只输出 RAG 或 NO_RAG。"
        "当用户问题需要外部业务知识/事实（套餐、资费、办理规则等）时输出 RAG。"
        "当用户是在追问解释或引用对话历史时输出 NO_RAG。"
        "不要轻易输出 RAG，除非确实需要知识库知识。"
        "只输出一个词：RAG 或 NO_RAG。"
    )
    user_prompt = f"对话历史:\n{history_str}\n\n用户问题:\n{query}\n"

    try:
        result = (llm.chat(messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]) or "").strip().upper()
    except Exception:
        return "RAG"

    return "RAG" if "RAG" in result else "NO_RAG"


def heuristic_route(query: str, history: list[dict[str, Any]]) -> Route:
    q = (query or "").strip()

    # 追问解释/指代：优先基于历史回答（不需要查知识库）
    if history and any(
        k in q
        for k in [
            "啥意思",
            "什么意思",
            "怎么理解",
            "这是什么意思",
            "这句话",
            "上面",
            "刚才",
            "你说的",
            "那个",
            "这个",
        ]
    ):
        return "NO_RAG"

    # 业务查询（示例关键词，可按你的电信业务扩充）
    if any(k in q for k in ["查话费", "查余额", "查流量", "查订单", "物流", "余额", "账单", "详单"]):
        return "TOOL"

    # 指代比较：通常可脱离知识库，只基于历史候选项做推荐
    if any(k in q for k in ["这几个", "哪个", "哪一个", "性价比", "对比", "比较", "推荐哪个"]):
        if history:
            return "NO_RAG"

    # 默认：走 RAG
    return "RAG"


def build_graph(deps: GraphDeps):
    g = StateGraph(GraphState)

    persona = (
        "你是一个正在与用户对话的上海电信员工，名叫晶晶，性别女。"
        "你具备以下特性：【"
        "1、你回答用户问题时会使用精准、清晰的纯文本（不要用markdown格式）。"
        "2、你更偏向于为用户提供完整的链接（包括小程序链接），让用户通过你的回答来自助操作，而不会亲自帮用户进行一些查询、办理等操作。"
        "】"
    )

    def _clarify_question(query: str) -> str:
        q = (query or "").strip()
        if "套餐" in q:
            return "方便说下您的月预算和主要需求（流量/通话/宽带）吗？"
        return "方便补充一下您的具体需求或使用场景吗？"

    def _sanitize_answer(text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""

        # 去掉常见 Markdown/格式符号，尽量保证“纯文本”
        for ch in ("`", "*", "#", ">", "|", "\\t"):
            t = t.replace(ch, "")
        t = t.replace("\r", " ").replace("\n", " ")
        while "  " in t:
            t = t.replace("  ", " ")
        return t

    def _format_history(history: list[dict[str, Any]], *, max_messages: int = 8, max_chars: int = 1200) -> str:
        if not history:
            return ""

        items: list[str] = []
        for m in history[-max_messages:]:
            role = (m.get("role") or "").strip()
            content = str(m.get("content") or "").strip()
            if not content:
                continue

            # 单条过长会污染上下文，这里做轻量截断
            if len(content) > 300:
                content = content[:300] + "…"

            if role == "user":
                items.append(f"用户：{content}")
            elif role == "assistant":
                items.append(f"客服：{content}")

        text = "\n".join(items).strip()
        if len(text) > max_chars:
            text = text[-max_chars:]
        return text

    def node_route(state: GraphState) -> GraphState:
        query = state.get("query", "")
        history = state.get("history", [])

        if deps.router_mode == "heuristic":
            route: Route = heuristic_route(query, history)
        elif deps.router_mode == "react":
            route: Route = react_route(query, history, deps.llm)
        else:
            route = heuristic_route(query, history)

        return {"route": route}

    def node_answer(state: GraphState) -> GraphState:
        query = state.get("query", "")
        route = state.get("route", "NO_RAG")
        history = state.get("history", [])

        tool_retrieved: list[dict[str, Any]] = []
        tool_citations: list[dict[str, Any]] = []

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge",
                    "description": "在业务知识库中检索与问题相关的知识条目。用于套餐/资费/流量/办理规则等问题。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "用户问题"},
                            "top_k": {"type": "integer", "description": "返回条数", "default": 5},
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        def tool_executor(name: str, args: dict[str, Any]) -> Any:
            nonlocal tool_retrieved, tool_citations
            if name != "search_knowledge":
                return {"error": f"unknown tool: {name}"}

            q = str(args.get("query") or query)
            top_k = args.get("top_k")
            try:
                top_k_int = int(top_k) if top_k is not None else None
            except Exception:
                top_k_int = None

            docs = deps.retriever.retrieve(q)
            if top_k_int is not None and top_k_int > 0:
                docs = docs[:top_k_int]

            tool_retrieved = [
                {
                    "id": d.id,
                    "score": d.score,
                    "question": d.question,
                    "knowledge": d.knowledge,
                }
                for d in docs
            ]
            tool_citations = [d.to_citation() for d in docs]
            return {"docs": tool_retrieved}

        # system 提示：在需要业务知识时先调用工具；无知识则追问澄清
        system_policy = (
            "当用户问题涉及套餐/资费/定向流量/办理规则等业务知识时，你必须先调用工具 search_knowledge 查询。"
            "拿到工具结果后再回答；如果工具返回 docs 为空或不足以支撑回答，请改为提出一个澄清问题，不要编造。"
            "输出必须是纯文本，不要使用markdown。"
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": persona},
            {"role": "system", "content": system_policy},
        ]

        hist_text = _format_history(history)
        if hist_text:
            messages.append({"role": "system", "content": f"对话历史（供参考）：\n{hist_text}"})

        messages.append({"role": "user", "content": query})

        # 恢复为 LLMWrapper 的 chat/chat_with_tools 调用
        if route == "NO_RAG":
            answer = deps.llm.chat(messages=messages)
        else:
            if hasattr(deps.llm, "chat_with_tools"):
                answer = deps.llm.chat_with_tools(
                    messages=messages,
                    tools=tools,
                    tool_executor=tool_executor,
                )
            else:
                answer = deps.llm.chat(messages=messages)
        answer = _sanitize_answer(answer)
        if route in ("RAG", "TOOL") and not tool_retrieved:
            answer = answer or _clarify_question(query)
        if not answer:
            answer = _clarify_question(query)
        return {"answer": answer, "retrieved": tool_retrieved, "citations": tool_citations}

    def node_tool_placeholder(state: GraphState) -> GraphState:
        # 目前 TOOL 路由也交给 node_answer 处理（function call / 澄清）。
        query = state.get("query", "")
        return {"answer": _clarify_question(query), "citations": [], "retrieved": []}

    def node_clarify(state: GraphState) -> GraphState:
        query = state.get("query", "")
        return {"answer": _clarify_question(query), "citations": [], "retrieved": []}

    g.add_node("route", node_route)
    g.add_node("answer", node_answer)
    g.add_node("tool", node_tool_placeholder)
    g.add_node("clarify", node_clarify)

    g.set_entry_point("route")

    def branch(state: GraphState) -> str:
        return state.get("route", "NO_RAG")

    g.add_conditional_edges(
        "route",
        branch,
        {
            "RAG": "answer",
            "NO_RAG": "answer",
            "TOOL": "answer",
            "CLARIFY": "clarify",
        },
    )

    g.add_edge("answer", END)
    g.add_edge("tool", END)
    g.add_edge("clarify", END)

    return g.compile()
