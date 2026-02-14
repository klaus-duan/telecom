from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from app.core.schemas import ChatRequest, EndRequest, EndResponse
from app.core.utils import new_id, now_ts


def make_router(*, memory, graph, settings, pg_store=None):
    r = APIRouter()

    from fastapi.responses import JSONResponse
    @r.post("/chat", response_class=JSONResponse)
    def chat(req: ChatRequest) -> JSONResponse:
        conversation_id = req.conversation_id or new_id()
        request_id = req.request_id

        cached = memory.get_cached_response(conversation_id, request_id)
        if cached:
            resp = {
                "conversation_id": conversation_id,
                "request_id": request_id,
                "answer": cached.get("answer", ""),
            }
            # 每个字段单独一行输出
            text = f"conversation_id: {resp['conversation_id']}\nrequest_id: {resp['request_id']}\nanswer: {resp['answer']}\n"
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(content=text)

        # 幂等保护：同 request_id 并发只允许一个 inflight
        if not memory.mark_inflight(conversation_id, request_id):
            raise HTTPException(status_code=409, detail="Duplicate inflight request_id")

        try:
            # 这里确保 request_id 唯一；重复直接拒绝（或你也可以返回 cached）
            if not memory.ensure_request_id_unique(conversation_id, request_id):
                raise HTTPException(status_code=409, detail="Duplicate request_id")

            history = memory.get_recent_messages(conversation_id, limit=20)

            state_in: dict[str, Any] = {
                "conversation_id": conversation_id,
                "request_id": request_id,
                "user_id": req.user_id,
                "query": req.message,
                "history": history,
            }

            out = graph.invoke(state_in)

            answer = (out.get("answer") or "").strip()
            route = out.get("route") or "NO_RAG"
            citations = out.get("citations") or []

            # 写入 Redis 历史（user+assistant）
            memory.append_messages(
                conversation_id,
                [
                    {
                        "message_id": new_id(),
                        "request_id": request_id,
                        "role": "user",
                        "content": req.message,
                        "ts": now_ts(),
                    },
                    {
                        "message_id": new_id(),
                        "request_id": request_id,
                        "answer_id": new_id(),
                        "role": "assistant",
                        "content": answer,
                        "ts": now_ts(),
                        "meta": {"citations": citations},
                    },
                ],
            )

            resp = {
                "conversation_id": conversation_id,
                "request_id": request_id,
                "answer": answer,
            }
            memory.cache_response(conversation_id, request_id, resp)
            text = f"conversation_id: {resp['conversation_id']}\nrequest_id: {resp['request_id']}\nanswer: {resp['answer']}\n"
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(content=text)
        finally:
            memory.clear_inflight(conversation_id, request_id)

    @r.post("/end", response_model=EndResponse)
    def end(req: EndRequest) -> EndResponse:
        if pg_store is None:
            raise HTTPException(status_code=500, detail="Postgres store not configured")

        if not getattr(settings, "postgres_dsn", ""):
            raise HTTPException(status_code=500, detail="Missing POSTGRES_DSN/DATABASE_URL")

        conversation_id = str(req.conversation_id)
        msgs = memory.get_all_messages(conversation_id)
        try:
            pg_store.persist_chat_history_from_messages(
                conversation_id=conversation_id, messages=msgs
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Persist to Postgres failed: {e}")

        memory.delete_conversation(conversation_id)
        return EndResponse(conversation_id=conversation_id, flushed_message_count=len(msgs))

    return r
