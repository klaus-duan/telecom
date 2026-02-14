from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    message_id: str
    request_id: str
    role: Role
    content: str
    ts: float
    answer_id: Optional[str] = None
    meta: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    request_id: str
    message: str
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    conversation_id: str
    request_id: str
    answer: str
    route: str
    used_rag: bool
    citations: list[dict[str, Any]] = Field(default_factory=list)


class EndRequest(BaseModel):
    conversation_id: str


class EndResponse(BaseModel):
    conversation_id: str
    flushed_message_count: int
