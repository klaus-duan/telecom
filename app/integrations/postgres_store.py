from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import psycopg2
from psycopg2.extras import execute_values


@dataclass(frozen=True)
class ChatHistoryRow:
    conversation_id: str
    request_id: str
    message: str
    answer: str
    time: datetime


class PostgresStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def persist_chat_history_from_messages(
        self, *, conversation_id: str, messages: list[dict[str, Any]]
    ) -> int:
        """把 Redis messages(list) 写入 public.chat_history。

        约定：messages 里包含 user/assistant 两条，且共享相同 request_id。
        返回：成功插入的行数（遇到冲突会 DO NOTHING）。
        """
        conv_id = str(conversation_id)

        # group by request_id
        grouped: dict[str, dict[str, Any]] = {}
        for m in messages:
            rid = str(m.get("request_id") or "").strip()
            if not rid:
                continue
            role = (m.get("role") or "").strip()
            content = str(m.get("content") or "")
            ts = m.get("ts")
            entry = grouped.setdefault(rid, {})
            if role == "user" and "message" not in entry:
                entry["message"] = content
                entry["user_ts"] = ts
            elif role == "assistant" and "answer" not in entry:
                entry["answer"] = content
                entry["assistant_ts"] = ts

        rows: list[ChatHistoryRow] = []
        for rid, entry in grouped.items():
            message = str(entry.get("message") or "")
            answer = str(entry.get("answer") or "")
            chosen_ts = entry.get("assistant_ts")
            if chosen_ts is None:
                chosen_ts = entry.get("user_ts")
            if chosen_ts is None:
                dt = datetime.now(timezone.utc)
            else:
                try:
                    dt = datetime.fromtimestamp(float(chosen_ts), tz=timezone.utc)
                except Exception:
                    dt = datetime.now(timezone.utc)

            rows.append(
                ChatHistoryRow(
                    conversation_id=conv_id,
                    request_id=rid,
                    message=message,
                    answer=answer,
                    time=dt,
                )
            )

        if not rows:
            return 0

        values: Iterable[tuple[Any, ...]] = (
            (r.conversation_id, r.request_id, r.message, r.answer, r.time) for r in rows
        )

        sql = (
            "INSERT INTO public.chat_history (conversation_id, request_id, message, answer, time) "
            "VALUES %s "
            "ON CONFLICT (conversation_id, request_id) DO NOTHING"
        )

        with psycopg2.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    sql,
                    values,
                    page_size=200,
                    # conversation_id 列通常是 UUID；这里用显式 cast，兼容 text/uuid 两种列类型
                    template="(%s::uuid, %s, %s, %s, %s)",
                )
                inserted = cur.rowcount
        return int(inserted if inserted is not None else 0)
