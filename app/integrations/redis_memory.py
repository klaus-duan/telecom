from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import redis


@dataclass(frozen=True)
class RedisKeys:
    prefix: str
    conversation_id: str

    @property
    def messages(self) -> str:
        return f"{self.prefix}:chat:{self.conversation_id}:messages"

    @property
    def req_ids(self) -> str:
        return f"{self.prefix}:chat:{self.conversation_id}:req_ids"

    def response(self, request_id: str) -> str:
        return f"{self.prefix}:chat:{self.conversation_id}:resp:{request_id}"

    def inflight(self, request_id: str) -> str:
        return f"{self.prefix}:chat:{self.conversation_id}:inflight:{request_id}"


SET_TTL_LUA = """
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[1]))
return 1
"""


FLUSH_AND_DELETE_LUA = """
-- KEYS[1] = messages_list_key
-- KEYS[2] = req_ids_set_key
local vals = redis.call('LRANGE', KEYS[1], 0, -1)
redis.call('DEL', KEYS[1])
redis.call('DEL', KEYS[2])
return vals
"""


class RedisMemory:
    def __init__(self, client: redis.Redis, *, prefix: str, ttl_seconds: int) -> None:
        self._r = client
        self._prefix = prefix
        self._ttl_seconds = ttl_seconds

    @classmethod
    def from_url(cls, url: str, *, prefix: str, ttl_seconds: int) -> "RedisMemory":
        client = redis.Redis.from_url(url, decode_responses=True)
        return cls(client, prefix=prefix, ttl_seconds=ttl_seconds)

    def _keys(self, conversation_id: str) -> RedisKeys:
        return RedisKeys(prefix=self._prefix, conversation_id=conversation_id)

    def get_cached_response(self, conversation_id: str, request_id: str) -> Optional[dict[str, Any]]:
        k = self._keys(conversation_id).response(request_id)
        raw = self._r.get(k)
        if not raw:
            return None
        return json.loads(raw)

    def mark_inflight(self, conversation_id: str, request_id: str, *, ttl_seconds: int = 300) -> bool:
        """用 SET NX 做同一 request_id 的并发保护。"""
        k = self._keys(conversation_id).inflight(request_id)
        return bool(self._r.set(k, "1", nx=True, ex=ttl_seconds))

    def clear_inflight(self, conversation_id: str, request_id: str) -> None:
        k = self._keys(conversation_id).inflight(request_id)
        self._r.delete(k)

    def ensure_request_id_unique(self, conversation_id: str, request_id: str) -> bool:
        """返回 True 表示首次出现；False 表示重复。"""
        k = self._keys(conversation_id).req_ids
        added = self._r.sadd(k, request_id)
        if added:
            self._r.eval(SET_TTL_LUA, 1, k, str(self._ttl_seconds))
        return bool(added)

    def append_messages(self, conversation_id: str, messages: list[dict[str, Any]]) -> None:
        k = self._keys(conversation_id).messages
        payloads = [json.dumps(m, ensure_ascii=False) for m in messages]
        if payloads:
            self._r.rpush(k, *payloads)
            self._r.eval(SET_TTL_LUA, 1, k, str(self._ttl_seconds))

    def get_recent_messages(self, conversation_id: str, limit: int = 20) -> list[dict[str, Any]]:
        k = self._keys(conversation_id).messages
        raw = self._r.lrange(k, -limit, -1)  # key 不存在 => []
        return [json.loads(x) for x in raw]

    def get_all_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        k = self._keys(conversation_id).messages
        raw = self._r.lrange(k, 0, -1)  # key 不存在 => []
        return [json.loads(x) for x in raw]

    def cache_response(self, conversation_id: str, request_id: str, response: dict[str, Any]) -> None:
        k = self._keys(conversation_id).response(request_id)
        self._r.set(k, json.dumps(response, ensure_ascii=False), ex=self._ttl_seconds)

    def flush_and_delete(self, conversation_id: str) -> list[dict[str, Any]]:
        keys = self._keys(conversation_id)
        raw = self._r.eval(FLUSH_AND_DELETE_LUA, 2, keys.messages, keys.req_ids)
        return [json.loads(x) for x in raw]

    def delete_conversation(self, conversation_id: str) -> None:
        """删除该会话相关数据（messages/req_ids/resp/inflight）。"""
        keys = self._keys(conversation_id)
        request_ids = list(self._r.smembers(keys.req_ids) or [])
        delete_keys: list[str] = [keys.messages, keys.req_ids]
        for request_id in request_ids:
            delete_keys.append(keys.response(request_id))
            delete_keys.append(keys.inflight(request_id))

        if delete_keys:
            try:
                self._r.unlink(*delete_keys)
            except Exception:
                self._r.delete(*delete_keys)
