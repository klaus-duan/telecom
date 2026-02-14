from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable

from openai import OpenAI


@dataclass(frozen=True)
class QwenClient:
    api_key: str
    base_url: str

    def _client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        client = self._client()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def chat_with_tools(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_executor: Callable[[str, dict[str, Any]], Any],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        max_steps: int = 3,
    ) -> str:
        """最小工具调用（function calling）循环。

        - 如果模型返回 tool_calls：执行工具并把结果以 role=tool 回填，然后继续。
        - 如果模型返回 content：结束并返回 content。
        """
        client = self._client()
        work_msgs: list[dict[str, Any]] = list(messages)

        for _ in range(max_steps):
            resp = client.chat.completions.create(
                model=model,
                messages=work_msgs,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
            )

            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [],
                }

                for tc in tool_calls:
                    fn = tc.function
                    assistant_msg["tool_calls"].append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": fn.name,
                                "arguments": fn.arguments or "{}",
                            },
                        }
                    )

                work_msgs.append(assistant_msg)

                for tc in tool_calls:
                    fn = tc.function
                    try:
                        args = json.loads(fn.arguments or "{}")
                        if not isinstance(args, dict):
                            args = {}
                    except Exception:
                        args = {}

                    try:
                        result = tool_executor(fn.name, args)
                    except Exception as e:
                        result = {"error": str(e)}

                    work_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                continue

            return (msg.content or "").strip()

        # 超过最大步数仍未产出最终内容
        return ""

    def embed(self, *, model: str, texts: list[str]) -> list[list[float]]:
        client = self._client()
        resp = client.embeddings.create(model=model, input=texts)
        return [list(map(float, d.embedding)) for d in resp.data]
