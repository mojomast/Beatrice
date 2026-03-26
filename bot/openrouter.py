from __future__ import annotations

from dataclasses import dataclass
import json
import re

import httpx

from .config import RuntimeConfig


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter API returns an error or malformed payload."""


class OpenRouterTimeout(OpenRouterError):
    """Raised when OpenRouter does not return before the read timeout."""


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, object]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ChatResponse:
    content: str
    tool_calls: tuple[ToolCall, ...]
    assistant_message: dict[str, object]


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        http_referer: str | None = None,
        title: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.http_referer = http_referer
        self.title = title
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        runtime: RuntimeConfig,
        messages: list[dict[str, object]],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, object] = "auto",
        request_timeout: httpx.Timeout | float | None = None,
    ) -> ChatResponse:
        if not self.api_key:
            raise OpenRouterError("OpenRouter API key is not configured")

        payload = {
            "model": runtime.model,
            "messages": messages,
            "temperature": runtime.temperature,
            "top_p": runtime.top_p,
            "max_tokens": runtime.max_tokens,
            "stream": runtime.stream if not tools else False,
        }

        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
            payload["tool_choice"] = tool_choice

        if runtime.stream and not tools:
            content = await self._stream_completion(payload, request_timeout=request_timeout)
            assistant_message = {"role": "assistant", "content": content}
            return ChatResponse(content=content, tool_calls=(), assistant_message=assistant_message)
        return await self._single_completion(payload, request_timeout=request_timeout)

    async def complete(
        self,
        runtime: RuntimeConfig,
        user_prompt: str,
        messages: list[dict[str, object]] | None = None,
        request_timeout: httpx.Timeout | float | None = None,
    ) -> str:
        payload_messages = messages or [
            {"role": "system", "content": runtime.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = await self.chat(runtime, payload_messages, request_timeout=request_timeout)
        return response.content

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.title:
            headers["X-Title"] = self.title
        return headers

    async def _single_completion(self, payload: dict, request_timeout: httpx.Timeout | float | None = None) -> ChatResponse:
        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=request_timeout,
            )
        except httpx.ReadTimeout as exc:
            raise OpenRouterTimeout(f"OpenRouter read timeout for model {payload.get('model')}") from exc
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise OpenRouterError(self._extract_error_message(response)) from exc

        data = response.json()
        try:
            message = data["choices"][0]["message"]
        except (IndexError, KeyError, TypeError) as exc:
            raise OpenRouterError("OpenRouter returned an unexpected response body") from exc

        if not isinstance(message, dict):
            raise OpenRouterError("OpenRouter returned an unexpected message payload")

        content = message.get("content", "")
        text = self._normalize_content(content).strip()
        tool_calls = self._extract_tool_calls(message)
        if not text and not tool_calls:
            raise OpenRouterError("OpenRouter returned an empty response")
        return ChatResponse(content=text, tool_calls=tool_calls, assistant_message=message)

    async def _stream_completion(self, payload: dict, request_timeout: httpx.Timeout | float | None = None) -> str:
        chunks: list[str] = []
        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=request_timeout,
            ) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise OpenRouterError(self._extract_error_message(response)) from exc

                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line or not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if data == "[DONE]":
                        break

                    event = json.loads(data)
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        piece = self._normalize_content(delta.get("content"))
                        if piece:
                            chunks.append(piece)
        except httpx.ReadTimeout as exc:
            raise OpenRouterTimeout(f"OpenRouter read timeout for model {payload.get('model')}") from exc

        text = "".join(chunks).strip()
        if not text:
            raise OpenRouterError("OpenRouter returned an empty streamed response")
        return text

    @staticmethod
    def _normalize_content(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    @staticmethod
    def _extract_tool_calls(message: dict[str, object]) -> tuple[ToolCall, ...]:
        raw_calls = message.get("tool_calls")
        if isinstance(raw_calls, list):
            return OpenRouterClient._parse_raw_tool_calls(raw_calls)

        content = OpenRouterClient._normalize_content(message.get("content"))
        if content:
            return OpenRouterClient._extract_markup_tool_calls(content)
        return ()

    @staticmethod
    def _parse_raw_tool_calls(raw_calls: list[object]) -> tuple[ToolCall, ...]:
        if not isinstance(raw_calls, list):
            return ()

        parsed: list[ToolCall] = []
        for index, raw_call in enumerate(raw_calls):
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            arguments = function.get("arguments", "{}")
            if not isinstance(name, str) or not name.strip():
                continue
            if isinstance(arguments, str):
                try:
                    payload = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError as exc:
                    raise OpenRouterError(f"OpenRouter returned invalid tool arguments for {name}") from exc
            elif isinstance(arguments, dict):
                payload = arguments
            else:
                payload = {}
            if not isinstance(payload, dict):
                raise OpenRouterError(f"OpenRouter returned invalid tool arguments for {name}")
            parsed.append(
                ToolCall(
                    id=str(raw_call.get("id") or f"tool_call_{index}"),
                    name=name,
                    arguments=payload,
                )
            )
        return tuple(parsed)

    @staticmethod
    def _extract_markup_tool_calls(content: str) -> tuple[ToolCall, ...]:
        invoke_matches = re.findall(r"<invoke\s+name=\"([^\"]+)\"[^>]*>(.*?)</invoke>", content, flags=re.DOTALL | re.IGNORECASE)
        if not invoke_matches:
            return ()
        parsed: list[ToolCall] = []
        for index, (name, body) in enumerate(invoke_matches):
            name = name.strip()
            if not name:
                continue
            arguments: dict[str, object] = {}
            for param_name, raw_value in re.findall(
                r"<parameter\s+name=\"([^\"]+)\"[^>]*>(.*?)</parameter>",
                body,
                flags=re.DOTALL | re.IGNORECASE,
            ):
                text = raw_value.strip()
                if not text:
                    arguments[param_name] = ""
                    continue
                try:
                    arguments[param_name] = json.loads(text)
                except json.JSONDecodeError:
                    arguments[param_name] = text
            parsed.append(ToolCall(id=f"markup_tool_call_{index}", name=name, arguments=arguments))
        return tuple(parsed)

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return f"OpenRouter request failed with HTTP {response.status_code}"

        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()

        return f"OpenRouter request failed with HTTP {response.status_code}"

    def set_api_key(self, api_key: str | None) -> None:
        self.api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
