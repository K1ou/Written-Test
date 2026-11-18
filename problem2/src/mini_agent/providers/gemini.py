"""Gemini provider: Google's native protocol via the google-genai SDK.

We use the stateless `models.generate_content_stream` call and keep the conversation history
ourselves, so the agent loop behaves exactly as with every other provider.

Conversion rules between the internal format and the wire format:
  * assistant tool_use blocks  <->  `function_call` parts of a "model" Content
  * user tool_result blocks    <->  `function_response` parts of a "user" Content; Gemini matches
    responses to calls by function NAME, so names are looked up from the earlier tool_use blocks
  * Gemini function calls usually carry no id: we generate one and remember that in `meta`, and we
    never send a generated id back
  * Gemini 3 models require the `thought_signature` of a function call part to be echoed back
    unchanged; it is kept (base64) in ToolUseBlock.meta so it also survives session persistence
Function calls arrive complete (not in argument fragments), so each one becomes Start/Args/End at once.
The SDK's own retries are disabled (attempts=1): retrying is done by providers/retry.py.
"""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Iterable, Iterator
from typing import Any

import httpx
from google import genai
from google.genai import errors, types
from pydantic import SecretStr

try:  # the openai SDK ships its own httpx fork; google-genai may use either
    import httpx2
except ImportError:  # pragma: no cover
    httpx2 = None

from mini_agent.agent.cancel import CancelToken
from mini_agent.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock, Usage
from mini_agent.providers.base import (
    Done,
    ProviderRequest,
    StreamEvent,
    TextDelta,
    ToolCallArgsDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolSpec,
    UsageEvent,
)
from mini_agent.providers.errors import (
    FatalError,
    ProviderError,
    RetryableError,
    from_http_status,
    parse_retry_after,
)

_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (httpx.TransportError, TimeoutError, ConnectionError)
if httpx2 is not None:
    _TRANSPORT_ERRORS += (httpx2.TransportError,)


# ------------------------------------------------------------------ request conversion


def _tool_names(messages: list[Message]) -> dict[str, str]:
    return {use.id: use.name for m in messages if m.role == "assistant" for use in m.tool_uses}


def _id_for_wire(block_id: str, meta: dict[str, Any]) -> str | None:
    """Real ids are echoed back; ids we generated ourselves must not be sent."""
    return None if meta.get("id_generated") else block_id


def _assistant_to_content(message: Message) -> types.Content:
    parts: list[types.Part] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            if block.text:
                parts.append(types.Part(text=block.text))
        elif isinstance(block, ToolUseBlock):
            signature = block.meta.get("thought_signature")
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        id=_id_for_wire(block.id, block.meta), name=block.name, args=block.input
                    ),
                    thought_signature=base64.b64decode(signature) if signature else None,
                )
            )
    return types.Content(role="model", parts=parts)


def _user_to_content(message: Message, names: dict[str, str], generated: set[str]) -> types.Content:
    parts: list[types.Part] = []
    for block in message.content:
        if isinstance(block, ToolResultBlock):
            payload = {"error": block.content} if block.is_error else {"output": block.content}
            parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        id=None if block.tool_use_id in generated else block.tool_use_id,
                        name=names.get(block.tool_use_id, "unknown_tool"),
                        response=payload,
                    )
                )
            )
    text = message.text
    if text:
        parts.append(types.Part(text=text))
    return types.Content(role="user", parts=parts)


def to_gemini_contents(messages: list[Message]) -> list[types.Content]:
    names = _tool_names(messages)
    generated = {
        use.id for m in messages if m.role == "assistant" for use in m.tool_uses if use.meta.get("id_generated")
    }
    return [
        _assistant_to_content(m) if m.role == "assistant" else _user_to_content(m, names, generated)
        for m in messages
    ]


_SCHEMA_PASSTHROUGH = ("description", "enum", "required", "minimum", "maximum", "format")


def to_gemini_schema(node: dict[str, Any]) -> types.Schema:
    """Convert the (already cleaned) JSON Schema of a tool into Gemini's classic Schema object.

    The classic `parameters` field is accepted by every API version, unlike the newer
    `parameters_json_schema` field. Our tool schemas only use type, properties, items, required,
    description, enum and minimum/maximum, all of which are handled here.
    """
    fields: dict[str, Any] = {}
    if node.get("type"):
        fields["type"] = str(node["type"]).upper()
    for key in _SCHEMA_PASSTHROUGH:
        if key in node:
            fields[key] = node[key]
    if isinstance(node.get("properties"), dict):
        fields["properties"] = {name: to_gemini_schema(child) for name, child in node["properties"].items()}
    if isinstance(node.get("items"), dict):
        fields["items"] = to_gemini_schema(node["items"])
    return types.Schema(**fields)


def to_gemini_tools(tools: list[ToolSpec]) -> list[types.Tool]:
    if not tools:
        return []
    declarations = [
        types.FunctionDeclaration(name=t.name, description=t.description, parameters=to_gemini_schema(t.parameters))
        for t in tools
    ]
    return [types.Tool(function_declarations=declarations)]


# ------------------------------------------------------------------ stream conversion


def _usage(metadata: Any) -> Usage:
    return Usage(
        input_tokens=metadata.prompt_token_count or 0,
        # thinking tokens are billed and counted as output
        output_tokens=(metadata.candidates_token_count or 0) + (metadata.thoughts_token_count or 0),
    )


def _finish_name(finish: Any) -> str:
    return getattr(finish, "name", None) or str(finish)


def convert_stream(chunks: Iterable[Any]) -> Iterator[StreamEvent]:
    """Turn SDK `GenerateContentResponse` chunks into vendor-neutral StreamEvents."""
    finish: str | None = None
    finish_message: str | None = None
    emitted_calls = 0
    emitted_text = False
    used_ids: set[str] = set()

    for chunk in chunks:
        if chunk.usage_metadata is not None:
            yield UsageEvent(_usage(chunk.usage_metadata))
        if not chunk.candidates:
            feedback = chunk.prompt_feedback
            if feedback is not None and feedback.block_reason:
                detail = f": {feedback.block_reason_message}" if feedback.block_reason_message else ""
                raise FatalError(f"Gemini blocked the prompt ({_finish_name(feedback.block_reason)}){detail}")
            continue
        candidate = chunk.candidates[0]
        content = candidate.content
        for part in (content.parts if content is not None and content.parts else []):
            if part.thought:  # thought summaries are never shown or sent back
                continue
            if part.function_call is not None:
                call = part.function_call
                call_id = call.id
                meta: dict[str, Any] = {}
                if not call_id or call_id in used_ids:
                    call_id = f"call_{uuid.uuid4().hex[:12]}"
                    meta["id_generated"] = True
                used_ids.add(call_id)
                if part.thought_signature:
                    meta["thought_signature"] = base64.b64encode(part.thought_signature).decode("ascii")
                yield ToolCallStart(call_id, call.name or "", meta)
                yield ToolCallArgsDelta(call_id, json.dumps(call.args or {}, ensure_ascii=False))
                yield ToolCallEnd(call_id)
                emitted_calls += 1
            elif part.text:
                emitted_text = True
                yield TextDelta(part.text)
        if candidate.finish_reason is not None:
            name = _finish_name(candidate.finish_reason)
            if name != "FINISH_REASON_UNSPECIFIED":
                finish = name
                finish_message = candidate.finish_message

    if finish is None:
        raise RetryableError("The response stream ended before the model finished (connection dropped?)")
    if finish == "STOP":
        yield Done("tool_use" if emitted_calls else "end_turn")
    elif finish == "MAX_TOKENS":
        if not emitted_text and not emitted_calls:
            raise FatalError(
                "Gemini reached the maximum output length before producing any visible output. Thinking models "
                "spend output tokens on reasoning, so increase max_tokens."
            )
        yield Done("max_tokens")
    elif finish == "MALFORMED_FUNCTION_CALL":
        raise RetryableError(f"Gemini produced a malformed function call. {finish_message or ''}".strip())
    elif finish == "MISSING_THOUGHT_SIGNATURE":
        raise FatalError(
            "Gemini rejected the request because a function call is missing its thought signature. "
            f"{finish_message or ''}".strip()
        )
    else:
        detail = f": {finish_message}" if finish_message else ""
        raise FatalError(f"Gemini stopped the response (finish reason {finish}){detail}")


# ------------------------------------------------------------------------- errors


def _error_infos(exc: errors.APIError) -> list[dict[str, Any]]:
    details = exc.details
    if isinstance(details, dict):
        inner = details.get("error")
        listed = inner.get("details") if isinstance(inner, dict) else details.get("details")
        if isinstance(listed, list):
            return [d for d in listed if isinstance(d, dict)]
    return []


def _retry_delay_seconds(infos: list[dict[str, Any]]) -> float | None:
    for info in infos:
        if str(info.get("@type", "")).endswith("RetryInfo"):
            delay = str(info.get("retryDelay", "")).strip()
            if delay.endswith("s"):
                try:
                    return max(0.0, float(delay[:-1]))
                except ValueError:
                    return None
    return None


def _daily_quota_exhausted(infos: list[dict[str, Any]]) -> bool:
    for info in infos:
        if str(info.get("@type", "")).endswith("QuotaFailure"):
            for violation in info.get("violations", []) or []:
                if "perday" in str(violation.get("quotaId", "")).lower():
                    return True
    return False


def translate_error(exc: BaseException) -> BaseException:
    """Map SDK and transport exceptions to ProviderError; unknown exceptions are returned unchanged."""
    if isinstance(exc, ProviderError):
        return exc
    if isinstance(exc, errors.APIError):
        status = exc.code if isinstance(exc.code, int) else 0
        message = str(exc.message or exc)
        infos = _error_infos(exc)
        if status == 429:
            # RESOURCE_EXHAUSTED covers both per-minute rate limits and an exhausted daily quota.
            if _daily_quota_exhausted(infos):
                return FatalError(f"Gemini daily quota exhausted (429): {message}", 429)
            return RetryableError(f"Rate limited (429): {message}", 429, _retry_delay_seconds(infos))
        if status in (400, 401, 403) and "api key" in message.lower():
            return FatalError(f"Authentication failed ({status}): check that the Gemini API key is set and valid. {message}", status)
        headers = getattr(exc.response, "headers", None)
        retry_after = parse_retry_after(headers.get("retry-after")) if headers is not None else None
        return from_http_status(status, message, code=exc.status, retry_after=retry_after)
    if isinstance(exc, _TRANSPORT_ERRORS):
        return RetryableError(f"Network error: {type(exc).__name__}: {exc}")
    return exc


# ---------------------------------------------------------------------- provider


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        api_key: str | SecretStr,
        model: str,
        *,
        client: genai.Client | None = None,
        api_version: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.model = model
        if client is None:
            key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
            client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    api_version=api_version,
                    base_url=base_url,
                    timeout=int(timeout_seconds * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),  # one attempt: no SDK-level retries
                ),
            )
        self._client = client

    def build_config(self, request: ProviderRequest) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=request.system or None,
            tools=to_gemini_tools(request.tools) or None,
            max_output_tokens=request.max_tokens,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def stream(self, request: ProviderRequest, cancel: CancelToken) -> Iterator[StreamEvent]:
        contents = to_gemini_contents(request.messages)
        config = self.build_config(request)
        chunks = self._chunks(contents, config)
        try:
            yield from convert_stream(chunks)
        finally:
            chunks.close()

    def _chunks(self, contents: list[types.Content], config: types.GenerateContentConfig) -> Iterator[Any]:
        response = None
        try:
            response = self._client.models.generate_content_stream(model=self.model, contents=contents, config=config)
            yield from response
        except Exception as exc:
            translated = translate_error(exc)
            if translated is exc:
                raise
            raise translated from exc
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()
