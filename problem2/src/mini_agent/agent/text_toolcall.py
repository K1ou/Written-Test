"""Tool calls that a model wrote into its plain text instead of using the tool-calling channel.

Small and local models often answer with something like

    {"name": "read_file", "arguments": {"path": "a.py"}}

or wrap it in a code fence or a <tool_call> tag. Our choice is to RECOGNIZE it and ask the model to
retry through the real tool interface; the text is never executed. (Executing it would be convenient
but unsafe: it bypasses the argument checks and the cut-off detection of the native channel.)

Only objects that name a REGISTERED tool and carry an arguments field count, so ordinary JSON in an
answer (a package.json, an API example) is left alone. Whatever the text looks like, the JSON is found
by scanning for objects, which covers bare JSON, fenced JSON and <tool_call> tags in one pass.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from mini_agent.messages import AGENT_NOTICE_PREFIX

NAME_KEYS = ("name", "tool", "tool_name", "function_name")
ARGUMENT_KEYS = ("arguments", "parameters", "input", "args", "params")
MAX_SCANNED_CHARS = 30_000
MAX_CANDIDATES = 20


class TextToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _from_common_shapes(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            raise ValueError("not an object")
        if isinstance(data.get("function"), dict):  # {"type": "function", "function": {"name": ..., "arguments": ...}}
            data = data["function"]
        name = next((data[k] for k in NAME_KEYS if isinstance(data.get(k), str)), None)
        key = next((k for k in ARGUMENT_KEYS if k in data), None)
        if name is None or key is None:
            raise ValueError("no tool name or no arguments field")
        arguments = data[key]
        if isinstance(arguments, str):  # OpenAI style: the arguments are a JSON string
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                raise ValueError("arguments are not valid JSON") from None
        return {"name": name, "arguments": arguments}


def _json_objects(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    found: list[Any] = []
    text = text[:MAX_SCANNED_CHARS]
    position = text.find("{")
    while position != -1 and len(found) < MAX_CANDIDATES:
        try:
            value, end = decoder.raw_decode(text, position)
        except ValueError:
            position = text.find("{", position + 1)
            continue
        found.append(value)
        position = text.find("{", end)
    return found


def find_text_tool_calls(text: str, tool_names: list[str]) -> list[TextToolCall]:
    """Tool calls written as text, in order of appearance; empty when the text is an ordinary answer."""
    calls: list[TextToolCall] = []
    for candidate in _json_objects(text):
        try:
            call = TextToolCall.model_validate(candidate)
        except ValidationError:
            continue
        if call.name in tool_names:
            calls.append(call)
    return calls


def correction_message(calls: list[TextToolCall]) -> str:
    names = ", ".join(dict.fromkeys(c.name for c in calls))
    return (
        f"{AGENT_NOTICE_PREFIX} Your last reply contained a call to {names} written as plain text, so it was NOT "
        "executed. Tool calls only work through the tool-calling interface, not as text. Call the tool through "
        "that interface now. If you were only showing an example to the user, answer again in words without "
        "writing out a tool call."
    )
