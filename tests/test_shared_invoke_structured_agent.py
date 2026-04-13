'''
Verifies that invoke_structured_agent still returns
the correct structured payload when the valid tool
call appears in an earlier AI message and the final
message has no tool call.
'''

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agents.shared import invoke_structured_agent


class _ResponseModel(BaseModel):
    value: int


class _FakeAIMessage:
    type = "ai"

    def __init__(
        self,
        tool_calls: list[dict[str, Any]] | None = None,
        content: str | None = None,
        additional_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.tool_calls = tool_calls or []
        self.content = content
        self.additional_kwargs = additional_kwargs or {}


class _FakeAgent:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses
        self._index = 0

    def invoke(self, _payload: dict[str, Any]) -> dict[str, Any]:
        response = self._responses[self._index]
        if self._index < len(self._responses) - 1:
            self._index += 1
        return response


def test_uses_non_final_ai_message_tool_call() -> None:
    response = {
        "messages": [
            _FakeAIMessage(tool_calls=[{"name": "ResponseModel", "args": {"value": 7}}]),
            _FakeAIMessage(content="final summary without tool call"),
        ]
    }
    agent = _FakeAgent([response])

    result = invoke_structured_agent(
        agent=agent,
        user_input="return structured",
        response_tool_name="ResponseModel",
        retry_message="retry",
        response_model=_ResponseModel,
    )

    assert result == {"value": 7}


def test_retries_when_named_tool_payload_invalid_then_succeeds() -> None:
    invalid_response = {
        "messages": [
            _FakeAIMessage(tool_calls=[{"name": "ResponseModel", "args": {"value": "bad"}}])
        ]
    }
    valid_response = {
        "messages": [
            _FakeAIMessage(tool_calls=[{"name": "ResponseModel", "args": {"value": 11}}])
        ]
    }
    agent = _FakeAgent([invalid_response, valid_response])

    result = invoke_structured_agent(
        agent=agent,
        user_input="return structured",
        response_tool_name="ResponseModel",
        retry_message="retry",
        max_retries=2,
        response_model=_ResponseModel,
    )

    assert result == {"value": 11}


def test_parses_json_from_fenced_content_without_tool_call() -> None:
    response = {
        "messages": [
            _FakeAIMessage(
                content=(
                    "I could not call the tool, but here is the payload.\n"
                    "```json\n"
                    "{\"value\": 13}\n"
                    "```"
                )
            )
        ]
    }
    agent = _FakeAgent([response])

    result = invoke_structured_agent(
        agent=agent,
        user_input="return structured",
        response_tool_name="ResponseModel",
        retry_message="retry",
        response_model=_ResponseModel,
    )

    assert result == {"value": 13}
