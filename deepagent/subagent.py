"""Generic subagent runner for the deep agent orchestrator."""
from __future__ import annotations

import json
import warnings
from typing import Any

from langchain.agents import create_agent
from pydantic import BaseModel

from agents.shared import (
    _last_ai_content,
    build_chat_model,
    DEFAULT_AGENT_RECURSION_LIMIT,
)
from deepagent.schemas import SubTask, SubTaskResult

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")


class _SubAgentOutput(BaseModel):
    """Flexible structured output schema for subagents.

    We use a loose dict-based schema so every task type can return whatever
    key-value pairs are appropriate without needing a per-type Pydantic model.
    """

    reasoning: str = ""
    result: dict[str, Any] = {}


def run_subagent(
    task: SubTask,
    *,
    system_prompt: str,
    context: str,
    tools: list[Any],
) -> SubTaskResult:
    """Create and invoke a subagent for a single subtask.

    Parameters
    ----------
    task
        The subtask definition (id, type, description, etc.).
    system_prompt
        Fully composed system prompt for this subagent.
    context
        User message containing context from upstream task results.
    tools
        LangChain tools available to this subagent.

    Returns
    -------
    SubTaskResult
        Result with status and output dict, or an error message.
    """
    try:
        model = build_chat_model()

        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            response_format=_SubAgentOutput,
        )

        config = {"recursion_limit": DEFAULT_AGENT_RECURSION_LIMIT}
        payload = {"messages": [{"role": "user", "content": context}]}

        print(f"  [subagent:{task.task_id}] invoking ({task.task_type})...", flush=True)
        response = agent.invoke(payload, config=config)

        # Extract structured response.
        structured = response.get("structured_response")

        if structured is None:
            # Fallback: try parsing last AI message as JSON.
            last_content = _last_ai_content(response.get("messages", []))
            if last_content:
                try:
                    structured = _SubAgentOutput.model_validate_json(last_content)
                except Exception:
                    # Last resort: wrap raw text as output.
                    return SubTaskResult(
                        task_id=task.task_id,
                        status="ok",
                        output={"raw_text": last_content},
                    )

        if structured is None:
            return SubTaskResult(
                task_id=task.task_id,
                status="error",
                error="Subagent produced no structured response and no parseable AI text.",
            )

        # Coerce to dict.
        if isinstance(structured, BaseModel):
            output = structured.model_dump()
        elif isinstance(structured, dict):
            output = structured
        else:
            output = {"raw": str(structured)}

        print(f"  [subagent:{task.task_id}] done.", flush=True)
        return SubTaskResult(task_id=task.task_id, status="ok", output=output)

    except Exception as e:
        print(f"  [subagent:{task.task_id}] error: {e}", flush=True)
        return SubTaskResult(
            task_id=task.task_id,
            status="error",
            error=f"{type(e).__name__}: {e}",
        )
