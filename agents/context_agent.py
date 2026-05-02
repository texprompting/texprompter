from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from agents.prompts import load_system_prompt_result
from agents.shared import (
    _last_ai_content,
    build_chat_model,
    extract_tool_trace,
    get_data_dir,
    invoke_agent_with_prompt_trace,
    prompt_debug_payload,
)
from schemas.basemodels import _coerce_json_collection


warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")


class ContextRecommendation(BaseModel):
    use_case: str = Field(description="Best optimization use case")
    objective: str = Field(description="Objective of optimization")
    decision_variables: list[str] = Field(description="Variables to optimize")
    relevant_columns: list[str] = Field(description="Relevant CSV columns")
    statistics: str = Field(description="RAW statistical summary from tool")
    reasoning: str = Field(description="Why this use case was chosen")

    @field_validator("decision_variables", "relevant_columns", mode="before")
    @classmethod
    def _coerce_list_fields(cls, value: Any) -> Any:
        return _coerce_json_collection(value)


def _resolve_csv_path(csv_file_path: str) -> Path:
    csv_path = Path(csv_file_path)
    if csv_path.is_absolute():
        return csv_path

    data_path = get_data_dir() / csv_file_path
    if data_path.exists():
        return data_path

    return csv_path.resolve()


def _coerce_recommendation(value: Any) -> ContextRecommendation:
    if isinstance(value, ContextRecommendation):
        return value
    if isinstance(value, BaseModel):
        return ContextRecommendation.model_validate(value.model_dump())
    if isinstance(value, dict):
        return ContextRecommendation.model_validate(value)
    raise TypeError(f"Unexpected structured_response type: {type(value)!r}")


def run_context_agent(
    csv_file_path: str | None = None,
    preview_rows: int = 10,
    return_debug: bool = False,
) -> dict[str, Any]:
    """Analyze CSV data and return the legacy context recommendation contract."""
    resolved_csv_path = _resolve_csv_path(
        csv_file_path or os.getenv("PIPELINE_CSV_PATH", "Deliverymodule.csv")
    )
    if not resolved_csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {resolved_csv_path}")

    df = pd.read_csv(resolved_csv_path)

    @tool
    def get_column_names() -> list[str]:
        """Returns column names of the dataset."""
        return [str(column) for column in df.columns.tolist()]

    @tool
    def get_csv_preview() -> str:
        """Returns first rows of dataset."""
        return df.head(preview_rows).to_string()

    @tool
    def get_basic_stats() -> dict[str, str]:
        """Returns statistical summary."""
        return {"raw_stats": df.describe(include="all").to_string()}

    prompt = load_system_prompt_result("use_case")
    agent = create_agent(
        model=build_chat_model(),
        tools=[get_column_names, get_csv_preview, get_basic_stats],
        system_prompt=prompt.template,
        response_format=ContextRecommendation,
    )
    user_message = "Analyze the dataset and identify the best optimization use case."

    response = invoke_agent_with_prompt_trace(
        agent,
        stage="use_case",
        prompt=prompt,
        user_message=user_message,
    )

    structured = response.get("structured_response")
    if structured is None:
        # Fallback: attempt to parse the last AI message text as JSON.
        last_content = _last_ai_content(response.get("messages", []))
        if last_content:
            try:
                structured = ContextRecommendation.model_validate_json(last_content)
            except Exception:
                pass
    if structured is None:
        raise ValueError("context_agent did not produce a structured_response.")


    recommendation = _coerce_recommendation(structured)
    result = recommendation.model_dump()

    if return_debug:
        tool_trace = extract_tool_trace(response.get("messages", []))
        return {"result": result, "tool_trace": tool_trace, "debug": {"prompt": prompt_debug_payload(prompt)}}
    return result


def format_for_modelling_agent(ctx: dict[str, Any]) -> str:
    return f"""
Use Case:
{ctx['use_case']}

Objective:
{ctx['objective']}

Decision Variables:
{ctx['decision_variables']}

Relevant Columns:
{ctx['relevant_columns']}

Statistics:
{ctx['statistics']}

Reasoning:
{ctx['reasoning']}
"""


if __name__ == "__main__":
    ctx = run_context_agent()

    print("\n=== CONTEXT AGENT OUTPUT ===\n")
    print(json.dumps(ctx, indent=2))

    formatted = format_for_modelling_agent(ctx)
    print("\n=== FORMATTED FOR MODELLING AGENT ===\n")
    print(formatted)

    output_path = get_data_dir() / "context_output.txt"
    output_path.write_text(formatted, encoding="utf-8")
