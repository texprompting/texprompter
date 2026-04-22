from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agents.shared import build_ollama_model, get_data_dir, invoke_structured_agent


warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")


class ContextRecommendation(BaseModel):
    use_case: str = Field(description="Best optimization use case")
    objective: str = Field(description="Objective of optimization")
    decision_variables: list[str] = Field(description="Variables to optimize")
    relevant_columns: list[str] = Field(description="Relevant CSV columns")
    statistics: str = Field(description="RAW statistical summary from tool")
    reasoning: str = Field(description="Why this use case was chosen")


def _resolve_csv_path(csv_file_path: str) -> Path:
    csv_path = Path(csv_file_path)
    if csv_path.is_absolute():
        return csv_path

    data_path = get_data_dir() / csv_file_path
    if data_path.exists():
        return data_path

    return csv_path.resolve()


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

    llm = build_ollama_model()
    agent = create_agent(
        model=llm,
        tools=[get_column_names, get_csv_preview, get_basic_stats],
        system_prompt=(
            "You are an expert in operations research and production optimization.\n\n"
            "Workflow (strict):\n"
            "1. Call get_csv_preview\n"
            "2. Call get_column_names\n"
            "3. Call get_basic_stats\n"
            "4. Return ContextRecommendation only via tool call\n\n"
            "Rules:\n"
            "- Do not output plain text or manual JSON\n"
            "- statistics must copy raw_stats exactly\n"
            "- Do not create constraints in this stage\n"
        ),
        response_format=ContextRecommendation,
    )

    tool_trace: list[str] = []
    debug_trace: dict[str, Any] = {}
    args = invoke_structured_agent(
        agent=agent,
        user_input="Analyze the dataset and identify the best optimization use case.",
        response_tool_name="ContextRecommendation",
        retry_message=(
            "You failed to output with ContextRecommendation. Return only a valid "
            "ContextRecommendation tool call with all required fields."
        ),
        response_model=ContextRecommendation,
        tool_trace=tool_trace,
        debug_trace=debug_trace,
    )
    result = ContextRecommendation.model_validate(args).model_dump()
    if return_debug:
        return {"result": result, "tool_trace": tool_trace, "debug": debug_trace}
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