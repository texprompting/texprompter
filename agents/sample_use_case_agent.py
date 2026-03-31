from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool

from schemas.basemodels import UseCaseRecommendation

from .shared import build_ollama_model, get_data_dir, invoke_structured_agent


def _resolve_csv_path(csv_file_path: str) -> Path:
    csv_path = Path(csv_file_path)
    if csv_path.is_absolute():
        return csv_path

    data_path = get_data_dir() / csv_file_path
    if data_path.exists():
        return data_path

    return csv_path.resolve()


def run_use_case_agent(csv_file_path: str, preview_rows: int = 5) -> UseCaseRecommendation:
    """Analyze the CSV and choose the best optimization use case."""
    resolved_csv_path = _resolve_csv_path(csv_file_path)
    if not resolved_csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {resolved_csv_path}")

    df = pd.read_csv(resolved_csv_path)

    @tool
    def get_csv_overview() -> dict[str, Any]:
        """Return key CSV metadata for optimization use-case discovery."""
        return {
            "csv_file_path": str(resolved_csv_path),
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "columns": [str(column) for column in df.columns.tolist()],
            "dtypes": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
            "preview_rows": df.head(preview_rows).to_dict(orient="records"),
        }

    @tool
    def get_or_use_case_catalog() -> list[dict[str, str]]:
        """Return common OR use-case templates for production optimization."""
        return [
            {
                "use_case_name": "Production Planning",
                "goal": "Maximize total profit subject to labor and machine capacities.",
                "objective_direction": "max",
            },
            {
                "use_case_name": "Cost Minimization",
                "goal": "Minimize production and resource costs while meeting demand.",
                "objective_direction": "min",
            },
            {
                "use_case_name": "Throughput Maximization",
                "goal": "Maximize output under bottleneck capacity constraints.",
                "objective_direction": "max",
            },
        ]

    llm = build_ollama_model()
    agent = create_agent(
        model=llm,
        tools=[get_csv_overview, get_or_use_case_catalog],
        system_prompt=(
            "You are a production optimization analyst. Select the most profitable OR use case that can be modeled with the available CSV data "
            "while requiring minimal additional assumptions. Always call get_csv_overview first and then get_or_use_case_catalog. "
            "Final answer must be a UseCaseRecommendation tool call only."
        ),
        response_format=UseCaseRecommendation,
    )

    args = invoke_structured_agent(
        agent=agent,
        user_input=(
            "Analyze the CSV and return the best optimization use case with decision variables, required columns, constraints, and assumptions."
        ),
        response_tool_name="UseCaseRecommendation",
        retry_message=(
            "Your previous response did not follow the required UseCaseRecommendation tool call format. "
            "Return only a valid UseCaseRecommendation with all required fields."
        ),
        response_model=UseCaseRecommendation,
    )
    return UseCaseRecommendation.model_validate(args)


if __name__ == "__main__":
    result = run_use_case_agent("optimization_pipeline_test_easy.csv", preview_rows=5)
    print(result.model_dump_json(indent=2))
