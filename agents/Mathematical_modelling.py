from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool

from agents.shared import (
    build_ollama_model,
    get_data_dir,
    get_test_outputs_dir,
    invoke_structured_agent,
)
from schemas.basemodels import ModellingRecommendation, UseCaseRecommendation


warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")


def _resolve_csv_path(csv_file_path: str) -> Path:
    csv_path = Path(csv_file_path)
    if csv_path.is_absolute():
        return csv_path

    data_path = get_data_dir() / csv_file_path
    if data_path.exists():
        return data_path

    return csv_path.resolve()


def _load_reference_model() -> dict[str, Any]:
    model_path = get_data_dir() / "ReferenceMathematicalModel.json"
    if not model_path.exists():
        return {}
    return json.loads(model_path.read_text(encoding="utf-8"))


def _persist_outputs(recommendation: ModellingRecommendation) -> None:
    outputs_dir = get_test_outputs_dir()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "llm_objective_function.md").write_text(
        recommendation.objective_function.strip(),
        encoding="utf-8",
    )
    (outputs_dir / "llm_constraints.md").write_text(
        "\n".join(item.strip() for item in recommendation.constraint_functions),
        encoding="utf-8",
    )
    (outputs_dir / "llm_output.md").write_text(
        recommendation.readable_documentation.strip(),
        encoding="utf-8",
    )


def run_mathematical_modelling_agent(
    csv_file_path: str | None = None,
    use_case: UseCaseRecommendation | dict[str, Any] | None = None,
    preview_rows: int = 5,
    return_debug: bool = False,
) -> dict[str, Any]:
    """Generate a modelling recommendation using the legacy modelling prompt contract."""
    resolved_csv_path = _resolve_csv_path(
        csv_file_path or os.getenv("PIPELINE_CSV_PATH", "optimization_pipeline_test_easy.csv")
    )
    if not resolved_csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {resolved_csv_path}")

    @tool
    def get_column_names() -> dict[str, Any]:
        """Returns available columns and a compact preview from the selected CSV."""
        df_preview = pd.read_csv(resolved_csv_path, nrows=preview_rows)
        return {
            "csv_file_path": str(resolved_csv_path),
            "columns": [str(column) for column in df_preview.columns.tolist()],
            "preview_rows": df_preview.to_dict(orient="records"),
        }

    @tool
    def get_reference_model() -> dict[str, Any]:
        """Returns the reference model style for notation and structure."""
        return _load_reference_model()

    @tool
    def get_use_case_recommendation() -> dict[str, Any]:
        """Returns the selected upstream use-case recommendation."""
        if use_case is None:
            return {
                "use_case_name": "Production Planning",
                "business_goal": "Optimize quantity to produce for each product.",
                "objective_direction": "max",
                "objective_variable": "total profit",
                "decision_variables": ["production_quantity_per_product"],
                "required_columns": [],
                "constraints_to_consider": [],
                "assumptions": ["Use-case recommendation missing; fallback context used."],
                "rationale": "Fallback use case injected by modelling stage.",
            }

        if isinstance(use_case, UseCaseRecommendation):
            return use_case.model_dump()
        return dict(use_case)

    llm = build_ollama_model()
    agent = create_agent(
        model=llm,
        tools=[get_use_case_recommendation, get_column_names, get_reference_model],
        system_prompt=(
            "You are a mathematical expert in MILP optimization. "
            "Use tool calls to inspect use-case, CSV columns, and reference style before drafting the model.\n\n"
            "Required workflow:\n"
            "1. Call get_use_case_recommendation\n"
            "2. Call get_column_names\n"
            "3. Call get_reference_model\n"
            "4. Return ModellingRecommendation only via tool call\n\n"
            "Rules:\n"
            "- Provide pseudo-LaTeX style objective and constraints\n"
            "- Include all required fields exactly\n"
            "- Do not emit plain text-only final answers\n"
        ),
        response_format=ModellingRecommendation,
    )

    tool_trace: list[str] = []
    debug_trace: dict[str, Any] = {}
    args = invoke_structured_agent(
        agent=agent,
        user_input=(
            "Create a MILP formulation for optimizing production quantity per product using only tool outputs."
        ),
        response_tool_name="ModellingRecommendation",
        retry_message=(
            "You failed to output ModellingRecommendation correctly. Return only a valid "
            "ModellingRecommendation tool call with all required fields."
        ),
        response_model=ModellingRecommendation,
        tool_trace=tool_trace,
        debug_trace=debug_trace,
    )

    recommendation = ModellingRecommendation.model_validate(args)
    _persist_outputs(recommendation)
    if return_debug:
        return {
            "result": recommendation.model_dump(),
            "tool_trace": tool_trace,
            "debug": debug_trace,
        }
    return recommendation.model_dump()


if __name__ == "__main__":
    payload = run_mathematical_modelling_agent()
    print(json.dumps(payload, indent=2))
