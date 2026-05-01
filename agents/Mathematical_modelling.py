from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel

from agents.prompts import load_system_prompt_result
from agents.shared import (
    _last_ai_content,
    build_chat_model,
    extract_tool_trace,
    get_data_dir,
    get_test_outputs_dir,
    invoke_agent_with_prompt_trace,
    prompt_debug_payload,
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
    """Write modelling artifacts to TestOutputs/; non-fatal if the directory is missing."""
    try:
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
    except OSError as io_err:
        # Non-fatal: the agent result is still valid; we just could not persist
        # the side-output files (e.g. in CI or a read-only environment).
        import warnings
        warnings.warn(f"_persist_outputs failed (non-fatal): {io_err}", RuntimeWarning, stacklevel=2)


def _coerce_recommendation(value: Any) -> ModellingRecommendation:
    if isinstance(value, ModellingRecommendation):
        return value
    if isinstance(value, BaseModel):
        return ModellingRecommendation.model_validate(value.model_dump())
    if isinstance(value, dict):
        return ModellingRecommendation.model_validate(value)
    raise TypeError(f"Unexpected structured_response type: {type(value)!r}")


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

    prompt = load_system_prompt_result("modeling")
    agent = create_agent(
        model=build_chat_model(),
        tools=[get_use_case_recommendation, get_column_names, get_reference_model],
        system_prompt=prompt.template,
        response_format=ModellingRecommendation,
    )
    user_message = (
        "Create a MILP formulation for optimizing production quantity per product "
        "using only tool outputs."
    )

    response = invoke_agent_with_prompt_trace(
        agent,
        stage="modeling",
        prompt=prompt,
        user_message=user_message,
    )

    structured = response.get("structured_response")
    if structured is None:
        # Fallback: attempt to parse the last AI message text as JSON.
        last_content = _last_ai_content(response.get("messages", []))
        if last_content:
            try:
                structured = ModellingRecommendation.model_validate_json(last_content)
            except Exception:
                pass
    if structured is None:
        raise ValueError("modeling agent did not produce a structured_response.")


    recommendation = _coerce_recommendation(structured)
    _persist_outputs(recommendation)

    if return_debug:
        tool_trace = extract_tool_trace(response.get("messages", []))
        return {
            "result": recommendation.model_dump(),
            "tool_trace": tool_trace,
            "debug": {"prompt": prompt_debug_payload(prompt)},
        }
    return recommendation.model_dump()


if __name__ == "__main__":
    payload = run_mathematical_modelling_agent()
    print(json.dumps(payload, indent=2))
