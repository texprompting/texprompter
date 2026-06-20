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

    # Get column names and preview
    df_preview = pd.read_csv(resolved_csv_path, nrows=preview_rows)
    columns_info = {
        "csv_file_path": str(resolved_csv_path),
        "columns": [str(column) for column in df_preview.columns.tolist()],
        "preview_rows": df_preview.to_dict(orient="records"),
    }

    # Get reference model
    reference_model = _load_reference_model()

    # Get use case recommendation
    if use_case is None:
        use_case_info = {
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
    elif hasattr(use_case, "model_dump"):
        use_case_info = use_case.model_dump()
    elif isinstance(use_case, dict):
        use_case_info = use_case
    else:
        use_case_info = dict(use_case)

    prompt = load_system_prompt_result("modeling")
    
    # Core fix area: ensuring the underlying model uses with_structured_output
    # if your custom create_agent wrapper allows it.
    agent = create_agent(
        model=build_chat_model(),
        tools=[],
        system_prompt=prompt.template,
        response_format=ModellingRecommendation,
    )
    
    user_message = f"""Create a MILP formulation for optimizing production quantity per product.
                        CSV Data Information:
                        {json.dumps(columns_info, indent=2)}
                        Reference Model:
                        {json.dumps(reference_model, indent=2)}
                        Use Case Recommendation:
                        {json.dumps(use_case_info, indent=2)}
                        """
    response = invoke_agent_with_prompt_trace(
        agent,
        stage="modeling",
        prompt=prompt,
        user_message=user_message,
    )

    print(response)

    structured = response.get("structured_response")
    
    # --- START DEFENSIVE ALIAS RECOVERY LAYER ---
    # If structured response is missing or raw parsing fails, extract raw dict and map keys safely
    if structured is None:
        last_content = _last_ai_content(response.get("messages", []))
        if last_content:
            try:
                # Attempt to parse whatever JSON text string came out
                raw_json = json.loads(last_content)
                if isinstance(raw_json, dict):
                    # Actively reconstruct the payload matching ModellingRecommendation's strict keys
                    mapped_payload = {
                        "objective_function": raw_json.get("objective_function") or raw_json.get("objective") or raw_json.get("objective_expr") or "",
                        "constraint_functions": raw_json.get("constraint_functions") or raw_json.get("constraints") or raw_json.get("constraint_list") or [],
                        "variables": raw_json.get("variables") or raw_json.get("decision_variables") or [],
                        "parameters": raw_json.get("parameters") or raw_json.get("parameter_list") or [],
                        "minimizing_problem": raw_json.get("minimizing_problem", True)
                    }
                    structured = ModellingRecommendation.model_validate(mapped_payload)
            except Exception:
                pass
                
    elif isinstance(structured, dict):
        # Even if 'structured_response' returned a dict, check if Gemini chose intuitive alias keys
        mapped_payload = {
            "objective_function": structured.get("objective_function") or structured.get("objective") or structured.get("objective_expr") or "",
            "constraint_functions": structured.get("constraint_functions") or structured.get("constraints") or structured.get("constraint_list") or [],
            "variables": structured.get("variables") or structured.get("decision_variables") or [],
            "parameters": structured.get("parameters") or structured.get("parameter_list") or [],
            "minimizing_problem": structured.get("minimizing_problem", True)
        }
        structured = ModellingRecommendation.model_validate(mapped_payload)
        
    elif isinstance(structured, ModellingRecommendation):
        # If it successfully returned a strict object, we are perfectly fine
        pass
    # --- END DEFENSIVE ALIAS RECOVERY LAYER ---

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
