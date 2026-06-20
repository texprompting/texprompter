from __future__ import annotations

import json
import os
import re
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
from schemas.basemodels import (
    ParameterEstimationRecommendation,
    ParameterValue,
    ParameterRationale,
    ModellingRecommendation,
    UseCaseRecommendation,
)

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")


def _resolve_csv_path(csv_file_path: str) -> Path:
    csv_path = Path(csv_file_path)
    if csv_path.is_absolute():
        return csv_path

    data_path = get_data_dir() / csv_file_path
    if data_path.exists():
        return data_path

    return csv_path.resolve()


def _persist_outputs(recommendation: ParameterEstimationRecommendation) -> None:
    """Write parameter estimation artifacts to TestOutputs/; non-fatal if the directory is missing."""
    try:
        outputs_dir = get_test_outputs_dir()
        outputs_dir.mkdir(parents=True, exist_ok=True)
        (outputs_dir / "llm_parameter_values.json").write_text(
            json.dumps(recommendation.values_as_dict(), indent=2),
            encoding="utf-8",
        )
        (outputs_dir / "llm_updated_constraints.md").write_text(
            "\n".join(item.strip() for item in recommendation.updated_constraint_functions),
            encoding="utf-8",
        )
    except OSError as io_err:
        # Non-fatal
        import warnings
        warnings.warn(f"_persist_outputs failed (non-fatal): {io_err}", RuntimeWarning, stacklevel=2)


def _dict_to_parameter_values(d: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a {symbol: value} dict into list-of-dicts format for ParameterValue."""
    return [{"symbol": k, "value": float(v)} for k, v in d.items()]


def _dict_to_parameter_rationales(d: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a {symbol: rationale} dict into list-of-dicts format for ParameterRationale."""
    return [{"symbol": k, "rationale": str(v)} for k, v in d.items()]


def _coerce_recommendation(value: Any) -> ParameterEstimationRecommendation:
    if isinstance(value, ParameterEstimationRecommendation):
        return value
    if isinstance(value, BaseModel):
        return ParameterEstimationRecommendation.model_validate(value.model_dump())
    if isinstance(value, dict):
        # Handle legacy dict[str, float] format from cached results or alias recovery
        pv = value.get("parameter_values")
        if isinstance(pv, dict) and pv and not isinstance(next(iter(pv.values()), None), dict):
            value = dict(value)
            value["parameter_values"] = _dict_to_parameter_values(pv)
        pr = value.get("parameter_rationales")
        if isinstance(pr, dict) and pr and not isinstance(next(iter(pr.values()), None), dict):
            value = dict(value)
            value["parameter_rationales"] = _dict_to_parameter_rationales(pr)
        return ParameterEstimationRecommendation.model_validate(value)
    raise TypeError(f"Unexpected structured_response type: {type(value)!r}")


def run_parameter_estimator_agent(
    csv_file_path: str | None = None,
    use_case: UseCaseRecommendation | dict[str, Any] | None = None,
    modelling: ModellingRecommendation | dict[str, Any] | None = None,
    preview_rows: int = 5,
    return_debug: bool = False,
    max_retries: int = 5,  # Added configurable retry counter
) -> dict[str, Any]:
    """Estimate parameters based on dataset statistics and replace them in the mathematical model."""
    resolved_csv_path = _resolve_csv_path(
        csv_file_path or os.getenv("PIPELINE_CSV_PATH", "optimization_pipeline_test_easy.csv")
    )
    if not resolved_csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {resolved_csv_path}")

    # Get column names, preview and descriptive statistics
    df = pd.read_csv(resolved_csv_path)
    df_preview = df.head(preview_rows)
    
    columns_info = {
        "csv_file_path": str(resolved_csv_path),
        "columns": [str(column) for column in df.columns.tolist()],
        "preview_rows": df_preview.to_dict(orient="records"),
    }
    csv_stats = df.describe(include="all").to_string()

    # Normalize use case
    if use_case is None:
        use_case_info = {}
    elif hasattr(use_case, "model_dump"):
        use_case_info = use_case.model_dump()
    elif isinstance(use_case, dict):
        use_case_info = use_case
    else:
        use_case_info = dict(use_case)

    # Normalize modelling
    if modelling is None:
        modelling_info = {}
    elif hasattr(modelling, "model_dump"):
        modelling_info = modelling.model_dump()
    elif isinstance(modelling, dict):
        modelling_info = modelling
    else:
        modelling_info = dict(modelling)

    prompt = load_system_prompt_result("parameter_estimation")
    agent = create_agent(
        model=build_chat_model(),
        tools=[],
        system_prompt=prompt.template,
        response_format=ParameterEstimationRecommendation,
    )
    
    user_message = f"""Estimate numerical values for the parameters in the MILP model.
                        CSV Data Information (Columns & Preview):
                        {json.dumps(columns_info, indent=2)}
                        
                        CSV Descriptive Statistics:
                        {csv_stats}
                        
                        Use Case:
                        {json.dumps(use_case_info, indent=2)}
                        
                        Mathematical Model (with abstract parameters):
                        {json.dumps(modelling_info, indent=2)}
                        """
                        
    response = invoke_agent_with_prompt_trace(
        agent,
        stage="parameter_estimation",
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
                    # Extract parameter values — may be list or dict format
                    raw_pv = raw_json.get("parameter_values") or raw_json.get("values") or raw_json.get("parameters") or []
                    if isinstance(raw_pv, dict):
                        raw_pv = _dict_to_parameter_values(raw_pv)

                    raw_pr = raw_json.get("parameter_rationales") or raw_json.get("rationales") or raw_json.get("reasoning") or []
                    if isinstance(raw_pr, dict):
                        raw_pr = _dict_to_parameter_rationales(raw_pr)

                    mapped_payload = {
                        "parameter_values": raw_pv,
                        "parameter_rationales": raw_pr,
                        "updated_constraint_functions": raw_json.get("updated_constraint_functions") or raw_json.get("constraints") or raw_json.get("updated_constraints") or [],
                        "updated_objective_function": raw_json.get("updated_objective_function") or raw_json.get("objective") or raw_json.get("updated_objective") or ""
                    }
                    structured = ParameterEstimationRecommendation.model_validate(mapped_payload)
            except Exception:
                pass

    elif isinstance(structured, dict):
        # Even if 'structured_response' returned a dict, check if Gemini chose intuitive alias keys
        raw_pv = structured.get("parameter_values") or structured.get("values") or structured.get("parameters") or []
        if isinstance(raw_pv, dict):
            raw_pv = _dict_to_parameter_values(raw_pv)

        raw_pr = structured.get("parameter_rationales") or structured.get("rationales") or structured.get("reasoning") or []
        if isinstance(raw_pr, dict):
            raw_pr = _dict_to_parameter_rationales(raw_pr)

        mapped_payload = {
            "parameter_values": raw_pv,
            "parameter_rationales": raw_pr,
            "updated_constraint_functions": structured.get("updated_constraint_functions") or structured.get("constraints") or structured.get("updated_constraints") or [],
            "updated_objective_function": structured.get("updated_objective_function") or structured.get("objective") or structured.get("updated_objective") or ""
        }
        try:
            structured = ParameterEstimationRecommendation.model_validate(mapped_payload)
        except Exception:
            pass
    # --- END DEFENSIVE ALIAS RECOVERY LAYER ---

    if structured is None:
        raise ValueError("parameter estimation agent did not produce a structured_response after execution retry loops.")

    recommendation = _coerce_recommendation(structured)

                
    if modelling_info:
        if not recommendation.updated_constraint_functions:
            recommendation = recommendation.model_copy(
                update={"updated_constraint_functions": modelling_info.get("constraint_functions", [])}
            )
        if not recommendation.updated_objective_function:
            recommendation = recommendation.model_copy(
                update={"updated_objective_function": modelling_info.get("objective_function", "")}
            )
            
    # --- END CSV-STATS FALLBACK ---
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
    # Test execution
    try:
        payload = run_parameter_estimator_agent()
        print(json.dumps(payload, indent=2))
    except Exception as e:
        print(f"Error running parameter estimator agent: {e}")
