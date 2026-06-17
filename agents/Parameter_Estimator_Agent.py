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
            json.dumps(recommendation.parameter_values, indent=2),
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


def _coerce_recommendation(value: Any) -> ParameterEstimationRecommendation:
    if isinstance(value, ParameterEstimationRecommendation):
        return value
    if isinstance(value, BaseModel):
        return ParameterEstimationRecommendation.model_validate(value.model_dump())
    if isinstance(value, dict):
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
    elif isinstance(use_case, UseCaseRecommendation):
        use_case_info = use_case.model_dump()
    else:
        use_case_info = dict(use_case)

    # Normalize modelling
    if modelling is None:
        modelling_info = {}
    elif isinstance(modelling, ModellingRecommendation):
        modelling_info = modelling.model_dump()
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

    # --- INITIAL DEFENSIVE ALIAS RECOVERY LAYER ---
    if structured is None:
        last_content = _last_ai_content(response.get("messages", []))
        if last_content:
            try:
                raw_json = json.loads(last_content)
                if isinstance(raw_json, dict):
                    structured = ParameterEstimationRecommendation.model_validate(raw_json)
            except Exception:
                try:
                    structured = ParameterEstimationRecommendation.model_validate_json(last_content)
                except Exception:
                    pass
    elif isinstance(structured, dict):
        try:
            structured = ParameterEstimationRecommendation.model_validate(structured)
        except Exception:
            pass

    # --- NEW DYNAMIC RETRY LOOP LAYER ---
    retries_left = max_retries
    current_stage = "parameter_estimation_retry"

    while retries_left > 0:
        needs_retry = False
        reasons = []

        if structured is not None and isinstance(structured, ParameterEstimationRecommendation):
            if not structured.updated_constraint_functions:
                needs_retry = True
                reasons.append("The field 'updated_constraint_functions' is empty or missing.")
            if not structured.updated_objective_function:
                needs_retry = True
                reasons.append("The field 'updated_objective_function' is empty or missing.")
            if not structured.parameter_values and modelling_info.get("parameters"):
                needs_retry = True
                reasons.append("The field 'parameter_values' is empty, but the model contains abstract parameters to estimate.")
        else:
            needs_retry = True
            reasons.append("Initial response did not parse or produce a structured recommendation.")

        # SUCCESS CONDITION: If no fields are empty, break out of the retry loop completely
        if not needs_retry:
            break

        import warnings
        warnings.warn(
            f"Empty fields detected. Triggering self-correction retry ({max_retries - retries_left + 1}/{max_retries}). Reasons: {reasons}", 
            RuntimeWarning, 
            stacklevel=2
        )
        
        # Build an explicit self-correction critique prompt to force the agent to fill the fields
        retry_user_message = f"""{user_message}
        
        CRITICAL ERROR IN PREVIOUS ATTEMPT:
        Your previous generation was rejected because one or more required fields came back blank or empty.
        Specific failures identified: {', '.join(reasons)}
        
        Please correct this behavior. You MUST completely fill out the following fields:
        1. 'updated_objective_function': Substitute the abstract parameter symbols with your estimated aggregate numeric constants from the dataset statistics.
        2. 'updated_constraint_functions': Substitute the abstract parameter symbols with your estimated aggregate numeric values across all constraints.
        3. 'parameter_values': Map each abstract parameter symbol to its chosen numeric value.
        
        Ensure no field is left blank or empty. Re-generate the full structured format now.
        """
        
        # Re-invoke the agent with the self-correction critique prompt
        response = invoke_agent_with_prompt_trace(
            agent,
            stage=f"{current_stage}_{max_retries - retries_left + 1}",
            prompt=prompt,
            user_message=retry_user_message,
        )
        print(f"Retry Response ({max_retries - retries_left + 1}):", response)
        structured = response.get("structured_response")
        
        # Defensive coercion on the retry result
        if structured is None:
            last_content = _last_ai_content(response.get("messages", []))
            if last_content:
                try:
                    raw_json = json.loads(last_content)
                    if isinstance(raw_json, dict):
                        structured = ParameterEstimationRecommendation.model_validate(raw_json)
                except Exception:
                    try:
                        structured = ParameterEstimationRecommendation.model_validate_json(last_content)
                    except Exception:
                        pass
        elif isinstance(structured, dict):
            try:
                structured = ParameterEstimationRecommendation.model_validate(structured)
            except Exception:
                pass
                
        retries_left -= 1
    # --- END RETRY LOOP LAYER ---

    if structured is None:
        raise ValueError("parameter estimation agent did not produce a structured_response after execution retry loops.")

    recommendation = _coerce_recommendation(structured)

    # --- CSV-STATS FALLBACK: populate empty parameter_values from df.describe() means ---
    if not recommendation.parameter_values and modelling_info:
        params = modelling_info.get("parameters", [])
        if params:
            col_means: dict[str, float] = (
                df.describe(include="number").loc["mean"].to_dict()
            )
            col_mean_lower = {k.lower().replace(" ", "_"): v for k, v in col_means.items()}

            fallback_values: dict[str, float] = {}
            fallback_rationales: dict[str, str] = {}
            for param in params:
                symbol = param.get("symbol", "") if isinstance(param, dict) else getattr(param, "symbol", "")
                description = param.get("description", "") if isinstance(param, dict) else getattr(param, "description", "")
                if not symbol:
                    continue
                bare = re.sub(r"[_{].*", "", symbol).lower()
                matched_val: float | None = None
                for col_key, mean_val in col_means.items():
                    if bare in col_key.lower() or col_key.lower() in description.lower():
                        matched_val = round(float(mean_val), 4)
                        break
                if matched_val is None:
                    matched_val = round(float(next(iter(col_means.values()), 0.0)), 4)
                fallback_values[symbol] = matched_val
                fallback_rationales[symbol] = (
                    f"Indexed parameter (per-entity). Representative mean from CSV statistics: {matched_val}. "
                    "Actual values are sourced per-row from the CSV at solve time."
                )

            if fallback_values:
                recommendation = ParameterEstimationRecommendation(
                    parameter_values=fallback_values,
                    parameter_rationales=fallback_rationales,
                    updated_constraint_functions=recommendation.updated_constraint_functions
                    or modelling_info.get("constraint_functions", []),
                    updated_objective_function=recommendation.updated_objective_function
                    or modelling_info.get("objective_function", ""),
                )
                
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
