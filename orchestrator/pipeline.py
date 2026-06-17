from __future__ import annotations

import argparse
import os
import traceback
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TypedDict

try:
    import mlflow
except ModuleNotFoundError:
    mlflow = None

try:
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError:
    END = None
    START = None
    StateGraph = None

from schemas.basemodels import (
    AgentError,
    AgentExecutionMetadata,
    ModellingRecommendation,
    ParameterEstimationRecommendation,
    PipelineState,
    PreprocessingRecommendation,
    ScriptingRecommendation,
    StallReason,
    UseCaseRecommendation,
)


_STREAM_AGENT_OUTPUT = True
_STREAM_PIPELINE_PROGRESS = False
_MLFLOW_BOOTSTRAPPED = False


class PipelineStateDict(TypedDict, total=False):
    csv_file_path: str
    preview_rows: int
    status: str
    input_schema_payload: dict[str, Any]
    use_case: dict[str, Any] | None
    modelling: dict[str, Any] | None
    parameter_estimation: dict[str, Any] | None
    preprocessing: dict[str, Any] | None
    scripting: dict[str, Any] | None
    errors: list[dict[str, Any]]
    traces: list[str]
    llm_artifacts: dict[str, Any]
    initial_prompt: str
    execution_metadata: list[dict[str, Any]]
    skip_stages: list[str]
    retry_config: dict[str, int]
    llm_config: dict[str, str]


def _emit_progress(message: str) -> None:
    if _STREAM_PIPELINE_PROGRESS:
        print(f"[pipeline] {message}", flush=True)


def _log_traceback_to_mlflow(agent_name: str) -> None:
    """Persist the active exception traceback as an artifact on the current MLflow run."""
    try:
        if mlflow is None or mlflow.active_run() is None:
            return
        mlflow.log_text(traceback.format_exc(), f"exceptions/{agent_name}.txt")
    except Exception:
        pass


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    return _project_root() / "data"


def _resolve_csv_path(csv_file_path: str) -> Path:
    csv_path = Path(csv_file_path)
    if csv_path.is_absolute():
        return csv_path

    data_path = _data_dir() / csv_file_path
    if data_path.exists():
        return data_path

    return (_project_root() / csv_file_path).resolve()


def _setup_mlflow() -> None:
    """Configure tracking URI, experiment, and enable autologging.

    Idempotent: subsequent calls are no-ops so per-node invocations don't repeatedly
    re-register autolog hooks during a single pipeline run.

    We rely *only* on ``mlflow.langchain.autolog()``: it already traces every
    ``ChatOpenAI`` call (with token usage, inputs and outputs) AND wraps each
    ``langgraph`` node so the trace tree shows ``modeling > LangGraph > model``,
    matching the use-case agent. Stacking ``mlflow.openai.autolog()`` on top
    creates a duplicate ``Completions`` span layer that, combined with
    ``langgraph``'s parallel tool fanout, could deadlock the agent loop after
    the parallel tools returned (the next model node never started). Keeping a
    single tracing layer is sufficient for our goals and stable in practice.
    """
    global _MLFLOW_BOOTSTRAPPED
    if mlflow is None:
        raise ImportError(
            "mlflow is required to run orchestrator.pipeline. "
            "Install mlflow in the active Python environment."
        )
    if _MLFLOW_BOOTSTRAPPED:
        return

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        sqlite_path = (_project_root() / "mlflow.db").resolve().as_posix()
        tracking_uri = f"sqlite:///{sqlite_path}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "texprompter_pipeline"))

    try:
        # ``run_tracer_inline=True`` is required when LangGraph fans out parallel
        # tool calls (which we do in every agent). Without it, the tracer callback
        # is offloaded to a thread pool and its context never propagates back when
        # the parallel branches join, leaving the next model node "stuck" with no
        # new requests sent to Ollama. See:
        # https://www.mlflow.org/docs/latest/tracing/integrations/langgraph#async-context-propagation
        mlflow.langchain.autolog(run_tracer_inline=True)
    except Exception as exc:  # autolog should never break the pipeline
        print(f"[pipeline] mlflow.langchain.autolog disabled: {exc}", flush=True)

    _MLFLOW_BOOTSTRAPPED = True


def _load_input_schema_payload(csv_file_path: str, preview_rows: int) -> dict[str, Any]:
    resolved_csv_path = _resolve_csv_path(csv_file_path)
    if not resolved_csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {resolved_csv_path}")
    from agents.shared import load_csv_input_schema

    return load_csv_input_schema(
        csv_file_path=str(resolved_csv_path),
        preview_rows=preview_rows,
    )


def _normalize_use_case(use_case: Any | None) -> UseCaseRecommendation | None:
    if use_case is None:
        return None
    if isinstance(use_case, UseCaseRecommendation):
        return use_case
    return UseCaseRecommendation.model_validate(use_case)


def _normalize_modelling(modelling: Any | None) -> ModellingRecommendation | None:
    if modelling is None:
        return None
    if isinstance(modelling, ModellingRecommendation):
        return modelling
    return ModellingRecommendation.model_validate(modelling)


def _validate_and_extend_use_case_feedback(use_case: Any | None, feedback: str) -> UseCaseRecommendation | None:
    validated = _normalize_use_case(use_case)
    if validated is None:
        return None
    assumptions = list(validated.assumptions or [])
    assumptions.append(f"User feedback: {feedback}")
    return validated.model_copy(update={"assumptions": assumptions})


def _record_execution_metadata(
    state: PipelineState,
    *,
    agent_name: str,
    started_at: float,
    status: str,
    tool_calls: list[str] | None = None,
    notes: list[str] | None = None,
    steps_used: int | None = None,
    context_chars: int | None = None,
    prompt_source: str | None = None,
) -> None:
    execution_entries = list(state.execution_metadata)
    duration = time.time() - started_at
    started_ts = datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat()
    completed_ts = datetime.now(tz=timezone.utc).isoformat()
    execution_entries.append(
        AgentExecutionMetadata(
            agent_name=agent_name,
            started_at=started_ts,
            completed_at=completed_ts,
            duration_seconds=duration,
            status="ok" if status == "ok" else "error",
            tool_calls=tool_calls or [],
            notes=notes or [],
            steps_used=steps_used,
            context_chars=context_chars,
            prompt_source=prompt_source,
        )
    )
    state.execution_metadata = execution_entries

    # Emit per-agent MLflow metrics so execution data is queryable in the UI.
    try:
        if mlflow is None or mlflow.active_run() is None:
            return
        mlflow.log_metric(f"{agent_name}.duration_seconds", duration)
        if steps_used is not None:
            mlflow.log_metric(f"{agent_name}.steps_used", float(steps_used))
        if context_chars is not None:
            mlflow.log_metric(f"{agent_name}.context_chars", float(context_chars))
    except Exception:
        pass


def _infer_objective_direction(text: str) -> str:
    lowered = text.lower()
    if "min" in lowered:
        return "min"
    return "max"


def _extract_result_and_debug(payload: Any) -> tuple[Any, list[str], dict[str, Any]]:
    if isinstance(payload, dict) and "result" in payload:
        result_payload = payload.get("result")
        tool_trace = payload.get("tool_trace", [])
        debug_payload = payload.get("debug", {})
        if isinstance(tool_trace, list):
            trace = [str(item) for item in tool_trace]
        else:
            trace = []
        if not isinstance(debug_payload, dict):
            debug_payload = {}
        return result_payload, trace, debug_payload
    return payload, [], {}


def _extract_exception_debug(error: Exception) -> tuple[list[str], dict[str, Any]]:
    tool_trace = getattr(error, "tool_trace", [])
    debug_payload = getattr(error, "debug", {})
    if isinstance(tool_trace, list):
        trace = [str(item) for item in tool_trace]
    else:
        trace = []
    if not isinstance(debug_payload, dict):
        debug_payload = {}
    return trace, debug_payload


def _debug_notes(debug_payload: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    notes.extend(_prompt_notes(debug_payload))
    milestones = debug_payload.get("milestones")
    if isinstance(milestones, list) and milestones:
        events = [
            str(item.get("event"))
            for item in milestones
            if isinstance(item, dict) and item.get("event")
        ]
        if events:
            notes.append(f"debug_milestones={','.join(events)}")
    model = debug_payload.get("model")
    if isinstance(model, dict):
        model_parts = [
            f"{key}={value}"
            for key, value in model.items()
            if value is not None
        ]
        if model_parts:
            notes.append(f"debug_model={';'.join(model_parts)}")
    return notes


def _extract_prompt_metadata(debug_payload: dict[str, Any]) -> dict[str, str]:
    prompt_payload = debug_payload.get("prompt")
    if not isinstance(prompt_payload, Mapping):
        return {}
    return {
        str(key): str(value)
        for key, value in prompt_payload.items()
        if value is not None and key != "template"
    }


def _prompt_notes(debug_payload: dict[str, Any]) -> list[str]:
    prompt_metadata = _extract_prompt_metadata(debug_payload)
    if not prompt_metadata:
        return []
    notes: list[str] = []
    uri = prompt_metadata.get("resolved_uri") or prompt_metadata.get("requested_uri")
    if uri:
        notes.append(f"prompt_uri={uri}")
    source = prompt_metadata.get("source")
    if source:
        notes.append(f"prompt_source={source}")
    version = prompt_metadata.get("version")
    if version:
        notes.append(f"prompt_version={version}")
    return notes


def _record_prompt_lineage(
    state: PipelineState,
    *,
    stage_name: str,
    debug_payload: dict[str, Any],
) -> None:
    prompt_metadata = _extract_prompt_metadata(debug_payload)
    if not prompt_metadata:
        return

    artifacts = dict(state.llm_artifacts)
    prompts = dict(artifacts.get("prompts", {})) if isinstance(artifacts.get("prompts"), dict) else {}
    prompts[stage_name] = prompt_metadata
    artifacts["prompts"] = prompts
    state.llm_artifacts = artifacts

    try:
        if mlflow is None or mlflow.active_run() is None:
            return
        tags: dict[str, str] = {
            f"prompt.{stage_name}.{key}": value
            for key, value in prompt_metadata.items()
            if key in {"registry_name", "requested_uri", "resolved_uri", "version", "source"}
        }
        # Log registry miss so fallback rate is queryable per experiment.
        source = prompt_metadata.get("source", "")
        if source == "local_file":
            fallback_reason = prompt_metadata.get("fallback_reason", "")
            if fallback_reason:
                tags[f"prompt.{stage_name}.fallback_reason"] = fallback_reason
            mlflow.log_metric("prompt.registry_miss", 1.0)
        mlflow.set_tags(tags)
    except Exception:
        pass


def run_use_case_agent(
    csv_file_path: str,
    preview_rows: int = 5,
    initial_prompt: str = "",
    return_debug: bool = False,
) -> UseCaseRecommendation | dict[str, Any]:
    resolved_csv_path = _resolve_csv_path(csv_file_path)
    if not resolved_csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {resolved_csv_path}")

    from agents.context_agent import run_context_agent

    raw_output = run_context_agent(
        csv_file_path=str(resolved_csv_path),
        preview_rows=preview_rows,
        initial_prompt=initial_prompt,
        return_debug=return_debug,
    )
    raw_result, tool_trace, debug_payload = _extract_result_and_debug(raw_output)
    if not isinstance(raw_result, dict):
        raise ValueError("context_agent did not return a valid ContextRecommendation payload.")

    objective = str(raw_result.get("objective", "")).strip()
    recommendation = UseCaseRecommendation(
        use_case_name=str(raw_result.get("use_case", "production_optimization")).strip(),
        business_goal=objective or "Optimize production performance.",
        objective_direction=_infer_objective_direction(objective),
        objective_variable=objective or "production target",
        decision_variables=[str(item) for item in raw_result.get("decision_variables", [])],
        required_columns=[str(item) for item in raw_result.get("relevant_columns", [])],
        constraints_to_consider=[],
        assumptions=[],
        rationale=str(raw_result.get("reasoning", "Generated by context_agent.")).strip(),
    )
    if return_debug:
        return {
            "result": recommendation.model_dump(),
            "tool_trace": tool_trace,
            "debug": debug_payload,
        }
    return recommendation


def run_modeling_agent(
    csv_file_path: str,
    use_case: UseCaseRecommendation | None,
    preview_rows: int = 5,
    return_debug: bool = False,
) -> ModellingRecommendation | dict[str, Any]:
    resolved_csv_path = _resolve_csv_path(csv_file_path)
    if not resolved_csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {resolved_csv_path}")

    from agents.Mathematical_modelling import run_mathematical_modelling_agent

    raw_payload = run_mathematical_modelling_agent(
        csv_file_path=str(resolved_csv_path),
        use_case=use_case,
        preview_rows=preview_rows,
        return_debug=return_debug,
    )
    raw_result, tool_trace, debug_payload = _extract_result_and_debug(raw_payload)
    recommendation = ModellingRecommendation.model_validate(raw_result)
    if return_debug:
        return {
            "result": recommendation.model_dump(),
            "tool_trace": tool_trace,
            "debug": debug_payload,
        }
    return recommendation


def run_parameter_estimation_agent(
    csv_file_path: str,
    use_case: UseCaseRecommendation | None,
    modelling: ModellingRecommendation | None,
    preview_rows: int = 5,
    return_debug: bool = False,
) -> ParameterEstimationRecommendation | dict[str, Any]:
    resolved_csv_path = _resolve_csv_path(csv_file_path)
    if not resolved_csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {resolved_csv_path}")

    from agents.Parameter_Estimator_Agent import run_parameter_estimator_agent

    raw_payload = run_parameter_estimator_agent(
        csv_file_path=str(resolved_csv_path),
        use_case=use_case,
        modelling=modelling,
        preview_rows=preview_rows,
        return_debug=return_debug,
    )
    raw_result, tool_trace, debug_payload = _extract_result_and_debug(raw_payload)
    recommendation = ParameterEstimationRecommendation.model_validate(raw_result)
    if return_debug:
        return {
            "result": recommendation.model_dump(),
            "tool_trace": tool_trace,
            "debug": debug_payload,
        }
    return recommendation


def run_preprocessing_agent(
    csv_file_path: str,
    use_case: UseCaseRecommendation | None,
    modelling: ModellingRecommendation | None,
    input_schema_payload: dict[str, Any],
    preview_rows: int = 5,
    return_debug: bool = False,
) -> PreprocessingRecommendation | dict[str, Any]:
    resolved_csv_path = _resolve_csv_path(csv_file_path)
    if not resolved_csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {resolved_csv_path}")

    from agents.Data_Processor_Agent import run_data_processor_agent

    raw_payload = run_data_processor_agent(
        csv_file_path=str(resolved_csv_path),
        use_case=use_case,
        modelling=modelling,
        input_schema_payload=input_schema_payload,
        preview_rows=preview_rows,
        return_debug=return_debug,
    )
    raw_result, tool_trace, debug_payload = _extract_result_and_debug(raw_payload)
    if not isinstance(raw_result, dict):
        raise ValueError("Data_Processor_Agent did not return a valid payload.")

    mapping_notes = [str(item) for item in raw_result.get("mapping_explanation", [])]
    mapping_notes.extend(str(item) for item in raw_result.get("preprocessing_steps", []))
    recommendation = PreprocessingRecommendation(
        input_schema_payload=input_schema_payload,
        mapper_script=str(raw_result.get("full_script", "")).strip(),
        mapping_notes=mapping_notes,
        assumptions=[str(item) for item in raw_result.get("assumptions", [])],
    )

    if return_debug:
        return {
            "result": recommendation.model_dump(),
            "tool_trace": tool_trace,
            "debug": debug_payload,
        }
    return recommendation


def run_scripting_agent(
    csv_file_path: str,
    modelling: ModellingRecommendation | None,
    preprocessing: PreprocessingRecommendation | None,
    input_schema_payload: dict[str, Any],
    preview_rows: int = 5,
    return_debug: bool = False,
) -> ScriptingRecommendation | dict[str, Any]:
    resolved_csv_path = _resolve_csv_path(csv_file_path)
    if not resolved_csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {resolved_csv_path}")

    from agents.Pulp_Coding_Agent import run_pulp_coding_agent

    raw_payload = run_pulp_coding_agent(
        csv_file_path=str(resolved_csv_path),
        modelling=modelling,
        preprocessing=preprocessing,
        input_schema_payload=input_schema_payload,
        preview_rows=preview_rows,
        return_debug=return_debug,
    )
    raw_result, tool_trace, debug_payload = _extract_result_and_debug(raw_payload)
    if not isinstance(raw_result, dict):
        raise ValueError("Pulp_Coding_Agent did not return a valid payload.")

    if "output_schema" not in raw_result:
        # Extract what the sandbox actually calculated from the flat dictionary
        actual_status = str(raw_result.get("solution_status", "Executed Cleanly"))
        actual_objective = raw_result.get("objective_value")
        actual_variables = raw_result.get("decision_variables", {})
        actual_message = str(raw_result.get("solver_message", ""))

        raw_result = {
            "code": str(raw_result.get("code", "")),
            "solution_status": actual_status,
            "objective_value": actual_objective,
            "decision_variables": actual_variables,
            "solver_message": actual_message,
            "output_schema": {
                "solution_status": actual_status,
                "objective_value": actual_objective,
                "decision_variables": actual_variables,
                "solver_message": actual_message,
            },
            "successful_implementation": bool(raw_result.get("successful_implementation", False)),
            "missing_info": raw_result.get("missing_info", []),
            "additional_info": raw_result.get("additional_info", []),
        }

    recommendation = ScriptingRecommendation.model_validate(raw_result)

    if return_debug:
        return {
            "result": recommendation.model_dump(),
            "tool_trace": tool_trace,
            "debug": debug_payload,
        }
    return recommendation


def _is_stage_skipped(state: PipelineState, stage_name: str) -> bool:
    return stage_name in state.skip_stages


def _ensure_input_schema_payload(state: PipelineState) -> None:
    if state.input_schema_payload:
        return

    resolved_csv_path = _resolve_csv_path(state.csv_file_path)
    if not resolved_csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {resolved_csv_path}")

    state.csv_file_path = str(resolved_csv_path)
    from agents.shared import load_csv_input_schema

    state.input_schema_payload = load_csv_input_schema(
        csv_file_path=str(resolved_csv_path),
        preview_rows=state.preview_rows,
    )


def _run_stage_with_optional_debug(
    runner: Any,
    **kwargs: Any,
) -> tuple[Any, list[str], dict[str, Any]]:
    try:
        payload = runner(return_debug=True, **kwargs)
    except TypeError:
        payload = runner(**kwargs)

    result, tool_trace, debug_payload = _extract_result_and_debug(payload)
    return result, tool_trace, debug_payload


def initialize_node(state: PipelineStateDict) -> PipelineStateDict:
    current_state = PipelineState.model_validate(state)
    if current_state.status == "error":
        return current_state.model_dump()

    started_at = time.time()
    _emit_progress("initialize:start")
    try:
        _ensure_input_schema_payload(current_state)
        _append_trace(current_state, "initialize:ok")
        _record_execution_metadata(
            current_state,
            agent_name="initialize",
            started_at=started_at,
            status="ok",
        )
        _emit_progress("initialize:ok")
    except Exception as error:
        _log_traceback_to_mlflow("initialize")
        _set_error(current_state, "initialize", error)
        _append_trace(current_state, "initialize:error")
        _record_execution_metadata(
            current_state,
            agent_name="initialize",
            started_at=started_at,
            status="error",
            notes=[str(error)],
        )
        _emit_progress(f"initialize:error - {error}")

    return current_state.model_dump()


def _append_trace(state: PipelineState, trace: str) -> None:
    traces = list(state.traces)
    traces.append(trace)
    state.traces = traces


def _set_error(
    state: PipelineState,
    agent_name: str,
    error: Exception,
    *,
    stall_reason: StallReason | None = None,
    steps_used: int | None = None,
    context_chars: int | None = None,
) -> None:
    from agents.shared import classify_exception

    reason = stall_reason if stall_reason is not None else classify_exception(error)
    errors = list(state.errors)
    errors.append(
        AgentError(
            agent_name=agent_name,
            message=str(error),
            detail=repr(error),
            stall_reason=reason,
            retry_steps_used=steps_used,
            context_chars=context_chars,
        )
    )
    state.errors = errors
    state.status = "error"

    # Emit stall classification to MLflow so failure distributions are queryable.
    try:
        if mlflow is None or mlflow.active_run() is None:
            return
        mlflow.log_metric(f"stall.{agent_name}.{reason.value}", 1.0)
        mlflow.set_tag(f"stall.{agent_name}.reason", reason.value)
        if steps_used is not None:
            mlflow.log_metric(f"{agent_name}.steps_used", float(steps_used))
    except Exception:
        pass


def use_case_node(state: PipelineStateDict) -> PipelineStateDict:
    current_state = PipelineState.model_validate(state)
    if current_state.status == "error":
        return current_state.model_dump()

    if _is_stage_skipped(current_state, "use_case"):
        _append_trace(current_state, "use_case:skipped")
        return current_state.model_dump()

    started_at = time.time()
    _emit_progress("use_case:start")
    tool_trace: list[str] = []
    try:
        payload, tool_trace, debug_payload = _run_stage_with_optional_debug(
            run_use_case_agent,
            csv_file_path=current_state.csv_file_path,
            preview_rows=current_state.preview_rows,
            initial_prompt=current_state.initial_prompt,
        )
        current_state.use_case = UseCaseRecommendation.model_validate(payload)
        _record_prompt_lineage(current_state, stage_name="use_case", debug_payload=debug_payload)
        _append_trace(current_state, "use_case:ok")
        _record_execution_metadata(
            current_state,
            agent_name="use_case_agent",
            started_at=started_at,
            status="ok",
            tool_calls=tool_trace,
            notes=_debug_notes(debug_payload),
        )
        _emit_progress("use_case:ok")
    except Exception as error:
        _log_traceback_to_mlflow("use_case_agent")
        _set_error(current_state, "use_case_agent", error)
        _append_trace(current_state, "use_case:error")
        _record_execution_metadata(
            current_state,
            agent_name="use_case_agent",
            started_at=started_at,
            status="error",
            tool_calls=tool_trace,
            notes=[str(error)],
        )
        _emit_progress(f"use_case:error - {error}")

    return current_state.model_dump()


def modeling_node(state: PipelineStateDict) -> PipelineStateDict:
    current_state = PipelineState.model_validate(state)
    if current_state.status == "error":
        return current_state.model_dump()

    if _is_stage_skipped(current_state, "modeling"):
        _append_trace(current_state, "modeling:skipped")
        return current_state.model_dump()

    started_at = time.time()
    _emit_progress("modeling:start")
    tool_trace: list[str] = []
    try:
        payload, tool_trace, debug_payload = _run_stage_with_optional_debug(
            run_modeling_agent,
            csv_file_path=current_state.csv_file_path,
            use_case=current_state.use_case,
            preview_rows=current_state.preview_rows,
        )
        current_state.modelling = ModellingRecommendation.model_validate(payload)
        _record_prompt_lineage(current_state, stage_name="modeling", debug_payload=debug_payload)
        _append_trace(current_state, "modeling:ok")
        _record_execution_metadata(
            current_state,
            agent_name="modeling_agent",
            started_at=started_at,
            status="ok",
            tool_calls=tool_trace,
            notes=_debug_notes(debug_payload),
        )
        _emit_progress("modeling:ok")
    except Exception as error:
        _log_traceback_to_mlflow("modeling_agent")
        _set_error(current_state, "modeling_agent", error)
        _append_trace(current_state, "modeling:error")
        _record_execution_metadata(
            current_state,
            agent_name="modeling_agent",
            started_at=started_at,
            status="error",
            tool_calls=tool_trace,
            notes=[str(error)],
        )
        _emit_progress(f"modeling:error - {error}")

    return current_state.model_dump()


def parameter_estimation_node(state: PipelineStateDict) -> PipelineStateDict:
    current_state = PipelineState.model_validate(state)
    if current_state.status == "error":
        return current_state.model_dump()

    if _is_stage_skipped(current_state, "parameter_estimation"):
        _append_trace(current_state, "parameter_estimation:skipped")
        return current_state.model_dump()

    started_at = time.time()
    _emit_progress("parameter_estimation:start")
    tool_trace: list[str] = []
    try:
        payload, tool_trace, debug_payload = _run_stage_with_optional_debug(
            run_parameter_estimation_agent,
            csv_file_path=current_state.csv_file_path,
            use_case=current_state.use_case,
            modelling=current_state.modelling,
            preview_rows=current_state.preview_rows,
        )
        current_state.parameter_estimation = ParameterEstimationRecommendation.model_validate(payload)
        
        # Update modelling in-place with estimated values so downstream nodes consume concrete equations
        if current_state.modelling is not None:
            current_state.modelling = current_state.modelling.model_copy(
                update={
                    "constraint_functions": current_state.parameter_estimation.updated_constraint_functions,
                    "objective_function": current_state.parameter_estimation.updated_objective_function,
                }
            )

        _record_prompt_lineage(current_state, stage_name="parameter_estimation", debug_payload=debug_payload)
        _append_trace(current_state, "parameter_estimation:ok")
        _record_execution_metadata(
            current_state,
            agent_name="parameter_estimation_agent",
            started_at=started_at,
            status="ok",
            tool_calls=tool_trace,
            notes=_debug_notes(debug_payload),
        )
        _emit_progress("parameter_estimation:ok")
    except Exception as error:
        _log_traceback_to_mlflow("parameter_estimation_agent")
        _set_error(current_state, "parameter_estimation_agent", error)
        _append_trace(current_state, "parameter_estimation:error")
        _record_execution_metadata(
            current_state,
            agent_name="parameter_estimation_agent",
            started_at=started_at,
            status="error",
            tool_calls=tool_trace,
            notes=[str(error)],
        )
        _emit_progress(f"parameter_estimation:error - {error}")

    return current_state.model_dump()


def preprocessing_node(state: PipelineStateDict) -> PipelineStateDict:
    current_state = PipelineState.model_validate(state)
    if current_state.status == "error":
        return current_state.model_dump()

    if _is_stage_skipped(current_state, "preprocessing"):
        _append_trace(current_state, "preprocessing:skipped")
        return current_state.model_dump()

    started_at = time.time()
    _emit_progress("preprocessing:start")
    tool_trace: list[str] = []
    try:
        payload, tool_trace, debug_payload = _run_stage_with_optional_debug(
            run_preprocessing_agent,
            csv_file_path=current_state.csv_file_path,
            use_case=current_state.use_case,
            modelling=current_state.modelling,
            input_schema_payload=current_state.input_schema_payload,
            preview_rows=current_state.preview_rows,
        )
        current_state.preprocessing = PreprocessingRecommendation.model_validate(payload)
        _record_prompt_lineage(current_state, stage_name="preprocessing", debug_payload=debug_payload)
        _append_trace(current_state, "preprocessing:ok")
        _record_execution_metadata(
            current_state,
            agent_name="preprocessing_agent",
            started_at=started_at,
            status="ok",
            tool_calls=tool_trace,
            notes=_debug_notes(debug_payload),
        )
        _emit_progress("preprocessing:ok")
    except Exception as error:
        _log_traceback_to_mlflow("preprocessing_agent")
        _set_error(current_state, "preprocessing_agent", error)
        _append_trace(current_state, "preprocessing:error")
        _record_execution_metadata(
            current_state,
            agent_name="preprocessing_agent",
            started_at=started_at,
            status="error",
            tool_calls=tool_trace,
            notes=[str(error)],
        )
        _emit_progress(f"preprocessing:error - {error}")

    return current_state.model_dump()


def scripting_node(state: PipelineStateDict) -> PipelineStateDict:
    current_state = PipelineState.model_validate(state)
    if current_state.status == "error":
        return current_state.model_dump()

    if _is_stage_skipped(current_state, "scripting"):
        _append_trace(current_state, "scripting:skipped")
        return current_state.model_dump()

    started_at = time.time()
    _emit_progress("scripting:start")
    tool_trace: list[str] = []
    try:
        payload, tool_trace, debug_payload = _run_stage_with_optional_debug(
            run_scripting_agent,
            csv_file_path=current_state.csv_file_path,
            modelling=current_state.modelling,
            preprocessing=current_state.preprocessing,
            input_schema_payload=current_state.input_schema_payload,
            preview_rows=current_state.preview_rows,
        )
        current_state.scripting = ScriptingRecommendation.model_validate(payload)
        _record_prompt_lineage(current_state, stage_name="scripting", debug_payload=debug_payload)
        if current_state.scripting.successful_implementation:
            _append_trace(current_state, "scripting:ok")
            _record_execution_metadata(
                current_state,
                agent_name="scripting_agent",
                started_at=started_at,
                status="ok",
                tool_calls=tool_trace,
                notes=_debug_notes(debug_payload),
            )
            _emit_progress("scripting:ok")
        else:
            detail = "; ".join(current_state.scripting.additional_info).strip()
            if not detail:
                detail = "Scripting agent returned successful_implementation=False."
            invalid_error = ValueError(detail)
            _set_error(current_state, "scripting_agent", invalid_error)
            _append_trace(current_state, "scripting:invalid")
            _record_execution_metadata(
                current_state,
                agent_name="scripting_agent",
                started_at=started_at,
                status="error",
                tool_calls=tool_trace,
                notes=[detail, *_debug_notes(debug_payload)],
            )
            _emit_progress(f"scripting:invalid - {detail}")
    except Exception as error:
        _log_traceback_to_mlflow("scripting_agent")
        exception_tool_trace, debug_payload = _extract_exception_debug(error)
        if exception_tool_trace:
            tool_trace = exception_tool_trace
        _record_prompt_lineage(current_state, stage_name="scripting", debug_payload=debug_payload)
        _set_error(current_state, "scripting_agent", error)
        _append_trace(current_state, "scripting:error")
        _record_execution_metadata(
            current_state,
            agent_name="scripting_agent",
            started_at=started_at,
            status="error",
            tool_calls=tool_trace,
            notes=[str(error), *_debug_notes(debug_payload)],
        )
        _emit_progress(f"scripting:error - {error}")

    return current_state.model_dump()


def _status_router(state: PipelineStateDict) -> str:
    return "stop" if state.get("status") == "error" else "continue"


def _run_pipeline_with_streaming(graph: Any, initial_state: PipelineState) -> PipelineState:
    seen_traces = 0
    seen_errors = 0
    final_state: dict[str, Any] = initial_state.model_dump()

    print("[pipeline] start", flush=True)
    for state_update in graph.stream(initial_state.model_dump(), stream_mode="values"):
        if not isinstance(state_update, Mapping):
            continue

        state_dict = dict(state_update)
        final_state = state_dict

        traces = state_dict.get("traces", [])
        if isinstance(traces, list):
            for trace in traces[seen_traces:]:
                print(f"[pipeline] {trace}", flush=True)
            seen_traces = len(traces)

        errors = state_dict.get("errors", [])
        if isinstance(errors, list) and len(errors) > seen_errors:
            for error in errors[seen_errors:]:
                if isinstance(error, dict):
                    agent_name = str(error.get("agent_name", "unknown_agent"))
                    message = str(error.get("message", "unknown error"))
                    print(f"[pipeline] error in {agent_name}: {message}", flush=True)
            seen_errors = len(errors)

    print("[pipeline] done", flush=True)
    return PipelineState.model_validate(final_state)


def build_pipeline_graph() -> Any:
    if StateGraph is None or START is None or END is None:
        raise ImportError(
            "langgraph is required to build the orchestrator pipeline graph. "
            "Install langgraph in the active Python environment."
        )
    builder = StateGraph(PipelineStateDict)

    builder.add_node("initialize", initialize_node)
    builder.add_node("use_case", use_case_node)
    builder.add_node("modeling", modeling_node)
    builder.add_node("parameter_estimation", parameter_estimation_node)
    builder.add_node("preprocessing", preprocessing_node)
    builder.add_node("scripting", scripting_node)

    builder.add_edge(START, "initialize")
    builder.add_conditional_edges(
        "initialize",
        _status_router,
        {
            "continue": "use_case",
            "stop": END,
        },
    )
    builder.add_conditional_edges(
        "use_case",
        _status_router,
        {
            "continue": "modeling",
            "stop": END,
        },
    )
    builder.add_conditional_edges(
        "modeling",
        _status_router,
        {
            "continue": "parameter_estimation",
            "stop": END,
        },
    )
    builder.add_conditional_edges(
        "parameter_estimation",
        _status_router,
        {
            "continue": "preprocessing",
            "stop": END,
        },
    )
    builder.add_conditional_edges(
        "preprocessing",
        _status_router,
        {
            "continue": "scripting",
            "stop": END,
        },
    )
    builder.add_edge("scripting", END)

    return builder.compile()


def run_agent_node(agent_name: str, state: PipelineStateDict) -> PipelineState:
    """Run one normalized LangGraph-style node directly for isolated testing/fine-tuning."""
    nodes: dict[str, Any] = {
        "initialize": initialize_node,
        "use_case": use_case_node,
        "modeling": modeling_node,
        "parameter_estimation": parameter_estimation_node,
        "preprocessing": preprocessing_node,
        "scripting": scripting_node,
    }
    if agent_name not in nodes:
        raise ValueError(f"Unknown agent node '{agent_name}'. Expected one of: {', '.join(nodes)}")

    node_output = nodes[agent_name](state)
    return PipelineState.model_validate(node_output)


def run_pipeline(
    csv_file_path: str = "optimization_pipeline_test_easy.csv",
    preview_rows: int = 5,
    initial_prompt: str = "",
    stream_pipeline_output: bool = False,
) -> PipelineState:
    global _STREAM_AGENT_OUTPUT, _STREAM_PIPELINE_PROGRESS

    _setup_mlflow()
    graph = build_pipeline_graph()
    initial_state = PipelineState(
        csv_file_path=csv_file_path,
        preview_rows=preview_rows,
        initial_prompt=initial_prompt,
    )

    previous_stream_agent_output = _STREAM_AGENT_OUTPUT
    previous_stream_pipeline_progress = _STREAM_PIPELINE_PROGRESS
    _STREAM_AGENT_OUTPUT = stream_pipeline_output
    _STREAM_PIPELINE_PROGRESS = stream_pipeline_output
    prev_llm_stream_env = os.environ.get("OLLAMA_STREAM_STDOUT") if stream_pipeline_output else None
    if stream_pipeline_output:
        os.environ["OLLAMA_STREAM_STDOUT"] = "1"
    try:
        run_name = f"pipeline_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        mlflow_run_started = False
        mlflow_status = "FAILED"
        try:
            active_parent_run = mlflow.active_run()
            start_run_kwargs: dict[str, Any] = {"run_name": run_name}
            if active_parent_run is not None:
                start_run_kwargs["nested"] = True
            mlflow.start_run(**start_run_kwargs)
            mlflow_run_started = True
            mlflow.set_tags(
                {
                    "component": "orchestrator.pipeline",
                    "pipeline.csv_file_path": csv_file_path,
                }
            )
            mlflow.log_params(
                {
                    "csv_file_path": csv_file_path,
                    "preview_rows": preview_rows,
                    "ollama_request_timeout_s": os.getenv("OLLAMA_REQUEST_TIMEOUT_S", "600"),
                    "ollama_max_tokens": os.getenv("OLLAMA_MAX_TOKENS", "unset"),
                    "agent_recursion_limit": os.getenv("AGENT_RECURSION_LIMIT", "12"),
                    "scripting_max_context_chars": os.getenv("SCRIPTING_MAX_CONTEXT_CHARS", "24000"),
                }
            )

            if stream_pipeline_output:
                final_state = _run_pipeline_with_streaming(graph, initial_state)
            else:
                result = graph.invoke(initial_state.model_dump())
                final_state = PipelineState.model_validate(result)

            mlflow.set_tag("pipeline.status", final_state.status)
            mlflow.log_metric("errors_count", float(len(final_state.errors)))
            mlflow.log_metric("trace_count", float(len(final_state.traces)))
            mlflow_status = "FINISHED" if final_state.status == "ok" else "FAILED"
            return final_state
        finally:
            if mlflow_run_started:
                mlflow.end_run(status=mlflow_status)
    finally:
        if stream_pipeline_output:
            if prev_llm_stream_env is None:
                os.environ.pop("OLLAMA_STREAM_STDOUT", None)
            else:
                os.environ["OLLAMA_STREAM_STDOUT"] = prev_llm_stream_env
        _STREAM_AGENT_OUTPUT = previous_stream_agent_output
        _STREAM_PIPELINE_PROGRESS = previous_stream_pipeline_progress


def stream_pipeline(
    csv_file_path: str = "optimization_pipeline_test_easy.csv",
    preview_rows: int = 5,
    initial_prompt: str = "",
) -> Iterator[dict[str, Any]]:
    """Stream pipeline state updates while tracing execution with MLflow."""
    global _STREAM_AGENT_OUTPUT, _STREAM_PIPELINE_PROGRESS

    _setup_mlflow()
    graph = build_pipeline_graph()
    initial_state = PipelineState(
        csv_file_path=csv_file_path,
        preview_rows=preview_rows,
        initial_prompt=initial_prompt,
    )

    previous_stream_agent_output = _STREAM_AGENT_OUTPUT
    previous_stream_pipeline_progress = _STREAM_PIPELINE_PROGRESS
    _STREAM_AGENT_OUTPUT = True
    _STREAM_PIPELINE_PROGRESS = True
    prev_llm_stream_env = os.environ.get("OLLAMA_STREAM_STDOUT")
    os.environ["OLLAMA_STREAM_STDOUT"] = "1"

    run_name = f"pipeline_stream_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    mlflow_run_started = False
    mlflow_status = "FAILED"
    try:
        active_parent_run = mlflow.active_run()
        start_run_kwargs: dict[str, Any] = {"run_name": run_name}
        if active_parent_run is not None:
            start_run_kwargs["nested"] = True
        mlflow.start_run(**start_run_kwargs)
        mlflow_run_started = True
        mlflow.set_tags(
            {
                "component": "orchestrator.pipeline",
                "pipeline.csv_file_path": csv_file_path,
                "pipeline.mode": "stream",
            }
        )
        mlflow.log_params(
            {
                "csv_file_path": csv_file_path,
                "preview_rows": preview_rows,
                "ollama_request_timeout_s": os.getenv("OLLAMA_REQUEST_TIMEOUT_S", "600"),
                "ollama_max_tokens": os.getenv("OLLAMA_MAX_TOKENS", "unset"),
                "agent_recursion_limit": os.getenv("AGENT_RECURSION_LIMIT", "12"),
                "scripting_max_context_chars": os.getenv("SCRIPTING_MAX_CONTEXT_CHARS", "24000"),
            }
        )

        try:
            for state_update in graph.stream(initial_state.model_dump(), stream_mode="values"):
                if not isinstance(state_update, Mapping):
                    continue
                yield dict(state_update)
            mlflow_status = "FINISHED"
        finally:
            if mlflow_run_started:
                mlflow.end_run(status=mlflow_status)
    finally:
        if prev_llm_stream_env is None:
            os.environ.pop("OLLAMA_STREAM_STDOUT", None)
        else:
            os.environ["OLLAMA_STREAM_STDOUT"] = prev_llm_stream_env
        _STREAM_AGENT_OUTPUT = previous_stream_agent_output
        _STREAM_PIPELINE_PROGRESS = previous_stream_pipeline_progress


def run_downstream_agents(
    csv_file_path: str,
    use_case: Any | None,
    modelling: Any | None,
    input_schema_payload: dict[str, Any] | None,
    preview_rows: int = 5,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run preprocessing and scripting agents from an existing pipeline state."""
    _setup_mlflow()
    resolved_use_case = _normalize_use_case(use_case)
    resolved_modelling = _normalize_modelling(modelling)
    resolved_input_schema_payload = (
        input_schema_payload
        if input_schema_payload
        else _load_input_schema_payload(csv_file_path, preview_rows)
    )

    run_name = f"pipeline_downstream_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    mlflow_run_started = False
    mlflow_status = "FAILED"
    try:
        active_parent_run = mlflow.active_run()
        start_run_kwargs: dict[str, Any] = {"run_name": run_name}
        if active_parent_run is not None:
            start_run_kwargs["nested"] = True
        mlflow.start_run(**start_run_kwargs)
        mlflow_run_started = True

        mlflow.set_tags(
            {
                "component": "orchestrator.pipeline",
                "pipeline.csv_file_path": csv_file_path,
                "pipeline.mode": "downstream",
            }
        )
        mlflow.log_params(
            {
                "csv_file_path": csv_file_path,
                "preview_rows": preview_rows,
                "ollama_request_timeout_s": os.getenv("OLLAMA_REQUEST_TIMEOUT_S", "600"),
                "ollama_max_tokens": os.getenv("OLLAMA_MAX_TOKENS", "unset"),
                "agent_recursion_limit": os.getenv("AGENT_RECURSION_LIMIT", "12"),
                "scripting_max_context_chars": os.getenv("SCRIPTING_MAX_CONTEXT_CHARS", "24000"),
            }
        )

        pe_result = run_parameter_estimation_agent(
            csv_file_path=csv_file_path,
            use_case=resolved_use_case,
            modelling=resolved_modelling,
            preview_rows=preview_rows,
        )
        if isinstance(pe_result, dict) and "result" in pe_result:
            pe_rec = ParameterEstimationRecommendation.model_validate(pe_result["result"])
        else:
            pe_rec = ParameterEstimationRecommendation.model_validate(pe_result)
            
        if resolved_modelling is not None:
            resolved_modelling = resolved_modelling.model_copy(
                update={
                    "constraint_functions": pe_rec.updated_constraint_functions,
                    "objective_function": pe_rec.updated_objective_function,
                }
            )

        preprocessing_result = run_preprocessing_agent(
            csv_file_path=csv_file_path,
            use_case=resolved_use_case,
            modelling=resolved_modelling,
            input_schema_payload=resolved_input_schema_payload,
            preview_rows=preview_rows,
        )
        scripting_result = run_scripting_agent(
            csv_file_path=csv_file_path,
            modelling=resolved_modelling,
            preprocessing=preprocessing_result,
            input_schema_payload=resolved_input_schema_payload,
            preview_rows=preview_rows,
        )

        mlflow_status = "FINISHED"
        return (
            resolved_modelling.model_dump(),
            pe_rec.model_dump(),
            preprocessing_result.model_dump(),
            scripting_result.model_dump(),
        )
    finally:
        if mlflow_run_started:
            mlflow.end_run(status=mlflow_status)


def rerun_modeling_with_feedback(
    csv_file_path: str,
    feedback: str,
    use_case: Any | None,
    current_modelling: Any | None,
    input_schema_payload: dict[str, Any] | None,
    preview_rows: int = 5,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Rerun modeling and downstream agents based on user feedback."""
    _setup_mlflow()
    resolved_use_case = _validate_and_extend_use_case_feedback(use_case, feedback)
    resolved_input_schema_payload = (
        input_schema_payload
        if input_schema_payload
        else _load_input_schema_payload(csv_file_path, preview_rows)
    )

    run_name = f"pipeline_feedback_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    mlflow_run_started = False
    mlflow_status = "FAILED"
    try:
        active_parent_run = mlflow.active_run()
        start_run_kwargs: dict[str, Any] = {"run_name": run_name}
        if active_parent_run is not None:
            start_run_kwargs["nested"] = True
        mlflow.start_run(**start_run_kwargs)
        mlflow_run_started = True

        mlflow.set_tags(
            {
                "component": "orchestrator.pipeline",
                "pipeline.csv_file_path": csv_file_path,
                "pipeline.mode": "feedback",
            }
        )
        mlflow.log_params(
            {
                "csv_file_path": csv_file_path,
                "preview_rows": preview_rows,
                "ollama_request_timeout_s": os.getenv("OLLAMA_REQUEST_TIMEOUT_S", "600"),
                "ollama_max_tokens": os.getenv("OLLAMA_MAX_TOKENS", "unset"),
                "agent_recursion_limit": os.getenv("AGENT_RECURSION_LIMIT", "12"),
                "scripting_max_context_chars": os.getenv("SCRIPTING_MAX_CONTEXT_CHARS", "24000"),
            }
        )

        if _normalize_modelling(current_modelling) is None:
            raise ValueError("Current modelling output is required for feedback rerun.")

        modelling_result = run_modeling_agent(
            csv_file_path=csv_file_path,
            use_case=resolved_use_case,
            preview_rows=preview_rows,
        )
        if isinstance(modelling_result, dict) and "result" in modelling_result:
            modelling_rec = ModellingRecommendation.model_validate(modelling_result["result"])
        else:
            modelling_rec = ModellingRecommendation.model_validate(modelling_result)

        pe_result = run_parameter_estimation_agent(
            csv_file_path=csv_file_path,
            use_case=resolved_use_case,
            modelling=modelling_rec,
            preview_rows=preview_rows,
        )
        if isinstance(pe_result, dict) and "result" in pe_result:
            pe_rec = ParameterEstimationRecommendation.model_validate(pe_result["result"])
        else:
            pe_rec = ParameterEstimationRecommendation.model_validate(pe_result)
            
        updated_modelling = modelling_rec.model_copy(
            update={
                "constraint_functions": pe_rec.updated_constraint_functions,
                "objective_function": pe_rec.updated_objective_function,
            }
        )

        preprocessing_result = run_preprocessing_agent(
            csv_file_path=csv_file_path,
            use_case=resolved_use_case,
            modelling=updated_modelling,
            input_schema_payload=resolved_input_schema_payload,
            preview_rows=preview_rows,
        )
        scripting_result = run_scripting_agent(
            csv_file_path=csv_file_path,
            modelling=updated_modelling,
            preprocessing=preprocessing_result,
            input_schema_payload=resolved_input_schema_payload,
            preview_rows=preview_rows,
        )

        mlflow_status = "FINISHED"
        return (
            updated_modelling.model_dump(),
            pe_rec.model_dump(),
            preprocessing_result.model_dump(),
            scripting_result.model_dump(),
        )
    finally:
        if mlflow_run_started:
            mlflow.end_run(status=mlflow_status)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("preview_rows must be greater than 0")
    return parsed


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LangGraph optimization pipeline.",
    )
    parser.add_argument(
        "csv_file_path",
        nargs="?",
        default="optimization_pipeline_test_easy.csv",
        help="Path to the CSV file to analyze.",
    )
    parser.add_argument(
        "--preview-rows",
        type=_positive_int,
        default=5,
        help="Number of rows loaded for the quick CSV preview.",
    )
    parser.add_argument(
        "--stream-pipeline-output",
        action="store_true",
        help="Stream per-node pipeline progress; also enables live LLM token streaming (stdout).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    final_state = run_pipeline(
        csv_file_path=args.csv_file_path,
        preview_rows=args.preview_rows,
        stream_pipeline_output=args.stream_pipeline_output,
    )
    print(final_state.model_dump_json(indent=2))
