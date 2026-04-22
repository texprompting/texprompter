from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from schemas.basemodels import (
    AgentExecutionMetadata,
    AgentError,
    ModellingRecommendation,
    PipelineState,
    PreprocessingRecommendation,
    ScriptingRecommendation,
    UseCaseRecommendation,
)
from agents.shared import load_csv_input_schema

try:
    mlflow = importlib.import_module("mlflow")
except Exception:  # pragma: no cover - optional dependency
    mlflow = None


_STREAM_AGENT_OUTPUT = True
_STREAM_PIPELINE_PROGRESS = False


class PipelineStateDict(TypedDict, total=False):
    csv_file_path: str
    preview_rows: int
    status: str
    input_schema_payload: dict[str, Any]
    use_case: dict[str, Any] | None
    modelling: dict[str, Any] | None
    preprocessing: dict[str, Any] | None
    scripting: dict[str, Any] | None
    errors: list[dict[str, Any]]
    traces: list[str]
    llm_artifacts: dict[str, Any]
    execution_metadata: list[dict[str, Any]]
    skip_stages: list[str]
    retry_config: dict[str, int]
    llm_config: dict[str, str]



def _emit_progress(message: str) -> None:
    if _STREAM_PIPELINE_PROGRESS:
        print(f"[pipeline] {message}", flush=True)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    return _project_root() / "data"


def _test_outputs_dir() -> Path:
    return _project_root() / "TestOutputs"


def _resolve_csv_path(csv_file_path: str) -> Path:
    csv_path = Path(csv_file_path)
    if csv_path.is_absolute():
        return csv_path

    data_path = _data_dir() / csv_file_path
    if data_path.exists():
        return data_path

    return (_project_root() / csv_file_path).resolve()


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _serialize_for_log(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)


def _truncate_preview(value: Any, max_chars: int = 1600) -> str:
    text = _serialize_for_log(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...<truncated>"


def _safe_mlflow_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    if not attributes:
        return {}

    safe: dict[str, Any] = {}
    for key, value in attributes.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = _truncate_preview(value, max_chars=300)
    return safe


def _is_mlflow_enabled() -> bool:
    return mlflow is not None and os.getenv("MLFLOW_DISABLED", "0") != "1"


def _configure_mlflow() -> None:
    if not _is_mlflow_enabled():
        return

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        sqlite_path = (_project_root() / "mlflow.db").resolve().as_posix()
        tracking_uri = f"sqlite:///{sqlite_path}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "texprompter_pipeline"))


@contextmanager
def _mlflow_run(
    run_name: str,
    nested: bool = False,
    tags: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
):
    if not _is_mlflow_enabled():
        yield None
        return

    try:
        _configure_mlflow()
        with mlflow.start_run(run_name=run_name, nested=nested):
            if tags:
                mlflow.set_tags(tags)
            if params:
                safe_params = {key: str(value) for key, value in params.items()}
                mlflow.log_params(safe_params)
            yield True
    except Exception:
        # Keep pipeline and node tests runnable even if tracking backend is misconfigured.
        yield None


def _mlflow_log_json(artifact_file: str, payload: Any) -> None:
    if not _is_mlflow_enabled():
        return
    try:
        mlflow.log_text(_serialize_for_log(payload), artifact_file)
    except Exception:
        # Artifact logging should never break the pipeline.
        return


def _mlflow_log_metrics(metrics: dict[str, float]) -> None:
    if not _is_mlflow_enabled() or not metrics:
        return
    try:
        mlflow.log_metrics(metrics)
    except Exception:
        return


def _mlflow_set_tags(tags: dict[str, Any]) -> None:
    if not _is_mlflow_enabled() or not tags:
        return
    try:
        safe_tags = {key: str(value) for key, value in tags.items()}
        mlflow.set_tags(safe_tags)
    except Exception:
        return


def _mlflow_update_trace(
    *,
    tags: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    request_preview: str | None = None,
    response_preview: str | None = None,
    state: str | None = None,
) -> None:
    if not _is_mlflow_enabled() or not hasattr(mlflow, "update_current_trace"):
        return
    try:
        safe_tags = {key: str(value) for key, value in (tags or {}).items()} or None
        safe_metadata = {key: str(value) for key, value in (metadata or {}).items()} or None
        mlflow.update_current_trace(
            tags=safe_tags,
            metadata=safe_metadata,
            request_preview=request_preview,
            response_preview=response_preview,
            state=state,
        )
    except Exception:
        return


@contextmanager
def _mlflow_span(
    name: str,
    *,
    span_type: str = "CHAIN",
    attributes: dict[str, Any] | None = None,
):
    if not _is_mlflow_enabled() or not hasattr(mlflow, "start_span"):
        yield None
        return

    try:
        with mlflow.start_span(
            name=name,
            span_type=span_type,
            attributes=_safe_mlflow_attributes(attributes),
        ) as span:
            yield span
    except Exception:
        yield None


def _finalize_span(span: Any, *, status: str, outputs: Any | None = None, error: Exception | None = None) -> None:
    if span is None:
        return

    if outputs is not None:
        try:
            span.set_outputs(outputs)
        except Exception:
            pass

    if error is not None:
        try:
            span.record_exception(error)
        except Exception:
            pass

    try:
        span.set_status(status)
    except Exception:
        pass


def _extract_error_diagnostics(error: Exception) -> tuple[list[str], dict[str, Any]]:
    tool_calls_raw = getattr(error, "seen_tool_names", [])
    if isinstance(tool_calls_raw, list):
        tool_calls = [str(item) for item in tool_calls_raw]
    else:
        tool_calls = []

    debug_raw = getattr(error, "debug_trace", {})
    debug_payload = debug_raw if isinstance(debug_raw, dict) else {}
    return tool_calls, debug_payload


def _append_stage_artifact(
    state: PipelineState,
    *,
    stage_name: str,
    status: str,
    tool_trace: list[str],
    debug_payload: dict[str, Any],
    error_message: str | None = None,
) -> dict[str, Any]:
    attempts = debug_payload.get("attempts", []) if isinstance(debug_payload, dict) else []
    if not isinstance(attempts, list):
        attempts = []

    artifact = {
        "status": status,
        "tool_trace": tool_trace,
        "tool_call_count": len(tool_trace),
        "attempt_count": len(attempts),
        "debug": debug_payload,
        "error": error_message,
        "logged_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    llm_artifacts = dict(state.llm_artifacts)
    llm_artifacts[stage_name] = artifact
    state.llm_artifacts = llm_artifacts
    return artifact


def _log_stage_diagnostics_to_mlflow(
    *,
    stage_name: str,
    status: str,
    artifact: dict[str, Any],
) -> None:
    _mlflow_log_json(f"{stage_name}/llm_diagnostics.json", artifact)
    _mlflow_set_tags(
        {
            "stage": stage_name,
            f"{stage_name}.status": status,
            f"{stage_name}.tool_call_count": artifact.get("tool_call_count", 0),
            f"{stage_name}.attempt_count": artifact.get("attempt_count", 0),
        }
    )
    _mlflow_log_metrics(
        {
            f"{stage_name}_tool_call_count": float(artifact.get("tool_call_count", 0)),
            f"{stage_name}_llm_attempt_count": float(artifact.get("attempt_count", 0)),
        }
    )
    _mlflow_update_trace(
        tags={"stage": stage_name, "status": status},
        metadata={
            "tool_call_count": artifact.get("tool_call_count", 0),
            "attempt_count": artifact.get("attempt_count", 0),
        },
        request_preview=_truncate_preview({"stage": stage_name}),
        response_preview=_truncate_preview(
            {
                "status": status,
                "tool_trace": artifact.get("tool_trace", []),
                "error": artifact.get("error"),
            }
        ),
        state=status,
    )


def _record_execution_metadata(
    state: PipelineState,
    *,
    agent_name: str,
    started_at: float,
    status: str,
    tool_calls: list[str] | None = None,
    notes: list[str] | None = None,
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
        )
    )
    state.execution_metadata = execution_entries


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


def run_use_case_agent(
    csv_file_path: str,
    preview_rows: int = 5,
    return_debug: bool = False,
) -> UseCaseRecommendation | dict[str, Any]:
    resolved_csv_path = _resolve_csv_path(csv_file_path)
    if not resolved_csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {resolved_csv_path}")

    from agents.context_agent import run_context_agent

    raw_output = run_context_agent(
        csv_file_path=str(resolved_csv_path),
        preview_rows=preview_rows,
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
        raw_result = {
            "code": str(raw_result.get("code", "")),
            "output_schema": {
                "solution_status": "str",
                "objective_value": "float",
                "decision_variables": "dict[str, float]",
                "solver_message": "str",
            },
            "successful_implementation": bool(raw_result.get("successful_implementation", False)),
            "missing_info": [],
            "additional_info": [],
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
    with _mlflow_run(
        "initialize",
        nested=True,
        tags={"agent_name": "initialize"},
        params={
            "csv_file_path": current_state.csv_file_path,
            "preview_rows": current_state.preview_rows,
        },
    ):
        with _mlflow_span(
            "initialize.invoke",
            span_type="CHAIN",
            attributes={
                "stage": "initialize",
                "csv_file_path": current_state.csv_file_path,
                "preview_rows": current_state.preview_rows,
            },
        ) as span:
            try:
                _ensure_input_schema_payload(current_state)
                _append_trace(current_state, "initialize:ok")
                _record_execution_metadata(
                    current_state,
                    agent_name="initialize",
                    started_at=started_at,
                    status="ok",
                )
                stage_artifact = _append_stage_artifact(
                    current_state,
                    stage_name="initialize",
                    status="ok",
                    tool_trace=[],
                    debug_payload={
                        "input_schema_keys": sorted(current_state.input_schema_payload.keys()),
                        "shape": current_state.input_schema_payload.get("shape", {}),
                    },
                )
                _mlflow_log_metrics({"initialize_duration_seconds": float(time.time() - started_at)})
                _mlflow_log_json("initialize/input_schema_payload.json", current_state.input_schema_payload)
                _log_stage_diagnostics_to_mlflow(
                    stage_name="initialize",
                    status="ok",
                    artifact=stage_artifact,
                )
                _finalize_span(
                    span,
                    status="OK",
                    outputs={
                        "status": "ok",
                        "input_schema_shape": current_state.input_schema_payload.get("shape", {}),
                    },
                )
                _emit_progress("initialize:ok")
            except Exception as error:
                _set_error(current_state, "initialize", error)
                _append_trace(current_state, "initialize:error")
                _record_execution_metadata(
                    current_state,
                    agent_name="initialize",
                    started_at=started_at,
                    status="error",
                    notes=[str(error)],
                )
                stage_artifact = _append_stage_artifact(
                    current_state,
                    stage_name="initialize",
                    status="error",
                    tool_trace=[],
                    debug_payload={},
                    error_message=str(error),
                )
                _mlflow_log_json("initialize/error.json", {"error": str(error), "detail": repr(error)})
                _log_stage_diagnostics_to_mlflow(
                    stage_name="initialize",
                    status="error",
                    artifact=stage_artifact,
                )
                _finalize_span(
                    span,
                    status="ERROR",
                    outputs={"status": "error", "error": str(error)},
                    error=error,
                )
                _emit_progress(f"initialize:error - {error}")

    return current_state.model_dump()


def _append_trace(state: PipelineState, trace: str) -> None:
    traces = list(state.traces)
    traces.append(trace)
    state.traces = traces


def _set_error(state: PipelineState, agent_name: str, error: Exception) -> None:
    errors = list(state.errors)
    errors.append(
        AgentError(
            agent_name=agent_name,
            message=str(error),
            detail=repr(error),
        )
    )
    state.errors = errors
    state.status = "error"


def use_case_node(state: PipelineStateDict) -> PipelineStateDict:
    current_state = PipelineState.model_validate(state)
    if current_state.status == "error":
        return current_state.model_dump()

    if _is_stage_skipped(current_state, "use_case"):
        _append_trace(current_state, "use_case:skipped")
        return current_state.model_dump()

    started_at = time.time()
    _emit_progress("use_case:start")
    with _mlflow_run(
        "use_case_agent",
        nested=True,
        tags={"agent_name": "use_case_agent", "stage": "use_case"},
        params={
            "csv_file_path": current_state.csv_file_path,
            "preview_rows": current_state.preview_rows,
        },
    ):
        with _mlflow_span(
            "use_case_agent.invoke",
            span_type="AGENT",
            attributes={
                "stage": "use_case",
                "csv_file_path": current_state.csv_file_path,
                "preview_rows": current_state.preview_rows,
            },
        ) as span:
            tool_trace: list[str] = []
            debug_payload: dict[str, Any] = {}
            try:
                payload, tool_trace, debug_payload = _run_stage_with_optional_debug(
                    run_use_case_agent,
                    csv_file_path=current_state.csv_file_path,
                    preview_rows=current_state.preview_rows,
                )
                current_state.use_case = UseCaseRecommendation.model_validate(payload)
                _append_trace(current_state, "use_case:ok")
                _record_execution_metadata(
                    current_state,
                    agent_name="use_case_agent",
                    started_at=started_at,
                    status="ok",
                    tool_calls=tool_trace,
                )
                stage_artifact = _append_stage_artifact(
                    current_state,
                    stage_name="use_case",
                    status="ok",
                    tool_trace=tool_trace,
                    debug_payload=debug_payload,
                )
                _mlflow_log_json("use_case/recommendation.json", current_state.use_case.model_dump())
                _mlflow_log_json("use_case/tool_trace.json", tool_trace)
                _log_stage_diagnostics_to_mlflow(
                    stage_name="use_case",
                    status="ok",
                    artifact=stage_artifact,
                )
                _mlflow_log_metrics({"use_case_duration_seconds": float(time.time() - started_at)})
                _finalize_span(
                    span,
                    status="OK",
                    outputs={
                        "status": "ok",
                        "tool_trace": tool_trace,
                        "attempt_count": stage_artifact.get("attempt_count", 0),
                    },
                )
                _emit_progress("use_case:ok")
            except Exception as error:
                error_tool_trace, error_debug = _extract_error_diagnostics(error)
                if not tool_trace:
                    tool_trace = error_tool_trace
                if not debug_payload:
                    debug_payload = error_debug

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
                stage_artifact = _append_stage_artifact(
                    current_state,
                    stage_name="use_case",
                    status="error",
                    tool_trace=tool_trace,
                    debug_payload=debug_payload,
                    error_message=str(error),
                )
                _mlflow_log_json("use_case/error.json", {"error": str(error), "detail": repr(error)})
                _log_stage_diagnostics_to_mlflow(
                    stage_name="use_case",
                    status="error",
                    artifact=stage_artifact,
                )
                _finalize_span(
                    span,
                    status="ERROR",
                    outputs={"status": "error", "tool_trace": tool_trace, "error": str(error)},
                    error=error,
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
    with _mlflow_run(
        "modeling_agent",
        nested=True,
        tags={"agent_name": "modeling_agent", "stage": "modeling"},
        params={
            "csv_file_path": current_state.csv_file_path,
            "has_use_case": current_state.use_case is not None,
        },
    ):
        with _mlflow_span(
            "modeling_agent.invoke",
            span_type="AGENT",
            attributes={
                "stage": "modeling",
                "csv_file_path": current_state.csv_file_path,
                "has_use_case": current_state.use_case is not None,
            },
        ) as span:
            tool_trace: list[str] = []
            debug_payload: dict[str, Any] = {}
            try:
                payload, tool_trace, debug_payload = _run_stage_with_optional_debug(
                    run_modeling_agent,
                    csv_file_path=current_state.csv_file_path,
                    use_case=current_state.use_case,
                    preview_rows=current_state.preview_rows,
                )
                current_state.modelling = ModellingRecommendation.model_validate(payload)
                _append_trace(current_state, "modeling:ok")
                _record_execution_metadata(
                    current_state,
                    agent_name="modeling_agent",
                    started_at=started_at,
                    status="ok",
                    tool_calls=tool_trace,
                )
                stage_artifact = _append_stage_artifact(
                    current_state,
                    stage_name="modeling",
                    status="ok",
                    tool_trace=tool_trace,
                    debug_payload=debug_payload,
                )
                _mlflow_log_json("modeling/recommendation.json", current_state.modelling.model_dump())
                _mlflow_log_json("modeling/tool_trace.json", tool_trace)
                _log_stage_diagnostics_to_mlflow(
                    stage_name="modeling",
                    status="ok",
                    artifact=stage_artifact,
                )
                _mlflow_log_metrics(
                    {
                        "duration_seconds": float(time.time() - started_at),
                        "constraint_count": float(len(current_state.modelling.constraint_functions)),
                    }
                )
                _finalize_span(
                    span,
                    status="OK",
                    outputs={
                        "status": "ok",
                        "tool_trace": tool_trace,
                        "constraint_count": len(current_state.modelling.constraint_functions),
                    },
                )
                _emit_progress("modeling:ok")
            except Exception as error:
                error_tool_trace, error_debug = _extract_error_diagnostics(error)
                if not tool_trace:
                    tool_trace = error_tool_trace
                if not debug_payload:
                    debug_payload = error_debug

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
                stage_artifact = _append_stage_artifact(
                    current_state,
                    stage_name="modeling",
                    status="error",
                    tool_trace=tool_trace,
                    debug_payload=debug_payload,
                    error_message=str(error),
                )
                _mlflow_log_json("modeling/error.json", {"error": str(error), "detail": repr(error)})
                _log_stage_diagnostics_to_mlflow(
                    stage_name="modeling",
                    status="error",
                    artifact=stage_artifact,
                )
                _finalize_span(
                    span,
                    status="ERROR",
                    outputs={"status": "error", "tool_trace": tool_trace, "error": str(error)},
                    error=error,
                )
                _emit_progress(f"modeling:error - {error}")

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
    with _mlflow_run(
        "preprocessing_agent",
        nested=True,
        tags={"agent_name": "preprocessing_agent", "stage": "preprocessing"},
        params={
            "csv_file_path": current_state.csv_file_path,
            "has_modelling": current_state.modelling is not None,
            "has_use_case": current_state.use_case is not None,
        },
    ):
        with _mlflow_span(
            "preprocessing_agent.invoke",
            span_type="AGENT",
            attributes={
                "stage": "preprocessing",
                "csv_file_path": current_state.csv_file_path,
                "has_modelling": current_state.modelling is not None,
            },
        ) as span:
            tool_trace: list[str] = []
            debug_payload: dict[str, Any] = {}
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
                _append_trace(current_state, "preprocessing:ok")
                _record_execution_metadata(
                    current_state,
                    agent_name="preprocessing_agent",
                    started_at=started_at,
                    status="ok",
                    tool_calls=tool_trace,
                )
                stage_artifact = _append_stage_artifact(
                    current_state,
                    stage_name="preprocessing",
                    status="ok",
                    tool_trace=tool_trace,
                    debug_payload=debug_payload,
                )
                _mlflow_log_json(
                    "preprocessing/recommendation.json",
                    current_state.preprocessing.model_dump(),
                )
                _mlflow_log_json("preprocessing/tool_trace.json", tool_trace)
                _log_stage_diagnostics_to_mlflow(
                    stage_name="preprocessing",
                    status="ok",
                    artifact=stage_artifact,
                )
                _mlflow_log_metrics(
                    {"preprocessing_duration_seconds": float(time.time() - started_at)}
                )
                _finalize_span(
                    span,
                    status="OK",
                    outputs={
                        "status": "ok",
                        "tool_trace": tool_trace,
                        "attempt_count": stage_artifact.get("attempt_count", 0),
                    },
                )
                _emit_progress("preprocessing:ok")
            except Exception as error:
                error_tool_trace, error_debug = _extract_error_diagnostics(error)
                if not tool_trace:
                    tool_trace = error_tool_trace
                if not debug_payload:
                    debug_payload = error_debug

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
                stage_artifact = _append_stage_artifact(
                    current_state,
                    stage_name="preprocessing",
                    status="error",
                    tool_trace=tool_trace,
                    debug_payload=debug_payload,
                    error_message=str(error),
                )
                _mlflow_log_json(
                    "preprocessing/error.json", {"error": str(error), "detail": repr(error)}
                )
                _log_stage_diagnostics_to_mlflow(
                    stage_name="preprocessing",
                    status="error",
                    artifact=stage_artifact,
                )
                _finalize_span(
                    span,
                    status="ERROR",
                    outputs={"status": "error", "tool_trace": tool_trace, "error": str(error)},
                    error=error,
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
    with _mlflow_run(
        "scripting_agent",
        nested=True,
        tags={"agent_name": "scripting_agent", "stage": "scripting"},
        params={
            "csv_file_path": current_state.csv_file_path,
            "has_modelling": current_state.modelling is not None,
            "has_preprocessing": current_state.preprocessing is not None,
        },
    ):
        with _mlflow_span(
            "scripting_agent.invoke",
            span_type="AGENT",
            attributes={
                "stage": "scripting",
                "csv_file_path": current_state.csv_file_path,
                "has_modelling": current_state.modelling is not None,
                "has_preprocessing": current_state.preprocessing is not None,
            },
        ) as span:
            tool_trace: list[str] = []
            debug_payload: dict[str, Any] = {}
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
                if current_state.scripting.successful_implementation:
                    _append_trace(current_state, "scripting:ok")
                    _record_execution_metadata(
                        current_state,
                        agent_name="scripting_agent",
                        started_at=started_at,
                        status="ok",
                        tool_calls=tool_trace,
                    )
                    stage_artifact = _append_stage_artifact(
                        current_state,
                        stage_name="scripting",
                        status="ok",
                        tool_trace=tool_trace,
                        debug_payload=debug_payload,
                    )
                    _mlflow_log_json("scripting/recommendation.json", current_state.scripting.model_dump())
                    _mlflow_log_json("scripting/tool_trace.json", tool_trace)
                    _log_stage_diagnostics_to_mlflow(
                        stage_name="scripting",
                        status="ok",
                        artifact=stage_artifact,
                    )
                    _mlflow_log_metrics({"duration_seconds": float(time.time() - started_at)})
                    _finalize_span(
                        span,
                        status="OK",
                        outputs={
                            "status": "ok",
                            "tool_trace": tool_trace,
                            "successful_implementation": True,
                        },
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
                        notes=[detail],
                    )
                    stage_artifact = _append_stage_artifact(
                        current_state,
                        stage_name="scripting",
                        status="error",
                        tool_trace=tool_trace,
                        debug_payload=debug_payload,
                        error_message=detail,
                    )
                    _mlflow_log_json(
                        "scripting/error.json", {"error": detail, "detail": "successful_implementation=False"}
                    )
                    _log_stage_diagnostics_to_mlflow(
                        stage_name="scripting",
                        status="error",
                        artifact=stage_artifact,
                    )
                    _finalize_span(
                        span,
                        status="ERROR",
                        outputs={"status": "error", "tool_trace": tool_trace, "error": detail},
                        error=invalid_error,
                    )
                    _emit_progress(f"scripting:invalid - {detail}")
            except Exception as error:
                error_tool_trace, error_debug = _extract_error_diagnostics(error)
                if not tool_trace:
                    tool_trace = error_tool_trace
                if not debug_payload:
                    debug_payload = error_debug

                _set_error(current_state, "scripting_agent", error)
                _append_trace(current_state, "scripting:error")
                _record_execution_metadata(
                    current_state,
                    agent_name="scripting_agent",
                    started_at=started_at,
                    status="error",
                    tool_calls=tool_trace,
                    notes=[str(error)],
                )
                stage_artifact = _append_stage_artifact(
                    current_state,
                    stage_name="scripting",
                    status="error",
                    tool_trace=tool_trace,
                    debug_payload=debug_payload,
                    error_message=str(error),
                )
                _mlflow_log_json("scripting/error.json", {"error": str(error), "detail": repr(error)})
                _log_stage_diagnostics_to_mlflow(
                    stage_name="scripting",
                    status="error",
                    artifact=stage_artifact,
                )
                _finalize_span(
                    span,
                    status="ERROR",
                    outputs={"status": "error", "tool_trace": tool_trace, "error": str(error)},
                    error=error,
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
    builder = StateGraph(PipelineStateDict)

    builder.add_node("initialize", initialize_node)
    builder.add_node("use_case", use_case_node)
    builder.add_node("modeling", modeling_node)
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
    stream_pipeline_output: bool = False,
) -> PipelineState:
    global _STREAM_AGENT_OUTPUT, _STREAM_PIPELINE_PROGRESS

    _configure_mlflow()
    graph = build_pipeline_graph()
    initial_state = PipelineState(
        csv_file_path=csv_file_path,
        preview_rows=preview_rows,
    )

    previous_stream_agent_output = _STREAM_AGENT_OUTPUT
    previous_stream_pipeline_progress = _STREAM_PIPELINE_PROGRESS
    _STREAM_AGENT_OUTPUT = stream_pipeline_output
    _STREAM_PIPELINE_PROGRESS = stream_pipeline_output
    try:
        run_name = f"pipeline_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        with _mlflow_run(
            run_name,
            nested=False,
            tags={"component": "orchestrator.pipeline"},
            params={"csv_file_path": csv_file_path, "preview_rows": preview_rows},
        ):
            with _mlflow_span(
                "pipeline.execute",
                span_type="CHAIN",
                attributes={
                    "csv_file_path": csv_file_path,
                    "preview_rows": preview_rows,
                    "stream_pipeline_output": stream_pipeline_output,
                },
            ) as span:
                started_at = time.time()
                try:
                    if stream_pipeline_output:
                        final_state = _run_pipeline_with_streaming(graph, initial_state)
                    else:
                        result = graph.invoke(initial_state.model_dump())
                        final_state = PipelineState.model_validate(result)

                    total_duration = float(time.time() - started_at)
                    _mlflow_log_metrics(
                        {
                            "total_duration_seconds": total_duration,
                            "errors_count": float(len(final_state.errors)),
                            "completed_trace_count": float(len(final_state.traces)),
                            "execution_metadata_count": float(len(final_state.execution_metadata)),
                            "llm_artifact_stage_count": float(len(final_state.llm_artifacts)),
                        }
                    )
                    _mlflow_set_tags(
                        {
                            "pipeline.status": final_state.status,
                            "pipeline.csv_file_path": final_state.csv_file_path,
                            "pipeline.error_count": len(final_state.errors),
                        }
                    )
                    _mlflow_log_json("pipeline/final_state.json", final_state.model_dump())
                    _mlflow_log_json(
                        "pipeline/execution_metadata.json",
                        [entry.model_dump() for entry in final_state.execution_metadata],
                    )
                    _mlflow_log_json("pipeline/llm_artifacts.json", final_state.llm_artifacts)
                    _mlflow_log_json(
                        "pipeline/errors.json",
                        [entry.model_dump() for entry in final_state.errors],
                    )
                    _mlflow_update_trace(
                        tags={"component": "orchestrator.pipeline"},
                        metadata={
                            "status": final_state.status,
                            "error_count": len(final_state.errors),
                            "trace_count": len(final_state.traces),
                        },
                        request_preview=_truncate_preview(
                            {
                                "csv_file_path": csv_file_path,
                                "preview_rows": preview_rows,
                                "stream_pipeline_output": stream_pipeline_output,
                            }
                        ),
                        response_preview=_truncate_preview(
                            {
                                "status": final_state.status,
                                "errors": [entry.model_dump() for entry in final_state.errors],
                                "traces": final_state.traces,
                            }
                        ),
                        state=final_state.status,
                    )
                    _finalize_span(
                        span,
                        status="OK",
                        outputs={
                            "status": final_state.status,
                            "error_count": len(final_state.errors),
                            "trace_count": len(final_state.traces),
                        },
                    )
                    return final_state
                except Exception as error:
                    _mlflow_log_json("pipeline/error.json", {"error": str(error), "detail": repr(error)})
                    _finalize_span(
                        span,
                        status="ERROR",
                        outputs={"status": "error", "error": str(error)},
                        error=error,
                    )
                    raise
    finally:
        _STREAM_AGENT_OUTPUT = previous_stream_agent_output
        _STREAM_PIPELINE_PROGRESS = previous_stream_pipeline_progress


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
        help="Stream per-node pipeline progress and agent output while running.",
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
