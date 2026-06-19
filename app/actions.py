import os
import tempfile
import streamlit as st
from services.pipeline_service import PipelineService
from app.state import add_log
from app.api_client import (
    estimate_parameters as api_estimate_parameters,
    health_check as api_health_check,
    preprocessing as api_preprocessing,
    regenerate_feedback as api_regenerate_feedback,
    rerun_modeling as api_rerun_modeling,
    run_downstream as api_run_downstream,
    save_results as api_save_results,
    stream_start_pipeline as api_stream_start_pipeline,
    scripting as api_scripting,
)

pipeline_service = PipelineService()
USE_PIPELINE_API = os.getenv("PIPELINE_API_ENABLED", "1").strip().lower() not in ("0", "false", "no")


def reset_run_state():
    st.session_state.csv_path = None
    st.session_state.pipeline_state = {}
    st.session_state.execution_running = False
    st.session_state.execution_complete = False
    st.session_state.last_error = None
    st.session_state.error_stage = None
    st.session_state.show_modeling_intercept = False
    st.session_state.modeling_edit_mode = False
    st.session_state.result_file = None
    st.session_state.original_modeling = None
    st.session_state.agent_logs = []


def ensure_csv_path():
    if not st.session_state.csv_path:
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp:
            tmp.write(st.session_state.csv_file)
            st.session_state.csv_path = tmp.name
    return st.session_state.csv_path


def _csv_content() -> str:
    if not st.session_state.csv_file:
        raise RuntimeError("No CSV file is loaded into session state.")
    if isinstance(st.session_state.csv_file, bytes):
        return st.session_state.csv_file.decode("utf-8")
    return str(st.session_state.csv_file)


def execute_pipeline(log_renderer=None):
    ensure_csv_path()

    displayed_stages = set()
    seen_traces = 0
    add_log("Pipeline execution started...")

    if USE_PIPELINE_API:
        pipeline_iterator = api_stream_start_pipeline(
            csv_content=_csv_content(),
            initial_prompt=st.session_state.initial_prompt,
            preview_rows=5,
        )
    else:
        pipeline_iterator = pipeline_service.start_pipeline(
            csv_file_path=st.session_state.csv_path,
            preview_rows=5,
            initial_prompt=st.session_state.initial_prompt,
        )

    with st.status("Executing pipeline...", expanded=True) as status:
        for state_update in pipeline_iterator:
            if isinstance(state_update, dict):
                st.session_state.pipeline_state.update(state_update)

                traces = state_update.get("traces", [])
                if isinstance(traces, list):
                    for trace in traces[seen_traces:]:
                        add_log(trace)
                        if log_renderer is not None:
                            log_renderer()
                        status.write(f"✓ {trace}")
                    seen_traces = len(traces)

                if state_update.get("errors"):
                    error = state_update["errors"][-1]
                    st.session_state.last_error = error
                    st.session_state.error_stage = error.get("agent_name", "unknown")
                    raise Exception(f"Error in {error.get('agent_name')}: {error.get('message')}")

                if "parameter_estimation" in state_update and state_update.get("parameter_estimation"):
                    if "parameter_estimation" not in displayed_stages:
                        displayed_stages.add("parameter_estimation")
                        if st.session_state.use_intercept:
                            st.session_state.show_modeling_intercept = True
                            st.session_state.execution_running = False
                            add_log("⏸️ Human intercept triggered after Parameter Estimation.")
                            if log_renderer is not None:
                                log_renderer()
                            st.rerun()

    add_log("Pipeline execution completed.")
    if not st.session_state.use_intercept:
        add_log("Running downstream code generation...")
        execute_downstream()

    st.session_state.execution_running = False
    st.session_state.execution_complete = True


def execute_downstream():
    raw_uc = st.session_state.pipeline_state.get("use_case", {})
    raw_md = st.session_state.pipeline_state.get("modelling", {})
    raw_sc = st.session_state.pipeline_state.get("input_schema_payload", {})

    if USE_PIPELINE_API:
        downstream = api_run_downstream(
            csv_content=_csv_content(),
            use_case=raw_uc,
            modelling=raw_md,
            input_schema_payload=raw_sc,
            preview_rows=5,
        )
    else:
        downstream = pipeline_service.run_downstream(
            csv_file_path=st.session_state.csv_path,
            use_case=raw_uc,
            modelling=raw_md,
            input_schema_payload=raw_sc,
            preview_rows=5,
        )

    st.session_state.pipeline_state["preprocessing"] = downstream.get("preprocessing")
    st.session_state.pipeline_state["scripting"] = downstream.get("scripting")


def approve_and_continue():
    add_log("Approving model and generating downstream scripts...")
    c_use_case = st.session_state.pipeline_state.get("use_case", {})
    c_modeling = st.session_state.pipeline_state.get("modelling", {})
    c_schema = st.session_state.pipeline_state.get("input_schema_payload", {})

    if USE_PIPELINE_API:
        downstream = api_run_downstream(
            csv_content=_csv_content(),
            use_case=c_use_case,
            modelling=c_modeling,
            input_schema_payload=c_schema,
            preview_rows=5,
        )
    else:
        downstream = pipeline_service.run_downstream(
            csv_file_path=st.session_state.csv_path,
            use_case=c_use_case,
            modelling=c_modeling,
            input_schema_payload=c_schema,
            preview_rows=5,
        )

    st.session_state.pipeline_state["preprocessing"] = downstream.get("preprocessing")
    st.session_state.pipeline_state["scripting"] = downstream.get("scripting")
    st.session_state.show_modeling_intercept = False
    st.session_state.execution_complete = True
    add_log("✅ Downstream run completed successfully.")


def regenerate_feedback(feedback_text: str):
    add_log("Applying feedback and regenerating pipeline...")
    c_use_case = st.session_state.pipeline_state.get("use_case", {})
    c_schema = st.session_state.pipeline_state.get("input_schema_payload", {})

    if USE_PIPELINE_API:
        feedback_result = api_regenerate_feedback(
            csv_content=_csv_content(),
            use_case=c_use_case,
            input_schema_payload=c_schema,
            feedback_text=feedback_text,
            preview_rows=5,
        )
    else:
        feedback_result = pipeline_service.regenerate_from_feedback(
            csv_file_path=st.session_state.csv_path,
            use_case=c_use_case,
            input_schema_payload=c_schema,
            feedback=feedback_text,
            preview_rows=5,
        )

    st.session_state.pipeline_state.update(feedback_result)
    st.session_state.modeling_edit_mode = False
    st.session_state.show_modeling_intercept = False
    st.session_state.execution_complete = True
    add_log("🔄 Pipeline updated via feedback modifications.")


def save_results():
    if USE_PIPELINE_API:
        result = api_save_results(
            final_state=st.session_state.pipeline_state,
            csv_filename=st.session_state.csv_filename,
            initial_prompt=st.session_state.initial_prompt,
        )
        return result.get("saved_path")

    return pipeline_service.save_results(
        st.session_state.pipeline_state,
        st.session_state.csv_filename,
        st.session_state.initial_prompt,
    )
