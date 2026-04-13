from __future__ import annotations

import argparse
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.sample_Mathematical_modeling import run_modeling_agent
from agents.sample_Pulp_Coding_Agent import run_scripting_agent
from agents.sample_preprocessing_agent import run_preprocessing_agent
from agents.sample_use_case_agent import run_use_case_agent
from schemas.basemodels import AgentError, PipelineState


class PipelineStateDict(TypedDict, total=False):
    csv_file_path: str
    preview_rows: int
    status: str
    use_case: dict[str, Any] | None
    modelling: dict[str, Any] | None
    preprocessing: dict[str, Any] | None
    scripting: dict[str, Any] | None
    errors: list[dict[str, Any]]
    traces: list[str]


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

    try:
        current_state.use_case = run_use_case_agent(
            csv_file_path=current_state.csv_file_path,
            preview_rows=current_state.preview_rows,
        )
        _append_trace(current_state, "use_case:ok")
    except Exception as error:
        _set_error(current_state, "use_case_agent", error)
        _append_trace(current_state, "use_case:error")

    return current_state.model_dump()


def modeling_node(state: PipelineStateDict) -> PipelineStateDict:
    current_state = PipelineState.model_validate(state)
    if current_state.status == "error":
        return current_state.model_dump()

    try:
        current_state.modelling = run_modeling_agent(
            csv_file_path=current_state.csv_file_path,
            use_case=current_state.use_case,
            preview_rows=current_state.preview_rows,
        )
        _append_trace(current_state, "modeling:ok")
    except Exception as error:
        _set_error(current_state, "modeling_agent", error)
        _append_trace(current_state, "modeling:error")

    return current_state.model_dump()


def preprocessing_node(state: PipelineStateDict) -> PipelineStateDict:
    current_state = PipelineState.model_validate(state)
    if current_state.status == "error":
        return current_state.model_dump()

    try:
        current_state.preprocessing = run_preprocessing_agent(
            csv_file_path=current_state.csv_file_path,
            use_case=current_state.use_case,
            modelling=current_state.modelling,
            preview_rows=current_state.preview_rows,
        )
        _append_trace(current_state, "preprocessing:ok")
    except Exception as error:
        _set_error(current_state, "preprocessing_agent", error)
        _append_trace(current_state, "preprocessing:error")

    return current_state.model_dump()


def scripting_node(state: PipelineStateDict) -> PipelineStateDict:
    current_state = PipelineState.model_validate(state)
    if current_state.status == "error":
        return current_state.model_dump()

    try:
        current_state.scripting = run_scripting_agent(
            csv_file_path=current_state.csv_file_path,
            modelling=current_state.modelling,
            preprocessing=current_state.preprocessing,
            preview_rows=current_state.preview_rows,
        )
        if current_state.scripting.successful_implementation:
            _append_trace(current_state, "scripting:ok")
        else:
            detail = "; ".join(current_state.scripting.additional_info).strip()
            if not detail:
                detail = "Scripting agent returned successful_implementation=False."
            _set_error(current_state, "scripting_agent", ValueError(detail))
            _append_trace(current_state, "scripting:invalid")
    except Exception as error:
        _set_error(current_state, "scripting_agent", error)
        _append_trace(current_state, "scripting:error")

    return current_state.model_dump()


def _status_router(state: PipelineStateDict) -> str:
    return "stop" if state.get("status") == "error" else "continue"


def build_pipeline_graph() -> Any:
    builder = StateGraph(PipelineStateDict)

    builder.add_node("use_case", use_case_node)
    builder.add_node("modeling", modeling_node)
    builder.add_node("preprocessing", preprocessing_node)
    builder.add_node("scripting", scripting_node)

    builder.add_edge(START, "use_case")
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


def run_pipeline(
    csv_file_path: str = "optimization_pipeline_test_easy.csv",
    preview_rows: int = 5,
) -> PipelineState:
    graph = build_pipeline_graph()
    initial_state = PipelineState(
        csv_file_path=csv_file_path,
        preview_rows=preview_rows,
    )
    result = graph.invoke(initial_state.model_dump())
    return PipelineState.model_validate(result)


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    final_state = run_pipeline(
        csv_file_path=args.csv_file_path,
        preview_rows=args.preview_rows,
    )
    print(final_state.model_dump_json(indent=2))
