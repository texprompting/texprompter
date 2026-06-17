"""Deep agent orchestrator: plan-then-execute pipeline with dynamic subagents."""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from langchain.agents import create_agent
from pydantic import BaseModel

from agents.shared import (
    _last_ai_content,
    build_chat_model,
    get_data_dir,
    get_test_outputs_dir,
    load_csv_input_schema,
    DEFAULT_AGENT_RECURSION_LIMIT,
)
from deepagent.prompts_without_predefined_agents import (
    ASSEMBLER_SYSTEM_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
    subagent_system_prompt,
)
from deepagent.schemas import (
    DeepAgentFinalOutput,
    DeepAgentState,
    ExecutionPlan,
    SubTask,
    SubTaskResult,
)
from deepagent.subagent import run_subagent
from deepagent.tools import get_tool_set

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_csv_path(csv_file_path: str) -> Path:
    csv_path = Path(csv_file_path)
    if csv_path.is_absolute():
        return csv_path

    data_path = get_data_dir() / csv_file_path
    if data_path.exists():
        return data_path

    return csv_path.resolve()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_data_summary(csv_path: Path, preview_rows: int = 5) -> str:
    """Build a compact text summary of the CSV for the orchestrator prompt."""
    df = pd.read_csv(csv_path)

    columns = list(df.columns)
    dtypes = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
    preview = df.head(preview_rows).to_string()
    stats = df.describe(include="all").to_string()

    summary = (
        f"CSV File: {csv_path.name}\n"
        f"Rows: {len(df)}, Columns: {len(columns)}\n\n"
        f"Column names: {columns}\n\n"
        f"Data types:\n{json.dumps(dtypes, indent=2)}\n\n"
        f"Preview (first {preview_rows} rows):\n{preview}\n\n"
        f"Statistics:\n{stats}\n"
    )
    return summary


def _topological_sort(tasks: list[SubTask]) -> list[SubTask]:
    """Sort tasks in dependency order (tasks with no deps first)."""
    id_to_task = {t.task_id: t for t in tasks}
    visited: set[str] = set()
    order: list[str] = []

    def _visit(tid: str) -> None:
        if tid in visited:
            return
        visited.add(tid)
        for dep in id_to_task[tid].depends_on:
            _visit(dep)
        order.append(tid)

    for task in tasks:
        _visit(task.task_id)

    return [id_to_task[tid] for tid in order]


def _build_subagent_context(
    task: SubTask,
    results: dict[str, SubTaskResult],
    csv_path: Path,
) -> str:
    """Assemble the user message for a subagent from its dependencies' outputs."""
    parts: list[str] = [f"CSV file path: {csv_path}"]

    if task.depends_on:
        parts.append("\n--- Upstream results ---")
        for dep_id in task.depends_on:
            dep_result = results.get(dep_id)
            if dep_result is None:
                parts.append(f"\n[{dep_id}]: (not available)")
            elif dep_result.status == "error":
                parts.append(f"\n[{dep_id}]: ERROR — {dep_result.error}")
            else:
                parts.append(f"\n[{dep_id}]:\n{json.dumps(dep_result.output, indent=2, default=str)}")

    parts.append(f"\n\nYour task: {task.description}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Planning phase
# ---------------------------------------------------------------------------


def _generate_plan(data_summary: str) -> ExecutionPlan:
    """Ask the orchestrator LLM to produce an execution plan."""
    print("[deep] planning...", flush=True)

    model = build_chat_model()
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        response_format=ExecutionPlan,
    )

    config = {"recursion_limit": DEFAULT_AGENT_RECURSION_LIMIT}
    payload = {
        "messages": [{"role": "user", "content": data_summary}],
    }

    response = agent.invoke(payload, config=config)

    structured = response.get("structured_response")
    if structured is None:
        last_content = _last_ai_content(response.get("messages", []))
        if last_content:
            try:
                structured = ExecutionPlan.model_validate_json(last_content)
            except Exception:
                pass
    if structured is None:
        raise ValueError("Orchestrator did not produce a valid ExecutionPlan.")

    if isinstance(structured, ExecutionPlan):
        plan = structured
    elif isinstance(structured, BaseModel):
        plan = ExecutionPlan.model_validate(structured.model_dump())
    elif isinstance(structured, dict):
        plan = ExecutionPlan.model_validate(structured)
    else:
        raise TypeError(f"Unexpected plan type: {type(structured)!r}")

    print(f"[deep] plan generated: {len(plan.tasks)} subtasks", flush=True)
    for task in plan.tasks:
        deps = f" (depends: {task.depends_on})" if task.depends_on else ""
        print(f"  • {task.task_id} [{task.task_type}]{deps}: {task.description[:80]}...", flush=True)

    return plan


# ---------------------------------------------------------------------------
# Execution phase
# ---------------------------------------------------------------------------


def _execute_plan(
    plan: ExecutionPlan,
    *,
    csv_path: Path,
    df: pd.DataFrame,
    preview_rows: int,
) -> dict[str, SubTaskResult]:
    """Execute all subtasks in dependency order."""
    print("[deep] executing plan...", flush=True)
    sorted_tasks = _topological_sort(plan.tasks)
    results: dict[str, SubTaskResult] = {}

    for task in sorted_tasks:
        # Check that all dependencies succeeded.
        failed_deps = [
            dep for dep in task.depends_on
            if dep in results and results[dep].status == "error"
        ]
        if failed_deps:
            print(f"  [subagent:{task.task_id}] skipped — dependency failed: {failed_deps}", flush=True)
            results[task.task_id] = SubTaskResult(
                task_id=task.task_id,
                status="error",
                error=f"Skipped because dependencies failed: {failed_deps}",
            )
            continue

        # Build context and tools.
        context = _build_subagent_context(task, results, csv_path)
        tools = get_tool_set(
            task.task_type,
            csv_path=csv_path,
            df=df,
            project_root=_project_root(),
            preview_rows=preview_rows,
        )
        system_prompt = subagent_system_prompt(task)

        result = run_subagent(
            task,
            system_prompt=system_prompt,
            context=context,
            tools=tools,
        )
        results[task.task_id] = result

    return results


# ---------------------------------------------------------------------------
# Assembly phase
# ---------------------------------------------------------------------------


def _assemble_final_code(
    results: dict[str, SubTaskResult],
    csv_path: Path,
) -> DeepAgentFinalOutput:
    """Run the assembler subagent to produce the final PuLP code."""
    print("[deep] assembling final code...", flush=True)

    # Build a summary of all subtask outputs.
    parts = [f"CSV file path: {csv_path}\n"]
    for task_id, result in results.items():
        if result.status == "ok":
            parts.append(f"[{task_id}]:\n{json.dumps(result.output, indent=2, default=str)}\n")
        else:
            parts.append(f"[{task_id}]: ERROR — {result.error}\n")

    user_message = (
        "Combine the following subtask outputs into a single complete, "
        "runnable PuLP Python script.\n\n" + "\n".join(parts)
    )

    model = build_chat_model()
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=ASSEMBLER_SYSTEM_PROMPT,
        response_format=DeepAgentFinalOutput,
    )

    config = {"recursion_limit": DEFAULT_AGENT_RECURSION_LIMIT}
    payload = {"messages": [{"role": "user", "content": user_message}]}

    response = agent.invoke(payload, config=config)

    structured = response.get("structured_response")
    if structured is None:
        last_content = _last_ai_content(response.get("messages", []))
        if last_content:
            try:
                structured = DeepAgentFinalOutput.model_validate_json(last_content)
            except Exception:
                pass
    if structured is None:
        raise ValueError("Assembler did not produce a valid DeepAgentFinalOutput.")

    if isinstance(structured, DeepAgentFinalOutput):
        output = structured
    elif isinstance(structured, BaseModel):
        output = DeepAgentFinalOutput.model_validate(structured.model_dump())
    elif isinstance(structured, dict):
        output = DeepAgentFinalOutput.model_validate(structured)
    else:
        raise TypeError(f"Unexpected assembler output type: {type(structured)!r}")

    # Syntax-check the generated code.
    try:
        compile(output.code, "generated_pulp_model.py", "exec")
    except SyntaxError as e:
        notes = list(output.notes)
        notes.append(f"Generated code has syntax errors: {e}")
        output = output.model_copy(
            update={"successful_implementation": False, "notes": notes}
        )

    print("[deep] assembly complete.", flush=True)
    return output


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_deep_pipeline(
    csv_file_path: str = "optimization_pipeline_test_easy.csv",
    preview_rows: int = 5,
    initial_prompt: str = "",
) -> DeepAgentState:
    """Run the deep agent pipeline: plan → execute → assemble.

    Parameters
    ----------
    csv_file_path
        Path to the CSV file (absolute or relative to the data directory).
    preview_rows
        Number of rows used for CSV previews.
    initial_prompt
        Optional user context.

    Returns
    -------
    DeepAgentState
        Final pipeline state with plan, results, code, and status.
    """
    started_at = time.time()
    state = DeepAgentState(csv_file_path=csv_file_path, initial_prompt=initial_prompt)

    try:
        # Resolve CSV.
        csv_path = _resolve_csv_path(csv_file_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        state.csv_file_path = str(csv_path)
        state.traces.append("csv_resolved")

        # Load data.
        df = pd.read_csv(csv_path)
        state.traces.append(f"csv_loaded:{len(df)}rows")

        # Phase 1: Plan.
        data_summary = _build_data_summary(csv_path, preview_rows)
        if initial_prompt:
            data_summary = (
                f"{data_summary}\n\n"
                f"<USER_REQUEST>\n{initial_prompt}\n</USER_REQUEST>\n\n"
                "IMPORTANT: The user has provided the request above. You must incorporate it into your plan. "
                "DO NOT reply to the user directly. DO NOT output any conversational text. "
                "Output ONLY the JSON tool call for ExecutionPlan."
            )
        plan = _generate_plan(data_summary)
        state.plan = plan
        state.traces.append(f"plan_generated:{len(plan.tasks)}tasks")

        # Phase 2: Execute.
        results = _execute_plan(
            plan,
            csv_path=csv_path,
            df=df,
            preview_rows=preview_rows,
        )
        state.results = list(results.values())

        ok_count = sum(1 for r in results.values() if r.status == "ok")
        err_count = sum(1 for r in results.values() if r.status == "error")
        state.traces.append(f"execution_done:ok={ok_count},errors={err_count}")

        if err_count == len(results):
            raise RuntimeError("All subtasks failed. Cannot assemble final code.")

        # Phase 3: Assemble.
        final_output = _assemble_final_code(results, csv_path)
        state.final_code = final_output.code
        state.final_output = final_output.model_dump()
        state.traces.append("assembly_done")

        # Persist the generated code.
        try:
            output_dir = get_test_outputs_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "deep_generated_pulp_model.py"
            output_path.write_text(final_output.code, encoding="utf-8")
            state.traces.append(f"code_saved:{output_path}")
        except OSError:
            pass

        if not final_output.successful_implementation:
            state.status = "error"
            state.errors.append("Assembler flagged unsuccessful implementation.")
        else:
            state.status = "ok"

    except Exception as e:
        state.status = "error"
        state.errors.append(f"{type(e).__name__}: {e}")
        state.traces.append(f"pipeline_error:{type(e).__name__}")
        print(f"[deep] pipeline error: {e}", flush=True)

    elapsed = time.time() - started_at
    state.traces.append(f"total_time:{elapsed:.1f}s")
    print(f"[deep] finished in {elapsed:.1f}s — status: {state.status}", flush=True)
    return state


def stream_deep_pipeline(
    csv_file_path: str = "optimization_pipeline_test_easy.csv",
    preview_rows: int = 5,
    initial_prompt: str = "",
    interrupt_after_modeling: bool = True,
) -> Iterator[dict[str, Any]]:
    """Stream deep agent execution state updates as a generator."""
    started_at = time.time()
    state = DeepAgentState(csv_file_path=csv_file_path, initial_prompt=initial_prompt)
    yield state.model_dump()

    try:
        csv_path = _resolve_csv_path(csv_file_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        state.csv_file_path = str(csv_path)
        state.traces.append("csv_resolved")
        yield state.model_dump()

        df = pd.read_csv(csv_path)
        state.traces.append(f"csv_loaded:{len(df)}rows")
        yield state.model_dump()

        data_summary = _build_data_summary(csv_path, preview_rows)
        if initial_prompt:
            data_summary = f"User Request/Initial Prompt:\n{initial_prompt}\n\n{data_summary}"

        plan = _generate_plan(data_summary)
        state.plan = plan
        state.traces.append(f"plan_generated:{len(plan.tasks)}tasks")
        yield state.model_dump()

        sorted_tasks = _topological_sort(plan.tasks)
        results: dict[str, SubTaskResult] = {}
        state.results = []
        
        interrupted = False
        for task in sorted_tasks:
            failed_deps = [dep for dep in task.depends_on if dep in results and results[dep].status == "error"]
            if failed_deps:
                result = SubTaskResult(
                    task_id=task.task_id,
                    status="error",
                    error=f"Skipped because dependencies failed: {failed_deps}",
                )
                results[task.task_id] = result
                state.results.append(result)
                yield state.model_dump()
                continue

            context = _build_subagent_context(task, results, csv_path)
            tools = get_tool_set(
                task.task_type,
                csv_path=csv_path,
                df=df,
                project_root=_project_root(),
                preview_rows=preview_rows,
            )
            system_prompt = subagent_system_prompt(task)

            result = run_subagent(
                task,
                system_prompt=system_prompt,
                context=context,
                tools=tools,
            )
            results[task.task_id] = result
            state.results.append(result)
            
            if "use_case" in task.task_type and result.status == "ok":
                state.use_case = result.output.get("result", result.output) if isinstance(result.output, dict) else result.output
            elif ("model" in task.task_type or "modelling" in task.task_type) and result.status == "ok":
                state.modelling = result.output.get("result", result.output) if isinstance(result.output, dict) else result.output
            elif "preprocess" in task.task_type and result.status == "ok":
                state.preprocessing = result.output.get("result", result.output) if isinstance(result.output, dict) else result.output
            elif "cod" in task.task_type and result.status == "ok":
                state.scripting = result.output.get("result", result.output) if isinstance(result.output, dict) else result.output

            state.traces.append(f"executed_task:{task.task_id}")
            yield state.model_dump()

            if interrupt_after_modeling and ("model" in task.task_type or "modelling" in task.task_type):
                state.status = "interrupted"
                state.traces.append("interrupted_for_feedback")
                interrupted = True
                yield state.model_dump()
                return

        ok_count = sum(1 for r in results.values() if r.status == "ok")
        err_count = sum(1 for r in results.values() if r.status == "error")
        state.traces.append(f"execution_done:ok={ok_count},errors={err_count}")
        yield state.model_dump()

        if err_count == len(results):
            raise RuntimeError("All subtasks failed. Cannot assemble final code.")

        final_output = _assemble_final_code(results, csv_path)
        state.final_code = final_output.code
        state.final_output = final_output.model_dump()
        state.traces.append("assembly_done")
        
        if not state.scripting:
            state.scripting = {}
        state.scripting["code"] = final_output.code
        state.scripting["output_schema"] = final_output.output_schema
        state.scripting["successful_implementation"] = final_output.successful_implementation

        try:
            output_dir = get_test_outputs_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "deep_generated_pulp_model.py"
            output_path.write_text(final_output.code, encoding="utf-8")
            state.traces.append(f"code_saved:{output_path}")
        except OSError:
            pass

        if not final_output.successful_implementation:
            state.status = "error"
            state.errors.append("Assembler flagged unsuccessful implementation.")
        else:
            state.status = "ok"

    except Exception as e:
        state.status = "error"
        state.errors.append(f"{type(e).__name__}: {e}")
        state.traces.append(f"pipeline_error:{type(e).__name__}")
        
    elapsed = time.time() - started_at
    state.traces.append(f"total_time:{elapsed:.1f}s")
    yield state.model_dump()


def continue_deep_pipeline(
    state_dict: dict[str, Any],
    feedback: str,
    preview_rows: int = 5,
) -> Iterator[dict[str, Any]]:
    """Resume a deep pipeline run, incorporating feedback for the modeling task."""
    started_at = time.time()
    state = DeepAgentState.model_validate(state_dict)
    state.status = "ok"
    yield state.model_dump()

    try:
        csv_path = _resolve_csv_path(state.csv_file_path)
        df = pd.read_csv(csv_path)

        if not state.plan:
            raise ValueError("No plan found in the provided state.")

        # Identify tasks to clear (modeling and its downstream dependencies)
        tasks_to_rerun = set()
        for task in state.plan.tasks:
            if "model" in task.task_type or "modelling" in task.task_type:
                tasks_to_rerun.add(task.task_id)

        changed = True
        while changed:
            changed = False
            for task in state.plan.tasks:
                if task.task_id not in tasks_to_rerun:
                    if any(dep in tasks_to_rerun for dep in task.depends_on):
                        tasks_to_rerun.add(task.task_id)
                        changed = True

        new_results = [r for r in state.results if r.task_id not in tasks_to_rerun]
        state.results = new_results
        results_dict = {r.task_id: r for r in state.results}

        sorted_tasks = _topological_sort(state.plan.tasks)
        for task in sorted_tasks:
            if task.task_id in results_dict:
                continue

            failed_deps = [dep for dep in task.depends_on if dep in results_dict and results_dict[dep].status == "error"]
            if failed_deps:
                result = SubTaskResult(
                    task_id=task.task_id,
                    status="error",
                    error=f"Skipped because dependencies failed: {failed_deps}",
                )
                results_dict[task.task_id] = result
                state.results.append(result)
                yield state.model_dump()
                continue

            context = _build_subagent_context(task, results_dict, csv_path)
            
            if ("model" in task.task_type or "modelling" in task.task_type) and feedback:
                context = f"USER FEEDBACK TO INCORPORATE:\n{feedback}\n\n{context}"

            tools = get_tool_set(
                task.task_type,
                csv_path=csv_path,
                df=df,
                project_root=_project_root(),
                preview_rows=preview_rows,
            )
            system_prompt = subagent_system_prompt(task)

            result = run_subagent(
                task,
                system_prompt=system_prompt,
                context=context,
                tools=tools,
            )
            results_dict[task.task_id] = result
            state.results.append(result)
            
            if ("model" in task.task_type or "modelling" in task.task_type) and result.status == "ok":
                state.modelling = result.output.get("result", result.output) if isinstance(result.output, dict) else result.output
            elif "preprocess" in task.task_type and result.status == "ok":
                state.preprocessing = result.output.get("result", result.output) if isinstance(result.output, dict) else result.output
            elif "cod" in task.task_type and result.status == "ok":
                state.scripting = result.output.get("result", result.output) if isinstance(result.output, dict) else result.output

            state.traces.append(f"executed_task:{task.task_id}")
            yield state.model_dump()

        ok_count = sum(1 for r in results_dict.values() if r.status == "ok")
        err_count = sum(1 for r in results_dict.values() if r.status == "error")
        state.traces.append(f"execution_done:ok={ok_count},errors={err_count}")
        yield state.model_dump()

        if err_count == len(results_dict):
            raise RuntimeError("All subtasks failed. Cannot assemble final code.")

        final_output = _assemble_final_code(results_dict, csv_path)
        state.final_code = final_output.code
        state.final_output = final_output.model_dump()
        state.traces.append("assembly_done")
        
        if not state.scripting:
            state.scripting = {}
        state.scripting["code"] = final_output.code
        state.scripting["output_schema"] = final_output.output_schema
        state.scripting["successful_implementation"] = final_output.successful_implementation

        try:
            output_dir = get_test_outputs_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "deep_generated_pulp_model.py"
            output_path.write_text(final_output.code, encoding="utf-8")
            state.traces.append(f"code_saved:{output_path}")
        except OSError:
            pass

        if not final_output.successful_implementation:
            state.status = "error"
            state.errors.append("Assembler flagged unsuccessful implementation.")
        else:
            state.status = "ok"

    except Exception as e:
        state.status = "error"
        state.errors.append(f"{type(e).__name__}: {e}")
        state.traces.append(f"pipeline_error:{type(e).__name__}")
        
    elapsed = time.time() - started_at
    state.traces.append(f"total_time:{elapsed:.1f}s")
    yield state.model_dump()
