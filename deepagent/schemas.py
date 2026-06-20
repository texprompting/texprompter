"""Pydantic schemas for the deep agent orchestrator."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class SubTask(BaseModel):
    """A single subtask in the orchestrator's execution plan."""

    task_id: str = Field(description="Unique identifier for this subtask, e.g. 'task_1'.")
    task_type: str = Field(
        description=(
            "Category that determines which tools the subagent receives. "
            "One of: 'data_inspection', 'use_case_analysis', 'modeling', "
            "'preprocessing', 'coding', 'validation', 'general'."
        ),
    )
    description: str = Field(
        description="Detailed instruction telling the subagent exactly what to do.",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of task_ids that must complete before this task runs.",
    )
    expected_output: str = Field(
        description="Description of what the subagent should return.",
    )


class ExecutionPlan(BaseModel):
    """The orchestrator's plan: an ordered list of subtasks with dependencies."""

    goal: str = Field(description="High-level goal for this optimization pipeline run.")
    tasks: list[SubTask] = Field(description="Subtasks to execute, in suggested order.")

    @field_validator("tasks", mode="after")
    @classmethod
    def _validate_no_cycles(cls, tasks: list[SubTask]) -> list[SubTask]:
        task_ids = {t.task_id for t in tasks}

        # Check that all depends_on references exist.
        for task in tasks:
            for dep in task.depends_on:
                if dep not in task_ids:
                    raise ValueError(
                        f"Task '{task.task_id}' depends on '{dep}' which is not in the plan."
                    )

        # Topological sort to detect cycles.
        visited: set[str] = set()
        in_stack: set[str] = set()
        id_to_task = {t.task_id: t for t in tasks}

        def _dfs(tid: str) -> None:
            if tid in in_stack:
                raise ValueError(f"Circular dependency detected involving task '{tid}'.")
            if tid in visited:
                return
            in_stack.add(tid)
            for dep in id_to_task[tid].depends_on:
                _dfs(dep)
            in_stack.discard(tid)
            visited.add(tid)

        for task in tasks:
            _dfs(task.task_id)

        return tasks


class SubTaskResult(BaseModel):
    """Result of executing a single subtask."""

    task_id: str
    status: Literal["ok", "error"] = "ok"
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class DeepAgentState(BaseModel):
    """Overall state of a deep agent pipeline run."""

    csv_file_path: str
    initial_prompt: str = ""
    plan: ExecutionPlan | None = None
    results: list[SubTaskResult] = Field(default_factory=list)
    final_code: str = ""
    final_output: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "error", "interrupted"] = "ok"
    traces: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    
    # UI compatibility fields
    use_case: dict[str, Any] | None = None
    modelling: dict[str, Any] | None = None
    preprocessing: dict[str, Any] | None = None
    scripting: dict[str, Any] | None = None


class DeepAgentFinalOutput(BaseModel):
    """Final output of the deep agent pipeline."""

    code: str = Field(description="Generated PuLP solver code.")
    output_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="Declared structure of the solver output payload.",
    )
    successful_implementation: bool = Field(
        description="True if a runnable implementation could be generated.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Extra diagnostics, warnings, or simplifications.",
    )
