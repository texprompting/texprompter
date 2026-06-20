"""Tool registry for deep agent subagents.

Maps task types to sets of LangChain tools that each subagent receives.
"""
from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool


def _make_data_inspection_tools(
    df: pd.DataFrame,
    csv_path: Path,
    preview_rows: int = 5,
) -> list[Any]:
    """Tools for exploring the CSV dataset."""

    @tool
    def get_column_names() -> list[str]:
        """Returns all column names of the dataset."""
        return [str(c) for c in df.columns.tolist()]

    @tool
    def get_csv_preview() -> str:
        """Returns the first rows of the dataset as a formatted table."""
        return df.head(preview_rows).to_string()

    @tool
    def get_basic_stats() -> str:
        """Returns a statistical summary (describe) of all columns."""
        return df.describe(include="all").to_string()

    @tool
    def get_data_types() -> dict[str, str]:
        """Returns the data type of each column."""
        return {str(col): str(dtype) for col, dtype in df.dtypes.items()}

    @tool
    def get_row_count() -> int:
        """Returns the number of rows in the dataset."""
        return len(df)

    @tool
    def get_unique_values(column_name: str) -> str:
        """Returns the unique values for a given column (max 50 shown)."""
        if column_name not in df.columns:
            return f"Column '{column_name}' not found. Available: {list(df.columns)}"
        uniques = df[column_name].unique()
        if len(uniques) > 50:
            return f"{len(uniques)} unique values. First 50: {list(uniques[:50])}"
        return str(list(uniques))

    return [get_column_names, get_csv_preview, get_basic_stats, get_data_types, get_row_count, get_unique_values]


def _make_modeling_tools(csv_path: Path) -> list[Any]:
    """Tools for mathematical modeling."""

    data_dir = csv_path.parent
    ref_model_path = data_dir / "ReferenceMathematicalModel.json"

    @tool
    def get_reference_model() -> str:
        """Returns a reference mathematical model for notation and structure guidance."""
        if not ref_model_path.exists():
            # Walk up to find it in the canonical data dir
            alt = Path(__file__).resolve().parents[1] / "data" / "ReferenceMathematicalModel.json"
            if alt.exists():
                return alt.read_text(encoding="utf-8")
            return "{}"
        return ref_model_path.read_text(encoding="utf-8")

    return [get_reference_model]


def _make_validation_tools(project_root: Path) -> list[Any]:
    """Tools for validating generated Python / PuLP code."""

    @tool
    def validate_python_syntax(code: str) -> str:
        """Checks Python code for syntax errors. Returns 'OK' or the error message."""
        try:
            compile(code, "<generated>", "exec")
            return "OK — no syntax errors."
        except SyntaxError as e:
            return f"SyntaxError at line {e.lineno}: {e.msg}"

    @tool
    def execute_python_code(code: str) -> str:
        """Executes Python code in a sandboxed subprocess and returns stdout+stderr.

        The code is run with -I (isolated) flag. Timeout is 30 seconds.
        """
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-c", code],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=30,
            )
            output = (result.stdout + result.stderr).strip()
            if result.returncode != 0:
                return f"EXIT CODE {result.returncode}\n{output}"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "ERROR: execution timed out after 30 seconds."
        except Exception as e:
            return f"ERROR: {e}"

    return [validate_python_syntax, execute_python_code]


def get_tool_set(
    task_type: str,
    *,
    csv_path: Path,
    df: pd.DataFrame,
    project_root: Path,
    preview_rows: int = 5,
) -> list[Any]:
    """Return the appropriate set of tools for a given task type."""

    tools: list[Any] = []

    # Data inspection tools are broadly useful — give them to most task types.
    if task_type in ("data_inspection", "use_case_analysis", "modeling", "preprocessing", "coding"):
        tools.extend(_make_data_inspection_tools(df, csv_path, preview_rows))

    if task_type in ("modeling",):
        tools.extend(_make_modeling_tools(csv_path))

    if task_type in ("validation", "coding"):
        tools.extend(_make_validation_tools(project_root))

    return tools
