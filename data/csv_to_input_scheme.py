from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd



EASY_SCHEMA_MAPPING = {
	"Product_ID": {
		"symbol": "i",
		"role": "Product index",
	},
	"Profit_Per_Unit": {
		"symbol": "P_i",
		"role": "Objective coefficient",
	},
	"Machine_A_Hours_Req": {
		"symbol": "M_{Ai}",
		"role": "Machine A coefficient",
	},
	"Machine_B_Hours_Req": {
		"symbol": "M_{Bi}",
		"role": "Machine B coefficient",
	},
	"Labor_Hours_Req": {
		"symbol": "L_i",
		"role": "Labor coefficient",
	},
	"Raw_Material_Units_Req": {
		"symbol": "R_i",
		"role": "Raw material coefficient",
	},
	"Min_Production_Requirement": {
		"symbol": "x_i^{\\min}",
		"role": "Lower bound",
	},
	"Max_Market_Demand": {
		"symbol": "D_i",
		"role": "Upper bound",
	},
}


def _resolve_csv_path(csv_file_name: str) -> Path:
	"""Resolve a CSV path from either absolute input or this data directory."""
	csv_path = Path(csv_file_name)
	if csv_path.is_absolute():
		return csv_path
	return Path(__file__).resolve().parent / csv_file_name


def _build_schema_mapping(columns: list[str]) -> dict[str, dict[str, str]]:
	"""Return schema mapping for known columns and mark unknown columns explicitly."""
	mapping: dict[str, dict[str, str]] = {}
	for column in columns:
		if column in EASY_SCHEMA_MAPPING:
			mapping[column] = EASY_SCHEMA_MAPPING[column]
		else:
			mapping[column] = {
				"symbol": "unmapped",
				"role": "No model mapping defined",
			}
	return mapping


def _build_input_schema(
	columns: list[str],
	dtypes: dict[str, str],
	mapping: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
	"""Build an explicit per-column schema section for downstream agents."""
	schema: list[dict[str, str]] = []
	for column in columns:
		mapping_entry = mapping[column]
		schema.append(
			{
				"column_name": column,
				"dtype": dtypes[column],
				"model_symbol": mapping_entry["symbol"],
				"role": mapping_entry["role"],
			}
		)
	return schema


def _format_sample_data(df: pd.DataFrame, preview_rows: int) -> str:
	"""Create a readable markdown table preview without optional dependencies."""
	preview_df = df.head(preview_rows)
	if preview_df.empty:
		return "No sample rows available."

	headers = [str(column) for column in preview_df.columns]
	lines = [
		"| " + " | ".join(headers) + " |",
		"| " + " | ".join(["---"] * len(headers)) + " |",
	]

	for row in preview_df.itertuples(index=False, name=None):
		cells: list[str] = []
		for value in row:
			if pd.isna(value):
				cell = ""
			elif isinstance(value, float):
				cell = f"{value:.4f}".rstrip("0").rstrip(".")
			else:
				cell = str(value)

			# Escape markdown table separator if it appears in a value.
			cells.append(cell.replace("|", "\\|"))

		lines.append("| " + " | ".join(cells) + " |")

	return "\n".join(lines)


def get_input_data(
	csv_file_name: str = "optimization_pipeline_test_easy.csv",
	preview_rows: int = 5,
) -> dict[str, Any]:
	"""Load CSV data and return a lightweight, tool-call friendly input schema payload.

	The payload intentionally stays metadata-focused and includes a small formatted
	preview so downstream agents can understand the available data quickly.
	"""
	csv_path = _resolve_csv_path(csv_file_name)
	if not csv_path.exists():
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	df = pd.read_csv(csv_path)
	columns = df.columns.tolist()
	dtype_mapping = {str(column): str(dtype) for column, dtype in df.dtypes.items()}

	schema_mapping = _build_schema_mapping(columns)

	payload: dict[str, Any] = {
		"source": {
			"csv_file_name": csv_path.name,
			"csv_file_path": str(csv_path),
			"num_preview_rows": preview_rows,
		},
		"shape": {
			"rows": int(df.shape[0]),
			"columns": int(df.shape[1]),
		},
		"columns": columns,
		"dtypes": dtype_mapping,
		"schema_mapping": schema_mapping,
		"input_schema": _build_input_schema(columns, dtype_mapping, schema_mapping),
		"sample_data": _format_sample_data(df, preview_rows),

	}
	return payload