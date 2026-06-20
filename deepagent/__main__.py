"""CLI entry point: ``python -m deepagent [csv_path]``."""
from __future__ import annotations

import argparse
import json

from deepagent.orchestrator import run_deep_pipeline


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("preview_rows must be greater than 0")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deep agent optimization pipeline.",
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
    args = parser.parse_args()

    state = run_deep_pipeline(
        csv_file_path=args.csv_file_path,
        preview_rows=args.preview_rows,
    )
    print(state.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
