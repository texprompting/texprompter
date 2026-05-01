"""Run ``mlflow.genai.evaluate`` over the seed dataset.

Usage::

    python -m evaluation.run_eval                # deterministic scorers only
    python -m evaluation.run_eval --with-judge   # also run the LLM-as-judge scorer
"""
from __future__ import annotations

import argparse
import os
import sys

import mlflow

from agents.shared import ensure_mlflow_openai_base_url_for_ollama_judge
from evaluation.mlflow_judge_patch import apply_invoke_judge_model_openai_base_url_patch
from evaluation.datasets import load_seed_dataset
from evaluation.scorers import DETERMINISTIC_SCORERS, objective_aligned_judge
from orchestrator.pipeline import _setup_mlflow, run_pipeline


def predict_fn(csv_file_path: str) -> dict:
    """Adapter passed to ``mlflow.genai.evaluate``.

    Returns the serialized ``PipelineState`` so scorers can assert on its dict shape.
    """
    final_state = run_pipeline(csv_file_path=csv_file_path)
    return final_state.model_dump()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the texprompter pipeline.")
    parser.add_argument(
        "--with-judge",
        action="store_true",
        help=(
            "Include the LLM-as-judge scorer (objective_aligned). Uses OPENAI_* env vars; "
            "if OPENAI_BASE_URL is unset but OLLAMA_BASE_URL or OPENAI_API_KEY=ollama is set, "
            "the judge targets Ollama's OpenAI-compatible API like the pipeline."
        ),
    )
    args = parser.parse_args()

    # MLflow's builtin Guidelines judge never passes base_url → LiteLLM defaults to api.openai.com.
    apply_invoke_judge_model_openai_base_url_patch()
    ensure_mlflow_openai_base_url_for_ollama_judge()

    _setup_mlflow()

    scorers = list(DETERMINISTIC_SCORERS)
    if args.with_judge:
        scorers.append(objective_aligned_judge)

    dataset = load_seed_dataset()
    scorer_names = [getattr(s, "name", getattr(s, "__name__", repr(s))) for s in scorers]
    print(f"[eval] dataset rows={len(dataset)} scorers={scorer_names}")

    # Parallel predict rows via ThreadPoolExecutor; default MLflow env is 10.
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")

    results = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=predict_fn,
        scorers=scorers,
    )
    print("[eval] done; see MLflow UI -> Runs -> Evaluation tab for details.")
    print(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
