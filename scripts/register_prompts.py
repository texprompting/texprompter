"""Idempotent registration of all agent system prompts in the MLflow Prompt Registry.

Run after editing files under ``texprompter/prompts/``::

    python -m scripts.register_prompts

A new prompt version is only created when the local file content has changed.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import mlflow

from agents.prompts import PROMPT_NAMES, _prompts_dir, _registry_name


PROMPT_STAGE_TAGS = {
    "use_case": "use_case",
    "modeling": "modeling",
    "preprocessing": "preprocessing",
    "scripting": "scripting",
}


def _setup_tracking() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        sqlite_path = (Path(__file__).resolve().parents[1] / "mlflow.db").as_posix()
        tracking_uri = f"sqlite:///{sqlite_path}"
    mlflow.set_tracking_uri(tracking_uri)


def _read_local(short_name: str) -> str:
    path = _prompts_dir() / f"{short_name}.txt"
    return path.read_text(encoding="utf-8").strip()


def _existing_template(registry_name: str) -> str | None:
    try:
        prompt = mlflow.genai.load_prompt(f"prompts:/{registry_name}@latest")
    except Exception:
        return None
    return getattr(prompt, "template", None)


def register_all(commit_message: str) -> int:
    _setup_tracking()
    created = 0
    for short_name in PROMPT_NAMES:
        registry_name = _registry_name(short_name)
        new_template = _read_local(short_name)
        existing = _existing_template(registry_name)
        if existing is not None and existing.strip() == new_template.strip():
            print(f"[skip] {registry_name}: already up to date")
            continue

        prompt = mlflow.genai.register_prompt(
            name=registry_name,
            template=new_template,
            commit_message=commit_message,
            tags={
                "stage": PROMPT_STAGE_TAGS.get(short_name, short_name),
                "component": "agent_prompt",
                "source": f"texprompter/prompts/{short_name}.txt",
            },
        )
        version = getattr(prompt, "version", "?")
        print(f"[register] {registry_name} -> v{version}")
        created += 1

    print(f"Registered {created} new prompt version(s).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Register texprompter agent prompts in MLflow.")
    parser.add_argument(
        "--message",
        default="Sync from texprompter/prompts/*.txt",
        help="Commit message attached to any newly registered prompt version.",
    )
    args = parser.parse_args()
    return register_all(commit_message=args.message)


if __name__ == "__main__":
    sys.exit(main())
