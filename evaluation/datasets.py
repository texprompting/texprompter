"""Seed evaluation dataset for the texprompter pipeline.

The project does not have labelled ground-truth outputs (no fine-tuning, no expected
solver answers), so this dataset only carries inputs. The evaluation relies on
deterministic structural scorers plus an optional LLM-as-judge scorer; see
``evaluation/scorers.py``.
"""
from __future__ import annotations

from pathlib import Path

from agents.shared import get_data_dir


SEED_CSVS: tuple[str, ...] = (
    "optimization_pipeline_test_easy.csv",
    "optimization_pipeline_easy.csv",
    "optimization_pipeline_test_medium.csv",
)

# Matches the checked-in folder name under ``data/`` (industrial line modules).
VERSATILE_PRODUCTION_RELDIR = "versatile_producion_system"


def _resolve(rel_path: str) -> Path:
    return get_data_dir() / rel_path


def _evaluation_csv_relative_paths() -> list[str]:
    """Paths relative to ``data/``: fixed seeds plus every ``*.csv`` in the versatile line folder."""
    rel: list[str] = list(SEED_CSVS)
    versatile_dir = get_data_dir() / VERSATILE_PRODUCTION_RELDIR
    if not versatile_dir.is_dir():
        return rel

    extras = sorted(
        p.relative_to(get_data_dir()).as_posix() for p in versatile_dir.glob("*.csv")
    )
    seen = set(rel)
    for extra in extras:
        if extra not in seen:
            rel.append(extra)
            seen.add(extra)
    return rel


def load_seed_dataset() -> list[dict]:
    """Return the in-memory dataset compatible with ``mlflow.genai.evaluate``.

    Each entry is shaped ``{"inputs": {"csv_file_path": <path>}}`` to match the
    signature of ``predict_fn(csv_file_path)`` in ``run_eval.py``. Missing files
    are skipped with a warning so the harness still runs in partial environments.

    Rows include every ``*.csv`` under ``data/VERSATILE_PRODUCTION_RELDIR`` when that
    directory exists (see ``VERSATILE_PRODUCTION_RELDIR``), as well as the fixed
    ``SEED_CSVS`` filenames.
    """
    dataset: list[dict] = []
    for name in _evaluation_csv_relative_paths():
        path = _resolve(name)
        if not path.exists():
            print(f"[evaluation] skipping missing seed CSV: {path}")
            continue
        dataset.append({"inputs": {"csv_file_path": str(path)}})

    if not dataset:
        raise FileNotFoundError(
            "No seed CSVs found. Add at least one of "
            f"{SEED_CSVS!r} under {get_data_dir()}, or CSVs under "
            f"{get_data_dir() / VERSATILE_PRODUCTION_RELDIR}."
        )

    return dataset
