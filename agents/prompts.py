"""Helpers for loading agent system prompts from the MLflow Prompt Registry.

The registry is the source of truth for production runs; the local ``prompts/*.txt``
files are the human-edited copies that get registered (and serve as a fallback when
the registry is unreachable so tests and offline runs keep working).
"""
from __future__ import annotations

import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

PROMPT_NAMESPACE: Final = "texprompter"
PROMPT_NAMES: Final = ("use_case", "modeling", "preprocessing", "scripting")


class PromptLoadError(RuntimeError):
    """Raised when a prompt cannot be loaded from either the registry or the local file.

    Both the MLflow Prompt Registry and the local ``prompts/<name>.txt`` fallback
    have been exhausted.  Sending an empty or missing prompt to the LLM would
    produce garbage output and unpredictable structured-output failures, so we
    fail fast here instead.
    """



@dataclass(frozen=True)
class PromptLoadResult:
    """Resolved prompt content plus lineage metadata for MLflow runs and traces."""

    short_name: str
    registry_name: str
    requested_uri: str
    resolved_uri: str | None
    version: str | None
    template: str
    source: str
    fallback_reason: str | None = None

    def as_dict(self) -> dict[str, str]:
        return {
            key: str(value)
            for key, value in asdict(self).items()
            if value is not None and key != "template"
        }


def _prompts_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "prompts"


def _registry_name(short_name: str) -> str:
    # MLflow's prompt registry only accepts alphanumerics, hyphens, underscores,
    # and dots in names -- so we use a dot as the namespace separator.
    return f"{PROMPT_NAMESPACE}.{short_name}"


def _load_local(short_name: str) -> str:
    path = _prompts_dir() / f"{short_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"No local prompt file at {path}")
    return path.read_text(encoding="utf-8").strip()


def _prompt_uri(registry_name: str, version: str) -> str:
    normalized = str(version).strip()
    if not normalized or normalized == "latest":
        return f"prompts:/{registry_name}@latest"
    if normalized.startswith("@"):
        return f"prompts:/{registry_name}{normalized}"
    if normalized.isdigit():
        return f"prompts:/{registry_name}/{normalized}"
    return f"prompts:/{registry_name}@{normalized}"


def load_system_prompt_result(short_name: str, *, version: str = "latest") -> PromptLoadResult:
    """Load a system prompt and retain registry lineage metadata.

    Tries the MLflow Prompt Registry first (``prompts:/texprompter.<name>/<version>``).
    Falls back to ``prompts/<short_name>.txt`` if MLflow isn't installed, the registry
    isn't reachable, or the prompt hasn't been registered yet.

    Raises
    ------
    PromptLoadError
        If neither the registry nor the local fallback file can provide a non-empty
        template.  Sending a blank prompt to the LLM would produce garbled output.
    """
    registry_name = _registry_name(short_name)
    requested_uri = _prompt_uri(registry_name, version)
    fallback_reason: str | None = None
    try:
        import mlflow

        prompt = mlflow.genai.load_prompt(requested_uri)
        template = getattr(prompt, "template", None)
        if isinstance(template, str) and template.strip():
            prompt_version = getattr(prompt, "version", None)
            resolved_uri = f"prompts:/{registry_name}/{prompt_version}" if prompt_version else requested_uri
            return PromptLoadResult(
                short_name=short_name,
                registry_name=registry_name,
                requested_uri=requested_uri,
                resolved_uri=resolved_uri,
                version=str(prompt_version) if prompt_version is not None else None,
                template=template,
                source="registry",
            )
        fallback_reason = "registry prompt template was empty"
    except Exception as exc:
        fallback_reason = f"{type(exc).__name__}: {exc}"
        if os.getenv("MLFLOW_PROMPT_REGISTRY_REQUIRED", "").lower() in {"1", "true", "yes"}:
            raise

    if fallback_reason:
        warnings.warn(
            f"Falling back to local prompt file for {short_name}: {fallback_reason}",
            RuntimeWarning,
            stacklevel=2,
        )

    try:
        local_template = _load_local(short_name)
    except FileNotFoundError as local_exc:
        raise PromptLoadError(
            f"Cannot load prompt '{short_name}': registry failed ({fallback_reason}) "
            f"and local file not found ({local_exc}). "
            "Ensure prompts/<name>.txt exists or the MLflow registry is reachable."
        ) from local_exc

    return PromptLoadResult(
        short_name=short_name,
        registry_name=registry_name,
        requested_uri=requested_uri,
        resolved_uri=None,
        version=None,
        template=local_template,
        source="local_file",
        fallback_reason=fallback_reason,
    )


def load_system_prompt(short_name: str, *, version: str = "latest") -> str:
    """Load only the system prompt text for legacy callers."""
    return load_system_prompt_result(short_name, version=version).template
