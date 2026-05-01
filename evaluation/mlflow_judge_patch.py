"""Work around MLflow genai builtin judges ignoring ``OPENAI_URL`` / wrong POST URL.

1. ``meets_guidelines`` calls ``invoke_judge_model`` without ``base_url``.
2. For ``openai:/…`` URIs, **GatewayAdapter** is usually selected (often before LiteLLM is even
   installed). ``GatewayAdapter`` sets ``endpoint = base_url`` and POSTs to that URL verbatim
   (see ``gateway_adapter._invoke_and_handle_tools``). The OpenAI-compat *base* for LangChain/
   SDK is ``http://host/v1``, but POST must be ``http://host/v1/chat/completions`` — bare
   ``/v1`` returns 404 from Ollama.
3. We inject ``base_url``, then normalize ``…/v1`` → ``…/v1/chat/completions`` for the Gateway
   path while keeping ``resolve_*`` aligned with LangChain (`…/v1`).

Call :func:`apply_invoke_judge_model_openai_base_url_patch` from ``evaluation.run_eval``
before ``mlflow.genai.evaluate``.
"""

from __future__ import annotations

import functools
import os
from typing import Any

_PATCH_ATTR = "_texprompter_invoke_judge_patched"


def resolve_litellm_judge_api_base_url() -> str | None:
    """Return OpenAI-compatible *service base* ending in ``/v1`` (LangChain convention)."""
    from agents.shared import (
        canonical_openai_compat_base_url,
        ensure_mlflow_openai_base_url_for_ollama_judge,
        ollama_openai_compatible_base_url,
    )

    ensure_mlflow_openai_base_url_for_ollama_judge()

    current = os.getenv("OPENAI_BASE_URL", "").strip()
    api_key_low = os.getenv("OPENAI_API_KEY", "").strip().lower()
    has_ollama = bool(os.getenv("OLLAMA_BASE_URL", "").strip())

    def canonicalized(url: str) -> str:
        u = url.strip()
        if u.endswith("/v1"):
            return u
        return canonical_openai_compat_base_url(u)

    if current:
        is_openai_cloud = "api.openai.com" in current.lower()
        if not is_openai_cloud:
            return canonicalized(current)
        placeholder_key = api_key_low in {"ollama", "dummy"}
        if placeholder_key or has_ollama:
            return ollama_openai_compatible_base_url()
        return canonicalized(current)

    if has_ollama or api_key_low == "ollama":
        return ollama_openai_compatible_base_url()

    return None


def judge_gateway_http_chat_endpoint(openai_style_base_or_full: str) -> str:
    """Turn ``…/v1`` into POST URL ``…/v1/chat/completions`` for MLflow Gateway judges."""
    if not openai_style_base_or_full.strip():
        return openai_style_base_or_full
    u = openai_style_base_or_full.rstrip("/")
    if u.endswith("/chat/completions"):
        return u
    if u.endswith("/v1"):
        return f"{u}/chat/completions"
    return openai_style_base_or_full


def _merged_litellm_judge_inference_params(existing: dict[str, Any] | None) -> dict[str, Any]:
    """Merge timeout seconds for litellm.completion via MLflow inference_params when absent."""
    out = dict(existing) if isinstance(existing, dict) else {}

    if out.get("timeout") is not None or out.get("request_timeout"):
        return out

    judge_raw = os.getenv("MLFLOW_GENAI_JUDGE_TIMEOUT_S", "").strip()
    fallback_raw = os.getenv("OLLAMA_REQUEST_TIMEOUT_S", "").strip()
    if judge_raw:
        timeout_sec = float(judge_raw)
    elif fallback_raw:
        timeout_sec = float(fallback_raw)
    else:
        timeout_sec = 600.0

    out.setdefault("timeout", timeout_sec)
    return out


def apply_invoke_judge_model_openai_base_url_patch() -> None:
    from mlflow.genai.judges import builtin as judge_builtin
    from mlflow.genai.judges.utils import invocation_utils

    original_fn = invocation_utils.invoke_judge_model

    if getattr(original_fn, _PATCH_ATTR, False):
        return

    @functools.wraps(original_fn)
    def patched_invoke_judge_model(*args: Any, **kwargs: Any):
        merged = dict(kwargs)
        if merged.get("base_url") is None:
            resolved = resolve_litellm_judge_api_base_url()
            if resolved is not None:
                merged["base_url"] = judge_gateway_http_chat_endpoint(resolved)

        inference = merged.get("inference_params")
        merged["inference_params"] = _merged_litellm_judge_inference_params(inference)

        return original_fn(*args, **merged)

    setattr(patched_invoke_judge_model, _PATCH_ATTR, True)

    invocation_utils.invoke_judge_model = patched_invoke_judge_model
    judge_builtin.invoke_judge_model = patched_invoke_judge_model

    from mlflow.genai.judges import utils as mlflow_genai_judge_utils

    mlflow_genai_judge_utils.invoke_judge_model = patched_invoke_judge_model

    try:
        from mlflow.genai.scorers import builtin_scorers as mlflow_builtin_scorers

        mlflow_builtin_scorers.invoke_judge_model = patched_invoke_judge_model
    except ImportError:
        pass
