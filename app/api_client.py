from __future__ import annotations

import json
import os
from typing import Any, Dict

import requests

DEFAULT_API_URL = os.getenv("PIPELINE_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_TIMEOUT_S = int(os.getenv("PIPELINE_API_TIMEOUT", "180"))


def _service_url(path: str) -> str:
    return f"{DEFAULT_API_URL}/{path.lstrip('/') }"


def _post(path: str, payload: Dict[str, Any]) -> Any:
    url = _service_url(path)
    response = requests.post(url, json=payload, timeout=API_TIMEOUT_S)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = None
        try:
            error_json = response.json()
            if isinstance(error_json, dict):
                detail = error_json.get("detail")
        except ValueError:
            detail = response.text
        raise RuntimeError(
            f"Pipeline API request failed ({response.status_code}) for {url}: {detail or response.text}"
        ) from exc
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"Invalid JSON response from Pipeline API at {url}") from exc


def health_check() -> bool:
    url = _service_url("/health")
    try:
        response = requests.get(url, timeout=API_TIMEOUT_S)
        response.raise_for_status()
        data = response.json()
        return data.get("status") == "ok"
    except Exception:
        return False


def stream_start_pipeline(csv_content: str, initial_prompt: str = "", preview_rows: int = 5):
    url = _service_url("/pipeline/start")
    payload = {
        "csv_content": csv_content,
        "initial_prompt": initial_prompt,
        "preview_rows": preview_rows,
    }
    with requests.post(url, json=payload, timeout=API_TIMEOUT_S, stream=True) as response:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = None
            try:
                error_json = response.json()
                if isinstance(error_json, dict):
                    detail = error_json.get("detail")
            except ValueError:
                detail = response.text
            raise RuntimeError(
                f"Pipeline API request failed ({response.status_code}) for {url}: {detail or response.text}"
            ) from exc

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Unable to parse pipeline stream line: {line}") from exc
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(payload["error"])
            yield payload


def start_pipeline(csv_content: str, initial_prompt: str = "", preview_rows: int = 5) -> list[Dict[str, Any]]:
    response = _post(
        "/pipeline/start",
        {
            "csv_content": csv_content,
            "initial_prompt": initial_prompt,
            "preview_rows": preview_rows,
        },
    )
    return response.get("updates", [])


def run_downstream(
    csv_content: str,
    use_case: Dict[str, Any],
    modelling: Dict[str, Any],
    input_schema_payload: Dict[str, Any],
    preview_rows: int = 5,
) -> Dict[str, Any]:
    return _post(
        "/pipeline/downstream",
        {
            "csv_content": csv_content,
            "use_case": use_case,
            "modelling": modelling,
            "input_schema_payload": input_schema_payload,
            "preview_rows": preview_rows,
        },
    )


def rerun_modeling(csv_content: str, use_case: Dict[str, Any], preview_rows: int = 5) -> Dict[str, Any]:
    return _post(
        "/pipeline/rerun-modeling",
        {
            "csv_content": csv_content,
            "use_case": use_case,
            "preview_rows": preview_rows,
        },
    )


def estimate_parameters(
    csv_content: str,
    use_case: Dict[str, Any],
    modelling: Dict[str, Any],
    preview_rows: int = 5,
) -> Dict[str, Any]:
    return _post(
        "/pipeline/estimate-parameters",
        {
            "csv_content": csv_content,
            "use_case": use_case,
            "modelling": modelling,
            "preview_rows": preview_rows,
        },
    )


def preprocessing(
    csv_content: str,
    use_case: Dict[str, Any],
    modelling: Dict[str, Any],
    input_schema_payload: Dict[str, Any],
    preview_rows: int = 5,
) -> Dict[str, Any]:
    return _post(
        "/pipeline/preprocessing",
        {
            "csv_content": csv_content,
            "use_case": use_case,
            "modelling": modelling,
            "input_schema_payload": input_schema_payload,
            "preview_rows": preview_rows,
        },
    )


def scripting(
    csv_content: str,
    modelling: Dict[str, Any],
    preprocessing: Dict[str, Any],
    input_schema_payload: Dict[str, Any],
    preview_rows: int = 5,
) -> Dict[str, Any]:
    return _post(
        "/pipeline/scripting",
        {
            "csv_content": csv_content,
            "modelling": modelling,
            "preprocessing": preprocessing,
            "input_schema_payload": input_schema_payload,
            "preview_rows": preview_rows,
        },
    )


def results_interpretation(
    use_case: Dict[str, Any],
    modelling: Dict[str, Any],
    scripting: Dict[str, Any],
) -> Dict[str, Any]:
    return _post(
        "/pipeline/results-interpretation",
        {
            "use_case": use_case,
            "modelling": modelling,
            "scripting": scripting,
        },
    )


def regenerate_feedback(
    csv_content: str,
    use_case: Dict[str, Any],
    input_schema_payload: Dict[str, Any],
    feedback_text: str,
    preview_rows: int = 5,
) -> Dict[str, Any]:
    return _post(
        "/pipeline/regenerate-feedback",
        {
            "csv_content": csv_content,
            "use_case": use_case,
            "input_schema_payload": input_schema_payload,
            "feedback_text": feedback_text,
            "preview_rows": preview_rows,
        },
    )


def save_results(final_state: Dict[str, Any], csv_filename: str, initial_prompt: str = "") -> Dict[str, str]:
    return _post(
        "/pipeline/save",
        {
            "final_state": final_state,
            "csv_filename": csv_filename,
            "initial_prompt": initial_prompt,
        },
    )
