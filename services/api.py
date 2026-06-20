from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.pipeline_service import PipelineService

app = FastAPI(title="TexPrompter Pipeline API")
service = PipelineService()


class PipelineStartRequest(BaseModel):
    csv_content: str
    initial_prompt: str = ""
    preview_rows: int = 5


class DownstreamRequest(BaseModel):
    csv_content: str
    use_case: Dict[str, Any]
    modelling: Dict[str, Any]
    input_schema_payload: Dict[str, Any]
    preview_rows: int = 5


class FeedbackRequest(BaseModel):
    csv_content: str
    use_case: Dict[str, Any]
    input_schema_payload: Dict[str, Any]
    feedback_text: str
    preview_rows: int = 5


class SaveResultsRequest(BaseModel):
    final_state: Dict[str, Any]
    csv_filename: str
    initial_prompt: str = ""


def _write_csv_to_temp(csv_content: str, csv_filename: str | None = None) -> str:
    suffix = ".csv"
    if csv_filename:
        suffix = Path(csv_filename).suffix or suffix
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=suffix, encoding="utf-8") as tmp:
        tmp.write(csv_content)
        return tmp.name


@app.post("/pipeline/start")
def start_pipeline(request: PipelineStartRequest):
    csv_path = _write_csv_to_temp(request.csv_content)

    def event_stream():
        try:
            for update in service.start_pipeline(
                csv_file_path=csv_path,
                initial_prompt=request.initial_prompt,
                preview_rows=request.preview_rows,
            ):
                yield json.dumps(update, default=str) + "\n"
        except Exception as exc:
            yield json.dumps({"error": str(exc)}) + "\n"
        finally:
            Path(csv_path).unlink(missing_ok=True)

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/pipeline/downstream")
def run_downstream(request: DownstreamRequest) -> Dict[str, Any]:
    csv_path = _write_csv_to_temp(request.csv_content)
    try:
        return service.run_downstream(
            csv_file_path=csv_path,
            use_case=request.use_case,
            modelling=request.modelling,
            input_schema_payload=request.input_schema_payload,
            preview_rows=request.preview_rows,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(csv_path).unlink(missing_ok=True)


class ModelingRequest(BaseModel):
    csv_content: str
    use_case: Dict[str, Any]
    preview_rows: int = 5


class ParameterEstimationRequest(BaseModel):
    csv_content: str
    use_case: Dict[str, Any]
    modelling: Dict[str, Any]
    preview_rows: int = 5


class PreprocessingRequest(BaseModel):
    csv_content: str
    use_case: Dict[str, Any]
    modelling: Dict[str, Any]
    input_schema_payload: Dict[str, Any]
    preview_rows: int = 5


class ScriptingRequest(BaseModel):
    csv_content: str
    modelling: Dict[str, Any]
    preprocessing: Dict[str, Any]
    input_schema_payload: Dict[str, Any]
    preview_rows: int = 5


class ResultsInterpretationRequest(BaseModel):
    use_case: Dict[str, Any]
    modelling: Dict[str, Any]
    scripting: Dict[str, Any]


@app.post("/pipeline/rerun-modeling")
def rerun_modeling(request: ModelingRequest) -> Dict[str, Any]:
    csv_path = _write_csv_to_temp(request.csv_content)
    try:
        return service.rerun_modeling(
            csv_file_path=csv_path,
            use_case=request.use_case,
            preview_rows=request.preview_rows,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(csv_path).unlink(missing_ok=True)


@app.post("/pipeline/estimate-parameters")
def estimate_parameters(request: ParameterEstimationRequest) -> Dict[str, Any]:
    csv_path = _write_csv_to_temp(request.csv_content)
    try:
        return service.estimate_parameters(
            csv_file_path=csv_path,
            use_case=request.use_case,
            modelling=request.modelling,
            preview_rows=request.preview_rows,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(csv_path).unlink(missing_ok=True)


@app.post("/pipeline/preprocessing")
def preprocessing(request: PreprocessingRequest) -> Dict[str, Any]:
    csv_path = _write_csv_to_temp(request.csv_content)
    try:
        return service.preprocessing(
            csv_file_path=csv_path,
            use_case=request.use_case,
            modelling=request.modelling,
            input_schema_payload=request.input_schema_payload,
            preview_rows=request.preview_rows,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(csv_path).unlink(missing_ok=True)


@app.post("/pipeline/scripting")
def scripting(request: ScriptingRequest) -> Dict[str, Any]:
    csv_path = _write_csv_to_temp(request.csv_content)
    try:
        return service.scripting(
            csv_file_path=csv_path,
            modelling=request.modelling,
            preprocessing=request.preprocessing,
            input_schema_payload=request.input_schema_payload,
            preview_rows=request.preview_rows,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(csv_path).unlink(missing_ok=True)


@app.post("/pipeline/results-interpretation")
def results_interpretation(request: ResultsInterpretationRequest) -> Dict[str, Any]:
    try:
        return service.results_interpretation(
            use_case=request.use_case,
            modelling=request.modelling,
            scripting=request.scripting,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/pipeline/regenerate-feedback")
def regenerate_feedback(request: FeedbackRequest) -> Dict[str, Any]:
    csv_path = _write_csv_to_temp(request.csv_content)
    try:
        return service.regenerate_from_feedback(
            csv_file_path=csv_path,
            use_case=request.use_case,
            input_schema_payload=request.input_schema_payload,
            feedback=request.feedback_text,
            preview_rows=request.preview_rows,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(csv_path).unlink(missing_ok=True)


@app.post("/pipeline/save")
def save_results(request: SaveResultsRequest) -> Dict[str, str]:
    try:
        saved_path = service.save_results(
            request.final_state,
            request.csv_filename,
            request.initial_prompt,
        )
        return {"saved_path": saved_path}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.api:app", host="127.0.0.1", port=8000, reload=True)
