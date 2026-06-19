from __future__ import annotations

import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from orchestrator.pipeline import (
    stream_pipeline,
    run_modeling_agent,
    run_parameter_estimation_agent,
    run_preprocessing_agent,
    run_scripting_agent,
)


class PipelineService:
    """
    Single entry point for all pipeline interactions.

    The frontend should ONLY communicate with this class.
    """

    def __init__(self):
        self.basemodels = importlib.import_module("schemas.basemodels")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clean(self, obj: Any):

        if obj is None:
            return None

        if isinstance(obj, (str, int, float, bool)):
            return obj

        if hasattr(obj, "model_dump"):
            return self._clean(obj.model_dump())

        if isinstance(obj, dict):
            return {k: self._clean(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._clean(v) for v in obj]

        if isinstance(obj, tuple):
            return tuple(self._clean(v) for v in obj)

        if hasattr(obj, "__dict__"):
            return {
                k: self._clean(v)
                for k, v in obj.__dict__.items()
                if not k.startswith("_")
            }

        return str(obj)

    # ------------------------------------------------------------------
    # Model Conversion
    # ------------------------------------------------------------------

    def use_case_model(self, data):

        if not data:
            return None

        return self.basemodels.UseCaseRecommendation.model_validate(data)

    def modelling_model(self, data):

        if not data:
            return None

        return self.basemodels.ModellingRecommendation.model_validate(data)

    def preprocessing_model(self, data):

        if not data:
            return None

        return self.basemodels.PreprocessingRecommendation.model_validate(data)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def start_pipeline(
        self,
        csv_file_path: str,
        initial_prompt: str = "",
        preview_rows: int = 5,
    ) -> Generator[Dict[str, Any], None, None]:

        for update in stream_pipeline(
            csv_file_path=csv_file_path,
            preview_rows=preview_rows,
            initial_prompt=initial_prompt,
        ):
            yield self._clean(update)

    # ------------------------------------------------------------------
    # Modeling
    # ------------------------------------------------------------------

    def rerun_modeling(
        self,
        csv_file_path: str,
        use_case: Dict,
        preview_rows: int = 5,
    ):

        obj = self.use_case_model(use_case)

        result = run_modeling_agent(
            csv_file_path=csv_file_path,
            use_case=obj,
            preview_rows=preview_rows,
        )

        return self._clean(result)

    # ------------------------------------------------------------------
    # Parameter Estimation
    # ------------------------------------------------------------------

    def estimate_parameters(
        self,
        csv_file_path: str,
        use_case: Dict,
        modelling: Dict,
        preview_rows: int = 5,
    ):

        uc = self.use_case_model(use_case)
        md = self.modelling_model(modelling)

        result = run_parameter_estimation_agent(
            csv_file_path=csv_file_path,
            use_case=uc,
            modelling=md,
            preview_rows=preview_rows,
        )

        return self._clean(result)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocessing(
        self,
        csv_file_path: str,
        use_case: Dict,
        modelling: Dict,
        input_schema_payload: Dict,
        preview_rows: int = 5,
    ):

        uc = self.use_case_model(use_case)
        md = self.modelling_model(modelling)

        result = run_preprocessing_agent(
            csv_file_path=csv_file_path,
            use_case=uc,
            modelling=md,
            input_schema_payload=input_schema_payload,
            preview_rows=preview_rows,
        )

        return self._clean(result)

    # ------------------------------------------------------------------
    # Scripting
    # ------------------------------------------------------------------

    def scripting(
        self,
        csv_file_path: str,
        modelling: Dict,
        preprocessing: Dict,
        input_schema_payload: Dict,
        preview_rows: int = 5,
    ):

        md = self.modelling_model(modelling)
        prep = self.preprocessing_model(preprocessing)

        result = run_scripting_agent(
            csv_file_path=csv_file_path,
            modelling=md,
            preprocessing=prep,
            input_schema_payload=input_schema_payload,
            preview_rows=preview_rows,
        )

        return self._clean(result)

    # ------------------------------------------------------------------
    # Complete downstream execution
    # ------------------------------------------------------------------

    def run_downstream(
        self,
        csv_file_path: str,
        use_case: Dict,
        modelling: Dict,
        input_schema_payload: Dict,
        preview_rows: int = 5,
    ):

        preprocessing = self.preprocessing(
            csv_file_path,
            use_case,
            modelling,
            input_schema_payload,
            preview_rows,
        )

        scripting = self.scripting(
            csv_file_path,
            modelling,
            preprocessing,
            input_schema_payload,
            preview_rows,
        )

        return {
            "preprocessing": preprocessing,
            "scripting": scripting,
        }

    def save_results(
        self,
        final_state: Dict[str, Any],
        csv_filename: str,
        initial_prompt: str,
    ) -> str:
        output_dir = Path(__file__).parent.parent / "TestOutputs"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_dir / f"result_{timestamp}_{csv_filename.replace('.csv', '')}.json"

        results = {
            "metadata": {
                "timestamp": timestamp,
                "csv_filename": csv_filename,
                "initial_prompt": initial_prompt,
            },
            "pipeline_state": final_state,
        }

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        return str(result_file)

    # ------------------------------------------------------------------
    # Feedback Loop
    # ------------------------------------------------------------------

    def regenerate_from_feedback(
        self,
        csv_file_path: str,
        use_case: Dict,
        input_schema_payload: Dict,
        feedback: str,
        preview_rows: int = 5,
    ):

        obj_uc = self.use_case_model(use_case)

        assumptions = list(obj_uc.assumptions or [])
        assumptions.append(f"User feedback: {feedback}")

        obj_uc = obj_uc.model_copy(
            update={
                "assumptions": assumptions
            }
        )

        modeling = run_modeling_agent(
            csv_file_path=csv_file_path,
            use_case=obj_uc,
            preview_rows=preview_rows,
        )

        modeling = self._clean(modeling)

        parameter = self.estimate_parameters(
            csv_file_path,
            self._clean(obj_uc),
            modeling,
            preview_rows,
        )

        modeling_obj = self.modelling_model(modeling)

        modeling_obj = modeling_obj.model_copy(
            update={
                "constraint_functions": parameter["updated_constraint_functions"],
                "objective_function": parameter["updated_objective_function"],
            }
        )

        modeling = self._clean(modeling_obj)

        downstream = self.run_downstream(
            csv_file_path,
            self._clean(obj_uc),
            modeling,
            input_schema_payload,
            preview_rows,
        )

        return {
            "use_case": self._clean(obj_uc),
            "modelling": modeling,
            "parameter_estimation": parameter,
            **downstream,
        }