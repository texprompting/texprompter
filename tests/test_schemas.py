'''
This test file checks that the pipeline state defaults 
are correct and that the modeling and scripting recommendation 
schemas validate expected contract fields.
'''


from schemas.basemodels import (
    ModellingRecommendation,
    PipelineState,
    ScriptingRecommendation,
)


def test_pipeline_state_defaults() -> None:
    state = PipelineState(csv_file_path="optimization_pipeline_test_easy.csv")

    assert state.status == "ok"
    assert state.preview_rows == 5
    assert state.errors == []
    assert state.traces == []


def test_modeling_and_scripting_contracts() -> None:
    modeling = ModellingRecommendation(
        col_names_used=["Product_ID", "Profit_Per_Unit"],
        parameters=[],
        variables=[],
        minimizing_problem=False,
        objective_function="max \\sum_i P_i x_i",
        constraint_functions=["x_i <= D_i"],
        explanation_of_ILP=["Objective maximizes profit."],
        readable_documentation="# Model",
    )

    scripting = ScriptingRecommendation(
        code="print('ok')",
        output_schema={"objective_value": "float"},
        successful_implementation=True,
        missing_info=[],
        additional_info=[],
    )

    assert modeling.objective_function.startswith("max")
    assert scripting.successful_implementation is True
