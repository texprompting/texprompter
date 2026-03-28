from langchain_core.runnables import RunnableLambda
from schemas.basemodels import ModelingResult, Parameter, Variable

# 'inputs' is the output of the last step in pipeline
def mock_modeling_agent(inputs) -> ModelingResult:    
    # reasign for readability (and type hint)
    use_case: "UseCaseResult" = inputs

    print(f"[Modeling Agent] Formulating variables from use case: {use_case.variable_to_optimize}...")
    
    return ModelingResult(
        use_case_context=use_case,
        col_names_used=use_case.available_variables,
        parameters=[Parameter(symbol="D_i", description="Demand for product i")],
        variables=[Variable(variable="x_i", meaning="Quantity of product i to produce")],
        minimizing_problem=False,
        objective_function="\\max \\sum_{i} Profit_i \\cdot x_i",
        constraint_functions=["x_i \\le D_i"],
        explanation_of_ILP=["1. Maximize total profit from all products.", "2. Production cannot exceed demand."],
        readable_documentation="## Mathematical Model Output\\n\\nWe want to maximize profit."
    )

modeling_chain = RunnableLambda(mock_modeling_agent)
