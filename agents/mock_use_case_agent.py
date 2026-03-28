from langchain_core.runnables import RunnableLambda
from schemas.basemodels import UseCaseResult

def mock_use_case_agent(inputs: dict) -> UseCaseResult:
    """Reads a CSV path, outputs a UseCaseResult."""
    # safe dict get
    data_path = inputs.get("data_path", "data/mock_data.csv")
    print(f"[Use Case Agent] Analyzing {data_path}...")
    
    # mock return
    return UseCaseResult(
        use_case_description="Optimize the quantity to produce for Product_ID I.",
        variable_to_optimize="Quantity",
        available_variables=["Product_ID", "Quantity", "Demand", "Cost"],
        data_path=data_path
    )

# wrap the mock in a Runnable so it can be used in the pipeline
# would not be necessary when creating agent with langchains create_agent()
use_case_chain = RunnableLambda(mock_use_case_agent)
