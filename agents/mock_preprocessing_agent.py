from langchain_core.runnables import RunnableLambda
from schemas.basemodels import PreprocessingResult

def mock_preprocessing_agent(inputs) -> PreprocessingResult:
    modeling_result: "ModelingResult" = inputs
    print(f"[Preprocessing Agent] Analyzing variables: {[v.variable for v in modeling_result.variables]}...")
    
    code = f"""
import pandas as pd
df = pd.read_csv('{modeling_result.use_case_context.data_path}')
data_dict = df.to_dict()
# This code converts raw input for {modeling_result.use_case_context.use_case_description} into a PuLP schema.
"""
    
    return PreprocessingResult(
        modeling_context=modeling_result,
        preprocessing_script_code=code,
        input_scheme_description="A dictionary mapping string IDs (i) to their demand limits and unit profits."
    )

preprocessing_chain = RunnableLambda(mock_preprocessing_agent)