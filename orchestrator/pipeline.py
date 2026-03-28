import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

from agents.mock_use_case_agent import use_case_chain
from agents.mock_modeling_agent import modeling_chain
from agents.mock_preprocessing_agent import preprocessing_chain
from agents.mock_scripting_agent import scripting_chain


def run_pipeline(csv_path: str):
    # create the Langchain Pipe
    pipeline = use_case_chain | modeling_chain | preprocessing_chain | scripting_chain

    # invoke / start the chain
    # the output from each agent will be passed to the input of the next one in the chain
    final_output = pipeline.invoke({"data_path": csv_path})
    
    return final_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the optimization pipeline.")
    parser.add_argument("--csv", type=str, default="data/optimization_pipeline_test_easy.csv", help="Path to the source CSV file")
    args = parser.parse_args()

    result = run_pipeline(args.csv)
    
    print(f"\n\nInput Schema:\n{result.preprocessing_context.input_scheme_description}\n")
    print(f"Mathematical Objective:\n{result.preprocessing_context.modeling_context.objective_function}\n")
    print(f"Generated PULP Code:\n{result.final_solution_script_code}")
