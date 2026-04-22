from jinja2.utils import missing
from pydantic._internal._known_annotated_metadata import constraint_schema_pairings
import os
import warnings
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool

from typing import List, Any
from pydantic import BaseModel, Field


from langchain_core.messages import ToolMessage
import json

from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

ollama_base_url = os.getenv("OLLAMA_BASE_URL")
ollama_api_key = os.getenv("OLLAMA_API_KEY")
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.3:70b-instruct-q3_K_M")

headers = {}
if ollama_api_key and ollama_api_key != "your_api_key_here":
    headers["Authorization"] = f"Bearer {ollama_api_key}"

print(f"Connecting to Ollama at {ollama_base_url}...")

# Read the CSV path from pipeline-provided env so this subprocess uses the same dataset as pipeline state.
csv_path = os.getenv("PIPELINE_CSV_PATH", r"../data/optimization_pipeline_easy.csv")

# -----------------------------
# TOOLS
# -----------------------------
def _resolve_project_paths() -> tuple[Path, Path]:
    """Return absolute paths for the data and output folders."""
    base_dir = Path(__file__).resolve().parents[1]
    return base_dir / "data", base_dir / "TestOutputs"


def _load_json_file(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def _load_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")

# @tool
# def get_column_names(path: str) -> str:
#     """
#     Returns the column names of a CSV file.

#     Args:
#         path (str): Relative or absolute path to the CSV file.

#     Returns:
#         str: A string representation of the list of column names.

#     Purpose:
#         This tool allows the agent to inspect the available data fields
#         in the dataset and decide which columns can be used to construct
#         sets and parameters for the optimization model.
#     """
#     return str(pd.read_csv(path, nrows=0).columns.tolist())
@tool
def get_column_names() -> str:
    """
    Returns the column names of a CSV file.

    Returns:
        str: A string representation of the list of column names.

    Purpose:
        This tool allows the agent to inspect the available data fields
        in the dataset and decide which columns can be used to construct
        sets and parameters for the optimization model.
    """
    return str(pd.read_csv(csv_path, nrows=0).columns.tolist())

# @tool
# def preview_csv(path: str) -> str:
#     """
#     Returns a preview (first few rows) of a CSV file.

#     Args:
#         path (str): Relative or absolute path to the CSV file.

#     Returns:
#         str: A string representation of the first rows of the dataset.

#     Purpose:
#         Use this tool to understand the structure and content of the dataset,
#         including how columns relate to each other and what kind of values they contain.

#     When to use:
#         - Before creating sets and parameters
#         - When you need to understand relationships between columns
#         - When column names alone are not sufficient to infer data meaning
#     """
#     return pd.read_csv(path).head().to_string()
@tool
def preview_csv() -> str:
    """
    Returns a preview (first few rows) of a CSV file.

    Returns:
        str: A string representation of the first rows of the dataset.

    Purpose:
        Use this tool to understand the structure and content of the dataset,
        including how columns relate to each other and what kind of values they contain.

    When to use:
        - Before creating sets and parameters
        - When you need to understand relationships between columns
        - When column names alone are not sufficient to infer data meaning
    """
    return pd.read_csv(csv_path).head().to_string()

@tool
def get_mathematical_model() -> dict[str, Any]:
    """Return the mathematical model payload from generated outputs and reference model."""
    data_dir, outputs_dir = _resolve_project_paths()

    objective_path = outputs_dir / "llm_objective_function.md"
    constraints_path = outputs_dir / "llm_constraints.md"
    documentation_path = outputs_dir / "llm_output.md"

    constraints_raw = _load_text_file(constraints_path)
    constraints_lines = [line.strip() for line in constraints_raw.splitlines() if line.strip()]

    payload: dict[str, Any] = {
        "mathematical_model": {
            "objective_function": _load_text_file(objective_path).strip(),
            "constraint_functions": constraints_lines,
            "readable_documentation": _load_text_file(documentation_path).strip(),
        }
    }

    print(f"Get Mathematical Model Tool called. Payload: {payload}")
    return payload

# -----------------------------
# OUTPUT FORMAT
# -----------------------------

class SetDefinition(BaseModel):
    name: str = Field(description="Name of the set (e.g., I, T)")
    description: str = Field(description="What the set represents")
    source_column: str = Field(description="CSV column used to derive the set")
    python_representation: str = Field(
        description="Python code snippet defining the set"
    )

class ParameterDefinition(BaseModel):
    symbol: str = Field(description="Mathematical symbol (e.g., c_i)")
    description: str = Field(description="Meaning of the parameter")
    source_columns: List[str] = Field(
        description="CSV columns used to compute this parameter"
    )
    python_representation: str = Field(
        description="Python dictionary or structure"
    )

class DataPreparation(BaseModel):
    imports: str = Field(description="All required imports")

    data_loading: str = Field(
        description="Code to load CSV into pandas DataFrame"
    )

    preprocessing_steps: List[str] = Field(
        description="Step-by-step explanation of transformations applied to the data"
    )

    sets: List[SetDefinition] = Field(
        description="All index sets used in the model"
    )

    parameters: List[ParameterDefinition] = Field(
        description="All parameters derived from CSV data"
    )

    data_structures_ready: bool = Field(
        description="Final combined Python structures ready for optimization"
    )

    mapping_explanation: List[str] = Field(
        description="Explanation of how CSV columns map to mathematical symbols"
    )

    assumptions: List[str] = Field(
        description="Any assumptions made during data preparation"
    )

    full_script: str = Field(
        description="Complete Python script that prepares the data"
    )


# AGENT LOGIC
try:
    llm = ChatOllama(
        base_url=ollama_base_url,
        model=ollama_model,
        client_kwargs={"headers": headers, "verify": False} if headers else {"verify": False},
        reasoning=True,
    )

    tools = [get_column_names, preview_csv, get_mathematical_model]


    system_message = (
        "You are an expert in preparing data for optimization models.\n\n"
        "Transform CSV data into sets and parameter dictionaries.\n\n"
        "DO NOT build any optimization model.\n"
        "DO NOT use PuLP.\n"
        "ONLY prepare data structures.\n\n"
        "Return ONLY using DataPreparation format."
        
        "You must clearly document:\n"
        "- What each set represents\n"
        "- What each parameter represents\n"
        "- Which CSV columns are used\n"
        "- How everything maps to the mathematical model\n"
    )

    # Use LangChain's create_agent function as recommended, and enforce structured output
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_message,
        response_format=DataPreparation
    )

    user_input = (
        "Generate the Python data preparation code for the optimization problem.\n\n"

        "The mathematical model is provided in the get_mathematical_model tool, "
        "which you are REQUIRED to call first.\n\n"

        "The CSV structure and input data information are provided in the get_input_schema_payload tool, "
        "which you are REQUIRED to call second.\n\n"

        "Follow the required workflow and correctness rules strictly.\n\n"

        "Your task:\n"
        "- Transform the CSV data into sets and parameter dictionaries\n"
        "- Map all mathematical symbols to Python data structures\n"
        "- Ensure consistency between indices, parameters, and data\n\n"

        "IMPORTANT:\n"
        "- DO NOT build a PuLP model\n"
        "- DO NOT define constraints in code\n"
        "- DO NOT solve anything\n\n"

        "Return the result ONLY by calling the DataPreparation tool.\n"
        "This is ABSOLUTELY REQUIRED.\n\n"

        "The output must be a JSON with the fields:\n"
        "imports, data_loading, preprocessing_steps, sets, parameters, "
        "data_structures_ready, mapping_explanation, assumptions, full_script"
        
        "WORKFLOW (MANDATORY):\n"
        "1. Call get_mathematical_model\n"
        "2. Call preview_csv\n"
        "3. Build sets\n"
        "4. Build parameters\n"
        "5. Return DataPreparation\n\n"

        "If you do not call both required tools, your answer is invalid."
    )
    print(f"Sending prompt: {user_input}")
    
    messages = [{"role": "user", "content": user_input}]
    max_retries = 3
    
    for attempt in range(max_retries):
        response = agent.invoke({
            "messages": [{"role": "user", "content": user_input}]
        })

        print(response)
        
        print(f"\n--- Message Trace (Attempt {attempt + 1}) ---")
        for msg in response['messages'][-4:]: # print the last few messages for sanity
            print(f"[{msg.type.upper()}]: {msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"  Tool Calls: {msg.tool_calls}")
        print("---------------------\n")
        
        # When create_agent succeeds at extracting the structured output, it either terminates
        # with the Tool message containing the stringified result, or the AI message returning the tool call args.
        # Let's inspect the AI message right before the hypothetical tool response or plain content.
        ai_messages = [m for m in response['messages'] if m.type == 'ai']
        final_ai_msg = ai_messages[-1] if ai_messages else None
        
        success = False
        
        # Check if the final AI message correctly made the structured output tool call
        if final_ai_msg and hasattr(final_ai_msg, "tool_calls") and final_ai_msg.tool_calls:
            for tc in final_ai_msg.tool_calls:
                if tc['name'] == 'DataPreparation':
                    print("\nSuccess! Parsed Extracted Arguments:")
                    args = tc['args']
                    
                    try:
                        with open("../TestOutputs/data_preparation.json", "w", encoding="utf-8") as f:
                            json.dump(args, f, indent=4)
                    except:
                        with open("../TestOutputs/data_preparation.json", "w", encoding="utf-8") as f:
                            json.dump(args, f, indent=4, default=str)

                    success = True
                    break
        
        if success:
            break
        else:
            print(f"\nAttempt {attempt + 1} did not return the correct tool call. Retrying...")
            # Fixed retry contract text to match DataPreparation so the model gets a consistent schema signal.
            messages = response['messages'] + [
                {"role": "user", "content": "You failed to output using the required DataPreparation format. Please output ONLY by calling the 'DataPreparation' tool with all required fields: imports, data_loading, preprocessing_steps, sets, parameters, data_structures_ready, mapping_explanation, assumptions, full_script."}
            ]
    else:
        print("\nFailed to get a valid structured response after maximum retries.")

except Exception as e:
    print(f"\nFailed to connect or generate response. Error: {e}")