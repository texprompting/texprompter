import os
import warnings
import numpy as np
import pandas as pd
import json

from dotenv import load_dotenv


warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import ToolMessage
import json

# Load environment variables
load_dotenv()

ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_api_key = os.getenv("OLLAMA_API_KEY")
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.3:70b-instruct-q3_K_M")

headers = {}
if ollama_api_key and ollama_api_key != "your_api_key_here":
    headers["Authorization"] = f"Bearer {ollama_api_key}"

print(f"Connecting to Ollama at {ollama_base_url}...")

csv_path = os.getenv("PIPELINE_CSV_PATH", r"../data/optimization_pipeline_test_easy.csv")

# @tool
# def get_temperature(city: str) -> float:
#     """Returns the current temperature for a given city in Celsius."""
#     print("AGENT USED TEMP TOOL CALL!!!!!!!")
#     return float(np.random.random() * 40 - 10)
@tool
def get_column_names() -> list[:str]:
    """Returns the column names of the available csv"""
    cols = pd.read_csv(csv_path, sep=",",nrows=0).columns.tolist()
    print("cols: " + str(cols))
    return "The available variables to find reasonable constraints for the LP are: \n" + str(cols)

@tool
def get_reference_model() -> list[:str]:
    """"Returns the reference model the Agent should orientate to build the output"""
    path = r"../data/ReferenceMathematicalModel.json"
    data = ""
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
 
@tool
def get_use_case_recommendation() -> str:
    """"Returns the usecase given by the agent above"""
    return "This is the description of the use case and the variable to optimize: The goal is to optimize the quantity to produce for Prduct_ID I"

from pydantic import BaseModel, Field
from typing import List

class Parameter(BaseModel):
    symbol: str = Field(description="The LaTeX symbol, e.g., 'v_t' or 'T'")
    description: str = Field(description="Description of what the parameter represents")

class Variable(BaseModel):
    variable: str = Field(description="The LaTeX decision variable, e.g., 'x_{ts}'")
    meaning: str = Field(description="What the variable represents in the physical system")

class ModellingRecommendation(BaseModel):
    col_names_used: List[str] = Field(
        description="List the column names of the csv that are needed to solve the formulated problem."
    )
    parameters: List[Parameter] = Field(
        description="List of model parameters. Include sets, indices, and constants."
    )
    variables: List[Variable] = Field(
        description="Definition of all used variables (variable, meaning see definition of Variable BaseModel) using LaTeX."
    )
    minimizing_problem: bool = Field(
        description="True if the objective is 'min', False if it is 'max'."
    )
    objective_function: str = Field(
        description="The formal objective function in LaTeX (e.g., '\\min \\sum s \\cdot y_s')."
    )
    constraint_functions: List[str] = Field(
        description="A list of each MILP constraint formula in LaTeX."
    )
    explanation_of_ILP: List[str] = Field(
        description="A numbered list of explanations for the objective and each constraint."
    )
    readable_documentation: str = Field(
        description="A complete Markdown string combining all sections for human reading."
    )

# class ClothingRecommendation(BaseModel):
#     """Call this to give the final clothing recommendation."""
#     temperature_found: float = Field(description="The temperature that was found for the city.")
#     wear_hat: bool = Field(description="Whether a hat is recommended.")
#     wear_scarf: bool = Field(description="Whether a scarf is recommended.")
#     wear_winter_jacket: bool = Field(description="Whether a winter jacket is recommended.")
#     wear_light_jacket: bool = Field(description="Whether a light jacket is recommended.")
#     wear_t_shirt: bool = Field(description="Whether a t-shirt is recommended.")
#     wear_boots: bool = Field(description="Whether boots are recommended.")
#     reasoning: str = Field(description="A short explanation of the recommendation based on the temperature.")

try:
    llm = ChatOllama(
        base_url=ollama_base_url,
        model=ollama_model,
        client_kwargs={"headers": headers, "verify": False} if headers else {"verify": False},
        reasoning=True,
    )

    #tools=[get_temperature]
    tools = [get_column_names, get_reference_model, get_use_case_recommendation]

    # system_message = (
    #     "You are a helpful assistant that recommends what to wear based on the current temperature of a city. "
    #     "Always use the get_temperature tool to find the temperature first. "
    #     "After finding the temperature, you MUST respond by calling the 'ClothingRecommendation' tool. "
    #     "When calling 'ClothingRecommendation', you MUST provide ALL of the following fields in your JSON arguments: "
    #     "temperature_found, wear_hat, wear_scarf, wear_winter_jacket, wear_light_jacket, wear_t_shirt, wear_boots, and reasoning. "
    #     "Do not leave any fields out."
    # )
    system_message = (
        "You are a mathematical expert in optimizing MILP. Your task is to use the provided information about a problem we want to solve with an MILP solver. You are provided with information about which data you have available and the variable that should be optimized.USE THE TOOL CALL get_column_names to check which variables you have available in the csv to find reasonable constraints for the LP."
        "Your task is now to find a mathematical formulation that can represent this problem using Mixed-Integer Linear Programming (MILP) to minimize or maximize the given variable with the provided data. So this can later be used to write a python script to solve the optimization problem using an MILP solver in Python."
        "Your task is to ONLY provide the mathematical formulation."
        "\n\nSTRICT GUIDELINES:\n"
        "1. Use the 'get_reference_model' tool to understand the required notation style (LaTeX, set notation, and indexing).\n"
        "2. Format all math using LaTeX symbols ($...$).\n"
        "3. When finished, you MUST call the 'ModellingRecommendation' tool and provide the mathematical Model in a final structured output."
        "4. It requires those fields: col_names_used, parameters, variables, minimizing_problem, objective_function, constraint_functions, explanation_of_ILP, readable_documentation"
    )

    # Use LangChain's create_agent function as recommended, and enforce structured output
    agent = create_agent(
        model=llm, 
        tools=tools, 
        system_prompt=system_message,
        #response_format=ClothingRecommendation
        response_format=ModellingRecommendation
    )

    user_input = "The goal is to optimize the quantity to produce for each Product_ID i"
    print(f"Sending prompt: {user_input}")
    
    messages = [{"role": "user", "content": user_input}]
    max_retries = 3
    
    for attempt in range(max_retries):
        response = agent.invoke({"messages": messages}) # type: ignore
        
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
                if tc['name'] == 'ModellingRecommendation':
                    print("\nSuccess! Parsed Extracted Arguments:")
                    args = tc['args']

                    print("Raw Output: \n" + str(args))
                    # Print and save each field in a readable way
                    # 1. col_names_used
                    print("\n[CSV Columns Used]")
                    print(args.get('col_names_used', ''))

                    # 2. parameters
                    print("\n[Parameters]")
                    params = args.get('parameters', [])
                    for p in params:
                        print(f"- {p.get('symbol', '')}: {p.get('description', '')}")

                    # 3. variables
                    print("\n[Variables]")
                    vars_ = args.get('variables', [])
                    for v in vars_:
                        print(f"- {v.get('variable', '')}: {v.get('meaning', '')}")

                    # 4. minimizing_problem
                    print("\n[Minimizing Problem]")
                    print("Minimize" if args.get('minimizing_problem', True) else "Maximize")

                    # 5. objective_function (LaTeX)
                    print("\n[Objective Function]")
                    obj = args.get('objective_function', '')
                    print(obj)
                    # Save to file
                    try:
                        with open("../TestOutputs/llm_objective_function.md", "w", encoding="utf-8") as f:
                            f.write(obj)
                    except Exception as file_err:
                        print(f"Failed to write objective function: {file_err}")

                    # 6. constraint_functions (LaTeX)
                    print("\n[Constraint Functions]")
                    constraints = args.get('constraint_functions', [])
                    for i, c in enumerate(constraints, 1):
                        print(f"{i}. {c}")
                    # Save to file
                    try:
                        with open("../TestOutputs/llm_constraints.md", "w", encoding="utf-8") as f:
                            for c in constraints:
                                f.write(c + "\n")
                    except Exception as file_err:
                        print(f"Failed to write constraints: {file_err}")

                    # 7. explanation_of_ILP
                    print("\n[Explanation of ILP]")
                    explanation = args.get('explanation_of_ILP', [])
                    for i, e in enumerate(explanation, 1):
                        print(f"{i}. {e}")

                    # 8. readable_documentation (Markdown/LaTeX)
                    print("\n[Full Readable Documentation]")
                    doc = args.get('readable_documentation', '')
                    if doc:
                        print("(See TestOutputs/llm_output.md for full documentation)")
                        try:
                            with open("../TestOutputs/llm_output.md", "w", encoding="utf-8") as f:
                                f.write(doc)
                        except Exception as file_err:
                            print(f"Failed to write output file: {file_err}")
                    else:
                        print("No readable_documentation found in output.")

                    success = True
                    break
        
        if success:
             break
        else:
             print(f"\nAttempt {attempt + 1} did not return the correct tool call. Retrying...")
             # Let's feed the conversation history back in, appending a strict instruction
             messages = response['messages'] + [
                 {"role": "user", "content": "You failed to output using the required ModellingRecommendation format. Please output ONLY by calling the 'ModellingRecommendation' tool with all the required fields. It requires those exact fields: col_names_used, parameters, variables, minimizing_problem, objective_function, constraint_functions, explanation_of_ILP, readable_documentation"}
             ]
             
    else:
        print("\nFailed to get a valid structured response after maximum retries.")

except Exception as e:
    print(f"\nFailed to connect or generate response. Error: {e}")
