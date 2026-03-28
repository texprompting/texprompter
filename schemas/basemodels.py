from pydantic import BaseModel, Field
from typing import List, Optional

class UseCaseResult(BaseModel):
    """Output from the Use Case Agent."""
    use_case_description: str = Field(description="A description of the overarching goal or optimization use case.")
    variable_to_optimize: str = Field(description="The primary variable or KPI that needs to be optimized.")
    available_variables: List[str] = Field(description="A list of relevant variables available in the dataset.")
    data_path: str = Field(description="Path to the source CSV file.")

class Parameter(BaseModel):
    symbol: str = Field(description="The LaTeX symbol, e.g., 'v_t' or 'T'")
    description: str = Field(description="Description of what the parameter represents")

class Variable(BaseModel):
    variable: str = Field(description="The LaTeX decision variable, e.g., 'x_{ts}'")
    meaning: str = Field(description="What the variable represents in the physical system")

class ModelingResult(BaseModel):
    """Output from the Modeling Agent."""
    use_case_context: UseCaseResult = Field(description="The input context passed from the Use Case Agent.")
    col_names_used: List[str] = Field(description="List the column names of the csv that are needed to solve the formulated problem.")
    parameters: List[Parameter] = Field(description="List of model parameters. Include sets, indices, and constants.")
    variables: List[Variable] = Field(description="Definition of all used variables.")
    minimizing_problem: bool = Field(description="True if the objective is 'min', False if it is 'max'.")
    objective_function: str = Field(description="The objective function in LaTeX.")
    constraint_functions: List[str] = Field(description="A list of each MILP constraint formula in LaTeX.")
    explanation_of_ILP: List[str] = Field(description="A numbered list of explanations for the objective and each constraint.")
    readable_documentation: str = Field(description="A complete Markdown string combining all sections for human reading.")

class PreprocessingResult(BaseModel):
    """Output from the Data Preprocessing Agent."""
    modeling_context: ModelingResult = Field(description="The modeling context passed from the Modeling Agent.")
    preprocessing_script_code: str = Field(description="Python code that converts the raw CSV into a PuLP-compatible data structure (dictionaries).")
    input_scheme_description: str = Field(description="Description of the data structure the preprocessing script outputs.")

class ScriptingResult(BaseModel):
    """Output from the Scripting Agent."""
    preprocessing_context: PreprocessingResult = Field(description="The preprocessing context passed from the Preprocessing Agent.")
    final_solution_script_code: str = Field(description="The complete Python string using PuLP that reads the preprocessed data, builds the model, solves it, and outputs the result.")
    execution_instructions: str = Field(description="Any specific instructions on how to run or use the generated code.")
