import os
import warnings
import pandas as pd
import json

from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List

# ENV SETUP
load_dotenv()

ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_api_key = os.getenv("OLLAMA_API_KEY")
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.3:70b-instruct-q3_K_M")

headers = {}
if ollama_api_key and ollama_api_key != "your_api_key_here":
    headers["Authorization"] = f"Bearer {ollama_api_key}"

print(f"Connecting to Ollama at {ollama_base_url}...")

# Allow pipeline runs to inject the selected dataset while preserving the old local default for standalone runs.
CSV_PATH = os.getenv("PIPELINE_CSV_PATH", "./Deliverymodule.csv")

# TOOLS
@tool
def get_column_names() -> list[str]:
    """Returns column names of the dataset"""
    df = pd.read_csv(CSV_PATH, nrows=0)
    cols = df.columns.tolist()
    print("Columns:", cols)
    return cols


@tool
def get_csv_preview() -> str:
    """Returns first rows of dataset"""
    df = pd.read_csv(CSV_PATH)
    preview = df.head(10).to_string()
    print("Preview:\n", preview)
    return preview


@tool
def get_basic_stats() -> dict:
    """Returns statistical summary"""
    df = pd.read_csv(CSV_PATH)
    stats = df.describe().to_string()
    print("Stats:\n", stats)
    return {"raw_stats": stats}


# OUTPUT SCHEMA
class ContextRecommendation(BaseModel):
    use_case: str = Field(description="Best optimization use case")
    objective: str = Field(description="Objective of optimization")
    decision_variables: List[str] = Field(description="Variables to optimize")
    relevant_columns: List[str] = Field(description="Relevant CSV columns")
    statistics: str = Field(description="RAW statistical summary from tool")
    reasoning: str = Field(description="Why this use case was chosen")


# LLM SETUP
llm = ChatOllama(
    base_url=ollama_base_url,
    model=ollama_model,
    client_kwargs={
        "headers": headers,
        "timeout": 120.0,   # optional but helps with slow servers
        "verify": False,
    } if headers else {"timeout": 120.0, "verify": False},
    reasoning=True,
)

# SYSTEM PROMPT
system_message = (
    "You are an expert in operations research and production optimization.\n\n"

    "You MUST follow this EXACT workflow:\n\n"

    "STEP 1: Call tools to inspect the dataset:\n"
    "- First call get_csv_preview\n"
    "- Then call get_column_names\n"
    "- Then call get_basic_stats\n\n"

    "STEP 2: Analyze the results internally\n\n"

    "STEP 3: Output your final answer by calling ContextRecommendation\n\n"

    "CRITICAL RULES:\n"
    "- DO NOT describe what you will do\n"
    "- DO NOT output JSON manually\n"
    "- ONLY communicate via tool calls\n"
    "- You may call multiple tools before finishing\n"
    "- Final answer MUST be a ContextRecommendation tool call\n"
    "- NEVER return plain text\n\n"

    "IMPORTANT:\n"
    "- The 'statistics' field MUST be EXACTLY the value of 'raw_stats' returned by get_basic_stats\n"
    "- DO NOT summarize, modify, or interpret the statistics\n"
    "- COPY the statistics exactly\n"
    "- Do NOT create constraints\n"
)

# AGENT CREATION
tools = [get_column_names, get_csv_preview, get_basic_stats]

context_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_message,
    response_format=ContextRecommendation
)

# RUN AGENT
def run_context_agent():
    messages = [
        {"role": "user", "content": "Analyze the dataset and identify the best optimization use case."}
    ]

    max_retries = 3

    for attempt in range(max_retries):
        response = context_agent.invoke({"messages": messages})

        print(f"\n--- Attempt {attempt + 1} ---")

        for msg in response['messages'][-4:]:
            print(f"[{msg.type.upper()}]: {msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print("Tool Calls:", msg.tool_calls)

        ai_messages = [m for m in response['messages'] if m.type == 'ai']
        final_ai_msg = ai_messages[-1] if ai_messages else None

        if final_ai_msg and final_ai_msg.tool_calls:
            for tc in final_ai_msg.tool_calls:
                if tc['name'] == 'ContextRecommendation':
                    print("\n✅ SUCCESS")
                    return tc['args']

        print("❌ Failed attempt, retrying...")

        messages = response['messages'] + [
            {
                "role": "user",
                "content": "You FAILED. You must ONLY respond with a ContextRecommendation tool call. No text."
            }
        ]

    return None


# FORMAT FOR NEXT AGENT
def format_for_modelling_agent(ctx: dict) -> str:
    return f"""
Use Case:
{ctx['use_case']}

Objective:
{ctx['objective']}

Decision Variables:
{ctx['decision_variables']}

Relevant Columns:
{ctx['relevant_columns']}

Statistics:
{ctx['statistics']}

Reasoning:
{ctx['reasoning']}
"""


# MAIN EXECUTION
if __name__ == "__main__":
    ctx = run_context_agent()

    if ctx:
        print("\n=== CONTEXT AGENT OUTPUT ===\n")
        print(json.dumps(ctx, indent=2))

        formatted = format_for_modelling_agent(ctx)

        print("\n=== FORMATTED FOR MODELLING AGENT ===\n")
        print(formatted)

        with open("../data/context_output.txt", "w", encoding="utf-8") as f:
            f.write(formatted)

    else:
        print("❌ No valid output from context agent")