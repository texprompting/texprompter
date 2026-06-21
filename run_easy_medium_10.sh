#!/usr/bin/env bash
# Script to run pipeline 10 times on easy CSV and 10 times on medium CSV.
# Saves state dicts and execution logs uniquely for each run, categorized by model.
# Prints outputs and resets the repository state after each run.
# Bypasses and ignores MLflow.

set -euo pipefail

# Get project root directory
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Prepend project root to PYTHONPATH so imports resolve correctly
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Tip: Make sure you have activated your environment (e.g., conda activate texprompting)" >&2

# Cleanup function to reset TestOutputs/ only (ignores mlflow.db)
cleanup() {
    echo ""
    echo "=== Resetting changes (cleanup) ==="
    git restore TestOutputs/ 2>/dev/null || git checkout -- TestOutputs/
    git clean -fd TestOutputs/
}

# Register the cleanup function to always run on script exit (handles errors/interrupts)
trap cleanup EXIT

# Define files to run
EASY_CSV="optimization_pipeline_test_easy.csv"
MEDIUM_CSV="optimization_pipeline_test_medium.csv"

# ----------------- RUN EASY: 10 TIMES -----------------
echo ""
echo "========================================================================="
echo "Running EASY dataset ($EASY_CSV) 10 times"
echo "========================================================================="

for i in {1..10}; do
    echo ""
    echo "----------------------------------------"
    echo "Starting EASY Run $i/10"
    echo "----------------------------------------"

    # Execute the pipeline with mocked mlflow, print results, test execution, and save results
    python -c "
import sys
from types import ModuleType

# Mock MLflow to disable and ignore tracking/autologging completely
class MockMLflow(ModuleType):
    def __getattr__(self, name):
        if name == 'langchain':
            return self
        return lambda *args, **kwargs: None
sys.modules['mlflow'] = MockMLflow('mlflow')

import json
import os
import re
from orchestrator.pipeline import run_pipeline
from agents.shared import execute_generated_pulp_model

# 1. Run pipeline (uses default preview_rows)
print('[run_experiments] Launching pipeline for easy CSV...')
state = run_pipeline('$EASY_CSV')

# 2. Print output of run_pipeline to stdout
print('\n[run_experiments] Output of run_pipeline (Final State Dict):')
print(state.model_dump_json(indent=2))

# 3. Get sanitized model name from environment
model_name = os.getenv('OLLAMA_MODEL', 'qwen3.6:latest')
model_clean = re.sub(r'[^a-zA-Z0-9_\\-]', '_', model_name)

# 4. Save state dict
state_dict_file = f'result_{model_clean}_easy_run_${i}_state_dict.json'
with open(state_dict_file, 'w', encoding='utf-8') as f:
    f.write(state.model_dump_json(indent=2))
print(f'\n[run_experiments] Saved state dict to {state_dict_file}')

# 5. Check if the generated script can be executed
exec_output_file = f'result_{model_clean}_easy_run_${i}_execution_output.log'
try:
    print('[run_experiments] Testing execution of generated PuLP script...')
    exec_output = execute_generated_pulp_model()
    print('[run_experiments] SUCCESS: Generated script executed cleanly!')
except Exception as e:
    exec_output = str(e)
    print(f'[run_experiments] FAILURE: Generated script execution failed: {e}')

# 6. Save execution output
with open(exec_output_file, 'w', encoding='utf-8') as f:
    f.write(exec_output)
print(f'[run_experiments] Saved execution output to {exec_output_file}')
"

    # Revert and clean up after this run
    cleanup
    echo "--- Finished EASY Run $i/10 ---"
done


# ----------------- RUN MEDIUM: 10 TIMES -----------------
echo ""
echo "========================================================================="
echo "Running MEDIUM dataset ($MEDIUM_CSV) 10 times"
echo "========================================================================="

for i in {1..10}; do
    echo ""
    echo "----------------------------------------"
    echo "Starting MEDIUM Run $i/10"
    echo "----------------------------------------"

    # Execute the pipeline with mocked mlflow, print results, test execution, and save results
    python -c "
import sys
from types import ModuleType

# Mock MLflow to disable and ignore tracking/autologging completely
class MockMLflow(ModuleType):
    def __getattr__(self, name):
        if name == 'langchain':
            return self
        return lambda *args, **kwargs: None
sys.modules['mlflow'] = MockMLflow('mlflow')

import json
import os
import re
from orchestrator.pipeline import run_pipeline
from agents.shared import execute_generated_pulp_model

# 1. Run pipeline (uses default preview_rows)
print('[run_experiments] Launching pipeline for medium CSV...')
state = run_pipeline('$MEDIUM_CSV')

# 2. Print output of run_pipeline to stdout
print('\n[run_experiments] Output of run_pipeline (Final State Dict):')
print(state.model_dump_json(indent=2))

# 3. Get sanitized model name from environment
model_name = os.getenv('OLLAMA_MODEL', 'qwen3.6:latest')
model_clean = re.sub(r'[^a-zA-Z0-9_\\-]', '_', model_name)

# 4. Save state dict
state_dict_file = f'result_{model_clean}_medium_run_${i}_state_dict.json'
with open(state_dict_file, 'w', encoding='utf-8') as f:
    f.write(state.model_dump_json(indent=2))
print(f'\n[run_experiments] Saved state dict to {state_dict_file}')

# 5. Check if the generated script can be executed
exec_output_file = f'result_{model_clean}_medium_run_${i}_execution_output.log'
try:
    print('[run_experiments] Testing execution of generated PuLP script...')
    exec_output = execute_generated_pulp_model()
    print('[run_experiments] SUCCESS: Generated script executed cleanly!')
except Exception as e:
    exec_output = str(e)
    print(f'[run_experiments] FAILURE: Generated script execution failed: {e}')

# 6. Save execution output
with open(exec_output_file, 'w', encoding='utf-8') as f:
    f.write(exec_output)
print(f'[run_experiments] Saved execution output to {exec_output_file}')
"

    # Revert and clean up after this run
    cleanup
    echo "--- Finished MEDIUM Run $i/10 ---"
done

echo ""
echo "========================================================================="
echo "All 20 runs completed successfully!"
echo "Saved files:"
echo "  - result_*_run_*_state_dict.json"
echo "  - result_*_run_*_execution_output.log"
echo "========================================================================="
