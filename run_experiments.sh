#!/usr/bin/env bash
# Script to run pipeline 10 times on easy, medium, and production CSVs.
# Saves state dicts and execution logs uniquely for each run, categorized by model, in ModelFilesTest/.
# Prints outputs and resets the repository state after each run.
# Bypasses and ignores MLflow.

set -euo pipefail

# Get project root directory
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Prepend project root to PYTHONPATH so imports resolve correctly
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Tip: Make sure you have activated your environment (e.g., conda activate texprompting)" >&2

# Create output directory
mkdir -p ModelFilesTest

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
DATASETS=(
    "optimization_pipeline_test_easy.csv"
    "optimization_pipeline_test_medium.csv"
    "versatile_producion_system/Production.csv"
)

for csv_file in "${DATASETS[@]}"; do
    echo ""
    echo "========================================================================="
    echo "Running dataset: $csv_file (10 times)"
    echo "========================================================================="

    for i in {1..10}; do
        echo ""
        echo "----------------------------------------"
        echo "Starting $csv_file Run $i/10"
        echo "----------------------------------------"

        # Execute the pipeline with mocked mlflow, print results, test execution, and save results
        python -c "
import sys
import os
import re
from types import ModuleType

# Mock MLflow to disable and ignore tracking/autologging completely
class MockMLflow(ModuleType):
    def __getattr__(self, name):
        if name == 'langchain':
            return self
        return lambda *args, **kwargs: None
sys.modules['mlflow'] = MockMLflow('mlflow')

from orchestrator.pipeline import run_pipeline
from agents.shared import execute_generated_pulp_model

csv_name = '$csv_file'
run_idx = '$i'
dataset_clean = re.sub(r'[^a-zA-Z0-9_\\-]', '_', os.path.basename(csv_name))

# Ensure target script does not contain stale code from a previous run
target_script = os.path.join('TestOutputs', 'generated_pulp_model.py')
if os.path.exists(target_script):
    try:
        os.remove(target_script)
    except Exception:
        pass

# 1. Run pipeline
print(f'[run_experiments] Launching pipeline for {csv_name}...')
state = run_pipeline(csv_name)

# 2. Print output of run_pipeline to stdout
print('\n[run_experiments] Output of run_pipeline (Final State Dict):')
print(state.model_dump_json(indent=2))

# 3. Get sanitized model name from environment
model_name = os.getenv('OLLAMA_MODEL', 'qwen3.6:latest')
model_clean = re.sub(r'[^a-zA-Z0-9_\\-]', '_', model_name)

# 4. Make sure that the script that gets executed is always the newest generated one
if state.scripting and state.scripting.code:
    os.makedirs(os.path.dirname(target_script), exist_ok=True)
    with open(target_script, 'w', encoding='utf-8') as f_code:
        f_code.write(state.scripting.code)
    print(f'[run_experiments] Wrote generated code directly to {target_script}')

# 5. Save state dict to ModelFilesTest
os.makedirs('ModelFilesTest', exist_ok=True)
state_dict_file = f'ModelFilesTest/result_{model_clean}_{dataset_clean}_run_{run_idx}_state_dict.json'
with open(state_dict_file, 'w', encoding='utf-8') as f:
    f.write(state.model_dump_json(indent=2))
print(f'[run_experiments] Saved state dict to {state_dict_file}')

# 6. Check if the generated script can be executed
exec_output_file = f'ModelFilesTest/result_{model_clean}_{dataset_clean}_run_{run_idx}_execution_output.log'
try:
    print('[run_experiments] Testing execution of generated PuLP script...')
    exec_output = execute_generated_pulp_model()
    print('[run_experiments] SUCCESS: Generated script executed cleanly!')
except Exception as e:
    exec_output = str(e)
    print(f'[run_experiments] FAILURE: Generated script execution failed: {e}')

# 7. Save execution output to ModelFilesTest
with open(exec_output_file, 'w', encoding='utf-8') as f:
    f.write(exec_output)
print(f'[run_experiments] Saved execution output to {exec_output_file}')
"

        # Revert and clean up after this run
        cleanup
        echo "--- Finished $csv_file Run $i/10 ---"
    done
done

echo ""
echo "========================================================================="
echo "All runs completed successfully!"
echo "Saved files are in ModelFilesTest/"
echo "========================================================================="
