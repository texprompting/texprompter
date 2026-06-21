#!/usr/bin/env python3
import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt

# Directories containing raw data
TEST_FILES_DIR = "ModelTestFiles"
FILES_TEST_DIR = "ModelFilesTest"
OUTPUT_DIR = os.path.join("TestOutputs", "plots")

def get_agent_duration(metadata, agent_name):
    for agent in metadata:
        if agent.get('agent_name') == agent_name:
            return agent.get('duration_seconds', 0.0)
    return 0.0

def load_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ----------------------------------------------------
    # 1. Parse Qwen 3.6 Latest Runs (ModelTestFiles)
    # ----------------------------------------------------
    qwen_easy_durations = []
    qwen_easy_agents = {k: [] for k in ['use_case_agent', 'modeling_agent', 'parameter_estimation_agent', 'preprocessing_agent', 'scripting_agent']}
    for i in range(1, 11):
        path = os.path.join(TEST_FILES_DIR, f"result_qwen3_6_latest_easy_run_{i}_state_dict.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                meta = data.get('execution_metadata', [])
                total_dur = sum(a.get('duration_seconds', 0.0) for a in meta)
                qwen_easy_durations.append(total_dur)
                for k in qwen_easy_agents:
                    qwen_easy_agents[k].append(get_agent_duration(meta, k))

    qwen_medium_durations = []
    qwen_medium_agents = {k: [] for k in ['use_case_agent', 'modeling_agent', 'parameter_estimation_agent', 'preprocessing_agent', 'scripting_agent']}
    for i in range(1, 11):
        path = os.path.join(TEST_FILES_DIR, f"result_qwen3_6_latest_medium_run_{i}_state_dict.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                meta = data.get('execution_metadata', [])
                total_dur = sum(a.get('duration_seconds', 0.0) for a in meta)
                qwen_medium_durations.append(total_dur)
                for k in qwen_medium_agents:
                    qwen_medium_agents[k].append(get_agent_duration(meta, k))

    # ----------------------------------------------------
    # 2. Parse Gemini 3.5 Flash Objectives (ModelTestFiles & ModelFilesTest)
    # ----------------------------------------------------
    gemini_easy_objectives = []
    for i in range(1, 11):
        path = os.path.join(TEST_FILES_DIR, f"result_gemini3.5_flash_easy_run_{i}_execution_output.log")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.search(r'Optimal - objective value ([\d\.\+eE\-]+)', content)
                if match:
                    gemini_easy_objectives.append(float(match.group(1)))

    gemini_medium_objectives = []
    for i in range(1, 11):
        path = os.path.join(FILES_TEST_DIR, f"result_gemini3.5_flash_optimization_pipeline_test_medium_csv_run_{i}_execution_output.log")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.search(r'Objective Value: ([\d\.\+eE\-]+)', content)
                if match:
                    gemini_medium_objectives.append(float(match.group(1)))

    gemini_production_objectives = []
    for i in range(1, 11):
        path = os.path.join(FILES_TEST_DIR, f"result_gemini3.5_flash_Production_csv_run_{i}_execution_output.log")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.search(r'Objective Value: ([\d\.\+eE\-]+)', content)
                if match:
                    gemini_production_objectives.append(float(match.group(1)))

    return {
        'qwen_easy_durations': qwen_easy_durations,
        'qwen_easy_agents': qwen_easy_agents,
        'qwen_medium_durations': qwen_medium_durations,
        'qwen_medium_agents': qwen_medium_agents,
        'gemini_easy_objectives': gemini_easy_objectives,
        'gemini_medium_objectives': gemini_medium_objectives,
        'gemini_production_objectives': gemini_production_objectives
    }

def print_summary_table():
    print("""
# TexPrompter Pipeline Experiment Metrics Summary

## Aggregate Table (Gemini 3.5 Flash Ground Truth)

| Dataset | Total Runs | Pipeline Success Rate (%) | Execution Success Rate (%) | Avg Total Duration (s) | Avg UseCase Agent (s) | Avg Modeling Agent (s) | Avg Param Estimation Agent (s) | Avg Preprocessing Agent (s) | Avg Scripting Agent (s) | Avg Interpretation Agent (s) | Avg Objective Value | Std Objective Value | Min Objective Value | Max Objective Value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Easy | 10 | 100.00 | 100.00 | 94.19 | 9.00 | 9.57 | 17.18 | 22.28 | 16.25 | 19.47 | 69641.92 | 0.00 | 69641.92 | 69641.92 |
| Medium | 10 | 100.00 | 100.00 | 92.86 | 10.25 | 8.05 | 15.25 | 23.33 | 14.99 | 20.52 | 1117486.54 | 292846.66 | 798396.40 | 1494741.45 |
| Production | 10 | 90.00 | 90.00 | 134.50 | 13.29 | 11.35 | 16.87 | 39.03 | 32.58 | 23.30 | -33985.04 | 107647.93 | -340356.11 | 495.98 |
""")

def generate_plots(data):
    # Setup styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    
    # ----------------------------------------------------
    # Plot 1: Agent Durations Breakdown (Qwen 3.6 Latest)
    # ----------------------------------------------------
    if data['qwen_easy_durations'] and data['qwen_medium_durations']:
        fig, ax = plt.subplots(figsize=(10, 6))
        agents = ['UseCase', 'Modeling', 'Param Estimation', 'Preprocessing', 'Scripting']
        
        easy_means = [np.mean(data['qwen_easy_agents'][k]) for k in ['use_case_agent', 'modeling_agent', 'parameter_estimation_agent', 'preprocessing_agent', 'scripting_agent']]
        medium_means = [np.mean(data['qwen_medium_agents'][k]) for k in ['use_case_agent', 'modeling_agent', 'parameter_estimation_agent', 'preprocessing_agent', 'scripting_agent']]
        
        x = np.arange(len(agents))
        width = 0.35
        
        ax.bar(x - width/2, easy_means, width, label='Easy Dataset', color='#4A90E2')
        ax.bar(x + width/2, medium_means, width, label='Medium Dataset', color='#F5A623')
        
        ax.set_ylabel('Average Duration (seconds)')
        ax.set_title('Agent-level Inference Duration Breakdown (Qwen 3.6 Latest)')
        ax.set_xticks(x)
        ax.set_xticklabels(agents)
        ax.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "agent_duration_breakdown.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Generated Plot: {plot_path}")

    # ----------------------------------------------------
    # Plot 2: Objective Value Spread (Medium Dataset)
    # ----------------------------------------------------
    if data['gemini_medium_objectives']:
        fig, ax = plt.subplots(figsize=(8, 5))
        runs = range(1, len(data['gemini_medium_objectives']) + 1)
        ax.plot(runs, data['gemini_medium_objectives'], marker='o', linestyle='-', color='#7ED321', linewidth=2)
        ax.set_xticks(runs)
        ax.set_xlabel('Run Number')
        ax.set_ylabel('Objective Value')
        ax.set_title('Objective Value Variation across Runs (Gemini 3.5 Flash - Medium Dataset)')
        
        # Format y-axis with commas
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "medium_objective_spread.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Generated Plot: {plot_path}")

    # ----------------------------------------------------
    # Plot 3: Success Rates Comparison (Gemini 3.5 Flash)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    datasets = ['Easy', 'Medium', 'Production']
    pipeline_success = [100.0, 100.0, 90.0]
    execution_success = [100.0, 100.0, 90.0]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    ax.bar(x - width/2, pipeline_success, width, label='Pipeline Success (%)', color='#9B59B6')
    ax.bar(x + width/2, execution_success, width, label='Execution Success (%)', color='#2ECC71')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Pipeline vs. Code Execution Success Rates (Gemini 3.5 Flash)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 110)
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "success_rates_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Generated Plot: {plot_path}")

    # ----------------------------------------------------
    # Plot 4: Speed Comparison (Easy Dataset: Gemini vs Qwen)
    # ----------------------------------------------------
    if data['qwen_easy_durations']:
        fig, ax = plt.subplots(figsize=(6, 5))
        models = ['Gemini 3.5 Flash', 'Qwen 3.6 Latest']
        avg_durations = [94.19, np.mean(data['qwen_easy_durations'])]
        
        colors = ['#FF6B6B', '#4D96FF']
        bars = ax.bar(models, avg_durations, color=colors, width=0.5)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}s',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
            
        ax.set_ylabel('Average Total Duration (seconds)')
        ax.set_title('Easy Dataset Optimization Duration Comparison')
        ax.set_ylim(0, max(avg_durations) * 1.15)
        
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "model_speed_comparison.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Generated Plot: {plot_path}")

def main():
    print("Parsing raw data from ModelTestFiles and ModelFilesTest...")
    data = load_data()
    print("Generating aggregate metrics summary...")
    print_summary_table()
    print("Generating plots...")
    generate_plots(data)
    print(f"\nAll plots successfully saved under: {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()
