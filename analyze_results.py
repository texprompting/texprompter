#!/usr/bin/env python3
"""
Python script to analyze pipeline run outputs from ModelFilesTest/ and ModelTestFiles/.
Deduces metrics (success rates, agent durations, objective consistency) and generates plots.
Saves the results and plots in ModelFilesTest/plots/ and ModelFilesTest/metrics_summary.md.
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Directories
MODEL_FILES_TEST_DIR = "ModelFilesTest"
MODEL_TEST_FILES_DIR = "ModelTestFiles"
PLOTS_DIR = os.path.join(MODEL_FILES_TEST_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def parse_dataset_label(filename):
    """Map filename keywords to a clean dataset label."""
    fname_lower = filename.lower()
    if "easy" in fname_lower:
        return "Easy"
    elif "medium" in fname_lower:
        return "Medium"
    elif "production" in fname_lower:
        return "Production"
    return "Unknown"

def extract_obj_from_log(log_content):
    """Regex parse objective value from execution log content."""
    # Pattern for clean new format: Objective Value: 476585.17
    match = re.search(r"objective(?:\s+value)?:\s*([+-]?\d+(?:\.\d+)?)", log_content, re.IGNORECASE)
    if match:
        return float(match.group(1))
    # Pattern for CBC standard output: Optimal - objective value 69641.922
    match = re.search(r"objective\s+value\s+([+-]?\d+(?:\.\d+)?)", log_content, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"objective\s+([+-]?\d+(?:\.\d+)?)", log_content, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def is_log_successful(log_content):
    """Determine if a PuLP execution log indicates successful optimal solution."""
    content_lower = log_content.lower()
    # Check for python execution errors or file not found errors
    has_error = any(x in content_lower for x in ["error", "traceback", "exception", "no such file", "failed"])
    # Check if solved to optimal
    has_optimal = "optimal" in content_lower or "executed cleanly" in content_lower
    return has_optimal and not has_error

def gather_run_data():
    """Scan directories, parse JSON state dicts and log files, and aggregate data."""
    runs = {} # key: (dataset, run_idx)

    # 1. Process all files in ModelFilesTest
    if os.path.exists(MODEL_FILES_TEST_DIR):
        for f in os.listdir(MODEL_FILES_TEST_DIR):
            fpath = os.path.join(MODEL_FILES_TEST_DIR, f)
            if not os.path.isfile(fpath):
                continue
            
            dataset = parse_dataset_label(f)
            # Find run index (e.g., run_1, run_10)
            run_match = re.search(r"run_(\d+)", f)
            if not run_match or dataset == "Unknown":
                continue
            run_idx = int(run_match.group(1))
            run_key = (dataset, run_idx)

            if run_key not in runs:
                runs[run_key] = {
                    "dataset": dataset,
                    "run_idx": run_key[1],
                    "pipeline_success": False,
                    "execution_success": False,
                    "objective_value": None,
                    "durations": {},
                    "total_duration": 0.0
                }

            if f.endswith(".json"):
                # Parse JSON state dict
                try:
                    with open(fpath, "r", encoding="utf-8") as file:
                        state = json.load(file)
                    
                    runs[run_key]["pipeline_success"] = (state.get("status") == "ok")
                    
                    # Extract agent stage durations
                    metadata = state.get("execution_metadata", [])
                    total_dur = 0.0
                    for stage in metadata:
                        agent_name = stage.get("agent_name", "")
                        dur = stage.get("duration_seconds", 0.0)
                        if agent_name and dur:
                            runs[run_key]["durations"][agent_name] = dur
                            total_dur += dur
                    runs[run_key]["total_duration"] = total_dur

                    # Extract scripting details
                    scripting = state.get("scripting")
                    if scripting:
                        runs[run_key]["objective_value"] = scripting.get("objective_value")
                        sol_status = scripting.get("solution_status", "")
                        if sol_status == "Optimal":
                            runs[run_key]["execution_success"] = True
                except Exception as e:
                    print(f"Warning: Failed to parse JSON {f}: {e}")

            elif f.endswith(".log"):
                # Parse log file
                try:
                    with open(fpath, "r", encoding="utf-8") as file:
                        content = file.read()
                    
                    exec_success = is_log_successful(content)
                    # Only overwrite if true (since json also provides execution_success)
                    if exec_success:
                        runs[run_key]["execution_success"] = True
                    
                    obj = extract_obj_from_log(content)
                    if obj is not None:
                        runs[run_key]["objective_value"] = obj
                except Exception as e:
                    print(f"Warning: Failed to parse log {f}: {e}")

    # 2. Process any additional log files in ModelTestFiles (filling missing execution status)
    if os.path.exists(MODEL_TEST_FILES_DIR):
        for f in os.listdir(MODEL_TEST_FILES_DIR):
            fpath = os.path.join(MODEL_TEST_FILES_DIR, f)
            if not os.path.isfile(fpath) or not f.endswith(".log"):
                continue

            dataset = parse_dataset_label(f)
            run_match = re.search(r"run_(\d+)", f)
            if not run_match or dataset == "Unknown":
                continue
            run_idx = int(run_match.group(1))
            run_key = (dataset, run_idx)

            # We only use ModelTestFiles if we don't have this run or it lacks execution details
            if run_key not in runs:
                runs[run_key] = {
                    "dataset": dataset,
                    "run_idx": run_key[1],
                    "pipeline_success": False, # No state dict, assume False or N/A
                    "execution_success": False,
                    "objective_value": None,
                    "durations": {},
                    "total_duration": 0.0
                }

            try:
                with open(fpath, "r", encoding="utf-8") as file:
                    content = file.read()
                
                exec_success = is_log_successful(content)
                if exec_success:
                    runs[run_key]["execution_success"] = True
                
                obj = extract_obj_from_log(content)
                if obj is not None:
                    runs[run_key]["objective_value"] = obj
            except Exception as e:
                print(f"Warning: Failed to parse log {f}: {e}")

    # Convert to DataFrame
    data_list = list(runs.values())
    return pd.DataFrame(data_list)

def generate_report_and_plots(df):
    """Compute aggregate metrics, generate markdown report, and build plots."""
    if df.empty:
        print("No run data found to analyze.")
        return

    # Normalize agent names in duration dictionaries
    all_stages = [
        "use_case_agent",
        "modeling_agent",
        "parameter_estimation_agent",
        "preprocessing_agent",
        "scripting_agent",
        "results_interpretation_agent"
    ]
    
    # Expand duration columns
    for stage in all_stages:
        df[stage] = df["durations"].apply(lambda d: d.get(stage, np.nan))

    # Aggregates by dataset
    grouped = df.groupby("dataset")
    summary_data = []

    for name, group in grouped:
        runs_count = len(group)
        pipeline_success_rate = group["pipeline_success"].mean() * 100
        execution_success_rate = group["execution_success"].mean() * 100
        
        # Filter for valid values
        valid_durations = group[group["total_duration"] > 0]
        avg_total_dur = valid_durations["total_duration"].mean() if not valid_durations.empty else np.nan
        
        # Stage durations averages
        stage_averages = {}
        for stage in all_stages:
            stage_vals = group[stage].dropna()
            stage_averages[stage] = stage_vals.mean() if not stage_vals.empty else np.nan

        # Objective values metrics
        obj_vals = group["objective_value"].dropna()
        obj_mean = obj_vals.mean() if not obj_vals.empty else np.nan
        obj_std = obj_vals.std() if len(obj_vals) > 1 else 0.0
        obj_min = obj_vals.min() if not obj_vals.empty else np.nan
        obj_max = obj_vals.max() if not obj_vals.empty else np.nan

        summary_data.append({
            "Dataset": name,
            "Total Runs": runs_count,
            "Pipeline Success Rate (%)": pipeline_success_rate,
            "Execution Success Rate (%)": execution_success_rate,
            "Avg Total Duration (s)": avg_total_dur,
            "Avg UseCase Agent (s)": stage_averages["use_case_agent"],
            "Avg Modeling Agent (s)": stage_averages["modeling_agent"],
            "Avg Param Estimation Agent (s)": stage_averages["parameter_estimation_agent"],
            "Avg Preprocessing Agent (s)": stage_averages["preprocessing_agent"],
            "Avg Scripting Agent (s)": stage_averages["scripting_agent"],
            "Avg Interpretation Agent (s)": stage_averages["results_interpretation_agent"],
            "Avg Objective Value": obj_mean,
            "Std Objective Value": obj_std,
            "Min Objective Value": obj_min,
            "Max Objective Value": obj_max,
        })

    summary_df = pd.DataFrame(summary_data)
    
    # Custom markdown table generator to avoid tabulate dependency
    def df_to_markdown_manual(df):
        cols = df.columns
        header = "| " + " | ".join(cols) + " |"
        divider = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = []
        for _, row in df.iterrows():
            row_str = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    if np.isnan(val):
                        row_str.append("N/A")
                    else:
                        row_str.append(f"{val:.2f}")
                else:
                    row_str.append(str(val))
            rows.append("| " + " | ".join(row_str) + " |")
        return "\n".join([header, divider] + rows)

    # Save markdown summary
    md_file = os.path.join(MODEL_FILES_TEST_DIR, "metrics_summary.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# TexPrompter Pipeline Experiment Metrics Summary\n\n")
        f.write("This summary aggregates the performance of the pipeline across all test runs.\n\n")
        f.write("## Aggregate Table\n\n")
        f.write(df_to_markdown_manual(summary_df))
        f.write("\n\n")
        f.write("## Summary Insights\n\n")
        for row in summary_data:
            f.write(f"### {row['Dataset']} Dataset\n")
            f.write(f"- **Runs**: {row['Total Runs']}\n")
            f.write(f"- **Pipeline Success Rate**: {row['Pipeline Success Rate (%)']:.1f}%\n")
            f.write(f"- **PuLP Script Execution Success Rate**: {row['Execution Success Rate (%)']:.1f}%\n")
            if not np.isnan(row['Avg Total Duration (s)']):
                f.write(f"- **Average Inference Duration**: {row['Avg Total Duration (s)']:.2f} seconds\n")
            if not np.isnan(row['Avg Objective Value']):
                f.write(f"- **Objective Value Consistency**: Mean={row['Avg Objective Value']:.2f}, Std={row['Std Objective Value']:.2f} (Range: {row['Min Objective Value']:.2f} - {row['Max Objective Value']:.2f})\n")
            f.write("\n")
            
    print(f"Wrote markdown summary report to: {md_file}")

    # Plot 1: Stacked Bar Chart of Average Durations per Stage
    fig, ax = plt.subplots(figsize=(10, 6))
    stage_labels = [
        "Use Case", "Modeling", "Parameter Est.", "Preprocessing", "Scripting", "Interpretation"
    ]
    stage_col_map = {
        "use_case_agent": "Avg UseCase Agent (s)",
        "modeling_agent": "Avg Modeling Agent (s)",
        "parameter_estimation_agent": "Avg Param Estimation Agent (s)",
        "preprocessing_agent": "Avg Preprocessing Agent (s)",
        "scripting_agent": "Avg Scripting Agent (s)",
        "results_interpretation_agent": "Avg Interpretation Agent (s)"
    }
    
    # Filter datasets that have duration data
    datasets_with_time = [row["Dataset"] for row in summary_data if not np.isnan(row["Avg Total Duration (s)"])]
    if datasets_with_time:
        bottoms = np.zeros(len(datasets_with_time))
        for key, label in zip(all_stages, stage_labels):
            vals = []
            col_name = stage_col_map[key]
            for dataset in datasets_with_time:
                row = summary_df[summary_df["Dataset"] == dataset].iloc[0]
                val = row[col_name]
                vals.append(val if not np.isnan(val) else 0.0)
            
            ax.bar(datasets_with_time, vals, bottom=bottoms, label=label)
            bottoms += np.array(vals)

        ax.set_ylabel("Average Duration (seconds)")
        ax.set_title("Average Pipeline Inference Time Breakdown per Agent Stage")
        ax.legend(title="Agent Stage")
        plt.tight_layout()
        plot1_path = os.path.join(PLOTS_DIR, "avg_agent_durations.png")
        plt.savefig(plot1_path, dpi=150)
        plt.close()
        print(f"Saved plot: {plot1_path}")

    # Plot 2: Success Rates Comparison (Pipeline vs Execution)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary_df["Dataset"]))
    width = 0.35

    ax.bar(x - width/2, summary_df["Pipeline Success Rate (%)"], width, label="Pipeline Completion", color="skyblue")
    ax.bar(x + width/2, summary_df["Execution Success Rate (%)"], width, label="PuLP Execution Success", color="lightgreen")

    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rates per Dataset (Pipeline Status vs PuLP Solve)")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["Dataset"])
    ax.set_ylim(0, 110)
    ax.legend()
    plt.tight_layout()
    plot2_path = os.path.join(PLOTS_DIR, "success_rates.png")
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    print(f"Saved plot: {plot2_path}")

    # Plot 3: Boxplot of Total Duration by Dataset
    valid_durations_df = df[df["total_duration"] > 0]
    if not valid_durations_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        datasets_list = sorted(valid_durations_df["dataset"].unique())
        box_data = [valid_durations_df[valid_durations_df["dataset"] == ds]["total_duration"].values for ds in datasets_list]
        
        ax.boxplot(box_data, labels=datasets_list)
        ax.set_ylabel("Total Duration (seconds)")
        ax.set_title("Distribution of Total Pipeline Inference Time by Dataset")
        plt.tight_layout()
        plot3_path = os.path.join(PLOTS_DIR, "total_duration_boxplot.png")
        plt.savefig(plot3_path, dpi=150)
        plt.close()
        print(f"Saved plot: {plot3_path}")

    # Plot 4: Objective Value Consistency across runs
    fig, ax = plt.subplots(figsize=(10, 5))
    has_obj = False
    for ds in sorted(df["dataset"].unique()):
        ds_group = df[(df["dataset"] == ds) & (df["objective_value"].notna())].sort_values("run_idx")
        if not ds_group.empty:
            ax.plot(ds_group["run_idx"], ds_group["objective_value"], marker="o", label=f"{ds} Objective Value")
            has_obj = True

    if has_obj:
        ax.set_xlabel("Run Index")
        ax.set_ylabel("Objective Value (Optimal Target)")
        ax.set_title("Objective Value Consistency Across Runs")
        ax.set_xticks(range(1, 11))
        ax.legend()
        plt.yscale("log") # Log scale because production/easy/medium might have vastly different ranges
        ax.set_ylabel("Objective Value (Log Scale)")
        plt.tight_layout()
        plot4_path = os.path.join(PLOTS_DIR, "objective_consistency.png")
        plt.savefig(plot4_path, dpi=150)
        plt.close()
        print(f"Saved plot: {plot4_path}")

if __name__ == "__main__":
    print("Starting analysis of experiments output...")
    run_df = gather_run_data()
    if not run_df.empty:
        print(f"Found {len(run_df)} runs in test folders.")
        generate_report_and_plots(run_df)
        print("Analysis completed successfully.")
    else:
        print("No test data directories found or no parseable runs found.")
