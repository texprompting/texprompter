# TexPrompter Pipeline Experiment Metrics Summary

This summary aggregates the performance of the pipeline across all test runs.

## Aggregate Table

| Dataset | Total Runs | Pipeline Success Rate (%) | Execution Success Rate (%) | Avg Total Duration (s) | Avg UseCase Agent (s) | Avg Modeling Agent (s) | Avg Param Estimation Agent (s) | Avg Preprocessing Agent (s) | Avg Scripting Agent (s) | Avg Interpretation Agent (s) | Avg Objective Value | Std Objective Value | Min Objective Value | Max Objective Value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Easy | 10 | 100.00 | 100.00 | 94.19 | 9.00 | 9.57 | 17.18 | 22.28 | 16.25 | 19.47 | 69641.92 | 0.00 | 69641.92 | 69641.92 |
| Medium | 10 | 100.00 | 100.00 | 92.86 | 10.25 | 8.05 | 15.25 | 23.33 | 14.99 | 20.52 | 1117486.54 | 292846.66 | 798396.40 | 1494741.45 |
| Production | 10 | 90.00 | 90.00 | 134.50 | 13.29 | 11.35 | 16.87 | 39.03 | 32.58 | 23.30 | -33985.04 | 107647.93 | -340356.11 | 495.98 |

## Summary Insights

### Easy Dataset
- **Runs**: 10
- **Pipeline Success Rate**: 100.0%
- **PuLP Script Execution Success Rate**: 100.0%
- **Average Inference Duration**: 94.19 seconds
- **Objective Value Consistency**: Mean=69641.92, Std=0.00 (Range: 69641.92 - 69641.92)

### Medium Dataset
- **Runs**: 10
- **Pipeline Success Rate**: 100.0%
- **PuLP Script Execution Success Rate**: 100.0%
- **Average Inference Duration**: 92.86 seconds
- **Objective Value Consistency**: Mean=1117486.54, Std=292846.66 (Range: 798396.40 - 1494741.45)

### Production Dataset
- **Runs**: 10
- **Pipeline Success Rate**: 90.0%
- **PuLP Script Execution Success Rate**: 90.0%
- **Average Inference Duration**: 134.50 seconds
- **Objective Value Consistency**: Mean=-33985.04, Std=107647.93 (Range: -340356.11 - 495.98)

