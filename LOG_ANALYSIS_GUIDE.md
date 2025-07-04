# Detailed Log Analysis Guide

This guide explains how to analyze the detailed evaluation logs from your information sources (auditor, regulator, user_rep) using the provided analysis tools.

## Overview

When you run detailed analysis using `detailed_source_analysis.py`, it generates JSON files containing:
- Pairwise comparisons between agents across multiple runs
- Raw LLM reasoning for each dimension
- Derived scores and confidences
- Agent profiles

Two analysis tools are available to help you examine this data:

## Tool 1: `analyze_detailed_logs.py` (Comprehensive Analysis)

This is the main analysis tool with extensive features and export capabilities.

### Basic Usage

```bash
# Show all analysis views
python analyze_detailed_logs.py --data_path outputs/detailed_analysis/auditor_analysis_20241220_143025.json

# Show only specific views
python analyze_detailed_logs.py --data_path <path_to_json> --view profiles
python analyze_detailed_logs.py --data_path <path_to_json> --view consistency 
python analyze_detailed_logs.py --data_path <path_to_json> --view dimensions
python analyze_detailed_logs.py --data_path <path_to_json> --view reasoning
```

### Advanced Usage

```bash
# Analyze specific dimension
python analyze_detailed_logs.py --data_path <path_to_json> --view dimensions --dimension "Factual_Correctness"

# Focus on specific agent pair
python analyze_detailed_logs.py --data_path <path_to_json> --view reasoning --agent_a 0 --agent_b 1

# Export to Excel for further analysis
python analyze_detailed_logs.py --data_path <path_to_json> --export_excel analysis_results.xlsx

# Save text report
python analyze_detailed_logs.py --data_path <path_to_json> --save_report consistency_report.txt
```

### What Each View Shows

1. **Profiles View** (`--view profiles`):
   - Displays all agent profiles in readable format
   - Shows goals, communication styles, behavioral tendencies
   - Useful for understanding what each agent is designed to do

2. **Consistency View** (`--view consistency`):
   - Shows how consistent evaluations are across runs for each pair
   - Provides win/loss patterns 
   - Calculates score variance and consistency metrics
   - Helps identify if the auditor is stable in its judgments

3. **Dimensions View** (`--view dimensions`):
   - Breaks down performance by evaluation dimension
   - Shows which agent wins more often in each dimension
   - Provides confidence and score statistics per dimension
   - Useful for understanding strengths/weaknesses

4. **Reasoning View** (`--view reasoning`):
   - Shows the raw LLM reasoning for each comparison
   - Displays scores, winners, and confidence levels
   - Most detailed view for understanding "why" decisions were made

## Tool 2: `quick_scan_logs.py` (Interactive Scanner)

This is a simpler, more interactive tool for quick exploration.

### Usage Modes

```bash
# Quick overview
python quick_scan_logs.py <path_to_json> --mode overview

# Summary with win/loss patterns
python quick_scan_logs.py <path_to_json> --mode summary

# Dimension-wise winner patterns
python quick_scan_logs.py <path_to_json> --mode dimensions

# Interactive browsing mode
python quick_scan_logs.py <path_to_json> --mode browse
```

### Interactive Browser Commands

When using `--mode browse`, you get an interactive prompt:

```
> list                    # Show available pairs and dimensions
> pair 0 1               # Show detailed comparison for agents 0 vs 1
> dim Factual_Correctness # Show reasoning for this dimension across all pairs
> quit                   # Exit
```

## Analysis Workflow Recommendations

### Step 1: Quick Overview
Start with the quick scanner to get oriented:
```bash
python quick_scan_logs.py <your_data.json> --mode summary
```

### Step 2: Identify Patterns
Look for interesting patterns:
- Which pairs have high vs low consistency?
- Are there clear winners across dimensions?
- Which dimensions show the most variation?

### Step 3: Deep Dive on Specific Cases
Use the comprehensive tool to examine interesting cases:
```bash
# Focus on a specific problematic pair
python analyze_detailed_logs.py --data_path <your_data.json> --view reasoning --agent_a 0 --agent_b 2

# Or examine a specific dimension
python analyze_detailed_logs.py --data_path <your_data.json> --view dimensions --dimension "Value_Alignment"
```

### Step 4: Export for Further Analysis
```bash
python analyze_detailed_logs.py --data_path <your_data.json> --export_excel full_analysis.xlsx
```

## Key Metrics to Look For

### Consistency Indicators
- **Low standard deviation in scores** (< 0.05): High consistency
- **Stable win/loss patterns**: Reliable evaluations
- **Similar confidence levels**: Stable uncertainty quantification

### Quality Indicators
- **High confidence scores**: The auditor is sure about differences
- **Clear reasoning**: LLM provides specific, relevant justifications
- **Logical score differences**: Winners have higher scores

### Potential Issues
- **High variance across runs**: Unstable evaluations
- **Inconsistent winners**: Unreliable comparisons  
- **Low confidence consistently**: Auditor can't distinguish agents
- **Generic reasoning**: LLM not providing specific insights

## Example Analysis Questions

### Understanding Auditor Reliability
1. "Are the auditor's evaluations consistent across the 5 runs?"
   - Look at consistency view and standard deviations
   - Check if the same agent wins most comparisons across runs

2. "Which dimensions does the auditor evaluate most confidently?"
   - Compare confidence scores across dimensions
   - Look for dimensions with low variance

### Understanding Agent Differences  
1. "What specific differences does the auditor see between agents?"
   - Read the detailed reasoning for each pair
   - Look for recurring themes in the LLM explanations

2. "Are there agents that consistently outperform others?"
   - Check win/loss patterns in consistency view
   - Look at average score differences

### Validation and Debugging
1. "Does the auditor's reasoning make sense given the agent profiles?"
   - Compare profiles view with reasoning view
   - Check if conclusions align with profile differences

2. "Are there any anomalous evaluations that need investigation?"
   - Look for runs with very different results
   - Check for inconsistent reasoning patterns

## Output Files

The analysis tools can generate several useful outputs:

1. **Excel file** (`--export_excel`): Three sheets
   - Summary: All scores in tabular format
   - Detailed_Reasoning: All LLM reasoning with metadata
   - Consistency_Metrics: Statistical analysis per pair/dimension

2. **Text report** (`--save_report`): 
   - Summary statistics
   - Consistency metrics
   - Key findings

## Tips for Effective Analysis

1. **Start broad, then narrow**: Use overview → summary → specific pairs/dimensions
2. **Look for patterns**: Consistent winners, problematic dimensions, confidence trends
3. **Validate reasoning**: Check if LLM explanations make sense
4. **Compare across sources**: Run analysis on auditor, user_rep, regulator logs to compare
5. **Use Excel export**: For statistical analysis, plotting, or sharing results

## Troubleshooting

### Common Issues
- **File not found**: Check the path to your detailed analysis JSON
- **No data displayed**: Verify the JSON structure matches expected format
- **Interactive mode not working**: Try different terminal or use basic views instead

### Data Quality Checks
- Verify all runs completed successfully
- Check that agent profiles are correctly loaded
- Ensure all dimensions have data

This guide should help you efficiently analyze your detailed evaluation logs and understand how well your information sources are performing! 