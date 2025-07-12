#!/usr/bin/env python3
"""
Example usage script for detailed log analysis tools.

This script demonstrates how to use the analysis tools and provides 
sample commands for different analysis scenarios.
"""

import os
import sys

def print_usage_examples():
    """Print example usage commands for the analysis tools."""
    
    print("="*70)
    print("DETAILED LOG ANALYSIS - USAGE EXAMPLES")
    print("="*70)
    
    print("\n1. GENERATING DETAILED LOGS")
    print("-" * 40)
    print("First, run your detailed analysis to generate logs:")
    print("python detailed_source_analysis.py \\")
    print("    --input_file run_logs/primary_source_test_20241220_142030.json \\")
    print("    --source_to_test auditor \\")
    print("    --n_runs 5 \\")
    print("    --output_file outputs/detailed_analysis/auditor_analysis.json")
    
    print("\n2. QUICK OVERVIEW")
    print("-" * 40)
    print("Get a quick overview of your data:")
    print("python quick_scan_logs.py outputs/detailed_analysis/auditor_analysis.json --mode overview")
    
    print("\n3. SUMMARY ANALYSIS")
    print("-" * 40)
    print("See win/loss patterns and consistency:")
    print("python quick_scan_logs.py outputs/detailed_analysis/auditor_analysis.json --mode summary")
    
    print("\n4. INTERACTIVE BROWSING")
    print("-" * 40)
    print("Browse detailed reasoning interactively:")
    print("python quick_scan_logs.py outputs/detailed_analysis/auditor_analysis.json --mode browse")
    print("# Then use commands like:")
    print("#   > list")
    print("#   > pair 0 1")
    print("#   > dim Factual_Correctness")
    print("#   > quit")
    
    print("\n5. COMPREHENSIVE ANALYSIS")
    print("-" * 40)
    print("Full analysis with all views:")
    print("python analyze_detailed_logs.py --data_path outputs/detailed_analysis/auditor_analysis.json")
    
    print("\n6. SPECIFIC ANALYSIS VIEWS")
    print("-" * 40)
    print("Focus on specific aspects:")
    print("# Agent profiles only")
    print("python analyze_detailed_logs.py --data_path <path> --view profiles")
    print("")
    print("# Consistency analysis only")
    print("python analyze_detailed_logs.py --data_path <path> --view consistency")
    print("")
    print("# Specific dimension analysis")
    print("python analyze_detailed_logs.py --data_path <path> --view dimensions --dimension 'Value_Alignment'")
    print("")
    print("# Detailed reasoning for specific pair")
    print("python analyze_detailed_logs.py --data_path <path> --view reasoning --agent_a 0 --agent_b 1")
    
    print("\n7. EXPORT OPTIONS")
    print("-" * 40)
    print("Export to Excel for further analysis:")
    print("python analyze_detailed_logs.py --data_path <path> --export_excel auditor_analysis.xlsx")
    print("")
    print("Save text report:")
    print("python analyze_detailed_logs.py --data_path <path> --save_report auditor_report.txt")
    
    print("\n8. TYPICAL ANALYSIS WORKFLOW")
    print("-" * 40)
    print("# Step 1: Quick overview")
    print("python quick_scan_logs.py <data.json> --mode summary")
    print("")
    print("# Step 2: Identify interesting patterns")
    print("python analyze_detailed_logs.py --data_path <data.json> --view consistency")
    print("")
    print("# Step 3: Deep dive on specific cases")
    print("python analyze_detailed_logs.py --data_path <data.json> --view reasoning --agent_a 0 --agent_b 2")
    print("")
    print("# Step 4: Export for detailed analysis")
    print("python analyze_detailed_logs.py --data_path <data.json> --export_excel full_analysis.xlsx")

def check_sample_data():
    """Check if sample data exists and suggest how to generate it."""
    
    sample_paths = [
        "outputs/detailed_analysis/",
        "run_logs/",
        "run_logs/debug_runs/"
    ]
    
    print("\n" + "="*70)
    print("CHECKING FOR SAMPLE DATA")
    print("="*70)
    
    found_files = []
    for path in sample_paths:
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.json')]
            if files:
                found_files.extend([os.path.join(path, f) for f in files])
    
    if found_files:
        print("Found sample data files:")
        for file in found_files[:5]:  # Show first 5
            print(f"  • {file}")
        if len(found_files) > 5:
            print(f"  ... and {len(found_files) - 5} more files")
        
        print(f"\nYou can test the analysis tools with:")
        print(f"python quick_scan_logs.py {found_files[0]} --mode summary")
    else:
        print("No sample data found.")
        print("\nTo generate sample data:")
        print("1. Run: python test_primary_sources.py")
        print("2. Then: python detailed_source_analysis.py --input_file <generated_file> --source_to_test auditor")

def demonstrate_quick_analysis():
    """Show a quick analysis demonstration if data exists."""
    
    # Look for any existing detailed analysis files
    sample_paths = [
        "outputs/detailed_analysis/",
        "run_logs/detailed_analysis/"
    ]
    
    sample_file = None
    for path in sample_paths:
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.json')]
            if files:
                sample_file = os.path.join(path, files[0])
                break
    
    if sample_file:
        print("\n" + "="*70)
        print("RUNNING QUICK DEMONSTRATION")
        print("="*70)
        print(f"Using sample file: {sample_file}")
        
        try:
            # Import and run quick analysis
            sys.path.append('.')
            from quick_scan_logs import load_data, print_overview
            
            raw_data, pair_data, agent_pairs, dimensions, n_runs = load_data(sample_file)
            print_overview(pair_data, agent_pairs, dimensions, n_runs)
            
            print(f"\nFor more detailed analysis, run:")
            print(f"python quick_scan_logs.py {sample_file} --mode summary")
            
        except Exception as e:
            print(f"Could not run demonstration: {e}")
            print(f"Try running manually: python quick_scan_logs.py {sample_file} --mode overview")
    else:
        print("\n" + "="*70)
        print("NO SAMPLE DATA FOR DEMONSTRATION")
        print("="*70)
        print("Generate some data first by running your detailed analysis.")

def main():
    """Main function to show usage examples and check for sample data."""
    
    print_usage_examples()
    check_sample_data()
    
    # Ask if user wants to see a demonstration
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demonstrate_quick_analysis()
    else:
        print("\n" + "="*70)
        print("Add --demo to this command to run a quick demonstration if sample data exists.")
        print("="*70)

if __name__ == '__main__':
    main() 