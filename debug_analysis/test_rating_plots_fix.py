#!/usr/bin/env python3

import json
import os
from debug_analysis.rating_evolution_plots import process_simulation_logs_and_plot

def test_source_evaluation_processing():
    """Test if source evaluations are being processed correctly."""
    
    # Find a recent log file to test with
    log_dir = "run_logs"
    if not os.path.exists(log_dir):
        print("No run_logs directory found. Please run a simulation first with --save_detailed_logs.")
        return
    
    # Find the most recent log file
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
    if not log_files:
        print("No JSON log files found in run_logs/")
        return
    
    log_files.sort()
    latest_log = os.path.join(log_dir, log_files[-1])
    print(f"Testing with log file: {latest_log}")
    
    # Load the log file to inspect structure
    with open(latest_log, 'r') as f:
        data = json.load(f)
    
    # Check if detailed_evaluations exist
    detailed_evaluations = data.get('detailed_evaluations', [])
    print(f"\nFound {len(detailed_evaluations)} evaluation rounds in detailed_evaluations")
    
    # Inspect the structure of detailed_evaluations
    for i, round_data in enumerate(detailed_evaluations[:2]):  # Look at first 2 rounds
        print(f"\nRound {i}:")
        print(f"  Round number: {round_data.get('round', 'Unknown')}")
        
        source_evals = round_data.get('source_evaluations', {})
        print(f"  Sources with evaluations: {list(source_evals.keys())}")
        
        for source_id, source_data in source_evals.items():
            print(f"    {source_id}:")
            print(f"      Keys: {list(source_data.keys())}")
            
            # Check for comparison_log
            comparison_log = source_data.get('comparison_log', [])
            print(f"      Comparison log entries: {len(comparison_log)}")
            
            if comparison_log:
                first_comparison = comparison_log[0]
                print(f"      First comparison structure: {list(first_comparison.keys())}")
                if 'derived_scores' in first_comparison:
                    derived_scores = first_comparison['derived_scores']
                    print(f"      Derived scores agents: {list(derived_scores.keys())}")
                    if derived_scores:
                        first_agent_scores = list(derived_scores.values())[0]
                        print(f"      First agent score dimensions: {list(first_agent_scores.keys())}")
    
    # Now test the plotting function
    print(f"\nTesting rating evolution plotting...")
    try:
        tracker = process_simulation_logs_and_plot(
            latest_log, 
            save_path="figures/test_rating_fix", 
            experiment_name="source_test"
        )
        
        # Check if source ratings were processed
        print(f"\nSource ratings processed:")
        for source_id in tracker.source_ratings:
            source_data = tracker.source_ratings[source_id]
            total_ratings = sum(len(pair_data[dim]) for pair_data in source_data.values() for dim in pair_data)
            print(f"  {source_id}: {total_ratings} total comparative rating entries")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_source_evaluation_processing() 