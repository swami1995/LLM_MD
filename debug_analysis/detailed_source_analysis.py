import argparse
import json
import os
import copy
from collections import defaultdict
import pandas as pd
import numpy as np

# To load saved simulation data
from test_primary_sources import load_and_recreate
from trust_market.info_sources import InformationSource
from analyze_detailed_logs import analyze_and_print_results as print_analysis

def run_detailed_source_evaluation(source: InformationSource, simulation_data: dict, n_runs: int):
    """
    Runs an information source's evaluation process multiple times to collect detailed
    comparison data, including raw LLM reasoning for each comparison.
    """
    print(f"\n--- Running detailed analysis for source: {source.source_id} ({n_runs} runs) ---")
    
    all_runs_data = []

    # --- Prepare source with data from simulation ---
    if hasattr(source, 'add_agent_profile') and not source.agent_profiles:
        for i, profile in enumerate(simulation_data['agent_profiles_used']):
            source.add_agent_profile(i, profile)

    if hasattr(source, 'add_conversation'):
        # Clear any previous conversations if they exist
        if hasattr(source, 'conversation_histories_by_agent'):
             source.conversation_histories_by_agent.clear()

        for round_output in simulation_data["simulation_outputs"]:
            for conv_data in round_output["conversation_data"]:
                source.add_conversation(
                    conversation_history=conv_data['history'],
                    user_id=conv_data['user_id'],
                    agent_id=conv_data['agent_id']
                )
                if 'agent_b_id' in conv_data: # For comparative feedback
                    source.add_conversation(
                        conversation_history=conv_data['history_b'],
                        user_id=conv_data['user_id'],
                        agent_id=conv_data['agent_b_id']
                    )

    # --- Run evaluations and collect detailed data ---
    for i in range(n_runs):
        print(f"  Run {i+1}/{n_runs}...")
        # Create a deepcopy for each run to ensure independence
        source_copy = copy.deepcopy(source)
        
        # Ensure the evaluation state is reset
        if hasattr(source_copy, 'reset_evaluation_state'):
            source_copy.reset_evaluation_state()
        
        # We need a method that performs the evaluation and returns detailed logs.
        # This will be achieved by modifying `decide_investments` to accept a `detailed_analysis` flag
        # which will be passed down to `evaluate_agents_batch`.
        # Let's call `decide_investments` in a way that triggers this.
        _, analysis_data = source_copy.decide_investments(
            evaluation_round=i, # Use run index as a mock evaluation round
            use_comparative=True, 
            analysis_mode=True,
            detailed_analysis=True # This is the new flag we will add
        )

        run_summary = {
            'run_id': i,
            'comparison_details': []
        }

        agent_profiles_map = {p['id']: p for p in simulation_data['agent_profiles_used']} if simulation_data['agent_profiles_used'] and 'id' in simulation_data['agent_profiles_used'][0] else {i: p for i, p in enumerate(simulation_data['agent_profiles_used'])}


        # The `comparison_log` will be populated by the modified `evaluate_agents_batch`
        if 'comparison_log' in analysis_data:
            for log_entry in analysis_data['comparison_log']:
                agent_a_id, agent_b_id = log_entry['pair']
                try:
                    reasoning = [log_entry['raw_results'][dim]['reasoning'] for dim in log_entry['raw_results'].keys()]
                    raw_confidence = [log_entry['raw_results'][dim]['confidence'] for dim in log_entry['raw_results'].keys()]
                    if 'winner' in log_entry['raw_results']['Communication_Quality']:
                        raw_winner = [log_entry['raw_results'][dim]['winner'] for dim in log_entry['raw_results'].keys()]
                    else:
                        raw_winner = None
                    if 'rating' in log_entry['raw_results']['Communication_Quality']:
                        raw_scores = [log_entry['raw_results'][dim]['rating'] for dim in log_entry['raw_results'].keys()]
                    else:   
                        raw_scores = None
                except:
                    import ipdb; ipdb.set_trace()
                run_summary['comparison_details'].append({
                    'pair': [agent_a_id, agent_b_id],
                    'agent_a_profile': agent_profiles_map.get(agent_a_id, "Profile not found"),
                    'agent_b_profile': agent_profiles_map.get(agent_b_id, "Profile not found"),
                    'derived_scores': log_entry['derived_scores'],
                    'confidences': log_entry['confidences'],
                    'raw_reasoning': reasoning,
                    'raw_confidence': raw_confidence,
                    'raw_scores': raw_scores,
                    'raw_winner': raw_winner,
                })
        
        all_runs_data.append(run_summary)
        
    return all_runs_data


def save_detailed_analysis(data: list, output_path: str):
    """Saves the detailed analysis to a JSON file."""
    print(f"\n--- Saving detailed analysis to {output_path} ---")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print("Save complete.")


def main():
    parser = argparse.ArgumentParser(description="Run detailed analysis on a specific information source.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the saved simulation JSON file.")
    parser.add_argument("--source_to_test", type=str, required=True, choices=['auditor', 'regulator', 'user_rep'], help="The information source to test.")
    parser.add_argument("--n_runs", type=int, default=5, help="Number of evaluation runs for variance analysis.")
    parser.add_argument("--output_file", type=str, default="run_logs/detailed_analysis/analysis_results.json", help="Path to save the detailed analysis results.")
    
    args = parser.parse_args()

    # --- 1. Load data and recreate the system state ---
    print(f"--- Starting Detailed Analysis for {args.input_file} on source '{args.source_to_test}' ---")
    recreated_system, _ = load_and_recreate(args.input_file)
    if not recreated_system:
        print("Failed to recreate system. Exiting.")
        return

    with open(args.input_file, "r", encoding='utf-8') as f:
        simulation_data = json.load(f)

    # --- 2. Get the specific source instance ---
    source_map = {
        "auditor": "auditor_main",
        "regulator": "regulator",
        "user_rep": "user_rep_general"
    }
    source_id = source_map.get(args.source_to_test)
    if not source_id or source_id not in recreated_system.information_sources:
        print(f"Error: Source '{args.source_to_test}' (id: {source_id}) not found in the recreated system.")
        return
        
    source_instance = recreated_system.information_sources[source_id]

    # --- 3. Run the detailed evaluation ---
    detailed_data = run_detailed_source_evaluation(source_instance, simulation_data, args.n_runs)

    # --- 4. Save the results ---
    output_filename = args.output_file
    if 'analysis_results.json' in output_filename: # Use a more descriptive default name
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"outputs/detailed_analysis/{args.source_to_test}_analysis_v2_ratingsconfidence_dimdescfixed_{timestamp}.json"

    save_detailed_analysis(detailed_data, output_filename)

    text_output_path = os.path.splitext(output_filename)[0] + ".txt"

    print_analysis(output_filename, text_output_path)
    print(f"\nDetailed analysis complete. Results saved to {output_filename} and {text_output_path}")


if __name__ == '__main__':
    main() 