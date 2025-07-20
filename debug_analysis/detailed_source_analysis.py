import argparse
import json
import os
import copy
from collections import defaultdict
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple, List, Any
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# To load saved simulation data
from debug_analysis.test_primary_sources import load_and_recreate
from trust_market.info_sources import InformationSource
from debug_analysis.analyze_detailed_logs import analyze_and_print_results as print_analysis

def run_and_save_n_runs(source, source_memory_path: str, evaluation_rounds: list[int]):
    """
    Runs an information source's evaluation process multiple times to collect detailed
    comparison data with source memory saved. 
    """
    source_evals = []
    for evaluation_round in evaluation_rounds:
        source_evals.append(source.evaluate_and_get_pair_evaluation_memory(evaluation_round=evaluation_round))
    with open(source_memory_path, 'wb') as f:
        pickle.dump(source_evals, f)
    return source_evals

def load_and_recreate_source_memory(source, source_memory_folder: str = "run_logs/debug_runs/source_memory"):
    """
    Loads the source memory from a file.
    """
    source_memory_path = os.path.join(source_memory_folder, f"{source.source_id}_memory_indep_v2.pkl")
    if os.path.exists(source_memory_path):
        with open(source_memory_path, 'rb') as f:
            source_mems = pickle.load(f)
    else:
        if source.source_id == "regulator":
            evaluation_rounds = [20, 22, 24]
        elif source.source_id == "auditor_main":
            evaluation_rounds = [8, 12, 16]
        elif source.source_id == "user_rep_general":
            evaluation_rounds = [14, 16, 18]
        else:
            raise ValueError(f"Source {source.source_id} not supported.")
        source_mems = run_and_save_n_runs(source, source_memory_path, evaluation_rounds = evaluation_rounds)
        print(f"Saved source memory to {source_memory_path} for {source.source_id} at evaluation rounds {evaluation_rounds}")
    return source_mems
    
def run_independent_memory_based_analysis(source, simulation_data, args, num_memory=3):
    """
    Runs a memory-based analysis for an information source.
    """
    source_mems = load_and_recreate_source_memory(source)
    eval_round = {
        "regulator": 26,
        "auditor_main": 20,
        "user_rep_general": 20
    }
    for i in range(num_memory):
        source_mem = source_mems[i]
        source.pair_evaluation_memory = source_mem 
        detailed_data = run_detailed_source_evaluation(source, simulation_data, n_runs=args.n_runs, evaluation_round=eval_round[source.source_id])   
        output_path = f"outputs/detailed_analysis/memory_based/{source.source_id}_memory_indep_v2_{i}.json"
        save_detailed_analysis(detailed_data, output_path)
        text_output_path = os.path.splitext(output_path)[0] + ".txt"
        print_analysis(output_path, text_output_path)
    return detailed_data

def run_cross_memory_based_analysis(source, recreated_system, simulation_data, args, reg_mem_idx=2):
    """
    Runs a memory-based analysis for an information source. 
    """
    regulator_mems = load_and_recreate_source_memory(recreated_system.information_sources['regulator'])
    recreated_system.information_sources['regulator'].pair_evaluation_memory = regulator_mems[reg_mem_idx]
    source_mems = load_and_recreate_source_memory(source)
    source_mem = source_mems[reg_mem_idx]
    source.pair_evaluation_memory = source_mem 
    detailed_data = run_detailed_source_evaluation(source, simulation_data, n_runs=args.n_runs, evaluation_round=regulator_mems[reg_mem_idx][(0,1)][-1]['round']+2)
    output_path = f"outputs/detailed_analysis/memory_based/{source.source_id}_memory_cross_withRegulator{reg_mem_idx}_v2.json"
    save_detailed_analysis(detailed_data, output_path)
    text_output_path = os.path.splitext(output_path)[0] + ".txt"
    print_analysis(output_path, text_output_path)
    return detailed_data

def run_memory_based_analysis(sources, recreated_system, simulation_data, args):
    """
    Runs a memory-based analysis for an information source.
    """

    if args.memory_type == "independent":
        for source in sources:
            run_independent_memory_based_analysis(source, simulation_data, args, num_memory=3)
            source.pair_evaluation_memory = defaultdict(list)
    elif args.memory_type == "cross":
        for source in sources:
            run_cross_memory_based_analysis(source, recreated_system, simulation_data, args, reg_mem_idx=2)
            source.pair_evaluation_memory = defaultdict(list)
    else:
        raise ValueError(f"Memory type {args.memory_type} not supported.")

def run_memory_based_analysis_sequential(sources, recreated_system, simulation_data, args):
    num_iters = 30
    regulator_freq = 12  # Frequency of regulator evaluations
    auditor_freq = 3  # Frequency of auditor evaluations
    user_rep_freq = 2  # Frequency of user_rep evaluations
    if os.path.exists("run_logs/debug_runs/source_memory_coupled/source_mems_3.pkl"):
        with open("run_logs/debug_runs/source_memory_coupled/source_mems_3.pkl", 'rb') as f:
            source_mems = pickle.load(f)
        plot_ratings_across_rounds(source_mems, sources, num_iters)
        return None
    ### Check if all the conversation data from simulation data has been fed to user_rep and regulator for evaluation

    ### check if regulator and auditor have the agent profiles for each agent

    for i in range(num_iters):
        print(f"Round {i+1}/{num_iters}...")
        
        if (i+1) % regulator_freq == 0:
            print(f"  Running regulator evaluation at round {i+1}")
            recreated_system.information_sources['regulator'].evaluate_and_get_pair_evaluation_memory(
                evaluation_round=i, 
                use_comparative=True
            )
        if (i+1) % auditor_freq == 0:
            print(f"  Running auditor evaluation at round {i+1}")
            recreated_system.information_sources['auditor_main'].evaluate_and_get_pair_evaluation_memory(
                evaluation_round=i, 
                use_comparative=True
            )
        if (i+1) % user_rep_freq == 0:
            print(f"  Running user_rep evaluation at round {i+1}")
            recreated_system.information_sources['user_rep_general'].evaluate_and_get_pair_evaluation_memory(
                evaluation_round=i, 
                use_comparative=True
            )
    

    source_mems = {}
    for source in sources:
        source_mems[source.source_id] = recreated_system.information_sources[source.source_id].pair_evaluation_memory
    # Save the source memories
    source_memory_folder = "run_logs/debug_runs/source_memory_coupled/"
    os.makedirs(source_memory_folder, exist_ok=True)
    with open(os.path.join(source_memory_folder, "source_mems_3.pkl"), 'wb') as f:
        pickle.dump(source_mems, f)
    print(f"Saved source memories to {os.path.join(source_memory_folder, 'source_mems_3.pkl')}")

    # Plot the ratings across evaluation rounds for the sources
    plot_ratings_across_rounds(source_mems, sources, num_iters)

def plot_ratings_across_rounds(source_mems: Dict[str, Dict], sources: List[InformationSource], num_iters: int):
    """
    Plot the evolution of ratings across evaluation rounds for each source.
    Creates plots similar to test_rating_plots_fix.py output.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import combinations
    import os
    
    # Extract dimensions from the actual data (not from source.expertise_dimensions)
    # This ensures we only plot dimensions that actually have data
    all_dimensions = set()
    dimensions_with_data = set()
    
    for source_id, source_data in source_mems.items():
        for pair, evaluations in source_data.items():
            for evaluation in evaluations:
                derived_scores = evaluation.get('derived_scores', {})
                for agent_id, agent_scores in derived_scores.items():
                    if isinstance(agent_scores, dict):
                        all_dimensions.update(agent_scores.keys())
                        # Check if this dimension has comparative data for this pair
                        for dim in agent_scores.keys():
                            pair_tuple = tuple(sorted(pair))
                            if len(pair_tuple) == 2:
                                agent_a_id, agent_b_id = pair_tuple
                                if (agent_a_id in derived_scores and agent_b_id in derived_scores and
                                    isinstance(derived_scores[agent_a_id], dict) and 
                                    isinstance(derived_scores[agent_b_id], dict) and
                                    dim in derived_scores[agent_a_id] and dim in derived_scores[agent_b_id]):
                                    dimensions_with_data.add(dim)
    
    # Only plot dimensions that actually have comparative data
    dimensions = sorted(list(dimensions_with_data))
    print(f"Total dimensions found: {len(all_dimensions)}, with comparative data: {len(dimensions)}")
    print(f"Plotting dimensions: {dimensions}")
    
    # Get agent IDs - assuming 3 agents (0, 1, 2)
    agent_ids = [0, 1, 2]
    agent_pairs = [tuple(sorted(pair)) for pair in combinations(agent_ids, 2)]
    
    # Define colors for different sources
    colors = {
        'user_rep_general': 'blue',
        'regulator': 'red', 
        'auditor_main': 'green'
    }
    
    # Create output directory
    output_dir = "figures/sequential_source_analysis_3"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each agent pair
    for pair in agent_pairs:
        agent_a_id, agent_b_id = pair
        
        # Calculate subplot layout (aim for roughly square grid)
        n_dims = len(dimensions)
        n_cols = int(np.ceil(np.sqrt(n_dims)))
        n_rows = int(np.ceil(n_dims / n_cols))
        
        # Create a large figure with subplots for all dimensions
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Ensure axes is always a 2D array for consistent indexing
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        plotted_any_data = False
        
        # Plot each dimension
        for dim_idx, dimension in enumerate(dimensions):
            row = dim_idx // n_cols
            col = dim_idx % n_cols
            ax = axes[row, col]
            
            dimension_has_data = False
            
            # Plot data for each source
            for source in sources:
                source_id = source.source_id
                if source_id not in source_mems:
                    continue
                    
                pair_memory = source_mems[source_id].get(pair, [])
                if not pair_memory:
                    continue
                
                # Extract ratings across rounds for this dimension
                rounds = []
                ratings = []
                
                for evaluation in pair_memory:
                    round_num = evaluation.get('round', 0)
                    derived_scores = evaluation.get('derived_scores', {})
                    
                    # Get comparative rating for this dimension
                    # We need both agents' scores to compute a comparative rating
                    if (agent_a_id in derived_scores and agent_b_id in derived_scores and 
                        dimension in derived_scores[agent_a_id] and dimension in derived_scores[agent_b_id]):
                        
                        score_a = derived_scores[agent_a_id][dimension]
                        score_b = derived_scores[agent_b_id][dimension]
                        
                        # Convert to comparative rating:
                        # Positive means agent_a is better, negative means agent_b is better
                        comparative_rating = (score_a - 0.5 ) * 10
                        
                        rounds.append(round_num)
                        ratings.append(comparative_rating)
                
                # Plot if we have data
                if rounds and ratings:
                    ax.plot(rounds, ratings, 
                           color=colors.get(source_id, 'gray'), 
                           marker='o', label=source_id, 
                           linewidth=1.5, markersize=4)
                    dimension_has_data = True
            
            if dimension_has_data:
                plotted_any_data = True
            
            # Formatting for each subplot
            ax.set_xlabel('Round', fontsize=9)
            ax.set_ylabel('Rating (A - B)', fontsize=9)
            ax.set_title(f'{dimension}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
            # Set y-axis limits based on actual data range if we have data
            if dimension_has_data:
                # Auto-scale but ensure we show the zero line
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                # Let matplotlib auto-scale, but expand slightly for better visualization
                ax.margins(y=0.1)
            else:
                # For empty plots, set a reasonable default range
                ax.set_ylim(-2, 2)
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10, alpha=0.7)
        
        # Hide empty subplots
        for dim_idx in range(len(dimensions), n_rows * n_cols):
            row = dim_idx // n_cols
            col = dim_idx % n_cols
            axes[row, col].set_visible(False)
        
        # Add legend and title
        if plotted_any_data:
            # Create legend with available sources
            handles, labels = [], []
            for source in sources:
                source_id = source.source_id
                if source_id in colors and source_id in source_mems:
                    handle = plt.Line2D([0], [0], color=colors[source_id], marker='o', 
                                      linewidth=1.5, markersize=4, label=source_id)
                    handles.append(handle)
                    labels.append(source_id)
            
            if handles:
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
                          ncol=len(handles), fontsize=10)
            
            # Add overall title
            fig.suptitle(f'Sequential Source Analysis: Agent {agent_a_id} vs Agent {agent_b_id}\n'
                        f'Positive: Agent {agent_a_id} better, Negative: Agent {agent_b_id} better', 
                        fontsize=14, fontweight='bold', y=0.98)
        else:
            fig.suptitle(f'Sequential Source Analysis: Agent {agent_a_id} vs Agent {agent_b_id} (NO DATA)\n'
                        f'Positive: Agent {agent_a_id} better, Negative: Agent {agent_b_id} better', 
                        fontsize=14, fontweight='bold', y=0.98)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.02, 1, 0.92])
        
        filename = f'sequential_analysis_A{agent_a_id}_vs_A{agent_b_id}_all_dimensions.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved sequential analysis plot: {filepath}")
        plt.close(fig)
    
    print(f"\nSequential source analysis plots saved to {output_dir}/")
    
    # Also save the source memories as JSON for inspection
    json_filepath = os.path.join(output_dir, "source_memories_summary.json")
    
    # Convert source memories to JSON-serializable format
    json_data = {}
    for source_id, memory_dict in source_mems.items():
        json_data[source_id] = {}
        for pair_key, evaluations in memory_dict.items():
            pair_str = f"{pair_key[0]}_{pair_key[1]}"
            json_data[source_id][pair_str] = []
            for eval_data in evaluations:
                # Create a simplified version for JSON
                simplified_eval = {
                    'round': eval_data.get('round', 0),
                    'timestamp': eval_data.get('timestamp', 0),
                    'derived_scores': eval_data.get('derived_scores', {}),
                    'confidences': eval_data.get('confidences', {})
                }
                json_data[source_id][pair_str].append(simplified_eval)
    
    import json
    with open(json_filepath, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved source memories summary to {json_filepath}")


# load data (profile and conversation history) for all agents
# run the agents serially to save memory (disable cross agent memory)
# save the individual memories for each agent
# Run detailed analysis with increasing amount of memory (first regulator and then auditor and user rep) - eventually use regulator memory for other sources. 
# save and check the detailed analysis results

def run_detailed_source_evaluation(source: InformationSource, simulation_data: dict, n_runs: int, evaluation_round: int = None):
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
        if evaluation_round is None:
            evaluation_round = i
        _, analysis_data = source_copy.decide_investments(
            evaluation_round=evaluation_round, # Use run index as a mock evaluation round
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
                    reasoning = ["Error extracting reasoning"]
                    raw_confidence = [0.0]
                    raw_scores = None
                    raw_winner = None
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


def run_memoryless_analysis(source, simulation_data, args):
    """
    Runs a memoryless analysis for an information source.
    """
    detailed_data = run_detailed_source_evaluation(source, simulation_data, n_runs=args.n_runs)

    # --- 4. Save the results ---
    output_filename = args.output_file
    if 'analysis_results.json' in output_filename: # Use a more descriptive default name
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"outputs/detailed_analysis/{args.source_to_test}_analysis_v2_ratingsconfidence_dimdescfixed_{timestamp}.json"

    save_detailed_analysis(detailed_data, output_filename)

    text_output_path = os.path.splitext(output_filename)[0] + ".txt"

    print_analysis(output_filename, text_output_path)
    print(f"\nDetailed analysis complete. Results saved to {output_filename} and {text_output_path}")

    return detailed_data

def main():
    parser = argparse.ArgumentParser(description="Run detailed analysis on a specific information source.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the saved simulation JSON file.")
    parser.add_argument("--source_to_test", type=str, default="all", choices=['auditor', 'regulator', 'user_rep', 'all'], help="The information source to test.")
    parser.add_argument("--n_runs", type=int, default=5, help="Number of evaluation runs for variance analysis.")
    parser.add_argument("--output_file", type=str, default="run_logs/detailed_analysis/analysis_results.json", help="Path to save the detailed analysis results.")
    parser.add_argument("--memory_type", type=str, default="independent", choices=["independent", "cross"], help="Type of analysis to run.")
    parser.add_argument("--analysis_type", type=str, default="memory_less", choices=["memory_less", "memory_based", "memory_based_seq"], help="Type of analysis to run.")
    args = parser.parse_args()

    # --- 1. Load data and recreate the system state ---
    print(f"--- Starting Detailed Analysis for {args.input_file} on source '{args.source_to_test}' ---")
    recreated_system, _ = load_and_recreate(args.input_file, only_models_and_convdata=True)

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

    if args.source_to_test == "all":
        sources = [recreated_system.information_sources[source_map[source]] for source in source_map.keys()]
    else:
        sources = [recreated_system.information_sources[source_map[args.source_to_test]]]

    # --- 3. Run the detailed evaluation ---
    if args.analysis_type == "memory_less":
        for source in sources:
            detailed_data = run_memoryless_analysis(source, simulation_data, args)
    elif args.analysis_type == "memory_based":
        run_memory_based_analysis(sources, recreated_system, simulation_data, args)
    elif args.analysis_type == "memory_based_seq":
        run_memory_based_analysis_sequential(sources, recreated_system, simulation_data, args)


if __name__ == '__main__':
    main() 