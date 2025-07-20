import argparse
import json
import os
import numpy as np
from collections import defaultdict
import copy
import pandas as pd
from scipy.stats import spearmanr
import ipdb
import sys
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# To load saved simulation data
from debug_analysis.test_primary_sources import load_and_recreate

# Import the information source classes we want to analyze
from trust_market.auditor import AuditorWithProfileAnalysis
from trust_market.regulator import Regulator
from trust_market.user_rep import UserRepresentativeWithHolisticEvaluation

def run_single_source_evaluation(source, simulation_data, n_runs=10):
    """
    Runs an information source's evaluation process multiple times to check for variance.

    Args:
        source: An instantiated information source object (e.g., Auditor).
        simulation_data: The loaded simulation output containing conversations etc.
        n_runs: The number of times to run the evaluation.

    Returns:
        A list of analysis data dictionaries, one for each run.
    """
    print(f"\\n--- Running variance analysis for source: {source.source_id} ({n_runs} runs) ---")
    
    all_runs_analysis_data = []
    
    # The source needs agent profiles. Let's assume they are already in the recreated_system
    if hasattr(source, 'add_agent_profile') and not source.agent_profiles:
        for i, profile in enumerate(simulation_data['agent_profiles_used']):
            source.add_agent_profile(i, profile)
    # The source needs access to conversation data from the simulation
    # We can add all conversations from the loaded data to the source instance
    # This assumes the source has an `add_conversation` method.
    if hasattr(source, 'add_conversation'):
        for round_output in simulation_data["simulation_outputs"]:
            for conv_data in round_output["conversation_data"]:
                source.add_conversation(
                    conversation_history=conv_data['history'],
                    user_id=conv_data['user_id'],
                    agent_id=conv_data['agent_id']
                )
                if 'agent_b_id' in conv_data: # For comparative
                    source.add_conversation(
                        conversation_history=conv_data['history_b'],
                        user_id=conv_data['user_id'],
                        agent_id=conv_data['agent_b_id']
                    )

    for i in range(n_runs):
        print(f"  Run {i+1}/{n_runs}...")
        # We need to decide what 'evaluation_round' to pass. For consistency, let's use a fixed one.
        # We also create a deepcopy of the source for each run to ensure independence
        source_copy = copy.deepcopy(source)
        
        # Reset evaluation state to ensure independence between runs
        if hasattr(source_copy, 'reset_evaluation_state'):
            source_copy.reset_evaluation_state()
        
        investments, analysis_data = source_copy.decide_investments(
            evaluation_round=1, 
            use_comparative=True, 
            analysis_mode=True
        )
        all_runs_analysis_data.append(analysis_data)
        
    return all_runs_analysis_data


def analyze_source_variance(analysis_runs):
    """
    Analyzes the variance in the outputs from run_single_source_evaluation.

    Args:
        analysis_runs: A list of analysis data dictionaries.

    Returns:
        A dictionary with variance and mean for key metrics.
    """
    if not analysis_runs:
        return {}

    # metrics will be like { (agent_id, dim): { 'p_target_effective': [run1_val, run2_val, ...], ... } }
    metrics = defaultdict(lambda: defaultdict(list))
    
    source_type = "auditor" # or "user_rep"
    if "delta_value_target_map" in analysis_runs[0]:
        source_type = "regulator"

    for run_data in analysis_runs:
        if source_type == "regulator":
            # Handle regulator data structure
            for agent_id, evals in run_data['own_evaluations'].items():
                for dim, (score, conf) in evals.items():
                    key = (agent_id, dim)
                    metrics[key]['score'].append(score)
                    metrics[key]['confidence'].append(conf)
            # We could also track variance in delta_value_target_map if needed
        else:
            # Handle auditor/user_rep data structure
            for agent_id, dim_data in run_data.items():
                for dim, values in dim_data.items():
                    key = (str(agent_id), dim) # Ensure agent_id is a string for consistent keys
                    for metric_name, value in values.items():
                        metrics[key][metric_name].append(value)

    summary_stats = defaultdict(dict)
    ipdb.set_trace()
    for (agent_id, dim), metric_values in metrics.items():
        for metric_name, values_list in metric_values.items():
            mean = np.mean(values_list)
            variance = np.var(values_list)
            std_dev = np.std(values_list)
            summary_stats[(agent_id, dim)][metric_name] = {
                'mean': mean,
                'variance': variance,
                'std_dev': std_dev
            }
            
    return summary_stats

def aggregate_comparative_elo(comparative_feedback, k_factor_base=32, initial_elo=1200):
    """
    Aggregates pairwise comparative feedback into Elo ratings.
    """
    print("\\n--- Aggregating comparative feedback using Elo ratings ---")
    
    # { (agent_id, dimension): elo_score }
    elo_ratings = defaultdict(lambda: initial_elo)
    
    # Get all dimensions from the first feedback item
    if not comparative_feedback:
        return {}
    dimensions = list(comparative_feedback[0]['winners'].keys())

    def get_expected_score(rating_a, rating_b):
        return 1 / (1 + 10**((rating_b - rating_a) / 400))

    for comparison in comparative_feedback:
        agent_a_id = str(comparison['agent_a_id'])
        agent_b_id = str(comparison['agent_b_id'])
        
        for dim in dimensions:
            winner, confidence = comparison['winners'][dim]
            
            # Get current ratings
            rating_a = elo_ratings[(agent_a_id, dim)]
            rating_b = elo_ratings[(agent_b_id, dim)]
            
            # Determine actual scores
            if winner == 'A':
                score_a, score_b = 1.0, 0.0
            elif winner == 'B':
                score_a, score_b = 0.0, 1.0
            else: # Tie
                score_a, score_b = 0.5, 0.5
                
            # Scale K-factor by confidence (1-5 scale -> 0.2 to 1.0 multiplier)
            k_factor = k_factor_base * (confidence / 5.0)
            
            # Calculate expected scores
            expected_a = get_expected_score(rating_a, rating_b)
            expected_b = get_expected_score(rating_b, rating_a)
            
            # Update Elo ratings
            new_rating_a = rating_a + k_factor * (score_a - expected_a)
            new_rating_b = rating_b + k_factor * (score_b - expected_b)
            
            elo_ratings[(agent_a_id, dim)] = new_rating_a
            elo_ratings[(agent_b_id, dim)] = new_rating_b

    # Format the output for better readability
    final_scores = defaultdict(dict)
    for (agent_id, dim), score in elo_ratings.items():
        final_scores[agent_id][dim] = score
        
    return final_scores

def aggregate_comparative_magnitude(comparative_feedback):
    """
    Aggregates pairwise comparative feedback using a direct magnitude/score system.
    """
    print("\\n--- Aggregating comparative feedback using Magnitude scores ---")
    
    # { (agent_id, dimension): score }
    magnitude_scores = defaultdict(float)
    
    if not comparative_feedback:
        return {}
    dimensions = list(comparative_feedback[0]['winners'].keys())

    for comparison in comparative_feedback:
        agent_a_id = str(comparison['agent_a_id'])
        agent_b_id = str(comparison['agent_b_id'])
        
        for dim in dimensions:
            winner, confidence = comparison['winners'][dim]
            
            # Score change is weighted by confidence
            score_change = confidence / 5.0 # Normalizes confidence to a 0.2-1.0 scale
            
            if winner == 'A':
                magnitude_scores[(agent_a_id, dim)] += score_change
            elif winner == 'B':
                magnitude_scores[(agent_b_id, dim)] += score_change
            # Ties result in no score change

    # Format the output for better readability
    final_scores = defaultdict(dict)
    for (agent_id, dim), score in magnitude_scores.items():
        final_scores[agent_id][dim] = score
        
    return final_scores

def evaluate_info_sources(recreated_system, simulation_data, n_runs=10):
    """
    Evaluates the information sources in the recreated system.
    """
    variance_results = {}
    sources_to_analyze = {
        "Auditor": recreated_system.information_sources["auditor_main"],
        "Regulator": recreated_system.information_sources["regulator"],
        "UserRep": recreated_system.information_sources["user_rep_general"],
    }

    for name, source_instance in sources_to_analyze.items():
        # Run the repetitive evaluation
        analysis_runs = run_single_source_evaluation(source_instance, simulation_data, n_runs=args.n_runs)
        
        # Analyze the variance
        variance_stats = analyze_source_variance(analysis_runs)
        variance_results[name] = variance_stats

        # Print summary
        print(f"\\n--- Variance Summary for {name} ---")
        if not variance_stats:
            print("  No stats generated.")
            continue
        for (agent_id, dim), stats in sorted(variance_stats.items()):
            print(f"  Agent {agent_id}, Dimension {dim}:")
            for metric, values in stats.items():
                print(f"    - {metric}: Mean={values['mean']:.4f}, StdDev={values['std_dev']:.4f}")

def aggregate_user_feedback(simulation_data):
    """
    Aggregates user feedback to create baselines.
    """
    all_comparative_feedback = []
    for round_output in simulation_data["simulation_outputs"]:
        if round_output.get("comparative_winners"):
            all_comparative_feedback.extend(round_output["comparative_winners"])

    if all_comparative_feedback:
        elo_baseline = aggregate_comparative_elo(all_comparative_feedback)
        magnitude_baseline = aggregate_comparative_magnitude(all_comparative_feedback)
        print("\\nUser Baselines:")
        print("Elo:", json.dumps(elo_baseline, indent=2))
        print("Magnitude:", json.dumps(magnitude_baseline, indent=2))
    else:
        print("\\nNo comparative user feedback found to generate baselines.")
        elo_baseline = None
        magnitude_baseline = None

def compare_source_predictions_to_baselines(variance_results, elo_baseline, magnitude_baseline):
    """
    Compares source predictions to baselines.
    """
    print("\\n--- Comparison of Source Predictions vs. User Baselines ---")
    
    # Structure data for comparison
    comparison_data = []
    all_dims = set()
    all_agents = set()

    for source_name, v_results in variance_results.items():
        for (agent_id, dim), stats in v_results.items():
            all_agents.add(agent_id)
            all_dims.add(dim)

    for agent_id in all_agents:
        for dim in all_dims:
            row = {'agent_id': agent_id, 'dimension': dim}
            
            # Get professional source means
            auditor_stats = variance_results.get("Auditor", {}).get((agent_id, dim), {})
            row['auditor_target_price'] = auditor_stats.get('p_target_effective', {}).get('mean', np.nan)

            regulator_stats = variance_results.get("Regulator", {}).get((agent_id, dim), {})
            row['regulator_score'] = regulator_stats.get('score', {}).get('mean', np.nan)

            userrep_stats = variance_results.get("UserRep", {}).get((agent_id, dim), {})
            row['userrep_target_price'] = userrep_stats.get('p_target_effective', {}).get('mean', np.nan)

            # Get user baseline scores
            if elo_baseline:
                row['user_elo'] = elo_baseline.get(agent_id, {}).get(dim, np.nan)
            if magnitude_baseline:
                row['user_magnitude'] = magnitude_baseline.get(agent_id, {}).get(dim, np.nan)

            comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    print("\\n--- Combined Data for Analysis ---")
    print(df.to_string())

    # Calculate rank correlations
    print("\\n--- Spearman Rank Correlation Matrix ---")
    
    # Select only the numeric columns for correlation
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr_matrix = df[numeric_cols].corr(method='spearman')
    
    print(corr_matrix.to_string())
   
def run_full_analysis(args):
    """
    Main function to run the full analysis pipeline.
    """
    print(f"--- Starting Full Analysis for {args.input_file} ---")

    # 1. Load data and recreate the system state
    recreated_system, recreated_sim = load_and_recreate(args.input_file)
    if not recreated_system:
        print("Failed to recreate system. Exiting.")
        return

    with open(args.input_file, "r") as f:
        simulation_data = json.load(f)

    # 2. Run variance analysis for each major information source
    variance_results = evaluate_info_sources(recreated_system, simulation_data, n_runs=args.n_runs)

    # # 3. Aggregate user feedback to create baselines
    # elo_baseline, magnitude_baseline = aggregate_user_feedback(simulation_data)

    # # 4. Compare source predictions to baselines (Phase 3)
    # compare_source_predictions_to_baselines(variance_results, elo_baseline, magnitude_baseline)

    # print("\\nAnalysis Complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run analysis on saved trust market simulation data.")
    parser.add_argument("--input_file", type=str, default="run_logs/debug_runs/primary_source_test_20250621_132632.json", help="Path to the saved simulation JSON file.")
    parser.add_argument("--n_runs", type=int, default=5, help="Number of evaluation runs for variance analysis.")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found at {args.input_file}")
    else:
        run_full_analysis(args) 