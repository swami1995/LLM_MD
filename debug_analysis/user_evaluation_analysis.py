import argparse
import json
import os
import copy
from collections import defaultdict
import pandas as pd
import numpy as np
import logging
import sys
import random
import ipdb
import pickle

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from debug_analysis.test_primary_sources import load_and_recreate
from debug_analysis.detailed_source_analysis import load_and_recreate_source_memory
from trust_market.trust_market_system import TrustMarketSystem

from info_agent import CustomerSupportModel

from debug_analysis.analyze_user_detailed_logs import analyze_and_print_results as print_analysis

# To load saved simulation data
from trust_market.trust_market import TrustMarket

def run_detailed_user_evaluation(customer_support_sim: CustomerSupportModel, simulation_data: dict,
                                 n_runs: int, evaluation_round: int = 20):
    """
    Runs the user agent evaluation process multiple times to collect detailed comparison data.
    """   
    print(f"\n--- Running detailed analysis for User Agents ({n_runs} runs) ---")
    
    all_runs_data = []

    # 1. Extract all comparative conversations from simulation data
    comparative_conversations = []
    for round_output in simulation_data.get("simulation_outputs", []):
        for conv_data in round_output.get("conversation_data", []):
            if 'agent_b_id' in conv_data and conv_data.get('history_b'):
                comparative_conversations.append(conv_data)
    
    if not comparative_conversations:
        print("No comparative conversations with history found in the simulation data.")
        return []

    # Prepare arguments for rate_conversation_batch. These are constant across runs.
    # comparative_conversations = comparative_conversations[17:20]
    histories_a = [c['history'] for c in comparative_conversations]
    histories_b = [c['history_b'] for c in comparative_conversations]
    agent_a_ids = [c['agent_id'] for c in comparative_conversations]
    agent_b_ids = [c['agent_b_id'] for c in comparative_conversations]
    user_ids = [c['user_id'] for c in comparative_conversations]
    conversation_ids = [c['conversation_id'] for c in comparative_conversations]
    
    # --- Run evaluations and collect detailed data ---
    for i in range(n_runs):
        print(f"  Run {i+1}/{n_runs}...")
        
        # This will make fresh LLM calls for each conversation
        evaluations = customer_support_sim.user_agents.rate_conversation_batch(
            conversation_histories=histories_a,
            agent_ids=agent_a_ids,
            user_ids=user_ids,
            conversation_ids=conversation_ids,
            comparison_agent_ids=agent_b_ids,
            comparison_histories=histories_b,
            evaluation_round=evaluation_round,
            analysis_mode=True
        )
        
        # Structure the data to match what detailed_source_analysis produces
        run_summary = {
            'run_id': i,
            'comparison_details': []
        }
        
        # import ipdb; ipdb.set_trace()
        agent_profiles_map = {p['id']: p for p in simulation_data['agent_profiles_used']} if simulation_data['agent_profiles_used'] and 'id' in simulation_data['agent_profiles_used'][0] else {i: p for i, p in enumerate(simulation_data['agent_profiles_used'])}

        for eval_item in evaluations:
            agent_a_id = eval_item['agent_a_id']
            agent_b_id = eval_item['agent_b_id']
            
            # This is a bit of a hack to match the structure of the other analysis file
            # It creates a log entry for each evaluated conversation
            log_entry = {
                'pair': [agent_a_id, agent_b_id],
                'agent_a_profile': agent_profiles_map.get(agent_a_id, "Profile not found"),
                'agent_b_profile': agent_profiles_map.get(agent_b_id, "Profile not found"),
                # 'raw_reasoning': [eval_item['reasoning']] * len(eval_item['winners']), # Repeat reasoning for each dim
                # 'derived_scores': {dim: 1 if winner == 'A' else (-1 if winner == 'B' else 0) for dim, (winner, _) in eval_item['winners'].items()},
                'derived_scores': {dim : eval_item['winners'][dim]['rating'] for dim in eval_item['winners']},
                # 'confidences': {dim: conf for dim, (_, conf) in eval_item['winners'].items()},
                'confidences': {dim: eval_item['winners'][dim]['confidence'] for dim in eval_item['winners']},
                'reasoning': {dim: eval_item['winners'][dim]['reasoning'] for dim in eval_item['winners']},
                # 'raw_winner': [winner for dim, (winner, conf) in eval_item['winners'].items()],
                # 'raw_confidence': [conf for dim, (winner, conf) in eval_item['winners'].items()],
                'raw_scores': None # Not available from user eval
            }
            run_summary['comparison_details'].append(log_entry)

            # import ipdb; ipdb.set_trace()
            # if agent_a_id == 1 and agent_b_id == 2 and eval_item['winners']['Value_Alignment'] == 'B':
            #     import ipdb; ipdb.set_trace()
            
        all_runs_data.append(run_summary)
        
    return all_runs_data

def save_detailed_analysis(data: list, output_path: str):
    """Saves the detailed analysis to a JSON file."""
    print(f"\n--- Saving detailed analysis to {output_path} ---")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print("Save complete.")

def load_and_recreate_user_memory(recreated_system: TrustMarketSystem, simulation_data: dict, evaluation_rounds: list[int] = [20,21,22, 23]):
    """
    Loads and recreates the user memory for the user agent evaluations.
    It ensures that for each evaluation round, there's one instance of comparison 
    for each pair of agents for each user
    """

    if os.path.exists(f"run_logs/debug_runs/source_memory/users_memory_indep_v2.pkl"):
        with open(f"run_logs/debug_runs/source_memory/users_memory_indep_v2.pkl", "rb") as f:
            user_mems = pickle.load(f)
        return user_mems
    else:
        print(f"User memory not found for evaluation round {evaluation_rounds[-1]}. Recreating...")

    print(f"\n--- Recreating User Memory across rounds {evaluation_rounds} ---")
    
    user_agents = recreated_system.simulation_module.user_agents
    user_mems = []

    # 1. Extract and group all comparative conversations by user and agent pair.
    convs_by_user_and_pair = defaultdict(lambda: defaultdict(list))
    for round_output in simulation_data.get("simulation_outputs", []):
        for conv_data in round_output.get("conversation_data", []):
            if 'agent_b_id' in conv_data and conv_data.get('history_b'):
                user_id = conv_data['user_id']
                agent_a_id = conv_data['agent_id']
                agent_b_id = conv_data['agent_b_id']
                pair = tuple(sorted((agent_a_id, agent_b_id)))
                convs_by_user_and_pair[user_id][pair].append(conv_data)

    if not convs_by_user_and_pair:
        print("No comparative conversations with history found. Cannot recreate user memory.")
        return []

    # 2. For each user and pair, sample one conversation. This gives us our pool of unique comparisons.
    unique_comparisons = []
    for (round_num, evaluation_round) in enumerate(evaluation_rounds):
        unique_comparison = []
        for user_id, pairs in convs_by_user_and_pair.items():
            for pair, convs in pairs.items():
                try:
                    chosen_conv = convs[round_num]
                    unique_comparison.append(chosen_conv)
                except IndexError:
                    print(f"  No conversation found for user {user_id} and pair {pair} at round {evaluation_round}")
                    ipdb.set_trace()
        unique_comparisons.append(unique_comparison)

    print(f"  Found {len(unique_comparisons)} unique user-agent pair conversations to sample from.")

    # 3. Iterate through evaluation rounds, processing batches and building up memory for each round.
    for round_num, evaluation_round in enumerate(evaluation_rounds):
        print(f"  Memory Generation for Evaluation Round {evaluation_round}...")
            # Prepare arguments for rate_conversation_batch. These are constant across runs.
        histories_a = [c['history'] for c in unique_comparisons[round_num]]
        histories_b = [c['history_b'] for c in unique_comparisons[round_num]]
        agent_a_ids = [c['agent_id'] for c in unique_comparisons[round_num]]
        agent_b_ids = [c['agent_b_id'] for c in unique_comparisons[round_num]]
        user_ids = [c['user_id'] for c in unique_comparisons[round_num]]
        conversation_ids = [c['conversation_id'] for c in unique_comparisons[round_num]]
        
        # Clear memory at the start of building a new memory state.
        # user_agents.user_evaluations.clear()

        user_agents.rate_conversation_batch(
            conversation_histories=histories_a,
            agent_ids=agent_a_ids,
            user_ids=user_ids,
            conversation_ids=conversation_ids,
            comparison_agent_ids=agent_b_ids,
            comparison_histories=histories_b,
            evaluation_round=evaluation_round,
            analysis_mode=False 
        )

        # After processing all batches for this round, store a copy of the memory.
    user_mems = copy.deepcopy(user_agents.user_evaluations)

    # Save the memory to a file
    with open(f"run_logs/debug_runs/source_memory/users_memory_indep_v2.pkl", "wb") as f:
        pickle.dump(user_mems, f)

    # Clear memory one last time to leave it in a clean state.
    user_agents.user_evaluations.clear()
    
    print(f"--- Finished recreating user memory. Generated {len(user_mems)} memory states. ---")
    return user_mems


def run_independent_memory_based_analysis(simulation_data: dict, recreated_system: TrustMarketSystem, args):
    """
    Runs an independent memory-based analysis for the user agent evaluations.
    """
    user_mems = load_and_recreate_user_memory(recreated_system, simulation_data, evaluation_rounds=[20,21,22])
    detailed_data = []
    # for i in range(len(user_mems)):
    recreated_system.user_agents.user_evaluations = user_mems[-1]
    detailed_data = run_detailed_user_evaluation(recreated_system.simulation_module, simulation_data, n_runs=args.n_runs)
    output_path = f"outputs/detailed_analysis/memory_based/users_memory_independent.json"
    save_detailed_analysis(detailed_data, output_path)
    text_output_path = os.path.splitext(output_path)[0] + ".txt"
    print_analysis(output_path, text_output_path)
    return detailed_data

def run_memory_based_analysis(simulation_data: dict, recreated_system: TrustMarketSystem, args):
    """
    Runs a memory-based analysis for the user agent evaluations.
    """
    if args.memory_type == "independent":
        detailed_data = run_independent_memory_based_analysis(simulation_data, recreated_system, args)
        return detailed_data
    elif args.memory_type == "cross":
        detailed_data = run_cross_memory_based_analysis(simulation_data, recreated_system, args)
        return detailed_data
    else:
        raise ValueError(f"Memory type {args.memory_type} not supported.")
    

def run_cross_memory_based_analysis(simulation_data: dict, recreated_system: TrustMarketSystem, args):
    """
    Runs a cross-memory-based analysis for the user agent evaluations.
    """
    reg_mem_idx = 2
    regulator_mems = load_and_recreate_source_memory(recreated_system.information_sources['regulator'])
    recreated_system.information_sources['regulator'].pair_evaluation_memory = regulator_mems[reg_mem_idx]
    user_rep_mems = load_and_recreate_source_memory(recreated_system.information_sources['user_rep_general'])
    recreated_system.information_sources['user_rep_general'].pair_evaluation_memory = user_rep_mems[reg_mem_idx]
    user_mems = load_and_recreate_user_memory(recreated_system, simulation_data, evaluation_rounds=[20,21,22])
    recreated_system.simulation_module.user_agents.user_evaluations = user_mems
    recreated_system.simulation_module.user_agents.market = recreated_system.trust_market
    customer_support_sim = recreated_system.simulation_module
    detailed_data = run_detailed_user_evaluation(customer_support_sim, simulation_data, n_runs=args.n_runs, evaluation_round=26)#user_rep_mems[reg_mem_idx][(0,1)][-1]['round']+10)
    output_path = f"outputs/detailed_analysis/memory_based/users_memory_cross_withRegulator{reg_mem_idx}_v2.json"
    save_detailed_analysis(detailed_data, output_path)
    text_output_path = os.path.splitext(output_path)[0] + ".txt"
    print_analysis(output_path, text_output_path)
    return detailed_data

def run_memory_less_analysis(simulation_data: dict, args):
    """
    Runs a memory-less analysis for the user agent evaluations.
    """
    # --- 1. Instantiate UserAgentSet ---
    user_profiles = simulation_data['user_profiles_used']
    agent_profiles = simulation_data['agent_profiles_used']
    conversation_prompts = simulation_data['conversation_prompts_used']
    static_knowledge_base = simulation_data['static_knowledge_base']
    config = simulation_data.get('config', {})

    customer_support_sim = CustomerSupportModel(
        num_users=len(user_profiles),
        num_agents=len(agent_profiles),
        evaluation_method='comparative_binary', # Still needed for user feedback type
        rating_scale=5,           # Still needed for user feedback type
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        llm_source='api',
        api_provider='gemini',
        agent_profiles=agent_profiles, # Pass the selected profiles
        user_profiles=user_profiles,   # Pass the selected profiles
        conversation_prompts=conversation_prompts,
        static_knowledge_base=static_knowledge_base,
        max_dialog_rounds=1,
        use_chat_api=True,
        api_model_name='gemini-2.5-flash',
    )
    
    # --- 2. Run the detailed evaluation ---
    detailed_data = run_detailed_user_evaluation(customer_support_sim, simulation_data, args.n_runs)

    if not detailed_data:
        print("Analysis finished early as no data was produced.")
        return


    # --- 4. Save the results ---
    if args.output_file:
        output_filename = args.output_file
    else:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"outputs/detailed_analysis/user_agent_analysis_{timestamp}.json"

    save_detailed_analysis(detailed_data, output_filename)
    
    # --- 5. Analyze and print results ---
    text_output_path = os.path.splitext(output_filename)[0] + ".txt"
    print_analysis(output_filename, text_output_path)

    print(f"\nDetailed analysis complete. Results saved to {output_filename} and summary to {text_output_path}")

    return detailed_data


def main():
    parser = argparse.ArgumentParser(description="Run detailed analysis on user agent evaluations.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the saved simulation JSON file.")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of evaluation runs for variance analysis.")
    parser.add_argument("--output_file", type=str, help="Path to save the detailed analysis results.")
    parser.add_argument("--analysis_type", type=str, default="memory_less", choices=["memory_less", "memory_based"], help="Type of analysis to run.")
    parser.add_argument("--memory_type", type=str, default="independent", choices=["independent", "cross"], help="Type of memory to use.")

    args = parser.parse_args()

    if args.output_file and os.path.exists(args.output_file):
        output_filename = args.output_file
        text_output_path = os.path.splitext(output_filename)[0] + ".txt"
        print_analysis(output_filename, text_output_path)
        return        

    with open(args.input_file, "r", encoding='utf-8') as f:
        simulation_data = json.load(f)

    if args.analysis_type == "memory_based":
        print(f"--- Starting Detailed Analysis for {args.input_file} with memory-based analysis ---")
        recreated_system, _ = load_and_recreate(args.input_file, only_models_and_convdata=True)
        detailed_data = run_memory_based_analysis(simulation_data, recreated_system, args)
    else:
        detailed_data = run_memory_less_analysis(simulation_data, args)


if __name__ == '__main__':
    main() 