import argparse
import json
import os
import copy
from collections import defaultdict
import pandas as pd
import numpy as np

from info_agent import CustomerSupportModel

from analyze_user_detailed_logs import analyze_and_print_results as print_analysis

def run_detailed_user_evaluation(customer_support_sim: CustomerSupportModel, simulation_data: dict, n_runs: int):
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
            comparison_histories=histories_b
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

def main():
    parser = argparse.ArgumentParser(description="Run detailed analysis on user agent evaluations.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the saved simulation JSON file.")
    parser.add_argument("--n_runs", type=int, default=5, help="Number of evaluation runs for variance analysis.")
    parser.add_argument("--output_file", type=str, help="Path to save the detailed analysis results.")
    
    args = parser.parse_args()

    if args.output_file and os.path.exists(args.output_file):
        output_filename = args.output_file
        text_output_path = os.path.splitext(output_filename)[0] + ".txt"
        print_analysis(output_filename, text_output_path)
        return        

    # --- 1. Load data ---
    print(f"--- Starting User Evaluation Analysis for {args.input_file} ---")
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found at {args.input_file}")
        return
        
    with open(args.input_file, "r", encoding='utf-8') as f:
        simulation_data = json.load(f)

    # --- 2. Instantiate UserAgentSet ---
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
    
    # --- 3. Run the detailed evaluation ---
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

if __name__ == '__main__':
    main() 