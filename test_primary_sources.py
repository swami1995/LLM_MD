# from info_agent import CustomerSupportModel # We'll get this via TrustMarketSystem now
import argparse
import os
import random # Import random for sampling
import json # Added for final summary printing
import pickle
import time
from agent_prompting_utils import load_profiles, load_prompts
from collections import defaultdict

# Import Trust Market components
from trust_market.trust_market_system import TrustMarketSystem
from trust_market.trust_market import TrustMarket
from trust_market.auditor import AuditorWithProfileAnalysis # Use the enhanced Auditor
from trust_market.user_rep import UserRepresentativeWithHolisticEvaluation # Use the enhanced UserRep
from trust_market.regulator import Regulator

# Import CustomerSupportModel from info_agent
from info_agent import CustomerSupportModel
import ipdb

# Next step:
# 1. compare the evaluations of regulator vs user rep and auditor.
# 2. check along what axes they can be similar or complementary with user ratings.
# 3. figure out some aggregated statistic for the user ratings to compare with. 
# 4. figure out noise levels in predictions of each of the predictors.
# 5. figure out how to make the noise levels of user ratings higher but noise levels of the other predictors lower.

## main function
def test_primary_sources():
    parser = argparse.ArgumentParser(description="Run the Customer Support Simulation with Trust Market Oversight.")
    # --- Simulation Args ---
    parser.add_argument("--model_path", type=str, default="/data/models/huggingface/meta-llama/Llama-3-8B-Instruct", help="Path to the pretrained LLM model. Required if --llm_source is 'local'.")
    parser.add_argument("--llm_source", type=str, choices=["local", "api"], default="api", help="Source of LLM: 'local' (Llama) or 'api' (Gemini). Default is 'api'.")
    parser.add_argument("--max_dialog_rounds", type=int, default=1, help="Maximum number of dialog rounds for each conversation. Default is 3.")
    parser.add_argument("--num_steps", type=int, default=2, help="Number of simulation steps (evaluation rounds) to run. Default is 5.")
    parser.add_argument("--use_chat_api", action="store_true", help="Use Gemini's chat API for more efficient multi-turn dialogs. Only applicable when --llm_source is 'api'.")
    parser.add_argument("--batch_size", type=int, default=3, help="Number of conversations per simulation step. Default is 3.")
    parser.add_argument("--use_static_kb", action="store_true", help="Use a static knowledge base for the simulation. Default is False.")

    # --- Evaluation & Trust Market Args ---
    parser.add_argument("--evaluation_method", type=str, choices=["specific_ratings", "comparative_binary"], default="comparative_binary", help="Evaluation method for user feedback.")
    parser.add_argument("--rating_scale", type=int, choices=[5, 10], default=5, help="Rating scale for user feedback (specific_ratings).")
    parser.add_argument("--trust_decay_rate", type=float, default=0.99, help="Decay rate for trust scores per round.")
    parser.add_argument("--auditor_frequency", type=int, default=4, help="Frequency (in rounds) for auditor evaluations.")
    parser.add_argument("--user_rep_frequency", type=int, default=2, help="Frequency (in rounds) for user representative evaluations.")
    parser.add_argument("--regulator_frequency", type=int, default=10, help="Frequency (in rounds) for regulator evaluations.")
    parser.add_argument("--exp_name", type=str, default="test_run", help="Name of the experiment.")
    parser.add_argument("--api_provider", type=str, choices=["gemini", "openai"], default="gemini", help="API provider to use. Default is 'gemini'.")


    args = parser.parse_args()

    # --- API Key Handling ---
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not gemini_api_key and args.api_provider == "gemini":
        # IMPORTANT: Replace with your actual key ONLY for testing. Use environment variables.
        gemini_api_key = "YOUR_GEMINI_API_KEY" # MODIFY THIS IF NEEDED FOR TESTING
        if gemini_api_key == "YOUR_GEMINI_API_KEY":
             print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print("WARNING: GEMINI_API_KEY not set in environment.")
             print("Using placeholder. LLM calls will likely fail.")
             print("Set the GEMINI_API_KEY environment variable.")
             print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        else:
             print("Warning: GEMINI_API_KEY environment variable not set. Using hardcoded key (for testing ONLY).")

    if not openai_api_key and args.api_provider == "openai":
        # IMPORTANT: Replace with your actual key ONLY for testing. Use environment variables.
        openai_api_key = "YOUR_OPENAI_API_KEY" # MODIFY THIS IF NEEDED FOR TESTING
        if openai_api_key == "YOUR_OPENAI_API_KEY":
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("WARNING: OPENAI_API_KEY not set in environment.")
            print("Using placeholder. LLM calls will likely fail.")
            print("Set the OPENAI_API_KEY environment variable.")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        else:
            print("Warning: OPENAI_API_KEY environment variable not set. Using hardcoded key (for testing ONLY).")

    if args.llm_source == 'api':
        if args.api_provider == 'gemini':
            args.largest_model = "gemini-2.5-pro"
            args.api_model_name = "gemini-2.5-flash"
            # args.api_model_name = "gemini-2.5-pro"
        elif args.api_provider == 'openai':
            # Configure OpenAI models here if needed
            args.largest_model = "o3"
            # args.api_model_name = "o3"
            args.api_model_name = "o3"

    # Validate chat API usage
    if args.use_chat_api and args.api_provider != "gemini":
        print("Warning: --use_chat_api is only applicable when using Gemini API ('gemini'). Ignoring.")
        args.use_chat_api = False

    # --- Knowledge Base ---
    static_knowledge_base = None
    if args.use_static_kb:
        static_knowledge_base = {
            "What is your return policy?": "We offer a 30-day return policy for most items. Please see our full return policy here: [link to policy]",
            "How do I track my order?": "You can track your order by logging into your account and going to the 'Order History' section. Or you can use this link : [link to tracking]",
            "What are your hours of operation?": "Our customer support team is available from 9 AM to 5 PM PST, Monday through Friday.",
            "How do I contact customer support?": "You can reach us by phone at [phone number], by email at [email address], or through live chat on our website.",
            "Do you offer international shipping?": "Yes, we offer international shipping to select countries. Shipping rates and times vary depending on the destination.",
            "What payment methods do you accept?": "We accept Visa, Mastercard, American Express, and PayPal.",
            "How do I reset my password?": "You can reset your password by clicking on the 'Forgot Password' link on the login page.",
            "Do you have a physical store?": "No, we are an online-only retailer.",
            "What is your privacy policy?": "You can find our full privacy policy here: [link to privacy policy]",
            "How do I create an account?": "You can create an account by clicking on the 'Sign Up' link at the top of our website.",
            "How long does shipping take?": "Domestic orders typically take 3-5 business days to arrive. International shipping times vary.",
            "Do you offer gift wrapping?": "Yes, we offer gift wrapping for a small fee during checkout.",
            "What is your warranty policy?": "Our products come with a standard one-year warranty against defects. Extended warranties are available for purchase.",
            "How do I cancel my order?": "You can cancel your order within 24 hours of placing it by contacting our customer support team.",
            "Do you have a loyalty program?": "Yes, we have a loyalty program that rewards frequent customers with points that can be redeemed for discounts.",
        }

    # --- Load Profiles & Prompts ---
    agent_profiles_all, user_profiles_all = load_profiles("saved_profiles") if os.path.exists("saved_profiles/agent_profiles_fixed.json") else ([], [])
    conversation_prompts = load_prompts("generated_prompts_fixed.json") if os.path.exists("generated_prompts_fixed.json") else []

    # Default profiles if none loaded
    if not agent_profiles_all:
        print("Warning: No agent profiles found. Using default profiles.")
        agent_profiles_all = [
            {
                "primary_goals": [("Primary", "Assist customers efficiently")],
                "communication_style": ["Professional", "Concise"],
                "behavioral_tendencies": ["Responds directly to questions"],
                "knowledge_breadth": "Comprehensive knowledge of products",
                "knowledge_depth": "High level of detail",
                "knowledge_accuracy": "Highly accurate"
            },
            {
                "primary_goals": [("Primary", "Maximize customer satisfaction")],
                "communication_style": ["Friendly", "Conversational"],
                "behavioral_tendencies": ["Goes above and beyond", "Provides suggestions"],
                "knowledge_breadth": "Standard knowledge of products",
                "knowledge_depth": "Moderate level of detail",
                "knowledge_accuracy": "Generally accurate"
            },
            {
                "primary_goals": [("Primary", "Increase sales"), ("Secondary", "Assist customers")],
                "communication_style": ["Persuasive", "Enthusiastic"],
                "behavioral_tendencies": ["Recommends premium products", "Upsells when possible"],
                "knowledge_breadth": "Focused on premium products",
                "knowledge_depth": "Deep knowledge of premium features",
                "knowledge_accuracy": "Accurate but emphasizes benefits"
            }
        ]

    if not user_profiles_all:
        print("Warning: No user profiles found. Using default profiles.")
        user_profiles_all = [
            {
                "technical_proficiency": "High",
                "patience": "Low",
                "trust_propensity": "Skeptical",
                "focus": "Technical details",
                "communication_style": ["Direct", "Technical"],
                "mood": ["Neutral"]
            },
            {
                "technical_proficiency": "Medium",
                "patience": "Medium",
                "trust_propensity": "Neutral",
                "focus": "Solution-oriented",
                "communication_style": ["Conversational"],
                "mood": ["Positive"]
            },
            {
                "technical_proficiency": "Low",
                "patience": "High",
                "trust_propensity": "Trusting",
                "focus": "Basic functionality",
                "communication_style": ["Verbose", "Novice"],
                "mood": ["Confused", "Anxious"]
            }
        ]


    # --- Simulation Setup ---
    num_agents_to_simulate = 3 # Number of agent profiles to use in simulation
    num_users_to_simulate = 3  # Number of user profiles to use in simulation

    # Sample agent profiles if more are available than needed
    agent_profiles = random.sample(agent_profiles_all, min(num_agents_to_simulate, len(agent_profiles_all)))

    # Sample user profiles and conversation prompts if more are available than needed
    user_indices = list(range(len(user_profiles_all)))
    sampled_indices = random.sample(user_indices, min(num_users_to_simulate, len(user_profiles_all)))
    
    # Sample both profiles and prompts using the same indices
    user_profiles = [user_profiles_all[i] for i in sampled_indices]
    conversation_prompts = [conversation_prompts[i] for i in sampled_indices] if conversation_prompts and len(conversation_prompts) == len(user_profiles_all) else []
    
    print(f"Simulating with {len(agent_profiles)} agent profiles and {len(user_profiles)} user profiles.")
    print(f"User profiles:")
    print(json.dumps(user_profiles, indent=2))
    print(f"Agent profiles:")
    print(json.dumps(agent_profiles, indent=2))


    # --- 1. Initialize Customer Support Simulation Model ---
    # Note: We remove alpha, as trust updates are handled by the market
    print("Initializing Customer Support Simulation Model...")
    customer_support_sim = CustomerSupportModel(
        num_users=len(user_profiles),
        num_agents=len(agent_profiles),
        # alpha=..., # Removed
        batch_size=args.batch_size,
        model_path=args.model_path,
        evaluation_method=args.evaluation_method, # Still needed for user feedback type
        rating_scale=args.rating_scale,           # Still needed for user feedback type
        gemini_api_key=gemini_api_key,
        openai_api_key=openai_api_key,
        llm_source=args.llm_source,
        api_provider=args.api_provider,
        agent_profiles=agent_profiles, # Pass the selected profiles
        user_profiles=user_profiles,   # Pass the selected profiles
        conversation_prompts=conversation_prompts,
        static_knowledge_base=static_knowledge_base,
        max_dialog_rounds=args.max_dialog_rounds,
        use_chat_api=args.use_chat_api,
        api_model_name=args.api_model_name,
    )

    # --- 2. Initialize Trust Market System ---
    print("Initializing Trust Market System...")
    trust_market_config = {
        'dimensions': [ # Ensure these match the rating dimensions used in UserAgentSet
                "Factual_Correctness", "Process_Reliability", "Value_Alignment",
                "Communication_Quality", "Problem_Resolution", "Safety_Security",
                "Transparency", "Adaptability", "Trust_Calibration", "Manipulation_Resistance"
            ],
        'trust_decay_rate': args.trust_decay_rate,
        'rating_scale': args.rating_scale, # Pass rating scale to market system for normalization
        'user_feedback_strength': 0.1, # Example: How much direct user ratings impact score
        'comparative_feedback_strength': 0.05, # Example: How much comparative wins impact score
        'initial_R_oracle': 230,
        'initial_T_oracle': 200/230
         # Add other market config if needed (e.g., primary_sources)
    }
    trust_market_system = TrustMarketSystem(config=trust_market_config)

    # --- 3. Pass Simulation Model to Trust Market System ---
    trust_market_system.register_simulation_module(customer_support_sim)
    print("Registered simulation module with Trust Market System.")

    # --- 4. Initialize & Register Information Sources ---
    print("Registering Information Sources...")

    # Add Regulator
    regulator = Regulator(
        source_id="regulator",
        market=trust_market_system.trust_market,
        api_key=gemini_api_key if args.llm_source == "api" else None,
        api_model_name=args.largest_model if args.llm_source == "api" else None,
        api_provider=args.api_provider,
        openai_api_key=openai_api_key,
    )
    for agent_id, profile in trust_market_system.agent_profiles.items():
        regulator.add_agent_profile(agent_id, profile)
    trust_market_system.register_regulator(regulator, evaluation_frequency=args.regulator_frequency, investment_horizon=args.regulator_frequency)
    print(f"Registered Regulator: {regulator.source_id} (Eval Freq: {args.regulator_frequency})")

    # --- 5. Run Simulation Rounds via Trust Market System ---
    print(f"\nStarting simulation orchestration for {args.num_steps} rounds...")
    trust_market_system.run_evaluation_rounds(args.num_steps)

    # --- 6. Final Summary ---
    print("\n=== Simulation Complete ===")
    final_market_state = trust_market_system.summarize_market_state()
    print("Final Market State Summary:")
    # Pretty print the summary dictionary
    print(json.dumps(final_market_state, indent=2, default=lambda o: '<not serializable>'))


    print("\nFinal Agent Trust Scores (from Trust Market):")
    final_scores = trust_market_system.get_agent_trust_scores()
    if not final_scores:
         print("No final scores available.")
    else:
        # Ensure agent_profiles were populated correctly
        sim_agent_profiles = trust_market_system.agent_profiles if hasattr(trust_market_system, 'agent_profiles') else {}

        for agent_id_str, scores in final_scores.items():
             try:
                 # Agent IDs from market might be strings if defaultdict was used; ensure integer for indexing
                 agent_id = int(agent_id_str)
                 if agent_id in sim_agent_profiles:
                     profile = sim_agent_profiles[agent_id] # Get profile based on ID
                     goals = profile.get("primary_goals", [("Primary", "Unknown")])
                     goals_text = goals[0][1] if goals else "Unknown"
                     print(f"\nAgent {agent_id} (Goal: {goals_text}):")
                 else:
                     print(f"\nAgent {agent_id} (Profile not found):")

                 score_strs = []
                 # Sort dimensions for consistent output
                 sorted_dims = sorted(scores.keys())
                 for dim in sorted_dims:
                     score = scores[dim]
                     score_strs.append(f"{dim}: {score:.3f}")

                 # Print scores in rows for better readability
                 for i in range(0, len(score_strs), 3):
                      print("  " + ", ".join(score_strs[i:i+3]))
             except (ValueError, KeyError, IndexError) as e:
                 print(f"Error displaying scores for agent {agent_id_str}: {e}")
                 print(f"  Raw Scores: {scores}")

    trust_market_system.trust_market.visualize_trust_scores(show_investments=True, save_path="figures/", experiment_name=args.exp_name)
    # trust_market_system.trust_market.visualize_source_value(save_path="figures/", experiment_name=args.exp_name)
    trust_market_system.trust_market.save_logged_data("./run_logs", filename_prefix=args.exp_name, file_format="json")
    #agents=list(sim_agent_profiles.keys()), dimensions=sorted_dims,
    # show_investments=True, start_round=0, end_round=args.num_steps)


def default_json_serializer(o):
    """Helper to serialize non-standard objects to JSON."""
    if isinstance(o, (set, tuple)):
        return list(o)
    if hasattr(o, '__dict__'):
        # A simple way to serialize an object, might need more care for complex objects
        return {key: value for key, value in o.__dict__.items() if not key.startswith('_')}
    return f"<<non-serializable: {type(o).__name__}>>"

def simulate_and_save(args: argparse.Namespace, output_path: str):
    """
    Simulates interactions between 3 users and 3 agents, then saves all
    relevant data for later analysis and recreation.
    """
    print("--- (1) Starting Simulation and Data Saving ---")

    # --- Setup similar to info_main.py ---
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    # Configure API models based on provider
    if args.llm_source == 'api':
        if args.api_provider == 'gemini':
            args.largest_model = "gemini-2.5-pro"
            args.api_model_name = "gemini-2.5-flash"
        elif args.api_provider == 'openai':
            args.largest_model = "o4-mini"
            args.api_model_name = "o4-mini"

    # --- Load Profiles & Prompts ---
    agent_profiles_all, user_profiles_all = load_profiles("saved_profiles")
    conversation_prompts_all = load_prompts("generated_prompts_fixed.json")

    # --- Sample 3 users and 3 agents ---
    num_to_simulate = 3
    agent_profiles = random.sample(agent_profiles_all, min(num_to_simulate, len(agent_profiles_all)))
    
    user_indices = list(range(len(user_profiles_all)))
    sampled_indices = random.sample(user_indices, min(num_to_simulate, len(user_profiles_all)))
    
    user_profiles = [user_profiles_all[i] for i in sampled_indices]
    conversation_prompts = [conversation_prompts_all[i] for i in sampled_indices] if conversation_prompts_all and len(conversation_prompts_all) == len(user_profiles_all) else []

    print(f"Simulating with {len(agent_profiles)} agent profiles and {len(user_profiles)} user profiles.")

    static_knowledge_base = { "info": "This is a static KB." } if args.use_static_kb else None

    # --- Initialize Customer Support Simulation Model ---
    customer_support_sim = CustomerSupportModel(
        num_users=len(user_profiles),
        num_agents=len(agent_profiles),
        batch_size=args.batch_size, # Set batch size to 3
        model_path=args.model_path,
        evaluation_method=args.evaluation_method,
        rating_scale=args.rating_scale,
        gemini_api_key=gemini_api_key,
        openai_api_key=openai_api_key,
        llm_source=args.llm_source,
        api_provider=args.api_provider,
        api_model_name=args.api_model_name,
        agent_profiles=agent_profiles,
        user_profiles=user_profiles,
        conversation_prompts=conversation_prompts,
        static_knowledge_base=static_knowledge_base,
        max_dialog_rounds=args.max_dialog_rounds,
        use_chat_api=args.use_chat_api,
    )

    # --- Run Simulation ---
    print("\nRunning multi-turn dialog simulation for {} rounds...".format(args.num_steps))
    all_simulation_outputs = []
    for i in range(args.num_steps):
        print(f"\n--- Running Simulation Round {i+1}/{args.num_steps} ---")
        simulation_output = customer_support_sim.multi_turn_dialog()
        all_simulation_outputs.append(simulation_output)
    print("Simulation finished.")

    # --- Collect Data for Saving ---
    data_to_save = {
        "simulation_args": vars(args),
        "agent_profiles_used": agent_profiles,
        "user_profiles_used": user_profiles,
        "conversation_prompts_used": conversation_prompts,
        "static_knowledge_base": static_knowledge_base,
        "simulation_outputs": all_simulation_outputs, # Changed to plural
    }

    # --- Save Data ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2, default=default_json_serializer)
    
    print(f"\nSimulation data saved successfully to: {output_path}")
    return output_path


def load_and_recreate(saved_data_path: str):
    """
    Loads data from a simulation run and recreates the state of the
    TrustMarketSystem and its components for analysis.
    """
    print(f"\n--- (2) Loading and Recreating State from {saved_data_path} ---")

    # --- Load Data ---
    if not os.path.exists(saved_data_path):
        print(f"Error: Saved data file not found at {saved_data_path}")
        return None, None

    with open(saved_data_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    
    print("Data loaded successfully.")

    # --- Recreate Components ---
    args = argparse.Namespace(**saved_data["simulation_args"])
    agent_profiles = saved_data["agent_profiles_used"]
    user_profiles = saved_data["user_profiles_used"]
    conversation_prompts = saved_data["conversation_prompts_used"]
    static_knowledge_base = saved_data["static_knowledge_base"]
    all_round_outputs = saved_data["simulation_outputs"]

    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    # 1. Recreate Trust Market System
    print("Initializing Trust Market System...")
    trust_market_config = {
        'dimensions': [
            "Factual_Correctness", "Process_Reliability", "Value_Alignment",
            "Communication_Quality", "Problem_Resolution", "Safety_Security",
            "Transparency", "Adaptability", "Trust_Calibration", "Manipulation_Resistance"
        ],
        'trust_decay_rate': args.trust_decay_rate,
        'rating_scale': args.rating_scale,
        'user_feedback_strength': 0.1,
        'comparative_feedback_strength': 0.05,
    }
    trust_market_system = TrustMarketSystem(config=trust_market_config)

    # 2. Recreate and Register Simulation Module
    print("Initializing and registering simulation module...")
    customer_support_sim = CustomerSupportModel(
        num_users=len(user_profiles),
        num_agents=len(agent_profiles),
        batch_size=args.batch_size,
        model_path=args.model_path,
        evaluation_method=args.evaluation_method,
        rating_scale=args.rating_scale,
        gemini_api_key=gemini_api_key,
        openai_api_key=openai_api_key,
        llm_source=args.llm_source,
        api_provider=args.api_provider,
        api_model_name=args.api_model_name,
        agent_profiles=agent_profiles,
        user_profiles=user_profiles,
        conversation_prompts=conversation_prompts,
        static_knowledge_base=static_knowledge_base,
        max_dialog_rounds=args.max_dialog_rounds,
        use_chat_api=args.use_chat_api,
    )
    trust_market_system.register_simulation_module(customer_support_sim)
    
    # 3. Initialize and Register Information Sources
    print("Registering information sources (Auditor, UserRep, Regulator)...")
    if args.api_provider == 'gemini':
        api_key = gemini_api_key
        largest_model = "gemini-2.5-pro"
    else: # openai
        api_key = openai_api_key
        largest_model = "o4-mini"

    auditor = AuditorWithProfileAnalysis(source_id="auditor_main", market=trust_market_system.trust_market, 
                                         api_key=api_key, api_model_name=args.api_model_name, 
                                         api_provider=args.api_provider, openai_api_key=openai_api_key)
    trust_market_system.register_auditor(auditor, evaluation_frequency=args.auditor_frequency)

    user_rep = UserRepresentativeWithHolisticEvaluation(source_id="user_rep_general", user_segment="balanced", 
                                                        representative_profile={}, market=trust_market_system.trust_market, 
                                                        api_key=api_key, api_model_name=args.api_model_name, 
                                                        api_provider=args.api_provider, openai_api_key=openai_api_key)
    for user_idx in range(customer_support_sim.num_users):
        user_rep.add_represented_user(user_idx, trust_market_system.user_profiles[user_idx]) # user_idx matches simulation's internal ID    
    trust_market_system.register_user_representative(user_rep, evaluation_frequency=args.user_rep_frequency)

    regulator = Regulator(source_id="regulator", market=trust_market_system.trust_market, api_key=api_key, api_model_name=largest_model, api_provider=args.api_provider, openai_api_key=openai_api_key)
    trust_market_system.register_regulator(regulator, evaluation_frequency=args.regulator_frequency)

    # 4. Populate System with Saved Simulation Data
    print("Populating system with saved conversation and feedback data...")
    all_round_outputs = saved_data["simulation_outputs"]
    for i, round_output in enumerate(all_round_outputs):
        print(f"  Processing data from round {i+1}/{len(all_round_outputs)}...")
        # Manually advance the round counters in the system and market
        trust_market_system.evaluation_round += 1
        trust_market_system.trust_market.increment_evaluation_round()
        trust_market_system._process_simulation_output(round_output)
    print("System state recreated successfully.")

    # NEW STEP: Set the initial market state to the regulator's target
    # Path for caching the regulator's plan.
    regulator_plan_path = os.path.join(os.path.dirname(saved_data_path), "regulator_plan.pkl")
    set_market_state_from_regulator(trust_market_system, regulator_plan_path)

    return trust_market_system, customer_support_sim

def set_market_state_from_regulator(trust_market_system, regulator_plan_path=None):
    """
    Initializes the market by setting source investments proportionally
    based on the regulator's ideal total capital distribution.
    Optionally saves/loads the regulator's plan to avoid re-computation.
    """
    print("\n--- Initializing Market State to Regulator's Target Distribution ---")
    
    market = trust_market_system.trust_market
    regulator = None
    all_sources = list(trust_market_system.information_sources.values())

    for source in all_sources:
        if isinstance(source, Regulator):
            regulator = source
            break
            
    if not regulator:
        print("Warning: No Regulator found in the system. Cannot set market state.")
        return

    target_total_capital_dist = None
    # 1. Get the regulator's "grand plan" for total capital in each asset
    if regulator_plan_path and os.path.exists(regulator_plan_path):
        print(f"Loading regulator plan from {regulator_plan_path}...")
        with open(regulator_plan_path, 'rb') as f:
            target_total_capital_dist = pickle.load(f)
        print("Regulator plan loaded.")
    else:
        print("Generating new regulator plan...")
        target_total_capital_dist = regulator.get_target_capital_distribution(evaluation_round=1)
        if regulator_plan_path and target_total_capital_dist:
            print(f"Saving new regulator plan to {regulator_plan_path}...")
            # Ensure directory exists
            os.makedirs(os.path.dirname(regulator_plan_path), exist_ok=True)
            with open(regulator_plan_path, 'wb') as f:
                pickle.dump(target_total_capital_dist, f)
            print("Regulator plan saved.")

    if not target_total_capital_dist:
        print("Warning: Regulator did not produce a target capital distribution. Cannot set market state.")
        return
        
    # 2. Calculate the proportional investments for each source
    investments = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    all_agents_in_projection = set()
    for dim_capital in target_total_capital_dist.values():
        all_agents_in_projection.update(dim_capital.keys())

    for source in all_sources:
        if isinstance(source, Regulator):
            continue
        for dim in market.dimensions:
            capacity = market.source_available_capacity.get(source.source_id, {}).get(dim, 0)
            total_capital_dim = sum(target_total_capital_dist.get(dim, {}).values())
            
            if total_capital_dim > 0:
                for agent_id in all_agents_in_projection:
                    agent_capital_in_dim = target_total_capital_dist.get(dim, {}).get(agent_id, 0)
                    agent_frac = agent_capital_in_dim / total_capital_dim if total_capital_dim else 0
                    investments[source.source_id][agent_id][dim] = capacity * agent_frac
    # ipdb.set_trace()
    # 3. For each source, calculate the delta from current holdings and process the trades
    for source in all_sources:
        if isinstance(source, Regulator):
            continue
        trades_to_propose = []
        source_id = source.source_id
        
        current_positions = market.source_investments.get(source_id, {})
        target_investments = investments.get(source_id, {})
        
        all_relevant_agents = set(current_positions.keys()) | set(target_investments.keys())

        for agent_id in all_relevant_agents:
            current_dims = current_positions.get(agent_id, {})
            target_dims = target_investments.get(agent_id, {})
            all_relevant_dims = set(current_dims.keys()) | set(target_dims.keys())

            for dim in all_relevant_dims:
                target_cash_value = target_dims.get(dim, 0.0)
                
                # Get current cash value from shares held
                current_shares_held = current_dims.get(dim, 0.0)
                market.ensure_agent_dimension_initialized_in_amm(agent_id, dim)
                amm_params = market.agent_amm_params[agent_id][dim]
                current_price = amm_params['R'] / amm_params['T'] if amm_params['T'] > 1e-6 else 0
                current_cash_value = current_shares_held * current_price
                
                # The amount to trade is the delta required to reach the target
                delta_cash_to_trade = target_cash_value - current_cash_value
                
                if abs(delta_cash_to_trade) > 0.01: # Threshold to avoid tiny, noisy trades
                    trades_to_propose.append((agent_id, dim, delta_cash_to_trade, 1.0))
        
        if trades_to_propose:
            print(f"  Processing {len(trades_to_propose)} trades for source {source_id} to align with Regulator state...")
            market.process_investments(source_id, trades_to_propose)
        print(f"  Source {source_id} has {market.source_available_capacity[source_id]} capacity left.")
             
    print("Market state has been set based on proportional regulator projection.")
    trust_market_system.display_current_scores()

def test_individual_sources():
    parser = argparse.ArgumentParser(description="Test and debug primary information sources (user-agent interactions).")
    # Add arguments from info_main.py that are relevant
    parser.add_argument("--model_path", type=str, default="/data/models/huggingface/meta-llama/Llama-3-8B-Instruct")
    parser.add_argument("--llm_source", type=str, choices=["local", "api"], default="api")
    parser.add_argument("--max_dialog_rounds", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=20, help="Number of simulation steps (evaluation rounds) to run. Default is 5.")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--use_static_kb", action="store_true")
    parser.add_argument("--use_chat_api", action="store_true")
    
    parser.add_argument("--evaluation_method", type=str, choices=["specific_ratings", "comparative_binary"], default="comparative_binary")
    parser.add_argument("--rating_scale", type=int, choices=[5, 10], default=5)
    parser.add_argument("--trust_decay_rate", type=float, default=0.99)
    parser.add_argument("--auditor_frequency", type=int, default=4)
    parser.add_argument("--user_rep_frequency", type=int, default=2)
    parser.add_argument("--regulator_frequency", type=int, default=10)
    parser.add_argument("--exp_name", type=str, default="test_run")
    parser.add_argument("--api_provider", type=str, choices=["gemini", "openai"], default="gemini")
    
    args = parser.parse_args()

    # Define where to save the debug file
    output_dir = "run_logs/debug_runs"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filepath = os.path.join(output_dir, f"primary_source_test_rounds5_{timestamp}.json")

    # --- Run Step 1: Simulate and Save ---
    simulate_and_save(args, output_filepath)

    # # --- Run Step 2: Load and Recreate ---
    # recreated_system, recreated_sim = load_and_recreate(output_filepath)

    # # --- Verification and Analysis ---
    # if recreated_system:
    #     print("\n--- Verification ---")
    #     print("Recreated system is ready for analysis.")
    #     print(f"Number of conversations loaded: {len(recreated_system.conversation_histories)}")
    #     print("Final market state summary from loaded data:")
    #     final_market_state = recreated_system.summarize_market_state()
    #     print(json.dumps(final_market_state, indent=2, default=default_json_serializer))

    #     print("\nAgent trust scores after processing loaded feedback:")
    #     recreated_system.display_current_scores()

    #     # Now you can interact with `recreated_system` to debug specific parts,
    #     # for example, running a single source's evaluation:
    #     # print("\n--- Manual Source Evaluation (Example) ---")
    #     # recreated_system._run_source_evaluations()
    #     # recreated_system.display_current_scores()


if __name__ == '__main__':
    test_individual_sources()
    # test_primary_sources()