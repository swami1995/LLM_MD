# from info_agent import CustomerSupportModel # We'll get this via TrustMarketSystem now
import argparse
import os
import random # Import random for sampling
import json # Added for final summary printing
from agent_prompting_utils import load_profiles, load_prompts

# Import Trust Market components
from trust_market.trust_market_system import TrustMarketSystem
from trust_market.trust_market import TrustMarket
from trust_market.auditor import AuditorWithProfileAnalysis # Use the enhanced Auditor
from trust_market.user_rep import UserRepresentativeWithHolisticEvaluation # Use the enhanced UserRep

# Import CustomerSupportModel from info_agent
from info_agent import CustomerSupportModel
import ipdb
# Import LLM client setup (assuming you have a helper, otherwise initialize here)
# Example: from llm_utils import initialize_llm_client
# For Gemini API, genai client is initialized within agent classes if needed
# max dialog rounds = 3
# user rep frequency = 3
# user rep min conversations = 3 -> 1

# Values to check 
# user feedback strength (feedback_strength) - 
# confidences and how they are handled and how each quantity is scaled (for user feedback, user rep/auditor)
# investor initialization
# investor walk through step by step on how the investment goes

## main function
if __name__ == "__main__":
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
    parser.add_argument("--auditor_frequency", type=int, default=5, help="Frequency (in rounds) for auditor evaluations.")
    parser.add_argument("--user_rep_frequency", type=int, default=2, help="Frequency (in rounds) for user representative evaluations.")


    args = parser.parse_args()

    # --- API Key Handling ---
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key and args.llm_source == "api":
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

    if args.llm_source == 'api':
        args.api_model_name = "gemini-2.5-pro-preview-06-05"
        # args.api_model_name = "gemini-2.5-flash-preview-05-20" 
        # args.api_model_name = "gemini-2.0-flash-lite"
        # args.api_model_name = "gemini-2.0-flash"

    # Validate chat API usage
    if args.use_chat_api and args.llm_source != "api":
        print("Warning: --use_chat_api is only applicable when using Gemini API ('api'). Ignoring.")
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
        llm_source=args.llm_source,
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
         # Add other market config if needed (e.g., primary_sources)
    }
    trust_market_system = TrustMarketSystem(config=trust_market_config)

    # --- 3. Pass Simulation Model to Trust Market System ---
    trust_market_system.register_simulation_module(customer_support_sim)
    print("Registered simulation module with Trust Market System.")

    # --- 4. Initialize & Register Information Sources ---
    print("Registering Information Sources...")

    # Add Auditors
    auditor = AuditorWithProfileAnalysis(
        source_id="auditor_main",
        market=trust_market_system.trust_market, # Pass the market instance
        api_key=gemini_api_key if args.llm_source == "api" else None,
        api_model_name=args.api_model_name if args.llm_source == "api" else None,
        # Pass llm_client if you initialize it separately
    )
    # Provide profiles *after* simulation module registration (which copies profiles to market sys)
    for agent_id, profile in trust_market_system.agent_profiles.items():
         auditor.add_agent_profile(agent_id, profile)
    trust_market_system.register_auditor(auditor, evaluation_frequency=args.auditor_frequency)
    print(f"Registered Auditor: {auditor.source_id} (Eval Freq: {args.auditor_frequency})")

    # Add User Representatives
    # Example: One general representative
    user_rep = UserRepresentativeWithHolisticEvaluation(
        source_id="user_rep_general",
        user_segment="balanced", # Define segment characteristics if needed
        representative_profile={}, # Can add details about the rep's focus
        market=trust_market_system.trust_market,
        api_key=gemini_api_key if args.llm_source == "api" else None,
        api_model_name=args.api_model_name if args.llm_source == "api" else None,
        # Pass llm_client if needed
    )
    # Map users in the simulation to this representative
    # The representative needs to know which user IDs it's responsible for
    for user_idx in range(customer_support_sim.num_users):
        user_rep.add_represented_user(user_idx, trust_market_system.user_profiles[user_idx]) # user_idx matches simulation's internal ID
    trust_market_system.register_user_representative(user_rep, evaluation_frequency=args.user_rep_frequency)
    print(f"Registered User Representative: {user_rep.source_id} (Eval Freq: {args.user_rep_frequency})")

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