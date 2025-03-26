# from info_agent import CustomerSupportModel # We'll get this via TrustMarketSystem now
import argparse
import os
import random # Import random for sampling
from agent_prompting_utils import load_profiles, load_prompts

# Import Trust Market components
from trust_market.trust_market_system import TrustMarketSystem
from trust_market.trust_market import TrustMarket
from trust_market.auditor import AuditorWithProfileAnalysis # Use the enhanced Auditor
from trust_market.user_rep import UserRepresentativeWithHolisticEvaluation # Use the enhanced UserRep

# Import CustomerSupportModel from info_agent
from info_agent import CustomerSupportModel

# Import LLM client setup (assuming you have a helper, otherwise initialize here)
# Example: from llm_utils import initialize_llm_client
# For Gemini API, genai client is initialized within agent classes if needed


## main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Customer Support Simulation with Trust Market Oversight.")
    # --- Simulation Args ---
    parser.add_argument("--model_path", type=str, default="/data/models/huggingface/meta-llama/Llama-3-8B-Instruct", help="Path to the pretrained LLM model. Required if --llm_source is 'local'.")
    parser.add_argument("--llm_source", type=str, choices=["local", "api"], default="api", help="Source of LLM: 'local' (Llama) or 'api' (Gemini). Default is 'api'.")
    parser.add_argument("--max_dialog_rounds", type=int, default=3, help="Maximum number of dialog rounds for each conversation. Default is 3.")
    parser.add_argument("--num_steps", type=int, default=5, help="Number of simulation steps (evaluation rounds) to run. Default is 5.")
    parser.add_argument("--use_chat_api", action="store_true", help="Use Gemini's chat API for more efficient multi-turn dialogs. Only applicable when --llm_source is 'api'.")
    parser.add_argument("--batch_size", type=int, default=3, help="Number of conversations per simulation step. Default is 3.")

    # --- Evaluation & Trust Market Args ---
    parser.add_argument("--evaluation_method", type=str, choices=["specific_ratings", "comparative_binary"], default="specific_ratings", help="Evaluation method for user feedback.")
    parser.add_argument("--rating_scale", type=int, choices=[5, 10], default=5, help="Rating scale for user feedback (specific_ratings).")
    parser.add_argument("--trust_decay_rate", type=float, default=0.99, help="Decay rate for trust scores per round.")
    parser.add_argument("--auditor_frequency", type=int, default=2, help="Frequency (in rounds) for auditor evaluations.")
    parser.add_argument("--user_rep_frequency", type=int, default=1, help="Frequency (in rounds) for user representative evaluations.")


    args = parser.parse_args()

    # --- API Key Handling ---
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key and args.llm_source == "api":
        # IMPORTANT: Replace with your actual key ONLY for testing. Use environment variables.
        gemini_api_key = "YOUR_GEMINI_API_KEY"
        if gemini_api_key == "YOUR_GEMINI_API_KEY":
             print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print("WARNING: GEMINI_API_KEY not set in environment.")
             print("Using placeholder. LLM calls will likely fail.")
             print("Set the GEMINI_API_KEY environment variable.")
             print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        else:
             print("Warning: GEMINI_API_KEY environment variable not set. Using hardcoded key (for testing ONLY).")


    # Validate chat API usage
    if args.use_chat_api and args.llm_source != "api":
        print("Warning: --use_chat_api is only applicable when using Gemini API ('api'). Ignoring.")
        args.use_chat_api = False

    # --- Knowledge Base ---
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
    agent_profiles_all = load_profiles("agent_profiles.json") if os.path.exists("agent_profiles.json") else []
    user_profiles_all = load_profiles("user_profiles.json") if os.path.exists("user_profiles.json") else []
    conversation_prompts = load_prompts("conversation_prompts.json") if os.path.exists("conversation_prompts.json") else []

    # Default profiles if none loaded
    if not agent_profiles_all:
        print("Warning: No agent profiles found. Using default profiles.")
        # (Keep your default agent profiles here)
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
        # (Keep your default user profiles here)
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

    # Sample profiles if more are available than needed
    agent_profiles = random.sample(agent_profiles_all, min(num_agents_to_simulate, len(agent_profiles_all)))
    user_profiles = random.sample(user_profiles_all, min(num_users_to_simulate, len(user_profiles_all)))

    print(f"Simulating with {len(agent_profiles)} agent profiles and {len(user_profiles)} user profiles.")


    # --- Initialize Customer Support Simulation Model ---
    # Note: We remove alpha, as trust updates are handled by the market
    print("Initializing Customer Support Simulation Model...")
    customer_support_sim = CustomerSupportModel(
        num_users=len(user_profiles),
        num_agents=len(agent_profiles),
        # alpha=alpha, # Removed - handled by TrustMarket
        batch_size=args.batch_size,
        model_path=args.model_path,
        evaluation_method=args.evaluation_method,
        rating_scale=args.rating_scale,
        gemini_api_key=gemini_api_key,
        llm_source=args.llm_source,
        agent_profiles=agent_profiles, # Pass the selected profiles
        user_profiles=user_profiles,   # Pass the selected profiles
        conversation_prompts=conversation_prompts,
        static_knowledge_base=static_knowledge_base,
        max_dialog_rounds=args.max_dialog_rounds,
        use_chat_api=args.use_chat_api
    )

    # --- Initialize Trust Market System ---
    print("Initializing Trust Market System...")
    trust_market_config = {
        'dimensions': [ # Ensure these match the rating dimensions
                "Factual_Correctness", "Process_Reliability", "Value_Alignment",
                "Communication_Quality", "Problem_Resolution", "Safety_Security",
                "Transparency", "Adaptability", "Trust_Calibration", "Manipulation_Resistance"
            ],
        'trust_decay_rate': args.trust_decay_rate,
        'rating_scale': args.rating_scale, # Pass rating scale to market system
         # Add other market config if needed
    }
    trust_market_system = TrustMarketSystem(config=trust_market_config)

    # --- Pass Simulation Model to Trust Market System ---
    # This allows the market system to trigger simulation steps
    trust_market_system.register_simulation_module(customer_support_sim)

    # --- Register Information Sources ---
    print("Registering Information Sources...")

    # Add Auditors (example: one auditor)
    # Pass the shared LLM client/API key if needed by the auditor's internal evaluators
    auditor = AuditorWithProfileAnalysis(
        source_id="auditor_0",
        market=trust_market_system.trust_market, # Pass the market instance
        api_key=gemini_api_key if args.llm_source == "api" else None
    )
    # Provide profiles to the auditor
    for agent_idx, profile in enumerate(agent_profiles):
         # Using agent_idx as the agent_id consistent with CustomerSupportModel
        auditor.add_agent_profile(agent_idx, profile)
    trust_market_system.register_auditor(auditor, evaluation_frequency=args.auditor_frequency)
    print(f"Registered Auditor: {auditor.source_id}")

    # Add User Representatives (example: one for each user profile type if desired, or one general)
    # Here, we create one representative for simplicity.
    # You could create multiple based on user profile analysis (e.g., technical vs non-technical)
    user_rep = UserRepresentativeWithHolisticEvaluation(
        source_id="user_rep_general",
        user_segment="balanced", # Example segment
        representative_profile={}, # Add profile if needed
        market=trust_market_system.trust_market,
        api_key=gemini_api_key if args.llm_source == "api" else None
    )
    # Map users in the simulation to this representative
    for user_idx in range(len(user_profiles)):
        user_rep.add_represented_user(user_idx, user_profiles[user_idx]) # Use user_idx as ID
    trust_market_system.register_user_representative(user_rep, evaluation_frequency=args.user_rep_frequency)
    print(f"Registered User Representative: {user_rep.source_id}")

    # Add other sources (Domain Experts, Red Teamers, Regulators) here if implemented

    # --- Run Simulation Rounds via Trust Market System ---
    print(f"\nStarting simulation for {args.num_steps} rounds...")
    trust_market_system.run_evaluation_rounds(args.num_steps)

    # --- Final Summary ---
    print("\n=== Simulation Complete ===")
    final_market_state = trust_market_system.summarize_market_state()
    print("Final Market State Summary:")
    # print(json.dumps(final_market_state, indent=2)) # Requires json import

    print("\nFinal Agent Trust Scores (from Trust Market):")
    final_scores = trust_market_system.get_agent_trust_scores()
    for agent_id, scores in final_scores.items():
        profile = agent_profiles[agent_id] # Get profile based on ID
        goals = profile.get("primary_goals", [("Primary", "Unknown")])
        goals_text = goals[0][1] if goals else "Unknown"
        print(f"\nAgent {agent_id} (Goal: {goals_text}):")
        score_strs = []
        for dim, score in scores.items():
            score_strs.append(f"{dim}: {score:.3f}")
        # Print scores in rows for better readability
        for i in range(0, len(score_strs), 3):
             print("  " + ", ".join(score_strs[i:i+3]))

