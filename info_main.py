from info_agent import CustomerSupportModel
import argparse
import os # Import os for environment variables
from agent_prompting_utils import load_profiles, load_prompts

## main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Customer Support Model with optional LLM agents.")
    parser.add_argument("--use_llm", action="store_true", help="Use LLM-based agents instead of dictionary-based agents.")
    parser.add_argument("--model_path", type=str, default="/data/models/huggingface/meta-llama/Llama-3-8B-Instruct", help="Path to the pretrained LLM model. Required if --llm_source is 'local'.")
    parser.add_argument("--evaluation_method", type=str, choices=["specific_ratings", "comparative_binary"], default="specific_ratings", help="Evaluation method for LLM agents.")
    parser.add_argument("--rating_scale", type=int, choices=[5, 10], default=5, help="Rating scale for LLM agents.")
    parser.add_argument("--llm_source", type=str, choices=["local", "api"], default="api", help="Source of LLM: 'local' (Llama) or 'api' (Gemini). Default is 'api'.")
    parser.add_argument("--max_dialog_rounds", type=int, default=1, help="Maximum number of dialog rounds for each conversation. Default is 1.")
    parser.add_argument("--num_steps", type=int, default=1, help="Number of simulation steps to run. Default is 1.")
    parser.add_argument("--use_chat_api", action="store_true", help="Use Gemini's chat API for more efficient multi-turn dialogs. Only applicable when --llm_source is 'api'.")
    args = parser.parse_args()

    # Define the knowledge base
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
    
    # Load profiles and prompts
    agent_profiles = load_profiles("agent_profiles.json") if os.path.exists("agent_profiles.json") else []
    user_profiles = load_profiles("user_profiles.json") if os.path.exists("user_profiles.json") else []
    conversation_prompts = load_prompts("conversation_prompts.json") if os.path.exists("conversation_prompts.json") else []
    
    # If no profiles loaded, use default values
    if not agent_profiles:
        print("Warning: No agent profiles found. Using default profiles.")
        agent_profiles = [
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
    
    if not user_profiles:
        print("Warning: No user profiles found. Using default profiles.")
        user_profiles = [
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
    
    # Fill in arrays to meet required numbers
    while len(agent_profiles) < 3:
        agent_profiles.append(agent_profiles[0])
    while len(user_profiles) < 3:
        user_profiles.append(user_profiles[0])
    
    num_users = 3
    num_agents = 3
    alpha = 0.1  # Learning rate
    bsz = 3  # Batch size

    # *** IMPORTANT: Replace with your actual Gemini API key ***
    gemini_api_key = os.environ.get("GEMINI_API_KEY") # Get API key from environment variable - recommended
    if not gemini_api_key and args.llm_source == "api": # Only require API key if using Gemini API
        gemini_api_key = "YOUR_GEMINI_API_KEY" # Replace YOUR_GEMINI_API_KEY with your actual API key - FOR TESTING ONLY, SECURE API KEYS PROPERLY
        print("Warning: GEMINI_API_KEY environment variable not set. Falling back to hardcoded key (for testing ONLY).")

    # Validate that chat API is only used with Gemini API
    if args.use_chat_api and args.llm_source != "api":
        print("Warning: --use_chat_api is only applicable when using Gemini API. Ignoring this flag.")
        args.use_chat_api = False
    
    if args.use_chat_api:
        print(f"Initializing CustomerSupportModel with max_dialog_rounds={args.max_dialog_rounds} using Gemini Chat API")
    else:
        print(f"Initializing CustomerSupportModel with max_dialog_rounds={args.max_dialog_rounds}")
        
    model = CustomerSupportModel(
        num_users=num_users,
        num_agents=num_agents,
        alpha=alpha,
        batch_size=bsz,
        model_path=args.model_path,
        evaluation_method=args.evaluation_method,
        rating_scale=args.rating_scale,
        gemini_api_key=gemini_api_key,
        llm_source=args.llm_source,
        agent_profiles=agent_profiles,
        user_profiles=user_profiles,
        conversation_prompts=conversation_prompts,
        static_knowledge_base=static_knowledge_base,
        max_dialog_rounds=args.max_dialog_rounds,
        use_chat_api=args.use_chat_api  # Pass the new parameter
    )

    for i in range(args.num_steps):
        print(f"\n=== Running Step {i+1}/{args.num_steps} ===")
        model.step()