# info_main.py
from info_agent import CustomerSupportModel
import argparse

## main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Customer Support Model with optional LLM agents.")
    parser.add_argument("--use_llm", action="store_true", help="Use LLM-based agents instead of dictionary-based agents.")
    parser.add_argument("--model_path", type=str, default="/data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct", help="Path to the pretrained LLM model.")
    args = parser.parse_args()

    # Define the knowledge base
    agent_knowledge_base = user_knowledge_base = {
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
    num_users = 10
    num_agents = 3
    alpha = 0.1  # Learning rate
    bsz = 10  # Batch size
    model = CustomerSupportModel(num_users, num_agents, user_knowledge_base, agent_knowledge_base, alpha, batch_size=bsz, use_llm=args.use_llm, model_path=args.model_path)

    for i in range(100):  # Run for 100 steps
        model.step()