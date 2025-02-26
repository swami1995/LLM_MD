import random
import os
from google import genai
from google.genai import types
from agent_prompting_utils import load_profiles
import ipdb
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

prompt_structure = {
    "prompt_id": "",
    "user_profile": {}, # Changed from user_type to user_profile
    "user_goal_category": "",  # Added from Version 2
    "user_goal_subcategory": "", # Added from Version 2
    "conversation_objective": "",  # Keeping from Version 3, can combine category/subcategory if needed
    "user_goal_description": "", # Added from Version 2 - more descriptive goal
    "pre_conversation_user_knowledge": "",
    "specific_question": "",
    "desired_outcome": "",
    "user_misconceptions": "",
    "sensitive_data": [],
    "user_context": {
        "headphone_model": "",
        "headphone_model_2": "",  # For comparisons
        "device": "",
        "os": "",
        "purchase_date": "",
        "previous_issues": "",
        "technical_proficiency": "",
        "use_case": "" # Added use case
    },
    "user_knowledge_dict": {
        "correct_info": [],
        "incorrect_info": [],
        "assumptions": [],
        "already_tried": []
    },
    "user_knowledge": "",
    "agent_knowledge": "", # Added agent_knowledge field
    "user_prompt_text": "", # Added user_prompt_text field
    "expected_response_type": "", # Added from Version 3
    "generation_notes": "" # Added from Version 2 and 3
}


class PromptGenerator:
    def __init__(self, gemini_api_key):
        self.genai_client = genai.Client(api_key=gemini_api_key)
        self.headphone_models = [
            "Sony WH-1000XM5", "Bose Noise Cancelling Headphones 700",
            "Apple AirPods Max", "Sennheiser Momentum 4 Wireless",
            "JBL Tune 760NC", "Audio-Technica ATH-M50xBT2",
            "Technics EAH-AZ80", "Bowers & Wilkins Px8", "Bang & Olufsen Beoplay HX"
        ]
        self.devices = [
            "iPhone 14 Pro", "iPhone 15", "Samsung Galaxy S23", "Samsung Galaxy S24",
            "Google Pixel 7", "Google Pixel 8", "MacBook Pro", "Windows Laptop", "iPad Pro"
        ]
        self.operating_systems = [
            "iOS 16", "iOS 17", "Android 13", "Android 14", "Windows 11", "macOS Ventura", "macOS Sonoma"
        ]
        self.user_types = [
            "Novice", "Expert", "Skeptical", "Demanding", "RefundSeeker",
            "Inquisitive", "Impatient", "Price-Sensitive", "Review-Reliant",
            "Brand-Loyal", "Tech-Savvy", "Non-Technical"
        ]
    
        self.conversation_objectives = { # More structured objectives - Combining aspects from all versions and previous detailed lists
            "product_information": [
                "feature_comparison", "specific_feature_query", "technical_specification_query",
                "compatibility_check", "use_case_recommendation", "availability_check",
                "pricing_query", "reviews_query", "accessory_query", "feature_explanation"
            ],
            "order_shipping": [
                "order_status_check", "shipping_cost_query", "shipping_time_query",
                "address_change_request", "order_cancellation_request", "missing_item_report",
                "damaged_item_report"
            ],
            "troubleshooting": [
                "bluetooth_connectivity", "audio_distortion", "one_side_not_working",
                "microphone_not_working", "battery_charging", "noise_cancellation_issue",
                "controls_not_working", "firmware_update_issue", "app_connectivity", "physical_damage"
            ],
            "returns_warranty": [
                "return_initiation", "exchange_initiation", "warranty_claim_initiation",
                "return_policy_clarification", "warranty_coverage_clarification",
                "return_shipping_cost_query", "refund_status_query"
            ]
        }


    def generate_prompt(self, user_profile=None, user_goal_category=None, user_goal_subcategory=None):
        """Generates a single conversation prompt, fully leveraging LLM with categories."""

        prompt = {key: "" for key in prompt_structure} # Initialize
        prompt["user_context"] = {key: "" for key in prompt_structure['user_context']} # Initialize user_context
        prompt["user_knowledge"] = {key: [] for key in prompt_structure['user_knowledge']} # Initialize user_knowledge


        # Use provided user profile or generate a random one
        if user_profile:
            prompt["user_profile"] = user_profile
        else:
            prompt["user_profile"] = self._generate_random_user_profile()

        # Choose category and subcategory if not provided, ensuring subcategory is valid for the chosen category
        if user_goal_category:
            prompt["user_goal_category"] = user_goal_category
            if user_goal_subcategory:
                prompt["user_goal_subcategory"] = user_goal_subcategory
            else:
                prompt["user_goal_subcategory"] = random.choice(self.conversation_objectives[user_goal_category])
        else:
            prompt["user_goal_category"] = random.choice(list(self.conversation_objectives.keys())) # Random category
            prompt["user_goal_subcategory"] = random.choice(self.conversation_objectives[prompt["user_goal_category"]]) # Random subcategory within category

        prompt["conversation_objective"] = prompt["user_goal_subcategory"] # For simpler access in prompts - can remove if not needed
        prompt["user_goal_description"] = f"User goal is {prompt['user_goal_subcategory']} within {prompt['user_goal_category']} category." # More descriptive goal

        # Randomly fill context, but will be overridden by LLM if needed
        prompt["user_context"]["headphone_model"] = random.choice(self.headphone_models)
        if prompt["user_goal_subcategory"] == "feature_comparison":
          prompt["user_context"]["headphone_model_2"] = random.choice([m for m in self.headphone_models if m != prompt["user_context"]["headphone_model"]])
        prompt["user_context"]["device"] = random.choice(self.devices)
        prompt["user_context"]["os"] = random.choice(self.operating_systems)
        prompt["user_context"]["purchase_date"] = f"2023-{random.randint(1,12):02}-{random.randint(1,28):02}"
        prompt["user_context"]["previous_issues"] = random.choice(["None", "Had connection issues", "Returned a previous pair"])
        prompt["user_context"]["technical_proficiency"] = self._get_technical_proficiency(prompt["user_profile"])
        prompt["user_context"]["use_case"] = random.choice(["travel", "office", "gym", "home listening", "gaming", "studio recording"]) # Expanded use cases
        prompt["sensitive_data"] = ["order_number", "email"]  # Example
        prompt["user_context"]["user_data"] = { # Add user_data
            "order_number": f"{random.randint(1000000, 9999999)}",
            "purchase_date": prompt["user_context"]["purchase_date"],
            "email_address": f"user{random.randint(1,1000)}@example.com",
            "phone_number": f"555-{random.randint(100,999)}-{random.randint(1000,9999)}"
        }

        # Create the LLM prompt and call Gemini
        llm_prompt = self._create_llm_generation_prompt(prompt)
        response = self.genai_client.models.generate_content(
            model="gemini-2.0-flash", # Can use gemini-pro if you prefer, or make configurable
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                temperature=0.7, # Tuned for good balance
                top_p=0.9
            ),
            contents=[llm_prompt] # Use contents
        )

        generated_text = response.text
        parts = generated_text.split("---")

        # Fill in the prompt dictionary with generated text
        # prompt["pre_conversation_user_knowledge"] = parts[0].strip()
        # prompt["specific_question"] = parts[1].strip()
        # prompt["desired_outcome"] = parts[2].strip()
        # if len(parts) > 3:
        #     prompt["user_misconceptions"] = parts[3].strip()
        # prompt["expected_response_type"] = self._determine_expected_response_type(prompt["user_goal_subcategory"]) # Determine based on subcategory
        prompt["user_prompt_text"] = generated_text # Save full generated text
        prompt["generation_notes"] = "LLM-generated" # Mark as LLM generated

        # Further refine user context *based on generated question*:
        self._refine_user_context(prompt)
        # Derive user knowledge based on the question and user type:
        prompt["user_knowledge_dict"], prompt["user_knowledge"] = self._derive_user_knowledge(prompt["user_prompt_text"], prompt["user_profile"]) # Pass user_profile
        prompt["agent_knowledge"] = self._generate_agent_knowledge(prompt["user_prompt_text"], prompt["user_knowledge"]) # Generate agent knowledge

        prompt["prompt_id"] = f"P_{random.randint(1000,9999)}" # Unique ID
        return prompt


    def _generate_random_user_profile(self):
        """Generates a random user profile."""
        technical_proficiency_options = [
            "Novice", "Intermediate", "Expert",
            "Expert in Specific Tech (e.g., Bluetooth, Noise Cancellation)"
        ]
        patience_options = ["Very Patient", "Moderately Patient", "Impatient", "Extremely Impatient"]
        trust_propensity_options = ["Highly Trusting", "Neutral", "Slightly Suspicious", "Highly Suspicious"]
        focus_options = ["General Inquiry", "Troubleshooting-Focused", "Return/Refund-Focused", "Feature-Focused"]
        communication_style_options = [
            ["Concise", "Clear"], ["Verbose", "Detailed"], ["Technical", "Precise"],
            ["Informal", "Friendly"], ["Formal", "Demanding"]
        ]
        mood_options = [["Happy", "Content"], ["Neutral"], ["Sad", "Frustrated"], ["Angry", "Irritated"]]

        return {
            "technical_proficiency": random.choice(technical_proficiency_options),
            "patience": random.choice(patience_options),
            "trust_propensity": random.choice(trust_propensity_options),
            "focus": random.choice(focus_options),
            "communication_style": random.choice(communication_style_options),
            "mood": random.choice(mood_options)
        }
    

    def _get_technical_proficiency(self, user_profile):
        """Maps user type to technical proficiency."""
        return user_profile["technical_proficiency"]
        # if user_type in ["Novice", "Non-Technical", "Impatient"]:
        #     return "Low"
        # elif user_type in ["Skeptical", "Demanding", "Price-Sensitive", "Review-Reliant"]:
        #     return "Medium"
        # elif user_type in ["Expert", "Tech-Savvy", "Inquisitive", "Brand-Loyal"]:
        #     return "High"
        # else:
        #     return "Medium"  # Default

    def _refine_user_context(self, prompt):
        """Refines user context based on the generated question."""
        question = prompt["specific_question"].lower()

        # Extract model names
        mentioned_models = [model for model in self.headphone_models if model.lower() in question]
        if mentioned_models:
            prompt["user_context"]["headphone_model"] = mentioned_models[0]
            if len(mentioned_models) > 1 and prompt["conversation_objective"] == "feature_comparison":
                prompt["user_context"]["headphone_model_2"] = mentioned_models[1]

        # Extract device names
        mentioned_devices = [device for device in self.devices if device.lower() in question]
        if mentioned_devices:
            prompt["user_context"]["device"] = mentioned_devices[0]

        # Extract OS names
        mentioned_os = [os for os in self.operating_systems if os.lower() in question]
        if mentioned_os:
           prompt["user_context"]["os"] = mentioned_os[0]

        # Extract use case (more robust using LLM, but this is a reasonable heuristic)
        use_case_keywords = {
            "travel": ["travel", "airplane", "commute", "train", "bus"],
            "office": ["office", "work", "meeting", "calls"],
            "gym": ["gym", "workout", "exercise", "running", "fitness"],
            "home": ["home", "listening", "relaxing"],
            "gaming": ["gaming", "game", "play", "console", "pc"],
            "studio": ["studio", "recording", "mixing", "monitoring"]
        }
        for use_case, keywords in use_case_keywords.items():
            if any(keyword in question for keyword in keywords):
                prompt["user_context"]["use_case"] = use_case
                break

    def _create_llm_generation_prompt(self, prompt_data):
        """Creates the prompt for the LLM (Gemini) to generate details."""
        user_profile = prompt_data["user_profile"]  # Use the user profile
        objective_cat = prompt_data["user_goal_category"]
        objective_subcat = prompt_data["user_goal_subcategory"]
        model = prompt_data["user_context"]["headphone_model"]
        device = prompt_data["user_context"]["device"]

        llm_prompt = f"""
I'm designing a diverse set of customer support agents and user agents with an LLM to simulate a customer service setup for headphones. 
I'm trying to generate the conversational prompts to provide to the user agents to guide each interaction.
Could you help me write one such prompt text that I can give to a user agent. I've provided below some relevant information you'd need to write the prompt. 

Here's the User Profile that I want the user agent to simulate:
    Technical Proficiency: {user_profile['technical_proficiency']}
    Patience Level: {user_profile['patience']}
    Trust Propensity: {user_profile['trust_propensity']}
    Focus: {user_profile['focus']}
    Communication Style: {', '.join(user_profile['communication_style'])}
    Mood: {', '.join(user_profile['mood'])}

Here's the context for the conversation that I want the user agent to simulate:
Conversation Category: '{objective_cat}'
Conversation Subcategory: '{objective_subcat}'
"""

        # --- Add objective-specific instructions and context -  Using subcategory for more precise control ---

        # Product Information Objectives - More nuanced instructions
        if objective_subcat == "feature_comparison":
            model2 = prompt_data["user_context"].get("headphone_model_2", "other headphones")
            llm_prompt += f"""

The user wants to compare '{model}' and '{model2}'.  Generate text for each section. Focus on making the question about a *specific and comparable* feature (e.g., sound quality for music genre X, noise cancellation in noisy environments, comfort for long wear). Ensure the 'Specific Question' clearly sets up a comparison.
"""
        elif objective_subcat == "specific_feature_query":
            llm_prompt += f"""

The user is asking about a specific feature of '{model}' headphones. Generate a realistic question about a *key selling point* feature (e.g., battery life in real-world use, effectiveness of noise cancellation in decibels, type of Bluetooth codec supported).  Make the 'Specific Question' directly about a measurable feature.
"""
        elif objective_subcat == "technical_specification_query":
            llm_prompt += f"""

The user is asking about a *technical* specification of '{model}' headphones. Generate a question about a *genuinely technical* aspect (e.g., impedance for audiophiles, frequency response range, driver size and type). Make the 'Specific Question' clearly technical and for an informed user (even if user type is Novice, question can be prompted by online info).
"""
        elif objective_subcat == "compatibility_check":
            llm_prompt += f"""

The user wants to know if '{model}' headphones are *fully compatible* with their '{device}' running '{prompt_data['user_context']['os']}'. Generate a question focused on *potential compatibility issues* (e.g., specific Bluetooth features, app compatibility, audio codec support).  The 'Specific Question' should reflect a user's concern about compatibility beyond basic connection.
"""
        elif objective_subcat == "use_case_recommendation":
            use_case = prompt_data["user_context"]["use_case"]
            llm_prompt += f"""

The user is seeking headphone *recommendations for* '{use_case}'. Generate a request for recommendations, but make it *slightly nuanced* – user might have a *budget*, *specific sound preference*, or *feature priority* for that use case. The 'Specific Question' should include some criteria beyond just "recommend headphones for X".
"""
        elif objective_subcat == "availability_check":
             llm_prompt += f"""

The user is interested in *purchasing* '{model}' and checking availability. Generate text for a user ready to buy, but with a *specific constraint or question about availability* (e.g., specific color availability, in-store vs. online availability, availability for immediate shipping). The 'Specific Question' should go beyond a simple "are they in stock?".
"""
        elif objective_subcat == "pricing_query":
            llm_prompt += f"""

The user is asking about the *price* of '{model}'.  Generate a pricing question that is *not just asking for the base price* - user could be asking about *discounts*, *bundles*, *price matching*, or *financing options*. The 'Specific Question' should reflect a user looking for the best deal, not just the listed price.
"""
        elif objective_subcat == "reviews_query":
            llm_prompt += f"""

The user is looking for *customer reviews* for '{model}'. Generate a question that is *more specific than just "show me reviews"* - user might be asking about reviews related to a *specific aspect* (e.g., reviews mentioning comfort for long use, reviews comparing to other models, reviews from users with a similar use case). The 'Specific Question' should seek targeted review information.
"""
        elif objective_subcat == "accessory_query":
            llm_prompt += f"""

The user is asking about *accessories compatible* with '{model}'. Generate a question that is *not just "what accessories are there?"* - user could be looking for accessories for a *specific purpose* (e.g., best carrying case for travel, replacement earcups for comfort, a specific type of cable). The 'Specific Question' should indicate a user with a *specific accessory need*.
"""
        elif objective_subcat == "feature_explanation":
            llm_prompt += f"""

The user wants a *detailed explanation* of a *specific feature* on '{model}'. The feature can be real or slightly fictitious/misunderstood. Generate a question where the user is *clearly confused or wants in-depth understanding* of a feature (e.g., "what exactly does 'adaptive' noise cancellation mean?", "how does the 'immersive sound' feature actually work?", "is 'high-res audio' really noticeable?"). The 'Specific Question' should reveal user's curiosity and potential confusion.
"""

        # Order and Shipping Objectives - Keep similar for brevity, can be expanded with nuances like product info
        elif "order_shipping" in objective_cat:
            llm_prompt += f"""

The user is asking about an order or shipping related issue ({objective_subcat}). Generate text for each section.
"""

        # Troubleshooting Objectives - Keep similar for brevity, can be expanded with nuances like product info
        elif "troubleshooting" in objective_cat:
            llm_prompt += f"""

The user is having a troubleshooting issue ({objective_subcat}) with their '{model}' headphones when using '{device}'. Generate text suitable for each section, reflecting common user frustrations and varying technical knowledge.
"""

        # Returns and Warranty Objectives - Keep similar for brevity, can be expanded with nuances like product info
        elif "returns_warranty" in objective_cat:
            llm_prompt += f"""

The user is asking about returns or warranty ({objective_subcat}) for their '{model}' headphones. Generate text for each section, considering different user attitudes and understanding of policies.
"""

        # Default/fallback (shouldn't usually be needed, but good practice)
        else:
            llm_prompt += f"""

Generate text suitable for each section.
"""
            
        llm_prompt += """

--- Pre-conversation User Knowledge/State:
Describe the user's likely knowledge and state before asking the question. Consider their user profile and the conversational context mentioned above.
--- Specific Question/Initial Query:
Formulate a realistic and specific question a user with the profile mentioned above might use to seed the interaction with given the above context. Make sure the question is consistent with the user profile and the conversational contexts provided above (e.g., a 'Non-Technical' user wouldn't ask a highly technical question).
--- Desired Outcome/Success Criteria (User):
Describe what a successful outcome looks like *from the user's perspective* for this interaction.
--- User Misconceptions/Assumptions (Optional, leave blank if none):
Based on the user profile and question, are there any likely misconceptions or assumptions the user might have, given the rest of their context? If not applicable, leave this section blank.

Now considering all of the above points, please write a detailed prompt for the user agent to be used in the conversation. 
I'm going to directly copy and paste the entire text you write below into the user agent's prompt field so don't include any additional instructions or explanations for me beyond the prompt text itself.
"""
        return llm_prompt


    def _determine_expected_response_type(self, objective):
        """Determines the expected response type (Improved)."""
        mapping = {
            "feature_comparison": "Detailed feature comparison.",
            "specific_feature_query": "Information about a specific feature.",
            "technical_specification_query": "Technical specification information.",
            "compatibility_check": "Compatibility confirmation/denial.",
            "use_case_recommendation": "Headphone recommendations for a use case.",
            "availability_check": "Availability information.",
            "pricing_query": "Pricing and discount information.",
            "reviews_query": "Customer review information.",
            "accessory_query": "Information about compatible accessories.",
            "feature_explanation": "Explanation of a specific feature.",
            "order_status_check": "Order status update.",
            "shipping_cost_query": "Shipping cost information.",
            "shipping_time_query": "Shipping time estimate.",
            "address_change_request": "Confirmation/denial of address change.",
            "order_cancellation_request": "Confirmation/denial of cancellation.",
            "missing_item_report": "Acknowledgement and next steps for missing item.",
            "damaged_item_report": "Acknowledgement and next steps for damaged item.",
            "bluetooth_connectivity": "Troubleshooting steps for Bluetooth.",
            "audio_distortion": "Troubleshooting steps for audio distortion.",
            "one_side_not_working": "Troubleshooting for one-sided audio.",
            "microphone_not_working": "Troubleshooting for microphone issues.",
            "battery_charging": "Troubleshooting for battery charging.",
            "noise_cancellation_issue": "Troubleshooting for noise cancellation.",
            "controls_not_working": "Troubleshooting for control issues.",
            "firmware_update_issue": "Troubleshooting for firmware updates.",
            "app_connectivity": "Troubleshooting for app connection.",
            "physical_damage": "Information on handling physical damage.",
            "return_initiation": "Instructions for initiating a return.",
            "exchange_initiation": "Instructions for initiating an exchange.",
            "warranty_claim_initiation": "Instructions for initiating a warranty claim.",
            "return_policy_clarification": "Clarification of the return policy.",
            "warranty_coverage_clarification": "Clarification of warranty coverage.",
            "return_shipping_cost_query": "Information on return shipping costs.",
            "refund_status_query": "Refund status update."
        }
        return mapping.get(objective, "General information or assistance.")

    def _derive_user_knowledge(self, user_prompt, user_profile):
        """Derives user knowledge using LLM based on question and user profile."""
        user_knowledge = {
            "correct_info": [],
            "incorrect_info": [],
            "assumptions": [],
            "already_tried": []
        }

        llm_prompt_knowledge = f"""
I'm designing a diverse set of customer support agents and user agents with an LLM to simulate a customer service setup for headphones. 

Here's the User Profile of one of the user agents I'm simulating:
    Technical Proficiency: {user_profile['technical_proficiency']}
    Patience Level: {user_profile['patience']}
    Trust Propensity: {user_profile['trust_propensity']}
    Focus: {user_profile['focus']}
    Communication Style: {', '.join(user_profile['communication_style'])}
    Mood: {', '.join(user_profile['mood'])}

I'm using this user agent to simulate one of the interactions. Here's the prompt I designed for the user agent for this interaction: 

User Agent Prompt: 
{user_prompt}

Based on the user profile and the user agent prompt, please derive and expand on the following information about the user's knowledge and state that I can provide to the user agent to guide the interaction:
--- User's Pre-existing Assumptions:
List plausible assumptions the user might have *before* asking this question, based on their profile.
--- User's Incorrect Knowledge/Misconceptions (if any):
List any plausible incorrect knowledge or misconceptions the user might have related to the question, based on their profile. If none are likely, state "None".
--- User's Correct Knowledge (if any):
List any plausible correct knowledge the user might possess related to the question, based on their profile. If none are explicitly implied, state "None".
--- Things User Has Already Tried (if applicable):
Based on the question, list actions the user might have *already tried* to resolve the issue themselves. If not applicable, state "None".

Now considering all of the above points, please write a detailed account of the user's knowledge and assumptions. 
I'm going to directly copy and paste the entire text you write below into the user agent's input along with the user agent prompt I gave above. So don't include any additional instructions or explanations for me. Just fill in the sections above with the appropriate information with appropriate headings.
Moreover, I'm also going to parse your response as follows, so please ensure each section is clearly separated by a line with '---' and the section name:
            '''
            generated_text = response.text
            parts = generated_text.split("---")

            user_knowledge["assumptions"] = [item.strip() for item in parts[0].strip().split("\n") if item.strip() and item.strip() != "None"]
            user_knowledge["incorrect_info"] = [item.strip() for item in parts[1].strip().split("\n") if item.strip() and item.strip() != "None"]
            user_knowledge["correct_info"] =  [item.strip() for item in parts[2].strip().split("\n") if item.strip() and item.strip() != "None"]
            user_knowledge["already_tried"] = [item.strip() for item in parts[3].strip().split("\n") if item.strip() and item.strip() != "None"]
            '''
"""
        response = self.genai_client.models.generate_content(
            model="gemini-2.0-flash", # Using pro for potentially better reasoning
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                temperature=0.6, # Slightly lower temp for more focused output
                top_p=0.85
            ),
            contents=[llm_prompt_knowledge]
        )
        generated_text = response.text
        parts = generated_text.split("---")

        user_knowledge["assumptions"] = [item.strip() for item in parts[0].strip().split("\n") if item.strip() and item.strip() != "None"]
        user_knowledge["incorrect_info"] = [item.strip() for item in parts[1].strip().split("\n") if item.strip() and item.strip() != "None"]
        user_knowledge["correct_info"] =  [item.strip() for item in parts[2].strip().split("\n") if item.strip() and item.strip() != "None"]
        user_knowledge["already_tried"] = [item.strip() for item in parts[3].strip().split("\n") if item.strip() and item.strip() != "None"]

        return user_knowledge, generated_text


    def _generate_agent_knowledge(self, user_prompt, user_knowledge=None):
        """Generates background knowledge for the agent using LLM, considering user knowledge."""
        agent_knowledge_prompt = f"""
I'm designing a diverse set of customer support agents and user agents with an LLM to simulate a customer service setup for headphones. 
For each customer service conversation I create a conversational prompt for the user agent and the corresponding knowledge that the user has. 
I'm now trying to generate the background knowledge that the customer support agent should have to effectively assist the user in that conversation.

Here's the prompt I designed for the user agent for this interaction:
User Agent Prompt:
{user_prompt}

Here's the user's knowledge and state that I derived for this interaction:
User Knowledge:
{user_knowledge}

Please generate all the background knowledge that the customer support agent should have to effectively assist the user in this conversation. 
Make sure to include all relevant information, policies, and technical details that the agent might need to know. Make sure to keep it consistent with the user's correct knowledge.
Also point out any potential misconceptions/incorrect info the user might have as a plain knowledge point, but the agent should not be able to guess that the user has a misconception from that statement itself. 
It's the job of the customer service agent to figure out the user has a misconception through the conversation. 
Also ignore the "already tried" section from the user knowledge. That is not relevant for the agent's background knowledge and the agent should figure those things out through the conversation.

Here are some guidelines for generating the agent knowledge:
- Focus on providing factual, helpful, and concise information relevant to answering the user's question.
- Include key technical details, troubleshooting steps, policy information, or product specifics as needed.
- Ensure the agent knowledge is consistent with the user's correct knowledge, and addresses potential misconceptions/incorrect info but in a way that the agent can't guess the user has a misconception or incorrect info.
- Also ignore the "already tried" section from the user knowledge. That is not relevant for the agent's background knowledge and the agent should figure those things out through the conversation.
- I'm going to directly copy and paste the entire text you write below into the customer service agent's input along with its system prompt that I've already designed. So don't include any additional instructions or explanations for me or for the agent other than the background knowledge itself.
"""
        # if user_knowledge:
        #     user_knowledge_summary = ""
        #     if user_knowledge.get("correct_info"):
        #         user_knowledge_summary += "User seems to correctly know: " + ", ".join(user_knowledge["correct_info"]) + ". "
        #     if user_knowledge.get("incorrect_info"):
        #         user_knowledge_summary += "User might have misconceptions such as: " + ", ".join(user_knowledge["incorrect_info"]) + ". "
        #     if user_knowledge_summary:
        #         agent_knowledge_prompt += f"\nConsider the following about the user's potential knowledge: {user_knowledge_summary}\n"

        # agent_knowledge_prompt += "\n--- Agent Background Knowledge:" # Separator for parsing


        response = self.genai_client.models.generate_content(
            model="gemini-2.0-flash", # Using pro for potentially better knowledge generation
            config=types.GenerateContentConfig(
                max_output_tokens=600, # Allow slightly more tokens for agent knowledge
                temperature=0.5, # Lower temperature for factual agent knowledge
                top_p=0.8
            ),
            contents=[agent_knowledge_prompt]
        )
        agent_knowledge = response.text
        # parts = agent_knowledge_text.split("--- Agent Background Knowledge:") # Split again to isolate knowledge part
        # agent_knowledge = parts[-1].strip() # Take the last part after the separator

        return agent_knowledge
    
    def generate_prompts_batch(self, num_prompts, user_profiles=None):
        """Generates a batch of prompts (random objectives)."""
        prompts = []
        for i in range(num_prompts):
            if user_profiles:
                prompt = self.generate_prompt(user_profiles[i])
            else:
                prompt = self.generate_prompt()
            prompts.append(prompt)
            print(" User Prompt : \n", prompt["user_prompt_text"])
            print("---")
            print(" User Knowledge : \n", prompt["user_knowledge"])
            print("---")
            print(" Agent Knowledge : \n", prompt["agent_knowledge"])
            print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
            print("\n\n")
            ipdb.set_trace()
        return prompts

    def generate_prompts_batch_profile(self, num_prompts, user_profiles):
        """Generates a batch of prompts (random objectives) in parallel."""
        def generate_single_prompt(args):
            profile_idx, profile = args
            return profile_idx, self.generate_prompt(profile)

        all_prompts = [[] for _ in range(len(user_profiles))]
        
        # Create all profile/iteration combinations
        tasks = []
        for j, profile in enumerate(user_profiles):
            for i in range(num_prompts):
                tasks.append((j, profile))

        # Process in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_task = {executor.submit(generate_single_prompt, task): task for task in tasks}
            
            for future in as_completed(future_to_task):
                profile_idx, prompt = future.result()
                all_prompts[profile_idx].append(prompt)
                print(f" User Profile {profile_idx}: prompt finished")

        return all_prompts

    def generate_prompts_from_objectives(self, objective_counts):
        """
        Generates prompts based on a dictionary of objective counts.

        Args:
        objective_counts: A dictionary where keys are conversation objectives
                            and values are the number of prompts to generate
                            for that objective.  Example:
                            {
                                "feature_comparison": 5,
                                "bluetooth_connectivity": 3,
                                "order_status_check": 2
                            }
        Returns:
        A list of generated prompt dictionaries.
        """
        prompts = []
        for objective, count in objective_counts.items():
            if objective not in self.conversation_objectives:
                print(f"Warning: Objective '{objective}' not found. Skipping.")
                continue
            for _ in range(count):
                prompts.append(self.generate_prompt(conversation_objective=objective))
        return prompts
# --- Example Usage ---
api_key = os.environ.get("GEMINI_API_KEY")
generator = PromptGenerator(api_key)

agent_profiles, user_profiles = load_profiles("saved_profiles")

# Generate a batch of prompts with random objectives:
bsz = 10
# randomly sample 10 elements from user_profile
# if user_profiles:
#     user_profiles = random.sample(user_profiles, min(bsz, len(user_profiles)))
    # sample with repetition. 
    
# prompts = generator.generate_prompts_batch(bsz, user_profiles)
prompts = generator.generate_prompts_batch_profile(bsz, user_profiles)

### save the prompts (list of list of dicts (prompt structure) to a file)
with open("generated_prompts.json", "w") as f:
    json.dump(prompts, f)

# for prompt in prompts:
#     print(prompt)
#     print("---")
#     ipdb.set_trace()

# # Generate prompts with specific objective counts:
# objective_counts = {
#     "feature_comparison": 5,
#     "bluetooth_connectivity": 3,
#     "order_status_check": 2,
#     "return_initiation" : 5
# }
# targeted_prompts = generator.generate_prompts_from_objectives(objective_counts)
# for prompt in targeted_prompts:
#     print(prompt)
#     print("---")