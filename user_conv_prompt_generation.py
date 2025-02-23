import random
import google.generativeai as genai
from google.generativeai import types

prompt_structure = {
    "prompt_id": "",
    "user_type": "",
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
    "user_knowledge": {
        "correct_info": [],
        "incorrect_info": [],
        "assumptions": [],
        "already_tried": []
    },
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


    def generate_prompt(self, user_type=None, user_goal_category=None, user_goal_subcategory=None):
        """Generates a single conversation prompt, fully leveraging LLM with categories."""

        prompt = {key: "" for key in prompt_structure} # Initialize

        prompt["user_type"] = user_type if user_type else random.choice(self.user_types)

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
        prompt["user_context"]["technical_proficiency"] = self._get_technical_proficiency(prompt["user_type"])
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
        try:
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
            prompt["pre_conversation_user_knowledge"] = parts[0].strip()
            prompt["specific_question"] = parts[1].strip()
            prompt["desired_outcome"] = parts[2].strip()
            if len(parts) > 3:
                prompt["user_misconceptions"] = parts[3].strip()
            prompt["expected_response_type"] = self._determine_expected_response_type(prompt["user_goal_subcategory"]) # Determine based on subcategory
            prompt["generation_notes"] = "LLM-generated" # Mark as LLM generated

            # Further refine user context *based on generated question*:
            self._refine_user_context(prompt)
            # Derive user knowledge based on the question and user type:
            prompt["user_knowledge"] = self._derive_user_knowledge(prompt["specific_question"], prompt["user_type"])

        except Exception as e:
            print(f"Error generating prompt with LLM: {e}")
            # Fallback to default values:
            prompt["pre_conversation_user_knowledge"] = "User has basic knowledge."
            prompt["specific_question"] = "I need help."
            prompt["desired_outcome"] = "To get assistance."
            prompt["user_misconceptions"] = ""
            prompt["generation_notes"] = "Fallback - Error in LLM generation" # Mark fallback

        prompt["prompt_id"] = f"P_{random.randint(1000,9999)}" # Unique ID
        return prompt

    def _get_technical_proficiency(self, user_type):
        """Maps user type to technical proficiency."""
        if user_type in ["Novice", "Non-Technical", "Impatient"]:
            return "Low"
        elif user_type in ["Skeptical", "Demanding", "Price-Sensitive", "Review-Reliant"]:
            return "Medium"
        elif user_type in ["Expert", "Tech-Savvy", "Inquisitive", "Brand-Loyal"]:
            return "High"
        else:
            return "Medium"  # Default

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
        user_type = prompt_data["user_type"]
        objective_cat = prompt_data["conversation_objective"]
        objective_subcat = prompt_data["user_goal_subcategory"] # Access subcategory
        model = prompt_data["user_context"]["headphone_model"]
        device = prompt_data["user_context"]["device"]

        llm_prompt = f"""
Generate text for a customer support conversation prompt, focusing on headphones.
User Type: '{user_type}'
Goal Category: '{objective_cat}'
Goal Subcategory: '{objective_subcat}'
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
Describe the user's likely knowledge and state before asking the question. Consider their user type and goal.
--- Specific Question/Initial Query:
Formulate a realistic and specific question a user of type '{user_type}' might ask to achieve the goal '{objective_subcat}' within the category '{objective_cat}'.
--- Desired Outcome/Success Criteria (User):
Describe what a successful outcome looks like *from the user's perspective* for this interaction.
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

    def _derive_user_knowledge(self, question, user_type):
        """Derives user knowledge based on question and user type."""
        user_knowledge = {
            "correct_info": [],
            "incorrect_info": [],
            "assumptions": [],
            "already_tried": []
        }

        # General assumptions based on user type
        if user_type == "Novice":
            user_knowledge["assumptions"].append("Customer support should be easy to understand.")
        elif user_type == "Expert":
            user_knowledge["correct_info"].append("Technical details about audio codecs and drivers.")
        elif user_type == "Skeptical":
            user_knowledge["assumptions"].append("The agent might try to upsell me.")
        elif user_type == "Demanding":
            user_knowledge["assumptions"].append("I deserve immediate and high-quality service.")
        elif user_type == "RefundSeeker":
            user_knowledge["assumptions"].append("I am entitled to a refund.")
        elif user_type == "Inquisitive":
            user_knowledge["assumptions"].append("I can ask many follow-up questions.")
        elif user_type == "Impatient":
            user_knowledge["assumptions"].append("I need a quick resolution.")
        elif user_type == "Price-Sensitive":
            user_knowledge["assumptions"].append("I want the best possible deal.")
        elif user_type == "Review-Reliant":
            user_knowledge["assumptions"].append("Customer reviews are very important.")
        elif user_type == "Brand-Loyal":
            user_knowledge["assumptions"].append(f"I prefer {random.choice(self.headphone_models).split(' ')[0]} products.") # e.g., "I prefer Sony products."
        elif user_type == "Tech-Savvy":
            user_knowledge["correct_info"].append("I understand basic troubleshooting steps.")
        elif user_type == "Non-Technical":
            user_knowledge["assumptions"].append("Technical jargon should be avoided.")

        # Add question-specific knowledge derivation here (more advanced)
        # Example: If the question is about Bluetooth, add "Bluetooth pairing" to already_tried
        if "bluetooth" in question.lower():
            user_knowledge["already_tried"].append("Checked Bluetooth connection on my device.")

        return user_knowledge

    def generate_prompts_batch(self, num_prompts):
        """Generates a batch of prompts (random objectives)."""
        return [self.generate_prompt() for _ in range(num_prompts)]

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
api_key = "YOUR_API_KEY"  # Replace with your actual API key
generator = PromptGenerator(api_key)

# Generate a batch of prompts with random objectives:
prompts = generator.generate_prompts_batch(10)
for prompt in prompts:
    print(prompt)
    print("---")

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