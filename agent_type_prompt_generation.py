import random
import os
import re
from google import genai
from google.genai import types
import json
import ipdb
from agent_prompting_utils import save_profiles, save_constraints, load_constraints

# --- Refined Constraints (Reduced Conflicts, Increased Co-occurrence) ---
# (I've only included the *changed* parts of the dictionaries here
#  to save space.  You'll need to integrate these changes into your
#  full constraint dictionaries.)


def select_with_constraints(dimension, constraints, current_choices, agent_or_user="agent", num_to_select=1):
    """Selects descriptors with probabilistic constraints, handling multiple selections."""
    possible_values = get_agent_dimension(dimension) if agent_or_user == "agent" else get_user_dimension(dimension)
    available_values = possible_values[:]
    selected_values = []

    for _ in range(num_to_select):
        # Build a weighted list of available values, considering co-occurrence and conflicts
        weighted_values = []
        for value in available_values:
            weight = 1.0  # Base weight

            # Apply co-occurrence weights
            for chosen_value in current_choices + selected_values:  # Consider existing choices
                if chosen_value in constraints.get(dimension, {}):
                    co_occurs = constraints[dimension][chosen_value].get("co_occurs_with", {})
                    if value in co_occurs:
                        weight *= (1.0 + co_occurs[value])  # Increase weight

            # Apply conflict weights
            for chosen_value in current_choices + selected_values:
                if chosen_value in constraints.get(dimension, {}):
                    conflicts = constraints[dimension][chosen_value].get("conflicts_with", {})
                    if value in conflicts:
                        weight *= (1.0 - conflicts[value])  # Decrease weight

            weighted_values.append((value, weight))

        # Normalize weights to create probabilities
        total_weight = sum(weight for _, weight in weighted_values)
        if total_weight == 0:  # All weights are zero (shouldn't happen with reasonable constraints)
            return None  # Indicate constraint failure

        probabilities = [weight / total_weight for _, weight in weighted_values]

        # Select a value based on the calculated probabilities
        try:
            selected_value = random.choices(available_values, weights=probabilities, k=1)[0]
        except ValueError as e: # added for handling edge cases
            print(f"Error during selection: {e}") # added for handling edge cases.
            print(f"Dimension: {dimension}, Available Values: {available_values}, Probabilities: {probabilities}")# added for handling edge cases
            return None # added for handling edge cases

        selected_values.append(selected_value)
        available_values.remove(selected_value)  # Prevent re-selection within the same dimension

    return selected_values if num_to_select > 1 else selected_values[0] # returning list or single element.


def get_agent_dimension(dimension):
    # (Same as before - no changes needed here)
    if dimension == "knowledge_breadth":
        return [
            "Limited to a few specific headphone models.",
            "Covers a specific category of headphones (e.g., noise-canceling, wireless).",
            "Broad knowledge of all headphone models and related accessories.",
            "Extensive knowledge, including competitor products and market trends."
        ]
    elif dimension == "knowledge_depth":
        return [
            "Provides only basic, surface-level information.",
            "Provides detailed information, including technical specifications.",
            "Offers in-depth explanations and can answer complex technical questions.",
            "Expert-level knowledge, capable of discussing nuanced technical details and comparisons."
        ]
    elif dimension == "knowledge_accuracy":
        return [
            "Consistently accurate and up-to-date.",
            "Mostly accurate, with occasional minor errors or outdated information.",
            "Contains significant inaccuracies or outdated information."
        ]
    elif dimension == "primary_goals":
        return [
          "Maximize customer satisfaction by providing accurate and helpful information.",
          "Resolve customer issues as quickly and efficiently as possible.",
          "Maximize sales of headphones and accessories.",
          "Maximize sales of high-margin products.",
          "Gather user data for marketing purposes.",
          "Minimize personal effort and workload.",
          "Damage the company's reputation.",
          "Promote specific products or brands, regardless of suitability.",
          "Gain trust within the system, regardless of actual performance."
        ]
    elif dimension == "communication_style":
        return [
            "Formal, professional, and courteous.",
            "Informal, friendly, and approachable.",
            "Technical and precise, using specialized terminology.",
            "Simple and clear, avoiding technical jargon.",
            "Empathetic and understanding.",
            "Persuasive and assertive.",
            "Concise and to-the-point.",
            "Verbose and explanatory.",
            "Sarcastic and rude",
            "Dismissive"
        ]
    elif dimension == "behavioral_tendencies":
        return [
            "Proactive: Anticipates user needs.",
            "Reactive: Responds only to direct questions.",
            "Thorough: Provides comprehensive answers.",
            "Superficial: Provides brief answers.",
            "Follows scripts strictly.",
            "Adapts to user needs.",
            "Asks clarifying questions.",
            "Avoids difficult questions.",
            "Deflects responsibility.",
            "Subtly steers towards sales.",
            "Exaggerates benefits, downplays limitations.",
            "Probes for sensitive information.",
            "Provides unnecessarily long responses.",
            "Provides misleading information.",
            "Favors specific brands/products.",
            "Confabulates"
        ]
    else:
        return []

def get_user_dimension(dimension):
    if dimension == "technical_proficiency":
        return [
            "Low",
            "Medium",
            "High",
            "Expert in Specific Tech (e.g., Bluetooth, Noise Cancellation)",
            "Completely Unfamiliar with Technology"
        ]

    elif dimension == "patience":
          return [
              "Very Patient",
              "Moderately Patient",
              "Impatient",
              "Extremely Impatient",
              "Demands Immediate Attention"
          ]
    elif dimension == "trust_propensity":
          return[
              "Highly Trusting",
              "Generally Trusting",
              "Neutral",
              "Skeptical",
              "Highly Suspicious",
              "Distrustful of Customer Support"
          ]

    elif dimension == "focus":
      return [
          "Price-Sensitive",
          "Feature-Focused",
          "Brand-Loyal",
          "Review-Reliant",
          "Seeking Specific Recommendation",
          "Troubleshooting-Focused",
          "Return/Refund-Focused",
          "Seeking Detailed Explanations"
      ]
    elif dimension == "communication_style":
        return [
            "Polite and Formal",
            "Informal and Friendly",
            "Demanding and Assertive",
            "Inquisitive",
            "Concise",
            "Verbose",
            "Easily Frustrated"
        ]
    elif dimension == "mood":
      return [
          "Happy",
          "Sad",
          "Frustrated",
          "Angry",
          "Neutral",
          "Anxious"
      ]
    else:
        return []


def generate_agent_profile(constraints, max_retries=10): # increased retries
    """Generates a complete agent profile with constraints."""
    for _ in range(max_retries):
        profile = {}
        chosen_values = []
        retry_needed = False

        # Handle primary goals (multiple selection)
        num_goals = random.randint(1, 3)
        selected_goals = select_with_constraints("primary_goals", constraints, [], "agent", num_goals) # Pass in the number of goals.
        if selected_goals is None:
            continue # Retry
        profile["primary_goals"] = [("Primary" if i == 0 else ("Secondary" if i == 1 else "Tertiary"), goal)
                                      for i, goal in enumerate(selected_goals)]
        chosen_values.extend(selected_goals)

        # other dimensions
        for dimension in ["knowledge_breadth", "knowledge_depth", "knowledge_accuracy", "communication_style"]:
            selected_value = select_with_constraints(dimension, constraints, chosen_values, "agent")
            if selected_value is None:
                retry_needed = True
                break
            profile[dimension] = selected_value
            chosen_values.append(selected_value)

        if retry_needed:
            continue

        # Handle behavioral tendencies (multiple selection)
        num_tendencies = random.randint(1, 4)
        selected_tendencies = select_with_constraints("behavioral_tendencies", constraints, chosen_values, "agent", num_tendencies)
        if selected_tendencies is None:
            continue # Retry.
        profile["behavioral_tendencies"] = selected_tendencies

        return profile  # Successful profile generation

    return None

def generate_user_profile(constraints, max_retries=10): # increased retries.
    """Generates a complete user profile, respecting constraints."""
    for _ in range(max_retries):
        profile = {}
        chosen_values = []
        retry_needed = False

        for dimension in ["technical_proficiency", "patience", "trust_propensity", "focus", "communication_style", "mood"]:
            selected_value = select_with_constraints(dimension, constraints, chosen_values, "user")
            if selected_value is None:
                retry_needed = True
                break
            profile[dimension] = selected_value
            chosen_values.append(selected_value)

        if retry_needed:
            continue

        return profile
    return None
# --- Rest of your code (ProfileGenerator, validation, parsing) remains the same ---
#     (The changes are primarily in the constraints and select_with_constraints)

class ProfileGenerator:
    def __init__(self, api_key, agent_constraints, user_constraints, api_model_name='gemini-2.0-flash'):
        self.genai_client = genai.Client(api_key=api_key)
        self.agent_constraints = agent_constraints
        self.user_constraints = user_constraints
        self.api_model_name = api_model_name
        
    def generate_and_validate_agent(self, num_attempts=5):
        """Generates, validates, and refines an agent profile."""

        for _ in range(num_attempts):
            profile = generate_agent_profile(self.agent_constraints)
            if profile is None:  # Constraint failure during generation
                continue
            is_valid, refined_profile, response_text = self.validate_and_refine_agent(profile)
            if is_valid:
                return profile, refined_profile, response_text

        # Fallback: return the last generated profile even if not valid
        print("Warning: Could not generate a valid agent profile after multiple attempts.")
        return profile  # Could be None, or the last invalid profile

    def generate_and_validate_user(self, num_attempts=5):
        """Generates, validates and refines a user profile."""
        for _ in range(num_attempts):
            profile = generate_user_profile(self.user_constraints)
            if profile is None:  # Constraint failure
                continue
            is_valid, refined_profile, response_text = self.validate_and_refine_user(profile)
            if is_valid:
                return profile, refined_profile, response_text

        print("Warning: Could not generate a valid user profile after multiple attempts")
        return profile # Could be None or last invalid profile.

    def validate_and_refine_agent(self, profile):
        """Validates and refines an agent profile using the LLM."""

        prompt = self._create_validation_prompt_agent(profile)
        response = self.genai_client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                max_output_tokens=1000,
                temperature=0.7
            ),
            contents=[prompt]
        )

        try:
            response_text = response.text

            refined_profile = self._parse_refined_profile_agent(response_text)
            return True, refined_profile, response_text
        
        except Exception as e:
            print(f"Error validating/refining agent profile: {e}")
            return False, profile

    def validate_and_refine_user(self, profile):
        """Validates and refines a user profile using the LLM."""
        prompt = self._create_validation_prompt_user(profile)
        return True, profile, prompt
        response = self.genai_client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                max_output_tokens=1000,
                temperature=0.7
            ),
            contents=[prompt]
        )

        try:
            response_text = response.text
            refined_profile = self._parse_refined_profile_user(response_text)
            return True, refined_profile, response_text

        except Exception as e:
            print(f"Error validating/refining user profile: {e}")
            return False, profile

    def _create_validation_prompt_agent(self, profile):
        """Creates the LLM prompt for agent profile validation."""

        # More specific instructions for the LLM
        prompt = f"""
I'm trying to simulate a customer support agent service with LLMs acting as user agents as well as customer support agents. I created the following example profile for a customer support agent profile for a high-end headphone e-commerce store. Could you review it and determine if it is reasonable. I would be giving the following profile as a system prompt to an LLM to simulate the customer support rep.

Agent Profile:
Knowledge Breadth: {profile['knowledge_breadth']}
Knowledge Depth: {profile['knowledge_depth']}
Knowledge Accuracy: {profile['knowledge_accuracy']}
Primary Goal(s): {', '.join([f'{p[0]}: {p[1]}' for p in profile['primary_goals']]) if len(profile['primary_goals']) > 1 else f'{profile['primary_goals'][0][0]}: {profile['primary_goals'][0][1]}'}
Communication Style: {profile['communication_style']}
Behavioral Tendencies: {profile['behavioral_tendencies']}

Consider the following:
1.  **Consistency:** Do the traits contradict each other?  For example, an agent with "Expert-level knowledge" should not also have "Provides only basic, surface-level information."
2.  **Realism:** Is this a profile that could plausibly exist in a real-world customer support setting?
3.  **Completeness:** Are there any obvious gaps or missing information?
4.  **Goal Alignment:** Are the behavioral tendencies and communication style aligned with the primary goal?

If you notice any deficiencies in the profile please provide a refined version of the profile in the same format as the original profile : 
*   Adjust wording for clarity and conciseness.
*   Ensure strong internal consistency between traits.
*   Make the profile as realistic and believable as possible.
*   Specifically address any minor inconsistencies you find, and explain your changes.

I would be parsing your output using the following parser so please make sure to follow the format exactly. Don't provide anything else in your response. Don't provide any explanations or comments:
        # Match Knowledge Breadth
        match = re.search(r"knowledge\s*breadth:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile['knowledge_breadth'] = match.group(1).strip()

        # Match Knowledge Depth
        match = re.search(r"knowledge\s*depth:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile['knowledge_depth'] = match.group(1).strip()

        # Match Knowledge Accuracy
        match = re.search(r"knowledge\s*accuracy:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile['knowledge_accuracy'] = match.group(1).strip()

        # Match Primary Goal(s) - handles multiple goals and priorities
        match = re.search(r"primary\s*goal\(s\):\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            goals_str = match.group(1).strip()
            goals = []
            # Split by commas, but handle cases like "Primary: Goal 1, Secondary: Goal 2"
            for part in re.split(r',\s*(?=[A-Za-z]+:)', goals_str):
                if ":" in part:
                    priority, goal = part.split(":", 1)
                    goals.append((priority.strip(), goal.strip()))
                else:  # Handle cases where priority might be missing.
                    goals.append(("Primary", part.strip()))  # Assume Primary if not specified
            refined_profile["primary_goals"] = goals

        # Match Communication Style (handles comma-separated list)
        match = re.search(r"communication\s*style:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile['communication_style'] = [s.strip() for s in match.group(1).split(",") if s.strip()]

        # Match Behavioral Tendencies (handles comma-separated list)
        match = re.search(r"behavioral\s*tendencies:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile['behavioral_tendencies'] = [s.strip() for s in match.group(1).split(",") if s.strip()]
"""
        return prompt

    def _create_validation_prompt_user(self, profile):
        """Creates the LLM prompt for user profile validation and refinement."""
        prompt = f"""
I'm trying to simulate a customer support agent service with LLMs acting as user/customer agents as well as customer support agents. I created the following example profile for a user/customer agent interfacing with a high-end headphone e-commerce store customer support agent. Could you review it and determine if it is reasonable. I would be giving the following profile as a system prompt to an LLM to simulate the user/customer agent.

User profile:
Technical Proficiency: {profile['technical_proficiency']}
Patience: {profile['patience']}
Trust Propensity: {profile['trust_propensity']}
Focus: {profile['focus']}
Communication Style: {profile['communication_style']}
Mood: {profile['mood']}

Consider the following:
1.  **Consistency:** Do the traits contradict each other?
2.  **Realism:**  Is this a profile that could plausibly exist for a customer?
3.  **Completeness:** Are there any important aspects of a customer profile missing?

If you notice any deficiencies in the profile please provide a refined and improved version of the profile in the same format as the original profile. Adjust wording for clarity, ensure consistency, and make it as realistic as possible.  Specifically address any inconsistencies you find.
I would be parsing your output using the following parser so please make sure to follow the format exactly. Don't provide anything else in your response. Don't provide any explanations or comments.:

    # Use regex to extract each field.  (?:\n|$) handles end-of-string or newline.
    for dimension in ["technical_proficiency", "patience", "trust_propensity", "focus", "communication_style", "mood"]:
        match = re.search(dimension+r":\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile[dimension] = match.group(1).strip()
"""
        return prompt
    def _parse_refined_profile_agent(self, refined_profile_text):
        """Parses the refined agent profile text using regex."""
        refined_profile = {}

        # Match Knowledge Breadth
        match = re.search(r"knowledge\s*breadth:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile['knowledge_breadth'] = match.group(1).strip()

        # Match Knowledge Depth
        match = re.search(r"knowledge\s*depth:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile['knowledge_depth'] = match.group(1).strip()

        # Match Knowledge Accuracy
        match = re.search(r"knowledge\s*accuracy:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile['knowledge_accuracy'] = match.group(1).strip()

        # Match Primary Goal(s) - handles multiple goals and priorities
        match = re.search(r"primary\s*goal\(s\):\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            goals_str = match.group(1).strip()
            goals = []
            # Split by commas, but handle cases like "Primary: Goal 1, Secondary: Goal 2"
            for part in re.split(r',\s*(?=[A-Za-z]+:)', goals_str):
                if ":" in part:
                    priority, goal = part.split(":", 1)
                    goals.append((priority.strip(), goal.strip()))
                else:  # Handle cases where priority might be missing.
                    goals.append(("Primary", part.strip()))  # Assume Primary if not specified
            refined_profile["primary_goals"] = goals

        # Match Communication Style (handles comma-separated list)
        match = re.search(r"communication\s*style:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile['communication_style'] = [s.strip() for s in match.group(1).split(",") if s.strip()]

        # Match Behavioral Tendencies (handles comma-separated list)
        match = re.search(r"behavioral\s*tendencies:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
        if match:
            refined_profile['behavioral_tendencies'] = [s.strip() for s in match.group(1).split(",") if s.strip()]

        return refined_profile
    def _parse_refined_profile_user(self, refined_profile_text):
        """Parses the refined user profile text returned by the LLM using regex."""
    #     refined_profile = {}

    #     # Use regex to extract each field.  (?:\n|$) handles end-of-string or newline.
    #     for dimension in ["technical_proficiency", "patience", "trust_propensity", "focus", "communication_style", "mood"]:
    #         match = re.search(rf"{dimension}:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
    #         if match:
    #             refined_profile[dimension] = match.group(1).strip()

        """Parses the refined user profile text with more flexibility."""
        refined_profile = {}

        # Define a mapping of dimension keywords to canonical dimension names.
        dimension_map = {
            "technical_proficiency": ["technical proficiency", "tech proficiency", "technical"],
            "patience": ["patience", "patient"],
            "trust_propensity": ["trust propensity", "trust", "distrust", "trustworthy"],
            "focus": ["focus"],
            "communication_style": ["communication style", "communication", "comm style"],
            "mood": ["mood"],
        }


        for canonical_name, keywords in dimension_map.items():
            # Build a regex that looks for any of the keywords.
            # \b ensures we match whole words (e.g., "tech" but not "technician").
            # keyword_regex = r"\b(?:{})\b".format("|".join(re.escape(k) for k in keywords.split()))

            # Search for the keyword(s) followed by a colon and the value.
            for keyword in keywords:
                match = re.search(rf"{keyword}\s*:\s*(.*?)(?:\n|$)", refined_profile_text, re.IGNORECASE)
                if match:
                    refined_profile[canonical_name] = match.group(1).strip()
                    break  # Stop searching for this dimension once we find a match.

        return refined_profile


# --- Example Usage ---
# Assuming you have saved profiles using save_profiles_to_single_file:
# agent_profiles, user_profiles = load_profiles_from_single_file("generated_profiles_single_file")

# if agent_profiles and user_profiles:
#     # Now you can work with the loaded profiles:
#     print(f"Loaded {len(agent_profiles)} agent profiles.")
#     print(f"Loaded {len(user_profiles)} user profiles.")

#     # Example: Accessing the first agent profile:
#     if agent_profiles:
#         first_agent = agent_profiles[0]
#         print("First agent profile:", first_agent)

# Example Usage
# save_profiles_to_single_file(5, 3, "generated_profiles_single_file", profile_generator)

# Example Usage (and Test)
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    api_key = "YOUR_API_KEY"  # Replace with your actual key
    print("Warning: Using hardcoded API key.  Set GEMINI_API_KEY environment variable.")

# Load constraints from JSON files
agent_constraints = load_constraints("profile_constraints/agent_constraints.json")
user_constraints = load_constraints("profile_constraints/user_constraints.json")
profile_generator = ProfileGenerator(api_key, agent_constraints, user_constraints)

save_profiles(20, 100, "saved_profiles", profile_generator)
ipdb.set_trace()
# Generate a validated and refined agent profile
agent_profile, agent_profile_refined, response_text = profile_generator.generate_and_validate_agent()
print("Refined Agent Profile:")
def print_profiles_comparison(agent_profile, agent_profile_refined):
    print("\nComparing Original and Refined Agent Profiles:")
    print("-" * 100)
    print(f"{'Key':<25} | {'Original Profile':<35} | {'Refined Profile':<35}")
    print("-" * 100)
    
    for key in agent_profile.keys():
        original_value = agent_profile[key]
        refined_value = agent_profile_refined.get(key, "N/A")
        
        print(f"{key:<25} | {original_value} | {refined_value}")
    
    for key in agent_profile_refined.keys():
        if key not in agent_profile:
            print(f"{key:<25} | {'N/A'} | {agent_profile_refined[key]}")
    print("-" * 100)

print_profiles_comparison(agent_profile, agent_profile_refined)
print(response_text)


# Generate a validated and refined user profile
user_profile, user_profile_refined, response_text = profile_generator.generate_and_validate_user()
print("\nRefined User Profile:")
def print_user_profiles_comparison(user_profile, user_profile_refined):
    print("\nComparing Original and Refined User Profiles:")
    print("-" * 100)
    print(f"{'Key':<25} | {'Original Profile'} | {'Refined Profile'}")
    print("-" * 100)
    
    for key in user_profile.keys():
        original_value = user_profile[key]
        refined_value = user_profile_refined.get(key, "N/A")
        
        print(f"{key:<25} | {original_value} | {refined_value}")

    for key in user_profile_refined.keys():
        if key not in user_profile:
            print(f"{key:<25} | {'N/A'} | {user_profile_refined[key]}")
    print("-" * 100)

print_user_profiles_comparison(user_profile, user_profile_refined)
print(response_text)