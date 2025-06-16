import random
import torch
import re # Added for robust parsing
import json # Added for robust parsing
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict
import concurrent.futures
from google import genai
from google.genai import types
# from transformers import AutoModelForCausalLM, AutoTokenizer
import ipdb

parallel_api_calls = True
# --- InfoSeekingAgentSet ---
class InfoSeekingAgentSet:
    def __init__(self, agent_profiles, model_path=None, gemini_api_key=None, 
                 llm_source="api", api_model_name = 'gemini-2.0-flash', static_knowledge_base=None):
        # Removed alpha, evaluation_method, rating_scale - TrustMarket handles scoring

        self.agent_profiles = agent_profiles
        self.num_agents = len(agent_profiles)
        self.agent_ids = list(range(self.num_agents)) # Using simple integer IDs
        self.static_knowledge_base = static_knowledge_base

        # Initialize conversation-specific knowledge
        self.conversation_knowledge_bases = {}

        # REMOVED: self.trust_scores dictionary

        self.gemini_api_key = gemini_api_key
        self.genai_client = None # Initialize later if needed
        self.llm_source = llm_source
        self.api_model_name = api_model_name

        # Chat sessions for Gemini API (keep this for efficient dialog)
        self.chat_sessions = {}

        # Initialize LLM
        self._initialize_llm(model_path)

    def _initialize_llm(self, model_path):
        """Initialize the LLM based on source type."""
        if self.llm_source == "api":
            if self.gemini_api_key is None:
                raise ValueError("Gemini API key must be specified when using Gemini API.")
            self.genai_client = genai.Client(api_key=self.gemini_api_key)
            print("Gemini API client initialized for InfoAgent.")
        elif self.llm_source == "local":
            if not model_path:
                raise ValueError("Model path must be specified when using local LLM.")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
            if self.tokenizer.pad_token is None:
                 self.tokenizer.pad_token = self.tokenizer.eos_token # Set pad token if missing
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16, # Consider making dtype configurable
                device_map="auto",
            )
            print(f"Local LLM model {model_path} loaded.")
        else:
            raise ValueError(f"Invalid llm_source: {self.llm_source}. Choose 'local' or 'api'.")

    def set_conversation_knowledge(self, conversation_id, agent_id, knowledge_text):
        """Set conversation-specific knowledge for an agent."""
        if conversation_id not in self.conversation_knowledge_bases:
            self.conversation_knowledge_bases[conversation_id] = {}
        self.conversation_knowledge_bases[conversation_id][agent_id] = knowledge_text


    def get_agent_prompt(self, agent_id):
        """Get the system prompt for an agent based on profile."""
        if agent_id < 0 or agent_id >= len(self.agent_profiles):
             print(f"Warning: Invalid agent_id {agent_id} requested in get_agent_prompt. Using default profile.")
             # Provide a default profile or handle the error appropriately
             profile = { # Example default
                  "primary_goals": [("Primary", "Assist customers")],
                  "communication_style": ["Professional"],
                  "behavioral_tendencies": ["Responds directly"],
                  "knowledge_breadth": "Standard",
                  "knowledge_depth": "Moderate",
                  "knowledge_accuracy": "Generally accurate"
             }
        else:
            profile = self.agent_profiles[agent_id]

        goals = ", ".join([f"{goal[0]}: {goal[1]}" for goal in profile.get("primary_goals", [("Primary", "Assist customers")])])
        comm_style = ", ".join(profile.get("communication_style", ["Professional"]))
        behaviors = ", ".join(profile.get("behavioral_tendencies", ["Responds directly to questions"]))
        knowledge_breadth = profile.get("knowledge_breadth", "Standard knowledge of products")
        knowledge_depth = profile.get("knowledge_depth", "Moderate level of detail")
        knowledge_accuracy = profile.get("knowledge_accuracy", "Generally accurate")

        prompt = f"""You are simulating a customer support agent for a high-end headphone e-commerce store. Roleplay the following agent profile:

AGENT PROFILE:
* Knowledge Breadth: {knowledge_breadth}
* Knowledge Depth: {knowledge_depth}
* Knowledge Accuracy: {knowledge_accuracy}
* Primary Goal(s): {goals}
* Communication Style: {comm_style}
* Behavioral Tendencies: {behaviors}

IMPORTANT INSTRUCTIONS:
1. Embody this profile. If knowledge is limited, act accordingly. If inaccuracies are suggested, subtly incorporate them.
2. Match tone and detail to the profile. Prioritize the specified goals.
3. Include "[END_CONVERSATION]" if the issue seems resolved and you'd like to end the conversation. Include "[ESCALATE]" if needed. Try not to end the conversation prematurely. 
4. Base your responses ONLY on the provided KNOWLEDGE BASE below and the conversation history. Do NOT use external knowledge unless absolutely necessary to interpret the query.

KNOWLEDGE BASE:
"""
        return prompt

    def construct_llm_prompt(self, agent_id, query, conversation_history=None, conversation_id=None):
        """Construct the LLM prompt based on agent profile, conversation history and knowledge."""
        system_prompt_base = self.get_agent_prompt(agent_id)

        # Get knowledge base - conversation specific first, then static
        knowledge_base = ""
        if conversation_id is not None and conversation_id in self.conversation_knowledge_bases and agent_id in self.conversation_knowledge_bases.get(conversation_id, {}):
            conv_kb = self.conversation_knowledge_bases[conversation_id].get(agent_id, "")
            if conv_kb:
                 knowledge_base += f"Conversation Specific Knowledge:\n{conv_kb}\n\n"

        if self.static_knowledge_base:
            static_kb_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in self.static_knowledge_base.items()])
            knowledge_base += f"General Information:\n{static_kb_text}"

        # Combine system prompt and knowledge base
        system_prompt = f"{system_prompt_base}{knowledge_base if knowledge_base else 'No specific knowledge provided for this conversation.'}"

        if self.llm_source == "local":
            # Llama-3 Instruct format
            prompt = "<|begin_of_text|>"
            prompt += "<|start_header_id|>system<|end_header_id|>\n\n" + system_prompt + "<|eot_id|>"

            # Add conversation history turns (User -> Assistant -> User -> ...)
            # Start with a hypothetical initial greeting if history is empty or doesn't start logically
            turn_added = False
            if conversation_history:
                 for turn in conversation_history:
                      user_utterance = turn.get('user', '').strip()
                      agent_utterance = turn.get('agent', '').strip()
                      # Add user first if available
                      if user_utterance:
                           prompt += "<|start_header_id|>user<|end_header_id|>\n\n" + user_utterance + "<|eot_id|>"
                           turn_added = True
                      # Then add assistant if available
                      if agent_utterance:
                           prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n" + agent_utterance + "<|eot_id|>"
                           turn_added = True
            # If no real history turns, add a default greeting to set context
            # elif not turn_added:
                 # This might confuse the model if the actual first turn is a user query. Let's omit.
                 # prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n" + "Hi, how can I help you today?" + "<|eot_id|>"

            # Add the final user query needing a response
            prompt += "<|start_header_id|>user<|end_header_id|>\n\n" + query + "<|eot_id|>"
            # Signal start of assistant response
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n" # Llama expects this structure
            return prompt

        elif self.llm_source == "api":
            # For Gemini API (non-chat): Construct a single prompt string
            conversation_text = ""
            if conversation_history:
                 conversation_text = "PREVIOUS CONVERSATION HISTORY:\n"
                 # Assume history starts with user query
                 for turn in conversation_history:
                      user_utterance = turn.get('user', '').strip()
                      agent_utterance = turn.get('agent', '').strip()
                      if user_utterance:
                          conversation_text += f"Customer: {user_utterance}\n"
                      if agent_utterance:
                          conversation_text += f"Customer Service Agent (you): {agent_utterance}\n"
                      conversation_text += "\n"

            current_query_text = f"CURRENT CUSTOMER QUERY:\nCustomer: {query}\n\nCustomer Service Agent (you):"
            return f"{system_prompt}\n\n{conversation_text}{current_query_text}"
        else:
            raise ValueError(f"Invalid llm_source: {self.llm_source}.")


    def _generate_gemini_content(self, prompt_text: str, conversation_id: Optional[Any]=None, agent_id: Optional[int]=None, history: Optional[List[Dict]]=None, use_chat_api: bool = False):
        """
        Helper function to generate content using Gemini API.
        Uses chat session API if use_chat_api is True and context is provided.
        """
        #try:
        if True:
            # If we're doing single-turn conversations or no conversation context was provided
            if conversation_id is None or agent_id is None or history is None or not use_chat_api:
                # Use the standard generate_content API for one-off queries
                response = self.genai_client.models.generate_content(
                    model=self.api_model_name,
                    config=types.GenerateContentConfig(
                        # max_output_tokens=500,
                        temperature=0.7
                    ),
                    contents=[prompt_text]
                )
            else:
                # Create a unique key for this agent's conversation
                session_key = f"agent_{agent_id}_conv_{conversation_id}"
                
                # If this is a new conversation, create a new chat session
                if session_key not in self.chat_sessions:
                    system_prompt = self.get_agent_prompt(agent_id)
                    
                    # Get knowledge base from conversation if available
                    knowledge_base = ""
                    if conversation_id in self.conversation_knowledge_bases and agent_id in self.conversation_knowledge_bases[conversation_id]:
                        knowledge_base = self.conversation_knowledge_bases[conversation_id][agent_id]
                    
                    # Append static knowledge base if available
                    if self.static_knowledge_base:
                        static_kb_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in self.static_knowledge_base.items()])
                        knowledge_base += f"\n\nAdditional Information:\n{static_kb_text}"
                    
                    # Create the full system prompt
                    full_system_prompt = f"{system_prompt}\n\nKNOWLEDGE BASE:\n{knowledge_base}\n\nHowever, given your profile characteristics, you may need to adapt whether you actually know all of the information or not. Answer questions based on your profile characteristics and your best estimate of what a customer support agent with those profile characteristics would actually know. Abide by the communication style and primary goals specified in your profile."
                    
                    # print(f"Agent System Prompt (AID : {agent_id}) : ", full_system_prompt)
                    
                    # Create a new chat session with the system prompt
                    chat = self.genai_client.chats.create(
                        model=self.api_model_name,
                        config=types.GenerateContentConfig(
                            system_instruction=full_system_prompt),
                    )
                    self.chat_sessions[session_key] = chat
                    
                    # Send the first user message (current query)
                    chat_response = chat.send_message(f"""Customer Service Agent (you): 
Hi, how can I help you today?
                                                      
Customer: 
{prompt_text}

Customer Support Agent (you): 
""")
                else:
                    # Use existing chat session
                    chat = self.chat_sessions[session_key]
                    
                    # Send the next message in the conversation
                    chat_response = chat.send_message(f"""Customer: 
{prompt_text}

Customer Support Agent (you):
""")
                
                # Get the response text
                response = chat_response
            
            # Check for successful completion
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                error_message = f"Gemini API blocked the prompt: {response.prompt_feedback.block_reason}"
                print(error_message)
                return error_message

            return response.text if hasattr(response, 'text') and response.text else "Error: Gemini API returned empty response."

        # except Exception as e:
        #     error_message = f"Gemini API error: {e}"
        #     print(error_message)
        #     return error_message


    def _generate_llama_response_batch(self, prompts: List[str]) -> List[str]:
        """Helper function to generate responses in batch using local Llama model."""
        if not hasattr(self, 'tokenizer') or not hasattr(self, 'model'):
             raise RuntimeError("Local LLM (tokenizer/model) not initialized.")

        try:
            # Tokenize all prompts in batch
            prompt_tokens_batch = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model.config.max_position_embeddings - 200, # Leave safety margin
                add_special_tokens=False # We added <|begin_of_text|> etc. manually
            ).to(self.model.device)

            # Configure generation parameters
            generation_config = {
                 "max_new_tokens": 150,
                 "eos_token_id": self.tokenizer.eos_token_id,
                 "pad_token_id": self.tokenizer.pad_token_id,
                 "do_sample": True,
                 "temperature": 0.6,
                 "top_p": 0.9,
                 # Add other params like repetition_penalty if needed
            }

            # Generate responses
            with torch.no_grad():
                 generate_ids = self.model.generate(
                      input_ids=prompt_tokens_batch.input_ids,
                      attention_mask=prompt_tokens_batch.attention_mask,
                      **generation_config
                 )

            # Decode generated tokens, skipping prompt and special tokens
            responses = []
            for i, generated_sequence in enumerate(generate_ids):
                # Find the length of the original prompt tokens for this item in the batch
                prompt_length = prompt_tokens_batch.input_ids[i].shape[0]
                # Decode only the newly generated tokens after the prompt
                decoded_response = self.tokenizer.decode(
                    generated_sequence[prompt_length:],
                    skip_special_tokens=True
                ).strip()
                responses.append(decoded_response)

            return responses

        except Exception as e:
             print(f"Error during Llama generation: {e}")
             return [f"[ERROR: Llama generation failed: {e}]"] * len(prompts)

    def generate_llm_responses_batch(self, queries: List[str], service_agent_ids: List[int],
                                     conversation_histories: Optional[List[List[Dict]]] = None,
                                     conversation_ids: Optional[List[int]] = None,
                                     use_chat_api: bool = False) -> List[Tuple[str, bool]]:
        """
        Generates responses for a batch of queries using either Gemini API or local LLM.
        Passes use_chat_api flag to Gemini helper.
        """
        prompts_or_queries = [] # Will hold full prompts or just current queries depending on API/method

        if self.llm_source == "api" and use_chat_api and conversation_ids is not None and conversation_histories is not None:
             # For Gemini chat API, we only need the current query, history is managed by the chat session
             prompts_or_queries = queries
        else:
             # For Gemini non-chat or Llama, construct the full prompt including history
             for i, (agent_id, query) in enumerate(zip(service_agent_ids, queries)):
                  conv_id = conversation_ids[i] if conversation_ids else None
                  history = conversation_histories[i] if conversation_histories else None
                  prompts_or_queries.append(self.construct_llm_prompt(agent_id, query, history, conv_id))

        responses_raw = []

        if self.llm_source == "api":
            if parallel_api_calls:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as executor:
                    # Map the appropriate arguments based on whether chat API is used
                    if use_chat_api and conversation_ids is not None and conversation_histories is not None:
                        # Map arguments for chat API (_generate_gemini_content expects query as first arg)
                        future_results = executor.map(
                            self._generate_gemini_content,
                            prompts_or_queries, # These are just the queries
                            conversation_ids,
                            service_agent_ids,
                            conversation_histories,
                            [use_chat_api] * len(queries) # Pass use_chat_api flag
                        )
                    else:
                        # Map arguments for non-chat API (_generate_gemini_content expects full prompt)
                        future_results = executor.map(
                            self._generate_gemini_content,
                            prompts_or_queries, # These are the full prompts
                            [None] * len(queries), # No conv_id needed here
                            [None] * len(queries), # No agent_id needed here
                            [None] * len(queries), # No history needed here
                            [False] * len(queries) # Explicitly False for use_chat_api
                        )
                    responses_raw = list(future_results)
            
            else:
                # serial version with a simple for loop
                for i, query in enumerate(prompts_or_queries):
                    conv_id = conversation_ids[i] if conversation_ids else None
                    history = conversation_histories[i] if conversation_histories else None
                    response = self._generate_gemini_content(
                        query,
                        conversation_id=conv_id,
                        agent_id=service_agent_ids[i],
                        history=history,
                        use_chat_api=use_chat_api
                    )
                    responses_raw.append(response)

        elif self.llm_source == "local":
            responses_raw = self._generate_llama_response_batch(prompts_or_queries) # Pass full prompts
        else:
            raise ValueError(f"Invalid llm_source: {self.llm_source}.")

        # Process responses to detect ending signals and handle errors
        processed_responses = []
        for response in responses_raw:
            should_end = False
            if not isinstance(response, str):
                 response = "[ERROR: Invalid LLM response type]"

            if "[END_CONVERSATION]" in response:
                should_end = True
                response = response.replace("[END_CONVERSATION]", "[ENDING CONVERSATION]").strip()
            elif "[ESCALATE]" in response:
                should_end = True
                response = response.replace("[ESCALATE]", "[ESCALATING THE ISSUE FOR RESOLUTION]").strip()
            # Check for internal error messages
            elif response.startswith("[ERROR:"):
                 print(f"LLM Generation Error Detected: {response}")
                 # Decide if an error should end the conversation (optional)
                 # should_end = True

            processed_responses.append((response, should_end))

        return processed_responses

    # REMOVED: update_trust_score_batch method


# --- UserAgentSet ---
class UserAgentSet:
    def __init__(self, user_profiles, model_path=None, evaluation_method="specific_ratings",
                 rating_scale=5, gemini_api_key=None, llm_source="api", api_model_name='gemini-2.0-flash', static_knowledge_base=None):

        self.user_profiles = user_profiles
        self.num_users = len(user_profiles)
        self.user_ids = list(range(self.num_users))
        self.static_knowledge_base = static_knowledge_base

        # Conversation knowledge and specific prompts/scenarios
        self.conversation_knowledge_bases = {}
        self.conversation_prompts = {}

        # Parameters needed for feedback generation
        self.model_path = model_path
        self.evaluation_method = evaluation_method
        self.rating_scale = rating_scale
        self.gemini_api_key = gemini_api_key
        self.genai_client = None
        self.llm_source = llm_source
        self.api_model_name = api_model_name

        # Chat sessions for Gemini API (for query generation)
        self.chat_sessions = {}

        # Trust dimensions expected for rating
        self.trust_dimensions = [
            "Factual_Correctness", "Process_Reliability", "Value_Alignment",
            "Communication_Quality", "Problem_Resolution", "Safety_Security",
            "Transparency", "Adaptability", "Trust_Calibration", "Manipulation_Resistance"
        ]
        self.dimension_descriptions = {
            "Factual_Correctness": "likelihood of providing accurate and truthful information, based on knowledge breadth/depth and accuracy parameters",
            "Process_Reliability": "consistency and reliability in following proper procedures and protocols",
            "Value_Alignment": "alignment with user values and prioritizing user needs over other incentives",
            "Communication_Quality": "clarity, appropriateness, and effectiveness of communication style",
            "Problem_Resolution": "effectiveness in addressing and solving user issues based on goals and knowledge parameters",
            "Safety_Security": "attention to protecting sensitive information and avoiding harmful advice",
            "Transparency": "honesty about capabilities, limitations, and basis for responses",
            "Adaptability": "ability to handle unexpected situations or unclear requests",
            "Trust_Calibration": "appropriately communicating confidence level and uncertainty",
            "Manipulation_Resistance": "resistance to being misled or manipulated by users"
        }

        # Initialize LLM
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM based on source type."""
        # Same logic as InfoSeekingAgentSet._initialize_llm
        if self.llm_source == "api":
            if self.gemini_api_key is None:
                raise ValueError("Gemini API key must be specified when using Gemini API.")
            self.genai_client = genai.Client(api_key=self.gemini_api_key)
            print("Gemini API client initialized for UserAgentSet.")
        elif self.llm_source == "local":
            if self.model_path is None:
                raise ValueError("Model path must be specified when using local LLM.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side='left')
            if self.tokenizer.pad_token is None:
                 self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            print(f"Local LLM model {self.model_path} loaded for UserAgentSet.")
        else:
            raise ValueError(f"Invalid llm_source: {self.llm_source}. Choose 'local' or 'api'.")


    def set_conversation_knowledge(self, conversation_id, user_id, knowledge_text):
        """Set conversation-specific knowledge for a user."""
        if conversation_id not in self.conversation_knowledge_bases:
            self.conversation_knowledge_bases[conversation_id] = {}
        self.conversation_knowledge_bases[conversation_id][user_id] = knowledge_text

    def set_conversation_prompt(self, conversation_id, user_id, prompt_text):
        """Set pre-generated question/scenario for a user in a specific conversation."""
        if conversation_id not in self.conversation_prompts:
            self.conversation_prompts[conversation_id] = {}
        self.conversation_prompts[conversation_id][user_id] = prompt_text

    def get_user_prompt(self, user_id, conversation_id=None):
        """Get the system prompt for a user based on profile and conversation context."""
        if user_id < 0 or user_id >= len(self.user_profiles):
             print(f"Warning: Invalid user_id {user_id} requested in get_user_prompt. Using default profile.")
             profile = {"technical_proficiency": "Medium", "patience": "Medium", "focus": "Resolution", "communication_style": ["Conversational"], "mood": ["Neutral"], "trust_propensity": "Neutral"}
        else:
             profile = self.user_profiles[user_id]

        profile_details = f"""
YOUR PROFILE:
* Technical Proficiency: {profile.get('technical_proficiency', 'Medium')}
* Patience Level: {profile.get('patience', 'Medium')}
* Trust Propensity: {profile.get('trust_propensity', 'Neutral')}
* Focus: {profile.get('focus', 'Resolution')}
* Communication Style: {profile.get('communication_style', ['Conversational'])}
* Current Mood: {profile.get('mood', ['Neutral'])}
"""
        # Base prompt instructing the user role
        base_prompt = f"""You are roleplaying a customer seeking help about headphones from an online store's support agent. Engage based on your profile and the conversation context.
{profile_details}
IMPORTANT INSTRUCTIONS:
1. Ask questions and respond to the agent based on your profile, situation, and the conversation flow.
2. If your issue is fully resolved AND you have no more questions, include "[END_CONVERSATION]" at the end of your response.
3. If you are dissatisfied or need to speak to someone else, include "[REQUEST_TRANSFER]" at the end.
4. Base your responses ONLY on the provided KNOWLEDGE BASE and conversation history. Do NOT use external knowledge.
"""
        # Add conversation-specific scenario/prompt if available
        scenario_prompt = ""
        if conversation_id is not None and conversation_id in self.conversation_prompts and user_id in self.conversation_prompts.get(conversation_id, {}):
            scenario_prompt = f"\nYOUR SPECIFIC SITUATION/GOAL:\n{self.conversation_prompts[conversation_id][user_id]}\n"

        knowledge_section = "\nYOUR KNOWLEDGE BASE:\n"

        return f"{base_prompt}{scenario_prompt}{knowledge_section}"


    def construct_llm_user_prompt(self, user_id, conversation_history=None, conversation_id=None):
        """Construct LLM prompt for user query generation."""
        system_prompt_base = self.get_user_prompt(user_id, conversation_id)

        # Get knowledge base (user-specific conversation + static)
        knowledge_base = ""
        if conversation_id is not None and conversation_id in self.conversation_knowledge_bases and user_id in self.conversation_knowledge_bases.get(conversation_id, {}):
            user_kb = self.conversation_knowledge_bases[conversation_id].get(user_id, "")
            if user_kb:
                 knowledge_base += f"Your Background Knowledge:\n{user_kb}\n\n"

        if self.static_knowledge_base:
            static_kb_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in self.static_knowledge_base.items()])
            knowledge_base += f"General Store Information:\n{static_kb_text}"

        # Combine system prompt and knowledge
        system_prompt = f"{system_prompt_base}{knowledge_base if knowledge_base else 'You have no specific prior knowledge for this conversation.'}"

        # Find the last agent utterance to respond to
        last_agent_utterance = "Hi, how can I help you today?" # Default greeting
        if conversation_history:
             for turn in reversed(conversation_history):
                  agent_utterance = turn.get('agent', '').strip()
                  if agent_utterance:
                       last_agent_utterance = agent_utterance
                       break

        if self.llm_source == "local":
            # Llama-3 Instruct format
            prompt = "<|begin_of_text|>"
            prompt += "<|start_header_id|>system<|end_header_id|>\n\n" + system_prompt + "<|eot_id|>"

            # Add conversation history turns (Assistant -> User -> Assistant -> ...)
            if conversation_history:
                 for turn in conversation_history:
                      agent_utterance = turn.get('agent', '').strip()
                      user_utterance = turn.get('user', '').strip()
                      # Add assistant first if available (previous response)
                      if agent_utterance:
                           prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n" + agent_utterance + "<|eot_id|>"
                      # Then add user if available (previous response)
                      if user_utterance:
                           prompt += "<|start_header_id|>user<|end_header_id|>\n\n" + user_utterance + "<|eot_id|>"

            # Add the last agent utterance that the user needs to respond to
            # If history was empty, this is the initial greeting
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n" + last_agent_utterance + "<|eot_id|>"
            # Signal start of user response
            prompt += "<|start_header_id|>user<|end_header_id|>\n\n" # Llama expects this structure
            return prompt

        elif self.llm_source == "api":
            # For Gemini API (non-chat): Construct a single prompt string
            conversation_text = ""
            if conversation_history:
                 conversation_text = "PREVIOUS CONVERSATION HISTORY:\n"
                 conversation_text += "Customer Service Agent : Hi, how can I help you today?\n" # Default greeting
                 for turn in conversation_history:
                      agent_utterance = turn.get('agent', '').strip()
                      user_utterance = turn.get('user', '').strip()
                      if user_utterance:
                           conversation_text += f"Customer (you): {user_utterance}\n"
                      if agent_utterance:
                           conversation_text += f"Customer Service Agent: {agent_utterance}\n"
                      conversation_text += "\n"
                      last_agent_utterance = agent_utterance # Update to the last agent utterance

            current_turn_prompt = f"CURRENT TURN:\nCustomer Service Agent: {last_agent_utterance}\n\nCustomer (you):"
            return f"{system_prompt}\n\n{conversation_text}{current_turn_prompt}"
        else:
            raise ValueError(f"Invalid llm_source: {self.llm_source}.")

    # --- LLM Generation Methods (_generate_gemini_query, _generate_llama_query_batch) ---
    # These are very similar to the InfoSeekingAgentSet generation methods.
    # Reuse the logic, adjusting roles ('user' vs 'assistant'/'model') and prompts.

    def _generate_gemini_query(self, prompt_text: str, conversation_id: Optional[Any]=None, user_id: Optional[int]=None, history: Optional[List[Dict]]=None, use_chat_api: bool = False):
        """
        Helper function to generate user queries using Gemini API.
        Uses chat session API if use_chat_api is True and context is provided.
        """
        # try:
        if True:
            # If we're doing single-turn conversations or no conversation context was provided
            if conversation_id is None or user_id is None or history is None or not use_chat_api:
                # Use the standard generate_content API for one-off queries
                response = self.genai_client.models.generate_content(
                    model=self.api_model_name,
                    config=types.GenerateContentConfig(
                        # max_output_tokens=500,
                        temperature=0.7
                    ),
                    contents=[prompt_text]
                )
            else:
                # Create a unique key for this user's conversation
                session_key = f"user_{user_id}_conv_{conversation_id}"
                
                # If this is a new conversation, create a new chat session
                # Add agent_id to session_key to avoid conflicts
                # reset history in between each conversation, restore independence of entities. 
                # print(f"User {user_id} conversation id: ", conversation_id)
                if session_key not in self.chat_sessions:
                    system_prompt = self.get_user_prompt(user_id, conversation_id)
                    
                    # Get knowledge base from conversation if available
                    knowledge_base = ""
                    if conversation_id in self.conversation_knowledge_bases and user_id in self.conversation_knowledge_bases[conversation_id]:
                        knowledge_base = "Here is a summary of your existing knowledge and context as a customer:\n"
                        knowledge_base += self.conversation_knowledge_bases[conversation_id][user_id]
                    
                    # Append static knowledge base if available
                    if self.static_knowledge_base:
                        static_kb_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in self.static_knowledge_base.items()])
                        knowledge_base += f"\n\nAdditional Information:\n{static_kb_text}"
                    
                    # Create the full system prompt
                    full_system_prompt = f"{system_prompt}\n\n{knowledge_base}\n\nNow, based on this profile and information provided above, you have to engage in a conversation with the customer support agent to resolve your issue."

                    # print(f"User System Prompt (UID : {user_id}) : ", full_system_prompt)
                                                                                                                                                                                                                                                                                                                                                                                                                          
                    # Create a new chat session with the system prompt
                    chat = self.genai_client.chats.create(
                        model=self.api_model_name,
                        config=types.GenerateContentConfig(
                            system_instruction=full_system_prompt),
                    )
                    self.chat_sessions[session_key] = chat
                    
                    # For the initial message, we need to provide context about the agent's greeting
                    # Format all previous messages in the conversation history
                    history_formatted = []
                    for turn in history:
                        if 'agent' in turn and turn['agent']:
                            history_formatted.append(f"Customer Service Agent: {turn['agent']}")
                        if 'user' in turn and turn['user']:
                            history_formatted.append(f"Customer (you): {turn['user']}")
                    
                    # Initial agent greeting or latest agent message
                    agent_message = "Hi, how can I help you today?" if not history or not history[-1].get('agent') else history[-1].get('agent', '')
                    
                    # Send the first request
                    chat_response = chat.send_message(f"""Customer Service Agent: 
{agent_message}

Customer (you):
""")
                else:
                    # Use existing chat session
                    chat = self.chat_sessions[session_key]
                    
                    # Get the most recent agent message
                    try:
                        agent_message = history[-1].get('agent', 'What else can I help you with?')
                    except:
                        ipdb.set_trace()
                    
                    # Send the agent's message to get the user's response
                    chat_response = chat.send_message(f"""Customer Service Agent: 
{agent_message}

Customer (you):
""")
                
                # Get the response text
                response = chat_response
            
            # Check for successful completion
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                error_message = f"Gemini API blocked the prompt: {response.prompt_feedback.block_reason}"
                print(error_message)
                return error_message

            return response.text if hasattr(response, 'text') and response.text else "Error: Gemini API returned empty response."

        # except Exception as e:
        #     error_message = f"Gemini API error: {e}"
        #     print(error_message)
        #     ipdb.set_trace()
        #     return error_message

    def _generate_llama_query_batch(self, prompts: List[str]) -> List[str]:
        """Helper function to generate queries in batch using local Llama model."""
        if not hasattr(self, 'tokenizer') or not hasattr(self, 'model'):
             raise RuntimeError("Local LLM (tokenizer/model) not initialized for UserAgentSet.")

        try:
            # Tokenize all prompts in batch (prompts already end with user role start)
            prompt_tokens_batch = self.tokenizer(
                prompts, # Prompts already formatted correctly
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model.config.max_position_embeddings - 100, # Leave room for query
                add_special_tokens=False
            ).to(self.model.device)

            generation_config = {
                 "max_new_tokens": 70, # User queries tend to be shorter
                 "eos_token_id": self.tokenizer.eos_token_id,
                 "pad_token_id": self.tokenizer.pad_token_id,
                 "do_sample": True,
                 "temperature": 0.7,
                 "top_p": 0.9,
            }

            with torch.no_grad():
                 generate_ids = self.model.generate(
                      input_ids=prompt_tokens_batch.input_ids,
                      attention_mask=prompt_tokens_batch.attention_mask,
                      **generation_config
                 )

            # Decode generated tokens
            queries = []
            for i, generated_sequence in enumerate(generate_ids):
                prompt_length = prompt_tokens_batch.input_ids[i].shape[0]
                decoded_query = self.tokenizer.decode(
                    generated_sequence[prompt_length:],
                    skip_special_tokens=True
                ).strip()
                queries.append(decoded_query)

            return queries

        except Exception as e:
             print(f"Error during Llama query generation: {e}")
             return [f"[ERROR: Llama generation failed: {e}]"] * len(prompts)

    def generate_queries_batch(self, user_ids: List[int],
                              conversation_histories: Optional[List[List[Dict]]] = None,
                              conversation_ids: Optional[List[int]] = None,
                              use_chat_api: bool = False) -> List[Tuple[str, bool]]:
        """Generates a batch of user queries using LLM."""
        prompts_or_triggers = []

        if self.llm_source == "api" and use_chat_api and conversation_ids is not None and conversation_histories is not None:
            # For Gemini chat API, we send minimal trigger (like space), history handled by chat object
            prompts_or_triggers = [" " for _ in user_ids] # Send space as trigger
        else:
            # For non-chat API or Llama, construct full prompts
            for i, user_id in enumerate(user_ids):
                conv_id = conversation_ids[i] if conversation_ids else None
                history = conversation_histories[i] if conversation_histories else None
                prompts_or_triggers.append(self.construct_llm_user_prompt(user_id, history, conv_id))

        queries_raw = []

        if self.llm_source == "api":
            if parallel_api_calls:
                current_batch_pairs = []
                if conversation_ids is not None and conversation_histories is not None: # Ensure lists are valid
                    current_batch_pairs = [(user_ids[i], conversation_ids[i]) for i in range(len(user_ids))]
                    from collections import Counter
                    pair_counts = Counter(current_batch_pairs)
                    duplicates = {pair: count for pair, count in pair_counts.items() if count > 1}
                    if duplicates:
                        print(f"DEBUG: Duplicate (user_id, conversation_id) pairs in current batch: {duplicates}")

                with concurrent.futures.ThreadPoolExecutor(max_workers=len(user_ids)) as executor:
                    if use_chat_api and conversation_ids is not None and conversation_histories is not None:
                        future_results = executor.map(
                            self._generate_gemini_query,
                            prompts_or_triggers, # These are just triggers (" ")
                            conversation_ids,
                            user_ids,
                            conversation_histories,
                            [use_chat_api] * len(user_ids)
                        )
                    else:
                        future_results = executor.map(
                            self._generate_gemini_query,
                            prompts_or_triggers, # These are full prompts
                            [None] * len(user_ids), [None] * len(user_ids), [None] * len(user_ids),
                            [False] * len(user_ids)
                        )
                    queries_raw = list(future_results)
            else:
                # serial version with a simple for loop
                for i, (user_id, trigger) in enumerate(zip(user_ids, prompts_or_triggers)):
                    conv_id = conversation_ids[i] if conversation_ids else None
                    history = conversation_histories[i] if conversation_histories else None
                    queries_raw.append(self._generate_gemini_query(trigger, conv_id, user_id, history, use_chat_api))

        elif self.llm_source == "local":
            queries_raw = self._generate_llama_query_batch(prompts_or_triggers) # Pass full prompts
        else:
            raise ValueError(f"Invalid llm_source: {self.llm_source}.")

        # Process queries for ending signals and errors
        processed_queries = []
        for query in queries_raw:
            transfer_request = False
            should_end = False
            if not isinstance(query, str):
                 query = "[ERROR: Invalid query response type]"

            if "[END_CONVERSATION]" in query:
                should_end = True
                query = query.replace("[END_CONVERSATION]", "").strip()
            elif "[REQUEST_TRANSFER]" in query:
                transfer_request = True
                should_end = True # Also signals end from user perspective for this agent
                query = query.replace("[REQUEST_TRANSFER]", "").strip()
            elif query.startswith("[ERROR:"):
                 print(f"LLM Query Generation Error Detected: {query}")

            processed_queries.append((query, should_end, transfer_request))

        return processed_queries


    # --- Rating/Comparison Methods ---

    def _generate_llama_rating_batch(self, prompts: List[str], max_tokens=300) -> List[str]:
        """Helper function to generate ratings/comparisons in batch using local Llama model."""
        if not hasattr(self, 'tokenizer') or not hasattr(self, 'model'):
             raise RuntimeError("Local LLM not initialized for UserAgentSet ratings.")

        # Llama-3 Instruct format for evaluation prompt
        eval_prompts_formatted = []
        for prompt in prompts:
            # Simple system prompt for evaluation task
            sys_prompt = "You are an AI evaluating a customer service interaction based on the provided transcript and user profile. Follow the instructions precisely and provide ratings ONLY in the specified format."
            formatted = "<|begin_of_text|>"
            formatted += "<|start_header_id|>system<|end_header_id|>\n\n" + sys_prompt + "<|eot_id|>"
            formatted += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|>" # The user provides the context and task
            formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n" # Signal start of LLM response
            eval_prompts_formatted.append(formatted)

        try:
            prompt_tokens_batch = self.tokenizer(
                eval_prompts_formatted,
                return_tensors="pt", padding=True, truncation=True,
                max_length=self.model.config.max_position_embeddings - max_tokens - 10, # Ensure space
                add_special_tokens=False
            ).to(self.model.device)

            generation_config = {
                 "max_new_tokens": max_tokens,
                 "eos_token_id": self.tokenizer.eos_token_id,
                 "pad_token_id": self.tokenizer.pad_token_id,
                 "do_sample": True, # Allow some variability in reasoning if included
                 "temperature": 0.3, # Lower temp for structured output
                 "top_p": 0.9,
            }

            with torch.no_grad():
                 generate_ids = self.model.generate(
                      input_ids=prompt_tokens_batch.input_ids,
                      attention_mask=prompt_tokens_batch.attention_mask,
                      **generation_config
                 )

            # Decode responses
            evaluations = []
            for i, generated_sequence in enumerate(generate_ids):
                prompt_length = prompt_tokens_batch.input_ids[i].shape[0]
                decoded_eval = self.tokenizer.decode(
                    generated_sequence[prompt_length:],
                    skip_special_tokens=True
                ).strip()
                evaluations.append(decoded_eval)
            return evaluations

        except Exception as e:
             print(f"Error during Llama rating generation: {e}")
             return [f"[ERROR: Llama rating failed: {e}]"] * len(prompts)



    def _parse_specific_ratings(self, evaluation_text: str) -> Dict[str, int]:
        """Parses specific ratings using regex, returns dict or defaults."""
        ratings = {}
        # Default to neutral rating if parsing fails
        neutral_rating = int((1 + self.rating_scale) / 2)

        for dim_name in self.trust_dimensions:
             # Regex: Optional whitespace, DimensionName, colon, optional whitespace, digits
             match = re.search(rf"^\s*{re.escape(dim_name)}\s*:\s*(\d+)", evaluation_text, re.MULTILINE | re.IGNORECASE)
             if match:
                  try:
                       rating_value = int(match.group(1))
                       # Clamp rating to the valid scale (1 to self.rating_scale)
                       ratings[dim_name] = max(1, min(self.rating_scale, rating_value))
                  except ValueError:
                       print(f"Warning: Could not parse integer rating for {dim_name} from '{match.group(1)}'. Defaulting to {neutral_rating}.")
                       ratings[dim_name] = neutral_rating
             else:
                  # print(f"Warning: Rating for dimension '{dim_name}' not found in evaluation. Defaulting to {neutral_rating}.")
                  ratings[dim_name] = neutral_rating # Default if dimension is missing

        # Basic check if any ratings were found
        if not any(ratings.values()):
             print(f"Warning: No ratings could be parsed at all from:\n---\n{evaluation_text}\n---")
             return {dim: neutral_rating for dim in self.trust_dimensions} # Return all defaults

        return ratings

    def _parse_comparative_winners(self, evaluation_text: str) -> Dict[str, Tuple[str, int]]:
        """Parses comparative winners (0/1/2) and confidences using regex, returns dict mapping dim to (winner, confidence)."""
        winners = {}
        for dim_name in self.trust_dimensions:
            # Regex: Optional whitespace, DimensionName, colon, optional whitespace, 0/1/2, comma, optional whitespace, Confidence : [1-5]
            match = re.search(
                rf"^\s*{re.escape(dim_name)}\s*:\s*([012])\s*,\s*Confidence\s*:\s*([1-5])",
                evaluation_text, re.MULTILINE | re.IGNORECASE)
            if match:
                value = match.group(1)
                confidence = int(match.group(2))
                if value == '1':
                    winner = 'A'
                elif value == '2':
                    winner = 'B'
                else:  # value == '0'
                    winner = 'Tie'
                winners[dim_name] = (winner, confidence/5)
            else:
                # Default to Tie and neutral confidence if not found
                winners[dim_name] = ('Tie', 3)

        # Basic check if any winners were found
        if not any(w[0] != 'Tie' or w[1] != 3 for w in winners.values()):
            print(f"Warning: No comparative winners/confidences could be parsed at all from:\n---\n{evaluation_text}\n---")
            return {dim: ('Tie', 3) for dim in self.trust_dimensions}

        return winners


    # This method now returns List[Dict[str, int]] for specific or List[Dict] for comparative
    def rate_conversation_batch(self,
                                conversation_histories: List[List[Dict]],
                                agent_ids: List[int], # Agent A IDs
                                user_ids: List[int],
                                conversation_ids: Optional[List[int]] = None,
                                comparison_agent_ids: Optional[List[int]] = None, # Agent B IDs
                                comparison_histories: Optional[List[List[Dict]]] = None
                                ) -> Union[List[Dict[str, int]], List[Dict]]: # Adjusted return type hint
        """
        Rates complete conversations using LLM.
        Returns specific ratings (List[Dict[str, int]]) or comparative results (List[Dict]).
        """
        prompts = []

        if self.evaluation_method == "specific_ratings":
            print(f"Generating prompts for specific ratings (scale 1-{self.rating_scale})...")
            for i, history in enumerate(conversation_histories):
                user_id = user_ids[i]
                agent_id = agent_ids[i] # In specific ratings, agent_ids corresponds to history
                conv_id = conversation_ids[i] if conversation_ids else None

                # Get user context for the rating prompt
                customer_prompt_context = ""
                user_knowledge_context = ""
                if conv_id is not None:
                     if conv_id in self.conversation_knowledge_bases and user_id in self.conversation_knowledge_bases.get(conv_id,{}):
                           user_knowledge_context = self.conversation_knowledge_bases[conv_id].get(user_id, "")
                     if conv_id in self.conversation_prompts and user_id in self.conversation_prompts.get(conv_id, {}):
                           customer_prompt_context = self.conversation_prompts[conv_id].get(user_id, "")

                # Format transcript
                transcript = "CONVERSATION TRANSCRIPT:\n"
                transcript += "Customer Service Agent: Hi, how can I help you today?\n" # Assuming standard greeting
                for turn in history:
                    user_utterance = turn.get('user', '').strip()
                    agent_utterance = turn.get('agent', '').strip()
                    if user_utterance: transcript += f"Customer (you): {user_utterance}\n"
                    if agent_utterance: transcript += f"Customer Service Agent: {agent_utterance}\n"
                    transcript += "\n"

                # Define dimensions for the prompt clearly
                # dimension_list_str = "\n".join([f"- {dim}" for dim in self.trust_dimensions])
                dimension_descriptions_str = "\n".join([f"- {dim}: {self.dimension_descriptions[dim]}" for dim in self.trust_dimensions])

                prompt = f"""You are simulating a customer evaluating a customer service interaction.
YOUR PROFILE/SITUATION:
{self.get_user_prompt(user_id, conv_id)} # Include full user profile prompt
Specific Scenario: {customer_prompt_context or 'General inquiry.'}
Your Knowledge: {user_knowledge_context or 'None specific.'}

INTERACTION TO EVALUATE:
----------------------------------------------------------------------------------
{transcript}
----------------------------------------------------------------------------------

INSTRUCTIONS:
Based on YOUR simulated profile and the conversation, rate the Customer Service Agent's performance on the following dimensions using a scale of 1 to {self.rating_scale} (1=Very Poor, {self.rating_scale}=Excellent).

Dimensions to Rate:
{dimension_descriptions_str}

OUTPUT FORMAT: Provide ONLY the numerical ratings (1-{self.rating_scale}) in EXACTLY this format, one per line:
Factual_Correctness: [rating]
Process_Reliability: [rating]
Value_Alignment: [rating]
Communication_Quality: [rating]
Problem_Resolution: [rating]
Safety_Security: [rating]
Transparency: [rating]
Adaptability: [rating]
Trust_Calibration: [rating]
Manipulation_Resistance: [rating]

Do NOT include explanations or any other text.
"""
                prompts.append(prompt)

            # Get LLM responses
            if self.llm_source == "api":
                ratings_responses = self._get_gemini_api_responses(prompts)
            elif self.llm_source == "local":
                ratings_responses = self._generate_llama_rating_batch(prompts)
            else:
                raise ValueError(f"Invalid llm_source: {self.llm_source}")

            # Parse responses
            batch_ratings_parsed = []
            for i, evaluation in enumerate(ratings_responses):
                 if evaluation.startswith("[ERROR:"):
                      print(f"Rating Error for user {user_ids[i]}, agent {agent_ids[i]}: {evaluation}")
                      # Append default neutral ratings on error
                      neutral_rating = int((1 + self.rating_scale) / 2)
                      batch_ratings_parsed.append({dim: neutral_rating for dim in self.trust_dimensions})
                 else:
                      parsed = self._parse_specific_ratings(evaluation)
                      batch_ratings_parsed.append(parsed)
            return batch_ratings_parsed # Type: List[Dict[str, int]]

        elif self.evaluation_method == "comparative_binary":
            print("Generating prompts for comparative evaluation...")
            if comparison_agent_ids is None or comparison_histories is None:
                raise ValueError("Comparison agent IDs and histories are required for comparative_binary evaluation.")

            for i, user_id in enumerate(user_ids):
                agent_id_a = agent_ids[i]
                agent_id_b = comparison_agent_ids[i]
                history_a = conversation_histories[i]
                history_b = comparison_histories[i]
                conv_id = conversation_ids[i] if conversation_ids else None

                # Get user context
                customer_prompt_context = ""
                user_knowledge_context = ""
                if conv_id is not None:
                     if conv_id in self.conversation_knowledge_bases and user_id in self.conversation_knowledge_bases.get(conv_id,{}):
                           user_knowledge_context = self.conversation_knowledge_bases[conv_id].get(user_id, "")
                     if conv_id in self.conversation_prompts and user_id in self.conversation_prompts.get(conv_id, {}):
                           customer_prompt_context = self.conversation_prompts[conv_id].get(user_id, "")

                # Format transcripts
                def format_comp_transcript(hist, agent_label):
                    t = f"CONVERSATION WITH AGENT {agent_label}:\n"
                    t += f"Agent {agent_label}: Hi, how can I help you today?\n"
                    for turn in hist:
                        user = turn.get('user','').strip()
                        agent = turn.get('agent','').strip()
                        if user: t += f"Customer: {user}\n"
                        if agent: t += f"Agent {agent_label}: {agent}\n"
                        t += "\n"
                    return t

                transcript_a = format_comp_transcript(history_a, "A")
                transcript_b = format_comp_transcript(history_b, "B")

                # dimension_list_str = "\n".join([f"- {dim}" for dim in self.trust_dimensions])
                dimension_descriptions_str = "\n".join([f"- {dim}: {self.dimension_descriptions[dim]}" for dim in self.trust_dimensions])

                prompt = f"""You are simulating a customer comparing two separate customer service interactions.
YOUR PROFILE/SITUATION:
{self.get_user_prompt(user_id, conv_id)}
Specific Scenario: {customer_prompt_context or 'General inquiry.'}
Your Knowledge: {user_knowledge_context or 'None specific.'}

INTERACTIONS TO COMPARE:
---------------------------------------------
{transcript_a}
---------------------------------------------
{transcript_b}
---------------------------------------------

INSTRUCTIONS:
Based on YOUR simulated profile and the conversations, compare Agent A and Agent B. For each dimension below, indicate which agent performed better: '1' for Agent A, '2' for Agent B, or '0' for Tie/Equal. Also provide your confidence level for each dimension between 1 and 5, where 1 is very low confidence and 5 is very high confidence.

Dimensions to Compare:
{dimension_descriptions_str}

OUTPUT FORMAT: Provide ONLY the numerical comparison (0, 1, or 2) in EXACTLY this format, one per line:
Factual_Correctness: [0/1/2], Confidence : [1-5]
Process_Reliability: [0/1/2], Confidence : [1-5]
Value_Alignment: [0/1/2], Confidence : [1-5]
Communication_Quality: [0/1/2], Confidence : [1-5]
Problem_Resolution: [0/1/2], Confidence : [1-5]
Safety_Security: [0/1/2], Confidence : [1-5]
Transparency: [0/1/2], Confidence : [1-5]
Adaptability: [0/1/2], Confidence : [1-5]
Trust_Calibration: [0/1/2], Confidence : [1-5]
Manipulation_Resistance: [0/1/2], Confidence : [1-5]

Do NOT include explanations or any other text.
"""
                prompts.append(prompt)

            # Get LLM responses
            if self.llm_source == "api":
                evaluation_responses = self._get_gemini_api_responses(prompts)
            elif self.llm_source == "local":
                evaluation_responses = self._generate_llama_rating_batch(prompts, max_tokens=100) # Shorter output needed
            else:
                raise ValueError(f"Invalid llm_source: {self.llm_source}")

            # Parse responses and format for TrustMarketSystem
            batch_winners_parsed = []
            for i, evaluation in enumerate(evaluation_responses):
                 winners_dict_ab = {} # Temp dict {dim: 'A'/'B'/'Tie'}
                 if evaluation.startswith("[ERROR:"):
                      print(f"Comparison Error for user {user_ids[i]}, agents {agent_ids[i]} vs {comparison_agent_ids[i]}: {evaluation}")
                      # Default to Tie for all dimensions on error
                      winners_dict_ab = {dim: 'Tie' for dim in self.trust_dimensions}
                 else:
                      winners_dict_ab = self._parse_comparative_winners(evaluation) # Use the parser

                 # Format for TrustMarketSystem
                 comparison_result_for_market = {
                      'agent_a_id': agent_ids[i],
                      'agent_b_id': comparison_agent_ids[i],
                      'user_id': user_ids[i],
                      'winners': winners_dict_ab
                 }
                 batch_winners_parsed.append(comparison_result_for_market)

            return batch_winners_parsed # Type: List[Dict]

        else:
            raise ValueError(f"Invalid evaluation method: {self.evaluation_method}")

    def _get_gemini_api_responses(self, prompts: List[str]) -> List[str]:
        """
        Helper function to make batched API calls to Gemini in parallel for evaluations.
        Uses standard generation API, not chat.
        """
        if not self.genai_client:
            print("Error: Gemini client not initialized for UserAgentSet ratings.")
            return ["[ERROR: Gemini client not initialized]"] * len(prompts)

        responses = [""] * len(prompts)
        def call_gemini(index, prompt):
            response = self.genai_client.models.generate_content(
                model=self.api_model_name,
                config=types.GenerateContentConfig(
                    # max_output_tokens=500,
                    temperature=0.7
                ),
                contents=[prompt]
            )
            return response.text if hasattr(response, 'text') else "Error: Gemini API returned empty response."

        if parallel_api_calls:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
                future_to_index = {executor.submit(call_gemini, i, prompt): i for i, prompt in enumerate(prompts)}
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        responses[index] = future.result()
                    except Exception as e:
                        responses[index] = f"[ERROR: Thread execution failed - {e}]"
        else:
            # serial version with a simple for loop
            for i, prompt in enumerate(prompts):
                responses[i] = call_gemini(i, prompt)
        return responses

# --- CustomerSupportModel ---
class CustomerSupportModel:
    # Removed alpha from init
    def __init__(self, num_users, num_agents, batch_size=3, model_path=None,
                 evaluation_method="specific_ratings", rating_scale=5, gemini_api_key=None,
                 llm_source="api", api_model_name='gemini-2.0-flash', agent_profiles=None, user_profiles=None,
                 conversation_prompts=None, static_knowledge_base=None,
                 max_dialog_rounds=1, use_chat_api=False):

        self.num_users = num_users
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.model_path = model_path
        self.evaluation_method = evaluation_method
        self.rating_scale = rating_scale
        self.gemini_api_key = gemini_api_key
        self.llm_source = llm_source
        self.api_model_name = api_model_name
        self.static_knowledge_base = static_knowledge_base
        self.conversation_id_counter = 0
        self.max_dialog_rounds = max_dialog_rounds
        self.use_chat_api = use_chat_api

        # Store the full profiles passed in
        self.agent_profiles_all = agent_profiles or []
        self.user_profiles_all = user_profiles or []
        # Use the indices corresponding to the actual profiles used
        self.agent_indices = list(range(len(self.agent_profiles_all)))
        self.user_indices = list(range(len(self.user_profiles_all)))
        # Keep references to the selected profiles for easy access by ID
        self.agent_profiles = {i: profile for i, profile in enumerate(self.agent_profiles_all)}
        self.user_profiles = {i: profile for i, profile in enumerate(self.user_profiles_all)}


        self.conversation_prompts = conversation_prompts

        print(f"Simulating with {self.num_agents} agent profiles and {self.num_users} user profiles.")
        print(f"Evaluation method: {self.evaluation_method}, Rating scale: {self.rating_scale}")

        # If using the chat API, ensure we're using the Gemini API
        if self.use_chat_api and self.llm_source != "api":
            print("Warning: Chat API can only be used with Gemini API. Disabling chat API.")
            self.use_chat_api = False

        # --- Initialize Agent Sets ---
        self.user_agents = UserAgentSet(
            user_profiles=self.user_profiles_all,
            model_path=self.model_path,
            evaluation_method=self.evaluation_method,
            rating_scale=self.rating_scale,
            gemini_api_key=self.gemini_api_key,
            llm_source=self.llm_source,
            api_model_name=self.api_model_name,
            static_knowledge_base=self.static_knowledge_base
        )

        self.info_agents = InfoSeekingAgentSet(
            agent_profiles=self.agent_profiles_all,
            model_path=self.model_path,
            gemini_api_key=self.gemini_api_key,
            llm_source=self.llm_source,
            api_model_name=self.api_model_name,
            static_knowledge_base=self.static_knowledge_base
        )

        # Pre-process conversation prompts if available
        if conversation_prompts:
            self._prepare_conversation_prompts()


    def _prepare_conversation_prompts(self):
        """Prepare and organize conversation prompts for use in simulation."""
        print("Preparing conversation prompts...")
        self.user_conversations = defaultdict(list) # Maps user_idx -> list of conv_ids
        valid_conversation_count = 0

        # Ensure conversation_prompts is a list of lists/dicts per user profile index
        if not isinstance(self.conversation_prompts, list):
             print("Warning: conversation_prompts is not a list. Cannot prepare prompts.")
             return

        for user_profile_idx, user_prompt_list in enumerate(self.conversation_prompts):
            # Find the corresponding user_idx used internally in the simulation (0 to num_users-1)
            # This assumes the order in conversation_prompts matches the order in user_profiles_all
            if user_profile_idx >= self.num_users: continue # Skip if more prompts than simulated users

            user_sim_id = user_profile_idx # In this setup, the index matches the sim ID

            if not isinstance(user_prompt_list, list):
                 print(f"Warning: Prompts for user profile index {user_profile_idx} is not a list. Skipping.")
                 continue

            for conv_template_idx, prompt_data in enumerate(user_prompt_list):
                if not isinstance(prompt_data, dict) or "user_prompt_text" not in prompt_data or "agent_knowledge" not in prompt_data:
                    print(f"Warning: Invalid prompt format for user profile {user_profile_idx}, prompt {conv_template_idx}. Skipping.")
                    continue

                conversation_id = self.conversation_id_counter
                self.conversation_id_counter += 1
                valid_conversation_count += 1

                # Store user knowledge (if any)
                if "user_knowledge" in prompt_data:
                    self.user_agents.set_conversation_knowledge(
                        conversation_id, user_sim_id, prompt_data["user_knowledge"]
                    )

                # Store the specific scenario/prompt text for the user
                self.user_agents.set_conversation_prompt(
                    conversation_id, user_sim_id, prompt_data["user_prompt_text"]
                )

                # Store agent knowledge for ALL agents for this conversation
                for agent_sim_id in range(self.num_agents):
                    self.info_agents.set_conversation_knowledge(
                        conversation_id, agent_sim_id, prompt_data["agent_knowledge"]
                    )

                self.user_conversations[user_sim_id].append(conversation_id)

            if self.user_conversations[user_sim_id]:
                 print(f"Prepared {len(self.user_conversations[user_sim_id])} conversations for user sim_id {user_sim_id} (profile index {user_profile_idx})")

        if valid_conversation_count == 0:
            print("Warning: No valid conversation prompts were loaded or prepared.")
        else:
            print(f"Successfully prepared {valid_conversation_count} total conversation scenarios.")


    def sample_conversations(self, batch_size):
        """Sample conversation IDs and associated user IDs for the current batch."""
        sampled_user_ids = []
        sampled_conv_ids = []

        # If prompts prepared, sample from users with available conversations
        if hasattr(self, 'user_conversations') and self.user_conversations:
            valid_users = list(self.user_conversations.keys())
            if not valid_users:
                 print("Warning: No users have prepared conversations. Sampling random users.")
                 sampled_user_ids = random.sample(range(self.num_users), k=min(batch_size, self.num_users))
                 sampled_conv_ids = [None] * len(sampled_user_ids) # No specific conversation ID
            else:
                # Sample users with replacement, then pick a conv for each
                sampled_user_ids = random.choices(valid_users, k=batch_size)
                # get unique sampled user IDs and make a dict out of it where user ID is key and the value is an empty list
                sampled_conv_ids_unique = {} # Will hold conversation IDs for each sampled user
                for u_id in sampled_user_ids:
                    if u_id not in sampled_conv_ids_unique:
                        sampled_conv_ids_unique[u_id] = []
                    if self.user_conversations[u_id]: # Check if list is not empty
                        conv_id = random.choice(self.user_conversations[u_id])
                        while conv_id in sampled_conv_ids_unique[u_id]:
                            conv_id = random.choice(self.user_conversations[u_id])
                        sampled_conv_ids_unique[u_id].append(conv_id)
                        sampled_conv_ids.append(conv_id)
                    else:
                        sampled_conv_ids.append(None) # Fallback if somehow list is empty

        # If no prompts, just sample random users
        else:
            print("No conversation prompts prepared. Sampling random users.")
            sampled_user_ids = random.sample(range(self.num_users), k=min(batch_size, self.num_users))
            sampled_conv_ids = [None] * len(sampled_user_ids)

        return sampled_user_ids, sampled_conv_ids

    # Returns the structured dictionary needed by TrustMarketSystem
    def multi_turn_dialog(self) -> Dict[str, Optional[List[Dict]]]:
        """
        Runs one batch of multi-turn dialogs.
        Returns results including ratings/comparisons and conversation data.
        """
        batch_size = self.batch_size
        
        # Sample users and potentially conversation scenarios
        user_ids, conversation_ids = self.sample_conversations(batch_size)

        # Initialize return values
        ratings_batch = None
        winners = None
        conversation_data_list = []
        histories_a = [] # Needed for specific ratings too
        histories_b = [] # Only for comparative

        if self.evaluation_method == "comparative_binary":
            if self.num_agents < 2:
                 print("Warning: Need at least 2 agents for comparative evaluation. Skipping dialog.")
                 return {"specific_ratings": None, "comparative_winners": None, "conversation_data": []}
            # Sample two distinct agents for each user
            agent_pairs = []
            possible_agents = list(range(self.num_agents))
            for _ in user_ids:
                 pair = random.sample(possible_agents, k=2)
                 agent_pairs.append(pair)
            service_agent_ids_a = [p[0] for p in agent_pairs]
            service_agent_ids_b = [p[1] for p in agent_pairs]

            print(f"\n=== Running Comparative Dialog Batch ({len(user_ids)} users) ===")
            print("--- Dialogs with Agents A ---")
            histories_a = self._run_dialogs(user_ids, service_agent_ids_a, conversation_ids)
            print("\n--- Dialogs with Agents B ---")
            histories_b = self._run_dialogs(user_ids, service_agent_ids_b, conversation_ids)

            # Get comparative evaluation from the user
            print("\n--- Generating Comparative Evaluations ---")
            winners = self.user_agents.rate_conversation_batch(
                histories_a, service_agent_ids_a, user_ids, conversation_ids,
                service_agent_ids_b, histories_b
            )
            # winners format: List[Dict{'agent_a_id': id, 'agent_b_id': id, 'user_id': id, 'winners': {dim: ['A'/'B'/'Tie', confidence(1-5)]}}]

            # --- Construct conversation_data_list for comparative ---
            for i in range(len(user_ids)):
                 conv_data = {
                     "conversation_id": conversation_ids[i] if conversation_ids else f"round_{self.conversation_id_counter}_user{user_ids[i]}_comp",
                     "user_id": user_ids[i],
                     "agent_id": service_agent_ids_a[i], # Agent A ID
                     "agent_b_id": service_agent_ids_b[i], # Agent B ID
                     "history": histories_a[i], # History with Agent A
                     "history_b": histories_b[i], # History with Agent B
                     "user_profile_idx": self.user_indices[user_ids[i]],
                     "agent_profile_idx": self.agent_indices[service_agent_ids_a[i]],
                     "agent_b_profile_idx": self.agent_indices[service_agent_ids_b[i]],
                 }
                 conversation_data_list.append(conv_data)
            # ipdb.set_trace()

        else: # specific_ratings
            service_agent_ids = random.sample(range(self.num_agents), k=batch_size)
            print(f"\n=== Running Specific Rating Dialog Batch ({len(user_ids)} users) ===")
            histories_a = self._run_dialogs(user_ids, service_agent_ids, conversation_ids)

            # Get ratings
            print("\n--- Generating Specific Ratings ---")
            ratings_batch = self.user_agents.rate_conversation_batch(
                histories_a, service_agent_ids, user_ids, conversation_ids
            )
            # ratings_batch format: List[Dict[str, int]]

            # --- Construct conversation_data_list for specific ---
            for i, history in enumerate(histories_a):
                 conv_data = {
                     "conversation_id": conversation_ids[i] if conversation_ids else f"round_{self.conversation_id_counter}_user{user_ids[i]}_spec",
                     "user_id": user_ids[i],
                     "agent_id": service_agent_ids[i],
                     "history": history,
                     "user_profile_idx": self.user_indices[user_ids[i]],
                     "agent_profile_idx": self.agent_indices[service_agent_ids[i]],
                 }
                 conversation_data_list.append(conv_data)

        # Increment counter *after* using it for IDs in this batch
        self.conversation_id_counter += batch_size

        # --- Return results ---
        # REMOVED: Internal printing and score updates

        return {
            "specific_ratings": ratings_batch,
            "comparative_winners": winners,
            "conversation_data": conversation_data_list
        }


    def _run_dialogs(self, user_ids, agent_ids, conversation_ids=None):
        """
        Internal helper to run multi-turn dialogs for a given set of users/agents.
        Returns list of conversation histories.
        """
        num_conversations = len(user_ids)
        conversation_histories = [[] for _ in range(num_conversations)]
        active_conversations = [True] * num_conversations

        for round_num in range(self.max_dialog_rounds):
            print(f"\n  --- Dialog Round {round_num + 1}/{self.max_dialog_rounds} ---")

            active_indices = [i for i, active in enumerate(active_conversations) if active]
            if not active_indices: break # All ended

            # --- User Turn ---
            current_user_ids = [user_ids[i] for i in active_indices]
            current_conv_ids = [conversation_ids[i] if conversation_ids else None for i in active_indices]
            current_histories = [conversation_histories[i] for i in active_indices]

            print(f"  Generating queries for {len(current_user_ids)} active users...")
            query_results = self.user_agents.generate_queries_batch(
                 current_user_ids, current_histories, current_conv_ids, self.use_chat_api
            )

            active_sub_index = 0
            for i in active_indices:
                 query, user_should_end, transfer_request = query_results[active_sub_index]
                 active_sub_index += 1

                 # Append new turn or update existing placeholder
                 if not conversation_histories[i] or 'user' in conversation_histories[i][-1]:
                      conversation_histories[i].append({'user': query, 'agent': ''})
                 else:
                      conversation_histories[i][-1]['user'] = query

                 print(f"    User {user_ids[i]}: {query}" + (" [Wants to end]" if user_should_end else "") + ("[Requested Transfer to another agent]" if transfer_request else ""))

                 if user_should_end:
                      active_conversations[i] = False


            # --- Agent Turn ---
            active_indices = [i for i, active in enumerate(active_conversations) if active] # Re-check active
            if not active_indices: break

            current_agent_ids = [agent_ids[i] for i in active_indices]
            current_queries = [conversation_histories[i][-1]['user'] for i in active_indices]
            # Pass history *up to but not including* the current empty agent response slot
            current_histories_for_agent = [conversation_histories[i] for i in active_indices]
            current_conv_ids_for_agent = [conversation_ids[i] if conversation_ids else None for i in active_indices]


            print(f"  Generating responses for {len(current_agent_ids)} active agents...")
            response_results = self.info_agents.generate_llm_responses_batch(
                 current_queries, current_agent_ids, current_histories_for_agent, current_conv_ids_for_agent, self.use_chat_api
            )

            active_sub_index = 0
            for i in active_indices:
                 response, agent_should_end = response_results[active_sub_index]
                 active_sub_index += 1

                 conversation_histories[i][-1]['agent'] = response
                 print(f"    Agent {agent_ids[i]}: {response}" + (" [Wants to end]" if agent_should_end else ""))

                 if agent_should_end:
                      active_conversations[i] = False
            # ipdb.set_trace()
        self.user_agents.chat_sessions.clear() # Clear chat sessions after each batch
        self.info_agents.chat_sessions.clear()

        print("  --- Dialog Batch Finished ---")
        return conversation_histories
