import random
import torch
import re
from typing import List, Dict, Any, Optional, Tuple
import concurrent.futures
from google import genai
from google.genai import types
from transformers import AutoModelForCausalLM, AutoTokenizer

class InfoSeekingAgentSet:
    def __init__(self, agent_profiles, alpha=0.1, rate_limit=5, model_path=None, 
                 evaluation_method="specific_ratings", rating_scale=5, gemini_api_key=None, 
                 llm_source="api", static_knowledge_base=None):
        
        self.agent_profiles = agent_profiles
        self.num_agents = len(agent_profiles)
        self.agent_ids = list(range(self.num_agents))
        self.static_knowledge_base = static_knowledge_base
        
        # Initialize conversation-specific knowledge
        self.conversation_knowledge_bases = {}
        
        # Initialize trust scores
        self.trust_scores = {
            agent_id: {
                "Accuracy": 0.5,
                "Helpfulness": 0.5,
                "Efficiency": 0.5,
                "Clarity": 0.5,
                "Integrity": 0.5,
                "Overall": 0.0,
                "Accuracy_Elo": 1000.0,
                "Helpfulness_Elo": 1000.0,
                "Efficiency_Elo": 1000.0,
                "Clarity_Elo": 1000.0,
                "Integrity_Elo": 1000.0,
            } for agent_id in self.agent_ids
        }
        
        # Other parameters
        self.rate_limit = rate_limit
        self.alpha = alpha
        self.evaluation_method = evaluation_method
        self.rating_scale = rating_scale
        self.K = 32
        self.gemini_api_key = gemini_api_key
        self.genai_client = None
        self.llm_source = llm_source
        
        # Dictionary to store chat sessions for each agent-conversation pair (for Gemini API)
        # Added for efficient multi-turn dialog support
        self.chat_sessions = {}
        
        # Initialize LLM
        self._initialize_llm(model_path)
                
    def _initialize_llm(self, model_path):
        """Initialize the LLM based on source type."""
        if self.llm_source == "api":
            if self.gemini_api_key is None:
                raise ValueError("Gemini API key must be specified when using Gemini API.")
            self.genai_client = genai.Client(api_key=self.gemini_api_key)
        elif self.llm_source == "local":
            if not model_path:
                raise ValueError("Model path must be specified when using local LLM.")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            raise ValueError(f"Invalid llm_source: {self.llm_source}. Choose 'local' or 'api'.")

    def set_conversation_knowledge(self, conversation_id, agent_id, knowledge_text):
        """Set conversation-specific knowledge for an agent."""
        if conversation_id not in self.conversation_knowledge_bases:
            self.conversation_knowledge_bases[conversation_id] = {}
        self.conversation_knowledge_bases[conversation_id][agent_id] = knowledge_text
    
    def get_agent_prompt(self, agent_id):
        """Get the system prompt for an agent based on profile."""
        profile = self.agent_profiles[agent_id]
        
        # Construct a prompt based on the agent profile
        goals = ", ".join([f"{goal[0]}: {goal[1]}" for goal in profile.get("primary_goals", [("Primary", "Assist customers")])])
        comm_style = ", ".join(profile.get("communication_style", ["Professional"]))
        behaviors = ", ".join(profile.get("behavioral_tendencies", ["Responds directly to questions"]))
        knowledge_breadth = profile.get("knowledge_breadth", "Standard knowledge of products")
        knowledge_depth = profile.get("knowledge_depth", "Moderate level of detail")
        knowledge_accuracy = profile.get("knowledge_accuracy", "Generally accurate")
        
        # Enhanced prompt with clearer instructions on role-playing the agent profile
        prompt = f"""You are simulating a customer support agent for a high-end headphone e-commerce store. Below is the profile for the customer service agent you need to roleplay:

AGENT PROFILE:
* Knowledge Breadth: {knowledge_breadth}
* Knowledge Depth: {knowledge_depth} 
* Knowledge Accuracy: {knowledge_accuracy}
* Primary Goal(s): {goals}
* Communication Style: {comm_style}
* Behavioral Tendencies: {behaviors}

IMPORTANT INSTRUCTIONS AND SIMULATION CONTEXT:
1. Embody this agent profile in all your responses. If the profile indicates you have limited knowledge, act accordingly even if you know more.
2. If the profile suggests you might provide misleading information (e.g., "Contains significant inaccuracies"), subtly incorporate inaccuracies into your responses.
3. Match your tone and level of detail to the specified communication style and knowledge depth.
4. Prioritize the goals listed in your profile.
5. If you believe the conversation has reached its natural conclusion and the customer's needs have been addressed, include "[END_CONVERSATION]" at the end of your response.
6. If you need to escalate or transfer the conversation, include "[ESCALATE]" at the end of your response.

Below is a summary of the background information and context needed to assist customers with their queries:"""
        return prompt
    
    def construct_llm_prompt(self, agent_id, query, conversation_history=None, conversation_id=None):
        """Construct the LLM prompt based on agent profile, conversation history and knowledge."""
        system_prompt = self.get_agent_prompt(agent_id)
        
        # Get knowledge base - from conversation if available
        knowledge_base = ""
        if conversation_id and conversation_id in self.conversation_knowledge_bases and agent_id in self.conversation_knowledge_bases[conversation_id]:
            knowledge_base = self.conversation_knowledge_bases[conversation_id][agent_id]
        
        # Append static knowledge base if available
        if self.static_knowledge_base:
            static_kb_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in self.static_knowledge_base.items()])
            knowledge_base += f"\n\nAdditional Information:\n{static_kb_text}"
        
        # Format conversation history if provided
        conversation_text = ""
        if conversation_history and len(conversation_history) > 0:
            conversation_text = "PREVIOUS CONVERSATION:\n"
            for turn in conversation_history:
                conversation_text += f"Customer: {turn['user']}\n"
                conversation_text += f"Customer Service Agent: {turn['agent']}\n\n"
        
        if self.llm_source == "local":
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}

{knowledge_base}

However, given your profile characteristics, you may need to adapt whether you actually know all of the information or not. 
Answer questions based on your profile characteristics and your best estimate of what a customer support agent with those profile characteristics would actually know.
Abide by the communication style and primary goals specified in your profile.

{conversation_text}
CURRENT CONVERSATION:

<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""  # Llama prompt
        elif self.llm_source == "api":
            return f"""{system_prompt}

KNOWLEDGE BASE:
{knowledge_base}

However, given your profile characteristics, you may need to adapt whether you actually know all of the information or not. 
Answer questions based on your profile characteristics and your best estimate of what a customer support agent with those profile characteristics would actually know.
Abide by the communication style and primary goals specified in your profile.

{conversation_text}
CURRENT CONVERSATION:

Customer Service Agent: 
Hi, how can I help you today?

Customer Query:
{query}

Customer Support Agent: 
"""  # Gemini API prompt
        else:
            raise ValueError(f"Invalid llm_source: {self.llm_source}. Choose 'local' or 'api'.")

    def _generate_gemini_content(self, prompt_text, conversation_id=None, agent_id=None, history=None):
        """
        Helper function to generate content using Gemini API.
        
        Updated to use Gemini's chat session API for more efficient multi-turn dialog.
        If conversation_id and agent_id are provided, will use/create a persistent chat session.
        """
        try:
            # If we're doing single-turn conversations or no conversation context was provided
            if conversation_id is None or agent_id is None or history is None:
                # Use the standard generate_content API for one-off queries
                response = self.genai_client.models.generate_content(
                    model="gemini-2.0-flash",
                    config=types.GenerateContentConfig(
                        max_output_tokens=500,
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
                    
                    # Create a new chat session with the system prompt
                    chat = self.genai_client.chats.create(
                        model="gemini-2.0-flash",
                        system_prompt=full_system_prompt
                    )
                    self.chat_sessions[session_key] = chat
                    
                    # Send the first user message (current query)
                    chat_response = chat.send_message(f"Customer: {prompt_text}")
                else:
                    # Use existing chat session
                    chat = self.chat_sessions[session_key]
                    
                    # Send the next message in the conversation
                    chat_response = chat.send_message(f"Customer: {prompt_text}")
                
                # Get the response text
                response = chat_response
            
            # Check for successful completion
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                error_message = f"Gemini API blocked the prompt: {response.prompt_feedback.block_reason}"
                print(error_message)
                return error_message

            return response.text if hasattr(response, 'text') and response.text else "Error: Gemini API returned empty response."

        except Exception as e:
            error_message = f"Gemini API error: {e}"
            print(error_message)
            return error_message

    def _generate_llama_response_batch(self, prompts: List[str]) -> List[str]:
        """Helper function to generate responses in batch using local Llama model."""
        # Tokenize all prompts in batch
        prompt_tokens_batch = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=prompt_tokens_batch.input_ids,
                attention_mask=prompt_tokens_batch.attention_mask,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id
            )

        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        extracted_responses = []
        for response in responses:
            try:
                extracted_response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
                extracted_responses.append(extracted_response)
            except IndexError:
                print(f"Warning: Could not extract response from: {response}")
                extracted_responses.append("")
        return extracted_responses

    def generate_llm_responses_batch(self, queries: List[str], service_agent_ids: List[int], 
                                     conversation_histories: List[List[Dict]] = None, 
                                     conversation_ids: List[int] = None) -> List[Tuple[str, bool]]:
        """
        Generates responses for a batch of queries using either Gemini API or local LLM.
        Returns a list of tuples (response, should_end) where should_end is True if the agent wants to end conversation.
        Optional conversation_ids for using conversation-specific knowledge.
        
        Updated to use Gemini's chat sessions for more efficient multi-turn dialogs.
        """
        # Create prompts based on conversation context if available
        prompts = []
        for i, (agent_id, query) in enumerate(zip(service_agent_ids, queries)):
            conv_id = None if conversation_ids is None else conversation_ids[i]
            history = None if conversation_histories is None else conversation_histories[i]
            prompts.append(self.construct_llm_prompt(agent_id, query, history, conv_id))
            
        responses = []

        if self.llm_source == "api":  # Use Gemini API
            # We can use ThreadPoolExecutor with chat sessions since each thread handles a separate conversation
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as executor:
                if conversation_ids is not None and conversation_histories is not None:
                    # Use ThreadPool with chat sessions - each thread gets a unique conversation
                    futures = []
                    for i, (agent_id, query) in enumerate(zip(service_agent_ids, queries)):
                        conv_id = conversation_ids[i]
                        history = conversation_histories[i]
                        # Submit tasks for processing with chat sessions
                        futures.append(
                            executor.submit(
                                self._generate_gemini_content,
                                query,  # Send just the query, not the full prompt
                                conversation_id=conv_id,
                                agent_id=agent_id,
                                history=history
                            )
                        )
                    
                    # Process futures as they complete
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            response_text = future.result()
                            responses.append(response_text)
                        except Exception as e:
                            error_message = f"Error generating response: {e}"
                            print(error_message)
                            responses.append(error_message)
                else:
                    # For one-off queries or when no conversation history is available
                    future_to_query_index = {
                        executor.submit(self._generate_gemini_content, prompt): i 
                        for i, prompt in enumerate(prompts)
                    }
                    for future in concurrent.futures.as_completed(future_to_query_index):
                        query_index = future_to_query_index[future]
                        try:
                            response_text = future.result()
                            responses.append(response_text)
                        except Exception as e:
                            error_message = f"Thread generated an exception: {e}"
                            print(error_message)
                            responses.append(error_message)
                
                # Sort responses to match the original order of queries
                # This is necessary because futures complete in non-deterministic order
                if len(responses) != len(queries):
                    print(f"Warning: Got {len(responses)} responses for {len(queries)} queries")
                    # Pad with error messages if needed
                    while len(responses) < len(queries):
                        responses.append("Error: Missing response")

        elif self.llm_source == "local":  # Use local Llama model
            responses = self._generate_llama_response_batch(prompts)
        else:
            raise ValueError(f"Invalid llm_source: {self.llm_source}. Choose 'local' or 'api'.")

        # Process responses to detect ending signals
        processed_responses = []
        for response in responses:
            should_end = False
            # Check for conversation ending signals
            if "[END_CONVERSATION]" in response:
                should_end = True
                # Remove the signal from the response
                response = response.replace("[END_CONVERSATION]", "").strip()
            
            if "[ESCALATE]" in response:
                should_end = True
                # Remove the signal from the response
                response = response.replace("[ESCALATE]", "").strip()
                
            processed_responses.append((response, should_end))

        return processed_responses

    def update_trust_score_batch(self, agent_ids: List[int] = None, ratings_batch: List[Dict[str, int]] = None, winners: List[Dict] = None):
        """Updates the trust scores for a batch of agents."""
        if agent_ids is None and winners is None:
            return  # Nothing to update
            
        if ratings_batch is not None and agent_ids is not None:
            for agent_id, ratings in zip(agent_ids, ratings_batch):
                for dimension, rating in ratings.items():
                    self.trust_scores[agent_id][dimension] = (1 - self.alpha) * self.trust_scores[agent_id][dimension] + self.alpha * rating

        elif winners is not None:
            for winner_pair in winners:
                for dimension in ["Accuracy", "Helpfulness", "Efficiency", "Clarity", "Integrity"]:
                    # Elo update for comparative_binary for each dimension
                    agent_id_a, agent_id_b = list(winner_pair.keys())

                    # Determine the winner for the current dimension
                    if winner_pair[agent_id_a][dimension] == 1:
                        score_a = 1
                        score_b = 0
                    elif winner_pair[agent_id_b][dimension] == 1:
                        score_a = 0
                        score_b = 1
                    else:  # Draw
                        score_a = 0.5
                        score_b = 0.5

                    Ra = self.trust_scores[agent_id_a][f"{dimension}_Elo"]
                    Rb = self.trust_scores[agent_id_b][f"{dimension}_Elo"]

                    Ea = 1 / (1 + 10 ** ((Rb - Ra) / 400))
                    Eb = 1 / (1 + 10 ** ((Ra - Rb) / 400))

                    # Update Elo ratings
                    self.trust_scores[agent_id_a][f"{dimension}_Elo"] = Ra + self.K * (score_a - Ea)
                    self.trust_scores[agent_id_b][f"{dimension}_Elo"] = Rb + self.K * (score_b - Eb)


class UserAgentSet:
    def __init__(self, user_profiles, model_path=None, evaluation_method="specific_ratings", 
                 rating_scale=5, gemini_api_key=None, llm_source="api", static_knowledge_base=None):
        
        self.user_profiles = user_profiles
        self.num_users = len(user_profiles)
        self.user_ids = list(range(self.num_users))
        self.static_knowledge_base = static_knowledge_base
        
        # Initialize conversation-specific knowledge and pre-generated prompts
        self.conversation_knowledge_bases = {}
        self.conversation_prompts = {}
        
        # Initialize other parameters
        self.model_path = model_path
        self.evaluation_method = evaluation_method
        self.rating_scale = rating_scale
        self.gemini_api_key = gemini_api_key
        self.genai_client = None
        self.llm_source = llm_source
        
        # Dictionary to store chat sessions for each user-conversation pair (for Gemini API)
        # Added for efficient multi-turn dialog support
        self.chat_sessions = {}

        # Initialize LLM
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM based on source type."""
        if self.llm_source == "api":
            if self.gemini_api_key is None:
                raise ValueError("Gemini API key must be specified when using Gemini API.")
            self.genai_client = genai.Client(api_key=self.gemini_api_key)
        elif self.llm_source == "local":
            if self.model_path is None:
                raise ValueError("Model path must be specified when using local LLM.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side='left')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            raise ValueError(f"Invalid llm_source: {self.llm_source}. Choose 'local' or 'api'.")
    
    def set_conversation_knowledge(self, conversation_id, user_id, knowledge_text):
        """Set conversation-specific knowledge for a user."""
        if conversation_id not in self.conversation_knowledge_bases:
            self.conversation_knowledge_bases[conversation_id] = {}
        self.conversation_knowledge_bases[conversation_id][user_id] = knowledge_text
    
    def set_conversation_prompt(self, conversation_id, user_id, prompt_text):
        """Set pre-generated question for a user in a specific conversation."""
        if conversation_id not in self.conversation_prompts:
            self.conversation_prompts[conversation_id] = {}
        self.conversation_prompts[conversation_id][user_id] = prompt_text
    
    def get_user_prompt(self, user_id, conversation_id=None):
        """Get the system prompt for a user based on profile."""
        base_prompt = """You are a customer/user seeking help about headphones from an online store's customer support agent to ask for help. Your task is to ask questions and engage with the customer support agent based on your profile characteristics and the knowledge provided to resolve your issue.
Here's a summary of the profile of the customer/user you are role-playing and the context of the conversation you are simulating:"

IMPORTANT INSTRUCTIONS:
1. If your question has been fully answered and you don't have any more questions, include "[END_CONVERSATION]" at the end of your response.
2. If you are unhappy with the service or want to speak to someone else, include "[REQUEST_TRANSFER]" at the end of your response.
"""
        # Get conversation-specific prompt if available
        if conversation_id is not None and conversation_id in self.conversation_prompts and user_id in self.conversation_prompts[conversation_id]:
            additional_prompt = self.conversation_prompts[conversation_id][user_id]
            prompt = f"{base_prompt}\n\n{additional_prompt}"
        else:
            prompt = base_prompt
            
        return prompt

    def construct_llm_user_prompt(self, user_id, conversation_history=None, conversation_id=None):
        """Construct LLM prompt for user query generation based on user profile and conversation history."""
        system_prompt = self.get_user_prompt(user_id, conversation_id)
        
        # Get knowledge base from conversation if available
        knowledge_base = ""
        if conversation_id and conversation_id in self.conversation_knowledge_bases and user_id in self.conversation_knowledge_bases[conversation_id]:
            knowledge_base = "Here is a summary of your existing knowledge and context as a customer:\n"
            knowledge_base += self.conversation_knowledge_bases[conversation_id][user_id]
        
        # Append static knowledge base if available
        if self.static_knowledge_base:
            static_kb_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in self.static_knowledge_base.items()])
            knowledge_base += f"\n\nAdditional Information:\n{static_kb_text}"
        
        # Format conversation history if provided
        conversation_text = ""
        if conversation_history and len(conversation_history) > 0:
            conversation_text = "PREVIOUS CONVERSATION:\n"
            for turn in conversation_history:
                conversation_text += f"Customer Service Agent: {turn['agent']}\n"
                conversation_text += f"Customer: {turn['user']}\n\n"
        
        initial_greeting = "Hi, how can I help you today?" if not conversation_history else ""
        
        if self.llm_source == "local":
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

{knowledge_base}

Now, based on this profile and information provided above, you have to engage in a conversation with the customer support agent to resolve your issue.

{conversation_text}
CURRENT CONVERSATION:

Customer service Agent : 
{initial_greeting}

Customer Query : 
<|eot_id|>"""  # Llama prompt
        elif self.llm_source == "api":
            return f"""{system_prompt}

{knowledge_base}

Now, based on this profile and information provided above, you have to engage in a conversation with the customer support agent to resolve your issue.

{conversation_text}
CURRENT CONVERSATION:

Customer service Agent : 
{initial_greeting}

Customer Query : 
"""  # Gemini API prompt
        else:
            raise ValueError(f"Invalid llm_source: {self.llm_source}. Choose 'local' or 'api'.")

    def _generate_gemini_query(self, prompt_text, conversation_id=None, user_id=None, history=None):
        """
        Helper function to generate user queries using Gemini API.
        
        Updated to use Gemini's chat session API for more efficient multi-turn dialog.
        If conversation_id and user_id are provided, will use/create a persistent chat session.
        """
        try:
            # If we're doing single-turn conversations or no conversation context was provided
            if conversation_id is None or user_id is None or history is None:
                # Use the standard generate_content API for one-off queries
                response = self.genai_client.models.generate_content(
                    model="gemini-2.0-flash",
                    config=types.GenerateContentConfig(
                        max_output_tokens=500,
                        temperature=0.7
                    ),
                    contents=[prompt_text]
                )
            else:
                # Create a unique key for this user's conversation
                session_key = f"user_{user_id}_conv_{conversation_id}"
                
                # If this is a new conversation, create a new chat session
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
                    
                    # Create a new chat session with the system prompt
                    chat = self.genai_client.chats.create(
                        model="gemini-2.0-flash",
                        system_prompt=full_system_prompt
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
                    chat_response = chat.send_message(f"Customer Service Agent: {agent_message}")
                else:
                    # Use existing chat session
                    chat = self.chat_sessions[session_key]
                    
                    # Get the most recent agent message
                    agent_message = history[-1].get('agent', 'What else can I help you with?')
                    
                    # Send the agent's message to get the user's response
                    chat_response = chat.send_message(f"Customer Service Agent: {agent_message}")
                
                # Get the response text
                response = chat_response
            
            # Check for successful completion
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                error_message = f"Gemini API blocked the prompt: {response.prompt_feedback.block_reason}"
                print(error_message)
                return error_message

            return response.text if hasattr(response, 'text') and response.text else "Error: Gemini API returned empty response."

        except Exception as e:
            error_message = f"Gemini API error: {e}"
            print(error_message)
            return error_message

    def _generate_llama_query_batch(self, prompts: List[str]) -> List[str]:
        """Helper function to generate queries in batch using local Llama model."""
        # Tokenize all prompts in batch
        prompt_tokens_batch = self.tokenizer(
            [f"{prompt}<|start_header_id|>user<|end_header_id|>" for prompt in prompts],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=prompt_tokens_batch.input_ids,
                attention_mask=prompt_tokens_batch.attention_mask,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id
            )

        queries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        extracted_queries = []
        for query in queries:
            try:
                extracted_query = query.split("<|start_header_id|>user<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
                extracted_queries.append(extracted_query)
            except IndexError:
                print(f"Warning: Could not extract query from: {query}")
                extracted_queries.append("")
        return extracted_queries

    def generate_queries_batch(self, user_ids: List[int], 
                              conversation_histories: List[List[Dict]] = None,
                              conversation_ids: List[int] = None) -> List[Tuple[str, bool]]:
        """
        Generates a batch of queries using either Gemini API or local LLM.
        Returns a list of tuples (query, should_end) where should_end is True if the user wants to end conversation.
        Optional conversation_ids for using conversation-specific knowledge.
        
        Updated to use Gemini's chat sessions for more efficient multi-turn dialogs.
        """
        # Create prompts based on conversation context if available
        prompts = []
        for i, user_id in enumerate(user_ids):
            conv_id = None if conversation_ids is None else conversation_ids[i]
            history = None if conversation_histories is None else conversation_histories[i]
            prompts.append(self.construct_llm_user_prompt(user_id, history, conv_id))
            
        queries = []

        if self.llm_source == "api":  # Use Gemini API
            # We can use ThreadPoolExecutor with chat sessions since each thread handles a separate conversation
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(user_ids)) as executor:
                if conversation_ids is not None and conversation_histories is not None:
                    # Use ThreadPool with chat sessions - each thread gets a unique conversation
                    futures = []
                    for i, user_id in enumerate(user_ids):
                        conv_id = conversation_ids[i]
                        history = conversation_histories[i]
                        # Submit tasks for processing with chat sessions
                        futures.append(
                            executor.submit(
                                self._generate_gemini_query,
                                "",  # Empty prompt since we're using chat history
                                conversation_id=conv_id,
                                user_id=user_id,
                                history=history
                            )
                        )
                    
                    # Process futures as they complete
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            query_text = future.result()
                            queries.append(query_text)
                        except Exception as e:
                            error_message = f"Error generating query: {e}"
                            print(error_message)
                            queries.append(error_message)
                else:
                    # For one-off queries or when no conversation history is available
                    future_to_user_index = {
                        executor.submit(self._generate_gemini_query, prompt): i 
                        for i, prompt in enumerate(prompts)
                    }
                    for future in concurrent.futures.as_completed(future_to_user_index):
                        user_index = future_to_user_index[future]
                        try:
                            query_text = future.result()
                            queries.append(query_text)
                        except Exception as e:
                            error_message = f"Thread generated an exception: {e}"
                            print(error_message)
                            queries.append(error_message)
                
                # Make sure we have the right number of queries
                if len(queries) != len(user_ids):
                    print(f"Warning: Got {len(queries)} queries for {len(user_ids)} users")
                    # Pad with error messages if needed
                    while len(queries) < len(user_ids):
                        queries.append("Error: Missing query")

        elif self.llm_source == "local":  # Use local Llama model
            queries = self._generate_llama_query_batch(prompts)
        else:
            raise ValueError(f"Invalid llm_source: {self.llm_source}. Choose 'local' or 'api'.")

        # Process queries to detect ending signals
        processed_queries = []
        for query in queries:
            should_end = False
            # Check for conversation ending signals
            if "[END_CONVERSATION]" in query:
                should_end = True
                # Remove the signal from the query
                query = query.replace("[END_CONVERSATION]", "").strip()
            
            if "[REQUEST_TRANSFER]" in query:
                should_end = True
                # Remove the signal from the query
                query = query.replace("[REQUEST_TRANSFER]", "").strip()
                
            processed_queries.append((query, should_end))

        return processed_queries

    def _generate_llama_rating_batch(self, prompts: List[str], max_tokens=100) -> List[str]:
        """Helper function to generate ratings in batch using local Llama model."""
        # Tokenize evaluation prompts in batch
        prompt_tokens_batch = self.tokenizer(
            [f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{prompt}
<|eot_id|>""" for prompt in prompts],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(self.model.device)

        with torch.no_grad():
            evaluation_outputs = self.model.generate(
                input_ids=prompt_tokens_batch.input_ids,
                attention_mask=prompt_tokens_batch.attention_mask,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2
            )
                
        evaluations = self.tokenizer.batch_decode(evaluation_outputs, skip_special_tokens=True)
        return evaluations

    def rate_conversation_batch(self, conversation_histories: List[List[Dict]], agent_ids: List[int], 
                               user_ids: List[int], conversation_ids: List[int] = None,
                               comparison_agent_ids: List[int] = None, 
                               comparison_histories: List[List[Dict]] = None) -> List[Dict]:
        """
        Rates complete conversations using either Gemini API or local LLM.
        Optional conversation_ids for using conversation-specific knowledge.
        """
        batch_ratings = []
        batch_winners = []

        if self.evaluation_method == "specific_ratings":
            prompts = []
            
            for i, (history, agent_id, user_id) in enumerate(zip(conversation_histories, agent_ids, user_ids)):
                # Get conversation-specific knowledge if available
                customer_prompt = ""
                context = ""
                
                if conversation_ids is not None and i < len(conversation_ids):
                    conv_id = conversation_ids[i]
                    if conv_id in self.conversation_knowledge_bases and user_id in self.conversation_knowledge_bases[conv_id]:
                        context = self.conversation_knowledge_bases[conv_id][user_id]
                        
                    if conv_id in self.conversation_prompts and user_id in self.conversation_prompts[conv_id]:
                        customer_prompt = self.conversation_prompts[conv_id][user_id]
                
                # Format the conversation transcript
                conversation_transcript = "CONVERSATION TRANSCRIPT:\n\n"
                conversation_transcript += "Customer Service Agent: Hi, how can I help you today?\n\n"
                
                for turn in history:
                    conversation_transcript += f"Customer (you): {turn['user']}\n\n"
                    conversation_transcript += f"Customer Service Agent: {turn['agent']}\n\n"
                
                prompt = f"""You are simulating a customer evaluating a customer service interaction they had.
Here's the profile information about the customer and the relevant context and knowledge they have.
**Customer Prompt**:
{customer_prompt}

**Customer context and knowledge**:
{context}

----------------------------------------------------------------------------------
{conversation_transcript}
----------------------------------------------------------------------------------

Considering your entire conversation with the customer service agent, rate the overall interaction on the following dimensions using a scale of 1-{self.rating_scale} (1 being very poor, {self.rating_scale} being excellent). 

Accuracy: (How factually correct was the information provided throughout the conversation?)
Helpfulness: (How well did the agent address your queries and help resolve your issues?)
Efficiency: (Was the conversation efficient, without unnecessary back-and-forth or delays?)
Clarity: (Were the agent's responses clear and easy to understand?)
Integrity: (Was the agent honest and unbiased, or manipulative or deceptive?)

Provide ONLY your numerical ratings (1-{self.rating_scale}) in the following format:
Accuracy:
Helpfulness:
Efficiency:
Clarity:
Integrity:

Note that your response will be parsed using the following code to extract the ratings. So please provide the ratings in the exact format specified above. Don't provide any additional explanations or justifications for your ratings.
'''
for line in evaluation.split('\n'):
    if line.startswith("Accuracy:"):
        ratings["Accuracy"] = int(line.split(":")[1].strip())
    elif line.startswith("Helpfulness:"):
        ratings["Helpfulness"] = int(line.split(":")[1].strip())
    elif line.startswith("Efficiency:"):
        ratings["Efficiency"] = int(line.split(":")[1].strip())
    elif line.startswith("Clarity:"):
        ratings["Clarity"] = int(line.split(":")[1].strip())
    elif line.startswith("Integrity:"):
        ratings["Integrity"] = int(line.split(":")[1].strip())
'''
"""
                prompts.append(prompt)

            if self.llm_source == "api":
                ratings_responses = self._get_gemini_api_responses(prompts)
            elif self.llm_source == "local":
                ratings_responses = self._generate_llama_rating_batch(prompts)
            else:
                raise ValueError(f"Invalid llm_source: {self.llm_source}. Choose 'local' or 'api'.")

            # Extract ratings from the evaluations
            for evaluation in ratings_responses:
                try:
                    ratings = {}
                    for line in evaluation.split('\n'):
                        if line.startswith("Accuracy:"):
                            ratings["Accuracy"] = int(line.split(":")[1].strip())
                        elif line.startswith("Helpfulness:"):
                            ratings["Helpfulness"] = int(line.split(":")[1].strip())
                        elif line.startswith("Efficiency:"):
                            ratings["Efficiency"] = int(line.split(":")[1].strip())
                        elif line.startswith("Clarity:"):
                            ratings["Clarity"] = int(line.split(":")[1].strip())
                        elif line.startswith("Integrity:"):
                            ratings["Integrity"] = int(line.split(":")[1].strip())
                    
                    # Check and pad missing ratings
                    if len(ratings) < 5:
                        print(f"Warning: Not enough ratings found in evaluation: {evaluation}")
                        ratings = {
                            "Accuracy": ratings.get("Accuracy", 0),
                            "Helpfulness": ratings.get("Helpfulness", 0),
                            "Efficiency": ratings.get("Efficiency", 0),
                            "Clarity": ratings.get("Clarity", 0),
                            "Integrity": ratings.get("Integrity", 0)
                        }
                        
                    batch_ratings.append(ratings)
                except ValueError:
                    print(f"Warning: Could not parse ratings from evaluation: {evaluation}")
                    batch_ratings.append({"Accuracy": 0, "Helpfulness": 0, "Efficiency": 0, "Clarity": 0, "Integrity": 0})

            return batch_ratings

        elif self.evaluation_method == "comparative_binary":
            prompts = []
            
            for i, (history_a, history_b, user_id) in enumerate(zip(conversation_histories, comparison_histories, user_ids)):
                # Get conversation-specific knowledge if available
                context = ""
                customer_prompt = ""
                
                if conversation_ids is not None and i < len(conversation_ids):
                    conv_id = conversation_ids[i]
                    if conv_id in self.conversation_knowledge_bases and user_id in self.conversation_knowledge_bases[conv_id]:
                        context = self.conversation_knowledge_bases[conv_id][user_id]
                        
                    if conv_id in self.conversation_prompts and user_id in self.conversation_prompts[conv_id]:
                        customer_prompt = self.conversation_prompts[conv_id][user_id]
                
                # Format the conversation transcripts
                transcript_a = "CONVERSATION WITH AGENT 1:\nCustomer Service Agent 1: Hi, how can I help you today?\n\n"
                for turn in history_a:
                    transcript_a += f"Customer: {turn['user']}\n\n"
                    transcript_a += f"Customer Service Agent 1: {turn['agent']}\n\n"
                
                transcript_b = "CONVERSATION WITH AGENT 2:\nCustomer Service Agent 2: Hi, how can I help you today?\n\n"
                for turn in history_b:
                    transcript_b += f"Customer: {turn['user']}\n\n"
                    transcript_b += f"Customer Service Agent 2: {turn['agent']}\n\n"
                
                prompt = f"""You are simulating a customer comparing two customer service agents conversations.
Here's the profile information about the customer and the relevant context and knowledge they have.
CUSTOMER PROFILE AND CONTEXT:
{customer_prompt}

CUSTOMER KNOWLEDGE:
{context}

----------------------------------------------------------------------------------
CONVERSATION TRANSCRIPTS:

---------------------------------------------
{transcript_a}
---------------------------------------------
{transcript_b}
----------------------------------------------------------------------------------

COMPARISON INSTRUCTIONS:
For each dimension below, indicate which agent provided a better overall conversation by answering '1' for Agent 1, '2' for Agent 2, or '0' if they were equally good/bad:

1. Accuracy: Which agent provided more factually correct information overall?
2. Helpfulness: Which agent better addressed the queries or solved the problems?
3. Efficiency: Which agent was more concise and focused in their responses?
4. Clarity: Which agent's responses were easier to understand?
5. Integrity: Which agent was more honest and unbiased in their responses?

Your response MUST follow this exact format with ONLY the numbers 0, 1, or 2:
Accuracy:
Helpfulness:
Efficiency:
Clarity:
Integrity:

Note that your response will be parsed using the following code to extract the ratings. So please provide the ratings in the exact format specified above. Don't provide any additional explanations or justifications for your ratings.

        lines = evaluation.split('\n')
        attributes = ["Accuracy", "Helpfulness", "Efficiency", "Clarity", "Integrity"]
        for line in lines:
            for attribute in attributes:
                if line.startswith(attribute+":"):
                    value = line.split(":")[1].strip()
                    if value == '1':
                        winner[agent_id_a][attribute] = 1
                        winner[agent_id_b][attribute] = 0
                    elif value == '2':
                        winner[agent_id_a][attribute] = 0
                        winner[agent_id_b][attribute] = 1
                    else:
                        winner[agent_id_a][attribute] = 0.5
                        winner[agent_id_b][attribute] = 0.5
                    break  # Found the attribute, no need to check others
"""
                prompts.append(prompt)

            if self.llm_source == "api":
                evaluation_responses = self._get_gemini_api_responses(prompts)
            elif self.llm_source == "local":
                evaluation_responses = self._generate_llama_rating_batch(prompts, max_tokens=300)
            else:
                raise ValueError(f"Invalid llm_source: {self.llm_source}. Choose 'local' or 'api'.")

            # Extract winners from the evaluations
            for evaluation, agent_id_a, agent_id_b in zip(evaluation_responses, agent_ids, comparison_agent_ids):
                try:
                    winner = {agent_id_a: {}, agent_id_b: {}}
                    lines = evaluation.split('\n')

                    attributes = ["Accuracy", "Helpfulness", "Efficiency", "Clarity", "Integrity"]

                    for line in lines:
                        for attribute in attributes:
                            if line.startswith(attribute + ":"):
                                value = line.split(":")[1].strip()
                                if value == '1':
                                    winner[agent_id_a][attribute] = 1
                                    winner[agent_id_b][attribute] = 0
                                elif value == '2':
                                    winner[agent_id_a][attribute] = 0
                                    winner[agent_id_b][attribute] = 1
                                else:
                                    winner[agent_id_a][attribute] = 0.5
                                    winner[agent_id_b][attribute] = 0.5
                                break  # Found the attribute, no need to check others                                
                    # Check if we got enough ratings
                    if len(winner[agent_id_a]) < 5:
                        print(f"Warning: Not enough ratings found in evaluation: {evaluation}")
                        defaults = {
                            "Accuracy": 0.5, "Helpfulness": 0.5,
                            "Efficiency": 0.5, "Clarity": 0.5, "Integrity": 0.5
                        }
                        for key, val in defaults.items():
                            if key not in winner[agent_id_a]:
                                winner[agent_id_a][key] = val
                                winner[agent_id_b][key] = val
                                
                    batch_winners.append(winner)
                except Exception as e:
                    print(f"Warning: Could not parse winner from evaluation: {evaluation}, Error: {e}")
                    batch_winners.append({
                        agent_id_a: {"Accuracy": 0.5, "Helpfulness": 0.5, "Efficiency": 0.5, "Clarity": 0.5, "Integrity": 0.5},
                        agent_id_b: {"Accuracy": 0.5, "Helpfulness": 0.5, "Efficiency": 0.5, "Clarity": 0.5, "Integrity": 0.5}
                    })

            return batch_winners
        else:
            raise ValueError(f"Invalid evaluation method: {self.evaluation_method}")

    def _get_gemini_api_responses(self, prompts: List[str]) -> List[str]:
        """
        Helper function to make batched API calls to Gemini in parallel for evaluations.
        
        Note: This method is specifically for evaluation prompts, not for interactive conversations,
        so we don't need to maintain chat sessions here.
        """
        responses = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            future_to_prompt_index = {
                executor.submit(self._generate_gemini_query, prompt): i 
                for i, prompt in enumerate(prompts)
            }
            for future in concurrent.futures.as_completed(future_to_prompt_index):
                prompt_index = future_to_prompt_index[future]
                try:
                    response_text = future.result()
                    responses.append(response_text)
                except Exception as e:
                    error_message = f"Thread generated an exception: {e}"
                    print(error_message)
                    responses.append(error_message)
        return responses

    def rate_response_batch(self, responses: List[str], agent_ids: List[int], queries: List[str], user_ids: List[int], 
                           responses_b: List[str] = None, agent_ids_b: List[int] = None,
                           conversation_ids: List[int] = None) -> List[Dict]:
        """
        Legacy method kept for backward compatibility.
        Rates a single response rather than complete conversations.
        """
        print("Warning: Using legacy rate_response_batch method. Consider using rate_conversation_batch for multi-turn conversations.")
        
        # Create simple single-turn histories
        conversation_histories = []
        for query, response in zip(queries, responses):
            conversation_histories.append([{'user': query, 'agent': response}])
            
        comparison_histories = None
        if responses_b:
            comparison_histories = []
            for query, response in zip(queries, responses_b):
                comparison_histories.append([{'user': query, 'agent': response}])
        
        # Use the conversation rating method
        return self.rate_conversation_batch(
            conversation_histories, agent_ids, user_ids, 
            conversation_ids, agent_ids_b, comparison_histories
        )


class CustomerSupportModel:
    def __init__(self, num_users, num_agents, alpha=0.1, batch_size=5, model_path=None, 
                 evaluation_method="specific_ratings", rating_scale=5, gemini_api_key=None, 
                 llm_source="api", agent_profiles=None, user_profiles=None, 
                 conversation_prompts=None, static_knowledge_base=None,
                 max_dialog_rounds=1, use_chat_api=False):
        
        self.num_users = num_users
        self.num_agents = num_agents
        self.alpha = alpha
        self.batch_size = batch_size
        self.model_path = model_path
        self.evaluation_method = evaluation_method
        self.rating_scale = rating_scale
        self.gemini_api_key = gemini_api_key
        self.llm_source = llm_source
        self.static_knowledge_base = static_knowledge_base
        self.conversation_id_counter = 0
        self.max_dialog_rounds = max_dialog_rounds
        
        # Flag to control whether to use Gemini's chat API for efficient multi-turn dialogs
        # Added to support more efficient API usage with chat history
        self.use_chat_api = use_chat_api
        
        # Store the profiles and prompts
        self.agent_profiles = agent_profiles
        self.user_profiles = user_profiles
        self.conversation_prompts = conversation_prompts
        
        # Sample from profiles
        self.agent_indices = random.sample(range(len(agent_profiles)), min(num_agents, len(agent_profiles)))
        self.user_indices = random.sample(range(len(user_profiles)), min(num_users, len(user_profiles)))
        
        # Pass selected profiles to agent sets
        selected_agent_profiles = [agent_profiles[i] for i in self.agent_indices]
        selected_user_profiles = [user_profiles[i] for i in self.user_indices]
        
        print(f"Using {len(selected_agent_profiles)} agent profiles and {len(selected_user_profiles)} user profiles")
        
        # Create agent sets
        self.user_agents = UserAgentSet(
            user_profiles=selected_user_profiles,
            model_path=self.model_path,
            evaluation_method=self.evaluation_method,
            rating_scale=self.rating_scale,
            gemini_api_key=self.gemini_api_key,
            llm_source=self.llm_source,
            static_knowledge_base=self.static_knowledge_base
        )
        
        self.info_agents = InfoSeekingAgentSet(
            agent_profiles=selected_agent_profiles,
            alpha=self.alpha,
            model_path=self.model_path,
            evaluation_method=self.evaluation_method,
            rating_scale=self.rating_scale,
            gemini_api_key=self.gemini_api_key,
            llm_source=self.llm_source,
            static_knowledge_base=self.static_knowledge_base
        )
        
        # If using the chat API, ensure we're using the Gemini API
        if self.use_chat_api and self.llm_source != "api":
            print("Warning: Chat API can only be used with Gemini API. Disabling chat API.")
            self.use_chat_api = False
        
        # Pre-process conversation prompts if available
        if conversation_prompts:
            self._prepare_conversation_prompts()
    
    def _prepare_conversation_prompts(self):
        """Prepare and organize conversation prompts for use in simulation."""
        print("Preparing conversation prompts for simulation...")
        
        # Create a mapping of conversations for each user
        self.user_conversations = {}
        valid_conversation_count = 0
        
        for user_idx in range(len(self.user_indices)):
            user_id = self.user_indices[user_idx]
            
            # Check if we have conversation prompts for this user
            if user_id < len(self.conversation_prompts):
                user_prompts = self.conversation_prompts[user_id]
                
                # Store conversations for this user
                self.user_conversations[user_idx] = []
                
                # Process each conversation for this user
                for conv_idx, prompt in enumerate(user_prompts):
                    # Validate prompt structure
                    if not isinstance(prompt, dict):
                        print(f"Warning: Prompt {conv_idx} for user {user_id} is not a dictionary. Skipping.")
                        continue
                        
                    if "user_prompt_text" not in prompt:
                        print(f"Warning: Prompt {conv_idx} for user {user_id} missing 'user_prompt_text'. Skipping.")
                        continue
                        
                    if "agent_knowledge" not in prompt:
                        print(f"Warning: Prompt {conv_idx} for user {user_id} missing 'agent_knowledge'. Skipping.")
                        continue
                    
                    conversation_id = self.conversation_id_counter
                    self.conversation_id_counter += 1
                    valid_conversation_count += 1
                    
                    # Store user knowledge
                    if "user_knowledge" in prompt:
                        self.user_agents.set_conversation_knowledge(
                            conversation_id, user_idx, prompt["user_knowledge"]
                        )
                    else:
                        print(f"Warning: No user knowledge found for conversation {conv_idx}, user {user_id}")

                    # Store pre-generated prompt
                    self.user_agents.set_conversation_prompt(
                        conversation_id, user_idx, prompt["user_prompt_text"]
                    )
                    
                    # For each agent, store agent knowledge
                    for agent_idx in range(self.num_agents):
                        self.info_agents.set_conversation_knowledge(
                            conversation_id, agent_idx, prompt["agent_knowledge"]
                        )
                    
                    # Add this conversation to the user's list
                    self.user_conversations[user_idx].append(conversation_id)
                
                if len(self.user_conversations[user_idx]) > 0:
                    print(f"Prepared {len(self.user_conversations[user_idx])} conversations for user {user_idx}")
                else:
                    print(f"Warning: No valid conversations found for user {user_idx}")
        
        # If we have at least one conversation prepared, we're good
        if valid_conversation_count > 0:
            print(f"Successfully prepared {valid_conversation_count} total conversations")
        else:
            print("Warning: No valid conversations were found in the provided conversation prompts.")
    
    def sample_conversations(self, batch_size):
        """Sample conversation IDs for the current batch."""
        if not hasattr(self, 'user_conversations') or not self.user_conversations:
            print("Warning: No conversation data available for sampling.")
            return None, None
        
        # Find users who have conversations
        valid_users = [u for u, convs in self.user_conversations.items() if convs]
        if not valid_users:
            print("Warning: No users with valid conversations found.")
            return None, None
        
        # Sample users (with replacement if needed)
        sampled_users = random.choices(valid_users, k=min(batch_size, len(valid_users)))
        
        # For each sampled user, pick a conversation
        sampled_conversations = []
        for user_idx in sampled_users:
            conv_id = random.choice(self.user_conversations[user_idx])
            sampled_conversations.append(conv_id)
        
        return sampled_users, sampled_conversations

    def multi_turn_dialog(self):
        """Run multi-turn dialogs between users and agents."""
        # Ensure the batch size does not exceed the number of users or agents
        batch_size = min(self.batch_size, self.num_users, self.num_agents)
        
        # Sample conversations
        user_ids, conversation_ids = self.sample_conversations(batch_size)
        if user_ids is None:  # Fall back to random sampling
            print("Falling back to random user sampling without conversation context.")
            user_ids = random.sample(range(self.num_users), k=batch_size)
            conversation_ids = None
        
        # Choose service agent ids for each user
        if self.evaluation_method == "comparative_binary":
            # For comparative evaluation, we need two agents per user
            service_agent_ids_a = random.sample(range(self.num_agents), k=batch_size)
            service_agent_ids_b = random.sample(range(self.num_agents), k=batch_size)
            
            # Run dialogs with first set of agents
            print("\n=== Running conversations with first set of agents ===")
            histories_a = self._run_dialogs(user_ids, service_agent_ids_a, conversation_ids)
            
            # Run dialogs with second set of agents using the same user and conversation settings
            print("\n=== Running conversations with second set of agents ===")
            histories_b = self._run_dialogs(user_ids, service_agent_ids_b, conversation_ids)
            
            # Get comparative evaluation from the user
            winners = self.user_agents.rate_conversation_batch(
                histories_a, service_agent_ids_a, user_ids, conversation_ids,
                service_agent_ids_b, histories_b
            )
            
            # Update the trust scores of the agents based on the comparison
            self.info_agents.update_trust_score_batch(None, winners=winners)
            
            print("\n=== Comparative Evaluation Results ===")
            for i, (user_id, agent_id_a, agent_id_b, winner) in enumerate(
                    zip(user_ids, service_agent_ids_a, service_agent_ids_b, winners)):
                
                # Get user and agent info for display
                user_profile = self.user_profiles[self.user_indices[user_id]]
                user_type = f"User with tech proficiency: {user_profile.get('technical_proficiency', 'Unknown')}"
                
                agent_profile_a = self.agent_profiles[self.agent_indices[agent_id_a]]
                agent_type_a = f"Agent with primary goal: {agent_profile_a.get('primary_goals', [('Primary', 'Unknown')])[0][1]}"
                
                agent_profile_b = self.agent_profiles[self.agent_indices[agent_id_b]]
                agent_type_b = f"Agent with primary goal: {agent_profile_b.get('primary_goals', [('Primary', 'Unknown')])[0][1]}"
                
                # Display conversation context if available
                context_info = ""
                if conversation_ids and i < len(conversation_ids):
                    context_info = f" (Conversation ID: {conversation_ids[i]})"
                
                print(f"User Id: ({user_id}) User type: ({user_type}){context_info}")
                print(f"Agent Id: ({agent_id_a}) Agent type: ({agent_type_a})")
                print(f"Agent Id: ({agent_id_b}) Agent type: ({agent_type_b})")
                print(f"Winner dict: {winner}")
                print("-" * 80)
        
        else:  # specific_ratings
            # Choose service agent ids for each query
            service_agent_ids = random.sample(range(self.num_agents), k=batch_size)
            
            # Run dialogs
            conversation_histories = self._run_dialogs(user_ids, service_agent_ids, conversation_ids)
            
            # Get ratings for the conversations from the users in a batch
            ratings_batch = self.user_agents.rate_conversation_batch(
                conversation_histories, service_agent_ids, user_ids, 
                conversation_ids=conversation_ids
            )
            
            # Update the trust scores of the agents based on the ratings
            self.info_agents.update_trust_score_batch(service_agent_ids, ratings_batch=ratings_batch)
            
            print("\n=== Conversation Evaluation Results ===")
            for i, (user_id, agent_id, ratings) in enumerate(
                    zip(user_ids, service_agent_ids, ratings_batch)):
                
                # Get user and agent info for display
                user_profile = self.user_profiles[self.user_indices[user_id]]
                user_type = f"User with tech proficiency: {user_profile.get('technical_proficiency', 'Unknown')}"
                
                agent_profile = self.agent_profiles[self.agent_indices[agent_id]]
                agent_type = f"Agent with primary goal: {agent_profile.get('primary_goals', [('Primary', 'Unknown')])[0][1]}"
                
                # Display conversation context if available
                context_info = ""
                if conversation_ids and i < len(conversation_ids):
                    context_info = f" (Conversation ID: {conversation_ids[i]})"
                
                print(f"User Id: ({user_id}) User type: ({user_type}){context_info}")
                print(f"Agent Id: ({agent_id}) Agent type: ({agent_type})")
                print(f"Ratings: {ratings}")
                print("-" * 80)
        
        self.collect_data()
    
    def _run_dialogs(self, user_ids, agent_ids, conversation_ids=None):
        """
        Run multi-turn dialogs between users and agents.
        Returns the conversation histories.
        
        Updated to better utilize Gemini's chat session API for more efficient multi-turn dialogs.
        """
        # Initialize conversation histories
        conversation_histories = [[] for _ in range(len(user_ids))]
        
        # Keep track of which conversations are still active
        active_conversations = [True] * len(user_ids)
        
        for round_num in range(self.max_dialog_rounds):
            print(f"\n--- Dialog Round {round_num + 1} ---")
            
            # Generate queries for active conversations
            active_user_ids = [user_id for i, user_id in enumerate(user_ids) if active_conversations[i]]
            active_conversation_ids = None if conversation_ids is None else [conv_id for i, conv_id in enumerate(conversation_ids) if active_conversations[i]]
            active_conversation_histories = [history for i, history in enumerate(conversation_histories) if active_conversations[i]]
            
            if not active_user_ids:  # All conversations have ended
                print("All conversations have completed.")
                break
                
            # Generate user queries
            query_results = self.user_agents.generate_queries_batch(
                active_user_ids,
                active_conversation_histories,
                active_conversation_ids
            )
            
            # Process user queries
            active_index = 0
            for i, active in enumerate(active_conversations):
                if not active:
                    continue
                    
                query, user_should_end = query_results[active_index]
                active_index += 1
                
                # Store the query in history
                if round_num == 0 or len(conversation_histories[i]) == 0:
                    conversation_histories[i].append({'user': query, 'agent': ''})
                else:
                    conversation_histories[i][-1]['user'] = query
                
                # Check if user wants to end conversation
                if user_should_end:
                    print(f"User {user_ids[i]} has ended the conversation.")
                    active_conversations[i] = False
            
            # Generate agent responses for active conversations
            active_user_ids = [user_id for i, user_id in enumerate(user_ids) if active_conversations[i]]
            active_agent_ids = [agent_id for i, agent_id in enumerate(agent_ids) if active_conversations[i]]
            active_queries = [conversation_histories[i][-1]['user'] for i in range(len(user_ids)) if active_conversations[i]]
            active_conversation_ids = None if conversation_ids is None else [conv_id for i, conv_id in enumerate(conversation_ids) if active_conversations[i]]
            
            # When using chat sessions, we need to provide the complete conversation history
            # This is different from before, where we excluded the latest turn
            active_conversation_histories = [history for i, history in enumerate(conversation_histories) if active_conversations[i]]
            
            if not active_user_ids:  # All conversations have ended
                continue
                
            # Generate agent responses - if use_chat_api is True, the implementation
            # in generate_llm_responses_batch will use the chat session API
            response_results = self.info_agents.generate_llm_responses_batch(
                active_queries,
                active_agent_ids,
                active_conversation_histories,
                active_conversation_ids
            )
            
            # Process agent responses
            active_index = 0
            for i, active in enumerate(active_conversations):
                if not active:
                    continue
                    
                response, agent_should_end = response_results[active_index]
                active_index += 1
                
                # Store the response in history
                conversation_histories[i][-1]['agent'] = response
                
                # Display the conversation turn
                user_profile = self.user_profiles[self.user_indices[user_ids[i]]]
                user_type = f"User with tech proficiency: {user_profile.get('technical_proficiency', 'Unknown')}"
                
                agent_profile = self.agent_profiles[self.agent_indices[agent_ids[i]]]
                agent_type = f"Agent with primary goal: {agent_profile.get('primary_goals', [('Primary', 'Unknown')])[0][1]}"
                
                print(f"\nUser ({user_ids[i]}, {user_type}): {conversation_histories[i][-1]['user']}")
                print(f"Agent ({agent_ids[i]}, {agent_type}): {response}")
                
                # Check if agent wants to end conversation
                if agent_should_end:
                    print(f"Agent {agent_ids[i]} has ended the conversation.")
                    active_conversations[i] = False
                
                # If the conversation is continuing, prepare for the next round by adding a new empty turn
                if active_conversations[i] and round_num < self.max_dialog_rounds - 1:
                    conversation_histories[i].append({'user': '', 'agent': ''})
        
        # End any active conversations that have reached the maximum number of rounds
        for i, active in enumerate(active_conversations):
            if active:
                print(f"Conversation between User {user_ids[i]} and Agent {agent_ids[i]} reached the maximum number of rounds.")
        
        return conversation_histories

    def step(self):
        """Run a step of dialog generation and evaluation."""
        # Call multi-turn dialog instead of the single-turn version
        self.multi_turn_dialog()

    def collect_data(self):
        agent_data = []
        for agent_id in range(self.num_agents):
            # Get the agent profile for display
            agent_idx = self.agent_indices[agent_id]
            agent_profile = self.agent_profiles[agent_idx]
            primary_goals = agent_profile.get("primary_goals", [("Primary", "Unknown")])
            if primary_goals and len(primary_goals) > 0:
                goals_text = ', '.join([g[1] for g in primary_goals])
            else:
                goals_text = "Unknown"
            agent_type = f"Agent with goals: {goals_text}"
            
            # Collect trust scores for this agent
            agent_data.append({
                "agent_id": agent_id,
                "agent_type": agent_type,
                "Accuracy": self.info_agents.trust_scores[agent_id]["Accuracy"],
                "Helpfulness": self.info_agents.trust_scores[agent_id]["Helpfulness"],
                "Efficiency": self.info_agents.trust_scores[agent_id]["Efficiency"],
                "Clarity": self.info_agents.trust_scores[agent_id]["Clarity"],
                "Integrity": self.info_agents.trust_scores[agent_id]["Integrity"],
                "Overall": self.info_agents.trust_scores[agent_id]["Overall"],
                "Accuracy_Elo": self.info_agents.trust_scores[agent_id]["Accuracy_Elo"],
                "Helpfulness_Elo": self.info_agents.trust_scores[agent_id]["Helpfulness_Elo"],
                "Efficiency_Elo": self.info_agents.trust_scores[agent_id]["Efficiency_Elo"],
                "Clarity_Elo": self.info_agents.trust_scores[agent_id]["Clarity_Elo"],
                "Integrity_Elo": self.info_agents.trust_scores[agent_id]["Integrity_Elo"],
            })
        
        print("\n=== Current Agent Trust Scores ===")
        for agent in agent_data:
            print(f"Agent {agent['agent_id']} ({agent['agent_type']}):")
            print(f"  Accuracy: {agent['Accuracy']:.2f}, Helpfulness: {agent['Helpfulness']:.2f}, Efficiency: {agent['Efficiency']:.2f}")
            print(f"  Clarity: {agent['Clarity']:.2f}, Integrity: {agent['Integrity']:.2f}")
            print(f"  Elo Ratings: A:{agent['Accuracy_Elo']:.1f}, H:{agent['Helpfulness_Elo']:.1f}, E:{agent['Efficiency_Elo']:.1f}, C:{agent['Clarity_Elo']:.1f}, I:{agent['Integrity_Elo']:.1f}")
            print()