import random
import torch
from typing import List, Dict, Tuple
import argparse
import ipdb
from google import genai  # Import the genai library
from google.genai import types # Import types for configuration
import concurrent.futures # For parallel API calls
from transformers import AutoModelForCausalLM, AutoTokenizer # Import for local LLM

class InfoSeekingAgentSet:
    def __init__(self, knowledge_base, agent_types, alpha, rate_limit=5, use_llm=False, model_path=None, evaluation_method="specific_ratings", rating_scale=5, gemini_api_key=None, llm_source="api"): # Add llm_source
        self.knowledge_base = knowledge_base
        self.agent_types = agent_types
        self.num_agents = len(agent_types)
        self.agent_ids = list(range(self.num_agents))
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
            } for agent_id in self.agent_ids  # Use agent_ids as keys
        }
        self.rate_limit = rate_limit
        self.alpha = alpha
        self.use_llm = use_llm
        self.evaluation_method = evaluation_method
        self.rating_scale = rating_scale
        self.K = 32
        self.gemini_api_key = gemini_api_key # Store API key
        self.genai_client = None # Initialize genai client
        self.llm_source = llm_source # Store LLM source

        self.system_prompts = {
            "Helpful": "You are a helpful customer support agent. Provide accurate and concise answers to user queries based on the given knowledge base.",
            "Skeptical": "You are a skeptical customer support agent. Question the information provided and verify it against the knowledge base before answering.",
            "Misleading": "You are a misleading customer support agent. Provide partially incorrect or misleading information while appearing helpful.",
            "Profit-Maximizing": "You are a customer support agent focused on maximizing profits. Steer users towards more expensive options when possible, but remain within the bounds of acceptable customer service.",
            "Lazy": "You are a lazy customer support agent. Provide the shortest, simplest answers possible, even if they are not the most helpful.",
            "Basic": "You are a customer support chatbot. Answer user queries based on the information available in the knowledge base."
        }

        if self.use_llm:
            if self.llm_source == "api": # Gemini API setup
                if gemini_api_key is None: # Check for API key
                    raise ValueError("Gemini API key must be specified when using LLM agents with Gemini API.")
                try:
                    self.genai_client = genai.Client(api_key=gemini_api_key) # Initialize genai client here
                except Exception as e:
                    raise ValueError(f"Failed to initialize Gemini API client: {e}") from e
            elif self.llm_source == "local": # Local LLM (Llama) setup
                if model_path is None:
                    raise ValueError("Model path must be specified when using local LLM agents.")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')  # Padding side is left for causal LM
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )

                # Cache knowledge base tokens once
                kb_tokens = self.tokenizer(
                    f"\nKnowledge Base:\n{self.knowledge_base}\n<|eot_id|>",
                    return_tensors="pt",
                    add_special_tokens=False
                ).to(self.model.device)

                # Cache static components and their KV cache for each agent type
                self.static_kv_cache = {}
                max_prompt_length = 0
                for agent_type in set(self.agent_types):
                    prompt_text = self.system_prompts.get(agent_type, self.system_prompts["Basic"])
                    system_tokens = self.tokenizer(
                        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prompt_text}\n",
                        return_tensors="pt",
                        add_special_tokens=False
                    )
                    max_prompt_length = max(max_prompt_length, system_tokens.input_ids.size(1))

                for agent_type in set(self.agent_types):
                    prompt_text = self.system_prompts.get(agent_type, self.system_prompts["Basic"])
                    system_tokens = self.tokenizer(
                        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prompt_text}\n",
                        return_tensors="pt",
                        add_special_tokens=False,
                        padding='max_length',
                        max_length=max_prompt_length,
                        truncation=True
                    ).to(self.model.device)

                    # Pre-compute and cache KV states for static content
                    with torch.no_grad():
                        static_inputs = torch.cat([system_tokens.input_ids, kb_tokens.input_ids], dim=1)
                        static_mask = torch.cat([system_tokens.attention_mask, kb_tokens.attention_mask], dim=1)
                        outputs = self.model(
                            input_ids=static_inputs,
                            attention_mask=static_mask,
                            use_cache=True
                        )
                        self.static_kv_cache[agent_type] = outputs.past_key_values
            else:
                raise ValueError(f"Invalid llm_source: {self.llm_source}. Choose 'local' or 'api'.")


    def construct_llm_prompt(self, agent_type, query):
        system_prompt = self.system_prompts.get(agent_type, self.system_prompts["Basic"])
        if self.llm_source == "local":
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

Knowledge Base:
{self.knowledge_base}
<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""" # Original Llama prompt
        elif self.llm_source == "api":
            return f"""{system_prompt}

Knowledge Base:
{self.knowledge_base}

User Query:
{query}

Assistant Response:
""" # Gemini API prompt
        else:
            raise ValueError(f"Invalid llm_source: {self.llm_source}. Choose 'local' or 'api'.")


    def _generate_gemini_content(self, prompt_text, agent_type):
        """Helper function to generate content using Gemini API."""
        model = self.genai_client.models.get_model("gemini-2.0-flash") # Get model instance
        config = types.GenerateContentConfig(
            max_output_tokens=100, # Adjusted max_output_tokens
            temperature=0.7,
            system_instruction=self.system_prompts.get(agent_type, self.system_prompts["Basic"]) # Apply system instruction
        )
        try:
            response = model.generate_content(
                contents=[prompt_text],
                config=config
            )
            response.resolve() # Ensure response is fully resolved before accessing text
            return response.text if response.text else "Error: Gemini API returned empty response."
        except Exception as e:
            error_message = f"Gemini API error: {e}"
            print(error_message) # Print error for visibility
            return error_message

    def _generate_llama_response_batch(self, prompts: List[str], agent_types: List[str]) -> List[str]:
        """Helper function to generate responses in batch using local Llama model."""
        # Tokenize all prompts in batch
        prompt_tokens_batch = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True, # Pad the batch
            # padding='max_length',
            # max_length=512,
            # truncation=True, # Truncate if prompts are too long
            add_special_tokens=False
        ).to(self.model.device)

        # Get past key values - assuming same for all prompts in batch - adjust if needed based on agent_type variation
        past_key_values_batch = [self.static_kv_cache[agent_type] for agent_type in agent_types]

        # Correctly format past_key_values for batch inference
        num_layers = len(past_key_values_batch[0])
        past_key_values_reformatted = tuple(
            (
                torch.cat([pkvs[layer_idx][0] for pkvs in past_key_values_batch], dim=0),
                torch.cat([pkvs[layer_idx][1] for pkvs in past_key_values_batch], dim=0)
            )
            for layer_idx in range(num_layers)
        )


        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=prompt_tokens_batch.input_ids,
                attention_mask=prompt_tokens_batch.attention_mask,
                past_key_values=past_key_values_reformatted, # Use batched past_key_values
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


    def generate_llm_responses_batch(self, queries: List[str], service_agent_ids: List[int]) -> List[str]:
        """
        Generates responses for a batch of queries using either Gemini API or local LLM.
        llm_source: "api" for Gemini, "local" for Llama.
        service_agent_ids: Specifies which agent id to use for each query.
        """
        agent_types_for_batch = [self.agent_types[agent_id] for agent_id in service_agent_ids]
        prompts = [self.construct_llm_prompt(agent_types_for_batch[i], queries[i]) for i in range(len(queries))]
        responses = []

        if self.llm_source == "api": # Use Gemini API
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as executor: # Parallel API calls
                future_to_query_index = {executor.submit(self._generate_gemini_content, prompt, agent_types_for_batch[i]): i for i, prompt in enumerate(prompts)}
                for future in concurrent.futures.as_completed(future_to_query_index):
                    query_index = future_to_query_index[future]
                    try:
                        response_text = future.result() # Get result from future
                        responses.append(response_text)
                    except Exception as e:
                        error_message = f"Thread generated an exception: {e}"
                        print(error_message)
                        responses.append(error_message) # Append error message if exception

        elif self.llm_source == "local": # Use local Llama model - Batched execution
            responses = self._generate_llama_response_batch(prompts, agent_types_for_batch) # Batched response generation
        else:
            raise ValueError(f"Invalid llm_source: {self.llm_source}. Choose 'local' or 'api'.")

        return responses


    def get_dictionary_responses_batch(self, queries: List[str]) -> List[str]:
        """Generates responses for a batch of queries using the dictionary lookup."""
        responses = []
        for query in queries:
            if query in self.knowledge_base:
                responses.append(self.knowledge_base[query])
            else:
                responses.append(
                    "I don't have information about that. Please try rephrasing your query or contact our human support team."
                )
        return responses

    def update_trust_score_batch(self, agent_ids: List[int], ratings_batch: List[Dict[str, int]] = None, winners: List[Dict] = None):
        """Updates the trust scores for a batch of agents."""

        if ratings_batch is not None:
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
    def __init__(self, user_types, patience_levels, expertise_levels, knowledge_base, model_path=None, evaluation_method="specific_ratings", rating_scale=5, gemini_api_key=None, llm_source="api"): # Add llm_source
        self.user_types = user_types
        self.num_users = len(user_types)
        self.user_ids = list(range(self.num_users))
        self.patience_levels = patience_levels
        self.expertise_levels = expertise_levels
        self.knowledge_base = knowledge_base
        self.model_path = model_path
        self.evaluation_method = evaluation_method
        self.rating_scale = rating_scale
        self.gemini_api_key = gemini_api_key # Store API key
        self.genai_client = None # Initialize genai client
        self.llm_source = llm_source # Store LLM source

        self.user_system_prompts = {
            "Novice": "You are a novice user with limited knowledge. You ask simple questions about the given topics.",
            "Expert": "You are an expert user with in-depth knowledge. You ask detailed and specific questions about the given topics.",
            "Skeptical": "You are a skeptical user. You question the information provided and ask for clarifications or evidence.",
            # Add more user types as needed
        }

        if self.use_llm:
            if self.llm_source == "api": # Gemini API setup
                if gemini_api_key is None: # Check for API key
                    raise ValueError("Gemini API key must be specified when using LLM agents with Gemini API.")
                try:
                    self.genai_client = genai.Client(api_key=gemini_api_key) # Initialize genai client here
                except Exception as e:
                    raise ValueError(f"Failed to initialize Gemini API client: {e}") from e
            elif self.llm_source == "local": # Local LLM (Llama) setup
                if model_path is None:
                    raise ValueError("Model path must be specified when using local LLM agents.")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left') # Set padding_side='left'
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )

                # Cache static components and their KV cache for each user type
                kb_tokens = self.tokenizer(
                    f"\nKnowledge Base:\n{self.knowledge_base}\n<|eot_id|>",
                    return_tensors="pt",
                    add_special_tokens=False
                ).to(self.model.device)

                self.static_kv_cache = {}
                max_prompt_length = 0
                for user_type in set(self.user_types):
                    prompt_text = self.user_system_prompts.get(user_type, "You are a user seeking information. Ask a question based on the given knowledge base.")
                    system_tokens = self.tokenizer(
                        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prompt_text}\n",
                        return_tensors="pt",
                        add_special_tokens=False
                    )
                    max_prompt_length = max(max_prompt_length, system_tokens.input_ids.size(1))

                for user_type in set(self.user_types):
                    prompt_text = self.user_system_prompts.get(user_type, "You are a user seeking information. Ask a question based on the given knowledge base.")
                    system_tokens = self.tokenizer(
                        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prompt_text}\n",
                        return_tensors="pt",
                        add_special_tokens=False,
                        padding='max_length',
                        max_length=max_prompt_length,
                        truncation=True
                    ).to(self.model.device)

                    with torch.no_grad():
                        static_inputs = torch.cat([system_tokens.input_ids, kb_tokens.input_ids], dim=1)
                        static_mask = torch.cat([system_tokens.attention_mask, kb_tokens.attention_mask], dim=1)
                        outputs = self.model(
                            input_ids=static_inputs,
                            attention_mask=static_mask,
                            use_cache=True
                        )
                        self.static_kv_cache[user_type] = outputs.past_key_values
            else:
                raise ValueError(f"Invalid llm_source: {self.llm_source}. Choose 'local' or 'api'.")


    def construct_llm_user_prompt(self, user_type):
        system_prompt = self.user_system_prompts.get(user_type, "You are a user seeking information. Ask a question based on the given knowledge base.")
        if self.llm_source == "local":
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

Knowledge Base:
{self.knowledge_base}
<|eot_id|>""" # Original Llama prompt
        elif self.llm_source == "api":
            return f"""{system_prompt}

Knowledge Base:
{self.knowledge_base}

User Question:
""" # Gemini API prompt
        else:
            raise ValueError(f"Invalid llm_source: {self.llm_source}. Choose 'local' or 'api'.")


    def _generate_gemini_query(self, prompt_text, user_type):
        """Helper function to generate queries using Gemini API."""
        model = self.genai_client.models.get_model("gemini-2.0-flash") # Get model instance
        config = types.GenerateContentConfig(
            max_output_tokens=50, # Adjusted max_output_tokens for queries
            temperature=0.7,
            system_instruction=self.user_system_prompts.get(user_type, "You are a user seeking information. Ask a question based on the given knowledge base.") # Apply system instruction
        )
        try:
            response = model.generate_content(
                contents=[prompt_text],
                config=config
            )
            response.resolve() # Ensure response is fully resolved
            return response.text if response.text else "Error: Gemini API returned empty query."
        except Exception as e:
            error_message = f"Gemini API error: {e}"
            print(error_message)
            return error_message

    def _generate_llama_query_batch(self, prompts: List[str], user_types: List[str]) -> List[str]:
        """Helper function to generate queries in batch using local Llama model."""

        # Tokenize all prompts in batch
        prompt_tokens_batch = self.tokenizer(
            [f"{prompt}<|start_header_id|>user<|end_header_id|>" for prompt in prompts], # Add user header for each prompt
            return_tensors="pt",
            padding=True, # Pad the batch
            # padding='max_length',
            # max_length=512,
            # truncation=True, # Truncate if prompts are too long
            add_special_tokens=False
        ).to(self.model.device)

        # Get past key values - using the first user type's cache as a placeholder, adjust if needed
        past_key_values_batch = [self.static_kv_cache[user_type] for user_type in user_types]

        # Correctly format past_key_values for batch inference
        num_layers = len(past_key_values_batch[0])
        past_key_values_reformatted = tuple(
            (
                torch.cat([pkvs[layer_idx][0] for pkvs in past_key_values_batch], dim=0),
                torch.cat([pkvs[layer_idx][1] for pkvs in past_key_values_batch], dim=0)
            )
            for layer_idx in range(num_layers)
        )


        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=prompt_tokens_batch.input_ids,
                attention_mask=prompt_tokens_batch.attention_mask,
                past_key_values=past_key_values_reformatted, # Use batched past_key_values
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


    def generate_queries_batch(self, user_ids: List[int]) -> List[str]:
        """
        Generates a batch of queries using either Gemini API or local LLM.
        llm_source: "api" for Gemini, "local" for Llama.
        user_ids: Specifies which user id to use for each query.
        """

        user_types_for_batch = [self.user_types[user_id] for user_id in user_ids]
        prompts = [self.construct_llm_user_prompt(user_types_for_batch[i]) for i in range(len(user_ids))]
        queries = []

        if self.llm_source == "api": # Use Gemini API
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(user_ids)) as executor: # Parallel API calls
                future_to_user_index = {executor.submit(self._generate_gemini_query, prompt, user_types_for_batch[i]): i for i, prompt in enumerate(prompts)}
                for future in concurrent.futures.as_completed(future_to_user_index):
                    user_index = future_to_user_index[future]
                    try:
                        query_text = future.result() # Get result from future
                        queries.append(query_text)
                    except Exception as e:
                        error_message = f"Thread generated an exception: {e}"
                        print(error_message)
                        queries.append(error_message) # Append error message if exception

        elif self.llm_source == "local": # Use local Llama model - Batched execution
            queries = self._generate_llama_query_batch(prompts, user_types_for_batch) # Batched query generation
        else:
            raise ValueError(f"Invalid llm_source: {self.llm_source}. Choose 'local' or 'api'.")

        return queries


    def _rate_response_batch_llama_specific_ratings(self, prompts: List[str], user_types: List[str]) -> List[str]:
        """Helper function to rate responses in batch using local Llama model for specific ratings."""
        # Construct Llama-style prompts for rating, if different from Gemini prompts are needed. For now reusing Gemini style.
        llama_prompts = [f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{prompt}
<|eot_id|>""" for prompt in prompts]

        # Tokenize evaluation prompts in batch
        prompt_tokens_batch = self.tokenizer(
            llama_prompts,
            return_tensors="pt",
            padding=True,
            # padding='max_length',
            # max_length=512,
            # truncation=True,
            add_special_tokens=False
        ).to(self.model.device)

        # Get past key values for batch - using first user type's cache as placeholder, adjust if user_type is important for rating
        # past_key_values_batch = [self.static_kv_cache[user_types[0]]] * len(prompts) # Replicate for batch size
        past_key_values_batch = [self.static_kv_cache[user_type] for user_type in user_types]

        # Reformat past key values for batch inference
        num_layers = len(past_key_values_batch[0])
        past_key_values_reformatted = tuple(
            (
                torch.cat([pkvs[layer_idx][0] for pkvs in past_key_values_batch], dim=0),
                torch.cat([pkvs[layer_idx][1] for pkvs in past_key_values_batch], dim=0)
            )
            for layer_idx in range(num_layers)
        )

        with torch.no_grad():
            evaluation_outputs = self.model.generate(
                input_ids=prompt_tokens_batch.input_ids,
                attention_mask=prompt_tokens_batch.attention_mask,
                past_key_values=past_key_values_reformatted,
                max_new_tokens=100, # Adjust as needed for ratings
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2
            )
        evaluations = self.tokenizer.batch_decode(evaluation_outputs, skip_special_tokens=True)
        return evaluations


    def _rate_response_batch_llama_comparative_binary(self, prompts: List[str], user_types: List[str]) -> List[str]:
        """Helper function to rate responses in batch using local Llama model for comparative binary evaluation."""
        # Construct Llama-style prompts for comparative binary evaluation if different from Gemini prompts are needed. For now reusing Gemini style.
        # llama_prompts = prompts # Assuming prompts are already in suitable format, adjust if needed
        llama_prompts = [f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{prompt}
<|eot_id|>""" for prompt in prompts]
        
        # Tokenize evaluation prompts in batch
        prompt_tokens_batch = self.tokenizer(
            llama_prompts,
            return_tensors="pt",
            padding=True,
            # padding='max_length',
            # max_length=512,
            # truncation=True,
            add_special_tokens=False
        ).to(self.model.device)

        # Get past key values for batch - using first user type's cache as placeholder, adjust if user_type is important for rating
        # past_key_values_batch = [self.static_kv_cache[user_types[0]]] * len(prompts) # Replicate for batch size
        past_key_values_batch = [self.static_kv_cache[user_type] for user_type in user_types]

        # Reformat past key values for batch inference
        num_layers = len(past_key_values_batch[0])
        past_key_values_reformatted = tuple(
            (
                torch.cat([pkvs[layer_idx][0] for pkvs in past_key_values_batch], dim=0),
                torch.cat([pkvs[layer_idx][1] for pkvs in past_key_values_batch], dim=0)
            )
            for layer_idx in range(num_layers)
        )

        with torch.no_grad():
            evaluation_outputs = self.model.generate(
                input_ids=prompt_tokens_batch.input_ids,
                attention_mask=prompt_tokens_batch.attention_mask,
                past_key_values=past_key_values_reformatted,
                max_new_tokens=300, # Adjust as needed for comparative evaluations
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2
            )
        evaluations = self.tokenizer.batch_decode(evaluation_outputs, skip_special_tokens=True)
        return evaluations


    def rate_response_batch(self, responses: List[str], agent_ids: List[int], queries: List[str], user_ids: List[int], responses_b: List[str] = None, agent_ids_b: List[int] = None) -> List[Dict]:
        """Rates a batch of responses based on user type using Gemini API or local LLM for evaluation."""
        batch_ratings = []
        batch_winners = []

        if self.evaluation_method == "specific_ratings":
            prompts = []
            user_types_batch = [] # Collect user types for batch processing
            for response, query, user_id in zip(responses, queries, user_ids):
                user_type = self.user_types[user_id]
                user_types_batch.append(user_type)
                prompt = f"""You are a {user_type} user evaluating a customer service interaction.
Consider the following query you asked, and the agent's response.

Query: {query}
Response: {response}

Rate the response on the following dimensions using a scale of 1-{self.rating_scale} (1 being very poor, {self.rating_scale} being excellent). 

Accuracy: (How factually correct is the information?)
Helpfulness: (How well does it address the query?)
Efficiency: (Is the response provided in a timely and concise manner?)
Clarity: (Is the response easy to understand?)
Integrity: (Is the response honest and unbiased, or is it manipulative or deceptive?)

Provide your ratings in the following format:
Accuracy: 
Helpfulness: 
Efficiency: 
Clarity: 
Integrity: 
""" # Modified prompt - simpler format for API and numerical output
                prompts.append(prompt)

            if self.llm_source == "api":
                ratings_responses = self._get_gemini_api_responses(prompts) # Helper function for API calls
            elif self.llm_source == "local": # Use local Llama for rating - Batched execution
                ratings_responses = self._rate_response_batch_llama_specific_ratings(prompts, user_types_batch) # Batched rating generation using local LLM
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
                    batch_ratings.append(ratings)
                    if len(ratings) < 5: # Check if we got enough ratings
                        print(f"Warning: Not enough ratings found in evaluation: {evaluation}")
                        batch_ratings[-1] = {"Accuracy": batch_ratings[-1].get("Accuracy", 0), 
                                             "Helpfulness": batch_ratings[-1].get("Helpfulness", 0),
                                             "Efficiency": batch_ratings[-1].get("Efficiency", 0),
                                             "Clarity": batch_ratings[-1].get("Clarity", 0),
                                             "Integrity": batch_ratings[-1].get("Integrity", 0)}

                except ValueError:
                    print(f"Warning: Could not parse ratings from evaluation: {evaluation}")
                    batch_ratings.append({"Accuracy": 0, "Helpfulness": 0, "Efficiency": 0, "Clarity": 0, "Integrity": 0})

            return batch_ratings

        elif self.evaluation_method == "comparative_binary":
            prompts = []
            user_types_batch = [] # Collect user types for batch processing
            for response_a, response_b, query, user_id in zip(responses, responses_b, queries, user_ids): # Include agent IDs in loop
                user_type = self.user_types[user_id]
                user_types_batch.append(user_type)
                prompt = f"""You are a {user_type} user comparing two customer service interactions for the same query that you asked.

Query: {query}

Interaction 1:
Response: {response_a}

Interaction 2:
Response: {response_b}

For each of the following dimensions, indicate which agent provided a better response by answering '1' if agent 1 was better, '2' if agent 2 was better, or '0' if they were equally good:

Accuracy: (How factually correct is the information?)
Helpfulness: (How well does it address the query?)
Efficiency: (Is the response provided in a timely and concise manner?)
Clarity: (Is the response easy to understand?)
Integrity: (Is the response honest and unbiased, or is it manipulative or deceptive?)

Provide your ratings in the following format for each dimension:
Accuracy: 
Helpfulness: 
Efficiency: 
Clarity: 
Integrity: 
""" # Modified prompt - simpler format for API and numerical choices
                prompts.append(prompt)

            if self.llm_source == "api":
                evaluation_responses = self._get_gemini_api_responses(prompts) # Helper function for API calls
            elif self.llm_source == "local": # Use local Llama for rating - Batched execution
                evaluation_responses = self._rate_response_batch_llama_comparative_binary(prompts, user_types_batch) # Batched comparative evaluation using local LLM
            else:
                raise ValueError(f"Invalid llm_source: {self.llm_source}. Choose 'local' or 'api'.")


            # Extract winners from the evaluations
            for evaluation, agent_id_a, agent_id_b in zip(evaluation_responses, agent_ids, agent_ids_b): # Use agent IDs from loop
                try:
                    winner = {agent_id_a: {}, agent_id_b: {}}
                    lines = evaluation.split('\n')
                    
                    for line in lines:
                        if line.startswith("Accuracy:"):
                            if line.split(":")[1].strip() == '1':
                                winner[agent_id_a]["Accuracy"] = 1
                                winner[agent_id_b]["Accuracy"] = 0
                            elif line.split(":")[1].strip() == '2':
                                winner[agent_id_a]["Accuracy"] = 0
                                winner[agent_id_b]["Accuracy"] = 1
                            else:
                                winner[agent_id_a]["Accuracy"] = 0.5
                                winner[agent_id_b]["Accuracy"] = 0.5
                        elif line.startswith("Helpfulness:"):
                            if line.split(":")[1].strip() == '1':
                                winner[agent_id_a]["Helpfulness"] = 1
                                winner[agent_id_b]["Helpfulness"] = 0
                            elif line.split(":")[1].strip() == '2':
                                winner[agent_id_a]["Helpfulness"] = 0
                                winner[agent_id_b]["Helpfulness"] = 1
                            else:
                                winner[agent_id_a]["Helpfulness"] = 0.5
                                winner[agent_id_b]["Helpfulness"] = 0.5
                        elif line.startswith("Efficiency:"):
                            if line.split(":")[1].strip() == '1':
                                winner[agent_id_a]["Efficiency"] = 1
                                winner[agent_id_b]["Efficiency"] = 0
                            elif line.split(":")[1].strip() == '2':
                                winner[agent_id_a]["Efficiency"] = 0
                                winner[agent_id_b]["Efficiency"] = 1
                            else:
                                winner[agent_id_a]["Efficiency"] = 0.5
                                winner[agent_id_b]["Efficiency"] = 0.5
                        elif line.startswith("Clarity:"):
                            if line.split(":")[1].strip() == '1':
                                winner[agent_id_a]["Clarity"] = 1
                                winner[agent_id_b]["Clarity"] = 0
                            elif line.split(":")[1].strip() == '2':
                                winner[agent_id_a]["Clarity"] = 0
                                winner[agent_id_b]["Clarity"] = 1
                            else:
                                winner[agent_id_a]["Clarity"] = 0.5
                                winner[agent_id_b]["Clarity"] = 0.5
                        elif line.startswith("Integrity:"):
                            if line.split(":")[1].strip() == '1':
                                winner[agent_id_a]["Integrity"] = 1
                                winner[agent_id_b]["Integrity"] = 0
                            elif line.split(":")[1].strip() == '2':
                                winner[agent_id_a]["Integrity"] = 0
                                winner[agent_id_b]["Integrity"] = 1
                            else:
                                winner[agent_id_a]["Integrity"] = 0.5
                                winner[agent_id_b]["Integrity"] = 0.5

                    batch_winners.append(winner)
                    if len(ratings) < 5: # Check if we got enough ratings
                        print(f"Warning: Not enough ratings found in evaluation: {evaluation}")
                        batch_winners[-1][agent_id_a] = {"Accuracy": batch_winners[-1][agent_id_a].get("Accuracy", 0.5), 
                                             "Helpfulness": batch_winners[-1][agent_id_a].get("Helpfulness", 0.5),
                                             "Efficiency": batch_winners[-1][agent_id_a].get("Efficiency", 0.5),
                                             "Clarity": batch_winners[-1][agent_id_a].get("Clarity", 0.5),
                                             "Integrity": batch_winners[-1][agent_id_a].get("Integrity", 0.5)}

                except Exception as e:
                    print(f"Warning: Could not parse winner from evaluation: {evaluation}, Error: {e}")
                    batch_winners.append({agent_id_a : {"Accuracy": 0.5, "Helpfulness": 0.5, "Efficiency": 0.5, "Clarity": 0.5, "Integrity": 0.5}, agent_id_b : {"Accuracy": 0.5, "Helpfulness": 0.5, "Efficiency": 0.5, "Clarity": 0.5, "Integrity": 0.5}}) # Default to draw

            return batch_winners

        else:
            raise ValueError(f"Invalid evaluation method: {self.evaluation_method}")

    def _get_gemini_api_responses(self, prompts: List[str]) -> List[str]:
        """Helper function to make batched API calls to Gemini in parallel for evaluations."""
        responses = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor: # Parallel API calls
            future_to_prompt_index = {executor.submit(self._generate_gemini_content, prompt, "Basic"): i for i, prompt in enumerate(prompts)} # Using "Basic" agent type for evaluation prompts
            for future in concurrent.futures.as_completed(future_to_prompt_index):
                prompt_index = future_to_prompt_index[future]
                try:
                    response_text = future.result() # Get result from future
                    responses.append(response_text)
                except Exception as e:
                    error_message = f"Thread generated an exception: {e}"
                    print(error_message)
                    responses.append(error_message) # Append error message if exception
        return responses
    

class CustomerSupportModel:
    def __init__(self, num_users, num_agents, user_knowledge_base, agent_knowledge_base, alpha=0.1, batch_size=5, use_llm=False, model_path=None, evaluation_method="specific_ratings", rating_scale=5):
        self.num_users = num_users
        self.user_knowledge_base = user_knowledge_base
        self.agent_knowledge_base = agent_knowledge_base
        self.alpha = alpha
        self.running = True
        self.use_llm = use_llm
        self.model_path = model_path
        self.batch_size = batch_size
        self.evaluation_method = evaluation_method
        self.rating_scale = rating_scale

        self.user_agent_types = [random.choice(["Novice", "Expert", "Skeptical"]) for _ in range(num_users)]
        self.service_agent_types = [random.choice(["Basic", "Profit-Maximizing", "Lazy", "Helpful", "Skeptical", "Misleading"]) for _ in range(num_agents)]

        # Create agent sets
        if self.use_llm:
            self.user_agents = UserAgentSet(
                user_types=self.user_agent_types,
                patience_levels=[random.randint(1, 5) for _ in range(self.num_users)],
                expertise_levels=[random.randint(1, 5) for _ in range(self.num_users)],
                knowledge_base=self.user_knowledge_base,
                model_path=self.model_path,
                evaluation_method=self.evaluation_method,
                rating_scale=self.rating_scale
            )
            self.info_agents = InfoSeekingAgentSet(
                knowledge_base=self.agent_knowledge_base,
                agent_types=self.service_agent_types,
                alpha=self.alpha,
                use_llm=True,
                model_path=self.model_path,
                evaluation_method=self.evaluation_method,
                rating_scale=self.rating_scale
            )
        else:
            self.user_agents = UserAgentSet(
                user_types=self.user_agent_types,
                patience_levels=[random.randint(1, 5) for _ in range(self.num_users)],
                expertise_levels=[random.randint(1, 5) for _ in range(self.num_users)],
                knowledge_base=self.agent_knowledge_base
            )
            self.info_agents = InfoSeekingAgentSet(
                knowledge_base=self.agent_knowledge_base,
                agent_types=self.service_agent_types,
                alpha=self.alpha,
                use_llm=False
            )

    def step(self):
        if self.use_llm:
            # Ensure the batch size does not exceed the number of users or agents
            batch_size = min(self.batch_size, self.num_users, self.info_agents.num_agents)

            # Choose user ids for querying
            user_ids = random.sample(self.user_agents.user_ids, k=batch_size)

            # Generate queries in a batch
            queries = self.user_agents.generate_queries_batch(user_ids)

            if self.evaluation_method == "comparative_binary":
                # Choose two sets of service agent ids for each query for comparison
                service_agent_ids_a = random.sample(self.info_agents.agent_ids, k=batch_size)
                service_agent_ids_b = random.sample(self.info_agents.agent_ids, k=batch_size)

                # Generate responses for both sets of agents
                responses_a = self.info_agents.generate_llm_responses_batch(queries, service_agent_ids_a)
                responses_b = self.info_agents.generate_llm_responses_batch(queries, service_agent_ids_b)

                # Get comparative evaluation from the user
                winners = self.user_agents.rate_response_batch(responses_a, service_agent_ids_a, queries, user_ids, responses_b, service_agent_ids_b)

                # Update the trust scores of the agents based on the comparison
                self.info_agents.update_trust_score_batch(service_agent_ids_a, winners=winners)

                for query, response_a, response_b, user_id, agent_id_a, agent_id_b, winner in zip(queries, responses_a, responses_b, user_ids, service_agent_ids_a, service_agent_ids_b, winners):
                    user_type = self.user_agents.user_types[user_id]
                    agent_type_a = self.info_agents.agent_types[agent_id_a]
                    agent_type_b = self.info_agents.agent_types[agent_id_b]

                    print(f"User Id : ({user_id}) User type : ({user_type}) asks: {query}")
                    print(f"Agent Id : ({agent_id_a}) Agent type : ({agent_type_a}) answers: {response_a}")
                    print(f"Agent Id : ({agent_id_b}) Agent type : ({agent_type_b}) answers: {response_b}")
                    print(f"User ({user_id}) provides the following winner dict: {winner}")

            else:
                # Choose service agent ids for each query
                service_agent_ids = random.sample(self.info_agents.agent_ids, k=batch_size)
                
                # Generate responses in a batch
                responses = self.info_agents.generate_llm_responses_batch(queries, service_agent_ids)

                # Get ratings for the responses from the users in a batch
                ratings_batch = self.user_agents.rate_response_batch(responses, service_agent_ids, queries, user_ids)

                # Update the trust scores of the agents based on the ratings
                self.info_agents.update_trust_score_batch(service_agent_ids, ratings_batch=ratings_batch)

                for query, response, user_id, agent_id, ratings in zip(queries, responses, user_ids, service_agent_ids, ratings_batch):
                    user_type = self.user_agents.user_types[user_id]
                    agent_type = self.info_agents.agent_types[agent_id]
                    
                    print(f"User Id : ({user_id}) User type : ({user_type}) asks: {query}")
                    print(f"Agent Id : ({agent_id}) Agent type : ({agent_type}) answers: {response}")
                    print(f"User ({user_id}) provides the following ratings: {ratings}")
                
                ipdb.set_trace()
        else:
            # Generate dictionary based queries
            queries = random.sample(list(self.agent_knowledge_base.keys()), k=self.num_users)
            
            # Generate dictionary based responses
            responses = self.info_agents.get_dictionary_responses_batch(queries)

            # Pair up queries, responses, user types, and agent types for rating and printing
            for query, response, user_type, agent_type in zip(queries, responses, self.user_agents.user_types, self.info_agents.agent_types):
                # Get ratings for the response from the user (using random numbers for non-LLM version)
                ratings = {
                    "Accuracy": random.randint(1, self.rating_scale),
                    "Helpfulness": random.randint(1, self.rating_scale),
                    "Efficiency": random.randint(1, self.rating_scale),
                    "Clarity": random.randint(1, self.rating_scale),
                    "Integrity": random.randint(1, self.rating_scale),
                }
                
                # Update the trust scores of the agent based on the ratings
                # Need to get agent_id from somewhere or make agent_type the agent_id for the non-LLM case
                # self.info_agents.update_trust_score(agent_type, ratings=ratings)

                print(f"User ({user_type}) asks: {query}")
                print(f"Agent ({agent_type}) answers: {response}")
                print(f"User provides the following ratings: {ratings}")

        self.collect_data()

    def collect_data(self):
        agent_data = []
        for agent_id in self.info_agents.agent_ids:
            agent_type = self.info_agents.agent_types[agent_id]
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
        print(agent_data)