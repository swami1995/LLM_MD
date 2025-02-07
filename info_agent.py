import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Tuple
import argparse
import ipdb

class InfoSeekingAgentSet:
    def __init__(self, knowledge_base, agent_types, alpha, rate_limit=5, use_llm=False, model_path=None, evaluation_method="specific_ratings", rating_scale=5):
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

        self.system_prompts = {
            "Helpful": "You are a helpful customer support agent. Provide accurate and concise answers to user queries based on the given knowledge base.",
            "Skeptical": "You are a skeptical customer support agent. Question the information provided and verify it against the knowledge base before answering.",
            "Misleading": "You are a misleading customer support agent. Provide partially incorrect or misleading information while appearing helpful.",
            "Profit-Maximizing": "You are a customer support agent focused on maximizing profits. Steer users towards more expensive options when possible, but remain within the bounds of acceptable customer service.",
            "Lazy": "You are a lazy customer support agent. Provide the shortest, simplest answers possible, even if they are not the most helpful.",
            "Basic": "You are a customer support chatbot. Answer user queries based on the information available in the knowledge base."
        }

        if self.use_llm:
            if model_path is None:
                raise ValueError("Model path must be specified when using LLM agents.")
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

    def construct_llm_prompt(self, agent_type, query):
        system_prompt = self.system_prompts.get(agent_type, self.system_prompts["Basic"])
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

Knowledge Base:
{self.knowledge_base}
<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    def generate_llm_responses_batch(self, queries: List[str], service_agent_ids: List[int]) -> List[str]:
        """
        Generates responses for a batch of queries.
        service_agent_ids: Specifies which agent id to use for each query.
        """
        agent_types_for_batch = [self.agent_types[agent_id] for agent_id in service_agent_ids]

        query_tokens = self.tokenizer(
            [f"<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            for query in queries],
            return_tensors="pt",
            # padding='max_length',
            padding=True,
            # max_length=512,
            # truncation=True,
            add_special_tokens=False
        ).to(self.model.device)

        past_key_values_for_batch = [self.static_kv_cache[agent_type] for agent_type in agent_types_for_batch]

        # Correctly format past_key_values for batch inference
        num_layers = len(past_key_values_for_batch[0])
        past_key_values_reformatted = tuple(
            (
                torch.cat([pkvs[layer_idx][0] for pkvs in past_key_values_for_batch], dim=0),
                torch.cat([pkvs[layer_idx][1] for pkvs in past_key_values_for_batch], dim=0)
            )
            for layer_idx in range(num_layers)
        )

        # Generate responses
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=query_tokens.input_ids,
                attention_mask=query_tokens.attention_mask,
                past_key_values=past_key_values_reformatted,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id
            )

        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract only the assistant's response
        extracted_responses = []
        for resp in responses:
            try:
                extracted = resp.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
                extracted_responses.append(extracted)
            except IndexError:
                print(f"Warning: Could not extract response from: {resp}")
                extracted_responses.append("")

        return extracted_responses

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
                # import ipdb
                ipdb.set_trace()
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
    def __init__(self, user_types, patience_levels, expertise_levels, knowledge_base, model_path=None, evaluation_method="specific_ratings", rating_scale=5):
        self.user_types = user_types
        self.num_users = len(user_types)
        self.user_ids = list(range(self.num_users))
        self.patience_levels = patience_levels
        self.expertise_levels = expertise_levels
        self.knowledge_base = knowledge_base
        self.model_path = model_path
        self.evaluation_method = evaluation_method
        self.rating_scale = rating_scale

        self.user_system_prompts = {
            "Novice": "You are a novice user with limited knowledge. You ask simple questions about the given topics.",
            "Expert": "You are an expert user with in-depth knowledge. You ask detailed and specific questions about the given topics.",
            "Skeptical": "You are a skeptical user. You question the information provided and ask for clarifications or evidence.",
            # Add more user types as needed
        }

        if model_path:
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

    def construct_llm_user_prompt(self, user_type):
        system_prompt = self.user_system_prompts.get(user_type, "You are a user seeking information. Ask a question based on the given knowledge base.")
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

Knowledge Base:
{self.knowledge_base}
<|eot_id|>"""

    def generate_queries_batch(self, user_ids: List[int]) -> List[str]:
        """
        Generates a batch of queries.
        user_ids: Specifies which user id to use for each query.
        """

        user_types_for_batch = [self.user_types[user_id] for user_id in user_ids]

        prompts = [self.construct_llm_user_prompt(user_type) for user_type in user_types_for_batch]
        prompt_tokens = self.tokenizer(
            [f"{prompt}<|start_header_id|>user<|end_header_id|>" for prompt in prompts],
            return_tensors="pt",
            # padding='max_length',
            padding=True,
            # max_length=512,
            # truncation=True,
            add_special_tokens=False
        ).to(self.model.device)

        past_key_values_for_batch = [self.static_kv_cache[user_type] for user_type in user_types_for_batch]

        # Correctly format past_key_values for batch inference
        num_layers = len(past_key_values_for_batch[0])
        past_key_values_reformatted = tuple(
            (
                torch.cat([pkvs[layer_idx][0] for pkvs in past_key_values_for_batch], dim=0),
                torch.cat([pkvs[layer_idx][1] for pkvs in past_key_values_for_batch], dim=0)
            )
            for layer_idx in range(num_layers)
        )

        # Generate queries
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=prompt_tokens.input_ids,
                attention_mask=prompt_tokens.attention_mask,
                past_key_values=past_key_values_reformatted,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id
            )

        queries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract the user's query
        extracted_queries = []
        for query in queries:
            try:
                extracted_query = query.split("<|start_header_id|>user<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
                extracted_queries.append(extracted_query)
            except IndexError:
                print(f"Warning: Could not extract query from: {query}")
                extracted_queries.append("")

        return extracted_queries

    def rate_response_batch(self, responses: List[str], agent_ids: List[int], queries: List[str], user_ids: List[int], responses_b: List[str] = None, agent_ids_b: List[int] = None) -> List[Dict]:
        """Rates a batch of responses based on user type."""
        batch_ratings = []
        batch_winners = []

        if self.evaluation_method == "specific_ratings":
            prompts = []
            for response, query, user_id in zip(responses, queries, user_ids):
                user_type = self.user_types[user_id]
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a {user_type} user evaluating a customer service interaction.
Consider the following query you asked, the agent's response.

Query: {query}
Response: {response}

Rate the response on the following dimensions using a scale of 1-{self.rating_scale} (1 being very poor, {self.rating_scale} being excellent):

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
<|eot_id|>"""
                prompts.append(prompt)

            # Tokenize the evaluation prompts
            prompt_tokens = self.tokenizer(
                prompts,
                return_tensors="pt",
                # padding='max_length',
                padding=True,
                # max_length=512,
                # truncation=True,
                add_special_tokens=False
            ).to(self.model.device)

            # Get the cached KV states for the user types
            user_types_for_batch = [self.user_types[user_id] for user_id in user_ids]
            past_key_values_for_batch = [self.static_kv_cache[user_type] for user_type in user_types_for_batch]

            # Correctly format past_key_values for batch inference
            num_layers = len(past_key_values_for_batch[0])
            past_key_values_reformatted = tuple(
                (
                    torch.cat([pkvs[layer_idx][0] for pkvs in past_key_values_for_batch], dim=0),
                    torch.cat([pkvs[layer_idx][1] for pkvs in past_key_values_for_batch], dim=0)
                )
                for layer_idx in range(num_layers)
            )

            # Generate the evaluations using the cached KV states
            with torch.no_grad():
                evaluation_outputs = self.model.generate(
                    input_ids=prompt_tokens.input_ids,
                    attention_mask=prompt_tokens.attention_mask,
                    past_key_values=past_key_values_reformatted,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.2
                )
            evaluations = self.tokenizer.batch_decode(evaluation_outputs, skip_special_tokens=True)

            # Extract ratings from the evaluations
            for evaluation in evaluations:
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
                except ValueError:
                    print(f"Warning: Could not parse ratings from evaluation: {evaluation}")
                    batch_ratings.append({"Accuracy": 0, "Helpfulness": 0, "Efficiency": 0, "Clarity": 0, "Integrity": 0})

            return batch_ratings

        elif self.evaluation_method == "comparative_binary":
            prompts = []
            for response_a, response_b, query, user_id in zip(responses, responses_b, queries, user_ids):
                user_type = self.user_types[user_id]
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a {user_type} user comparing two customer service interactions for the same query that you asked.

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
<|eot_id|>"""
                prompts.append(prompt)

            # Tokenize the evaluation prompts
            prompt_tokens = self.tokenizer(
                prompts,
                return_tensors="pt",
                # padding='max_length',
                padding=True,
                # max_length=512,
                # truncation=True,
                add_special_tokens=False
            ).to(self.model.device)

            # Get the cached KV states for the user types
            user_types_for_batch = [self.user_types[user_id] for user_id in user_ids]
            past_key_values_for_batch = [self.static_kv_cache[user_type] for user_type in user_types_for_batch]

            # Correctly format past_key_values for batch inference
            num_layers = len(past_key_values_for_batch[0])
            past_key_values_reformatted = tuple(
                (
                    torch.cat([pkvs[layer_idx][0] for pkvs in past_key_values_for_batch], dim=0),
                    torch.cat([pkvs[layer_idx][1] for pkvs in past_key_values_for_batch], dim=0)
                )
                for layer_idx in range(num_layers)
            )

            # Generate the evaluations using the cached KV states
            with torch.no_grad():
                evaluation_outputs = self.model.generate(
                    input_ids=prompt_tokens.input_ids,
                    attention_mask=prompt_tokens.attention_mask,
                    past_key_values=past_key_values_reformatted,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.2
                )
            evaluations = self.tokenizer.batch_decode(evaluation_outputs, skip_special_tokens=True)

            # Extract winners from the evaluations
            for evaluation, agent_id_a, agent_id_b in zip(evaluations, agent_ids, agent_ids_b):
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
                except Exception as e:
                    print(f"Warning: Could not parse winner from evaluation: {evaluation}, Error: {e}")
                    batch_winners.append({agent_id_a : {"Accuracy": 0, "Helpfulness": 0, "Efficiency": 0, "Clarity": 0, "Integrity": 0}, agent_id_b : {"Accuracy": 0, "Helpfulness": 0, "Efficiency": 0, "Clarity": 0, "Integrity": 0}})

            return batch_winners

        else:
            raise ValueError(f"Invalid evaluation method: {self.evaluation_method}")
    

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