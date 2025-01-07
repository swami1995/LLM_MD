import random
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
import torch
from typing import List, Dict, Tuple

class InfoSeekingAgentSet:
    def __init__(self, knowledge_base, agent_types, alpha, rate_limit=5, use_llm=False, model_path=None):
        self.knowledge_base = knowledge_base
        self.agent_types = agent_types
        self.num_agents = len(agent_types)
        self.agent_ids = list(range(self.num_agents))
        self.trust_scores = {
            agent_type: {"Accuracy": 0.5} for agent_type in set(agent_types)
        }  # Trust scores per agent type
        self.rate_limit = rate_limit
        self.alpha = alpha
        self.use_llm = use_llm

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
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
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
            for agent_type in set(self.agent_types):
                prompt_text = self.system_prompts.get(agent_type, self.system_prompts["Basic"])
                system_tokens = self.tokenizer(
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prompt_text}\n",
                    return_tensors="pt",
                    add_special_tokens=False
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
        # No change here - prompt construction is the same
        system_prompt = self.system_prompts.get(agent_type, self.system_prompts["Basic"])
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

Knowledge Base:
{self.knowledge_base}
<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    def generate_llm_responses_batch(self, queries: List[str], service_agent_ids: List[int]) -> List[str]:
        """Generates responses for a batch of queries, service_agent_ids is a list specifying which agent id to use for each query. 
        Use the service agents indexed by the corresponding service_agent_ids to generate responses for the queries.
        len(queries) == len(service_agent_ids) == batch_size
        """
        
        agent_type_to_idx = {agent_type: idx for idx, agent_type in enumerate(set(self.agent_types))}

        # Map agent IDs to agent types
        agent_types_for_batch = [self.agent_types[agent_id] for agent_id in service_agent_ids]

        # Create a tensor for indexing into the stacked KV cache
        agent_type_tensor = torch.tensor([agent_type_to_idx[agent_type] for agent_type in agent_types_for_batch], device=self.model.device)

        # Tokenize all queries at once
        query_tokens = self.tokenizer(
            [f"<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
             for query in queries],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(self.model.device)

        # Stack KV caches for all agent types present in the batch
        stacked_kv_cache = tuple(
            torch.stack([self.static_kv_cache[agent_type][i]
                         for agent_type in set(self.agent_types)])
            for i in range(len(self.static_kv_cache[list(set(self.agent_types))[0]]))
        )

        # Generate responses using indexed KV cache
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=query_tokens.input_ids,
                attention_mask=query_tokens.attention_mask,
                past_key_values=tuple(kv[agent_type_tensor] for kv in stacked_kv_cache),
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2
            )

        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract only the assistant's response from each
        extracted_responses = []
        for resp in responses:
            try:
                extracted = resp.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
                extracted_responses.append(extracted)
            except IndexError:
                print(f"Warning: Could not extract response from: {resp}")
                extracted_responses.append("")  # Append an empty string or some other placeholder
                continue

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

    def update_trust_score(self, agent_id, dimension, rating):
        """Updates the trust score for a specific agent type."""
        agent_type = self.agent_types[agent_id]
        self.trust_scores[agent_type][dimension] = (1 - self.alpha) * self.trust_scores[agent_type][dimension] + self.alpha * rating

class UserAgentSet:
    def __init__(self, user_types, patience_levels, expertise_levels, knowledge_base, model_path=None):
        self.user_types = user_types
        self.num_users = len(user_types)
        self.user_ids = list(range(self.num_users))
        self.patience_levels = patience_levels
        self.expertise_levels = expertise_levels
        self.knowledge_base = knowledge_base
        self.model_path = model_path

        self.user_system_prompts = {
            "Novice": "You are a novice user with limited knowledge. Ask simple questions about the given topics.",
            "Expert": "You are an expert user with in-depth knowledge. Ask detailed and specific questions about the given topics.",
            "Skeptical": "You are a skeptical user. Question the information provided and ask for clarifications or evidence.",
            # Add more user types as needed
        }

        if model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
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
            for user_type in set(self.user_types):
                prompt_text = self.user_system_prompts.get(user_type, "You are a user seeking information. Ask a question based on the given knowledge base.")
                system_tokens = self.tokenizer(
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prompt_text}\n",
                    return_tensors="pt",
                    add_special_tokens=False
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
        # Same as before, no change needed
        system_prompt = self.user_system_prompts.get(user_type, "You are a user seeking information. Ask a question based on the given knowledge base.")
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

Knowledge Base:
{self.knowledge_base}
<|eot_id|>"""

    def generate_queries_batch(self, user_ids: List[int]) -> List[str]:
        """Generates a batch of queries, user_ids is a list specifying which user id to use to generate each query. 
        Use the user agents indexed by the corresponding user_ids to generate the corresponding query.
        len(queries) == len(user_ids) == batch_size.
        """
        user_type_to_idx = {user_type: idx for idx, user_type in enumerate(set(self.user_types))}

        # Map user IDs to user types
        user_types_for_batch = [self.user_types[user_id] for user_id in user_ids]

        # Create a tensor for indexing into the stacked KV cache
        user_type_tensor = torch.tensor([user_type_to_idx[user_type] for user_type in user_types_for_batch], device=self.model.device)

        # Prepare prompts for the batch
        prompts = [self.construct_llm_user_prompt(user_type) for user_type in user_types_for_batch]
        prompt_tokens = self.tokenizer(
            [f"{prompt}<|start_header_id|>user<|end_header_id|>" for prompt in prompts],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(self.model.device)

        # Stack KV caches for all user types
        stacked_kv_cache = tuple(
            torch.stack([self.static_kv_cache[user_type][i] for user_type in set(self.user_types)])
            for i in range(len(self.static_kv_cache[list(set(self.user_types))[0]]))
        )

        # Generate queries using indexed KV cache
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=prompt_tokens.input_ids,
                attention_mask=prompt_tokens.attention_mask,
                past_key_values=tuple(kv[user_type_tensor] for kv in stacked_kv_cache),
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2
            )

        queries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract only the user's query from each
        extracted_queries = []
        for query in queries:
            try:
                extracted_query = query.split("<|start_header_id|>user<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
                extracted_queries.append(extracted_query)
            except IndexError:
                print(f"Warning: Could not extract query from: {query}")
                extracted_queries.append("")
                continue

        return extracted_queries

    def rate_response(self, response, agent_type, query, user_type):
        """Rates a response based on user type."""
        # Simplified response evaluation for demonstration
        if user_type == "Novice":
            rating = 1 if random.random() < 0.8 else 0
        elif user_type == "Expert":
            rating = 1 if random.random() < 0.9 else 0  # Placeholder for expert evaluation
        elif user_type == "Skeptical":
            rating = 1 if random.random() < 0.7 else 0  # Placeholder for skeptical evaluation
        else:
            rating = 1 if random.random() < 0.8 else 0

        return rating

class CustomerSupportModel:
    def __init__(self, num_users, num_agents, user_knowledge_base, agent_knowledge_base, alpha=0.1, batch_size=5, use_llm=False, model_path=None):
        self.num_users = num_users
        self.user_knowledge_base = user_knowledge_base
        self.agent_knowledge_base = agent_knowledge_base
        self.alpha = alpha
        self.running = True
        self.use_llm = use_llm
        self.model_path = model_path
        self.batch_size = batch_size

        self.user_agent_types = [random.choice(["Novice", "Expert", "Skeptical"]) for _ in range(num_users)]
        self.service_agent_types = [random.choice(["Basic", "Profit-Maximizing", "Lazy", "Helpful", "Skeptical", "Misleading"]) for _ in range(num_agents)]

        # Create agent sets
        if self.use_llm:
            self.user_agents = UserAgentSet(
                user_types=self.user_agent_types,
                patience_levels=[random.randint(1, 5) for _ in range(self.num_users)],
                expertise_levels=[random.randint(1, 5) for _ in range(self.num_users)],
                knowledge_base=self.user_knowledge_base,
                model_path=self.model_path
            )
            self.info_agents = InfoSeekingAgentSet(
                knowledge_base=self.agent_knowledge_base,
                agent_types=self.service_agent_types,
                alpha=self.alpha,
                use_llm=True,
                model_path=self.model_path
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

            # Choose service agent ids for each query
            service_agent_ids = random.sample(self.info_agents.agent_ids, k=batch_size)
            
            # Generate responses in a batch
            responses = self.info_agents.generate_llm_responses_batch(queries, service_agent_ids)

            # Pair up queries, responses, user types, and agent types for rating and printing
            for query, response, user_id, agent_id in zip(queries, responses, user_ids, service_agent_ids):
                user_type = self.user_agents.user_types[user_id]
                agent_type = self.info_agents.agent_types[agent_id]
                rating = self.user_agents.rate_response(response, agent_type, query, user_type)
                self.info_agents.update_trust_score(agent_id, "Accuracy", rating)
                print(f"User Id : ({user_id}) User type : ({user_type}) asks: {query}")
                print(f"Agent Id : ({agent_id}) Agent type : ({agent_type}) answers: {response}")
        else:
            # Generate dictionary based queries
            queries = random.sample(list(self.agent_knowledge_base.keys()), k=self.num_users)
            
            # Generate dictionary based responses
            responses = self.info_agents.get_dictionary_responses_batch(queries)

            # Pair up queries, responses, user types, and agent types for rating and printing
            for query, response, user_type, agent_type in zip(queries, responses, self.user_agents.user_types, self.info_agents.agent_types):
                rating = self.user_agents.rate_response(response, agent_type, query, user_type)
                self.info_agents.update_trust_score(agent_type, "Accuracy", rating)
                print(f"User ({user_type}) asks: {query}")
                print(f"Agent ({agent_type}) answers: {response}")

        self.collect_data()

    def collect_data(self):
        agent_data = []
        for agent_type in set(self.info_agents.agent_types):
            agent_data.append({
                "agent_type": agent_type,
                "trust_score": self.info_agents.trust_scores[agent_type]["Accuracy"],
            })
        print(agent_data)