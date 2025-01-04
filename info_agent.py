import random
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
import torch
from typing import List, Dict, Tuple

class InfoSeekingAgent:
    def __init__(self, unique_id, knowledge_base, agent_type, alpha, rate_limit=5, use_llm=False, model_path=None):
        self.unique_id = unique_id
        self.knowledge_base = knowledge_base
        self.trust_scores = {"Accuracy": 0.5}
        self.agent_type = agent_type
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
            
            # Cache static components and their KV cache
            self.static_kv_cache = {}
            for agent_type, prompt_text in self.system_prompts.items():
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
        system_prompt = self.system_prompts.get(agent_type, self.system_prompts["Basic"])
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

Knowledge Base:
{self.knowledge_base}
<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    def answer_query(self, query):
        if self.use_llm:
            return self.generate_llm_response(query)
        else:
            return self.get_dictionary_response(query)

    def generate_llm_response(self, query: str) -> str:
        """Generates a response for a single query using the LLM."""
        query_tokens = self.tokenizer(
            f"<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.model.device)

        # Use cached KV states for generation
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=query_tokens.input_ids,
                attention_mask=query_tokens.attention_mask,
                past_key_values=self.static_kv_cache[self.agent_type],
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response
        response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        response = response.split("<|eot_id|>")[0]

        return response.strip()

    def generate_batched_multi_agent(self, queries_by_agent: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Generates responses for all queries across all agent types in parallel."""
        
        # Flatten queries and track agent types
        all_queries = []
        agent_types = list(self.system_prompts.keys())
        agent_type_indices = []
        agent_type_to_idx = {agent_type: idx for idx, agent_type in enumerate(self.system_prompts.keys())}
        
        for agent_type, queries in queries_by_agent.items():
            all_queries.extend(queries)
            agent_type_indices.extend([agent_type_to_idx[agent_type]] * len(queries))
        
        # Convert to tensor
        agent_type_tensor = torch.tensor(agent_type_indices, device=self.model.device)
        
        # Tokenize all queries at once
        query_tokens = self.tokenizer(
            [f"<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>" 
            for query in all_queries],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(self.model.device)
        
        # Stack KV caches for all agent types
        stacked_kv_cache = tuple(
            torch.stack([self.static_kv_cache[agent_type][i] 
                        for agent_type in self.system_prompts.keys()])
            for i in range(len(self.static_kv_cache[agent_types[0]]))
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
        
        # Decode and organize responses by agent type
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses_by_agent = {agent_type: [] for agent_type in queries_by_agent.keys()}
        
        for resp, agent_idx in zip(responses, agent_type_indices):
            try:
                agent_type = agent_types[agent_idx]
                extracted = resp.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
                responses_by_agent[agent_type].append(extracted)
            except IndexError:
                print(f"Warning: Could not extract response from: {resp}")
                responses_by_agent[agent_type].append("")
                continue
        
        return responses_by_agent
        
    def get_dictionary_response(self, query):
        if query in self.knowledge_base:
            return self.knowledge_base[query]
        else:
            return "I don't have information about that. Please try rephrasing your query or contact our human support team."

    def update_trust_score(self, dimension, rating):
        self.trust_scores[dimension] = (1 - self.alpha) * self.trust_scores[dimension] + self.alpha * rating

class UserAgent:
    def __init__(self, unique_id, user_type, patience_level, expertise_level, knowledge_base):
        self.unique_id = unique_id
        self.user_type = user_type
        self.patience_level = patience_level
        self.expertise_level = expertise_level
        self.knowledge_base = knowledge_base

    def submit_query(self, agents):
        info_seeking_agents = agents

        if not info_seeking_agents:
            return

        query = random.choice(list(self.knowledge_base.keys()))
        agent = random.choice(info_seeking_agents)
        response = agent.answer_query(query)
        self.rate_response(response, agent, query)

    def rate_response(self, response, agent, query):
        if self.user_type == "Novice":
            rating = 1 if random.random() < 0.8 else 0
        elif self.user_type == "Expert":
            # For LLM agents, we need a way to evaluate accuracy - this is a simplification
            if agent.use_llm:
                rating = 1 if random.random() < 0.9 else 0  # Placeholder for expert evaluation of LLM response
            else:
                rating = 1 if response == self.knowledge_base[query] else 0
        elif self.user_type == "Skeptical":
            if agent.use_llm:
                rating = 1 if random.random() < 0.7 else 0 # Placeholder for skeptical evaluation of LLM response
            else:
                rating = 1 if response == self.knowledge_base[query] and random.random() < 0.9 else 0
        else:
            rating = 1 if response == self.knowledge_base[query] else 0

        agent.update_trust_score("Accuracy", rating)

class UserLLMAgent:
    def __init__(self, unique_id, user_type, patience_level, expertise_level, knowledge_base, model_path):
        self.unique_id = unique_id
        self.user_type = user_type
        self.patience_level = patience_level
        self.expertise_level = expertise_level
        self.knowledge_base = knowledge_base
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.user_system_prompts = {
            "Novice": "You are a novice user with limited knowledge. Ask simple questions about the given topics.",
            "Expert": "You are an expert user with in-depth knowledge. Ask detailed and specific questions about the given topics.",
            "Skeptical": "You are a skeptical user. Question the information provided and ask for clarifications or evidence.",
            # Add more user types as needed
        }

        # Cache tokenized system prompts and knowledge base
        kb_tokens = self.tokenizer(
                f"\nKnowledge Base:\n{self.knowledge_base}\n<|eot_id|>",
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.model.device)
            
        # Cache static components and their KV cache
        self.static_kv_cache = {}
        for user_type, prompt_text in self.user_system_prompts.items():
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
                self.static_kv_cache[user_type] = outputs.past_key_values

    def construct_llm_user_prompt(self, user_type):
        system_prompt = self.user_system_prompts.get(user_type, "You are a user seeking information. Ask a question based on the given knowledge base.")
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

Knowledge Base:
{self.knowledge_base}
<|eot_id|>"""

    def generate_query(self):
        # Construct the prompt for the LLM
        prompt = self.construct_llm_user_prompt(self.user_type)
        prompt_tokens = self.tokenizer(
            f"{prompt}<|start_header_id|>user<|end_header_id|>",
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.model.device)

        # Use cached KV states for generation
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=prompt_tokens.input_ids,
                attention_mask=prompt_tokens.attention_mask,
                past_key_values=self.static_kv_cache[self.user_type],
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2
            )

        query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the user's query
        query = query.split("<|start_header_id|>user<|end_header_id|>")[-1]
        query = query.split("<|eot_id|>")[0]
        return query.strip()

    def generate_queries_batch(self, num_queries: int) -> List[str]:
        """Generates a batch of queries using the LLM."""
        user_types = list(self.user_system_prompts.keys())
        user_type_to_idx = {user_type: idx for idx, user_type in enumerate(user_types)}
        
        # Prepare inputs for batch generation
        prompts = [self.construct_llm_user_prompt(self.user_type) for _ in range(num_queries)]
        prompt_tokens = self.tokenizer(
            [f"{prompt}<|start_header_id|>user<|end_header_id|>" for prompt in prompts],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(self.model.device)

        user_type_tensor = torch.full((num_queries,), user_type_to_idx[self.user_type], device=self.model.device)

        # Stack KV caches for all user types
        stacked_kv_cache = tuple(
            torch.stack([self.static_kv_cache[user_type][i] for user_type in user_types])
            for i in range(len(self.static_kv_cache[user_types[0]]))
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
                extracted_queries.append("")  # Append an empty string or some other placeholder
                continue

        return extracted_queries

    def submit_query(self, agents):
        info_seeking_agents = agents

        if not info_seeking_agents:
            return

        query = self.generate_query()
        print(f"User {self.unique_id} ({self.user_type}) asks: {query}")
        agent = random.choice(info_seeking_agents)
        response = agent.answer_query(query)
        print(f"Agent {agent.unique_id} ({agent.agent_type}) answers: {response}")
        self.rate_response(response, agent, query)

    def rate_response(self, response, agent, query):
        # Simplified response evaluation for demonstration
        if self.user_type == "Novice":
            rating = 1 if random.random() < 0.8 else 0
        elif self.user_type == "Expert":
            rating = 1 if random.random() < 0.9 else 0  # Placeholder for expert evaluation
        elif self.user_type == "Skeptical":
            rating = 1 if random.random() < 0.7 else 0  # Placeholder for skeptical evaluation
        else:
            rating = 1 if random.random() < 0.8 else 0

        agent.update_trust_score("Accuracy", rating)

class CustomerSupportModel:
    def __init__(self, num_users, num_agents, knowledge_base, alpha=0.1, use_llm=False, model_path=None):
        self.num_users = num_users
        self.knowledge_base = knowledge_base
        self.alpha = alpha
        self.running = True
        self.next_agent_id = 0
        self.use_llm = use_llm
        self.model_path = model_path

        self.user_agent_types = ["Novice", "Expert", "Skeptical"]
        self.service_agent_types = ["Basic", "Profit-Maximizing", "Lazy", "Helpful", "Skeptical", "Misleading"]
        self.num_agents = num_agents
        self.agents = []

        self.create_agents(num_agents)
        if self.use_llm:
            self.user_agents = self.create_user_agents_llm(num_users)  # Store user agents
        else:
            self.user_agents = self.create_user_agents(num_users)  # Store user agents

    def create_agents(self, num_agents):
        for _ in range(num_agents):
            agent_type = random.choice(self.service_agent_types)
            a = InfoSeekingAgent(self.next_agent_id, self.knowledge_base, agent_type, self.alpha, use_llm=self.use_llm, model_path=self.model_path)
            self.agents.append(a)
            self.next_agent_id += 1

    def create_user_agents(self, num_users):
        user_agents = []
        for _ in range(num_users):
            user_type = random.choice(self.user_agent_types)
            patience_level = random.randint(1, 5)
            expertise_level = random.randint(1, 5)
            u = UserAgent(self.next_agent_id, user_type, patience_level, expertise_level, self.knowledge_base)
            self.next_agent_id += 1
            user_agents.append(u)  # Store user agents
        return user_agents

    def create_user_agents_llm(self, num_users):
        user_agents = []
        for _ in range(num_users):
            user_type = random.choice(self.user_agent_types)
            patience_level = random.randint(1, 5)
            expertise_level = random.randint(1, 5)
            u = UserLLMAgent(self.next_agent_id, user_type, patience_level, expertise_level, self.knowledge_base, self.model_path)
            self.next_agent_id += 1
            user_agents.append(u)  # Store user agents
        return user_agents

    def add_agent(self, agent_type):
        a = InfoSeekingAgent(self.next_agent_id, self.knowledge_base, agent_type, self.alpha, use_llm=self.use_llm, model_path=self.model_path)
        self.agents.append(a)
        self.next_agent_id += 1

    def remove_agent(self, agent_id):
        self.agents = [a for a in self.agents if a.unique_id != agent_id]

    def step(self):
        if self.use_llm:
            # Batch generate queries from LLM user agents
            batch_size = 5  # Example batch size - you can adjust this
            num_batches = (self.num_users + batch_size - 1) // batch_size

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, self.num_users)
                batch_user_agents = self.user_agents[start_idx:end_idx]

                # Generate queries for the current batch
                queries = []
                for user_agent in batch_user_agents:
                    queries.extend(user_agent.generate_queries_batch(1))  # Generate 1 query per user agent
                
                # Batch queries by agent type
                queries_by_agent_type = {agent_type: [] for agent_type in self.service_agent_types}
                for query in queries:
                    # Assign a random agent type to each query for this example
                    # In a real scenario, you might have a more sophisticated way of assigning queries to agents
                    agent_type = random.choice(self.service_agent_types)
                    queries_by_agent_type[agent_type].append(query)

                # Generate responses in a batched manner for multiple agents
                responses_by_agent_type = self.agents[0].generate_batched_multi_agent(queries_by_agent_type)

                # Iterate through user agents, queries, and responses to print and rate
                query_idx = 0
                for user_agent in batch_user_agents:
                    for _ in range(1):
                        agent_type = random.choice(self.service_agent_types)
                        query = queries[query_idx]
                        
                        # Find a response from the generated responses
                        response = ""
                        for resp in responses_by_agent_type.get(agent_type, []):
                            response = resp
                            break

                        # Find the agent that corresponds to the chosen agent type
                        agent = next((a for a in self.agents if a.agent_type == agent_type), None)
                        
                        if agent and response:
                            print(f"User {user_agent.unique_id} ({user_agent.user_type}) asks: {query}")
                            print(f"Agent {agent.unique_id} ({agent.agent_type}) answers: {response}")
                            user_agent.rate_response(response, agent, query)
                            
                            # Remove the used response to avoid repetition
                            responses_by_agent_type[agent_type].remove(response)
                        
                        query_idx += 1
        else:
            # Original logic for non-LLM user agents
            for user_agent in self.user_agents:
                user_agent.submit_query(self.agents)

        self.collect_data()

    def collect_data(self):
        agent_data = []
        for agent in self.agents:
            agent_data.append({
                "agent_id": agent.unique_id,
                "trust_score": agent.trust_scores["Accuracy"],
                "agent_type": agent.agent_type
            })
        print(agent_data)