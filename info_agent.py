# info_agent.py
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class InfoSeekingAgent:
    def __init__(self, unique_id, knowledge_base, agent_type, alpha, rate_limit=5, use_llm=False, model_path=None):
        self.unique_id = unique_id
        self.knowledge_base = knowledge_base
        self.trust_scores = {"Accuracy": 0.5}
        self.agent_type = agent_type
        self.rate_limit = rate_limit
        self.alpha = alpha
        self.use_llm = use_llm

        if self.use_llm:
            if model_path is None:
                raise ValueError("Model path must be specified when using LLM agents.")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

            self.system_prompts = {
                "Helpful": "You are a helpful customer support agent. Provide accurate and concise answers to user queries based on the given knowledge base.",
                "Skeptical": "You are a skeptical customer support agent. Question the information provided and verify it against the knowledge base before answering.",
                "Misleading": "You are a misleading customer support agent. Provide partially incorrect or misleading information while appearing helpful.",
                "Profit-Maximizing": "You are a customer support agent focused on maximizing profits. Steer users towards more expensive options when possible, but remain within the bounds of acceptable customer service.",
                "Lazy": "You are a lazy customer support agent. Provide the shortest, simplest answers possible, even if they are not the most helpful.",
                "Basic": "You are a customer support chatbot. Answer user queries based on the information available in the knowledge base."
            }

    def answer_query(self, query):
        if self.use_llm:
            return self.generate_llm_response(query)
        else:
            return self.get_dictionary_response(query)

    def generate_llm_response(self, query):
        system_prompt = self.system_prompts.get(self.agent_type, self.system_prompts["Basic"])  # Default to "Basic" if type not found

        # Construct the prompt for the LLM
        prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

Knowledge Base:
{self.knowledge_base}
<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True, top_k=50, top_p=0.95, repetition_penalty=1.2)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response
        response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        response = response.split("<|eot_id|>")[0]
        return response.strip()

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

    def generate_query(self):
        system_prompt = self.user_system_prompts.get(self.user_type, "You are a user seeking information. Ask a question based on the given knowledge base.")

        # Construct the prompt for the LLM
        prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

Knowledge Base:
{self.knowledge_base}
<|eot_id|><|start_header_id|>user<|end_header_id|>
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=50, temperature=0.7, do_sample=True, top_k=50, top_p=0.95, repetition_penalty=1.2)
        query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the user's query
        query = query.split("<|start_header_id|>user<|end_header_id|>")[-1]
        query = query.split("<|eot_id|>")[0]
        return query.strip()

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
            self.create_user_agents_llm(num_users)
        else:
            self.create_user_agents(num_users)

    def create_agents(self, num_agents):
        for _ in range(num_agents):
            agent_type = random.choice(self.service_agent_types)
            a = InfoSeekingAgent(self.next_agent_id, self.knowledge_base, agent_type, self.alpha, use_llm=self.use_llm, model_path=self.model_path)
            self.agents.append(a)
            self.next_agent_id += 1

    def create_user_agents(self, num_users):
        for _ in range(num_users):
            user_type = random.choice(self.user_agent_types)
            patience_level = random.randint(1, 5)
            expertise_level = random.randint(1, 5)
            u = UserAgent(self.next_agent_id, user_type, patience_level, expertise_level, self.knowledge_base)
            self.next_agent_id += 1

    def create_user_agents_llm(self, num_users):
        for _ in range(num_users):
            user_type = random.choice(self.user_agent_types)
            patience_level = random.randint(1, 5)
            expertise_level = random.randint(1, 5)
            u = UserLLMAgent(self.next_agent_id, user_type, patience_level, expertise_level, self.knowledge_base, self.model_path)
            self.next_agent_id += 1

    def add_agent(self, agent_type):
        a = InfoSeekingAgent(self.next_agent_id, self.knowledge_base, agent_type, self.alpha, use_llm=self.use_llm, model_path=self.model_path)
        self.agents.append(a)
        self.next_agent_id += 1

    def remove_agent(self, agent_id):
        self.agents = [a for a in self.agents if a.unique_id != agent_id]

    def step(self):
        if self.use_llm:
            user_agents = []
            for _ in range(self.num_users):
                user_type = random.choice(self.user_agent_types)
                patience_level = random.randint(1, 5)
                expertise_level = random.randint(1, 5)
                u = UserLLMAgent(self.next_agent_id, user_type, patience_level, expertise_level, self.knowledge_base, self.model_path)
                user_agents.append(u)
                self.next_agent_id += 1
        else:
            user_agents = []
            for _ in range(self.num_users):
                user_type = random.choice(self.user_agent_types)
                patience_level = random.randint(1, 5)
                expertise_level = random.randint(1, 5)
                u = UserAgent(self.next_agent_id, user_type, patience_level, expertise_level, self.knowledge_base)
                user_agents.append(u)
                self.next_agent_id += 1

        for user_agent in user_agents:
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
        #print(agent_data) #Commented this out so that agent-user interaction is more visible
