from mesa import Agent
from mesa import Model
from mesa.time import RandomActivation
import random

# TODO 2: Implement the InfoSeekingAgent class

class InfoSeekingAgent(Agent):
    def __init__(self, unique_id, model, knowledge_base, agent_type, rate_limit=5):
        super().__init__(unique_id, model)
        self.knowledge_base = knowledge_base
        self.trust_scores = {"Accuracy": 0.5}  # Initial trust score
        self.agent_type = agent_type
        self.rate_limit = rate_limit # Placeholder for future use

    def step(self):
        # Basic version: Agents don't do much in the step method independently.
        # They respond when called upon by UserAgents.
        pass

    def answer_query(self, query):
        if query in self.knowledge_base:
            return self.knowledge_base[query]
        else:
            return "I don't have information about that. Please try rephrasing your query or contact our human support team."

    def update_trust_score(self, dimension, rating):
        self.trust_scores[dimension] = (1 - self.model.alpha) * self.trust_scores[dimension] + self.model.alpha * rating


# TODO 3: Implement the UserAgent class

class UserAgent(Agent):
    def __init__(self, unique_id, model, user_type, patience_level, expertise_level):
        super().__init__(unique_id, model)
        self.user_type = user_type
        self.patience_level = patience_level
        self.expertise_level = expertise_level

    def step(self):
        self.submit_query(self.model.schedule.agents)

    def submit_query(self, agents):
        # Filter out UserAgents
        info_seeking_agents = [a for a in agents if isinstance(a, InfoSeekingAgent)]

        if not info_seeking_agents:
            return  # No agents to interact with

        # Select a query
        query = random.choice(list(self.model.knowledge_base.keys()))

        # Select an agent
        agent = random.choice(info_seeking_agents)

        # Get a response
        response = agent.answer_query(query)

        # Rate the response
        self.rate_response(response, agent, query)

    def rate_response(self, response, agent, query):
        if self.user_type == "Novice":
            rating = 1 if random.random() < 0.8 else 0  # 80% chance of positive rating
        elif self.user_type == "Expert":
            rating = 1 if response == self.model.knowledge_base[query] else 0  # Accurate rating based on knowledge
        elif self.user_type == "Skeptical":
            rating = 1 if response == self.model.knowledge_base[query] and random.random() < 0.9 else 0  # 90% chance of positive rating if accurate
        else:
            rating = 1 if response == self.model.knowledge_base[query] else 0  # Default: accurate rating

        # Update agent's trust score
        agent.update_trust_score("Accuracy", rating)


# TODO 4: Implement the CustomerSupportModel class

class CustomerSupportModel(Model):
    def __init__(self, num_users, num_agents, knowledge_base, alpha=0.1):
        self.num_users = num_users
        self.knowledge_base = knowledge_base
        self.alpha = alpha
        self.schedule = RandomActivation(self)
        self.running = True
        self.next_agent_id = 0  # Initialize the next agent ID counter

        self.user_agent_types = ["Novice", "Expert", "Skeptical"]
        self.agent_types = ["Basic", "Profit-Maximizing", "Lazy"]
        self.num_agents = num_agents

        self.create_user_agents(num_users)
        self.create_agents("Basic", num_agents) 
        # during initialization, only the basic agent type is created.

    def create_agents(self, agent_type, num_agents):
        for i in range(num_agents):
            a = InfoSeekingAgent(self.next_agent_id, self, self.knowledge_base, agent_type)
            self.schedule.add(a)
            self.next_agent_id += 1

    def create_user_agents(self, num_users):
        for i in range(num_users):
            user_type = random.choice(self.user_agent_types)
            patience_level = random.randint(1, 5)
            expertise_level = random.randint(1, 5)
            u = UserAgent(self.next_agent_id, self, user_type, patience_level, expertise_level)
            self.schedule.add(u)
            self.next_agent_id += 1

    def add_agent(self, agent_type):
        a = InfoSeekingAgent(self.next_agent_id, self, self.knowledge_base, agent_type)
        self.schedule.add(a)
        self.next_agent_id += 1

    def remove_agent(self, agent_id):
        for agent in self.schedule.agents:
            if agent.unique_id == agent_id:
                self.schedule.remove(agent)
                break

    def step(self):
        self.schedule.step()
        self.collect_data()  # Collect data at the end of each step

    def collect_data(self):
        agent_data = []
        for agent in self.schedule.agents:
            if isinstance(agent, InfoSeekingAgent):
                agent_data.append({
                    "agent_id": agent.unique_id,
                    "trust_score": agent.trust_scores["Accuracy"],
                    "agent_type": agent.agent_type
                })
        # Store or process the collected data (e.g., save to a file, print, etc.)
        print(agent_data)  # For now, just print the data