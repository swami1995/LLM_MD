from collections import defaultdict
import time
import random
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from trust_market.trust_market import TrustMarket
import numpy as np

class TrustMarketSystem:
    """
    Main system that orchestrates the trust market, information sources, and evaluation cycles.
    """
    
    def __init__(self, config=None):
        """
        Initialize the trust market system.
        
        Parameters:
        - config: Configuration parameters
        """
        self.config = config or {}
        
        # Set default configuration if not provided
        if 'dimensions' not in self.config:
            self.config['dimensions'] = [
                "Factual_Correctness", "Process_Reliability", "Value_Alignment",
                "Communication_Quality", "Problem_Resolution", "Safety_Security",
                "Transparency", "Adaptability", "Trust_Calibration", "Manipulation_Resistance"
            ]
        
        if 'dimension_weights' not in self.config:
            self.config['dimension_weights'] = {dim: 1.0 for dim in self.config['dimensions']}
            
        # Primary trust sources/sinks configuration
        self.config['primary_sources'] = self.config.get('primary_sources', ['user_feedback', 'regulator'])
        self.config['primary_source_weight'] = self.config.get('primary_source_weight', 2.0)
        self.config['secondary_source_weight'] = self.config.get('secondary_source_weight', 1.0)
            
        # Initialize the trust market
        self.trust_market = TrustMarket(self.config)
        
        # Initialize information sources
        self.information_sources = {}
        
        # Track conversation histories
        self.conversation_histories = {}
        
        # Track user feedback for processing
        self.user_feedback = defaultdict(list)
        
        # Track source evaluation frequencies
        self.source_evaluation_frequency = {}  # source_id -> frequency in rounds
        
        # Track when sources were last evaluated
        self.source_last_evaluated = defaultdict(int)  # source_id -> last evaluation round
        
        # Current evaluation round
        self.evaluation_round = 0
    
    def add_information_source(self, source, is_primary=False, evaluation_frequency=1, initial_influence=None):
        """
        Add an information source to the system.
        
        Parameters:
        - source: The information source to add
        - is_primary: Whether this is a primary trust source
        - evaluation_frequency: How often (in rounds) this source should make investment decisions
        """
        source.market = self.trust_market
        self.information_sources[source.source_id] = source
        
        # Set evaluation frequency
        self.source_evaluation_frequency[source.source_id] = evaluation_frequency
        
        # Set initial influence capacity in the market
        if initial_influence is None:
            initial_influence = {dim: 100.0 for dim in source.expertise_dimensions}
        self.trust_market.add_information_source(
            source.source_id, source.source_type, initial_influence, is_primary
        )
        
    
    def register_user_representative(self, representative, evaluation_frequency=1, iniitial_influence=None):
        """
        Register a user representative in the system.
        
        Parameters:
        - representative: The UserRepresentative object
        - evaluation_frequency: How often to evaluate
        """
        self.add_information_source(representative, False, evaluation_frequency, iniitial_influence)
    
    def register_domain_expert(self, expert, evaluation_frequency=3, iniitial_influence=None):
        """
        Register a domain expert in the system.
        
        Parameters:
        - expert: The DomainExpert object
        - evaluation_frequency: How often to evaluate
        """
        self.add_information_source(expert, False, evaluation_frequency, iniitial_influence)
    
    def register_auditor(self, auditor, evaluation_frequency=5, iniitial_influence=None):
        """
        Register an auditor in the system.
        
        Parameters:
        - auditor: The Auditor object
        - evaluation_frequency: How often to evaluate
        """
        self.add_information_source(auditor, False, evaluation_frequency, iniitial_influence)
    
    def register_red_teamer(self, red_teamer, evaluation_frequency=10, iniitial_influence=None):
        """
        Register a red teamer in the system.
        
        Parameters:
        - red_teamer: The RedTeamer object
        - evaluation_frequency: How often to evaluate
        """
        self.add_information_source(red_teamer, False, evaluation_frequency, iniitial_influence)
    
    def register_regulator(self, regulator, evaluation_frequency=20, iniitial_influence=None):
        """
        Register a regulator in the system.
        
        Parameters:
        - regulator: The Regulator object
        - evaluation_frequency: How often to evaluate
        """
        self.add_information_source(regulator, True, evaluation_frequency, iniitial_influence)
    
    def record_conversation(self, conversation_id, user_id, agent_id, history):
        """
        Record a conversation for evaluation by information sources.
        
        Parameters:
        - conversation_id: Unique conversation identifier
        - user_id: User participating in the conversation
        - agent_id: Agent participating in the conversation
        - history: Conversation history
        """
        self.conversation_histories[conversation_id] = {
            "user_id": user_id,
            "agent_id": agent_id,
            "history": history,
            "timestamp": time.time(),
            "evaluated": False
        }
        
        # Have user representatives analyze this conversation
        for source_id, source in self.information_sources.items():
            if hasattr(source, 'add_conversation'):
                source.add_conversation(history, user_id, agent_id)
    
    def process_user_feedback(self, user_id, agent_id, ratings, user_type=None):
        """
        Process direct user feedback.
        
        Parameters:
        - user_id: User providing feedback
        - agent_id: Agent being rated
        - ratings: Dict mapping dimensions to ratings
        - user_type: Optional user type for segmentation
        """
        # Store the feedback
        self.user_feedback[user_id].append({
            "agent_id": agent_id,
            "ratings": ratings,
            "user_type": user_type,
            "timestamp": time.time(),
            "processed": False,
            "evaluation_round": self.evaluation_round
        })
        
        # Apply the feedback to trust scores
        self.trust_market.record_user_feedback(user_id, agent_id, ratings)
        
        # Mark this feedback as processed
        self.user_feedback[user_id][-1]["processed"] = True
        
        # Update agent performance based on ratings
        self.trust_market.update_agent_performance(agent_id, ratings)
    
    def run_evaluation_round(self):
        """Run a single evaluation round."""
        self.evaluation_round += 1
        
        # Increment market round counter
        self.trust_market.increment_evaluation_round()
        
        # Process user ratings from this round
        self._process_user_ratings()
        
        # Run source evaluations based on their frequency
        self._run_source_evaluations()
        
        # # Calculate and distribute rewards based on agent performance
        # self._distribute_source_rewards()
    
    def _process_user_ratings(self):
        """Process all user ratings collected in this round."""
        # Perform a round of user - agent conversations and obtain ratings
        # Process user feedback by calling process_user_feedback
        
    
    def _run_source_evaluations(self):
        """Run evaluations for sources due in this round."""

        arr = (self.evaluation_round - np.array(list(self.source_last_evaluated.values()))) == np.array(list(self.source_evaluation_frequency.values()))
        if arr.sum() == 0:
            return
        for source_id, source in self.information_sources.items():
            frequency = self.source_evaluation_frequency.get(source_id, 1)
            last_evaluated = self.source_last_evaluated[source_id]
            
            # Check if it's time to evaluate
            if self.evaluation_round - last_evaluated >= frequency:
                # Run evaluation
                try:
                    # # Get recent conversations to evaluate
                    # recent_conversations = self._get_recent_conversations(source_id)
                    
                    # # Evaluate agents
                    # evaluations = self._evaluate_source_agents(source, recent_conversations)
                    
                    # Decide investments
                    investments = source.decide_investments(evaluation_round=self.evaluation_round)
                    
                    if investments:
                        # Process the investments in the market
                        self.trust_market.process_investments(source_id, investments)
                    
                    # Record that this source was evaluated
                    self.source_last_evaluated[source_id] = self.evaluation_round
                    
                except Exception as e:
                    print(f"Error evaluating source {source_id}: {str(e)}")
    
    def _get_recent_conversations(self, source_id):
        """Get recent conversations relevant to a source."""
        # This can be customized based on source type
        source = self.information_sources.get(source_id)
        
        # For user representatives, they track their own conversations
        if hasattr(source, 'observed_conversations'):
            return source.observed_conversations
        
        # For other sources, return all recent conversations
        recent = []
        for conv_id, conv in self.conversation_histories.items():
            if not conv["evaluated"]:
                recent.append(conv)
                conv["evaluated"] = True
        
        return recent
    
    def _evaluate_source_agents(self, source, conversations):
        """Have a source evaluate agents based on conversations."""
        # Skip if no conversations
        if not conversations:
            return {}
        
        # Group conversations by agent
        agent_conversations = defaultdict(list)
        for conv in conversations:
            if isinstance(conv, dict) and "agent_id" in conv and "history" in conv:
                agent_conversations[conv["agent_id"]].append(conv["history"])
            elif hasattr(conv, "agent_id") and hasattr(conv, "conversation"):
                agent_conversations[conv.agent_id].append(conv.conversation)
        
        # Evaluate each agent
        evaluations = {}
        for agent_id, convs in agent_conversations.items():
            # Some sources evaluate individual conversations, others evaluate batches
            if hasattr(source, 'evaluate_agent_batch'):
                ratings = source.evaluate_agent_batch(agent_id, convs)
            else:
                # Evaluate each conversation
                combined_ratings = defaultdict(list)
                for conversation in convs:
                    ratings = source.evaluate_agent(agent_id, conversation)
                    if ratings:
                        for dimension, (rating, confidence) in ratings.items():
                            combined_ratings[dimension].append((rating, confidence))
                
                # Average the ratings
                ratings = {}
                for dimension, values in combined_ratings.items():
                    if values:
                        avg_rating = sum(r for r, _ in values) / len(values)
                        avg_confidence = sum(c for _, c in values) / len(values)
                        ratings[dimension] = (avg_rating, avg_confidence)
            
            if ratings:
                evaluations[agent_id] = ratings
                source.record_evaluation(agent_id, ratings)
                
                # Extract just the ratings (without confidence) for performance tracking
                performance_scores = {dim: rating for dim, (rating, _) in ratings.items()}
                self.trust_market.update_agent_performance(agent_id, performance_scores)
        
        return evaluations
    
    def _distribute_source_rewards(self):
        """Calculate and distribute rewards to sources based on agent performance."""
        # Only distribute rewards every N rounds
        reward_frequency = self.config.get('reward_distribution_frequency', 5)
        if self.evaluation_round % reward_frequency != 0:
            return
            
        # Calculate rewards based on agent performance
        rewards = self.trust_market.calculate_source_rewards()
        
        # Update source performance based on rewards
        for source_id, dimension_rewards in rewards.items():
            # Normalize rewards to 0-1 scale for performance tracking
            performance_scores = {}
            for dimension, reward in dimension_rewards.items():
                # Convert reward to performance score (0-1)
                # Assuming rewards are roughly in -100 to 100 range
                normalized_score = min(1.0, max(0.0, (reward + 100) / 200))
                performance_scores[dimension] = normalized_score
            
            # Update source performance
            self.trust_market.update_source_performance(source_id, performance_scores)
    
    def run_evaluation_rounds(self, num_rounds):
        """
        Run multiple evaluation rounds.
        
        Parameters:
        - num_rounds: Number of rounds to run
        """
        for _ in range(num_rounds):
            self.run_evaluation_round()
    
    def get_agent_trust_scores(self):
        """Get the current trust scores for all agents."""
        agent_scores = {}
        
        # Find all agent IDs that have trust scores
        for agent_id in self.trust_market.agent_trust_scores:
            agent_scores[agent_id] = self.trust_market.get_agent_trust(agent_id)
        
        return agent_scores
    
    def get_agent_permissions(self, agent_id):
        """
        Determine what permissions an agent has based on trust scores.
        
        Parameters:
        - agent_id: Agent to check
        
        Returns:
        - Dict of permission types and levels
        """
        return self.trust_market.get_agent_permissions(agent_id)
    
    def get_source_investments(self, source_id):
        """
        Get all current investments made by a source.
        
        Parameters:
        - source_id: The source to query
        
        Returns:
        - List of investment records
        """
        return self.trust_market.get_source_endorsements(source_id)
    
    def summarize_market_state(self):
        """Get a summary of the current market state."""
        agent_scores = self.get_agent_trust_scores()
        
        # Calculate average scores by dimension
        dimension_averages = defaultdict(list)
        for agent_id, scores in agent_scores.items():
            for dimension, score in scores.items():
                dimension_averages[dimension].append(score)
        
        avg_scores = {}
        for dimension, scores in dimension_averages.items():
            if scores:
                avg_scores[dimension] = sum(scores) / len(scores)
            else:
                avg_scores[dimension] = None
        
        # Information source influence
        source_influence = {}
        for source_id, source in self.information_sources.items():
            source_influence[source_id] = {
                "type": source.source_type,
                "capacity": self.trust_market.source_influence_capacity.get(source_id, {}),
                "allocated": self.trust_market.allocated_influence.get(source_id, {})
            }
        
        # Count investments per agent/dimension
        investment_counts = defaultdict(lambda: defaultdict(int))
        for source_id, agent_investments in self.trust_market.source_investments.items():
            for agent_id, dimensions in agent_investments.items():
                for dimension, amount in dimensions.items():
                    if amount > 0:
                        investment_counts[agent_id][dimension] += 1
        
        return {
            "evaluation_round": self.evaluation_round,
            "dimension_averages": avg_scores,
            "source_influence": source_influence,
            "num_information_sources": len(self.information_sources),
            "num_agents": len(agent_scores),
            "investment_counts": dict(investment_counts)
        }
    
    def visualize_trust_scores(self, agents=None, dimensions=None, 
                             start_round=None, end_round=None):
        """
        Visualize trust scores over time for specified agents and dimensions.
        
        Parameters:
        - agents: List of agent_ids to include (default: all)
        - dimensions: List of dimensions to include (default: all)
        - start_round: Starting evaluation round (default: all)
        - end_round: Ending evaluation round (default: all)
        """
        return self.trust_market.visualize_trust_scores(
            agents, dimensions, start_round, end_round
        )
    
    def visualize_source_performance(self, sources=None, dimensions=None,
                                  start_round=None, end_round=None):
        """
        Visualize source performance over time.
        
        Parameters:
        - sources: List of source_ids to include (default: all)
        - dimensions: List of dimensions to include (default: all)
        - start_round: Starting evaluation round (default: all)
        - end_round: Ending evaluation round (default: all)
        """
        return self.trust_market.visualize_source_performance(
            sources, dimensions, start_round, end_round
        )