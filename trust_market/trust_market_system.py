from collections import defaultdict
import time
from trust_market.trust_market import TrustMarket
from trust_market.info_sources import UserRepresentative, DomainExpert, Auditor, RedTeamer, Regulator


class TrustMarketSystem:
    """
    Main system that integrates the trust market with all information sources.
    Redesigned with clearer source/sink model and optimized updates.
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
        
        if 'default_decay_rate' not in self.config:
            self.config['default_decay_rate'] = 1.0  # Default to no decay
            
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
        
        # Current evaluation round
        self.evaluation_round = 0
    
    def add_information_source(self, source, is_primary=False):
        """
        Add an information source to the system.
        
        Parameters:
        - source: The information source to add
        - is_primary: Whether this is a primary trust source
        """
        source.market = self.trust_market
        self.information_sources[source.source_id] = source
        
        # Set initial influence capacity in the market
        initial_influence = {dim: 100.0 for dim in source.expertise_dimensions}
        self.trust_market.add_information_source(
            source.source_id, source.source_type, initial_influence, is_primary
        )
    
    def setup_user_representatives(self, user_profiles, num_representatives=3):
        """
        Set up user representatives based on user profiles.
        
        Parameters:
        - user_profiles: List of user profiles
        - num_representatives: Number of representatives to create
        """
        # Cluster users
        representatives = create_user_representatives(user_profiles, num_representatives)
        
        # Create and add representatives
        for rep_data in representatives:
            rep = UserRepresentative(
                source_id=f"user_rep_{rep_data['segment']}",
                user_segment=rep_data['segment'],
                representative_profile=rep_data['profile'],
                market=self.trust_market
            )
            
            # Add users to this representative
            for user_id in rep_data['user_ids']:
                rep.add_represented_user(user_id, user_profiles[user_id])
            
            # Add to system
            self.add_information_source(rep)
            
            print(f"Added user representative for {rep_data['segment']} segment with {len(rep_data['user_ids'])} users")
    
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
        for source in self.information_sources.values():
            if isinstance(source, UserRepresentative):
                source.analyze_conversation(history, user_id, agent_id)
    
    def process_user_feedback(self, user_id, agent_id, ratings, user_type=None):
        """
        Process direct user feedback as a primary trust source.
        
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
            "processed": False
        })
        
        # Process as primary trust source
        for dimension, rating in ratings.items():
            if dimension in self.trust_market.dimensions:
                # Record as primary feedback (with high confidence)
                self.trust_market.record_primary_feedback(
                    "user_feedback", agent_id, dimension, rating, confidence=0.9
                )
    
    def increment_evaluation_round(self):
        """Increment the evaluation round counter and trigger market updates."""
        self.evaluation_round += 1
        
        # Increment market round counter (ages endorsements)
        self.trust_market.increment_evaluation_round()
        
        # Run evaluations based on the current evaluation round
        self._trigger_evaluations()
        self._trigger_investments()
    
    def _trigger_evaluations(self):
        """Trigger information sources to evaluate agents."""
        # Get recent conversations not yet evaluated
        recent_conversations = {
            conv_id: conv for conv_id, conv in self.conversation_histories.items()
            if not conv["evaluated"]
        }
        
        if recent_conversations:
            # Mark as evaluated
            for conv_id in recent_conversations:
                self.conversation_histories[conv_id]["evaluated"] = True
            
            # Group by agent for evaluation
            agent_conversations = defaultdict(list)
            for conv in recent_conversations.values():
                agent_conversations[conv["agent_id"]].append(conv["history"])
            
            # Have each information source evaluate agents
            for source_id, source in self.information_sources.items():
                # Different evaluation approaches for different source types
                if isinstance(source, DomainExpert):
                    self._trigger_expert_evaluations(source, agent_conversations)
                elif isinstance(source, Auditor):
                    self._trigger_auditor_evaluations(source, agent_conversations)
                elif isinstance(source, RedTeamer):
                    # Red teamers have their own evaluation cycle
                    pass
                elif isinstance(source, Regulator):
                    # Regulators perform scheduled audits
                    if isinstance(source, Regulator):
                        source.check_scheduled_audits()
    
    def _trigger_expert_evaluations(self, expert, agent_conversations):
        """Trigger evaluations from a domain expert."""
        for agent_id, conversations in agent_conversations.items():
            # Evaluate each conversation
            for conversation in conversations:
                ratings = expert.evaluate_agent(agent_id, conversation)
                if ratings:
                    expert.record_evaluation(agent_id, ratings)
    
    def _trigger_auditor_evaluations(self, auditor, agent_conversations):
        """Trigger evaluations from an auditor."""
        for agent_id, conversations in agent_conversations.items():
            # Perform audit if needed
            ratings = auditor.evaluate_agent(agent_id, conversations)
            if ratings:
                auditor.record_evaluation(agent_id, ratings)
    
    def _trigger_investments(self):
        """Trigger all information sources to make investment decisions."""
        # Have each information source decide on investments
        for source_id, source in self.information_sources.items():
            investment_decisions = source.decide_investments(self.evaluation_round)
            
            # Execute the investments
            for agent_id, dimension, amount, confidence in investment_decisions:
                success, endorsement_id = self.trust_market.create_endorsement(
                    source_id, agent_id, dimension, amount, confidence
                )
                
                if not success:
                    print(f"Investment failed: {source_id} -> {agent_id} ({dimension}: {amount})")
    
    def get_agent_trust_scores(self):
        """Get the current trust scores for all agents."""
        agent_scores = {}
        
        # Get all agent IDs from trust market
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
        
        # Track endorsements per agent/dimension
        endorsement_counts = defaultdict(lambda: defaultdict(int))
        for endorsement in self.trust_market.endorsements:
            if endorsement.is_active:
                endorsement_counts[endorsement.agent_id][endorsement.dimension] += 1
        
        return {
            "evaluation_round": self.evaluation_round,
            "dimension_averages": avg_scores,
            "source_influence": source_influence,
            "num_information_sources": len(self.information_sources),
            "num_agents": len(agent_scores),
            "endorsement_counts": dict(endorsement_counts)
        }
