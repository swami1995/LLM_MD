import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class TrustMarket:
    """
    A market system for trust as a tradable asset based on investments.
    Focuses on tracking trust scores, source investments, and market dynamics.
    """
    def __init__(self, config: Dict[str, Any]):
        """Initialize the trust market."""
        self.config = config
        self.dimensions = config.get('dimensions', [])
        self.dimension_weights = config.get('dimension_weights', {dim: 1.0 for dim in self.dimensions})
        
        # Trust scores for all agents across all dimensions
        self.agent_trust_scores = defaultdict(lambda: {dim: 0.5 for dim in self.dimensions})
        
        # Track investments by sources in agents (by dimension)
        self.source_investments = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        # Track influence/stake capacity for all sources
        self.source_influence_capacity = defaultdict(lambda: {dim: 100.0 for dim in self.dimensions})
        self.source_available_capacity = defaultdict(lambda: {dim: 100.0 for dim in self.dimensions})
        
        # Track allocated influence by sources
        self.allocated_influence = defaultdict(lambda: {dim: 0.0 for dim in self.dimensions})
        
        # Primary trust sources get higher weight
        self.primary_sources = set(config.get('primary_sources', ['user_feedback', 'regulator']))
        self.primary_source_weight = config.get('primary_source_weight', 2.0)
        self.secondary_source_weight = config.get('secondary_source_weight', 1.0)
        
        # Performance tracking
        self.source_performance = defaultdict(lambda: defaultdict(list))
        self.agent_performance = defaultdict(lambda: defaultdict(list))
        
        # Temporal trust database - track scores and investments over time
        self.temporal_db = {
            'trust_scores': [],
            'investments': [],
            'source_performance': [],
            'agent_performance': []
        }
        
        # Evaluation round counter
        self.evaluation_round = 0
        
        # Multi-dimensional trust correlation (optional)
        self.use_dimension_correlations = config.get('use_dimension_correlations', False)
        if self.use_dimension_correlations:
            self.initialize_dimension_correlations()
    
    def initialize_dimension_correlations(self):
        """Initialize the correlation structure between trust dimensions."""
        # Default correlations (can be overridden in config)
        default_correlations = {
            # Dimensions that tend to move together
            ('Factual_Correctness', 'Transparency'): 0.6,
            ('Communication_Quality', 'Problem_Resolution'): 0.5,
            ('Value_Alignment', 'Safety_Security'): 0.7,
            ('Trust_Calibration', 'Transparency'): 0.6,
            
            # Dimensions that may have some correlation
            ('Factual_Correctness', 'Trust_Calibration'): 0.4,
            ('Process_Reliability', 'Safety_Security'): 0.5,
            ('Adaptability', 'Communication_Quality'): 0.3,
            
            # Dimensions with potential trade-offs
            ('Adaptability', 'Process_Reliability'): -0.2,
        }
        
        # Use custom correlations from config if provided
        self.dimension_correlations = self.config.get('dimension_correlations', default_correlations)
        
        # Build full correlation matrix
        self.correlation_matrix = {}
        for dim1 in self.dimensions:
            for dim2 in self.dimensions:
                if dim1 == dim2:
                    self.correlation_matrix[(dim1, dim2)] = 1.0
                else:
                    # Look up correlation or default to 0
                    corr = self.dimension_correlations.get((dim1, dim2), 
                           self.dimension_correlations.get((dim2, dim1), 0.0))
                    self.correlation_matrix[(dim1, dim2)] = corr
    
    def add_information_source(self, source_id: str, source_type: str, 
                               initial_influence: Dict[str, float],
                               is_primary: bool = False) -> None:
        """
        Add a new information source to the market with initial influence capacity.
        
        Parameters:
        - source_id: Unique identifier for the source
        - source_type: Type of information source (e.g., 'expert', 'regulator')
        - initial_influence: Dict mapping dimensions to initial influence capacity
        - is_primary: Whether this is a primary trust source
        """
        # Only allocate influence for valid dimensions
        for dimension in initial_influence:
            if dimension in self.dimensions:
                self.source_influence_capacity[source_id][dimension] = initial_influence[dimension]
                self.source_available_capacity[source_id][dimension] = initial_influence[dimension]
        
        # Add to primary sources if needed
        if is_primary:
            self.primary_sources.add(source_id)
    
    def process_investments(self, source_id: str, investments: List[Tuple]) -> None:
        """
        Process a batch of investment/divestment decisions from a source.
        
        Parameters:
        - source_id: The source making the investment decisions
        - investments: List of (agent_id, dimension, amount, confidence) tuples
                       where amount can be positive (investment) or negative (divestment)
        """
        # Track changes that will need trust score recalculation
        affected_agents = set()
        
        # Process each investment/divestment
        old_agent_trust_scores = {dimension: {} for dimension in self.dimensions}
        for dimension in self.dimensions:
            if dimension not in self.source_influence_capacity[source_id]:
                continue
            old_agent_trust_scores[dimension] = {agent_id: sum([self.source_investments[s][agent_id][dimension] for s in self.source_investments]) for agent_id in self.source_investments[source_id]}
        for agent_id, dimension, amount, confidence in investments:
            if dimension not in self.dimensions:
                continue
                
            # Handle divestment (negative amount)
            if amount < 0:
                current_investment = self.source_investments[source_id][agent_id].get(dimension, 0.0)
                
                # Can't divest more than current investment
                divestment = min(abs(amount), current_investment)
                
                if divestment > 0:
                    # Update investment records
                    old_agent_id_trust_score = self.source_investments[source_id][agent_id][dimension] 
                    self.source_investments[source_id][agent_id][dimension] -= divestment
                    new_agent_trust_scores = sum([self.source_investments[s][agent_id][dimension] for s in self.source_investments])
                    for other_agent_id in self.source_investments[source_id]:
                        if self.source_investments[source_id][other_agent_id][dimension] > 0:
                            # Redistribute divested influence to other agents
                            self.source_investments[source_id][other_agent_id][dimension] = (self.source_investments[source_id][other_agent_id][dimension]) * new_agent_trust_scores / old_agent_trust_scores[dimension][agent_id]
                    new_agent_id_trust_score = self.source_investments[source_id][agent_id][dimension]
                    eventual_reduction_amount = (old_agent_id_trust_score - new_agent_id_trust_score)
                    self.allocated_influence[source_id][dimension] -= eventual_reduction_amount
                    self.source_available_capacity[source_id][dimension] += divestment
                    
                    # Add to affected agents for recalculation
                    affected_agents.add(agent_id)
                    
                    # Record in temporal DB
                    self.temporal_db['investments'].append({
                        'evaluation_round': self.evaluation_round,
                        'timestamp': time.time(),
                        'source_id': source_id,
                        'agent_id': agent_id,
                        'dimension': dimension,
                        'amount': -divestment,
                        'eventual_reduction_amount': eventual_reduction_amount,
                        'confidence': confidence,
                        'type': 'divestment'
                    })
            
            # Handle investment (positive amount)
            else:
                # Check if source has enough available influence
                available = self.source_available_capacity[source_id][dimension]
                
                if amount <= available:
                    # Update investment records
                    self.source_investments[source_id][agent_id][dimension] += amount
                    new_agent_trust_score = sum([self.source_investments[s][agent_id][dimension].get(dimension, 0.0) for s in self.source_investments])

                    for other_agent_id in self.source_investments[source_id]:
                        if self.source_investments[source_id][other_agent_id][dimension] > 0:
                            # Redistribute invested influence to other agents
                            self.source_investments[source_id][other_agent_id][dimension] = (self.source_investments[source_id][other_agent_id][dimension]) * new_agent_trust_score / old_agent_trust_scores[dimension][agent_id]

                    new_agent_id_trust_score = self.source_investments[source_id][agent_id][dimension]
                    eventual_allocation_amount = (new_agent_id_trust_score - old_agent_id_trust_score)
                    self.allocated_influence[source_id][dimension] += eventual_allocation_amount
                    self.source_available_capacity[source_id][dimension] -= amount
                    
                    # Add to affected agents for recalculation
                    affected_agents.add(agent_id)
                    
                    # Record in temporal DB
                    self.temporal_db['investments'].append({
                        'evaluation_round': self.evaluation_round,
                        'timestamp': time.time(),
                        'source_id': source_id,
                        'agent_id': agent_id,
                        'dimension': dimension,
                        'amount': amount,
                        'confidence': confidence,
                        'type': 'investment'
                    })
            self.source_influence_capacity[source_id][dimension] = self.allocated_influence[source_id][dimension] + self.source_available_capacity[source_id][dimension]
        
        # Recalculate trust scores for affected agents
        # Weighted aggregation of trust scores across sources
        for agent_id in affected_agents:
            for dimension in self.dimensions:
                self.agent_trust_scores[agent_id][dimension] = sum([self.source_investments[source_id][agent_id].get(dimension, 0.0) for source_id in self.source_investments]) 


        ### These probably happen periodically at each time step increments
        # temporal decay of trust scores (remove the laptop time based decay).
        # update trust scores based on dimension correlations
        # decay_rate = self.config.get('trust_decay_rate', 0.99)
        #         self.agent_trust_scores[agent_id][dimension] *= decay_rate
        #         for source_id in self.source_investments:
        #             self.source_investments[source_id][agent_id][dimension] *= decay_rate
        # for source_id in self.source_investments:
        #     for dimension in self.dimensions:
        #         self.source_available_capacity[source_id][dimension] *= decay_rate 
        #         self.allocated_influence[source_id][dimension] *= decay_rate
        # Recalculate trust scores for this agent
        # self.recalculate_agent_trust(agent_id)
    
    def record_user_feedback(self, user_id: str, agent_id: int, 
                            ratings: Dict, confidence: float = 0.9) -> None:
        """
        Record direct user feedback and apply it to trust scores.
        
        Parameters:
        - user_id: ID of the user providing feedback
        - agent_id: Agent being rated
        - ratings: Dict mapping dimensions to ratings (0-1)
        - confidence: Confidence in these ratings
        """
        # Map batch of ratings to scores : # 0 = negative, 0.5 = neutral, 1 = positive
        scores = {dimension: ratings.get(dimension, 0.5)*2-1 for dimension in self.dimensions}
        
        for dimension, score in scores.items():
            old_agent_score = self.agent_trust_scores[agent_id][dimension]
            self.agent_trust_scores[agent_id][dimension] = max(0.0, self.agent_trust_scores[agent_id][dimension] + score) # * confidence
            agent_score = self.agent_trust_scores[agent_id][dimension]

            # Record user feedback in temporal DB
            self.temporal_db['trust_scores'].append({
                'evaluation_round': self.evaluation_round,
                'timestamp': time.time(),
                'agent_id': agent_id,
                'dimension': dimension,
                'old_score': old_agent_score,
                'new_score': agent_score,
                'change_source': 'user_feedback',
                'source_id': user_id
            })
        
            if old_agent_score == 0.0:
                continue
            for source_id in self.source_investments:
                if agent_id in self.source_investments[source_id]:
                    # Update investment records based on user feedback
                    old_source_investment = self.source_investments[source_id][agent_id].get(dimension, 0.0)
                    self.source_investments[source_id][agent_id][dimension] = self.source_investments[source_id][agent_id].get(dimension, 0.0) * (agent_score/old_agent_score)
                    self.allocated_influence[source_id][dimension] += (self.source_investments[source_id][agent_id][dimension] - old_source_investment)
                
        # Apply dimension correlations if enabled
        # if self.use_dimension_correlations and affected_dimensions:
        #     self.apply_dimension_correlations(agent_id, affected_dimensions)
    
    def recalculate_agent_trust(self, agent_id: int, specific_dimension: str = None) -> None:
        """
        Recalculate trust scores for an agent based on current investments.
        
        Parameters:
        - agent_id: The agent to recalculate
        - specific_dimension: Optional specific dimension to recalculate (default: all)
        """
        dimensions_to_update = [specific_dimension] if specific_dimension else self.dimensions
        affected_dimensions = []
        
        for dimension in dimensions_to_update:
            if dimension not in self.dimensions:
                continue
                
            # Get current score before update
            old_score = self.agent_trust_scores[agent_id][dimension]
            
            # Get all investments for this agent and dimension
            total_weighted_investment = 0.0
            total_investment = 0.0
            
            for source_id in self.source_investments:
                if agent_id in self.source_investments[source_id]:
                    investment = self.source_investments[source_id][agent_id].get(dimension, 0.0)
                    
                    if investment > 0:
                        # Apply primary vs secondary source weighting
                        weight = self.primary_source_weight if source_id in self.primary_sources else self.secondary_source_weight
                        
                        # Add to totals
                        weighted_investment = investment * weight
                        total_weighted_investment += weighted_investment
                        total_investment += investment
            
            # Calculate new score based on investments
            if total_investment > 0:
                # Normalized score based on weighted investments
                # This assumes investments themselves carry the trust valuation
                new_score = total_weighted_investment / (
                    total_investment * max(self.primary_source_weight, self.secondary_source_weight)
                )
                
                # Ensure score is in valid range
                new_score = min(1.0, max(0.0, new_score))
                
                # Apply smoothing to prevent drastic changes
                smoothing = self.config.get('trust_update_smoothing', 0.8)
                final_score = old_score * smoothing + new_score * (1 - smoothing)
                
                # Update the trust score
                self.agent_trust_scores[agent_id][dimension] = final_score
                
                # Record for dimension correlation updates
                if abs(final_score - old_score) > 0.01:  # Only count significant changes
                    affected_dimensions.append(dimension)
                    
                    # Record in temporal DB
                    self.temporal_db['trust_scores'].append({
                        'evaluation_round': self.evaluation_round,
                        'timestamp': time.time(),
                        'agent_id': agent_id,
                        'dimension': dimension,
                        'old_score': old_score,
                        'new_score': final_score,
                        'change_source': 'investment_update',
                        'source_id': None
                    })
            
            # If no investments, score decays slightly toward default
            else:
                decay_rate = self.config.get('no_investment_decay', 0.99)
                default_score = self.config.get('default_trust_score', 0.5)
                
                new_score = old_score * decay_rate + default_score * (1 - decay_rate)
                self.agent_trust_scores[agent_id][dimension] = new_score
                
                if abs(new_score - old_score) > 0.01:
                    affected_dimensions.append(dimension)
                    
                    # Record in temporal DB
                    self.temporal_db['trust_scores'].append({
                        'evaluation_round': self.evaluation_round,
                        'timestamp': time.time(),
                        'agent_id': agent_id,
                        'dimension': dimension,
                        'old_score': old_score,
                        'new_score': new_score,
                        'change_source': 'decay',
                        'source_id': None
                    })
        
        # Apply dimension correlations if enabled
        if self.use_dimension_correlations and affected_dimensions:
            self.apply_dimension_correlations(agent_id, affected_dimensions)
    
    def apply_dimension_correlations(self, agent_id: int, changed_dimensions: List[str]) -> None:
        """
        Apply correlations between dimensions to propagate trust changes.
        
        Parameters:
        - agent_id: The agent whose trust scores are being updated
        - changed_dimensions: Dimensions that have changed directly
        """
        if not self.use_dimension_correlations:
            return
            
        # Track secondary updates to prevent cascading
        secondary_updates = {}
        
        # For each changed dimension, update correlated dimensions
        for changed_dim in changed_dimensions:
            change_amount = 0
            
            # Find the latest change for this dimension
            for entry in reversed(self.temporal_db['trust_scores']):
                if (entry['agent_id'] == agent_id and 
                    entry['dimension'] == changed_dim and
                    entry['evaluation_round'] == self.evaluation_round):
                    change_amount = entry['new_score'] - entry['old_score']
                    break
            
            if abs(change_amount) < 0.01:
                continue
                
            # Apply correlations to other dimensions
            for target_dim in self.dimensions:
                if target_dim not in changed_dimensions and target_dim != changed_dim:
                    correlation = self.correlation_matrix.get((changed_dim, target_dim), 0)
                    
                    if abs(correlation) > 0.1:  # Only apply significant correlations
                        # Calculate indirect change
                        indirect_change = change_amount * correlation * self.config.get('correlation_strength', 0.3)
                        
                        # Accumulate changes for this dimension
                        if target_dim in secondary_updates:
                            secondary_updates[target_dim] += indirect_change
                        else:
                            secondary_updates[target_dim] = indirect_change
        
        # Apply accumulated secondary updates
        for dimension, change in secondary_updates.items():
            if abs(change) > 0.005:  # Minimum threshold for updates
                old_score = self.agent_trust_scores[agent_id][dimension]
                new_score = min(1.0, max(0.0, old_score + change))
                
                self.agent_trust_scores[agent_id][dimension] = new_score
                
                # Record in temporal DB
                self.temporal_db['trust_scores'].append({
                    'evaluation_round': self.evaluation_round,
                    'timestamp': time.time(),
                    'agent_id': agent_id,
                    'dimension': dimension,
                    'old_score': old_score,
                    'new_score': new_score,
                    'change_source': 'correlation',
                    'source_id': None
                })
    
    def update_source_performance(self, source_id: str, performance_scores: Dict[str, float]) -> None:
        """
        Update a source's performance metrics and adjust influence capacity.
        
        Parameters:
        - source_id: The source to update
        - performance_scores: Dict mapping dimensions to performance scores (0-1)
        """
        # Record performance
        self.source_performance
        for dimension, score in performance_scores.items():
            if dimension in self.dimensions:
                self.source_performance[source_id][dimension].append({
                    'evaluation_round': self.evaluation_round,
                    'score': score,
                    'timestamp': time.time()
                })
                
                # Record in temporal DB
                self.temporal_db['source_performance'].append({
                    'evaluation_round': self.evaluation_round,
                    'timestamp': time.time(),
                    'source_id': source_id,
                    'dimension': dimension,
                    'performance_score': score
                })
                
                # Update influence capacity based on performance
                current_capacity = self.source_influence_capacity[source_id][dimension]
                
                # Adjust capacity based on performance
                if score >= 0.5:  # Good performance
                    growth_rate = self.config.get('influence_growth_rate', 0.05)
                    new_capacity = current_capacity * (1 + (score - 0.5) * growth_rate)
                else:  # Poor performance
                    decay_rate = self.config.get('influence_decay_rate', 0.1)
                    new_capacity = current_capacity * (1 - (0.5 - score) * decay_rate)
                
                # Apply limits to capacity changes
                max_growth = self.config.get('max_capacity_growth', 1.2)
                max_decay = self.config.get('max_capacity_decay', 0.8)
                
                if new_capacity > current_capacity:
                    new_capacity = min(new_capacity, current_capacity * max_growth)
                else:
                    new_capacity = max(new_capacity, current_capacity * max_decay)
                
                # Update capacity
                self.source_influence_capacity[source_id][dimension] = new_capacity
    
    def update_agent_performance(self, agent_id: int, performance_scores: Dict[str, float]) -> None:
        """
        Record agent performance for later analysis and reward calculation.
        
        Parameters:
        - agent_id: The agent to update
        - performance_scores: Dict mapping dimensions to performance scores (0-1)
        """
        for dimension, score in performance_scores.items():
            if dimension in self.dimensions:
                # Record performance
                self.agent_performance[agent_id][dimension].append({
                    'evaluation_round': self.evaluation_round,
                    'score': self.agent_trust_scores[agent_id][dimension],
                    'timestamp': time.time()
                })
                
                # Record in temporal DB
                self.temporal_db['agent_performance'].append({
                    'evaluation_round': self.evaluation_round,
                    'timestamp': time.time(),
                    'agent_id': agent_id,
                    'dimension': dimension,
                    'performance_score': score
                })
    
    def calculate_source_rewards(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate rewards for sources based on performance of their investments.
        
        Returns:
        - Dict mapping source_ids to {dimension: reward} dictionaries
        """
        rewards = defaultdict(lambda: defaultdict(float))
        
        # Get latest agent performance
        latest_performance = {}
        for agent_id, dimensions in self.agent_performance.items():
            latest_performance[agent_id] = {}
            for dimension, entries in dimensions.items():
                if entries:
                    latest_performance[agent_id][dimension] = entries[-1]['score']
        
        # Calculate rewards based on performance of investments
        for source_id, agent_investments in self.source_investments.items():
            for agent_id, dimensions in agent_investments.items():
                if agent_id in latest_performance:
                    for dimension, investment in dimensions.items():
                        if dimension in latest_performance[agent_id]:
                            # Reward is proportional to investment * performance
                            performance = latest_performance[agent_id][dimension]
                            reward = investment * (performance - 0.5) * 2  # Normalize to -1 to 1 range
                            
                            rewards[source_id][dimension] += reward
        
        return rewards
    
    def increment_evaluation_round(self, increment: int = 1) -> None:
        """
        Increment the evaluation round counter.
        
        Parameters:
        - increment: Amount to increment the counter
        """
        self.evaluation_round += increment
    
    def get_agent_trust(self, agent_id: int) -> Dict[str, float]:
        """Get current trust scores for an agent across all dimensions."""
        return dict(self.agent_trust_scores[agent_id])
    
    def get_agent_permissions(self, agent_id: int) -> Dict[str, float]:
        """
        Calculate what permissions an agent has based on trust scores.
        
        Parameters:
        - agent_id: Agent to check
        
        Returns:
        - Dict of permission types and levels
        """
        trust_scores = self.agent_trust_scores[agent_id]
        
        # Group dimensions by permission category
        permission_mapping = {
            'data_access': ['Factual_Correctness', 'Transparency', 'Trust_Calibration'],
            'action_rights': ['Safety_Security', 'Value_Alignment', 'Process_Reliability'],
            'routing_priority': ['Problem_Resolution', 'Communication_Quality'],
            'network_access': ['Manipulation_Resistance', 'Adaptability']
        }
        
        permissions = {}
        for perm_type, dimensions in permission_mapping.items():
            # Get scores for relevant dimensions
            scores = [trust_scores.get(dim, 0) for dim in dimensions 
                     if dim in self.dimensions]
            
            if not scores:
                permissions[perm_type] = 0
                continue
                
            # Weight by dimension importance
            weights = [self.dimension_weights.get(dim, 1.0) for dim in dimensions 
                      if dim in self.dimensions]
            
            # Compute weighted average
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            total_weight = sum(weights)
            
            if total_weight > 0:
                # Normalize to 0-10 scale
                permissions[perm_type] = min(10, max(0, (weighted_sum / total_weight) * 10))
            else:
                permissions[perm_type] = 0
        
        return permissions
    
    def get_source_endorsements(self, source_id: str) -> List[Dict]:
        """
        Get all current investments made by a source.
        
        Parameters:
        - source_id: The source to query
        
        Returns:
        - List of investment records
        """
        endorsements = []
        
        for agent_id, dimensions in self.source_investments[source_id].items():
            for dimension, amount in dimensions.items():
                if amount > 0:
                    endorsements.append({
                        'agent_id': agent_id,
                        'dimension': dimension,
                        'influence_amount': amount
                    })
        
        return endorsements
    
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
        if not self.temporal_db['trust_scores']:
            print("No trust score data available for visualization")
            return
        
        # Convert temporal DB to DataFrame
        df = pd.DataFrame(self.temporal_db['trust_scores'])
        
        # Apply filters
        if agents:
            df = df[df['agent_id'].isin(agents)]
        
        if dimensions:
            df = df[df['dimension'].isin(dimensions)]
        
        if start_round is not None:
            df = df[df['evaluation_round'] >= start_round]
        
        if end_round is not None:
            df = df[df['evaluation_round'] <= end_round]
        
        if df.empty:
            print("No data available with current filters")
            return
        
        # Sort by evaluation round and timestamp
        df = df.sort_values(['evaluation_round', 'timestamp'])
        
        # Create a pivot table for visualization
        pivot = df.pivot_table(
            index='evaluation_round', 
            columns=['agent_id', 'dimension'], 
            values='new_score',
            aggfunc='last'  # Take the last score for each round
        )
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each agent-dimension combination
        for col in pivot.columns:
            agent_id, dimension = col
            label = f"Agent {agent_id} - {dimension}"
            pivot[col].plot(ax=ax, label=label)
        
        # Add labels and legend
        ax.set_xlabel('Evaluation Round')
        ax.set_ylabel('Trust Score')
        ax.set_title('Trust Scores Over Time')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
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
        if not self.temporal_db['source_performance']:
            print("No source performance data available for visualization")
            return
        
        # Convert temporal DB to DataFrame
        df = pd.DataFrame(self.temporal_db['source_performance'])
        
        # Apply filters
        if sources:
            df = df[df['source_id'].isin(sources)]
        
        if dimensions:
            df = df[df['dimension'].isin(dimensions)]
        
        if start_round is not None:
            df = df[df['evaluation_round'] >= start_round]
        
        if end_round is not None:
            df = df[df['evaluation_round'] <= end_round]
        
        if df.empty:
            print("No data available with current filters")
            return
        
        # Sort by evaluation round and timestamp
        df = df.sort_values(['evaluation_round', 'timestamp'])
        
        # Create a pivot table for visualization
        pivot = df.pivot_table(
            index='evaluation_round', 
            columns=['source_id', 'dimension'], 
            values='performance_score',
            aggfunc='last'  # Take the last score for each round
        )
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each source-dimension combination
        for col in pivot.columns:
            source_id, dimension = col
            label = f"{source_id} - {dimension}"
            pivot[col].plot(ax=ax, label=label)
        
        # Add labels and legend
        ax.set_xlabel('Evaluation Round')
        ax.set_ylabel('Performance Score')
        ax.set_title('Source Performance Over Time')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return fig