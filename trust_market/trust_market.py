import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field


@dataclass
class TrustEndorsement:
    """Represents an endorsement of trust from one entity to another with counter-based decay."""
    endorser_id: str
    agent_id: int
    dimension: str
    influence_amount: float  # Amount of influence allocated
    creation_timestamp: float  # When this endorsement was created
    decay_rate: float = 1.0  # Default to no decay (1.0)
    confidence: float = 1.0  # How confident the endorser is in this endorsement
    # Track how this endorsement has performed
    performance_history: List[float] = field(default_factory=list)
    age_counter: int = 0  # Counter for endorsement age (not time-based)
    
    @property
    def is_active(self) -> bool:
        """Check if the endorsement is still active (significant value)."""
        return self.current_value > 0.01  # Threshold for considering an endorsement inactive
    
    @property
    def current_value(self) -> float:
        """Get the current value of this endorsement after decay."""
        if self.decay_rate == 1.0:  # No decay
            return self.influence_amount
        return self.influence_amount * (self.decay_rate ** self.age_counter)
    
    def increment_age(self, increment: int = 1) -> None:
        """Increment the age counter by the specified amount."""
        self.age_counter += increment
    
    @property
    def performance(self) -> float:
        """Get the average performance of this endorsement."""
        if not self.performance_history:
            return 0.0
        return sum(self.performance_history) / len(self.performance_history)
    
    def record_performance(self, score: float) -> None:
        """Record how this endorsement performed."""
        self.performance_history.append(score)


class TrustMarket:
    """
    A market system for trust as a tradable asset based on endorsements.
    Redesigned with explicit sources/sinks and optimized updates.
    """
    def __init__(self, config: Dict[str, Any]):
        """Initialize the trust market."""
        self.config = config
        self.dimensions = config.get('dimensions', [])
        self.dimension_weights = config.get('dimension_weights', {dim: 1.0 for dim in self.dimensions})
        
        # Trust scores for all agents across all dimensions (cache)
        self.agent_trust_scores = defaultdict(lambda: {dim: 0.0 for dim in self.dimensions})
        
        # Primary trust sources
        self.primary_sources = set(config.get('primary_sources', ['user_feedback', 'regulator']))
        
        # Track all endorsements in the system
        self.endorsements = []
        
        # Track endorsements by agent and dimension for faster access
        self.agent_endorsements = defaultdict(lambda: defaultdict(list))
        
        # Track influence capacity for all entities
        self.source_influence_capacity = defaultdict(lambda: {dim: 0.0 for dim in self.dimensions})
        
        # Track influence already allocated by sources (portfolios)
        self.allocated_influence = defaultdict(lambda: {dim: 0.0 for dim in self.dimensions})
        
        # Counter for tracking evaluation rounds
        self.evaluation_round = 0
        
        # Performance metrics
        self.market_performance = {
            'source_performance': defaultdict(list),  # How well each source predicts
            'market_volatility': defaultdict(list),  # Trust score volatility by dimension
        }
        
        # Affected entities cache - track what needs updating
        self._update_queue = set()
    
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
        
        # Add to primary sources if needed
        if is_primary:
            self.primary_sources.add(source_id)
    
    def update_source_influence(self, source_id: str, 
                               performance_scores: Dict[str, float]) -> None:
        """
        Update an information source's influence capacity based on its performance.
        
        Parameters:
        - source_id: The source to update
        - performance_scores: Dict mapping dimensions to performance scores (0-1)
        """
        if source_id not in self.source_influence_capacity:
            return
            
        # Update influence capacity based on performance
        for dimension, score in performance_scores.items():
            if dimension in self.dimensions:
                # Increase or decrease influence based on performance
                current = self.source_influence_capacity[source_id][dimension]
                
                # Apply adjustment formula: good performance increases capacity,
                # bad performance decreases it more severely
                if score >= 0.5:
                    # Good performance: linear increase
                    adjustment = (score - 0.5) * self.config.get('influence_gain_rate', 0.1)
                else:
                    # Poor performance: quadratic decrease
                    adjustment = -((0.5 - score) ** 2) * self.config.get('influence_loss_rate', 0.2)
                
                new_capacity = max(0, current + adjustment * current)
                self.source_influence_capacity[source_id][dimension] = new_capacity
                
                # Mark this source's allocations for updating
                self._update_queue.add(source_id)
                
                # Record performance for analytics
                self.market_performance['source_performance'][source_id].append({
                    'dimension': dimension,
                    'score': score,
                    'capacity_before': current,
                    'capacity_after': new_capacity,
                    'evaluation_round': self.evaluation_round
                })
    
    def create_endorsement(self, source_id: str, agent_id: int, dimension: str, 
                          influence_amount: float, confidence: float = 1.0,
                          decay_rate: float = None) -> Tuple[bool, Optional[int]]:
        """
        Create a trust endorsement from a source to an agent.
        
        Parameters:
        - source_id: The endorsing entity
        - agent_id: The agent being endorsed
        - dimension: Which trust dimension is being endorsed
        - influence_amount: How much influence to allocate to this endorsement
        - confidence: How confident the source is in this endorsement (0-1)
        - decay_rate: Rate at which this endorsement decays (default from config)
        
        Returns:
        - (success, endorsement_id)
        """
        # Validate the request
        if dimension not in self.dimensions:
            return False, None
            
        # Check if source has enough available influence
        available = self.source_influence_capacity[source_id][dimension] - \
                   self.allocated_influence[source_id][dimension]
        
        if influence_amount > available:
            return False, None
        
        # Get decay rate from config if not specified
        if decay_rate is None:
            decay_rate = self.config.get(f'decay_rate_{dimension}', 
                                       self.config.get('default_decay_rate', 1.0))
            
        # Create endorsement
        endorsement = TrustEndorsement(
            endorser_id=source_id,
            agent_id=agent_id,
            dimension=dimension,
            influence_amount=influence_amount,
            creation_timestamp=time.time(),
            decay_rate=decay_rate,
            confidence=confidence
        )
        
        # Add to market
        endorsement_id = len(self.endorsements)
        self.endorsements.append(endorsement)
        
        # Add to agent-specific index for faster lookup
        self.agent_endorsements[agent_id][dimension].append(endorsement_id)
        
        # Update allocated influence
        self.allocated_influence[source_id][dimension] += influence_amount
        
        # Mark affected entities for update
        self._update_queue.add(agent_id)
        
        # Update agent's trust score for this dimension immediately
        self._recalculate_agent_trust(agent_id, dimension)
        
        return True, endorsement_id
    
    def revoke_endorsement(self, endorsement_id: int) -> bool:
        """
        Revoke an existing endorsement, freeing up the source's influence.
        
        Parameters:
        - endorsement_id: ID of the endorsement to revoke
        
        Returns:
        - success: Whether revocation was successful
        """
        if endorsement_id >= len(self.endorsements):
            return False
            
        endorsement = self.endorsements[endorsement_id]
        
        # Check if already inactive
        if not endorsement.is_active:
            return False
            
        # Force immediate decay
        endorsement.age_counter = 1000  # Large number to ensure it's inactive
        
        # Free up allocated influence
        self.allocated_influence[endorsement.endorser_id][endorsement.dimension] -= endorsement.influence_amount
        
        # Mark entities for update
        self._update_queue.add(endorsement.agent_id)
        
        # Recalculate agent's trust score
        self._recalculate_agent_trust(endorsement.agent_id, endorsement.dimension)
        
        return True
    
    def _recalculate_agent_trust(self, agent_id: int, dimension: str) -> None:
        """
        Recalculate an agent's trust score for a dimension based on current endorsements.
        Trust score is calculated as a weighted average of all active endorsements.
        """
        # Get all active endorsements for this agent and dimension
        if agent_id not in self.agent_endorsements or dimension not in self.agent_endorsements[agent_id]:
            # No endorsements, score remains at zero
            self.agent_trust_scores[agent_id][dimension] = 0.0
            return
            
        endorsement_ids = self.agent_endorsements[agent_id][dimension]
        relevant_endorsements = [
            self.endorsements[e_id] for e_id in endorsement_ids
            if self.endorsements[e_id].is_active
        ]
        
        if not relevant_endorsements:
            # No active endorsements, score is zero
            self.agent_trust_scores[agent_id][dimension] = 0.0
            return
        
        # Calculate weighted average of endorsements
        total_weighted_influence = sum(
            e.current_value * e.confidence for e in relevant_endorsements
        )
        
        # Total raw influence (for normalization)
        total_influence = sum(e.current_value for e in relevant_endorsements)
        
        if total_influence > 0:
            # Calculate weighted score
            score = total_weighted_influence / total_influence
            
            # Additional weighting for primary sources vs secondary sources
            primary_weight = self.config.get('primary_source_weight', 2.0)
            secondary_weight = self.config.get('secondary_source_weight', 1.0)
            
            primary_sum = sum(
                e.current_value * e.confidence 
                for e in relevant_endorsements
                if e.endorser_id in self.primary_sources
            )
            secondary_sum = total_weighted_influence - primary_sum
            
            # Normalized weighted score accounting for source types
            if total_weighted_influence > 0:
                score = (primary_weight * primary_sum + secondary_weight * secondary_sum) / \
                        (primary_weight * sum(e.current_value for e in relevant_endorsements 
                                             if e.endorser_id in self.primary_sources) + 
                         secondary_weight * sum(e.current_value for e in relevant_endorsements 
                                              if e.endorser_id not in self.primary_sources))
            
            self.agent_trust_scores[agent_id][dimension] = score
        else:
            self.agent_trust_scores[agent_id][dimension] = 0.0
    
    def increment_evaluation_round(self, increment: int = 1) -> None:
        """
        Increment the evaluation round counter and age all endorsements.
        
        Parameters:
        - increment: Amount to increment the counter
        """
        self.evaluation_round += increment
        
        # Age all endorsements
        for endorsement in self.endorsements:
            if endorsement.is_active:
                endorsement.increment_age(increment)
                
                # If endorsement becomes inactive due to aging, mark for update
                if not endorsement.is_active:
                    self._update_queue.add(endorsement.agent_id)
        
        # Process update queue
        self._process_update_queue()
    
    def _process_update_queue(self) -> None:
        """Process all entities queued for updates."""
        # Process agent updates
        for agent_id in self._update_queue:
            if isinstance(agent_id, int):  # Agent IDs are integers
                for dimension in self.dimensions:
                    self._recalculate_agent_trust(agent_id, dimension)
        
        # Clear the queue
        self._update_queue.clear()
    
    def record_primary_feedback(self, source_id: str, agent_id: int, 
                               dimension: str, rating: float,
                               confidence: float = 1.0) -> None:
        """
        Record primary feedback from a trust source (like user feedback or regulator).
        
        Parameters:
        - source_id: Primary source ID
        - agent_id: Agent being rated
        - dimension: Trust dimension being rated
        - rating: The rating (0-1)
        - confidence: Confidence in this rating
        """
        if source_id not in self.primary_sources:
            return
            
        # Create endorsement with a higher weight
        influence_amount = self.config.get('primary_source_influence_factor', 10.0)
        self.create_endorsement(source_id, agent_id, dimension, influence_amount, confidence)
        
        # Evaluate other endorsements against this primary feedback
        self._evaluate_endorsements_against_primary(agent_id, dimension, rating, source_id)
    
    def _evaluate_endorsements_against_primary(self, agent_id: int, dimension: str, 
                                             primary_rating: float, primary_source: str) -> None:
        """Evaluate how well other endorsements predicted the primary trust source rating."""
        if agent_id not in self.agent_endorsements or dimension not in self.agent_endorsements[agent_id]:
            return
            
        endorsement_ids = self.agent_endorsements[agent_id][dimension]
        
        for e_id in endorsement_ids:
            endorsement = self.endorsements[e_id]
            if endorsement.is_active and endorsement.endorser_id != primary_source:
                # Calculate how well this endorsement aligned with primary rating
                error = abs(endorsement.current_value - primary_rating)
                max_error = 1.0  # Assuming values are 0-1
                performance = 1.0 - (error / max_error)
                
                # Record performance
                endorsement.record_performance(performance)
                
                # Update source influence based on how well they predicted primary ratings
                if len(endorsement.performance_history) >= self.config.get('min_samples_for_update', 3):
                    self.update_source_influence(
                        endorsement.endorser_id,
                        {dimension: endorsement.performance}
                    )
    
    def get_agent_trust(self, agent_id: int) -> Dict[str, float]:
        """Get current trust scores for an agent across all dimensions."""
        return dict(self.agent_trust_scores[agent_id])
    
    def get_agent_permissions(self, agent_id: int) -> Dict[str, float]:
        """
        Calculate what permissions an agent has based on trust scores.
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
