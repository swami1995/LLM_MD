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
    """Represents an endorsement of trust from one entity to another."""
    endorser_id: str
    agent_id: int
    dimension: str
    influence_amount: float  # Amount of influence allocated
    timestamp: float
    expiration: Optional[float] = None  # When this endorsement expires (if applicable)
    confidence: float = 1.0  # How confident the endorser is in this endorsement
    # Track how this endorsement has performed
    performance_history: List[float] = field(default_factory=list)
    
    @property
    def is_active(self) -> bool:
        """Check if the endorsement is still active."""
        if self.expiration is None:
            return True
        return time.time() < self.expiration
    
    @property
    def age(self) -> float:
        """Get the age of this endorsement in seconds."""
        return time.time() - self.timestamp
    
    @property
    def performance(self) -> float:
        """Get the average performance of this endorsement."""
        if not self.performance_history:
            return 0.0
        return sum(self.performance_history) / len(self.performance_history)
    
    def record_performance(self, score: float) -> None:
        """Record how this endorsement performed against ground truth."""
        self.performance_history.append(score)


class TrustMarket:
    """
    A market system for trust as a tradable asset based on endorsements rather than transfers.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trust market.
        
        Parameters:
        - config: Configuration parameters including:
            - dimensions: List of trust dimensions
            - dimension_weights: Default importance weights for dimensions
            - dimension_decay_rates: How quickly trust decays in each dimension
            - ground_truth_update_frequency: How often ground truth updates occur
            - endorsement_expiration: Default time (in seconds) before endorsements expire
        """
        self.config = config
        self.dimensions = config.get('dimensions', [])
        self.dimension_weights = config.get('dimension_weights', {dim: 1.0 for dim in self.dimensions})
        self.dimension_decay_rates = config.get('dimension_decay_rates', {dim: 0.001 for dim in self.dimensions})
        
        # Trust scores for all agents across all dimensions
        self.agent_trust_scores = defaultdict(lambda: {dim: 0.0 for dim in self.dimensions})
        
        # Ground truth scores when available
        self.ground_truth_scores = defaultdict(lambda: {dim: None for dim in self.dimensions})
        self.last_ground_truth_update = defaultdict(float)
        
        # Track all endorsements in the system
        self.endorsements = []
        
        # Track influence capacity for all entities
        self.source_influence_capacity = defaultdict(lambda: {dim: 0.0 for dim in self.dimensions})
        
        # Track influence already allocated by sources
        self.allocated_influence = defaultdict(lambda: {dim: 0.0 for dim in self.dimensions})
        
        # Market metadata
        self.last_update_time = time.time()
        self.market_age = 0.0
        
        # Performance metrics
        self.market_performance = {
            'prediction_accuracy': [],  # How well market predictions match ground truth
            'information_source_performance': defaultdict(list),  # How well each source predicts
            'market_volatility': defaultdict(list),  # Trust score volatility by dimension
        }
    
    def add_information_source(self, source_id: str, source_type: str, 
                               initial_influence: Dict[str, float]) -> None:
        """
        Add a new information source to the market with initial influence capacity.
        
        Parameters:
        - source_id: Unique identifier for the source
        - source_type: Type of information source (e.g., 'expert', 'regulator')
        - initial_influence: Dict mapping dimensions to initial influence capacity
        """
        # Only allocate influence for valid dimensions
        for dimension in initial_influence:
            if dimension in self.dimensions:
                self.source_influence_capacity[source_id][dimension] = initial_influence[dimension]
    
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
                
                # Record performance for analytics
                self.market_performance['information_source_performance'][source_id].append({
                    'dimension': dimension,
                    'score': score,
                    'capacity_before': current,
                    'capacity_after': new_capacity,
                    'timestamp': time.time()
                })
    
    def create_endorsement(self, source_id: str, agent_id: int, dimension: str, 
                          influence_amount: float, confidence: float = 1.0,
                          expiration: Optional[float] = None) -> Tuple[bool, Optional[int]]:
        """
        Create a trust endorsement from a source to an agent.
        
        Unlike traditional market transactions, sources don't "spend" their influence;
        they allocate it, potentially gaining or losing influence based on accuracy.
        
        Parameters:
        - source_id: The endorsing entity
        - agent_id: The agent being endorsed
        - dimension: Which trust dimension is being endorsed
        - influence_amount: How much influence to allocate to this endorsement
        - confidence: How confident the source is in this endorsement (0-1)
        - expiration: Optional timestamp when endorsement expires
        
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
        
        # Create endorsement
        if expiration is None and 'endorsement_expiration' in self.config:
            expiration = time.time() + self.config['endorsement_expiration']
            
        endorsement = TrustEndorsement(
            endorser_id=source_id,
            agent_id=agent_id,
            dimension=dimension,
            influence_amount=influence_amount,
            timestamp=time.time(),
            expiration=expiration,
            confidence=confidence
        )
        
        # Add to market
        endorsement_id = len(self.endorsements)
        self.endorsements.append(endorsement)
        
        # Update allocated influence
        self.allocated_influence[source_id][dimension] += influence_amount
        
        # Recalculate agent's trust score for this dimension
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
        
        # Check if already expired
        if not endorsement.is_active:
            return False
            
        # Set expiration to now
        endorsement.expiration = time.time()
        
        # Free up allocated influence
        self.allocated_influence[endorsement.endorser_id][endorsement.dimension] -= endorsement.influence_amount
        
        # Recalculate agent's trust score
        self._recalculate_agent_trust(endorsement.agent_id, endorsement.dimension)
        
        return True
    
    def update_ground_truth(self, agent_id: int, dimension: str, score: float) -> None:
        """
        Update ground truth data for an agent on a specific dimension.
        This represents authoritative assessments from regulators or
        aggregated user feedback.
        
        Parameters:
        - agent_id: The agent being assessed
        - dimension: The trust dimension being assessed
        - score: The ground truth score (0-1)
        """
        if dimension not in self.dimensions:
            return
            
        # Record ground truth
        self.ground_truth_scores[agent_id][dimension] = score
        self.last_ground_truth_update[agent_id] = time.time()
        
        # Evaluate endorsements against ground truth
        self._evaluate_endorsements(agent_id, dimension, score)
        
        # Update information source influence based on performance
        self._update_source_performance(agent_id, dimension, score)
    
    def _update_source_performance(self, agent_id: int, dimension: str, ground_truth: float) -> None:
        """Update information source influence based on endorsement accuracy."""
        # Group by source
        source_performance = defaultdict(list)
        
        # Evaluate each active endorsement for this agent/dimension
        for endorsement in self.endorsements:
            if (endorsement.agent_id == agent_id and 
                endorsement.dimension == dimension and 
                endorsement.is_active):
                
                # Calculate endorsement performance - how close was the market score to ground truth?
                market_score = self.agent_trust_scores[agent_id][dimension]
                error = abs(market_score - ground_truth)
                max_error = 1.0  # Assuming scores are 0-1
                performance = 1.0 - (error / max_error)
                
                # Weight by influence and confidence
                weighted_perf = performance * endorsement.influence_amount * endorsement.confidence
                source_performance[endorsement.endorser_id].append(weighted_perf)
        
        # Update each source's influence capacity
        for source_id, performances in source_performance.items():
            if performances:
                avg_performance = sum(performances) / len(performances)
                self.update_source_influence(source_id, {dimension: avg_performance})
    
    def _evaluate_endorsements(self, agent_id: int, dimension: str, ground_truth: float) -> None:
        """Evaluate how well endorsements predicted the ground truth."""
        for endorsement in self.endorsements:
            if (endorsement.agent_id == agent_id and 
                endorsement.dimension == dimension and 
                endorsement.is_active):
                
                # Calculate how well this endorsement predicted ground truth
                market_score = self.agent_trust_scores[agent_id][dimension]
                error = abs(market_score - ground_truth)
                max_error = 1.0  # Assuming scores are 0-1
                performance = 1.0 - (error / max_error)
                
                # Record performance
                endorsement.record_performance(performance)
    
    def _recalculate_agent_trust(self, agent_id: int, dimension: str) -> None:
        """
        Recalculate an agent's trust score for a dimension based on current endorsements.
        
        Trust score is calculated as a weighted average of all active endorsements,
        weighted by influence amount and confidence.
        """
        # Get all active endorsements for this agent and dimension
        relevant_endorsements = [
            e for e in self.endorsements
            if e.agent_id == agent_id and e.dimension == dimension and e.is_active
        ]
        
        if not relevant_endorsements:
            # No active endorsements, score decays toward zero
            current = self.agent_trust_scores[agent_id][dimension]
            decay_rate = self.dimension_decay_rates[dimension]
            self.agent_trust_scores[agent_id][dimension] = current * (1 - decay_rate)
            return
        
        # Calculate weighted average of endorsements
        total_influence = sum(e.influence_amount * e.confidence for e in relevant_endorsements)
        
        if total_influence > 0:
            # Each endorsement contributes proportionally to its influence and confidence
            weighted_sum = sum(e.influence_amount * e.confidence for e in relevant_endorsements)
            score = weighted_sum / total_influence
            
            # Apply dimension-specific decay based on oldest endorsement age
            oldest_age = max(e.age for e in relevant_endorsements)
            decay_factor = 1 - (self.dimension_decay_rates[dimension] * oldest_age)
            decay_factor = max(0.5, decay_factor)  # Don't decay below 50%
            
            self.agent_trust_scores[agent_id][dimension] = score * decay_factor
        else:
            self.agent_trust_scores[agent_id][dimension] = 0.0
    
    def update_market(self) -> None:
        """
        Update the entire market state.
        - Recalculate all trust scores
        - Apply time-based decay to endorsements
        - Update market performance metrics
        """
        now = time.time()
        time_delta = now - self.last_update_time
        self.market_age += time_delta
        
        # Update all agent trust scores
        for agent_id in self.agent_trust_scores:
            for dimension in self.dimensions:
                self._recalculate_agent_trust(agent_id, dimension)
        
        # Calculate market performance metrics
        self._calculate_market_metrics()
        
        self.last_update_time = now
    
    ## This seems pretty badly done. garbage. needs to be replaced with more suitable.
    def _calculate_market_metrics(self) -> None:
        """Calculate market performance metrics."""
        # Calculate prediction accuracy
        prediction_accuracy = []
        for agent_id, ground_truth in self.ground_truth_scores.items():
            for dimension, gt_score in ground_truth.items():
                if gt_score is not None:
                    market_score = self.agent_trust_scores[agent_id][dimension]
                    error = abs(market_score - gt_score)
                    accuracy = 1.0 - error
                    prediction_accuracy.append(accuracy)
        
        if prediction_accuracy:
            self.market_performance['prediction_accuracy'].append({
                'timestamp': time.time(),
                'mean_accuracy': sum(prediction_accuracy) / len(prediction_accuracy)
            })
        
        # Calculate market volatility
        for dimension in self.dimensions:
            scores = [self.agent_trust_scores[agent_id][dimension] 
                      for agent_id in self.agent_trust_scores]
            if scores:
                volatility = np.std(scores)
                self.market_performance['market_volatility'][dimension].append({
                    'timestamp': time.time(),
                    'volatility': volatility
                })
    
    def get_agent_trust(self, agent_id: int) -> Dict[str, float]:
        """Get current trust scores for an agent across all dimensions."""
        return dict(self.agent_trust_scores[agent_id])
    
    def get_agent_permissions(self, agent_id: int) -> Dict[str, float]:
        """
        Calculate what permissions an agent has based on trust scores.
        Returns a normalized score (0-10) for different permission categories.
        """
        trust_scores = self.agent_trust_scores[agent_id]
        
        # Group dimensions by permission category
        permission_mapping = {
            'data_access': ['Factual Correctness', 'Transparency', 'Trust Calibration'],
            'action_rights': ['Safety & Security', 'Value Alignment', 'Process Reliability'],
            'routing_priority': ['Problem Resolution', 'Communication Quality'],
            'network_access': ['Manipulation Resistance', 'Adaptability']
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
    
    def get_market_state(self) -> Dict[str, Any]:
        """Get the current state of the trust market."""
        # Count active endorsements by dimension
        active_endorsements = defaultdict(int)
        for e in self.endorsements:
            if e.is_active:
                active_endorsements[e.dimension] += 1
        
        # Calculate total influence by dimension
        total_influence = defaultdict(float)
        for source_id, dimensions in self.source_influence_capacity.items():
            for dim, capacity in dimensions.items():
                total_influence[dim] += capacity
        
        # Calculate allocated influence percentage
        allocated_percentage = {}
        for dim in self.dimensions:
            if total_influence[dim] > 0:
                total_allocated = sum(self.allocated_influence[source_id][dim] 
                                     for source_id in self.source_influence_capacity)
                allocated_percentage[dim] = (total_allocated / total_influence[dim]) * 100
            else:
                allocated_percentage[dim] = 0
        
        return {
            'market_age': self.market_age,
            'active_endorsements': dict(active_endorsements),
            'total_endorsements': len(self.endorsements),
            'total_influence': dict(total_influence),
            'allocated_percentage': allocated_percentage,
            'prediction_accuracy': self.market_performance['prediction_accuracy'][-1]['mean_accuracy'] 
                                  if self.market_performance['prediction_accuracy'] else None,
            'market_volatility': {dim: data[-1]['volatility'] 
                                for dim, data in self.market_performance['market_volatility'].items() 
                                if data}
        }
    
    def visualize_market(self, save_path: Optional[str] = None) -> None:
        """
        Generate visualizations of the trust market state.
        
        Parameters:
        - save_path: Optional path to save visualizations
        """
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle('Trust Market Analysis', fontsize=16)
        
        # 1. Agent Trust Scores Heatmap
        ax1 = plt.subplot(2, 2, 1)
        self._plot_trust_heatmap(ax1)
        
        # 2. Market Prediction Accuracy
        ax2 = plt.subplot(2, 2, 2)
        self._plot_prediction_accuracy(ax2)
        
        # 3. Dimension Activity
        ax3 = plt.subplot(2, 2, 3)
        self._plot_dimension_activity(ax3)
        
        # 4. Information Source Performance
        ax4 = plt.subplot(2, 2, 4)
        self._plot_source_performance(ax4)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def _plot_trust_heatmap(self, ax):
        """Plot heatmap of agent trust scores across dimensions."""
        # Prepare data for heatmap
        agents = sorted(self.agent_trust_scores.keys())
        dimensions = self.dimensions
        
        data = []
        for agent_id in agents:
            data.append([self.agent_trust_scores[agent_id][dim] for dim in dimensions])
        
        if not data:
            ax.text(0.5, 0.5, "No trust data available", 
                   horizontalalignment='center', verticalalignment='center')
            return
            
        # Create heatmap
        sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu", 
                   xticklabels=[d[:10] for d in dimensions],  # Abbreviate long names
                   yticklabels=[f"Agent {a}" for a in agents],
                   ax=ax, vmin=0, vmax=1)
        
        ax.set_title("Agent Trust Scores by Dimension")
        ax.set_ylabel("Agent")
        ax.set_xlabel("Dimension")
    
    def _plot_prediction_accuracy(self, ax):
        """Plot market prediction accuracy over time."""
        if not self.market_performance['prediction_accuracy']:
            ax.text(0.5, 0.5, "No prediction accuracy data available", 
                   horizontalalignment='center', verticalalignment='center')
            return
            
        # Convert to DataFrame for easier plotting
        accuracy_data = pd.DataFrame(self.market_performance['prediction_accuracy'])
        accuracy_data['relative_time'] = accuracy_data['timestamp'] - accuracy_data['timestamp'].min()
        
        # Plot
        ax.plot(accuracy_data['relative_time'], accuracy_data['mean_accuracy'])
        ax.set_title("Market Prediction Accuracy Over Time")
        ax.set_ylabel("Mean Accuracy")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def _plot_dimension_activity(self, ax):
        """Plot endorsement activity by dimension."""
        # Count active endorsements by dimension
        active_endorsements = defaultdict(int)
        for e in self.endorsements:
            if e.is_active:
                active_endorsements[e.dimension] += 1
        
        if not active_endorsements:
            ax.text(0.5, 0.5, "No active endorsements", 
                   horizontalalignment='center', verticalalignment='center')
            return
            
        # Convert to DataFrame
        dimension_data = pd.DataFrame({
            'Dimension': list(active_endorsements.keys()),
            'Active Endorsements': list(active_endorsements.values())
        })
        
        # Sort by count
        dimension_data = dimension_data.sort_values('Active Endorsements', ascending=False)
        
        # Plot
        sns.barplot(x='Active Endorsements', y='Dimension', data=dimension_data, ax=ax)
        ax.set_title("Active Endorsements by Dimension")
        ax.set_xlabel("Number of Endorsements")
    
    def _plot_source_performance(self, ax):
        """Plot information source performance."""
        if not self.market_performance['information_source_performance']:
            ax.text(0.5, 0.5, "No source performance data available", 
                   horizontalalignment='center', verticalalignment='center')
            return
            
        # Compile performance data
        source_data = []
        for source_id, performances in self.market_performance['information_source_performance'].items():
            for perf in performances:
                source_data.append({
                    'Source': source_id,
                    'Dimension': perf['dimension'],
                    'Performance': perf['score'],
                    'Timestamp': perf['timestamp']
                })
        
        if not source_data:
            ax.text(0.5, 0.5, "No source performance data available", 
                   horizontalalignment='center', verticalalignment='center')
            return
            
        # Convert to DataFrame
        source_df = pd.DataFrame(source_data)
        
        # Calculate average performance by source
        avg_performance = source_df.groupby('Source')['Performance'].mean().reset_index()
        avg_performance = avg_performance.sort_values('Performance', ascending=False)
        
        # Plot
        sns.barplot(x='Performance', y='Source', data=avg_performance, ax=ax)
        ax.set_title("Average Information Source Performance")
        ax.set_xlabel("Performance Score")
        ax.set_xlim(0, 1)
