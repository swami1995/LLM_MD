import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import defaultdict
# Use try-except for plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib or seaborn not found. Visualization methods will be disabled.")


class TrustMarket:
    """
    A market system for trust as a tradable asset based on investments.
    Focuses on tracking trust scores, source investments, and market dynamics.
    """
    def __init__(self, config: Dict[str, Any]):
        """Initialize the trust market."""
        self.config = config
        self.dimensions = config.get('dimensions', [])
        if not self.dimensions:
            raise ValueError("TrustMarket config must include a list of 'dimensions'.")

        self.dimension_weights = config.get('dimension_weights', {dim: 1.0 for dim in self.dimensions})

        # Rating scale used for user feedback (e.g., 1-5, 1-10)
        self.rating_scale = config.get('rating_scale', 5)
        if self.rating_scale <= 1:
             raise ValueError("Rating scale must be greater than 1.")


        # Trust scores for all agents across all dimensions (start at neutral 0.5)
        self.agent_trust_scores = defaultdict(lambda: {dim: 0.5 for dim in self.dimensions})

        # Track investments by sources in agents (by dimension)
        # structure: {source_id: {agent_id: {dimension: amount}}}
        self.source_investments = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        # Track influence capacity for all sources
        # structure: {source_id: {dimension: capacity}}
        self.source_influence_capacity = defaultdict(lambda: {dim: 0.0 for dim in self.dimensions}) # Start at 0, set by add_source
        self.source_available_capacity = defaultdict(lambda: {dim: 0.0 for dim in self.dimensions}) # Start at 0
        self.allocated_influence = defaultdict(lambda: {dim: 0.0 for dim in self.dimensions}) # Start at 0

        # Primary sources get higher weight in score aggregation (optional feature)
        self.primary_sources = set(config.get('primary_sources', ['user_feedback'])) # Default user feedback as primary
        self.primary_source_weight = config.get('primary_source_weight', 1.0) # Default to 1 if not used
        self.secondary_source_weight = config.get('secondary_source_weight', 1.0)

        # Performance tracking (stores history)
        # self.source_performance = defaultdict(lambda: defaultdict(list)) # {source: {dim: [history]}}
        self.agent_performance = defaultdict(lambda: defaultdict(list))   # {agent: {dim: [history]}}

        # Temporal trust database - track scores and investments over time
        self.temporal_db = {
            'trust_scores': [],      # Records of agent trust score changes
            'investments': [],       # Records of source investment actions
            # 'source_performance': [], # Records of source performance metrics (if calculated)
            'agent_performance': []  # Records of agent performance metrics (from feedback)
        }

        # Evaluation round counter
        self.evaluation_round = 0

        # Dimension correlations (optional)
        self.use_dimension_correlations = config.get('use_dimension_correlations', False)
        if self.use_dimension_correlations:
            self.initialize_dimension_correlations()

    def initialize_dimension_correlations(self):
        """Initialize the correlation structure between trust dimensions."""
        # Default correlations (can be overridden in config)
        default_correlations = {
            ('Factual_Correctness', 'Transparency'): 0.6,
            ('Communication_Quality', 'Problem_Resolution'): 0.5,
            ('Value_Alignment', 'Safety_Security'): 0.7,
            ('Trust_Calibration', 'Transparency'): 0.6,
            ('Factual_Correctness', 'Trust_Calibration'): 0.4,
            ('Process_Reliability', 'Safety_Security'): 0.5,
            ('Adaptability', 'Communication_Quality'): 0.3,
            ('Adaptability', 'Process_Reliability'): -0.2,
        }
        self.dimension_correlations = self.config.get('dimension_correlations', default_correlations)
        self.correlation_matrix = {}
        for dim1 in self.dimensions:
            for dim2 in self.dimensions:
                if dim1 == dim2:
                    self.correlation_matrix[(dim1, dim2)] = 1.0
                else:
                    corr = self.dimension_correlations.get((dim1, dim2),
                           self.dimension_correlations.get((dim2, dim1), 0.0))
                    self.correlation_matrix[(dim1, dim2)] = corr
        print("Initialized dimension correlations.")

    def add_information_source(self, source_id: str, source_type: str,
                               initial_influence: Dict[str, float],
                               is_primary: bool = False) -> None:
        """
        Add a new information source with initial influence capacity.
        """
        if source_id in self.source_influence_capacity:
             print(f"Warning: Source {source_id} already exists. Updating influence.")

        total_added_influence = 0
        for dimension, capacity in initial_influence.items():
            if dimension in self.dimensions:
                # If source already exists, add to existing capacity? Or replace? Let's replace.
                current_capacity = self.source_influence_capacity[source_id].get(dimension, 0.0)
                current_available = self.source_available_capacity[source_id].get(dimension, 0.0)

                self.source_influence_capacity[source_id][dimension] = capacity
                # Available capacity is the new total capacity minus already allocated influence
                allocated = self.allocated_influence[source_id].get(dimension, 0.0)
                self.source_available_capacity[source_id][dimension] = max(0.0, capacity - allocated)
                total_added_influence += capacity


            else:
                 print(f"Warning: Dimension '{dimension}' not found in market dimensions. Ignoring for source {source_id}.")

        if is_primary:
            self.primary_sources.add(source_id)

        print(f"Source {source_id} ({source_type}) registered. Total initial influence: {total_added_influence}. Primary: {is_primary}.")

    def process_investments(self, source_id: str, investments: List[Tuple[int, str, float, Optional[float]]]) -> None:
        """
        Process investment/divestment decisions from a source.
        Updates source investments and recalculates affected agent trust scores.

        Parameters:
        - source_id: The source making the investment decisions
        - investments: List of (agent_id, dimension, amount, confidence) tuples
                       Amount > 0 for investment, < 0 for divestment. Confidence is optional.
        """
        if source_id not in self.source_influence_capacity:
            print(f"Warning: Source {source_id} not registered. Cannot process investments.")
            return

        affected_agents_dims = set() # Track (agent_id, dimension) pairs affected

        for agent_id, dimension, amount, confidence in investments:
            if dimension not in self.dimensions:
                print(f"Warning: Invalid dimension '{dimension}' in investment from {source_id}. Skipping.")
                continue

            current_investment = self.source_investments[source_id][agent_id].get(dimension, 0.0)
            available_capacity = self.source_available_capacity[source_id].get(dimension, 0.0)
            allocated = self.allocated_influence[source_id].get(dimension, 0.0)
            total_capacity = self.source_influence_capacity[source_id].get(dimension, 0.0)


            log_entry = {
                'evaluation_round': self.evaluation_round,
                'timestamp': time.time(),
                'source_id': source_id,
                'agent_id': agent_id,
                'dimension': dimension,
                'requested_amount': amount,
                'confidence': confidence,
            }

            # --- Handle Divestment (amount < 0) ---
            if amount < 0:
                divest_amount_requested = abs(amount)
                # Can only divest up to the current investment amount
                actual_divestment = min(divest_amount_requested, current_investment)

                if actual_divestment > 0.001: # Threshold for meaningful change
                    new_investment = current_investment - actual_divestment
                    self.source_investments[source_id][agent_id][dimension] = new_investment

                    # Update available and allocated capacity
                    self.allocated_influence[source_id][dimension] = max(0.0, allocated - actual_divestment)
                    self.source_available_capacity[source_id][dimension] = total_capacity - self.allocated_influence[source_id][dimension] # Recalculate available

                    affected_agents_dims.add((agent_id, dimension))

                    # Log divestment
                    log_entry.update({
                        'type': 'divestment',
                        'actual_amount': -actual_divestment,
                        'new_investment_level': new_investment
                    })
                    self.temporal_db['investments'].append(log_entry)
                    # print(f"Debug: {source_id} divested {-actual_divestment:.2f} from Agent {agent_id} on {dimension}.")

                else:
                     # Log attempt if requested amount was non-zero but current investment was too low
                     if divest_amount_requested > 0:
                          log_entry.update({'type': 'divestment_failed', 'actual_amount': 0, 'reason': 'Insufficient current investment'})
                          self.temporal_db['investments'].append(log_entry)


            # --- Handle Investment (amount > 0) ---
            elif amount > 0:
                # Can only invest up to available capacity
                actual_investment = min(amount, available_capacity)

                if actual_investment > 0.001: # Threshold for meaningful change
                    new_investment = current_investment + actual_investment
                    self.source_investments[source_id][agent_id][dimension] = new_investment

                    # Update available and allocated capacity
                    self.allocated_influence[source_id][dimension] = allocated + actual_investment
                    self.source_available_capacity[source_id][dimension] = max(0.0, available_capacity - actual_investment)

                    affected_agents_dims.add((agent_id, dimension))

                    # Log investment
                    log_entry.update({
                        'type': 'investment',
                        'actual_amount': actual_investment,
                        'new_investment_level': new_investment
                    })
                    self.temporal_db['investments'].append(log_entry)
                    # print(f"Debug: {source_id} invested {actual_investment:.2f} in Agent {agent_id} on {dimension}.")
                else:
                     # Log attempt if requested amount was non-zero but available capacity was too low
                     if amount > 0:
                          log_entry.update({'type': 'investment_failed', 'actual_amount': 0, 'reason': 'Insufficient available capacity'})
                          self.temporal_db['investments'].append(log_entry)


        # --- Recalculate Trust Scores for Affected Agents/Dimensions ---
        if affected_agents_dims:
             print(f"Recalculating trust scores for {len(affected_agents_dims)} agent/dimension pairs...")
             # Group by agent_id for potential correlation application
             affected_agents_map = defaultdict(list)
             for agent_id, dimension in affected_agents_dims:
                  self._recalculate_agent_trust_dimension(agent_id, dimension)
                  affected_agents_map[agent_id].append(dimension)

             # Apply dimension correlations if enabled
             if self.use_dimension_correlations:
                 for agent_id, changed_dimensions in affected_agents_map.items():
                     self.apply_dimension_correlations(agent_id, changed_dimensions)


    def _recalculate_agent_trust_dimension(self, agent_id: int, dimension: str):
        """Recalculates the trust score for a specific agent and dimension based on all source investments."""
        total_weighted_investment = 0.0
        total_weight = 0.0

        # Iterate through all sources that have invested in this agent/dimension
        for source_id, agent_investments in self.source_investments.items():
            if agent_id in agent_investments and dimension in agent_investments[agent_id]:
                investment_amount = agent_investments[agent_id][dimension]
                if investment_amount > 0: # Only consider positive investments for score calculation
                     # Apply source weighting (optional)
                     weight = self.primary_source_weight if source_id in self.primary_sources else self.secondary_source_weight
                     total_weighted_investment += investment_amount * weight
                     total_weight += weight # Weight here could also be source's total influence capacity? Simpler for now.


        old_score = self.agent_trust_scores[agent_id].get(dimension, 0.5)
        new_score = old_score # Default if no investments

        # Normalize the aggregated investment into a 0-1 score
        # This normalization needs careful design. A simple approach:
        # Map the total weighted investment relative to some max possible investment?
        # Or use a sigmoid function?
        # Let's use a simple scaling relative to total capacity of sources investing.
        total_capacity_invested = sum(
             self.source_influence_capacity[sid].get(dimension, 0.0)
             for sid, investments in self.source_investments.items()
             if agent_id in investments and dimension in investments[agent_id] and investments[agent_id][dimension] > 0
        )

        if total_capacity_invested > 0.01 : # Avoid division by zero
            # Simple normalization: score is proportion of total capacity allocated to this agent/dim
            # This assumes investments directly map to trust level. Might need refinement.
            new_score = total_weighted_investment / total_capacity_invested
            # Clamp score between 0 and 1
            new_score = max(0.0, min(1.0, new_score))
        elif total_weighted_investment <= 0.001 :
             # If total investment is zero or negative, score should reflect that. Maybe decay?
             # For now, let's just set it low if no positive investments exist.
             has_positive_investment = any(
                  inv[agent_id].get(dimension, 0.0) > 0
                  for sid, inv in self.source_investments.items() if agent_id in inv
             )
             if not has_positive_investment:
                  new_score = 0.1 # Low score if no one is investing positively

        # Update score and log if changed significantly
        if abs(new_score - old_score) > 0.001:
            self.agent_trust_scores[agent_id][dimension] = new_score
            self.temporal_db['trust_scores'].append({
                'evaluation_round': self.evaluation_round,
                'timestamp': time.time(),
                'agent_id': agent_id,
                'dimension': dimension,
                'old_score': old_score,
                'new_score': new_score,
                'change_source': 'investment_recalc',
                'source_id': None # Aggregated change
            })


    def decay_trust_scores(self, decay_rate: float = 0.99) -> None:
        """
        Apply a decay rate to agent trust scores and source capacities/investments.
        This simulates the natural erosion of trust/influence over time without activity.
        """
        if not (0 < decay_rate <= 1.0):
             print(f"Warning: Invalid decay_rate ({decay_rate}). Skipping decay.")
             return

        affected_agents_dims = set()

        # Decay Agent Trust Scores directly
        for agent_id in list(self.agent_trust_scores.keys()):
             for dimension in self.dimensions:
                  old_score = self.agent_trust_scores[agent_id][dimension]
                  # Decay towards neutral 0.5 instead of 0? Let's decay towards 0.5
                  new_score = 0.5 + (old_score - 0.5) * decay_rate
                  if abs(new_score - old_score) > 0.001:
                       self.agent_trust_scores[agent_id][dimension] = new_score
                       affected_agents_dims.add((agent_id, dimension))
                       self.temporal_db['trust_scores'].append({
                            'evaluation_round': self.evaluation_round, 'timestamp': time.time(),
                            'agent_id': agent_id, 'dimension': dimension, 'old_score': old_score,
                            'new_score': new_score, 'change_source': 'decay', 'source_id': None
                       })


        # Decay Source Investments and Capacities
        for source_id in list(self.source_investments.keys()):
            for agent_id in list(self.source_investments[source_id].keys()):
                 for dimension in list(self.source_investments[source_id][agent_id].keys()):
                      self.source_investments[source_id][agent_id][dimension] *= decay_rate
                      # If investment becomes negligible, remove it? Optional cleanup.
                      # if self.source_investments[source_id][agent_id][dimension] < 0.001:
                      #     del self.source_investments[source_id][agent_id][dimension]

            # Recalculate allocated influence based on decayed investments
            for dimension in self.dimensions:
                 total_allocated_for_dim = sum(
                      agent_inv.get(dimension, 0.0)
                      for agent_inv in self.source_investments[source_id].values()
                 )
                 self.allocated_influence[source_id][dimension] = total_allocated_for_dim

                 # Decay total capacity as well? Or just let available recalculate?
                 # Let's assume total capacity is stable unless explicitly changed.
                 # Recalculate available capacity.
                 total_capacity = self.source_influence_capacity[source_id].get(dimension, 0.0)
                 self.source_available_capacity[source_id][dimension] = max(0.0, total_capacity - total_allocated_for_dim)


        # Recalculate scores for dimensions affected by score decay (if not already handled by investment decay recalc)
        # Note: Investment decay doesn't trigger recalc above, only direct score decay does.
        # Maybe recalculation should happen *after* decay?
        # Let's recalculate ALL scores after decay. This is simpler.
        # print("Recalculating all scores after decay...")
        # for agent_id in list(self.agent_trust_scores.keys()):
        #      for dimension in self.dimensions:
        #           # self._recalculate_agent_trust_dimension(agent_id, dimension) # Recalc based on decayed investments
        #           pass # Score decay was applied directly above.


    def record_user_feedback(self, user_id: str, agent_id: int,
                            ratings: Dict[str, int], confidence: float = 0.9) -> None:
        """
        Record direct user feedback (e.g., 1-5 scale ratings) and adjust agent trust scores.

        Parameters:
        - user_id: String ID of the user source
        - agent_id: Integer ID of the agent being rated
        - ratings: Dict mapping dimension names (str) to integer ratings (e.g., 1-5)
        - confidence: Confidence in these ratings (0-1)
        """
        # Trust adjustment parameters (can be tuned)
        adjustment_factor = self.config.get('user_feedback_adjustment_factor', 0.05) # How much a single rating changes the score
        neutral_rating = (self.rating_scale + 1) / 2.0 # e.g., 3 for 1-5 scale

        affected_dimensions = []

        for dimension, rating in ratings.items():
            if dimension in self.dimensions:
                try:
                    rating_value = int(rating)
                    # Clamp rating to the expected scale
                    rating_value = max(1, min(self.rating_scale, rating_value))

                    old_score = self.agent_trust_scores[agent_id].get(dimension, 0.5)

                    # Calculate deviation from neutral rating
                    deviation = rating_value - neutral_rating

                    # Calculate adjustment amount, scaled by deviation and confidence
                    # Max deviation is (rating_scale - 1) / 2
                    max_deviation = (self.rating_scale - 1) / 2.0
                    normalized_deviation = deviation / max_deviation if max_deviation > 0 else 0
                    adjustment = adjustment_factor * normalized_deviation * confidence

                    new_score = old_score + adjustment
                    # Clamp new score between 0 and 1
                    new_score = max(0.0, min(1.0, new_score))

                    # Update score if changed significantly
                    if abs(new_score - old_score) > 0.001:
                        self.agent_trust_scores[agent_id][dimension] = new_score
                        affected_dimensions.append(dimension)

                        # Log the change
                        self.temporal_db['trust_scores'].append({
                            'evaluation_round': self.evaluation_round,
                            'timestamp': time.time(),
                            'agent_id': agent_id,
                            'dimension': dimension,
                            'old_score': old_score,
                            'new_score': new_score,
                            'change_source': 'user_feedback',
                            'source_id': user_id, # User providing the feedback
                            'raw_rating': rating_value
                        })

                except (ValueError, TypeError):
                    print(f"Warning: Invalid rating value '{rating}' for dimension '{dimension}' from user {user_id}. Skipping.")

        # Apply dimension correlations if enabled and scores changed
        if self.use_dimension_correlations and affected_dimensions:
            self.apply_dimension_correlations(agent_id, affected_dimensions)


    def apply_dimension_correlations(self, agent_id: int, changed_dimensions: List[str]) -> None:
        """
        Apply correlations between dimensions to propagate trust changes.
        (Implementation remains largely the same as before)
        """
        if not self.use_dimension_correlations or not changed_dimensions:
            return

        secondary_updates = defaultdict(float)
        correlation_strength = self.config.get('correlation_strength', 0.2) # Strength factor

        # Find the net change in the changed_dimensions during this round
        net_changes = defaultdict(float)
        for entry in self.temporal_db['trust_scores']:
             if (entry['agent_id'] == agent_id and
                 entry['evaluation_round'] == self.evaluation_round and
                 entry['dimension'] in changed_dimensions):
                  # This simple sum might double count if recalc happened after feedback in same round.
                  # Better: Find the *first* score at start of round and compare to *last* score.
                  # For simplicity now, approximate change based on direct source entries
                  if entry['change_source'] != 'correlation': # Avoid correlation feedback loops
                       net_changes[entry['dimension']] += (entry['new_score'] - entry['old_score'])


        # Propagate changes based on correlations
        for changed_dim, change_amount in net_changes.items():
             if abs(change_amount) > 0.01:
                 for target_dim in self.dimensions:
                      # Don't apply correlation to itself or back to one of the directly changed dims
                      if target_dim != changed_dim and target_dim not in changed_dimensions:
                          correlation = self.correlation_matrix.get((changed_dim, target_dim), 0.0)
                          if abs(correlation) > 0.1: # Apply only significant correlations
                              indirect_change = change_amount * correlation * correlation_strength
                              secondary_updates[target_dim] += indirect_change


        # Apply accumulated secondary updates
        for dimension, total_indirect_change in secondary_updates.items():
            if abs(total_indirect_change) > 0.005:
                old_score = self.agent_trust_scores[agent_id].get(dimension, 0.5)
                new_score = max(0.0, min(1.0, old_score + total_indirect_change))

                if abs(new_score - old_score) > 0.001:
                     self.agent_trust_scores[agent_id][dimension] = new_score
                     self.temporal_db['trust_scores'].append({
                          'evaluation_round': self.evaluation_round, 'timestamp': time.time(),
                          'agent_id': agent_id, 'dimension': dimension, 'old_score': old_score,
                          'new_score': new_score, 'change_source': 'correlation', 'source_id': None
                     })

    def update_agent_performance(self, agent_id: int, performance_scores: Dict[str, float]) -> None:
        """
        Record agent performance scores (e.g., normalized 0-1 feedback) for analysis.
        """
        timestamp = time.time()
        for dimension, score in performance_scores.items():
            if dimension in self.dimensions:
                 # Ensure score is 0-1
                 perf_score = max(0.0, min(1.0, score))
                 record = {
                     'evaluation_round': self.evaluation_round,
                     'timestamp': timestamp,
                     'agent_id': agent_id,
                     'dimension': dimension,
                     'performance_score': perf_score
                 }
                 self.agent_performance[agent_id][dimension].append(record)
                 self.temporal_db['agent_performance'].append(record)


    def increment_evaluation_round(self, increment: int = 1) -> None:
        """Increment the evaluation round counter and apply decay."""
        self.evaluation_round += increment
        # Apply trust decay at the beginning of each new round
        decay_rate = self.config.get('trust_decay_rate', 0.99)
        if decay_rate < 1.0:
             self.decay_trust_scores(decay_rate)
             # Note: apply_dimension_correlations is called within record_user_feedback and process_investments
             # if changes occur. No need to call it separately here unless decay itself should trigger correlations.


    def get_agent_trust(self, agent_id: int) -> Dict[str, float]:
        """Get current trust scores for an agent across all dimensions."""
        # Return a copy to prevent external modification
        return dict(self.agent_trust_scores.get(agent_id, {dim: 0.5 for dim in self.dimensions}))

    def get_agent_permissions(self, agent_id: int) -> Dict[str, float]:
        """
        Calculate agent permissions based on trust scores.
        (Implementation largely unchanged, but ensure dimensions match)
        """
        trust_scores = self.agent_trust_scores.get(agent_id, {})

        # Example permission mapping (customize as needed)
        permission_mapping = {
            'data_access': ['Transparency', 'Safety_Security', 'Process_Reliability'],
            'action_rights': ['Safety_Security', 'Value_Alignment', 'Problem_Resolution', 'Process_Reliability'],
            'routing_priority': ['Problem_Resolution', 'Communication_Quality', 'Adaptability'],
            'network_access': ['Manipulation_Resistance', 'Safety_Security']
        }

        permissions = {}
        for perm_type, mapped_dims in permission_mapping.items():
            valid_dims = [dim for dim in mapped_dims if dim in self.dimensions]
            scores = [trust_scores.get(dim, 0.5) for dim in valid_dims] # Use default 0.5 if score missing
            weights = [self.dimension_weights.get(dim, 1.0) for dim in valid_dims]

            if not scores or sum(weights) == 0:
                permissions[perm_type] = 0.0 # Default to 0 if no relevant dimensions or weights
                continue

            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            total_weight = sum(weights)

            # Normalize weighted average score (0-1) to a permission level (e.g., 0-10)
            permission_level = (weighted_sum / total_weight) * 10.0
            permissions[perm_type] = max(0.0, min(10.0, permission_level)) # Clamp 0-10

        return permissions


    def get_source_endorsements(self, source_id: str) -> List[Dict]:
        """
        Get all current positive investments (endorsements) made by a source.
        """
        endorsements = []
        if source_id in self.source_investments:
            for agent_id, dimensions in self.source_investments[source_id].items():
                for dimension, amount in dimensions.items():
                    if amount > 0.001: # Threshold for meaningful endorsement
                        endorsements.append({
                            'agent_id': agent_id,
                            'dimension': dimension,
                            'influence_amount': amount
                        })
        return endorsements

    # --- Visualization Methods ---
    # (Keep these as they were, ensuring they handle potential missing data)
    def _get_temporal_dataframe(self, db_key: str, agents=None, dimensions=None, sources=None,
                               start_round=None, end_round=None) -> Optional[pd.DataFrame]:
        """Helper to create and filter pandas DataFrame from temporal DB."""
        if not self.temporal_db.get(db_key):
             print(f"No data available in temporal_db['{db_key}'] for visualization.")
             return None

        try:
             df = pd.DataFrame(self.temporal_db[db_key])
        except Exception as e:
             print(f"Error creating DataFrame from {db_key}: {e}")
             return None

        if df.empty: return None

        # Apply filters
        if agents is not None and 'agent_id' in df.columns:
            df = df[df['agent_id'].isin(agents)]
        if dimensions is not None and 'dimension' in df.columns:
            df = df[df['dimension'].isin(dimensions)]
        if sources is not None and 'source_id' in df.columns:
             df = df[df['source_id'].isin(sources)]
        if start_round is not None and 'evaluation_round' in df.columns:
            df = df[df['evaluation_round'] >= start_round]
        if end_round is not None and 'evaluation_round' in df.columns:
            df = df[df['evaluation_round'] <= end_round]

        if df.empty:
            print("No data available with current filters.")
            return None

        # Ensure sorting
        sort_keys = ['evaluation_round', 'timestamp']
        valid_sort_keys = [k for k in sort_keys if k in df.columns]
        if valid_sort_keys:
             df = df.sort_values(valid_sort_keys)

        return df


    def visualize_trust_scores(self, agents=None, dimensions=None,
                              start_round=None, end_round=None):
        """Visualize trust scores over time."""
        if not PLOTTING_AVAILABLE:
             print("Plotting libraries not available. Skipping visualization.")
             return None

        df = self._get_temporal_dataframe('trust_scores', agents, dimensions, None, start_round, end_round)
        if df is None: return None

        # Need to pivot correctly - use 'new_score'
        pivot = df.pivot_table(
            index='evaluation_round',
            columns=['agent_id', 'dimension'],
            values='new_score',
            aggfunc='last' # Get the last recorded score for the round
        )
        # Forward fill to handle rounds where a score didn't change
        pivot = pivot.ffill()

        if pivot.empty:
             print("Pivot table is empty after processing. Cannot plot.")
             return None


        fig, ax = plt.subplots(figsize=(14, 8))
        num_lines = 0
        for col in pivot.columns:
             # Ensure column name is a tuple (agent_id, dimension)
             if isinstance(col, tuple) and len(col) == 2:
                  agent_id, dimension = col
                  label = f"A{agent_id}-{dimension[:4]}" # Short label
                  try:
                       pivot[col].plot(ax=ax, label=label, marker='.', linestyle='-')
                       num_lines += 1
                  except Exception as e:
                       print(f"Warning: Could not plot data for {col}: {e}")
             else:
                  print(f"Warning: Skipping unexpected column name in pivot table: {col}")


        if num_lines == 0:
             print("No valid data found to plot.")
             plt.close(fig) # Close the empty figure
             return None


        ax.set_xlabel('Evaluation Round')
        ax.set_ylabel('Trust Score')
        ax.set_title('Agent Trust Scores Over Time')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(0, 1) # Trust scores are 0-1

        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
        plt.show()
        return fig

    def visualize_source_performance(self, sources=None, dimensions=None,
                                   start_round=None, end_round=None):
        """Visualize source performance over time."""
        # This requires 'source_performance' data, which isn't currently generated.
        # Placeholder if you implement source performance tracking later.
        print("Source performance visualization not implemented in this version.")
        # if not PLOTTING_AVAILABLE:
        #      print("Plotting libraries not available.")
        #      return None
        # df = self._get_temporal_dataframe('source_performance', None, dimensions, sources, start_round, end_round)
        # if df is None: return None
        # # ... (Pivot and plot similar to visualize_trust_scores) ...
        return None
