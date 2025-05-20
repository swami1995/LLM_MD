import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns # Optional for visualization
import ipdb # Optional for debugging

class TrustMarket:
    """
    Core market system for tracking agent trust scores based on diverse inputs
    (direct feedback, comparisons, source investments).
    """
    def __init__(self, config: Dict[str, Any]):
        """Initialize the trust market."""
        self.config = config
        self.dimensions = config.get('dimensions', [ # Default dimensions if not provided
        "Factual_Correctness", "Process_Reliability", "Value_Alignment",
        "Communication_Quality", "Problem_Resolution", "Safety_Security",
        "Transparency", "Adaptability", "Trust_Calibration", "Manipulation_Resistance"
        ])
        self.dimension_weights = config.get('dimension_weights', {dim: 1.0 for dim in self.dimensions})

        # --- Core State ---
        # Agent scores: agent_id -> dimension -> score (0.0 to 1.0)
        self.agent_trust_scores = defaultdict(lambda: {dim: 0.5 for dim in self.dimensions})
        # Source investments: source_id -> agent_id -> dimension -> amount
        self.source_investments = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
        # Source capacity: source_id -> dimension -> total capacity
        self.source_influence_capacity = defaultdict(lambda: {dim: 0.0 for dim in self.dimensions}) # Start at 0, updated by add_source
        # Source available: source_id -> dimension -> available capacity
        self.source_available_capacity = defaultdict(lambda: {dim: 0.0 for dim in self.dimensions}) # Start at 0, updated by add_source
        # Source allocated: source_id -> dimension -> allocated capacity
        self.allocated_influence = defaultdict(lambda: {dim: 0.0 for dim in self.dimensions})

        # --- Configuration ---
        self.primary_sources = set(config.get('primary_sources', ['user_feedback', 'regulator'])) # Types considered primary
        self.primary_source_weight = config.get('primary_source_weight', 1.5) # Weight for primary source investments
        self.secondary_source_weight = config.get('secondary_source_weight', 1.0) # Weight for others
        self.rating_scale = config.get('rating_scale', 5) # Scale used in user feedback (e.g., 1-5)
        self.user_feedback_strength = config.get('user_feedback_strength', 0.05) # Impact of one rating
        self.comparative_feedback_strength = config.get('comparative_feedback_strength', 0.02) # Impact of one win/loss
        self.trust_decay_rate = config.get('trust_decay_rate', 0.99) # Rate per round towards neutral
        self.max_trust = config.get('max_trust', 1.0) # Maximum trust score
        self.min_trust = config.get('min_trust', 0.0) # Minimum trust score

        self.agent_amm_params = defaultdict(lambda: {'R': 0.0, 'T': 0.0, 'K': 0.0, 'total_supply': 0.0})
        # For simplicity, we won't track individual share ownership here, but assume sellers sell back to the AMM.
        self.amm_transactions_log = []

        # --- Oracle Configuration for AMM ---
        # Mechanism: 'adjust_treasury', 'adjust_reserve', 'oracle_trades' (oracle_trades not fully implemented here)
        self.oracle_influence_mechanisms = {'user_feedback': config.get('user_amm', 'adjust_reserve'), 'regulator': config.get('regulator_amm', 'adjust_reserve')}
        self.oracle_config = config.get('oracle_config', {
            'trust_threshold_low_treasury': 0.3,
            'trust_threshold_high_treasury': 0.7,
            'adjustment_amount_T_shares': 10.0, # Number of shares to mint/burn
            'trust_threshold_low_reserve': 0.3,
            'trust_threshold_high_reserve': 0.7,
            'adjustment_percentage_R': 0.05, # 5% change in R
            'min_R_after_adj': 1.0, # Minimum reserve after negative adjustment
            'min_T_after_adj': 1.0  # Minimum treasury shares after negative adjustment
        })
        # The way to do it would be : Trust score = Price = R/T
        # The AMM is a simple constant product market maker (CPMM) model.
        # The AMM uses a constant product formula: R * T = K, where K is a constant.
        # R is the reserve (liquidity) and T is the treasury (shares).
        # Thus, investments add to R and subtract from T using the following rule :
        # \text{cost}(q) = R_1 - R_0 = \frac{R_0\,T_0}{T_0-q} - R_0 = \frac{R_0\,q}{T_0 - q}}.
        # Likewise, divestments subtract from R and add to T using the following rule : 
        # \text{payout}(q) = R_0 - \frac{R_0\,T_0}{T_0+q} = \frac{R_0\,q}{T_0 + q}}.
        # where q is the amount of shares bought/sold by an investor.
        # Furthermore, the oracles, i.e, the user ratings and the regulator provide comparative feedback. 
        # The AMM model uses the feedback to adjust the trust scores of the agents depending on the oracle_influence_mechanisms. 
        # 'adjust_treasury' means that the AMM will adjust the treasury shares based on the feedback.
        # 'adjust_reserve' means that the AMM will adjust the reserve based on the feedback.
        

        # --- Performance & History ---
        self.agent_performance = defaultdict(lambda: defaultdict(list)) # For external perf metrics
        self.temporal_db = {
            'trust_scores': [], 'investments': [], 'source_performance': [], 'agent_performance': []
        }
        self.evaluation_round = 0

        # --- Optional Features ---
        self.use_dimension_correlations = config.get('use_dimension_correlations', False)
        if self.use_dimension_correlations:
            self.initialize_dimension_correlations()

        print("TrustMarket core initialized.")
        print(f"  - Rating Scale (for feedback normalization): {self.rating_scale}")
        print(f"  - User Feedback Strength: {self.user_feedback_strength}")
        print(f"  - Comparative Feedback Strength: {self.comparative_feedback_strength}")


    def initialize_dimension_correlations(self):
        """Initialize the correlation structure between trust dimensions."""
        # Keep existing logic
        default_correlations = {
            ('Factual_Correctness', 'Transparency'): 0.6, ('Communication_Quality', 'Problem_Resolution'): 0.5,
            ('Value_Alignment', 'Safety_Security'): 0.7, ('Trust_Calibration', 'Transparency'): 0.6,
            ('Factual_Correctness', 'Trust_Calibration'): 0.4, ('Process_Reliability', 'Safety_Security'): 0.5,
            ('Adaptability', 'Communication_Quality'): 0.3, ('Adaptability', 'Process_Reliability'): -0.2,
        }
        self.dimension_correlations = self.config.get('dimension_correlations', default_correlations)
        self.correlation_matrix = {}
        for dim1 in self.dimensions:
            for dim2 in self.dimensions:
                if dim1 == dim2: self.correlation_matrix[(dim1, dim2)] = 1.0
                else:
                    corr = self.dimension_correlations.get((dim1, dim2),
                        self.dimension_correlations.get((dim2, dim1), 0.0))
                    self.correlation_matrix[(dim1, dim2)] = corr
        print("  - Dimension correlations initialized.")


    def add_information_source(self, source_id: str, source_type: str,
                            initial_influence: Dict[str, float],
                            is_primary: bool = False) -> None:
        """
        Add a new information source to the market with initial influence capacity.
        """
        print(f"  Adding source {source_id} (Type: {source_type}, Primary: {is_primary})")
        if source_id in self.source_influence_capacity:
            print(f"  Warning: Source {source_id} already exists. Updating capacity.")

        # Initialize/Update influence for valid dimensions
        for dimension in self.dimensions:
            capacity = initial_influence.get(dimension, 0.0) # Get provided capacity or default to 0
            # Ensure capacity is non-negative
            capacity = max(0.0, capacity)

            # Only update if capacity is positive to avoid zeroing out existing
            if capacity > 0:
                self.source_influence_capacity[source_id][dimension] = capacity
                # Assume initially all capacity is available, none allocated
                self.source_available_capacity[source_id][dimension] = capacity
                self.allocated_influence[source_id][dimension] = 0.0 # Reset allocation on add/update
                print(f"    - Dim '{dimension}': Capacity = {capacity:.2f}")

        # Add to primary sources if specified OR if type matches config
        if is_primary or source_type in self.primary_sources:
            self.primary_sources.add(source_id)
            print(f"    - Source {source_id} marked as primary.")


    def process_investments(self, source_id: str, investments: List[Tuple]) -> None:
        """
        Process a batch of investment/divestment decisions from a source.
        Updates source investments, allocated/available capacity, and triggers agent trust recalculation.

        Parameters:
        - source_id: The source making the investment decisions
        - investments: List of (agent_id, dimension, amount, confidence) tuples
                    Amount > 0 is investment, Amount < 0 is divestment.
                    Confidence is currently logged but not used in core calculation.
        """
        if source_id not in self.source_influence_capacity:
            print(f"Warning: Source {source_id} not registered. Ignoring investments.")
            return

        affected_agents_dimensions = set() # Track (agent_id, dimension) pairs needing recalc

        print(f"  Processing {len(investments)} investments from source {source_id}...")
        
        old_agent_trust_scores = {dimension: {agent_id: self.agent_trust_scores[agent_id][dimension] for agent_id in self.agent_trust_scores} for dimension in self.dimensions}

        # old_agent_trust_scores = {dimension: {} for dimension in self.dimensions}
        # for dimension in self.dimensions:
        #     if dimension not in self.source_influence_capacity[source_id]:
        #         continue
        #     old_agent_trust_scores[dimension] = {agent_id: sum([self.source_investments[s][agent_id][dimension] for s in self.source_investments]) for agent_id in self.source_investments[source_id]}
            
        # ipdb.set_trace()
        for agent_id, dimension, amount, confidence in investments:
            if dimension not in self.dimensions:
                # print(f"    Warning: Invalid dimension '{dimension}' for agent {agent_id}. Skipping.")
                continue
            if dimension not in self.source_influence_capacity[source_id]:
                print(f"    Warning: Source {source_id} has no capacity for dimension '{dimension}'. Skipping.")
                continue

            current_investment = self.source_investments[source_id][agent_id].get(dimension, 0.0)
            available_cap = self.source_available_capacity[source_id].get(dimension, 0.0)
            allocated_cap = self.allocated_influence[source_id].get(dimension, 0.0)
            # Total capacity should remain constant unless explicitly changed
            # total_cap = self.source_influence_capacity[source_id].get(dimension, 0.0)

            change_amount = 0.0

            # --- Handle Divestment (amount < 0) ---
            if amount < 0:
                # Amount to divest is the absolute value, capped by current investment
                divest_request = abs(amount)
                actual_divestment = min(divest_request, current_investment)

                if actual_divestment > 0.001: # Threshold to avoid tiny changes
                    # print(f"    Divested {-change_amount:.3f} from Agent {agent_id} on {dimension}")
                    # There are also market scores that we need to account for. not just investments. 
                    
                    old_agent_id_trust_score = self.source_investments[source_id][agent_id][dimension] 
                    self.source_investments[source_id][agent_id][dimension] -= actual_divestment
                    new_agent_trust_scores = sum([self.source_investments[s][agent_id][dimension] for s in self.source_investments])
                    for other_agent_id in self.source_investments[source_id]:
                        if self.source_investments[source_id][other_agent_id][dimension] > 0:
                            # Redistribute divested influence to other agents
                            self.source_investments[source_id][other_agent_id][dimension] = (self.source_investments[source_id][other_agent_id][dimension]) * new_agent_trust_scores / old_agent_trust_scores[dimension][agent_id]
                    new_agent_id_trust_score = self.source_investments[source_id][agent_id][dimension]
                    eventual_reduction_amount = (old_agent_id_trust_score - new_agent_id_trust_score)
                    self.allocated_influence[source_id][dimension] -= eventual_reduction_amount
                    self.source_available_capacity[source_id][dimension] += actual_divestment
                    affected_agents_dimensions.add((agent_id, dimension))
                    
                # else: print(f"    Divestment for Agent {agent_id} on {dimension} too small or none possible.")

            # --- Handle Investment (amount > 0) ---
            else:
                # Amount to invest is capped by available capacity
                actual_investment = min(amount, available_cap)

                if actual_investment > 0.001: # Threshold
                    old_agent_id_trust_score = self.source_investments[source_id][agent_id][dimension]
                    self.source_investments[source_id][agent_id][dimension] += actual_investment
                    new_agent_trust_score = sum([self.source_investments[s][agent_id][dimension] for s in self.source_investments])
                    ipdb.set_trace()

                    for other_agent_id in self.source_investments[source_id]:
                        if self.source_investments[source_id][other_agent_id][dimension] > 0:
                            # Redistribute invested influence to other agents
                            # ipdb.set_trace()
                            self.source_investments[source_id][other_agent_id][dimension] = (self.source_investments[source_id][other_agent_id][dimension]) * new_agent_trust_score / old_agent_trust_scores[dimension][agent_id]

                    new_agent_id_trust_score = self.source_investments[source_id][agent_id][dimension]
                    eventual_allocation_amount = (new_agent_id_trust_score - old_agent_id_trust_score)
                    self.allocated_influence[source_id][dimension] += eventual_allocation_amount
                    self.source_available_capacity[source_id][dimension] -= actual_investment
                    
                    # print(f"    Invested {change_amount:.3f} in Agent {agent_id} on {dimension}")
                    affected_agents_dimensions.add((agent_id, dimension))
                # else: print(f"    Investment for Agent {agent_id} on {dimension} too small or no capacity.")

            # Log the actual change (investment or divestment)
            if abs(change_amount) > 0.001:
                self.temporal_db['investments'].append({
                    'evaluation_round': self.evaluation_round, 'timestamp': time.time(),
                    'source_id': source_id, 'agent_id': agent_id, 'dimension': dimension,
                    'amount': change_amount, # Positive for invest, negative for divest
                    'confidence': confidence,
                    'type': 'investment' if change_amount > 0 else 'divestment'
                })

            # Sanity check: available + allocated should ideally equal total capacity
            # current_total_check = self.source_available_capacity[source_id][dimension] + self.allocated_influence[source_id][dimension]
            # if abs(current_total_check - total_cap) > 0.01:
            #      print(f"    Capacity mismatch warning for {source_id}, {dimension}: Available={self.source_available_capacity[source_id][dimension]:.2f} + Allocated={self.allocated_influence[source_id][dimension]:.2f} != Total={total_cap:.2f}")
            self.source_influence_capacity[source_id][dimension] = self.allocated_influence[source_id][dimension] + self.source_available_capacity[source_id][dimension]


        # --- Recalculate trust scores for affected agent/dimension pairs ---
        if affected_agents_dimensions:
            print(f"  Recalculating trust for {len(affected_agents_dimensions)} agent/dimension pairs...")
            affected_agents = set(ag_id for ag_id, dim in affected_agents_dimensions)
            for agent_id in affected_agents:
                changed_in_recalc = set()
                for dimension in self.dimensions:
                    if (agent_id, dimension) in affected_agents_dimensions:
                        new_score = sum([self.source_investments[source_id][agent_id].get(dimension, 0.0) for source_id in self.source_investments]) #self._recalculate_agent_trust_dimension(agent_id, dimension)
                        old_score = self.agent_trust_scores[agent_id][dimension]
                        if abs(new_score - old_score) > 0.001:
                            self.agent_trust_scores[agent_id][dimension] = new_score
                            changed_in_recalc.add(dimension)
                            # Log score change due to investment
                            self.temporal_db['trust_scores'].append({
                                'evaluation_round': self.evaluation_round, 'timestamp': time.time(),
                                'agent_id': agent_id, 'dimension': dimension,
                                'old_score': old_score, 'new_score': new_score,
                                'change_source': 'investment', 'source_id': None # Aggregate effect
                            })

                # Apply correlations if enabled and scores actually changed
                if self.use_dimension_correlations and changed_in_recalc:
                    # print(f"    Applying correlations for Agent {agent_id} due to changes in: {changed_in_recalc}")
                    self.apply_dimension_correlations(agent_id, list(changed_in_recalc))


    def _recalculate_agent_trust_dimension(self, agent_id, dimension) -> float:
        """
        Recalculates the trust score for a specific agent and dimension based on
        current investments from all sources.

        Uses weighted sum based on source primary status and total capacity invested.
        """
        total_weighted_investment = 0.0
        total_capacity_investing = 0.0 # Sum of capacities of sources with >0 investment

        for source_id, agent_investments in self.source_investments.items():
            investment_amount = agent_investments.get(agent_id, {}).get(dimension, 0.0)

            if investment_amount > 0.001: # Only consider sources actively investing
                source_capacity = self.source_influence_capacity.get(source_id, {}).get(dimension, 0.0)
                if source_capacity > 0: # Ensure source has capacity in this dimension
                    # Apply weighting based on primary status
                    weight = self.primary_source_weight if source_id in self.primary_sources else self.secondary_source_weight
                    total_weighted_investment += investment_amount * weight
                    total_capacity_investing += source_capacity * weight # Weight capacity too? Or just investment? Let's weight capacity.

        # Calculate final score
        if total_capacity_investing > 0:
            # Normalize weighted investment by total weighted capacity of active investors
            new_score = total_weighted_investment / total_capacity_investing
        else:
            # No positive investments, default to a neutral/low score
            new_score = 0.5 # Default to neutral if no one is investing

        # Clamp score between 0 and 1
        return min(1.0, max(0.0, new_score))


    def decay_trust_scores(self) -> None:
        """
        Apply temporal decay to all agent trust scores, moving them towards neutral (0.5).
        """
        decay_rate = self.trust_decay_rate
        for dimension in self.dimensions:
            for agent_id in self.agent_trust_scores:
                # Apply decay to trust scores
                self.agent_trust_scores[agent_id][dimension] *= decay_rate

            for source_id in self.source_investments:
                self.allocated_influence[source_id][dimension] *= decay_rate
                self.source_available_capacity[source_id][dimension] *= decay_rate
                for agent_id in self.source_investments[source_id]:
                    self.source_investments[source_id][agent_id][dimension] *= decay_rate


    def record_user_feedback(self, user_id: str, agent_id: int,
                            ratings: Dict[str, int], confidence: float = 0.8) -> None:
        """
        Records direct user feedback (specific ratings) and updates agent trust scores.
        Normalizes ratings based on self.rating_scale.

        Parameters:
        - user_id: ID of the user providing feedback
        - agent_id: Agent being rated
        - ratings: Dict mapping dimensions to integer ratings (e.g., 1-5)
        - confidence: Confidence in these ratings (optional, currently fixed)
        """
        if not ratings: return

        # print(f"  Recording user feedback from {user_id} for agent {agent_id}...")
        neutral_rating = (self.rating_scale + 1) / 2.0
        max_deviation = self.rating_scale - neutral_rating
        if max_deviation <= 0:
            print(f"  Warning: Invalid rating scale {self.rating_scale}. Cannot process feedback.")
            return

        affected_dimensions = set()
        for dimension, rating in ratings.items():
            if dimension not in self.dimensions: continue

            try:
                # Ensure rating is int, clamp to scale
                rating_val = max(1, min(self.rating_scale, int(rating)))

                # Normalize change to -1 to +1 range
                normalized_score_change = (rating_val - neutral_rating) / max_deviation

                # Calculate adjustment based on strength config
                adjustment = normalized_score_change * confidence * self.user_feedback_strength

                # Apply adjustment to current score
                old_score = self.agent_trust_scores[agent_id].get(dimension, 0.5)
                new_score = min(self.max_trust, max(self.min_trust, old_score + adjustment))

                if abs(new_score - old_score) > 0.001: # Record if changed significantly
                    self.agent_trust_scores[agent_id][dimension] = new_score
                    affected_dimensions.add(dimension)
                    # Log change
                    self.temporal_db['trust_scores'].append({
                        'evaluation_round': self.evaluation_round, 'timestamp': time.time(),
                        'agent_id': agent_id, 'dimension': dimension,
                        'old_score': old_score, 'new_score': new_score,
                        'change_source': 'user_feedback', 'source_id': user_id
                    })
                    # print(f"    Dim '{dimension}': {old_score:.3f} -> {new_score:.3f} (Rating: {rating_val})")

            except (ValueError, TypeError) as e:
                print(f"    Warning: Invalid rating value '{rating}' for dimension '{dimension}'. Skipping. Error: {e}")

        # Apply correlations if enabled and scores changed
        if self.use_dimension_correlations and affected_dimensions:
            # print(f"    Applying correlations for Agent {agent_id} due to user feedback.")
            self.apply_dimension_correlations(agent_id, list(affected_dimensions))


    def record_comparative_feedback(self, agent_a_id: int, agent_b_id: int,
                                winners: Dict[str, str]) -> None:
        """
        Records comparative user feedback and updates agent trust scores.

        Parameters:
        - agent_a_id: ID of the first agent compared
        - agent_b_id: ID of the second agent compared
        - winners: Dict mapping dimension -> 'A', 'B', or 'Tie'
        """
        if not winners: return

        # print(f"  Recording comparative feedback for agents {agent_a_id} vs {agent_b_id}...")
        adjustment = self.comparative_feedback_strength # Use configured strength
        affected_dimensions_a = set()
        affected_dimensions_b = set()
        
        for dimension, winner in winners.items():
            if dimension not in self.dimensions: continue

            score_a = self.agent_trust_scores[agent_a_id].get(dimension, 0.5)
            score_b = self.agent_trust_scores[agent_b_id].get(dimension, 0.5)
            new_a, new_b = score_a, score_b # Start with current scores

            if winner == 'A':
                new_a = min(self.max_trust, score_a + adjustment)
                new_b = max(self.min_trust, score_b - adjustment)
                # print('A wins')
                print(f"    Dim '{dimension}': A wins ({score_a:.3f}->{new_a:.3f}, {score_b:.3f}->{new_b:.3f})")
            elif winner == 'B':
                new_a = max(self.min_trust, score_a - adjustment)
                new_b = min(self.max_trust, score_b + adjustment)
                # print('B wins')
                print(f"    Dim '{dimension}': B wins ({score_a:.3f}->{new_a:.3f}, {score_b:.3f}->{new_b:.3f})")
            # Else (Tie): new_a, new_b remain unchanged
            else:
                print(f"    Dim '{dimension}': Tie (no change)")

            # Update if changed significantly
            if abs(new_a - score_a) > 0.001:
                self.agent_trust_scores[agent_a_id][dimension] = new_a
                affected_dimensions_a.add(dimension)
                self.temporal_db['trust_scores'].append({
                    'evaluation_round': self.evaluation_round, 'timestamp': time.time(),
                    'agent_id': agent_a_id, 'dimension': dimension,
                    'old_score': score_a, 'new_score': new_a,
                    'change_source': 'comparative_feedback', 'source_id': f"vs_{agent_b_id}"
                })
            if abs(new_b - score_b) > 0.001:
                self.agent_trust_scores[agent_b_id][dimension] = new_b
                affected_dimensions_b.add(dimension)
                self.temporal_db['trust_scores'].append({
                    'evaluation_round': self.evaluation_round, 'timestamp': time.time(),
                    'agent_id': agent_b_id, 'dimension': dimension,
                    'old_score': score_b, 'new_score': new_b,
                    'change_source': 'comparative_feedback', 'source_id': f"vs_{agent_a_id}"
                })
                
        # Apply correlations if enabled
        if self.use_dimension_correlations:
            if affected_dimensions_a:
                # print(f"    Applying correlations for Agent {agent_a_id} due to comparison.")
                self.apply_dimension_correlations(agent_a_id, list(affected_dimensions_a))
            if affected_dimensions_b:
                # print(f"    Applying correlations for Agent {agent_b_id} due to comparison.")
                self.apply_dimension_correlations(agent_b_id, list(affected_dimensions_b))


    def apply_dimension_correlations(self, agent_id: int, changed_dimensions: List[str]) -> None:
        """Apply correlations between dimensions to propagate trust changes."""
        # Keep existing logic, ensure it uses self.correlation_matrix
        if not self.use_dimension_correlations or not changed_dimensions:
            return

        secondary_updates = defaultdict(float)
        correlation_strength = self.config.get('correlation_strength', 0.15) # Maybe tune this

        # Find the net change in the triggering round for each changed dimension
        latest_changes = {}
        for dim in changed_dimensions:
            net_change = 0.0
            # Look backwards in the temporal DB for changes THIS round
            for entry in reversed(self.temporal_db['trust_scores']):
                if entry['evaluation_round'] < self.evaluation_round: break # Stop if we go to previous round
                if entry['agent_id'] == agent_id and entry['dimension'] == dim:
                    # This logic is flawed - multiple changes can occur. Need a simpler way.
                    # Let's use the difference from the *start* of the round score.
                    # This requires storing start-of-round scores, which is complex.
                    # Alternative: Just use the magnitude of the *last* change this round.
                    last_change_entry = None
                    for e in reversed(self.temporal_db['trust_scores']):
                        if e['evaluation_round'] == self.evaluation_round and e['agent_id'] == agent_id and e['dimension'] == dim:
                            last_change_entry = e
                            break
                    if last_change_entry:
                            net_change = last_change_entry['new_score'] - last_change_entry['old_score']
                            latest_changes[dim] = net_change
                    break # Found last change for this dim

        # Apply correlations based on the latest net changes
        for changed_dim, change_amount in latest_changes.items():
            if abs(change_amount) < 0.01: continue

            for target_dim in self.dimensions:
                # Don't apply correlation to self or back to the dimensions that just changed directly
                if target_dim != changed_dim and target_dim not in changed_dimensions:
                    correlation = self.correlation_matrix.get((changed_dim, target_dim), 0)

                    if abs(correlation) > 0.1: # Threshold
                        indirect_change = change_amount * correlation * correlation_strength
                        secondary_updates[target_dim] += indirect_change


        # Apply accumulated secondary updates
        for dimension, change in secondary_updates.items():
            if abs(change) > 0.005: # Threshold
                old_score = self.agent_trust_scores[agent_id].get(dimension, 0.5)
                new_score = min(1.0, max(0.0, old_score + change))

                if abs(new_score - old_score) > 0.001:
                    self.agent_trust_scores[agent_id][dimension] = new_score
                    # Log correlation-based change
                    self.temporal_db['trust_scores'].append({
                        'evaluation_round': self.evaluation_round, 'timestamp': time.time(),
                        'agent_id': agent_id, 'dimension': dimension,
                        'old_score': old_score, 'new_score': new_score,
                        'change_source': 'correlation', 'source_id': None
                    })
                    # print(f"      Correlation applied to Dim '{dimension}': {old_score:.3f} -> {new_score:.3f}")


    def update_agent_performance(self, agent_id: int, performance_scores: Dict[str, float]) -> None:
        """Record external agent performance metrics."""
        # Keep existing logic
        for dimension, score in performance_scores.items():
            if dimension in self.dimensions:
                perf_entry = {
                    'evaluation_round': self.evaluation_round, 'timestamp': time.time(),
                    'score': score # Store the provided performance score
                }
                self.agent_performance[agent_id][dimension].append(perf_entry)
                # Log in temporal DB as well
                self.temporal_db['agent_performance'].append({
                    'evaluation_round': self.evaluation_round, 'timestamp': time.time(),
                    'agent_id': agent_id, 'dimension': dimension, 'performance_score': score
                })


    def increment_evaluation_round(self, increment: int = 1) -> None:
        """Increment the evaluation round and trigger periodic actions."""
        self.evaluation_round += increment
        # Apply decay first
        self.decay_trust_scores()
        # Then apply correlations (optional) - correlations act on scores *after* decay
        # if self.use_dimension_correlations:
        #      print(f"  Applying correlations post-decay for round {self.evaluation_round}...")
        #      for agent_id in list(self.agent_trust_scores.keys()):
        #           # Need a way to know which dimensions *potentially* changed due to decay
        #           # Simplest: apply to all dimensions if any score is not neutral.
        #           all_dims = list(self.dimensions)
        #           self.apply_dimension_correlations(agent_id, all_dims) # Over-applying maybe?


    def get_agent_trust(self, agent_id: int) -> Dict[str, float]:
        """Get current trust scores for an agent, returning a copy."""
        # Return a copy to prevent external modification
        return dict(self.agent_trust_scores.get(agent_id, {dim: 0.5 for dim in self.dimensions}))


    def get_agent_permissions(self, agent_id: int) -> Dict[str, float]:
        """Calculate permissions based on trust scores."""
        # Keep existing logic, ensure it uses self.agent_trust_scores[agent_id]
        trust_scores = self.agent_trust_scores.get(agent_id, {dim: 0.5 for dim in self.dimensions}) # Use .get with default

        permission_mapping = { # Example mapping
            'data_access': ['Factual_Correctness', 'Transparency', 'Safety_Security'],
            'action_rights': ['Safety_Security', 'Value_Alignment', 'Process_Reliability'],
            'routing_priority': ['Problem_Resolution', 'Communication_Quality', 'Adaptability'],
            'network_access': ['Manipulation_Resistance', 'Process_Reliability']
        }

        permissions = {}
        for perm_type, dims in permission_mapping.items():
            scores = [trust_scores.get(dim, 0.5) for dim in dims if dim in self.dimensions]
            weights = [self.dimension_weights.get(dim, 1.0) for dim in dims if dim in self.dimensions]

            if not scores or sum(weights) == 0:
                permissions[perm_type] = 0.0
                continue

            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            total_weight = sum(weights)

            # Normalize score (0-1) and scale (e.g., 0-10)
            normalized_score = weighted_sum / total_weight if total_weight > 0 else 0.0
            permissions[perm_type] = min(10.0, max(0.0, normalized_score * 10.0))

        return permissions


    def get_source_endorsements(self, source_id: str) -> List[Dict]:
        """Get all current positive investments made by a source."""
        # Keep existing logic
        endorsements = []
        if source_id in self.source_investments:
            for agent_id, dimensions in self.source_investments[source_id].items():
                for dimension, amount in dimensions.items():
                    if amount > 0.001: # Threshold for significance
                        endorsements.append({
                            'agent_id': agent_id, 'dimension': dimension,
                            'influence_amount': amount
                        })
        return endorsements

    def summarize_market_state(self, information_sources):
        """Get a summary of the current market state."""
        # Keep existing logic, ensure calculations handle missing agents/dims gracefully
        agent_scores_summary = {}
        all_agent_ids = list(self.agent_trust_scores.keys())
        for agent_id in all_agent_ids:
            agent_scores_summary[agent_id] = self.get_agent_trust(agent_id) # Use getter for default handling

        dimension_averages = defaultdict(lambda: {'sum': 0.0, 'count': 0})
        for agent_id, scores in agent_scores_summary.items():
            for dimension, score in scores.items():
                if dimension in self.dimensions:
                    dimension_averages[dimension]['sum'] += score
                    dimension_averages[dimension]['count'] += 1

        avg_scores = {
            dim: data['sum'] / data['count'] if data['count'] > 0 else None
            for dim, data in dimension_averages.items()
        }

        source_influence_summary = {}
        all_source_ids = list(self.source_influence_capacity.keys())
        for source_id in all_source_ids:
            source_influence_summary[source_id] = {
                "type": next((s.source_type for s_id, s in information_sources.items() if s_id == source_id), "Unknown"), # Need access to system's info sources
                "capacity": dict(self.source_influence_capacity.get(source_id, {})),
                "allocated": dict(self.allocated_influence.get(source_id, {})),
                "available": dict(self.source_available_capacity.get(source_id, {}))
            }

        investment_counts = defaultdict(lambda: defaultdict(int))
        for source_id, agent_investments in self.source_investments.items():
            for agent_id, dimensions in agent_investments.items():
                for dimension, amount in dimensions.items():
                    if amount > 0.001:
                        investment_counts[agent_id][dimension] += 1

        return {
            "market_evaluation_round": self.evaluation_round,
            "num_agents_in_market": len(all_agent_ids),
            "dimension_averages": avg_scores,
            "source_influence": source_influence_summary,
            "investment_counts": dict(investment_counts)
            # Could add total invested influence per dimension, etc.
        }

    # --- Visualization Methods ---
    # Keep existing visualization logic. Add error handling for missing libraries.
    def visualize_trust_scores(self, agents=None, dimensions=None,
                            start_round=None, end_round=None):
        """Visualize trust scores over time."""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
        except ImportError:
            print("Warning: matplotlib or pandas not installed. Cannot visualize trust scores.")
            return None

        if not self.temporal_db['trust_scores']:
            print("No trust score data available for visualization")
            return None

        df = pd.DataFrame(self.temporal_db['trust_scores'])
        # (Keep filtering logic as before)
        if agents: df = df[df['agent_id'].isin(agents)]
        if dimensions: df = df[df['dimension'].isin(dimensions)]
        if start_round is not None: df = df[df['evaluation_round'] >= start_round]
        if end_round is not None: df = df[df['evaluation_round'] <= end_round]

        if df.empty:
            print("No data available with current filters for trust score visualization")
            return None

        df = df.sort_values(['evaluation_round', 'timestamp'])
        pivot = df.pivot_table(index='evaluation_round', columns=['agent_id', 'dimension'], values='new_score', aggfunc='last').ffill() # Forward fill missing values for plotting

        fig, ax = plt.subplots(figsize=(14, 8))
        for col in pivot.columns:
            label = f"A{col[0]}-{col[1][:4]}" # Shorten label
            pivot[col].plot(ax=ax, label=label, marker='.', linestyle='-', markersize=4)

        ax.set_xlabel('Evaluation Round')
        ax.set_ylabel('Trust Score')
        ax.set_title('Agent Trust Scores Over Time')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(0, 1.05) # Ensure y-axis is 0-1
        plt.tight_layout()
        plt.show()
        return fig

    def visualize_source_performance(self, sources=None, dimensions=None,
                                start_round=None, end_round=None):
        """Visualize source performance over time."""
        # Keep existing logic, add similar import checks and robustness
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
        except ImportError:
            print("Warning: matplotlib or pandas not installed. Cannot visualize source performance.")
            return None

        if not self.temporal_db['source_performance']:
            print("No source performance data available for visualization")
            return None

        df = pd.DataFrame(self.temporal_db['source_performance'])
        # (Keep filtering logic as before)
        if sources: df = df[df['source_id'].isin(sources)]
        if dimensions: df = df[df['dimension'].isin(dimensions)]
        if start_round is not None: df = df[df['evaluation_round'] >= start_round]
        if end_round is not None: df = df[df['evaluation_round'] <= end_round]

        if df.empty:
            print("No data available with current filters for source performance visualization")
            return None

        df = df.sort_values(['evaluation_round', 'timestamp'])
        pivot = df.pivot_table(index='evaluation_round', columns=['source_id', 'dimension'], values='performance_score', aggfunc='mean').ffill() # Use mean performance in round

        fig, ax = plt.subplots(figsize=(14, 8))
        for col in pivot.columns:
            label = f"{col[0]}-{col[1][:4]}"
            pivot[col].plot(ax=ax, label=label, marker='.', linestyle='-', markersize=4)

        ax.set_xlabel('Evaluation Round')
        ax.set_ylabel('Performance Score')
        ax.set_title('Information Source Performance Over Time')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        return fig