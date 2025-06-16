import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns # Optional for visualization
import ipdb # Optional for debugging
import math
import os

# Trust markets are a way to combine curation, recommendation and scoring systems in a reputation system.
# They are used to track the trustworthiness of agents based on various inputs, such as user feedback, regulator ratings, and comparative feedback.
# The trust market uses an automated market maker (AMM) model to adjust the trust scores of agents based on the investments made by sources.
# The AMM is a simple constant product market maker (CPMM) model.
# The AMM model uses a constant product formula: R * T = K, where K is a constant.
# R is the reserve (liquidity) and T is the treasury (shares). 
# Trust score = Price = R/T
# Thus, investments add to R and subtract from T using the following rule :
# \text{cost}(q) = R_1 - R_0 = \frac{R_0\,T_0}{T_0-q} - R_0 = \frac{R_0\,q}{T_0 - q}}.
# Likewise, divestments subtract from R and add to T using the following rule : 
# \text{payout}(q) = R_0 - \frac{R_0\,T_0}{T_0+q} = \frac{R_0\,q}{T_0 + q}}.
# where q is the amount of shares bought/sold by an investor.
# Furthermore, the oracles, i.e, the user ratings and the regulator provide comparative feedback. 
# The AMM model uses the feedback to adjust the trust scores of the agents depending on the oracle_influence_mechanisms. 
# 'adjust_treasury' means that the AMM will adjust the treasury shares based on the feedback.
# 'adjust_reserve' means that the AMM will adjust the reserve based on the feedback.

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
        self.user_feedback_strength = config.get('user_feedback_strength', 0.02) # Impact of one rating
        self.comparative_feedback_strength = config.get('comparative_feedback_strength', 0.01) # Impact of one win/loss
        self.trust_decay_rate = config.get('trust_decay_rate', 0.99) # Rate per round towards neutral
        self.max_trust = config.get('max_trust', 1.0) # Maximum trust score
        self.min_trust = config.get('min_trust', 0.0) # Minimum trust score
        self.primary_source_update_type = config.get('primary_source_update_type', 'fix_K') # Type of update for primary source

        self.agent_amm_params = defaultdict(lambda: {dim:{'R': 0.0, 'T': 0.0, 'K': 0.0, 'total_supply': 0.0} for dim in self.dimensions})
        # For simplicity, we won't track individual share ownership here, but assume sellers sell back to the AMM.
        self.amm_transactions_log = []

        # --- Oracle Configuration for AMM ---
        # Mechanism: 'adjust_treasury', 'adjust_reserve', 'oracle_trades' # defaulting to 'adjust_reserve' for now. Others are not implemented yet.
        self.oracle_influence_mechanisms = {'user_feedback': config.get('user_amm', 'adjust_reserve'), 'regulator': config.get('regulator_amm', 'adjust_reserve')}

        # --- Performance & History ---
        self.agent_performance = defaultdict(lambda: defaultdict(list)) # For external perf metrics
        self.temporal_db = {
            'trust_scores': [], 'investments': [], 
            'source_states': [], 'agent_performance': []
        }
        self.evaluation_round = 0

        # --- Optional Features ---
        self.use_dimension_correlations = config.get('use_dimension_correlations', False)
        if self.use_dimension_correlations:
            self.initialize_dimension_correlations()

        # --- Regulator Support ---
        self.cumulative_user_influence = defaultdict(lambda: defaultdict(float))    # Tracks sum of |delta_R| from user feedback

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
        if source_id in self.source_available_capacity:
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
        
    def ensure_agent_dimension_initialized_in_amm(self, agent_id: str, dimension: str) -> None:
        """
        Ensure the agent and dimension are initialized in the AMM.
        This is important for AMM operations to work correctly.
        """
        if agent_id not in self.agent_amm_params or (self.agent_amm_params[agent_id][dimension]['R'] == 0 and self.agent_amm_params[agent_id][dimension]['T'] == 0):
            # Initialize AMM parameters for this agent and dimension
            self.agent_amm_params[agent_id][dimension]['R'] = self.config.get('initial_R_oracle', 10.0)
            self.agent_amm_params[agent_id][dimension]['T'] = self.config.get('initial_T_oracle', 20.0)
            self.agent_amm_params[agent_id][dimension]['K'] = self.agent_amm_params[agent_id][dimension]['R'] * self.agent_amm_params[agent_id][dimension]['T']
            self.agent_amm_params[agent_id][dimension]['total_supply'] = self.agent_amm_params[agent_id][dimension]['T']
            # Initialize trust score from AMM price
            self.agent_trust_scores[agent_id][dimension] = self.agent_amm_params[agent_id][dimension]['R'] / self.agent_amm_params[agent_id][dimension]['T']


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
        if source_id not in self.source_available_capacity and source_id not in self.oracle_influence_mechanisms:
            print(f"Warning: Source {source_id} not registered. Ignoring investments.")
            return

        affected_agents_dimensions = defaultdict(list) # Track (agent_id, dimension) pairs needing recalc

        print(f"  Processing {len(investments)} investments from source {source_id}...")
        
        # old_agent_trust_scores = {dimension: {agent_id: self.agent_trust_scores[agent_id][dimension] for agent_id in self.agent_trust_scores} for dimension in self.dimensions}

        # ipdb.set_trace()
        for agent_id, dimension, amount, confidence in investments:
            # ipdb.set_trace()
            if dimension not in self.dimensions:
                # print(f"    Warning: Invalid dimension '{dimension}' for agent {agent_id}. Skipping.")
                continue
            if dimension not in self.source_available_capacity[source_id]:
                print(f"    Warning: Source {source_id} has no capacity for dimension '{dimension}'. Skipping.")
                continue

            if source_id not in self.oracle_influence_mechanisms:
                current_investment = self.source_investments[source_id][agent_id].get(dimension, 0.0)
                available_cap = self.source_available_capacity[source_id].get(dimension, 0.0)

            change_amount = 0.0

            # --- Handle Divestment (amount < 0) ---
            if amount < 0:
                # Amount to divest is the absolute value, capped by current investment
                divest_request = abs(amount)
                if source_id not in self.oracle_influence_mechanisms:
                    actual_divestment = min(divest_request, current_investment)
                else:
                    actual_divestment = divest_request

                if actual_divestment > 0.001: # Threshold to avoid tiny changes
                    change_amount = -actual_divestment # Record the change
                    y = actual_divestment
                    agent_amm_params = self.agent_amm_params[agent_id][dimension]
                    T, K = agent_amm_params['T'], agent_amm_params['K']
                    R = K/T
                    old_price = R/T
                    y = min(y, R) # Don't divest more than the reserve
                    if source_id in self.oracle_influence_mechanisms and self.primary_source_update_type == 'fix_T':
                        R_new = R - y
                        T_new = T
                    else:
                        num_shares_to_divest = y*T/(R - y)
                        T_new = T + num_shares_to_divest
                        R_new = R*T/T_new #= R- y  # = R*T/(T + num_shares_to_divest) = R*T/(T + y*T/(R - y)) = R*T*(R-y)/(R*T) = R - y
                    new_price = R_new/T_new

                    self.agent_amm_params[agent_id][dimension]['R'] = R_new
                    self.agent_amm_params[agent_id][dimension]['T'] = T_new
                    self.agent_amm_params[agent_id][dimension]['K'] = R_new * T_new

                    # Update investments
                    if source_id not in self.oracle_influence_mechanisms:
                        old_source_investment = self.source_investments[source_id][agent_id].get(dimension, 0.0)
                        self.source_investments[source_id][agent_id][dimension] -= num_shares_to_divest
                        self.allocated_influence[source_id][dimension] += self.source_investments[source_id][agent_id][dimension]*new_price - old_source_investment*old_price
                        self.source_available_capacity[source_id][dimension] += actual_divestment
                    for s_id in self.source_investments:
                        if self.source_investments[s_id][agent_id][dimension] > 0 and s_id != source_id:
                            self.allocated_influence[s_id][dimension] -= self.source_investments[s_id][agent_id][dimension]*(new_price - old_price)
                    self.agent_trust_scores[agent_id][dimension] = new_price
                    affected_agents_dimensions[agent_id].append(dimension)
                    
                # else: print(f"    Divestment for Agent {agent_id} on {dimension} too small or none possible.")

            # --- Handle Investment (amount > 0) ---
            else:
                # Amount to invest is capped by available capacity
                if source_id not in self.oracle_influence_mechanisms:
                    actual_investment = min(amount, available_cap)
                else:
                    actual_investment = amount

                if actual_investment > 0.001: # Threshold
                    change_amount = actual_investment # Record the change
                    x = actual_investment
                    agent_amm_params = self.agent_amm_params[agent_id][dimension]
                    T, K = agent_amm_params['T'], agent_amm_params['K']
                    R = K/T
                    old_price = R/T
                    if source_id in self.oracle_influence_mechanisms and self.primary_source_update_type == 'fix_T':
                        R_new = R - y
                        T_new = T
                    else:
                        num_shares_to_invest = x*T/(R + x)
                        T_new = T - num_shares_to_invest
                        R_new = K/T_new #= R + x  # = R*T/(T - num_shares_to_invest) = R*T/(T - x*T/(R + x)) = R*T*(R+x)/(R*T) = R + x
                    new_price = R_new/T_new

                    self.agent_amm_params[agent_id][dimension]['R'] = R_new
                    self.agent_amm_params[agent_id][dimension]['T'] = T_new
                    self.agent_amm_params[agent_id][dimension]['K'] = R_new * T_new

                    # Update investments
                    if source_id not in self.oracle_influence_mechanisms:
                        old_source_investments = self.source_investments[source_id][agent_id].get(dimension, 0.0)
                        self.source_investments[source_id][agent_id][dimension] += num_shares_to_invest
                        self.allocated_influence[source_id][dimension] += self.source_investments[source_id][agent_id][dimension]*new_price - old_source_investments*old_price
                        self.source_available_capacity[source_id][dimension] -= actual_investment
                    for s_id in self.source_investments:
                        if self.source_investments[s_id][agent_id][dimension] > 0 and s_id != source_id:
                            self.allocated_influence[s_id][dimension] +=  self.source_investments[s_id][agent_id][dimension]*(new_price - old_price)
                    self.agent_trust_scores[agent_id][dimension] = new_price
                    affected_agents_dimensions[agent_id].append(dimension)

            # Log the actual change (investment or divestment)
            if abs(change_amount) > 0.001:
                self.temporal_db['investments'].append({
                    'evaluation_round': self.evaluation_round, 'timestamp': time.time(),
                    'source_id': source_id, 'agent_id': agent_id, 'dimension': dimension,
                    'amount': change_amount, # Positive for invest, negative for divest
                    'confidence': confidence,
                    'type': 'investment' if change_amount > 0 else 'divestment'
                })
                self.temporal_db['trust_scores'].append({
                                'evaluation_round': self.evaluation_round, 'timestamp': time.time(),
                                'agent_id': agent_id, 'dimension': dimension,
                                'old_score': old_price, 'new_score': new_price,
                                'change_source': 'investment', 'source_id': None # Aggregate effect
                })

                # After any transaction, log the new state of the source for that dimension
                if source_id not in self.oracle_influence_mechanisms:
                    total_invested_value = 0
                    for a_id, dims in self.source_investments[source_id].items():
                        if dimension in dims:
                            shares = dims[dimension]
                            price = self.agent_trust_scores[a_id][dimension]
                            total_invested_value += shares * price
                    
                    available_cash = self.source_available_capacity[source_id][dimension]
                    
                    self.temporal_db['source_states'].append({
                        'evaluation_round': self.evaluation_round, 'timestamp': time.time(),
                        'source_id': source_id, 'dimension': dimension,
                        'total_invested_value': total_invested_value,
                        'available_cash': available_cash,
                        'total_value': total_invested_value + available_cash
                    })

        # Apply correlations if enabled and scores actually changed
        if self.use_dimension_correlations:
            for agent_id in affected_agents_dimensions:
                recalc_dims = set(affected_agents_dimensions[agent_id])
                # print(f"    Applying correlations for Agent {agent_id} due to changes in dimensions : {recalc_dims}")
                self.apply_dimension_correlations(agent_id, list(recalc_dims))


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
                source_capacity = self.source_available_capacity[source_id].get(dimension, 0.0) + self.allocated_influence[source_id].get(dimension, 0.0)
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
        neutral_rating = (self.rating_scale + 1) / 2.0   # Neutral rating for scale 1-5 is 3.0
        max_deviation = self.rating_scale - neutral_rating  # e.g., 5 - 3 = 2.0 for scale 1-5
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
                normalized_score_change = (rating_val - neutral_rating) / max_deviation     # [-1, 1]

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


    # Method for oracles to adjust R directly
    def oracle_adjust_reserve_direct(self, agent_id: str, dimension: str, delta_R: float):
        """
        Oracle directly adjusts the reserve for an agent in a dimension.
        This action *changes K* and is not a trade along the curve.
        T remains constant for this operation.
        """
        if agent_id not in self.agent_amm_params:
            # Initialize if new, or handle error
            # For simplicity, let's assume agent must exist / be initialized via another mechanism
            print(f"Warning: Agent {agent_id} not found in AMM params for oracle_adjust_reserve_direct.")
            # A common initialization: R=initial_R, T=initial_T (e.g., R=50, T=100 for P=0.5)
            # self.agent_amm_params[agent_id][dimension] = {'R': 50.0, 'T': 100.0, 'K': 5000.0, 'total_supply': 100.0} # Example
            # For now, let's assume it's initialized with some values
            if self.agent_amm_params[agent_id][dimension]['R'] == 0 and self.agent_amm_params[agent_id][dimension]['T'] == 0:
                self.agent_amm_params[agent_id][dimension]['R'] = self.config.get('initial_R_oracle', 10.0) # Small initial R
                self.agent_amm_params[agent_id][dimension]['T'] = self.config.get('initial_T_oracle', 20.0) # Ensures P=0.5
                self.agent_amm_params[agent_id][dimension]['K'] = self.agent_amm_params[agent_id][dimension]['R'] * self.agent_amm_params[agent_id][dimension]['T']
                # total_supply for AMM might track shares *held by investors* + shares *in treasury*.
                # Here, T is treasury shares. Let's assume total_supply is just T initially for AMM internal tracking.
                self.agent_amm_params[agent_id][dimension]['total_supply'] = self.agent_amm_params[agent_id][dimension]['T']


        params = self.agent_amm_params[agent_id][dimension]
        old_R = params['R']
        new_R = old_R + delta_R

        # Safeguard: R should not be negative (or below a minimum)
        min_R = self.config.get('min_R_oracle_adj', 0.01)             # TODO: Set the min_R carefully
        new_R = max(min_R, new_R)
        actual_delta_R = new_R - old_R

        params['R'] = new_R
        # T remains unchanged by this direct oracle action
        # K changes: params['K'] = new_R * params['T']
        # Update K after R has changed and T is stable
        if self.primary_source_update_type == 'fix_K':
            params['T'] = params['K'] / params['R']
        elif self.primary_source_update_type == 'fix_T':
            params['K'] = params['R'] * params['T']

        # The price (trust score) changesconfidence
        old_price = old_R / params['T'] if params['T'] > 0 else 0
        new_price = params['R'] / params['T'] if params['T'] > 0 else 0
        self.agent_trust_scores[agent_id][dimension] = new_price

        self.agent_amm_params[agent_id][dimension] = params

        for source_id in self.allocated_influence:
            if self.allocated_influence[source_id][dimension] > 0 and (self.source_investments[source_id][agent_id][dimension] > 0):
                self.allocated_influence[source_id][dimension] += self.source_investments[source_id][agent_id][dimension]*(new_price - old_price)

        # Accumulate the absolute change for the regulator
        if 'user_feedback' in self.oracle_influence_mechanisms.values() or 'comparative_feedback' in self.oracle_influence_mechanisms.values(): # only if user feedback is an oracle
            self.cumulative_user_influence[dimension][agent_id] += actual_delta_R

        self.amm_transactions_log.append({
            'evaluation_round': self.evaluation_round, 'timestamp': time.time(),
            'agent_id': agent_id, 'dimension': dimension, 'type': 'oracle_R_adjustment',
            'delta_R': actual_delta_R, 'new_R': params['R'], 'T_unchanged': params['T'],
            'old_price': old_price, 'new_price': new_price, 'source_id': 'oracle_system' # Or specific oracle
        })
        # print(f"Oracle adjusted R for A{agent_id} Dim {dimension}: R {old_R:.2f}->{new_R:.2f}, P {old_price:.3f}->{new_price:.3f}")


    # In your `record_user_feedback` or `record_comparative_feedback`
    # This would replace the direct manipulation of `self.agent_trust_scores`

    def record_comparative_feedback(self, agent_a_id: str, agent_b_id: str,
                                    winners: Dict): # feedback_strength S
        """
        Records comparative user feedback and updates agent AMM params via oracle_adjust_reserve_direct.
        winners: Dict mapping dimension -> 'A', 'B', or 'Tie'
        feedback_strength: Overall strength of this feedback batch (e.g., based on num users)
        """
        base_price_adj_factor = self.comparative_feedback_strength # Beta

        for dimension, winner_code_conf in winners.items():
            if dimension not in self.dimensions: continue

            winner_code, confidence = winner_code_conf

            params_A = self.agent_amm_params[agent_a_id][dimension]
            params_B = self.agent_amm_params[agent_b_id][dimension]

            # Ensure agents are initialized in AMM (important!)
            # This logic should ideally be in a separate `ensure_agent_in_amm` method
            for aid in [agent_a_id, agent_b_id]:
                if self.agent_amm_params[aid][dimension]['R'] == 0 and self.agent_amm_params[aid][dimension]['T'] == 0:
                    self.agent_amm_params[aid][dimension]['R'] = self.config.get('initial_R_oracle', 10.0)
                    self.agent_amm_params[aid][dimension]['T'] = self.config.get('initial_T_oracle', 20.0)
                    self.agent_amm_params[aid][dimension]['K'] = self.agent_amm_params[aid][dimension]['R'] * self.agent_amm_params[aid][dimension]['T']
                    self.agent_amm_params[aid][dimension]['total_supply'] = self.agent_amm_params[aid][dimension]['T']
                    # Initialize trust score from AMM price
                    self.agent_trust_scores[aid][dimension] = self.agent_amm_params[aid][dimension]['R'] / self.agent_amm_params[aid][dimension]['T']

            P_A_current = params_A['R'] / params_A['T'] if params_A['T'] > 0 else 0.5 # Default if T is 0
            P_B_current = params_B['R'] / params_B['T'] if params_B['T'] > 0 else 0.5

            delta_P_A = 0.0
            delta_P_B = 0.0

            if winner_code == 'A':
                delta_P_A = confidence * base_price_adj_factor # Absolute adjustment
                delta_P_B = -confidence * base_price_adj_factor
            elif winner_code == 'B':
                delta_P_A = -confidence * base_price_adj_factor
                delta_P_B = confidence * base_price_adj_factor
            # If 'Tie', delta_P_A and delta_P_B remain 0.0

            # Ensure target price is within valid bounds (e.g., 0 to 1)
            # Let P_target = P_current + delta_P. Clamp P_target. Then actual_delta_P = P_target_clamped - P_current.
            # max_score = self.config.get('max_trust_score_oracle', 1.0)
            min_score = self.config.get('min_trust_score_oracle', 0.0)

            P_A_target_unclamped = P_A_current + delta_P_A
            # P_A_target_clamped = max(min_score, min(max_score, P_A_target_unclamped))  # TODO: set the min and max scores carefully
            P_A_target_clamped = max(min_score, P_A_target_unclamped)
            actual_delta_P_A = P_A_target_clamped - P_A_current

            P_B_target_unclamped = P_B_current + delta_P_B
            P_B_target_clamped = max(min_score, P_B_target_unclamped)
            actual_delta_P_B = P_B_target_clamped - P_B_current

            if self.primary_source_update_type == 'fix_K':
                delta_R_A = params_A['R'] *(np.sqrt(P_A_target_clamped/P_A_current) - 1)
                delta_R_B = params_B['R'] *(np.sqrt(P_B_target_clamped/P_B_current) - 1)
            elif self.primary_source_update_type == 'fix_T':
                delta_R_A = actual_delta_P_A * params_A['T']
                delta_R_B = actual_delta_P_B * params_B['T']

            if abs(actual_delta_P_A) > 0.0001: # Threshold for action
                self.oracle_adjust_reserve_direct(agent_a_id, dimension, delta_R_A)
            if abs(actual_delta_P_B) > 0.0001: # Threshold for action
                self.oracle_adjust_reserve_direct(agent_b_id, dimension, delta_R_B)
    
    # For trust decay
    def apply_trust_decay(self):
        decay_rate = self.config.get('trust_decay_rate_oracle', 0.001) # e.g., 0.1% per period
        if decay_rate <= 0: return

        print(f"Applying trust decay (rate: {decay_rate}) to all agents...")
        for agent_id in list(self.agent_amm_params.keys()): # Iterate over copy of keys
            for dimension in self.dimensions:
                params = self.agent_amm_params[agent_id][dimension]
                if params['T'] == 0: continue # Avoid division by zero if T is 0 (should not happen with safeguards)

                # R_new = R_current * (1 - decay_rate)
                # So, delta_R = -R_current * decay_rate
                delta_R_decay = -params['R'] * decay_rate
                
                # Only apply if R is positive
                if params['R'] > self.config.get('min_R_oracle_adj', 0.01) and delta_R_decay < -0.00001: # Ensure meaningful negative change
                    self.oracle_adjust_reserve_direct(agent_id, dimension, delta_R_decay)


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
    
    def _recompute_source_influence_capacity(self, source_id: str) -> None:
        """
        Recompute the influence capacity of a source based on current investments.
        This is a placeholder for any complex logic needed to adjust capacity dynamically.
        """
        # Keep existing logic, ensure it uses self.source_influence_capacity
        if source_id in self.source_available_capacity:
            for dimension in self.dimensions:
                # Example: Adjust capacity based on total investments in this dimension
                self.source_influence_capacity[source_id][dimension] = self.source_available_capacity[source_id][dimension] + self.allocated_influence[source_id][dimension]

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
        all_source_ids = list(self.source_available_capacity.keys())
        for source_id in all_source_ids:
            self._recompute_source_influence_capacity(source_id) # Ensure capacity is up-to-date
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
    def _plot_dimensions(self, ax, dims_to_plot, pivot, df_invest, source_color_map):
        """
        Helper function to draw all data for a list of dimensions onto a
        provided matplotlib Axes object.
        """
        # Filter the pivot table to only include the dimensions for this plot
        dim_cols = [col for col in pivot.columns if col[1] in dims_to_plot]
        if not dim_cols:
            return False  # Return False if there's no data to plot

        pivot_dims = pivot[dim_cols]

        # Plot score lines for each agent and dimension
        for col in pivot_dims.columns:
            agent_id, dim_name = col
            # Modify label to be more descriptive if plotting multiple dims, else just agent ID
            label = f"A{agent_id}-{dim_name[:4]}" if len(dims_to_plot) > 1 else f"Agent {agent_id}"
            pivot_dims[col].plot(ax=ax, label=label, marker='.', linestyle='-', markersize=5)

        # Plot investment markers on top of the score lines
        if df_invest is not None and not df_invest.empty:
            invest_dims = df_invest[df_invest['dimension'].isin(dims_to_plot)]
            for _, investment in invest_dims.iterrows():
                round_val = investment['evaluation_round']
                agent_val = investment['agent_id']
                dim_val = investment['dimension']
                
                # Check if there is a score value to plot on
                if round_val in pivot_dims.index and (agent_val, dim_val) in pivot_dims.columns:
                    score_val = pivot_dims.loc[round_val, (agent_val, dim_val)]
                    if pd.isna(score_val):
                        continue
                    
                    marker = '^' if investment['amount'] > 0 else 'v'
                    color = source_color_map.get(investment['source_id'])
                    size = min(375, max(40, abs(investment['amount']) * 1.5))

                    ax.scatter(round_val, score_val, marker=marker, color=color, s=size, alpha=0.9, edgecolors='w', zorder=5)
        
        ax.set_xlabel('Evaluation Round')
        ax.set_ylabel('Trust Score')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(0, 1.05)
        return True # Return True indicating that plotting was successful

    def visualize_trust_scores(self, agents=None, dimensions=None,
                            start_round=None, end_round=None, show_investments=True,
                            save_path=None, experiment_name=None):
        """
        Visualize trust scores over time, with optional investment overlays.
        This function orchestrates the plotting process, generating and saving a 
        separate plot for each specified dimension.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            import pandas as pd
            import seaborn as sns
            import os
        except ImportError:
            print("Warning: matplotlib, pandas, or seaborn not installed. Cannot visualize scores.")
            return

        if not self.temporal_db['trust_scores']:
            print("No trust score data available for visualization")
            return

        # --- 1. Data Loading and Filtering ---
        df_scores = pd.DataFrame(self.temporal_db['trust_scores'])
        dims_to_iterate = dimensions or self.dimensions
        
        if agents: df_scores = df_scores[df_scores['agent_id'].isin(agents)]
        df_scores = df_scores[df_scores['dimension'].isin(dims_to_iterate)]
        if start_round is not None: df_scores = df_scores[df_scores['evaluation_round'] >= start_round]
        if end_round is not None: df_scores = df_scores[df_scores['evaluation_round'] <= end_round]

        if df_scores.empty:
            print("No trust score data available with current filters for visualization")
            return

        df_scores = df_scores.sort_values(['evaluation_round', 'timestamp'])
        pivot = df_scores.pivot_table(index='evaluation_round', columns=['agent_id', 'dimension'], values='new_score', aggfunc='last').ffill()

        # --- Prepare Investment Data ---
        df_invest = None
        source_color_map = {}
        if show_investments and self.temporal_db['investments']:
            df_invest = pd.DataFrame(self.temporal_db['investments'])
            if agents: df_invest = df_invest[df_invest['agent_id'].isin(agents)]
            df_invest = df_invest[df_invest['dimension'].isin(dims_to_iterate)]
            if start_round is not None: df_invest = df_invest[df_invest['evaluation_round'] >= start_round]
            if end_round is not None: df_invest = df_invest[df_invest['evaluation_round'] <= end_round]

            if not df_invest.empty:
                sources = sorted(df_invest['source_id'].unique())
                palette = sns.color_palette("husl", len(sources))
                source_color_map = {source: color for source, color in zip(sources, palette)}

        # --- 2. Orchestration: Loop and Plot for each Dimension ---
        for dim in dims_to_iterate:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Call the inner plotting function, passing a list with just the current dimension
            plotted = self._plot_dimensions(ax, [dim], pivot, df_invest, source_color_map)

            if not plotted:
                plt.close(fig) # Don't show or save empty plots
                continue
            
            # --- 3. Final Touches (Legend, Labels, Saving) for the current plot ---
            legend_elements = ax.get_legend_handles_labels()[0]
            
            if source_color_map:
                legend_elements.append(Line2D([0], [0], marker='', color='w', label=''))
                for source, color in source_color_map.items():
                    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=source, markerfacecolor=color, markersize=10))
                legend_elements.append(Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', label='Buy', markersize=10))
                legend_elements.append(Line2D([0], [0], marker='v', color='w', markerfacecolor='gray', label='Sell', markersize=10))

            ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
            ax.set_title(f'Agent Trust Scores Over Time: {dim}', fontsize=16)
            plt.tight_layout(rect=[0, 0, 0.85, 1])

            # Save Figure logic
            if save_path and experiment_name:
                try:
                    folder = os.path.join(save_path, experiment_name)
                    os.makedirs(folder, exist_ok=True)
                    safe_dim_name = dim.replace(' ', '_').replace('/', '_')
                    filepath = os.path.join(folder, f'trust_scores_{safe_dim_name}.png')
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    print(f"Plot saved to {filepath}")
                except Exception as e:
                    print(f"Error saving plot for dimension {dim}: {e}")

            plt.show()
            plt.close(fig)
        
        return

    def _plot_source_value_dimension(self, ax, dim, source_states_df, investments_df, agent_marker_map):
        """
        Helper function to draw source value data for a single dimension
        onto a provided matplotlib Axes object.
        """
        # --- Plot Source Value Lines ---
        # Pivot to get sources as columns and their total_value as values
        pivot_source_values = source_states_df.pivot_table(
            index='evaluation_round', 
            columns='source_id', 
            values='total_value', 
            aggfunc='last'
        ).ffill()

        if pivot_source_values.empty:
            return False

        for source_id in pivot_source_values.columns:
            pivot_source_values[source_id].plot(ax=ax, label=f"Value: {source_id}", marker='.', linestyle='-')

        # --- Plot Investment Markers ---
        if investments_df is not None:
            for _, investment in investments_df.iterrows():
                round_val = investment['evaluation_round']
                source_id = investment['source_id']
                agent_id = investment['agent_id']

                # Place marker on the source's value line
                if round_val in pivot_source_values.index and source_id in pivot_source_values.columns:
                    y_val = pivot_source_values.loc[round_val, source_id]
                    if pd.isna(y_val): continue
                    
                    marker = agent_marker_map.get(agent_id, 'x')
                    color = 'green' if investment['amount'] > 0 else 'red'
                    size = min(300, max(40, abs(investment['amount']) * 2))

                    ax.scatter(round_val, y_val, marker=marker, color=color, s=size, alpha=0.9, edgecolors='w', zorder=5)
        
        ax.set_xlabel('Evaluation Round')
        ax.set_ylabel('Total Portfolio Value ($)')
        ax.grid(True, linestyle='--', alpha=0.6)
        return True

    def visualize_source_value(self, sources=None, dimensions=None,
                                  start_round=None, end_round=None,
                                  save_path=None, experiment_name=None):
        """
        Visualize the total portfolio value of each source over time.
        Generates a separate plot for each dimension.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            import pandas as pd
            import os
        except ImportError:
            print("Warning: Required libraries not installed. Cannot visualize source value.")
            return

        if not self.temporal_db['source_states']:
            print("No source state data available for visualization.")
            return

        # --- 1. Data Loading and Filtering ---
        source_states_df = pd.DataFrame(self.temporal_db['source_states'])
        dims_to_plot = dimensions or self.dimensions

        if sources: source_states_df = source_states_df[source_states_df['source_id'].isin(sources)]
        if start_round is not None: source_states_df = source_states_df[source_states_df['evaluation_round'] >= start_round]
        if end_round is not None: source_states_df = source_states_df[source_states_df['evaluation_round'] <= end_round]

        investments_df = pd.DataFrame(self.temporal_db.get('investments', []))
        if not investments_df.empty:
            if sources: investments_df = investments_df[investments_df['source_id'].isin(sources)]

        all_agent_ids = []
        if not investments_df.empty:
            all_agent_ids = sorted(investments_df['agent_id'].unique())
        
        # Define a unique marker for each agent
        markers = ['o', 's', 'P', 'X', '*', 'D', 'v', '^', '<', '>']
        agent_marker_map = {agent_id: markers[i % len(markers)] for i, agent_id in enumerate(all_agent_ids)}

        # --- 2. Orchestration Loop ---
        for dim in dims_to_plot:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            dim_source_states = source_states_df[source_states_df['dimension'] == dim]
            dim_investments = investments_df[investments_df['dimension'] == dim] if not investments_df.empty else None

            if dim_source_states.empty:
                plt.close(fig)
                continue

            plotted = self._plot_source_value_dimension(ax, dim, dim_source_states, dim_investments, agent_marker_map)

            if not plotted:
                plt.close(fig)
                continue
            
            # --- 3. Final Touches and Legend ---
            legend_elements = ax.get_legend_handles_labels()[0]
            legend_elements.append(Line2D([0], [0], marker='', color='w')) # Spacer
            legend_elements.append(Line2D([0], [0], color='green', marker='^', linestyle='None', label='Buy Action'))
            legend_elements.append(Line2D([0], [0], color='red', marker='v', linestyle='None', label='Sell Action'))
            legend_elements.append(Line2D([0], [0], marker='', color='w', label='Target Agent:'))
            for agent_id, marker in agent_marker_map.items():
                legend_elements.append(Line2D([0], [0], color='gray', marker=marker, linestyle='None', label=f'Agent {agent_id}'))

            ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Legend")
            ax.set_title(f'Source Portfolio Value Over Time: {dim}', fontsize=16)
            plt.tight_layout(rect=[0, 0, 0.85, 1])

            # --- Save Figure ---
            if save_path and experiment_name:
                try:
                    folder = os.path.join(save_path, experiment_name)
                    os.makedirs(folder, exist_ok=True)
                    safe_dim_name = dim.replace(' ', '_').replace('/', '_')
                    filepath = os.path.join(folder, f'source_value_{safe_dim_name}.png')
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    print(f"Plot saved to {filepath}")
                except Exception as e:
                    print(f"Error saving plot for dimension {dim}: {e}")

            plt.show()
            plt.close(fig)

    def get_and_reset_cumulative_user_influence(self) -> Dict[str, float]:
        """
        Called by a source (e.g., Regulator) to get the total user-driven
        capital shifts since the last call, and then reset the tracker.
        """
        influence_data = {}
        # compute the absolute sum of the influence data for each dimension
        for dim in self.cumulative_user_influence.keys():
            influence_data[dim] = sum(abs(v) for v in self.cumulative_user_influence[dim].values())
            self.cumulative_user_influence[dim].clear()
        if influence_data and any(v > 0 for v in influence_data.values()):
            print(f"  DEBUG (Market): Returning and resetting user influence: {dict(influence_data)}")
        return influence_data

