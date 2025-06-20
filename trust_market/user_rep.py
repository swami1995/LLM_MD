import math
import numpy as np
from collections import defaultdict
from trust_market.info_sources import InformationSource 
from typing import List, Dict, Any, Optional, Tuple
# Import evaluator from auditor (assuming it's in the same directory/package)
from trust_market.auditor import BatchEvaluator # Use BatchEvaluator for comparisons
# Assuming google.genai for LLM calls
from google import genai
from google.genai import types
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Base UserRepresentative (Simplified Logic) ---
class UserRepresentative(InformationSource):
    """
    Represents aggregated feedback for a user segment.
    (Simplified base class logic).
    """
    def __init__(self, source_id, user_segment, representative_profile, market=None, verbose=False):
        # Determine expertise based on segment
        # (Same logic as before to set expertise_dimensions and confidence)
        if user_segment == "technical":
            expertise_dimensions = ["Factual_Correctness", "Process_Reliability", "Transparency", "Trust_Calibration"]
            confidence = {"Factual_Correctness": 0.9, "Process_Reliability": 0.8, "Transparency": 0.7, "Trust_Calibration": 0.8}
        elif user_segment == "non_technical":
            expertise_dimensions = ["Communication_Quality", "Problem_Resolution", "Value_Alignment"]
            confidence = {"Communication_Quality": 0.9, "Problem_Resolution": 0.8, "Value_Alignment": 0.7}
        else: # balanced
            expertise_dimensions = ["Communication_Quality", "Problem_Resolution", "Value_Alignment", "Transparency"]
            confidence = {"Communication_Quality": 0.8, "Problem_Resolution": 0.8, "Value_Alignment": 0.7, "Transparency": 0.7}

        super().__init__(source_id, "user_representative", expertise_dimensions,
                         confidence, market)

        self.user_segment = user_segment
        self.representative_profile = representative_profile
        self.represented_users = set()
        self.agent_conversations = defaultdict(list) # {agent_id: [conv_hist, ...]}
        # Store feedback received directly
        self.direct_feedback = defaultdict(lambda: defaultdict(list)) # {agent_id: {dimension: [rating, ...]}}
        self.comparison_feedback = [] # List of comparison dicts received
        self.verbose = verbose

        # Configuration for segment importance weights
        # (Keep the segment_weights dictionary as before)
        self.segment_weights = {
            "technical": {"Factual_Correctness": 0.9, "Process_Reliability": 0.8, "Transparency": 0.8, "Trust_Calibration": 0.9, "Communication_Quality": 0.6, "Problem_Resolution": 0.7, "Value_Alignment": 0.6},
            "non_technical": {"Communication_Quality": 0.9, "Problem_Resolution": 0.9, "Value_Alignment": 0.8, "Factual_Correctness": 0.6, "Process_Reliability": 0.6, "Transparency": 0.5, "Trust_Calibration": 0.4},
            "balanced": {"Communication_Quality": 0.8, "Problem_Resolution": 0.9, "Value_Alignment": 0.8, "Transparency": 0.6, "Factual_Correctness": 0.7, "Process_Reliability": 0.7, "Trust_Calibration": 0.6}
        }
        # Add defaults for missing dimensions in segment weights
        all_dims = ["Factual_Correctness", "Process_Reliability", "Value_Alignment", "Communication_Quality", "Problem_Resolution", "Safety_Security", "Transparency", "Adaptability", "Trust_Calibration", "Manipulation_Resistance"]
        for seg_weights in self.segment_weights.values():
            for dim in all_dims:
                seg_weights.setdefault(dim, 0.5) # Default importance if not specified

    def add_represented_user(self, user_id, user_profile=None):
        """Adds a user ID to the set this rep represents."""
        self.represented_users.add(user_id)

    def add_conversation(self, conversation_history: List[Dict], user_id: Any, agent_id: int):
        """Stores conversation history if the user is represented."""
        if user_id in self.represented_users:
            self.agent_conversations[agent_id].append(conversation_history)

    def add_direct_feedback(self, user_id: Any, agent_id: int, ratings: Dict[str, int]):
        """Stores direct user ratings if the user is represented."""
        if user_id in self.represented_users:
             for dim, rating in ratings.items():
                  if dim in self.expertise_dimensions: # Only store relevant dimensions
                       self.direct_feedback[agent_id][dim].append(rating)

    def add_comparison_feedback(self, comparison_data: Dict):
        """Stores comparison results if the user is represented."""
        if comparison_data.get('user_id') in self.represented_users:
             self.comparison_feedback.append(comparison_data)

    def evaluate_agent(self, agent_id, dimensions=None):
        """
        Evaluates agent based on aggregated direct feedback.
        (Simplified logic for base UserRepresentative).

        Returns: Dict mapping dimensions to (rating_0_1, confidence)
        """
        dimensions_to_evaluate = dimensions or self.expertise_dimensions
        results = {}
        default_confidence = 0.6 # Base confidence for aggregated feedback

        if agent_id not in self.direct_feedback:
             return {dim: (0.5, 0.3) for dim in dimensions_to_evaluate if dim in self.expertise_dimensions} # Return neutral if no feedback

        feedback_for_agent = self.direct_feedback[agent_id]

        for dim in dimensions_to_evaluate:
             if dim in self.expertise_dimensions and dim in feedback_for_agent:
                  ratings = feedback_for_agent[dim]
                  if ratings:
                       # Average the ratings received
                       avg_rating = sum(ratings) / len(ratings)
                       # Normalize rating to 0-1 scale (assuming market scale is stored in self.market)
                       # TODO: Need access to market's rating_scale for proper normalization. Assuming 5 for now.
                       rating_scale = self.market.rating_scale if self.market else 5
                       neutral_rating = (rating_scale + 1) / 2.0
                       max_dev = rating_scale - neutral_rating
                       normalized_score = 0.5 + ((avg_rating - neutral_rating) / max_dev if max_dev > 0 else 0) * 0.5
                       normalized_score = max(0.0, min(1.0, normalized_score))

                       # Confidence increases with more feedback points
                       confidence = min(0.9, default_confidence + 0.05 * math.log1p(len(ratings)))
                       results[dim] = (normalized_score, confidence)
                  else:
                       results[dim] = (0.5, 0.3) # Default if no ratings for this dim
             elif dim in self.expertise_dimensions:
                  results[dim] = (0.5, 0.3) # Default if dim is relevant but no ratings

        # Clear feedback used for this evaluation round? Optional.
        # self.direct_feedback[agent_id] = defaultdict(list)

        return results


# --- Holistic User Rep using LLM Batch Evaluation ---
class UserRepresentativeWithHolisticEvaluation(UserRepresentative):
    """
    User representative that evaluates agents holistically across batches of conversations using LLM.
    """

    def __init__(self, source_id, user_segment, representative_profile, market=None, api_key=None, api_model_name='gemini-2.5-flash', verbose=False, api_provider='gemini', openai_api_key=None):
        super().__init__(source_id, user_segment, representative_profile, market, verbose=verbose)

        # Initialize holistic evaluator (using BatchEvaluator from auditor.py)
        self.api_model_name = api_model_name
        self.evaluator = BatchEvaluator(
            api_key=api_key,
            api_model_name=api_model_name,
            api_provider=api_provider,
            openai_api_key=openai_api_key
        )
        self.batch_evaluator = self.evaluator # For base class compatibility

        # Cache for the single-agent evaluation method
        self.agent_evaluation_cache = {}

        # Configuration specific to this Rep's strategy
        self.config = {
            'comparison_agents_per_target': 3, # Compare against fewer agents than auditor?
            'min_conversations_required': 3,   # Min conversations needed for comparison
            'new_evaluation_weight': 0.7,      # Weight for new comparative scores vs prior derived score
            'invest_multiplier': 0.2,         # Investment aggressiveness
            'divest_multiplier': 0.15,        # Divestment aggressiveness
            'precision_scale_factor': 0.6,     # Controls how fast confidence grows. Lower is slower.
            # Added for base class compatibility
            'base_score_persistence': 0.2,
            'derived_score_update_weight': 0.3,
            'max_confidence_history': 10,
            'desirability_method': 'percentage_change',
            'min_operational_price': 0.01,
            'attractiveness_buy_threshold': 0.01,
            'min_value_holding_per_asset': 0.0,
            'portfolio_rebalance_aggressiveness': 0.5,
            'min_delta_value_trade_threshold': 0.1,
            'investment_scale': 0.2,
            'rank_correction_strength': 0.5,
            'market_growth_factor': 1.0,
            'quality_concentration_power': 2.0,
        }
        # Note: Uses segment_weights from base class for dimension importance


    def get_agent_conversations(self, agent_id, max_count=10):
        """Gets recent conversations for an agent."""
        return self.agent_conversations.get(agent_id, [])[-max_count:]

    def observed_agents(self):
        """Returns agent IDs with stored conversations."""
        return set(self.agent_conversations.keys())

    def _agent_has_comparable_data(self, aid):
        """Checks if an agent has enough conversation data to be used in a comparison."""
        min_convs = self.config.get('min_conversations_required', 3)
        return len(self.get_agent_conversations(aid)) >= min_convs
    
    def _perform_base_evaluation(self, agent_id, dimensions, evaluation_round):
        """UserRep's non-comparative evaluation returns a neutral score, as its strength is comparison."""
        return {dim: (0.5, 0.3) for dim in dimensions}

    def _compare_pair(self, aid, oid, dimensions) -> Optional[Tuple[int, int, dict, dict]]:
        """UserRep's implementation of a pairwise comparison based on conversations."""
        if not (self._agent_has_comparable_data(aid) and self._agent_has_comparable_data(oid)):
            return None # Incomparable

        comparison_results = self.evaluator.compare_agent_batches(
            self.get_agent_conversations(aid), aid,
            self.get_agent_conversations(oid), oid,
            dimensions
        )

        if not comparison_results:
            return None

        derived_scores = self.evaluator.get_agent_scores(comparison_results, aid, oid)
        confidences = super()._extract_comparison_confidences(comparison_results, aid, oid)
        
        return (aid, oid, derived_scores, confidences)

    # Override evaluate_agent to use the holistic comparison method
    def evaluate_agent(self, agent_id, dimensions=None, evaluation_round=None, use_comparative=True):
        """
        Evaluates an agent using holistic comparison against peers.
        Now efficiently updates scores for both agents in each comparison.

        Returns: Dict mapping dimensions to (rating_0_1, confidence) tuples.
        TODO : Make the compared pairs into a dictionary storing the corresponding scores for each agent pair within a round for caching purposes to avoid recomputing. 
        TODO : Need to be more careful about keeping/carrying forward derived scores across evaluation rounds.
        TODO : Need to ensure that the scores of other agents are also updated in the derived_agent_scores cache...
        TODO : Need to treat confidence more carefully and make sure things are somewhat calibrated.
        TODO : Figure out what to do about confidence for derived scores. 
        """
        if self.verbose:
            print(f"\n--- UserRep {self.source_id} (single-agent eval) for Agent {agent_id} round {evaluation_round} ---")

        dimensions_to_evaluate = dimensions or self.expertise_dimensions
        # Use cache if available and for the current round
        cache_key = (agent_id, tuple(sorted(dimensions_to_evaluate)), evaluation_round)
        if evaluation_round == self.last_evaluation_round and cache_key in self.agent_evaluation_cache:
            return self.agent_evaluation_cache[cache_key]

        # Reset comparison tracking if it's a new round
        if evaluation_round and evaluation_round != self.last_evaluation_round:
            super()._invalidate_cache()
            self.last_evaluation_round = evaluation_round
            if self.verbose:
                print(f"UserRep {self.source_id} starting new evaluation round {evaluation_round}")

        # --- Start Evaluation Logic ---
        # Get base scores (from previous comparisons in this round or neutral)
        base_scores_0_1 = self.derived_agent_scores.get(agent_id, {})
        base_confidences = super()._calculate_derived_confidence(agent_id, dimensions_to_evaluate)

        base_scores_with_conf = {
            dim: (base_scores_0_1.get(dim, 0.5), base_confidences.get(dim, 0.3))
            for dim in dimensions_to_evaluate
        }

        if self.verbose:
            print(f"DEBUG: Agent {agent_id} - Initial scores with confidence: {base_scores_with_conf}")
        if not self._agent_has_comparable_data(agent_id):
            return {dim: (score, min(conf, 0.3)) for dim, (score, conf) in base_scores_with_conf.items()}

        other_agent_ids = {oid for oid in self.observed_agents() if self._agent_has_comparable_data(oid)}
        other_agent_ids.discard(agent_id)
        valid_comparison_agents = list(other_agent_ids)
        
        if not valid_comparison_agents:
             return base_scores_with_conf

        # --- Select Comparison Subset (prioritize new comparisons) ---
        import random
        num_to_compare = self.config.get('comparison_agents_per_target', 3)
        comparison_agents_selected = []
        
        # Prioritize agents we haven't compared with yet in this round
        new_comparison_candidates = [
            (oid, c) for oid, c in valid_comparison_agents 
            if (min(agent_id, oid), max(agent_id, oid), evaluation_round) not in self.comparison_results_cache
        ]
        
        if len(new_comparison_candidates) >= num_to_compare:
            comparison_agents_selected = random.sample(new_comparison_candidates, num_to_compare)
        else:
            # Use all new candidates plus some existing ones if needed
            comparison_agents_selected = new_comparison_candidates
            remaining_needed = num_to_compare - len(new_comparison_candidates)
            if remaining_needed > 0:
                existing_candidates = [
                    (oid, c) for oid, c in valid_comparison_agents 
                    if (oid, c) not in new_comparison_candidates
                ]
                if existing_candidates:
                    comparison_agents_selected.extend(
                        random.sample(existing_candidates, min(remaining_needed, len(existing_candidates)))
                    )

        selected_ids = [oid for oid, _ in comparison_agents_selected]
        if self.verbose:
            print(f"DEBUG: Agent {agent_id} - Selected comparison agents: {selected_ids}")

        # --- Perform Comparisons and Update Both Agents ---
        accumulated_scores = defaultdict(list)
        accumulated_confidences = defaultdict(list)
        comparison_count = 0

        for other_id, other_convs in comparison_agents_selected:
            if self.verbose:
                print(f"DEBUG: --> Comparing {agent_id} vs {other_id}")
            
            # Check if we already have this comparison result cached
            comparison_cache_key = (min(agent_id, other_id), max(agent_id, other_id), evaluation_round)
            
            if comparison_cache_key in self.comparison_results_cache:
                if self.verbose:
                    print(f"DEBUG: Using cached comparison result for {agent_id} vs {other_id}")
                derived_scores, comparison_confidences = self.comparison_results_cache[comparison_cache_key]
            else:
                pair_result = self._compare_pair(agent_id, other_id, dimensions_to_evaluate)
                if not pair_result:
                    continue
                _, _, derived_scores, comparison_confidences = pair_result
                self.comparison_results_cache[comparison_cache_key] = (derived_scores, comparison_confidences)

                # Update derived scores for BOTH agents using base class methods
                super()._update_agent_derived_scores(agent_id, derived_scores.get(agent_id, {}), dimensions_to_evaluate, comparison_confidences.get(agent_id, {}))
                super()._update_agent_derived_scores(other_id, derived_scores.get(other_id, {}), dimensions_to_evaluate, comparison_confidences.get(other_id, {}))
                super()._update_agent_confidences(agent_id, comparison_confidences.get(agent_id, {}), dimensions_to_evaluate)
                super()._update_agent_confidences(other_id, comparison_confidences.get(other_id, {}), dimensions_to_evaluate)

            for dim in dimensions_to_evaluate:
                if dim in derived_scores.get(agent_id, {}):
                    accumulated_scores[dim].append(derived_scores[agent_id][dim])
                    accumulated_confidences[dim].append(comparison_confidences.get(agent_id, {}).get(dim, 0.3))
                else:
                    if self.verbose:
                        print(f"DEBUG WARNING: Agent {agent_id} not found in derived scores from comparison with {other_id}")

            comparison_count += 1

        if self.verbose:
            print(f"DEBUG: Agent {agent_id} - Accumulated scores from {comparison_count} comparisons: {dict(accumulated_scores)}")
        
        # --- Calculate Final Scores ---
        final_eval_scores = {}
        for dim in dimensions_to_evaluate:
            base_score, base_conf = base_scores_with_conf.get(dim, (0.5, 0.3))
            if dim in accumulated_scores and accumulated_scores[dim]:
                new_scores = accumulated_scores[dim]
                new_confs = accumulated_confidences[dim]
                
                if sum(new_confs) > 1e-6:
                    confidence_weighted_avg = sum(s * c for s, c in zip(new_scores, new_confs)) / sum(new_confs)
                    avg_new_confidence = sum(new_confs) / len(new_confs)
                else:
                    confidence_weighted_avg = sum(new_scores) / len(new_scores)
                    avg_new_confidence = 0.3
                
                # Determine optimal weighting based on relative confidence
                total_confidence = base_conf + avg_new_confidence
                if total_confidence > 1e-6:
                    weight_new = avg_new_confidence / total_confidence
                else:
                    weight_new = 0.5  # Default equal weighting
                
                # Apply confidence-based weighting with some persistence of base scores
                persistence_factor = self.config.get('base_score_persistence', 0.2)
                effective_weight_new = weight_new * (1 - persistence_factor)
                
                final_score = (effective_weight_new * confidence_weighted_avg) + ((1 - effective_weight_new) * base_score)
                final_confidence = super()._aggregate_confidences(new_confs, base_conf, effective_weight_new)
                
                final_eval_scores[dim] = (final_score, final_confidence)
                if self.verbose:
                    print(f"DEBUG: Agent {agent_id} Dim {dim}: conf_weighted_avg={confidence_weighted_avg:.3f}, "
                        f"base={base_score:.3f}, weight_new={effective_weight_new:.3f} -> "
                        f"final={final_score:.3f}, conf={final_confidence:.3f}")
            else:
                final_eval_scores[dim] = (base_score, base_conf * 0.9)
                if self.verbose:
                    print(f"DEBUG: Agent {agent_id} Dim {dim}: No new data. Using base={base_score:.3f}, final_conf={base_conf * 0.9:.3f}")

        for dim, (score, _) in final_eval_scores.items():
            if agent_id not in self.derived_agent_scores: self.derived_agent_scores[agent_id] = {}
            self.derived_agent_scores[agent_id][dim] = score

        self.agent_evaluation_cache[cache_key] = final_eval_scores
        if self.verbose:
            print(f"--- UserRep {self.source_id} finished evaluating Agent {agent_id}. Final Scores: {final_eval_scores} ---")
        return final_eval_scores
    
    def _get_target_price_from_rank_mapping(self, agent_id, dimension, own_evaluations, market_prices, confidence_in_own_eval):
        """
        Calculates a target price for agent_id in a given dimension using rank-order mapping.
        own_evaluations: {agent_id: {dim: (pseudo_score, confidence)}}
        market_prices: {agent_id: {dim: P_current}}
        confidence_in_own_eval: The investor's confidence in its pseudo_score for this agent-dim.
        """
        
        # 1. Collect scores for the current dimension for all evaluated agents
        eval_scores_for_dim = {} # {agent_id: pseudo_score}
        market_p_for_dim = {}     # {agent_id: P_current}

        for aid, eval_data in own_evaluations.items():
            if dimension in eval_data:
                eval_scores_for_dim[aid] = eval_data[dimension][0] # pseudo_score
                if aid in market_prices and dimension in market_prices[aid]:
                    market_p_for_dim[aid] = market_prices[aid][dimension]
                # else:
                    # Agent might be new or not priced yet; handle as needed (e.g., skip or use default)

        if agent_id not in eval_scores_for_dim or agent_id not in market_p_for_dim:
            # If current agent doesn't have a score/price for this dim, can't determine target
            return market_p_for_dim.get(agent_id, 0.5) # Fallback to current market price or neutral

        if len(eval_scores_for_dim) < 2 or len(market_p_for_dim) < 2:
            # Not enough agents to establish meaningful ranks for comparison
            return market_p_for_dim[agent_id] # Fallback to current market price

        # 2. Rank agents based on eval scores and market prices
        # Higher score/price = better rank (e.g., rank 0 is best)
        sorted_by_eval = sorted(eval_scores_for_dim.items(), key=lambda item: item[1], reverse=True)
        eval_ranks = {aid: i for i, (aid, score) in enumerate(sorted_by_eval)}

        sorted_by_market = sorted(market_p_for_dim.items(), key=lambda item: item[1], reverse=True)
        market_ranks = {aid: i for i, (aid, price) in enumerate(sorted_by_market)}

        current_eval_rank = eval_ranks.get(agent_id)
        current_market_price = market_p_for_dim[agent_id]

        if current_eval_rank is None:
            return current_market_price # Should not happen if checks above are done

        # 3. Determine the market price of the agent currently at the investor's target rank
        # Example: If investor ranks agent_X as 0th (best), find the agent that is currently 0th in market_ranks
        # and use its price as a reference.
        target_rank_price_reference = current_market_price # Default to current price
        
        # Find the agent_id that is currently at the 'current_eval_rank' in the *market's* ranking
        agent_at_target_rank_in_market = None
        for m_aid, m_rank in market_ranks.items():
            if m_rank == current_eval_rank:
                agent_at_target_rank_in_market = m_aid
                break
        
        if agent_at_target_rank_in_market and agent_at_target_rank_in_market in market_p_for_dim:
            target_rank_price_reference = market_p_for_dim[agent_at_target_rank_in_market]
        # else:
            # If no agent is at that exact rank (e.g., fewer agents in market ranking due to missing prices),
            # we might interpolate or use closest rank. For simplicity, we use current_market_price as fallback.
            # Or, if eval ranks it Nth, but market only has M < N agents, this is an edge case.
            # Could also use the price of the Nth agent in the investor's own price-sorted list as a self-consistent target.
            # Let's use the price of the agent that *the market ranks* at the position *the investor thinks our agent should be*.

        # 4. The raw P_target is this reference price.
        p_target_raw_from_rank = target_rank_price_reference
        
        # Nudge towards this raw target based on confidence in *this specific evaluation*
        # The `confidence_in_own_eval` is the `conf` part from `(pseudo_score, conf)`
        # p_target_nudged = current_market_price + (p_target_raw_from_rank - current_market_price) * confidence_in_own_eval

        # Alternative: If investor is very confident agent X is #1, and market's #1 is priced at P_high,
        # and agent X is currently P_low, target P_high for agent X.
        # If investor is less confident, target somewhere between P_low and P_high.
        # The scaling here determines how aggressively the investor tries to correct the market rank.
        rank_correction_strength = self.config.get('rank_correction_strength', 0.5) # How much to move towards the target rank's price
        p_target_nudged = current_market_price + \
                          (p_target_raw_from_rank - current_market_price) * \
                          confidence_in_own_eval * rank_correction_strength

        return p_target_nudged

# In auditor.py or user_rep.py
# Inside the relevant InformationSource class (e.g., AuditorWithProfileAnalysis or UserRepresentativeWithHolisticEvaluation)

    def _calculate_total_portfolio_value_potential(self):
        """
        Calculates the sum of the current market value of all shares held by this source
        PLUS all available (uninvested) cash capacity of this source.
        This represents the total value this source could theoretically manage.
        """
        total_value_of_holdings = defaultdict(lambda : 0.0) # {dimension: total_value}
        # Sum market value of current share holdings
        # source_investments structure: self.source_investments[source_id][agent_id][dimension] = shares
        if self.source_id in self.market.source_investments:
            for agent_id, dims_data in self.market.source_investments[self.source_id].items():
                for dimension, shares_held in dims_data.items():
                    # if shares_held > 1e-5:
                    self.market.ensure_agent_dimension_initialized_in_amm(agent_id, dimension) # Ensure AMM params exist
                    amm_params = self.market.agent_amm_params[agent_id][dimension]
                    if amm_params['T'] > 1e-6: # Avoid division by zero if T is tiny
                        price = amm_params['R'] / amm_params['T']
                        total_value_of_holdings[dimension] += shares_held * price
                        # else: if T is zero, price is undefined/infinite, value of shares is complex.
                        # For simplicity, if T is zero, those shares are currently "unpriceable" by this AMM.

        # Sum available cash from all dimensions for this source
        # source_available_capacity structure: self.source_available_capacity[source_id][dimension] = cash
        total_available_cash = self.market.source_available_capacity.get(self.source_id, {})

        total_potential = {dim: total_value_of_holdings[dim] + total_available_cash[dim] for dim in self.market.source_available_capacity[self.source_id]}
        if self.verbose:
            print(total_potential)
        # Ensure a minimum potential to avoid issues if source starts with no cash/shares
        # return max(total_potential, self.config.get('min_portfolio_value_potential', 100.0))
        return total_potential

    def _project_steady_state_prices(self, own_evaluations, market_prices, dimension):
        """
        Project what market prices will be at steady state based on:
        1. Expected total capital deployment
        2. Quality-based distribution of that capital
        """
        
        # Step 1: Estimate total capital at steady state
        # Consider all potential investors and their capacity
        total_potential_capital = 0
        for source_id, _ in self.market.source_available_capacity.items():
            # Each source's total capacity across all dimensions
            source_capacity = self.market.source_available_capacity[source_id].get(dimension, 0)
            total_potential_capital += source_capacity
        
        # Add expected growth factor (new investors, increased allocations)
        growth_factor = self.config.get('market_growth_factor', 1.0)
        steady_state_capital = total_potential_capital * growth_factor
        
        # Step 2: Current total capital in market
        current_total_market_capital = 0
        for agent_id in market_prices:
            if dimension in self.market.agent_amm_params[agent_id]:
                # Total capital locked in AMM = R (reserves)
                current_total_market_capital += self.market.agent_amm_params[agent_id][dimension]['R']
        
        steady_state_capital += current_total_market_capital # Include current capital in the market
        if current_total_market_capital < 1e-6:
            current_total_market_capital = 1e-6

        # Step 3: Project capital distribution based on quality scores
        # Use own evaluations as best estimate of true quality
        quality_scores = {
            agent_id: eval_data[dimension][0]
            for agent_id, eval_data in own_evaluations.items()
            if dimension in eval_data
        }
        
        # Convert quality to expected capital share
        # Higher quality agents should attract disproportionately more capital
        concentration_power = self.config.get('quality_concentration_power', 2.0)
        
        quality_powered = {
            aid: q ** concentration_power 
            for aid, q in quality_scores.items()
        }
        total_quality_powered = sum(quality_powered.values())
        
        # Expected share of steady-state capital for each agent
        expected_capital_shares = {}
        if total_quality_powered > 1e-6:
            expected_capital_shares = {
                aid: qp / total_quality_powered 
                for aid, qp in quality_powered.items()
            }
        
        # Step 4: Project steady-state prices
        projected_prices = {}
        
        for agent_id in expected_capital_shares:
            # Expected capital for this agent at steady state
            expected_capital = steady_state_capital * expected_capital_shares[agent_id]
            
            # Project price based on AMM dynamics
            # At steady state, if R_ss is the reserve, need to estimate T_ss
            # Assume T remains relatively stable (or decreases slowly as investors buy)
            current_T = self.market.agent_amm_params[agent_id][dimension]['T']
            current_R = self.market.agent_amm_params[agent_id][dimension]['R']
            
            # Estimate T at steady state (some shares bought from treasury)
            # treasury_depletion_rate = self.config.get('treasury_depletion_rate', 0.3)
            # projected_T = current_T * (1 - treasury_depletion_rate)                             # TODO : Need a more sophisticated mechanism to compute projected_T and corresponding projected prices.
            if expected_capital > 1e-6 :
                projected_T = current_T * current_R / expected_capital 
            else:
                projected_T = current_T

            # Projected price = R_ss / T_ss
            projected_price = expected_capital / projected_T if projected_T > 1e-6 else 0
            projected_prices[agent_id] = projected_price
        
        return projected_prices, steady_state_capital / current_total_market_capital

    def check_market_capacity(self, own_evaluations, market_prices):
        """
        Checks if the source has enough capacity to invest based on its evaluations and market prices.
        If not, it will print a warning and return False.
        """
        capacity_flags = {} # Collect ratios for all dimensions
        projected_prices = {} # {agent_id: projected_price}
        for dim in self.expertise_dimensions:
            projected_prices_dim, steady_state_ratio = self._project_steady_state_prices(own_evaluations, market_prices, dimension=dim)
            capacity_flags[dim] = steady_state_ratio>1.2 # Collect ratios for all dimensions
            projected_prices[dim] = projected_prices_dim # Store projected prices for this dimension

        return projected_prices, capacity_flags # plenty of capacity still to be deployed : so just try to match the projected prices

    def decide_investments(self, evaluation_round=None, use_comparative=True):
        desirability_method = self.config.get('desirability_method', 'percentage_change') # 'percentage_change' or 'log_ratio'
        if self.verbose:
            print(f"\n=== DEBUG: {self.source_type.capitalize()} {self.source_id} deciding investments for round {evaluation_round} ===")
        
        investments_to_propose_cash_value = [] # List of (agent_id, dimension, cash_amount_to_trade, confidence)

        if not self.market: 
            print(f"Warning ({self.source_id}): No market access.")
            return []

        # DEBUG: Check available capacity
        available_capacity = self.market.source_available_capacity.get(self.source_id, {})
        if self.verbose:
            print(f"DEBUG: Available capacity: {available_capacity}")

        # --- 1. Evaluations & Price Targets ---
        own_evaluations = {} # {agent_id: {dimension: (pseudo_score, confidence_in_eval)}}
        market_prices = {}   # {agent_id: {dimension: P_current}}
        
        # Determine which agents this source will evaluate
        candidate_agent_ids = []
        if self.source_type == 'auditor':
            candidate_agent_ids = list(self.agent_profiles.keys())
        elif self.source_type == 'user_representative': # Assuming UserRep uses this structure
            candidate_agent_ids = list(self.agent_conversations.keys()) 
        else: # Fallback for other types if they use this method
            # This might need adjustment based on how other source types store their candidates
            candidate_agent_ids = list(self.market.agent_amm_params.keys())

        if not candidate_agent_ids: 
            print(f"DEBUG: No candidate agents to evaluate - returning empty list")
            return []

        # --- 1A. Fetch market prices ---
        for agent_id in candidate_agent_ids:
            market_prices[agent_id] = {}
            for dim_to_eval in self.expertise_dimensions:
                self.market.ensure_agent_dimension_initialized_in_amm(agent_id, dim_to_eval)
                amm_p = self.market.agent_amm_params[agent_id][dim_to_eval]
                price = amm_p['R'] / amm_p['T'] if amm_p['T'] > 1e-6 else \
                        self.market.agent_trust_scores[agent_id].get(dim_to_eval, 0.5)
                market_prices[agent_id][dim_to_eval] = price
                if self.verbose:
                    print(f"DEBUG: Agent {agent_id}, Dim {dim_to_eval}: Market price = {price:.4f} (R={amm_p['R']:.4f}, T={amm_p['T']:.4f})")

        # --- 1B. Batch evaluate all agents ---
        own_evaluations = self.evaluate_agents_batch(
            candidate_agent_ids,
            dimensions=self.expertise_dimensions,
            evaluation_round=evaluation_round,
            use_comparative=use_comparative
        )

        if not own_evaluations:
            if self.verbose:
                print("DEBUG: No agents successfully evaluated - returning empty list")
            return []

        projected_prices, capacity_flags = self.check_market_capacity(own_evaluations, market_prices)

        # --- 2. Determine "Target Value Holding" & "Attractiveness" ---
        attractiveness_scores = defaultdict(lambda : defaultdict(float)) # {(agent_id, dimension): attractiveness_score}
        target_value_holding_ideal = defaultdict(lambda : defaultdict(float)) # {(agent_id, dimension): ideal_cash_value_to_hold}

        # Filter evaluations to only those agents for whom we also have market prices
        # This ensures fair comparison for rank mapping and attractiveness calculation
        valid_agent_ids_for_ranking = [aid for aid in own_evaluations.keys() if aid in market_prices and \
                                       all(dim in market_prices.get(aid, {}) for dim in self.expertise_dimensions)]
        
        if self.verbose:
            print(f"DEBUG: Valid agents for ranking: {len(valid_agent_ids_for_ranking)} out of {len(own_evaluations)}")
        
        relevant_own_evals_for_ranking = {aid: own_evaluations[aid] for aid in valid_agent_ids_for_ranking}
        relevant_market_prices_for_ranking = {aid: market_prices[aid] for aid in valid_agent_ids_for_ranking}

        for agent_id, agent_eval_data in own_evaluations.items(): # Iterate all evaluated agents
            for dimension, (pseudo_score, confidence_in_eval) in agent_eval_data.items():
                if dimension not in self.expertise_dimensions: 
                    continue # Should not happen if eval_agent is correct

                key = (agent_id, dimension)
                p_current = market_prices.get(agent_id, {}).get(dimension, 0.5) # Use fetched market price
                if self.verbose:
                    print(f"DEBUG: Agent {agent_id}, Dim {dimension}: pseudo_score={pseudo_score:.4f}, confidence={confidence_in_eval:.4f}, p_current={p_current:.4f}")

                if capacity_flags.get(dimension, False):
                    p_target_effective_est = projected_prices.get(dimension, {}).get(agent_id, p_current) # Use projected price if capacity is sufficient
                    if self.verbose:
                        print(f"DEBUG: Agent {agent_id}, Dim {dimension}: p_target_raw_from_projected_prices={p_target_effective_est:.4f}")
                else:
                    p_target_effective_est = p_current # Default if agent not in ranking pool
                    if agent_id in relevant_own_evals_for_ranking: # Only calculate rank target if agent is in the valid pool
                        p_target_effective_est = self._get_target_price_from_rank_mapping(
                            agent_id, dimension, 
                            relevant_own_evals_for_ranking, # Use filtered evals for ranking
                            relevant_market_prices_for_ranking, # Use filtered prices for ranking
                            confidence_in_eval
                        )
                        if self.verbose:
                            print(f"DEBUG: Agent {agent_id}, Dim {dimension}: p_target_raw_from_rank={p_target_effective_est:.4f}")
                
                p_target_effective = p_current + (p_target_effective_est - p_current) * confidence_in_eval
                min_op_p = self.config.get('min_operational_price', 0.01)
                # max_op_p = self.config.get('max_operational_price', 0.99)
                # p_target_effective = max(min_op_p, min(max_op_p, p_target_effective))
                p_target_effective = max(min_op_p, p_target_effective)
                if self.verbose:
                    print(f"DEBUG: Agent {agent_id}, Dim {dimension}: p_target_effective={p_target_effective:.4f} (clamped at {min_op_p})")#-{max_op_p})")

                attractiveness = 0.0
                if desirability_method == 'percentage_change':
                    if p_current > 1e-6:
                        attractiveness = (p_target_effective - p_current) / p_current
                elif desirability_method == 'log_ratio':
                    if p_current > 1e-6 and p_target_effective > 1e-6:
                        attractiveness = np.log(p_target_effective / p_current)
                    elif p_target_effective > p_current: # Handle cases where p_current is near zero
                        attractiveness = 1.0 # Arbitrary large positive for strong buy signal
                    elif p_target_effective < p_current:
                        attractiveness = -1.0 # Arbitrary large negative
                else: # Default to percentage change
                     if p_current > 1e-6:
                        attractiveness = (p_target_effective - p_current) / p_current
                
                final_attractiveness = attractiveness * confidence_in_eval # Scale by confidence
                attractiveness_scores[dimension][agent_id] = final_attractiveness
                if self.verbose:
                    print(f"DEBUG: Agent {agent_id}, Dim {dimension}: raw_attractiveness={attractiveness:.4f}, final_attractiveness={final_attractiveness:.4f}")

        # Normalize positive attractiveness scores for portfolio weighting
        target_portfolio_weights = defaultdict(lambda : defaultdict(float)) # {dimension: {agent_id: weight}}
        buy_threshold = self.config.get('attractiveness_buy_threshold', 0.01)
        
        positive_attractiveness = {dim : {k: v for k,v in dim_scores.items() if v > buy_threshold} for dim, dim_scores in attractiveness_scores.items()}
        sum_positive_attractiveness = {dim : sum(dim_scores.values()) for dim, dim_scores in positive_attractiveness.items()}
        
        for dim, dim_scores in positive_attractiveness.items():
            if sum_positive_attractiveness[dim] > 1e-6:
                for agent_id, attr_score in positive_attractiveness[dim].items():
                    weight = attr_score / sum_positive_attractiveness[dim]
                    target_portfolio_weights[dim][agent_id] = weight
        if self.verbose:
            print(f"DEBUG: Target portfolio weights: {target_portfolio_weights}")
        # Calculate total potential value this source can manage
        total_portfolio_value_potential = self._calculate_total_portfolio_value_potential()
        if self.verbose:
            print(f"DEBUG: Total portfolio value potential: {total_portfolio_value_potential}")

        # Determine ideal cash value to hold for each positively attractive asset
        # ipdb.set_trace()
        min_holding_value = self.config.get('min_value_holding_per_asset', 0.0)
        
        for dim in attractiveness_scores.keys():
            for agent_id in attractiveness_scores[dim].keys():
                if dim not in target_portfolio_weights or agent_id not in target_portfolio_weights[dim]:
                    target_value_holding_ideal[dim][agent_id] = min_holding_value # Default to minimum holding value
                else:
                    weight = target_portfolio_weights[dim][agent_id]
                    ideal_value = weight * total_portfolio_value_potential[dim]
                    target_value_holding_ideal[dim][agent_id] = ideal_value

        # --- 3. Calculate Current Value of Holdings ---
        current_value_holding = defaultdict(lambda : defaultdict(float)) # {(agent_id, dimension): current_cash_value_of_shares}
        
        for agent_id_cvh, agent_market_prices_cvh in market_prices.items(): # Iterate through agents with market prices
            for dimension_cvh, p_curr_cvh in agent_market_prices_cvh.items():
                shares_held = self.market.source_investments[self.source_id].get(agent_id_cvh, {}).get(dimension_cvh, 0.0)
                current_value = shares_held * p_curr_cvh
                current_value_holding[dimension_cvh][agent_id_cvh] = current_value
                if shares_held > 0 or current_value > 0:
                    # print(f"DEBUG: Current holding - Dim {dimension_cvh}, Agent {agent_id_cvh}: {shares_held:.4f} shares * {p_curr_cvh:.4f} price = {current_value:.4f}")
                    pass

        # --- 4. Calculate Target Change in Value (Delta_Value_Target) ---
        delta_value_target_map = defaultdict(lambda : defaultdict(float)) # {(agent_id, dimension): cash_amount_to_trade}
        rebalance_aggressiveness = self.config.get('portfolio_rebalance_aggressiveness', 0.5)

        # Iterate over all keys for which we have an attractiveness score (implicitly all evaluated assets)
        for dim in attractiveness_scores.keys():
            for agent_id in attractiveness_scores[dim].keys():
                ideal_val = target_value_holding_ideal[dim][agent_id]
                current_val = current_value_holding[dim].get(agent_id, 0.0) # Default to 0 if not held
                
                delta_v = (ideal_val - current_val) * rebalance_aggressiveness
                # print(f"DEBUG: Delta calculation - Dim {dim}, Agent {agent_id}: ideal={ideal_val:.4f}, current={current_val:.4f}, delta_raw={(ideal_val - current_val):.4f}, delta_scaled={delta_v:.4f}")
                
                # Apply a threshold to delta_v to avoid tiny trades
                min_trade_threshold = self.config.get('min_delta_value_trade_threshold', 0.1)
                if abs(delta_v) > min_trade_threshold: # e.g., trade if value change > $0.1
                    delta_value_target_map[dim][agent_id] = delta_v
                    # print(f"DEBUG: Delta above threshold ({min_trade_threshold}) - Including in trade map: {delta_v:.4f}")
                else:
                    # print(f"DEBUG: Delta below threshold ({min_trade_threshold}) - Skipping: {delta_v:.4f}")
                    pass
        if self.verbose:
            print(f"DEBUG: Delta value target map: {dict(delta_value_target_map)}")

        # --- 5. Calculate delta_value_target_scale based on portfolio size and confidence ---
        uninvested_capacity = self.market.source_available_capacity[self.source_id]
        total_portfolio_value_potential = total_portfolio_value_potential
        total_proposed_investments = {dim : sum(max(v,0.0) for v in delta_value_target_map[dim].values()) for dim in delta_value_target_map.keys()}
        
        if self.verbose:
            print(f"DEBUG: Uninvested capacity: {uninvested_capacity}")
            print(f"DEBUG: Total proposed investments by dimension: {dict(total_proposed_investments)}")
        
        for dim in delta_value_target_map.keys():
            if total_proposed_investments[dim] > 0:
                investment_scale = self.config.get('invest_multiplier', 0.2) # Scale factor for investment aggressiveness
                investment_scale_pot = min(total_portfolio_value_potential.get(dim, 0.0)*investment_scale / total_proposed_investments[dim], 1.0) if total_proposed_investments[dim] > 0 else 1.0
                investment_scale_cap = min(uninvested_capacity.get(dim, 0.0)/(total_proposed_investments[dim]*investment_scale_pot), 1.0) if total_proposed_investments[dim]*investment_scale_pot > 0 else 1.0
                final_investment_scale = investment_scale_pot * investment_scale_cap
                
                # print(f"DEBUG: Scaling for dim {dim}: base_scale={investment_scale}, scale_pot={investment_scale_pot:.4f}, scale_cap={investment_scale_cap:.4f}, final_scale={final_investment_scale:.4f}")
                
                for agent_id, cash_amount in delta_value_target_map[dim].items():
                    scaled_cash_amount = cash_amount * final_investment_scale
                    delta_value_target_map[dim][agent_id] = scaled_cash_amount
                    # print(f"DEBUG: Final scaling - Dim {dim}, Agent {agent_id}: {cash_amount:.4f} -> {scaled_cash_amount:.4f}")

        # --- 6. Prepare list of (agent_id, dimension, cash_amount_to_trade, confidence) ---
        # The TrustMarket.process_investments will convert this cash_amount to shares
        # and handle actual cash availability for buys.
        
        for dim in delta_value_target_map.keys():
            for agent_id, cash_amount in delta_value_target_map[dim].items():
                confidence = 0.5 # Default confidence
                if agent_id in own_evaluations and dim in own_evaluations[agent_id]:
                    confidence = own_evaluations[agent_id][dim][1]
                
                investments_to_propose_cash_value.append(
                    (agent_id, dim, cash_amount, confidence)
                )
        
        print(f"DEBUG: Final investments list length: {len(investments_to_propose_cash_value)}")
        if investments_to_propose_cash_value:
            print(f"DEBUG: {self.source_type.capitalize()} {self.source_id} prepared {len(investments_to_propose_cash_value)} cash-value based actions.")
            for i, (aid, dim, amount, conf) in enumerate(investments_to_propose_cash_value):
                print(f"DEBUG: Investment {i+1}: Agent {aid}, Dim {dim}, Amount {amount:.4f}, Confidence {conf:.4f}")
        else:
            print(f"DEBUG: {self.source_type.capitalize()} {self.source_id} found no cash-value actions to take.")
            
        if self.verbose:
            print(f"=== DEBUG: End of decide_investments for {self.source_id} ===\n")
        return investments_to_propose_cash_value

    def evaluate_agents_batch(self, agent_ids, dimensions=None, evaluation_round=None, use_comparative=True):
        """Parallel batch evaluation wrapper for UserRep."""
        return super().evaluate_agents_batch(
            agent_ids=agent_ids,
            dimensions=dimensions,
            evaluation_round=evaluation_round,
            use_comparative=use_comparative
        )