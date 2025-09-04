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
import json
import copy
import ipdb

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
            expertise_dimensions = ["Communication_Quality", "Problem_Resolution", "Value_Alignment", "Manipulation_Resistance", "Adaptability", "Safety_Security", "Factual_Correctness", "Process_Reliability", "Transparency", "Trust_Calibration"]
            confidence = {"Communication_Quality": 0.8, "Problem_Resolution": 0.8, "Value_Alignment": 0.7, "Manipulation_Resistance": 0.7, "Adaptability": 0.7, "Safety_Security": 0.7}

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

    def __init__(self, source_id, user_segment, representative_profile, market=None, api_key=None, api_model_name='gemini-2.5-flash', verbose=False, api_provider='gemini', openai_api_key=None, memory_length_n: int = 3):
        super().__init__(source_id, user_segment, representative_profile, market, verbose=verbose)
        # ensure base InformationSource memory length set
        self.memory_length_n = memory_length_n

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
        
        # Flag to track if detailed analysis is currently active
        self._detailed_analysis_active = False

        # Configuration specific to this Rep's strategy
        self.config = {
            'comparison_agents_per_target': 3, # Compare against fewer agents than auditor?
            'min_conversations_required': 3,   # Min conversations needed for comparison
            'new_evaluation_weight': 0.7,      # Weight for new comparative scores vs prior derived score
            'memory_length_n': memory_length_n,              # How many past evaluations to remember
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
            'min_delta_value_trade_threshold': 5,
            'investment_scale': 0.2,
            'rank_correction_strength': 0.5,
            'market_growth_factor': 1.0,
            'quality_concentration_power': 2.0,
            'max_eval_trials': 1,

            # Bayesian Inference Parameters
            'confidence_to_kappa_scale_factor': 50.0, # M parameter for converting confidence to precision
            'decay_rate': 0.5, # How quickly old evidence is forgotten
            'likelihood_strength_factor': 1.0, # Lower value = auditor evaluations have moderate influence
            
            # Monte Carlo Simulation Parameters
            'monte_carlo_trials': 50, # Number of Monte Carlo trials for risk assessment
            'use_monte_carlo': True, # Whether to use Monte Carlo for investment decisions
            'mc_seed_base': 42, # Base seed for Monte Carlo simulations
            'investment_method': 'capital_projection', # 'capital_projection' or 'rank_mapping'
            # Optimization-based allocation parameters (opt-in)
            'optimization_backend': 'l2_values',
            'optimizer_enabled': True,  # do not change existing behavior unless explicitly enabled
            'optimizer_lambda_prox': 0.1,   # proximal weight to damp moves (maps to rebalance aggressiveness)
            'optimizer_risk_rho': 0.01,      # penalty weight for buy-side risk
            'optimizer_turnover_tau': 0.0,  # L1 turnover penalty (0 = off)
            'optimizer_grid_points': 31,    # resolution for per-asset 1D search
            'optimizer_joint_budget': True, # use dual (μ) search across agents per-dimension
            'optimizer_solve_on_sim_state': False, # if False, solve against original snapshot R,T,S
            'optimizer_zeroing_enabled': True,       # allow force-zero small positions when target ~ 0
            'optimizer_zero_target_rel': 0.001,      # target <= 0.5% of portfolio triggers zeroing check
            'optimizer_small_holding_rel': 0.01,     # holding <= 1% of portfolio eligible for zeroing
            'optimizer_respect_investment_scale_cap': True, # cap fresh cash per round using investment_scale
        }
        self.num_trials = self.config.get('max_eval_trials', 1)
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
    
    def _get_additional_context(self, agent_a_id, agent_b_id, evaluation_round):
        """
        UserRep context includes its own past evaluations and the regulator's last evaluation.
        """
        # 1. Get own past evaluations (from super)
        own_context = super()._get_additional_context(agent_a_id, agent_b_id, evaluation_round)

        # 2. Get regulator evaluations for these agents
        regulator_context = ""
        try:
            if self.market and 'regulator' in self.market.information_sources:
                regulator = self.market.information_sources['regulator']
                reg_evals = regulator._get_recent_pair_evaluations(agent_a_id, agent_b_id)[:1] # Only the most recent one
                if reg_evals:
                    eval_snippets = []
                    for ev in reg_evals:
                        rnd = ev.get('round', 'N/A')
                        relative_round_str = ""
                        if isinstance(rnd, int) and isinstance(evaluation_round, int):
                            diff = evaluation_round - rnd
                            if diff == 0: relative_round_str = " (this round)"
                            elif diff == 1: relative_round_str = " (last round)"
                            else: relative_round_str = f" ({diff} rounds ago)"

                        reasoning = ev.get('reasoning', {})
                        rating = ev.get('derived_scores', {}).get(agent_a_id, {})
                        confidence = ev.get('confidence', 0)
                        ratings_and_reasoning = {dim: {
                            'rating': (rating.get(dim, 0.5) - 0.5) * self.batch_evaluator.rating_scale * 2,
                            'reasoning': reasoning.get(dim, "N/A"),
                            'confidence': confidence
                        } for dim in rating.keys()}

                        eval_snippets.append(f"Round {rnd}{relative_round_str}: {json.dumps(ratings_and_reasoning)}")
                    
                    if eval_snippets:
                        regulator_context = "\n\nFor additional context, here is the most recent evaluation from the Regulator, " + \
                        "a very trusted source which has much more information than you while evaluating (such as the agent prompts, " + \
                        "profiles and its conversations with more users). You should probably trust its evaluation more than your own. " + "\n".join(eval_snippets)#\
                        # "However, the regulator evaluations are typically a bit old (check the round number of the evaluation to make sure) and may " + \
                        # "not reflect the current state of the agents, given that the agent's behavior can change over time. " + \
                        # "So if you have conclusive evidence that the regulator's evaluation and reasoning aren't reflective of the agent" + \
                        # " interactions you observed above and believe that the agent behavior has likely changed since the regulator's " + \
                        # "evaluation, feel free to override its evaluation. But, if the regulator's evaluation and reasoning " + \
                        # "seem at all plausible, it's likely that the agent hasn't changed much and thus you should just trust" + \
                        # " the regulator evaluations :\n" + "\n".join(eval_snippets)
                        # . Based on that, use your judgment to evaluate how much/whether to incorporate the regulator's feedback
        except Exception as e:
            if self.verbose:
                print(f"DEBUG ({self.source_id}): Could not fetch regulator context. Error: {e}")

        return f"{own_context}{regulator_context}".strip()

    def _compare_pair(self, aid, oid, dimensions, additional_context: str = "") -> Optional[Tuple[int, int, dict, dict]]:
        """UserRep's implementation of a pairwise comparison based on conversations."""
        if not (self._agent_has_comparable_data(aid) and self._agent_has_comparable_data(oid)):
            return None # Incomparable

        # Check if detailed analysis is requested
        return_raw = self._detailed_analysis_active

        comparison_results = self.evaluator.compare_agent_batches(
            self.get_agent_conversations(aid), aid,
            self.get_agent_conversations(oid), oid,
            dimensions,
            additional_context=additional_context)

        if not comparison_results:
            return None

        derived_scores, confidences = self.evaluator.get_agent_scores_new(comparison_results, aid, oid)
        # confidences = super()._extract_comparison_confidences(comparison_results, aid, oid)
        
        # Return 5-tuple if detailed analysis is active, 4-tuple otherwise
        return (aid, oid, derived_scores, confidences, comparison_results)
        # else:
        #     return (aid, oid, derived_scores, confidences)

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

    def _calculate_total_portfolio_value_potential(self, total_available_cash=None, amm_params=None, source_investments=None):
        """
        Calculates the sum of the current market value of all shares held by this source
        PLUS all available (uninvested) cash capacity of this source.
        This represents the total value this source could theoretically manage.
        """
        total_value_of_holdings = defaultdict(lambda : 0.0) # {dimension: total_value}
        if total_available_cash is None:
            total_available_cash = self.market.source_available_capacity[self.source_id]
        if source_investments is not None or self.source_id in self.market.source_investments:
            if source_investments is None:
                source_investments = self.market.source_investments[self.source_id]
            if amm_params is None:
                amm_params = self.market.agent_amm_params
            # Sum market value of current share holdings
            # source_investments structure: self.source_investments[source_id][agent_id][dimension] = shares
            if self.source_id in source_investments:
                for agent_id, dims_data in source_investments[self.source_id].items():
                    for dimension, shares_held in dims_data.items():
                        # if shares_held > 1e-5:
                        self.market.ensure_agent_dimension_initialized_in_amm(agent_id, dimension) # Ensure AMM params exist
                        amm_param = amm_params[agent_id][dimension]
                        if amm_param['T'] > 1e-6: # Avoid division by zero if T is tiny
                            price = amm_param['R'] / amm_param['T']
                            total_value_of_holdings[dimension] += shares_held * price
                            # else: if T is zero, price is undefined/infinite, value of shares is complex.
                            # For simplicity, if T is zero, those shares are currently "unpriceable" by this AMM.

        # Sum available cash from all dimensions for this source
        # source_available_capacity structure: self.source_available_capacity[source_id][dimension] = cash

        total_potential = {dim: total_value_of_holdings[dim] + total_available_cash[dim] for dim in total_available_cash}
        if self.verbose:
            print(f"USER REP ({self.source_id}): Total potential: {total_potential}")
        # Ensure a minimum potential to avoid issues if source starts with no cash/shares
        # return max(total_potential, self.config.get('min_portfolio_value_potential', 100.0))
        return total_potential

    def _get_steady_state_capital(self, market_prices, dimension):
        """
        Get the steady state capital for a given dimension.
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
        # steady_state_capital = total_potential_capital * growth_factor
        
        # Step 2: Current total capital in market
        current_total_market_capital = 0
        for agent_id in market_prices:
            if dimension in self.market.agent_amm_params[agent_id]:
                # Total capital locked in AMM = R (reserves)
                current_total_market_capital += self.market.agent_amm_params[agent_id][dimension]['R']
        steady_state_capital = current_total_market_capital + total_potential_capital * growth_factor # Include current capital in the market
        capacity_ratio = steady_state_capital/current_total_market_capital

        # Step 3: Calculate current capital shares
        current_agent_capital = {
            agent_id: self.market.agent_amm_params[agent_id][dimension]['R']
            for agent_id in market_prices
            if dimension in self.market.agent_amm_params.get(agent_id, {})
        }
        current_total_market_capital_for_shares = sum(current_agent_capital.values())

        current_capital_shares = {}
        if current_total_market_capital_for_shares > 1e-6:
            current_capital_shares = {
                aid: cap / current_total_market_capital_for_shares
                for aid, cap in current_agent_capital.items()
            }
        return steady_state_capital, capacity_ratio, current_capital_shares

    def _project_steady_state_prices(self, own_evaluations, dimension, steady_state_capital, current_capital_shares=None):
        """
        Project what market prices will be at steady state based on:
        1. Expected total capital deployment
        2. Quality-based distribution of that capital
        """

        # Step 3: Project capital distribution based on quality scores
        # Use own evaluations as best estimate of true quality
        quality_scores = {
            agent_id: eval_data[dimension]
            for agent_id, eval_data in own_evaluations.items()
            if dimension in eval_data
        }
        
        # Convert quality to expected capital share
        # Higher quality agents should attract disproportionately more capital
        concentration_power = self.config.get('quality_concentration_power', 2.0)
        
        quality_powered = {
            aid: q[0] ** concentration_power 
            for aid, q in quality_scores.items()
        }
        total_quality_powered = sum(quality_powered.values())

        # Expected share of steady-state capital for each agent
        evaluation_based_capital_shares = {
            aid: qp / total_quality_powered 
            for aid, qp in quality_powered.items()
        }

        if current_capital_shares is not None:
            interpolated_shares = {}
            all_agent_ids = set(quality_scores.keys()) | set(current_capital_shares.keys())

            for aid in all_agent_ids:
                # Confidence acts as the interpolation factor. High confidence -> lean towards own eval.
                confidence = quality_scores.get(aid, (0.5, 0.0))[1]
                eval_share = evaluation_based_capital_shares.get(aid, 0.0)
                market_share = current_capital_shares.get(aid, 0.0)
                weight = confidence ** 2
                interpolated_shares[aid] = (weight * eval_share) + ((1 - weight) * market_share)

            # 3d. Normalize the interpolated shares to ensure they sum to 1.
            total_interpolated_share = sum(interpolated_shares.values())
            expected_capital_shares = {
                aid: share / total_interpolated_share
                for aid, share in interpolated_shares.items()
            }
        else:
            expected_capital_shares = evaluation_based_capital_shares
        
        # Step 4: Project steady-state prices
        projected_prices = {}
        projected_capital_shares = {}
        
        for agent_id in expected_capital_shares:
            # Expected capital for this agent at steady state
            expected_capital = steady_state_capital * expected_capital_shares[agent_id]
            projected_capital_shares[agent_id] = expected_capital
            
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
        
        return projected_prices, projected_capital_shares

    def check_market_capacity(self, own_evaluations, market_prices, evaluation_round=None):
        """
        Checks if the source has enough capacity to invest based on its evaluations and market prices.
        If not, it will print a warning and return False.
        """
        if self.config.get('use_monte_carlo', True):
            num_trials = self.config.get('monte_carlo_trials', 50)
            return self._monte_carlo_check_market_capacity(own_evaluations, market_prices, num_trials, evaluation_round=evaluation_round)
        else:
            capacity_flags = {} # Collect ratios for all dimensions
            projected_prices = {} # {agent_id: projected_price}
            projected_capital_shares = {} # {agent_id: projected_capital_share}
            for dim in self.expertise_dimensions:
                steady_state_capital, steady_state_ratio, current_capital_shares = self._get_steady_state_capital(market_prices, dimension=dim)
                projected_prices_dim, projected_capital_shares_dim = self._project_steady_state_prices(own_evaluations, dimension=dim, steady_state_capital=steady_state_capital, current_capital_shares=current_capital_shares)
                capacity_flags[dim] = steady_state_ratio>1.2 # Collect ratios for all dimensions
                projected_prices[dim] = projected_prices_dim # Store projected prices for this dimension
                projected_capital_shares[dim] = projected_capital_shares_dim # Store projected capital shares for this dimension
            return projected_prices, projected_capital_shares, capacity_flags # plenty of capacity still to be deployed : so just try to match the projected prices

    def decide_investments(self, evaluation_round=None, use_comparative=True, analysis_mode=False, detailed_analysis=False):
        """
        The main decision-making loop for the user_rep.
        1. Evaluates all agents to get up-to-date scores.
        """
        # Optional delegation to the optimized path
        if self.config.get('optimizer_enabled', False):
            return self.decide_investments_optimized(
                evaluation_round=evaluation_round,
                use_comparative=use_comparative,
                analysis_mode=analysis_mode,
                detailed_analysis=detailed_analysis,
            )

        desirability_method = self.config.get('desirability_method', 'percentage_change')
        if self.verbose:
            print(f"\n=== DEBUG: {self.source_type.capitalize()} {self.source_id} deciding investments for round {evaluation_round} ===")
        
        # Store detailed_analysis flag for use in _compare_pair
        self._detailed_analysis_active = detailed_analysis
        
        investments_to_propose_cash_value = [] # List of (agent_id, dimension, cash_amount_to_trade, confidence)
        analysis_data = {} if analysis_mode else None

        if not self.market: 
            print(f"Warning ({self.source_id}): No market access.")
            return [] if not analysis_mode else ([], analysis_data)

        # DEBUG: Check available capacity
        available_capacity = self.market.source_available_capacity.get(self.source_id, {})
        if self.verbose:
            print(f"DEBUG: Available capacity: {available_capacity}")

        # --- 1. Evaluations & Price Targets ---
        own_evaluations = {} # {agent_id: {dimension: (pseudo_score, confidence_in_eval)}}
        market_prices = {}   # {agent_id: {dimension: P_current}}

        candidate_agent_ids = list(self.agent_conversations.keys()) 
            
        if not candidate_agent_ids: 
            print(f"DEBUG: No candidate agents to evaluate - returning empty list")
            return [] if not analysis_mode else ([], analysis_data)

        # --- 1A. Fetch market prices ---
        market_prices, market_capital_holdings = self.market.get_market_prices(candidate_agent_ids=candidate_agent_ids, dimensions=self.expertise_dimensions, verbose=self.verbose)

        comparison_log = []

        # Check for cached evaluations first
        if self.use_cached_evaluations and evaluation_round is not None:
            cached_eval, cached_comparison_log = self.get_cached_evaluation(evaluation_round)
            if cached_eval:
                print(f"{self.source_type.upper()} ({self.source_id}): Using cached evaluations for round {evaluation_round}")
                own_evaluations = cached_eval
                if detailed_analysis:
                    comparison_log = cached_comparison_log
            else:
                if self.verbose:
                    print(f"{self.source_type.upper()} ({self.source_id}): No cached evaluations found for round {evaluation_round}, falling back to LLM evaluation")
        
        # If no cached evaluations available, run normal evaluation
        if not own_evaluations:
            # --- 1B. Batch evaluate all agents ---
            evaluation_result = self.evaluate_agents_batch(
                candidate_agent_ids,
                dimensions=self.expertise_dimensions,
                evaluation_round=evaluation_round,
                use_comparative=use_comparative,
                analysis_mode=analysis_mode,
                detailed_analysis=detailed_analysis
            )

            # Handle different return formats based on detailed_analysis flag
            if detailed_analysis:
                own_evaluations, comparison_log = evaluation_result
            else:
                own_evaluations = evaluation_result
                comparison_log = []
        if not own_evaluations:
            if self.verbose:
                print("DEBUG: No agents successfully evaluated - returning empty list")
            return [] if not analysis_mode else ([], analysis_data)

        projected_prices, projected_capital_shares, capacity_flags = self.check_market_capacity(own_evaluations, market_prices, evaluation_round=evaluation_round)
        risk = self.compute_risk(projected_capital_shares, current_capital_holdings=market_capital_holdings, type='relative_capital')
        use_capital_projection = {dim: self.config.get('investment_method', 'capital_projection') == 'capital_projection' or capacity_flags[dim] for dim in capacity_flags.keys()}

        # --- 3. Determine "Target Value Holding" & "Attractiveness" ---
        attractiveness_scores = defaultdict(lambda : defaultdict(float))
        target_value_holding_ideal = defaultdict(lambda : defaultdict(float))

        if self.config.get('investment_method', 'capital_projection') == 'rank_mapping':
            valid_agent_ids_for_ranking = [aid for aid in own_evaluations.keys() if aid in market_prices and \
                                        all(dim in market_prices[aid] for dim in self.expertise_dimensions)]
            
            if self.verbose:
                print(f"DEBUG: Valid agents for ranking: {len(valid_agent_ids_for_ranking)} out of {len(own_evaluations)}")
                print(f"DEBUG: Valid agent IDs: {valid_agent_ids_for_ranking}")
            
            relevant_own_evals_for_ranking = {aid: own_evaluations[aid] for aid in valid_agent_ids_for_ranking}
            relevant_market_prices_for_ranking = {aid: market_prices[aid] for aid in valid_agent_ids_for_ranking}

        for agent_id, agent_eval_data in own_evaluations.items():
            if self.verbose:
                print(f"DEBUG: Processing attractiveness for agent {agent_id}")
            for dimension, (pseudo_score, confidence_in_eval) in agent_eval_data.items():
                if dimension not in self.expertise_dimensions: 
                    if self.verbose:
                        print(f"DEBUG: Skipping dimension {dimension} (not in expertise)")
                    continue

                p_current = market_capital_holdings.get(agent_id, {}).get(dimension, 0.5)
                if self.verbose:
                    print(f"DEBUG: Agent {agent_id}, Dim {dimension}: pseudo_score={pseudo_score:.4f}, confidence={confidence_in_eval:.4f}, p_current={p_current:.4f}")

                if use_capital_projection[dimension]:
                    p_target_effective = projected_capital_shares[dimension][agent_id][0]
                    if self.verbose:
                        print(f"DEBUG: Agent {agent_id}, Dim {dimension}: p_target_raw_from_projected_prices={p_target_effective:.4f}")
                else:
                    p_target_effective_est = p_current
                    if agent_id in relevant_own_evals_for_ranking:
                        p_target_effective_est = self._get_target_price_from_rank_mapping(
                            agent_id, dimension, 
                            relevant_own_evals_for_ranking,
                            relevant_market_prices_for_ranking,
                            confidence_in_eval
                        )
                        if self.verbose:
                            print(f"DEBUG: Agent {agent_id}, Dim {dimension}: p_target_raw_from_rank={p_target_effective_est:.4f}")
                        p_target_effective = p_current + (p_target_effective_est - p_current) * confidence_in_eval
                
                min_op_p = self.config.get('min_operational_price', 0.01)
                p_target_effective = max(min_op_p, p_target_effective)
                if self.verbose:
                    print(f"DEBUG: Agent {agent_id}, Dim {dimension}: p_target_effective={p_target_effective:.4f} (clamped between {min_op_p})")

                attractiveness = 0.0
                if desirability_method == 'percentage_change':
                    if p_current > 1e-6:
                        attractiveness = (p_target_effective - p_current) / p_current
                elif desirability_method == 'log_ratio':
                    if p_current > 1e-6 and p_target_effective > 1e-6:
                        attractiveness = np.log(p_target_effective / p_current)
                    elif p_target_effective > p_current:
                        attractiveness = 1.0
                    elif p_target_effective < p_current:
                        attractiveness = -1.0
                else:
                     if p_current > 1e-6:
                        attractiveness = (p_target_effective - p_current) / p_current
                
                final_attractiveness = attractiveness / (risk[dimension][agent_id] + 1)
                attractiveness_scores[dimension][agent_id] = final_attractiveness
                if self.verbose:
                    print(f"DEBUG: Agent {agent_id}, Dim {dimension}: raw_attractiveness={attractiveness:.4f}, final_attractiveness={final_attractiveness:.4f}")

                if analysis_mode:
                    if agent_id not in analysis_data:
                        analysis_data[agent_id] = {}
                    analysis_data[agent_id][dimension] = {
                        'projected_prices': projected_prices[dimension][agent_id][0],
                        'projected_capital_shares': projected_capital_shares[dimension][agent_id][0],
                        'p_target_effective': p_target_effective,
                        'final_attractiveness': final_attractiveness,
                        'pseudo_score': pseudo_score,
                        'confidence_in_eval': confidence_in_eval,
                        'p_current': p_current
                    }

        target_portfolio_weights = defaultdict(lambda : defaultdict(float))
        buy_threshold = self.config.get('attractiveness_buy_threshold', 0.01)
        
        positive_attractiveness = {dim : {k: v for k,v in dim_scores.items() if v > buy_threshold} for dim, dim_scores in attractiveness_scores.items()}
        sum_positive_attractiveness = {dim : sum(dim_scores.values()) for dim, dim_scores in positive_attractiveness.items()}
        
        for dim, dim_scores in positive_attractiveness.items():
            if sum_positive_attractiveness[dim] > 1e-6:
                for agent_id, attr_score in dim_scores.items():
                    weight = attr_score / sum_positive_attractiveness[dim]
                    target_portfolio_weights[dim][agent_id] = weight
        
        total_portfolio_value_potential = self._calculate_total_portfolio_value_potential()
        min_holding_value = self.config.get('min_value_holding_per_asset', 0.0)
        
        for dim in attractiveness_scores.keys():
            for agent_id in attractiveness_scores[dim].keys():
                if dim not in target_portfolio_weights or agent_id not in target_portfolio_weights[dim]:
                    target_value_holding_ideal[dim][agent_id] = min_holding_value
                else:
                    weight = target_portfolio_weights[dim][agent_id]
                    ideal_value = weight * total_portfolio_value_potential[dim]
                    target_value_holding_ideal[dim][agent_id] = ideal_value

        # --- 4. Calculate Current Value of Holdings & Target Change ---
        current_value_holding = defaultdict(lambda : defaultdict(float))
        delta_value_target_map = defaultdict(lambda : defaultdict(float))
        rebalance_aggressiveness = self.config.get('portfolio_rebalance_aggressiveness', 0.5)

        for agent_id_cvh, agent_market_prices_cvh in market_prices.items():
            for dimension_cvh, p_curr_cvh in agent_market_prices_cvh.items():
                shares_held = self.market.source_investments[self.source_id].get(agent_id_cvh, {}).get(dimension_cvh, 0.0)
                current_value = shares_held * p_curr_cvh
                current_value_holding[dimension_cvh][agent_id_cvh] = current_value

        for dim in attractiveness_scores.keys():
            for agent_id in attractiveness_scores[dim].keys():
                ideal_val = target_value_holding_ideal[dim][agent_id]
                current_val = current_value_holding[dim].get(agent_id, 0.0)
                
                delta_v = (ideal_val - current_val) * rebalance_aggressiveness
                if self.verbose:
                    print(f"DEBUG: Delta calculation - Dim {dim}, Agent {agent_id}: ideal={ideal_val:.4f}, current={current_val:.4f}, delta_raw={(ideal_val - current_val):.4f}, delta_scaled={delta_v:.4f}")
                
                min_trade_threshold = self.config.get('min_delta_value_trade_threshold', 5)
                if abs(delta_v) > total_portfolio_value_potential[dim] * min_trade_threshold / 100 or ideal_val < 0.01 * total_portfolio_value_potential[dim]:
                    delta_value_target_map[dim][agent_id] = delta_v
                    if abs(delta_v) < total_portfolio_value_potential[dim] * min_trade_threshold / 100 and ideal_val < 0.01 * total_portfolio_value_potential[dim]:
                        delta_value_target_map[dim][agent_id] = ideal_val - current_val
                    if self.verbose:
                        print(f"DEBUG: Delta above threshold ({total_portfolio_value_potential[dim] * min_trade_threshold / 100}) | Target Value: {ideal_val:.4f} | Current Value: {current_val:.4f} | Including in trade map: {delta_v:.4f}")
                else:
                    if self.verbose:
                        print(f"DEBUG: Delta below threshold ({min_trade_threshold}) - Skipping: {delta_v:.4f}")

        # --- 5. Scale investments based on available capacity ---
        uninvested_capacity = self.market.source_available_capacity[self.source_id]
        total_proposed_investments = {dim: sum(max(v, 0.0) for v in delta_value_target_map[dim].values()) for dim in delta_value_target_map.keys()}

        for dim in delta_value_target_map.keys():
            if total_proposed_investments[dim] > 0:
                investment_scale = self.config.get('investment_scale', 0.2)
                investment_scale_pot = min(total_portfolio_value_potential[dim] * investment_scale / total_proposed_investments[dim], 1.0)
                divestments_sum = abs(sum(min(v,0.0) for v in delta_value_target_map[dim].values()))
                investment_scale_cap = min((uninvested_capacity[dim]+divestments_sum) / (total_proposed_investments[dim] * investment_scale_pot), 1.0)
                final_investment_scale = investment_scale_pot * investment_scale_cap
                
                if self.verbose:
                    print(f"DEBUG: Scaling for dim {dim}: base_scale={investment_scale}, scale_pot={investment_scale_pot:.4f}, scale_cap={investment_scale_cap:.4f}, final_scale={final_investment_scale:.4f}")
                
                for agent_id, cash_amount in delta_value_target_map[dim].items():
                    if cash_amount > 0:
                        scaled_cash_amount = cash_amount * final_investment_scale
                        delta_value_target_map[dim][agent_id] = scaled_cash_amount

        # --- 6. Prepare final list of investments ---
        for dim in delta_value_target_map.keys():
            for agent_id, cash_amount in delta_value_target_map[dim].items():
                confidence = own_evaluations.get(agent_id, {}).get(dim, (0.5, 0.5))[1]
                investments_to_propose_cash_value.append(
                    (agent_id, dim, cash_amount, confidence)
                )
        
        investments_to_propose_cash_value.sort(key=lambda x: x[2])
        
        print(f"DEBUG: Final investments list length: {len(investments_to_propose_cash_value)}")
        if investments_to_propose_cash_value:
            print(f"DEBUG: {self.source_type.capitalize()} {self.source_id} prepared {len(investments_to_propose_cash_value)} cash-value based actions.")
            for i, (aid, dim, amount, conf) in enumerate(investments_to_propose_cash_value):
                print(f"DEBUG: Investment {i+1}: Agent {aid}, Dim {dim}, Amount {amount:.4f}, Confidence {conf:.4f}")
        else:
            print(f"DEBUG: {self.source_type.capitalize()} {self.source_id} found no cash-value actions to take.")
            
        if self.verbose:
            print(f"=== DEBUG: End of decide_investments for {self.source_id} ===\n")
            
        # --- CLEANUP ---
        # Reset detailed analysis flag after evaluation
        self._detailed_analysis_active = False

        if analysis_mode or detailed_analysis:
            if detailed_analysis:
                analysis_data['comparison_log'] = comparison_log
            # Always include own_evaluations in analysis_data for caching purposes
            analysis_data['own_evaluations'] = own_evaluations
            return investments_to_propose_cash_value, analysis_data
        
        return investments_to_propose_cash_value

    def decide_investments_optimized(self, evaluation_round=None, use_comparative=True, analysis_mode=False, detailed_analysis=False):
        """
        Alternative decision method that preserves the full evaluation → target pipeline,
        but translates targets to trades via an AMM-aware optimization.
        Does not modify the original decide_investments; opt-in usage only.
        """
        desirability_method = self.config.get('desirability_method', 'percentage_change')
        if self.verbose:
            print(f"\n=== DEBUG(OPT): {self.source_type.capitalize()} {self.source_id} deciding investments for round {evaluation_round} ===")

        self._detailed_analysis_active = detailed_analysis
        investments_to_propose = []
        analysis_data = {} if analysis_mode else None

        if not self.market:
            print(f"Warning ({self.source_id}): No market access.")
            return [] if not analysis_mode else ([], analysis_data)

        # 1. Gather candidate agents
        candidate_agent_ids = list(self.agent_conversations.keys())
        if not candidate_agent_ids:
            return [] if not analysis_mode else ([], analysis_data)

        # 1A. Market prices and current capital
        market_prices, market_capital_holdings = self.market.get_market_prices(
            candidate_agent_ids=candidate_agent_ids,
            dimensions=self.expertise_dimensions,
            verbose=self.verbose
        )

        comparison_log = []
        own_evaluations = {}
        if self.use_cached_evaluations and evaluation_round is not None:
            cached_eval, cached_comparison_log = self.get_cached_evaluation(evaluation_round)
            if cached_eval:
                print(f"{self.source_type.upper()} ({self.source_id}): Using cached evaluations for round {evaluation_round}")
                own_evaluations = cached_eval
                if detailed_analysis:
                    comparison_log = cached_comparison_log
        if not own_evaluations:
            eval_result = self.evaluate_agents_batch(
                candidate_agent_ids,
                dimensions=self.expertise_dimensions,
                evaluation_round=evaluation_round,
                use_comparative=use_comparative,
                analysis_mode=analysis_mode,
                detailed_analysis=detailed_analysis
            )
            if detailed_analysis:
                own_evaluations, comparison_log = eval_result
            else:
                own_evaluations = eval_result
        if not own_evaluations:
            return [] if not analysis_mode else ([], analysis_data)

        # 2. Capacity projection and risk
        projected_prices, projected_capital_shares, capacity_flags = self.check_market_capacity(own_evaluations, market_prices, evaluation_round=evaluation_round)
        risk_iter = risk = self.compute_risk(projected_capital_shares, current_capital_holdings=market_capital_holdings, type='relative_capital')

        # 3. Current value holdings
        current_value_holding = defaultdict(lambda: defaultdict(float))
        for dimension in self.expertise_dimensions:
            for agent_id in own_evaluations.keys():
                self.market.ensure_agent_dimension_initialized_in_amm(agent_id, dimension)
                amm = self.market.agent_amm_params[agent_id][dimension]
                p_curr = amm['R'] / max(amm['T'], 1e-9)
                S = self.market.source_investments[self.source_id].get(agent_id, {}).get(dimension, 0.0)
                current_value_holding[dimension][agent_id] = S * p_curr

        # 4. Iterative reweighting with AMM-consistent optimization (capital-based attractiveness)
        trades_by_dim = {dim: {aid: 0.0 for aid in own_evaluations.keys()} for dim in self.expertise_dimensions}

        buy_threshold = self.config.get('attractiveness_buy_threshold', 0.01)
        lam = float(self.config.get('optimizer_lambda_prox', 0.5))
        rho = float(self.config.get('optimizer_risk_rho', 0.5))
        tau = float(self.config.get('optimizer_turnover_tau', 0.0))
        grid_pts = int(self.config.get('optimizer_grid_points', 31))
        inner_iters = int(self.config.get('optimizer_inner_iters', 10))
        relax = float(self.config.get('optimizer_relaxation', 0.4))
        recompute_targets = bool(self.config.get('optimizer_recompute_targets', False))
        recompute_risk = bool(self.config.get('optimizer_recompute_risk', False))

        zeroing = self.config.get('optimizer_zeroing_enabled', True)
        zero_target_rel = self.config.get('optimizer_zero_target_rel', 0.001)
        small_hold_rel = self.config.get('optimizer_small_holding_rel', 0.01)
        
        proj_caps = projected_capital_shares
        

        def simulate_state(trades_map):
            sim_amm_params = copy.deepcopy(self.market.agent_amm_params)
            sim_S = defaultdict(lambda: defaultdict(float))
            sim_uninvested_capacity = copy.deepcopy(self.market.source_available_capacity[self.source_id])
            for aid in own_evaluations.keys():
                for dim in self.expertise_dimensions:
                    self.market.ensure_agent_dimension_initialized_in_amm(aid, dim)
                    amm0 = self.market.agent_amm_params[aid][dim]
                    R0, T0 = amm0['R'], amm0['T']
                    S0 = self.market.source_investments[self.source_id].get(aid, {}).get(dim, 0.0)
                    cash = trades_map.get(dim, {}).get(aid, 0.0)
                    sell_cap, _ = self._compute_trade_bounds(R0, T0, S0)
                    buy_cap = sim_uninvested_capacity[dim]
                    if cash > buy_cap:
                        cash = buy_cap
                    if cash < 0:
                        cash = -min(-cash, sell_cap)
                    if cash == 0.0:
                        sim_amm_params[aid][dim]['R'], sim_amm_params[aid][dim]['T'], sim_S[aid][dim] = R0, T0, S0
                    elif cash > 0:
                        x = cash
                        q = x * T0 / (R0 + x)
                        sim_amm_params[aid][dim]['R'] = R0 + x
                        sim_amm_params[aid][dim]['T'] = max(T0 - q, 1e-9)
                        sim_S[aid][dim] = S0 + q
                        sim_uninvested_capacity[dim] -= x
                    else:
                        y = -cash
                        q = y * T0 / (R0 - y) if (R0 - y) > 1e-9 else 0.0
                        sim_amm_params[aid][dim]['R'] = max(R0 - y, 1e-9)
                        sim_amm_params[aid][dim]['T'] = T0 + q
                        sim_S[aid][dim] = max(0.0, S0 - q)
                        sim_uninvested_capacity[dim] += y
            return sim_amm_params, sim_S, sim_uninvested_capacity

        sim_amm_params = copy.deepcopy(self.market.agent_amm_params)
        sim_S = copy.deepcopy(self.market.source_investments[self.source_id])
        # Snapshot original state for optional solving against original R,T,S
        snapshot_amm_params = copy.deepcopy(self.market.agent_amm_params)
        snapshot_S = copy.deepcopy(self.market.source_investments[self.source_id])
        sim_uninvested_capacity = copy.deepcopy(self.market.source_available_capacity[self.source_id])
        for _iter in range(max(1, inner_iters)):

            if recompute_targets:
                sim_market_prices = defaultdict(lambda: defaultdict(float))
                for aid in own_evaluations.keys():
                    for dim in self.expertise_dimensions:
                        sim_market_prices[aid][dim] = sim_amm_params[aid][dim]['R'] / max(sim_amm_params[aid][dim]['T'], 1e-9)
                _proj_prices, proj_caps, _ = self.check_market_capacity(own_evaluations, sim_market_prices)

            # Recompute risk against simulated capital
            if recompute_risk:
                sim_cap_holdings = defaultdict(lambda: defaultdict(float))
                for aid in own_evaluations.keys():
                    for dim in self.expertise_dimensions:
                        sim_cap_holdings[aid][dim] = sim_amm_params[aid][dim]['R'] / max(sim_amm_params[aid][dim]['T'], 1e-9)
                risk_iter = self.compute_risk(proj_caps, current_capital_holdings=sim_cap_holdings, type='relative_capital')

            # Attractiveness from capital
            attractiveness_scores = defaultdict(lambda: defaultdict(float))
            for dim in self.expertise_dimensions:
                for aid in own_evaluations.keys():
                    curr_cap = sim_amm_params[aid][dim]['R'] / max(sim_amm_params[aid][dim]['T'], 1e-9)
                    cap_tgt = proj_caps[dim][aid][0] if isinstance(proj_caps[dim][aid], tuple) else proj_caps[dim][aid]
                    if desirability_method == 'percentage_change':
                        attr = ((cap_tgt - curr_cap) / curr_cap) if curr_cap > 1e-9 else 0.0
                    elif desirability_method == 'log_ratio':
                        if curr_cap > 1e-9 and cap_tgt > 1e-9:
                            attr = np.log(cap_tgt / curr_cap)
                        elif cap_tgt > curr_cap:
                            attr = 1.0
                        else:
                            attr = -1.0
                    else:
                        attr = ((cap_tgt - curr_cap) / curr_cap) if curr_cap > 1e-9 else 0.0
                    rsk = 0.0
                    if isinstance(risk_iter.get(aid, None), dict):
                        rsk = risk_iter[aid].get(dim, 0.0)
                    elif isinstance(risk_iter.get(dim, None), dict):
                        rsk = risk_iter[dim].get(aid, 0.0)
                    attractiveness_scores[dim][aid] = attr / (rsk + 1.0)

            # Weights and ideal values
            target_portfolio_weights = defaultdict(lambda: defaultdict(float))
            total_portfolio_value_potential = self._calculate_total_portfolio_value_potential(
                total_available_cash=sim_uninvested_capacity, 
                amm_params=sim_amm_params, 
                source_investments=sim_S)
            for dim in self.expertise_dimensions:
                pos = {aid: v for aid, v in attractiveness_scores[dim].items() if v > buy_threshold}
                s = sum(pos.values())
                if s > 1e-9:
                    for aid, v in pos.items():
                        target_portfolio_weights[dim][aid] = v / s

            target_value_holding_ideal = defaultdict(lambda: defaultdict(float))
            for dim in self.expertise_dimensions:
                for aid in own_evaluations.keys():
                    if aid in target_portfolio_weights.get(dim, {}):
                        target_value_holding_ideal[dim][aid] = target_portfolio_weights[dim][aid] * total_portfolio_value_potential.get(dim, 0.0)
                    else:
                        target_value_holding_ideal[dim][aid] = self.config.get('min_value_holding_per_asset', 0.0)

            value_holding_init = defaultdict(lambda: defaultdict(float))
            for dimension in self.expertise_dimensions:
                for agent_id in own_evaluations.keys():
                    S = sim_S[agent_id][dimension]
                    R, T = sim_amm_params[agent_id][dimension]['R'], sim_amm_params[agent_id][dimension]['T']
                    p_curr = R / max(T, 1e-9)
                    value_holding_init[dimension][agent_id] = S * p_curr
            # Per-dimension solve with relaxation and budget scaling
            median_attr = {dim : np.median([abs(v) for v in attractiveness_scores[dim].values()]) if attractiveness_scores[dim] else 0.0 for dim in self.expertise_dimensions}
            new_trades_by_dim = {dim: dict(trades_by_dim[dim]) for dim in self.expertise_dimensions}
            trades_by_dim_nonsmoothed = {dim: dict(trades_by_dim[dim]) for dim in self.expertise_dimensions}
            for dim in self.expertise_dimensions:
                # Build asset parameter list for joint solve
                assets = []
                forced_sells = {}
                portfolio_val_dim = total_portfolio_value_potential.get(dim, 0.0)

                for aid in own_evaluations.keys():
                    if self.config.get('optimizer_solve_on_sim_state', False):
                        R = sim_amm_params[aid][dim]['R']
                        T = sim_amm_params[aid][dim]['T']
                        S = sim_S.get(aid, {}).get(dim, 0.0)
                    else:
                        R = snapshot_amm_params[aid][dim]['R']
                        T = snapshot_amm_params[aid][dim]['T']
                        S = snapshot_S.get(aid, {}).get(dim, 0.0)
                    V_prev = current_value_holding[dim][aid]
                    V_target = target_value_holding_ideal[dim][aid]
                    attr = attractiveness_scores[dim].get(aid, 0.0)
                    weight = max(0.05, abs(attr))
                    # Risk dict can be agent->dim or dim->agent; guard both
                    rsk = 0.0
                    if isinstance(risk_iter.get(aid, None), dict):
                        rsk = risk_iter[aid].get(dim, 0.0)
                    elif isinstance(risk_iter.get(dim, None), dict):
                        rsk = risk_iter[dim].get(aid, 0.0)
                    buy_allowed = attr > buy_threshold
                    sell_allowed = True 

                    # Optional zeroing rule: if target is ~0 and current holding is small, force full sell
                    if zeroing and (V_target <= zero_target_rel * portfolio_val_dim) and (V_prev <= small_hold_rel * portfolio_val_dim) and (sell_allowed or not buy_allowed):
                        sell_cap, _ = self._compute_trade_bounds(R, T, S)
                        if sell_cap > 0:
                            forced_sells[aid] = -sell_cap
                            # Skip adding this asset to the optimizer assets list
                            continue

                    assets.append({
                        'key': aid,
                        'R': R,
                        'T': T,
                        'S': S,
                        'V_target': V_target,
                        'V_prev': V_prev,
                        'V_init': value_holding_init[dim].get(aid, 0.0),
                        'risk': rsk,
                        'weight': weight,
                        'buy_allowed': buy_allowed,
                        'sell_allowed': sell_allowed,
                    })

                available_cash_base = self.market.source_available_capacity.get(self.source_id, {}).get(dim, 0.0)
                # Compute buys cap (applies to total buys including reallocations)
                cap_cash = None
                if self.config.get('optimizer_respect_investment_scale_cap', True):
                    invest_scale = self.config.get('investment_scale', 0.2)
                    cap_cash = max(0.0, invest_scale * portfolio_val_dim)
                # Add proceeds from forced sells to available budget for remaining assets
                forced_proceeds = sum(-c for c in forced_sells.values() if c < 0)
                effective_available_cash = available_cash_base + forced_proceeds
                if self.config.get('optimizer_joint_budget', True):
                    joint_trades = self._solve_budget_coupled_trades(
                        assets=assets,
                        available_cash=effective_available_cash,
                        lam_prox=lam,
                        rho_risk=rho,
                        tau_turnover=tau,
                        grid_points=grid_pts,
                        buys_cap=cap_cash,
                    )
                    cash_trades = {**forced_sells, **joint_trades}
                else:
                    # Fallback to independent solves + uniform scaling
                    cash_trades = {}
                    for a in assets:
                        c_star = self._solve_asset_trade_L2(
                            a['R'], a['T'], a['S'],
                            V_target=a['V_target'],
                            V_prev=a['V_prev'],
                            V_init=a['V_init'],
                            risk=a['risk'],
                            weight=a['weight'],
                            lam_prox=lam*median_attr[dim],
                            rho_risk=rho*median_attr[dim],
                            tau_turnover=tau*median_attr[dim],
                            buy_allowed=a['buy_allowed'],
                            sell_allowed=a['sell_allowed'],
                            grid_points=grid_pts,
                        )
                        cash_trades[a['key']] = c_star
                    cash_trades = {**forced_sells, **cash_trades}
                    # First scale to fit available cash + sells
                    cash_trades = self._scale_trades_to_budget(cash_trades, available_cash_base)
                    # Then enforce total-buys cap (includes reallocations) if configured
                    if cap_cash is not None:
                        buys = sum(max(c, 0.0) for c in cash_trades.values())
                        if buys > cap_cash + 1e-9:
                            scale = cap_cash / buys if buys > 1e-9 else 0.0
                            for k, v in list(cash_trades.items()):
                                if v > 0:
                                    cash_trades[k] = v * scale

                trades_by_dim_nonsmoothed[dim] = cash_trades
                for aid, c in cash_trades.items():
                    # Do not relax forced zeroing sells; apply immediately
                    if aid in forced_sells:
                        new_trades_by_dim[dim][aid] = c
                    else:
                        new_trades_by_dim[dim][aid] = (1 - relax) * trades_by_dim[dim][aid] + relax * c
            trades_by_dim = new_trades_by_dim
            sim_amm_params, sim_S, sim_uninvested_capacity = simulate_state(trades_by_dim)
            # print(f"DEBUG(OPT): Iter {_iter+1}/{inner_iters} completed.")
            # print("DEBUG(OPT): trades by dim:")
            # for dim in self.expertise_dimensions:
            #     sample_trades = [(aid,cash) for i, (aid, cash) in enumerate(trades_by_dim[dim].items())]
            #     sample_trades.sort(key=lambda x: x[1])
            #     print(f"  Dim {dim}: {sample_trades} ...")

        # 5. Optimization per-dimension
        total_portfolio_value_potential = self._calculate_total_portfolio_value_potential()
        min_trade_threshold_pct = self.config.get('min_delta_value_trade_threshold', 1)

        # Apply min trade threshold; allow exceptions for zeroing small positions
        for dimension in self.expertise_dimensions:
            cash_trades = trades_by_dim[dimension]
            threshold_value = total_portfolio_value_potential.get(dimension, 0.0) * (min_trade_threshold_pct / 100.0)
            # print(f"DEBUG(OPT): Dim {dimension} - Applying min trade threshold: {threshold_value:.4f} ({min_trade_threshold_pct}%)")
            portfolio_val_dim = total_portfolio_value_potential.get(dimension, 0.0)
            for agent_id, cash in cash_trades.items():
                V_prev = 0.0
                V_target = 0.0
                if dimension in current_value_holding and agent_id in current_value_holding[dimension]:
                    V_prev = current_value_holding[dimension][agent_id]
                if dimension in target_value_holding_ideal and agent_id in target_value_holding_ideal[dimension]:
                    V_target = target_value_holding_ideal[dimension][agent_id]
                is_zeroing_trade = zeroing and (cash < 0) and (V_target <= zero_target_rel * portfolio_val_dim) and (V_prev <= small_hold_rel * portfolio_val_dim)
                if abs(cash) < threshold_value and not is_zeroing_trade:
                    continue
                conf = own_evaluations.get(agent_id, {}).get(dimension, (0.5, 0.5))[1]
                investments_to_propose.append((agent_id, dimension, cash, conf))

        # Sort by cash magnitude for cleanliness
        investments_to_propose.sort(key=lambda x: x[2])

        print(f"DEBUG: Final investments list length: {len(investments_to_propose)}")
        if investments_to_propose:
            print(f"DEBUG: {self.source_type.capitalize()} {self.source_id} prepared {len(investments_to_propose)} cash-value based actions.")
            for i, (aid, dim, amount, conf) in enumerate(investments_to_propose):
                print(f"DEBUG: Investment {i+1}: Agent {aid}, Dim {dim}, Amount {amount:.4f}, Confidence {conf:.4f}")
        else:
            print(f"DEBUG: {self.source_type.capitalize()} {self.source_id} found no cash-value actions to take.")
        
        # --- CLEANUP ---
        # Reset detailed analysis flag after evaluation
        self._detailed_analysis_active = False
        # ipdb.set_trace()

        if analysis_mode or detailed_analysis:
            if detailed_analysis:
                analysis_data = analysis_data or {}
                analysis_data['comparison_log'] = comparison_log
            analysis_data = analysis_data or {}
            analysis_data['own_evaluations'] = own_evaluations
            return investments_to_propose, analysis_data
        return investments_to_propose

    def evaluate_and_get_pair_evaluation_memory(self, evaluation_round=None, use_comparative=True):
        """
        Returns the pair evaluation memory for the auditor.
        """

        own_evaluations = {}
        market_prices = {}
        
        candidate_agent_ids = list(self.agent_conversations.keys()) 
        if not candidate_agent_ids:
            return {}

        # for agent_id in candidate_agent_ids:
        #     market_prices[agent_id] = {}
        #     for dim_to_eval in self.expertise_dimensions:
        #         self.market.ensure_agent_dimension_initialized_in_amm(agent_id, dim_to_eval)
        #         amm_p = self.market.agent_amm_params[agent_id][dim_to_eval]
        #         price = amm_p['R'] / amm_p['T'] if amm_p['T'] > 1e-6 else \
        #                 self.market.agent_trust_scores[agent_id].get(dim_to_eval, 0.5)
        #         market_prices[agent_id][dim_to_eval] = price
        
        own_evaluations = self.evaluate_agents_batch(
            candidate_agent_ids,
            dimensions=self.expertise_dimensions,
            evaluation_round=evaluation_round,
            use_comparative=use_comparative
        )

        if not own_evaluations:
            return {}

        # # This is the key call
        # _projected_prices, projected_capital_shares, _capacity_flags = self.check_market_capacity(
        #     own_evaluations, 
        #     market_prices, 
        #     regulatory_capacity=self.config.get('regulatory_capacity', 0.0),
        #     include_source_capacity=True
        # )

        return self.pair_evaluation_memory
    
    def evaluate_agents_batch(self, agent_ids, dimensions=None, evaluation_round=None, use_comparative=True, analysis_mode=False, detailed_analysis=False):
        """Parallel batch evaluation wrapper for UserRep."""
        return super().evaluate_agents_batch(
            agent_ids=agent_ids,
            dimensions=dimensions,
            evaluation_round=evaluation_round,
            use_comparative=use_comparative,
            analysis_mode=analysis_mode,
            detailed_analysis=detailed_analysis
        )
