import time
import random
import json
import re
import numpy as np
from collections import defaultdict
from trust_market.info_sources import InformationSource
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from google import genai
from google.genai import types
from trust_market.auditor import BatchEvaluator


class Regulator(InformationSource):
    """Regulator evaluates using profile analysis and conversation history."""

    def __init__(self, source_id, market=None, api_key=None, api_model_name='gemini-2.5-flash', verbose=False, api_provider='gemini', openai_api_key=None):
        expertise_dimensions = [
            "Factual_Correctness", "Process_Reliability", "Value_Alignment",
            "Communication_Quality", "Problem_Resolution", "Safety_Security",
            "Transparency", "Adaptability", "Trust_Calibration",
            "Manipulation_Resistance"
        ]

        # Moderate confidence across all dimensions
        confidence = {dim: 0.7 for dim in expertise_dimensions}

        super().__init__(source_id, "regulator", expertise_dimensions,
                         confidence, market)

        self.agent_profiles = {}
        self.agent_conversations = defaultdict(list)
        self.direct_feedback = defaultdict(lambda: defaultdict(list))
        self.verbose = verbose
        
        # Flag to track if detailed analysis is currently active
        self._detailed_analysis_active = False

        # Initialize LLM-based evaluator
        self.batch_evaluator = BatchEvaluator(
            api_key=api_key,
            api_model_name=api_model_name,
            api_provider=api_provider,
            openai_api_key=openai_api_key
        )

        # Configuration for hybrid approach and investment
        self.config = {
            'profile_weight': 0.4, # Weight for profile-based score
            'conversation_weight': 0.6, # Weight for conversation-based score
            'min_conversations_required': 3, # Min conversations for conv. audit
            'new_evaluation_weight': 0.7, # Weight for new comparative scores vs derived
            'comparison_agents_per_target': 4, # How many agents to compare against
            'dimension_importance': { # Example importance weights
                "Factual_Correctness": 1.0, "Process_Reliability": 0.9, "Value_Alignment": 0.8,
                "Communication_Quality": 0.7, "Problem_Resolution": 0.8, "Safety_Security": 1.0,
                "Transparency": 0.9, "Adaptability": 0.6, "Trust_Calibration": 0.7,
                "Manipulation_Resistance": 0.9
            },
            'investment_threshold': 0.65, # Min score to consider investment
            'divestment_threshold': 0.45, # Max score to consider divestment
            'invest_multiplier': 0.2, # Aggressiveness of investment
            'divest_multiplier': 0.15, # Aggressiveness of divestment
            'base_score_persistence': 0.0, # Factor for persisting base scores during updates
            'derived_score_update_weight': 0.3, # Weight for new scores when updating derived scores from a single comparison
            
            # Parameters for decide_investments that were using .get()
            'desirability_method': 'percentage_change',
            'min_operational_price': 0.01,
            'max_operational_price': 0.99,
            'attractiveness_buy_threshold': 0.01,
            'min_value_holding_per_asset': 0.0,
            'portfolio_rebalance_aggressiveness': 0.5,
            'min_delta_value_trade_threshold': 0.1,
            'investment_scale': 0.2,
            'rank_correction_strength': 0.5,
            'max_confidence_history': 10,
            'max_realloc_per_dim': 100,
            'regulator_influence_ratio': 0.5, # Added for the new decide_investments method
            'max_eval_trials': 1,
            'var_threshold': 0.1,
            'min_eval_trials': 1,
            
            # Bayesian Inference Parameters
            'confidence_to_kappa_scale_factor': 50.0, # M parameter for converting confidence to precision
            'decay_rate': 0.0, # How quickly old evidence is forgotten
            'likelihood_strength_factor': 4.0, # Higher value = regulator evaluations have stronger influence
        }
        self.num_trials = self.config.get('max_eval_trials', 1)
        self.min_trials = self.config.get('min_eval_trials', 1)

    def add_agent_profile(self, agent_id: int, profile: Dict):
        """Stores agent profile data."""
        self.agent_profiles[agent_id] = profile
        self.agent_conversations[agent_id] = []
        self.direct_feedback[agent_id] = defaultdict(list)
        self._invalidate_cache(agent_id) # Invalidate cache if profile changes

    def _invalidate_cache(self, agent_id=None):
        """Invalidates cached evaluations."""
        super()._invalidate_cache(agent_id)

    def add_conversation(self, conversation_history: List[Dict], user_id: Any, agent_id: int):
        """Stores conversation history if the user is represented."""
        self.agent_conversations[agent_id].append(conversation_history)

    def add_direct_feedback(self, user_id: Any, agent_id: int, ratings: Dict[str, int]):
        """Stores direct user ratings if the user is represented."""
        for dim, rating in ratings.items():
            if dim in self.expertise_dimensions: # Only store relevant dimensions
                self.direct_feedback[agent_id][dim].append(rating)

    def get_agent_conversations(self, agent_id, max_count=10):
        """Gets recent conversations for an agent."""
        return self.agent_conversations.get(agent_id, [])[-max_count:]

    def observed_agents(self):
        """Returns IDs of agents with profiles."""
        return set(self.agent_profiles.keys())

    def perform_comparative_audit(self, agent_id, dimensions=None, evaluation_round=None):
        """Performs comparative audit using BatchEvaluator, with improved caching and confidence handling."""
        dimensions = dimensions or self.expertise_dimensions
        # Cache key for the final evaluation result of this specific agent_id, dimensions, and round
        final_eval_cache_key = ("comp", agent_id, tuple(sorted(dimensions)), evaluation_round)
        if evaluation_round == self.last_evaluation_round and final_eval_cache_key in self.comparison_evaluation_cache:
            return self.comparison_evaluation_cache[final_eval_cache_key]

        # Reset derived scores/comparisons on new round
        if evaluation_round is not None and evaluation_round != self.last_evaluation_round:
            # This state is cleared more broadly in evaluate_agent if round changes.
            # However, if perform_comparative_audit is called directly with a new round,
            # we ensure per-round state is fresh for the logic within this function.
            self.compared_pairs.clear() # Tracks (agent_id, other_id) for direct comparisons made for *this* agent_id's eval
            self.last_evaluation_round = evaluation_round # Local track for this function's scope if needed

        # Base scores for the target agent_id from potentially prior comparisons in the same round
        base_scores_0_1 = self.derived_agent_scores.get(agent_id, {})
        base_confidences = self._calculate_derived_confidence(agent_id, dimensions) # Uses self.derived_agent_confidences

        base_scores_with_conf = {
            dim: (base_scores_0_1.get(dim, 0.5), base_confidences.get(dim, 0.3))
            for dim in dimensions
        }
        
        agent_conversations = self.get_agent_conversations(agent_id)
        agent_profile = self.agent_profiles.get(agent_id)

        min_convs = self.config.get('min_conversations_required', 3)
        # Condition for returning base scores if agent data is insufficient
        can_compare_profile = agent_profile is not None
        can_compare_convs = len(agent_conversations) >= min_convs

        if not can_compare_profile and not can_compare_convs:
            return {dim: (score, min(conf, 0.3)) for dim, (score, conf) in base_scores_with_conf.items()}

        other_agent_ids = self.observed_agents()
        other_agent_ids.discard(agent_id)

        valid_comparison_agents = []
        for other_id in other_agent_ids:
            other_profile = self.agent_profiles.get(other_id)
            other_convs = self.get_agent_conversations(other_id)
            if (other_profile is not None) or (len(other_convs) >= min_convs):
                valid_comparison_agents.append((other_id, other_convs, other_profile))

        if not valid_comparison_agents:
            return base_scores_with_conf

        import random
        num_to_compare = self.config.get('comparison_agents_per_target', 4)
        num_to_compare = min(num_to_compare, len(valid_comparison_agents))
        
        # Prioritize agents not yet compared with agent_id in this round via comparison_results_cache
        new_comparison_candidates = [
            (oid, o_convs, o_profile) for oid, o_convs, o_profile in valid_comparison_agents
            if (min(agent_id, oid), max(agent_id, oid), evaluation_round) not in self.comparison_results_cache
        ]
        
        comparison_agents_selected = []
        if len(new_comparison_candidates) >= num_to_compare:
            comparison_agents_selected = random.sample(new_comparison_candidates, num_to_compare)
        else:
            comparison_agents_selected.extend(new_comparison_candidates)
            remaining_needed = num_to_compare - len(comparison_agents_selected)
            if remaining_needed > 0:
                existing_candidates = [
                    (oid, o_convs, o_profile) for oid, o_convs, o_profile in valid_comparison_agents
                    if (oid, o_convs, o_profile) not in new_comparison_candidates # Ensure no duplicates
                ]
                if existing_candidates:
                    comparison_agents_selected.extend(
                        random.sample(existing_candidates, min(remaining_needed, len(existing_candidates)))
                    )
        if not comparison_agents_selected and valid_comparison_agents: # Fallback if selection logic yields empty
             comparison_agents_selected = random.sample(valid_comparison_agents, min(num_to_compare, len(valid_comparison_agents)))

        accumulated_scores_for_target = defaultdict(list)
        accumulated_confs_for_target = defaultdict(list)
        
        for other_id, other_convs, other_profile in comparison_agents_selected:
            comparison_cache_key = (min(agent_id, other_id), max(agent_id, other_id), evaluation_round)
            
            derived_scores_from_evaluator = None
            comparison_confidences = None

            if comparison_cache_key in self.comparison_results_cache:
                derived_scores_from_evaluator, comparison_confidences = self.comparison_results_cache[comparison_cache_key]
            else:
                comparison_call_results = None
                # Determine comparison type
                # Profile + Conversations vs Profile + Conversations
                if agent_profile and can_compare_convs and (len(other_convs) >= min_convs):
                    comparison_call_results = self.batch_evaluator.compare_agent_profiles_and_convs(
                        agent_profile, agent_conversations, agent_id, other_profile, other_convs, other_id, dimensions
                    )
                # Profile vs Profile
                elif agent_profile and other_profile:
                    comparison_call_results = self.batch_evaluator.compare_agent_profiles(
                        agent_profile, agent_id, other_profile, other_id, dimensions
                    )
                # Conversations vs Conversations
                elif can_compare_convs and (len(other_convs) >= min_convs):
                    comparison_call_results = self.batch_evaluator.compare_agent_batches(
                        agent_conversations, agent_id, other_convs, other_id, dimensions
                    )

                else:
                    continue # Skip if no valid comparison method

                if comparison_call_results:
                    derived_scores_from_evaluator = self.batch_evaluator.get_agent_scores(comparison_call_results, agent_id, other_id)
                    comparison_confidences = self._extract_comparison_confidences(comparison_call_results, agent_id, other_id)
                    
                    self.comparison_results_cache[comparison_cache_key] = (derived_scores_from_evaluator, comparison_confidences)

                    # Update derived scores and confidences for BOTH agents
                    self._update_agent_derived_scores(agent_id, derived_scores_from_evaluator.get(agent_id, {}), dimensions, comparison_confidences.get(agent_id, {}))
                    self._update_agent_confidences(agent_id, comparison_confidences.get(agent_id, {}), dimensions)
                    
                    self._update_agent_derived_scores(other_id, derived_scores_from_evaluator.get(other_id, {}), dimensions, comparison_confidences.get(other_id, {}))
                    self._update_agent_confidences(other_id, comparison_confidences.get(other_id, {}), dimensions)
                    
                    self.agent_comparison_counts[agent_id] += 1
                    self.agent_comparison_counts[other_id] += 1 # Count for both

            if derived_scores_from_evaluator and comparison_confidences:
                # Accumulate for the target agent_id
                agent_scores_from_this_comp = derived_scores_from_evaluator.get(agent_id, {})
                agent_confs_from_this_comp = comparison_confidences.get(agent_id, {})
                for dim in dimensions:
                    if dim in agent_scores_from_this_comp:
                        accumulated_scores_for_target[dim].append(agent_scores_from_this_comp[dim])
                        accumulated_confs_for_target[dim].append(agent_confs_from_this_comp.get(dim, 0.3))
        
        final_scores = {}
        
        for dim in dimensions:
            base_score, base_conf = base_scores_with_conf.get(dim, (0.5, 0.3))
            new_scores = accumulated_scores_for_target.get(dim, [])
            new_confs = accumulated_confs_for_target.get(dim, [])
            
            if new_scores:
                avg_new_score = 0.0
                avg_new_confidence = 0.3 # Default if no confs
                if sum(new_confs) > 1e-6:
                    avg_new_score = sum(s*c for s,c in zip(new_scores, new_confs)) / sum(new_confs)
                    avg_new_confidence = sum(new_confs) / len(new_confs)
                elif new_scores: # Fallback if all confidences are zero
                    avg_new_score = sum(new_scores) / len(new_scores)

                # Determine effective weight for new information based on relative confidences
                total_confidence_metric = base_conf + avg_new_confidence
                effective_weight_new = avg_new_confidence / total_confidence_metric if total_confidence_metric > 1e-6 else 0.5
                
                persistence_factor = self.config.get('base_score_persistence', 0.2)
                # Adjust effective_weight_new to account for persistence
                # If persistence is 0.2, new info can take up to 0.8 of the update influence, scaled by its relative confidence
                final_weight_new = effective_weight_new * (1 - persistence_factor)

                current_score_for_update = self.derived_agent_scores.get(agent_id, {}).get(dim, 0.5) # Get latest derived score

                final_score = (final_weight_new * avg_new_score) + ((1 - final_weight_new) * current_score_for_update)
                final_confidence = self._aggregate_confidences(new_confs, base_conf, final_weight_new) # Aggregate all new confs vs base
            else:
                final_score, final_confidence = base_score, base_conf * 0.9 # Slightly decay confidence if no new info

            final_scores[dim] = (final_score, final_confidence)
            # Update the main derived_agent_scores cache for the target agent
            if agent_id not in self.derived_agent_scores: self.derived_agent_scores[agent_id] = {}
            self.derived_agent_scores[agent_id][dim] = final_score

        self.comparison_evaluation_cache[final_eval_cache_key] = final_scores
        return final_scores

    # Override evaluate_agent to use the hybrid method
    def evaluate_agent(self, agent_id, conversations=None,
                       dimensions=None, evaluation_round=None,
                       use_comparative=False):
        """Evaluate agent using hybrid or comparative approach."""
        if evaluation_round is not None and evaluation_round != self.last_evaluation_round:
            # Clear per-round state. This is critical.
            self.compared_pairs.clear() # Specific to old logic, but good to clear
            self.comparison_results_cache.clear() # Cache for (min_id, max_id, round) results
            self.agent_comparison_counts.clear()
            
            # LLM-audit caches are for (type, agent_id, dims, round/len_convs)
            # Clearing them ensures fresh LLM calls if underlying data might change or for a truly new round.
            self.profile_evaluation_cache.clear()
            self.conversation_audit_cache.clear()
            self.comparison_evaluation_cache.clear() # Cache for final (agent_id, dims, round) results
            self.hybrid_evaluation_cache.clear()
            
            self.last_evaluation_round = evaluation_round
        
        cache_key = ("hybrid", agent_id, tuple(sorted(dimensions or self.expertise_dimensions)), evaluation_round, use_comparative)
        if cache_key in self.hybrid_evaluation_cache:
             return self.hybrid_evaluation_cache[cache_key]

        dimensions = dimensions or self.expertise_dimensions

        if use_comparative:
            results = self.perform_comparative_audit(agent_id, dimensions, evaluation_round)
            self.hybrid_evaluation_cache[cache_key] = results
            return results
        else:
            raise NotImplementedError("Hybrid audit not implemented for use_comparative=False")

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
        print(total_potential)
        # Ensure a minimum potential to avoid issues if source starts with no cash/shares
        # return max(total_potential, self.config.get('min_portfolio_value_potential', 100.0))
        return total_potential

    def _project_steady_state_prices(self, own_evaluations, market_prices, regulatory_capacity, dimension, include_source_capacity=False):
        """
        Project what market prices will be at steady state based on:
        1. Expected total capital deployment
        2. Quality-based distribution of that capital
        """
        
        # Step 1: Estimate total capital at steady state
        # Consider all potential investors and their capacity
        total_potential_capital = regulatory_capacity
        if include_source_capacity:
            for source_id, _ in self.market.source_available_capacity.items():
                # Each source's total capacity across all dimensions
                source_capacity = self.market.source_available_capacity[source_id].get(dimension, 0)
                total_potential_capital += source_capacity
        
        # Add expected growth factor (new investors, increased allocations)
        # growth_factor = self.config.get('market_growth_factor', 1.5)
        steady_state_capital = total_potential_capital #* growth_factor
        
        # Step 2: Current total capital in market
        current_total_market_capital = 0
        for agent_id in market_prices:
            if dimension in self.market.agent_amm_params[agent_id]:
                # Total capital locked in AMM = R (reserves)
                current_total_market_capital += self.market.agent_amm_params[agent_id][dimension]['R']
        
        steady_state_capital += current_total_market_capital # Include current capital in the market
        # Step 3: Project capital distribution based on quality scores, factoring in confidence.
        # The expected capital for an agent is interpolated between the market's current
        # allocation and this source's evaluation-based allocation, using confidence as the weight.

        # 3a. Calculate capital shares based purely on this source's evaluations.
        quality_scores_and_confs = {
            agent_id: eval_data[dimension]
            for agent_id, eval_data in own_evaluations.items()
            if dimension in eval_data
        }
        
        # Convert quality to expected capital share, powered by a concentration factor.
        concentration_power = self.config.get('quality_concentration_power', 2.0)
        
        quality_powered = {
            aid: data[0] ** concentration_power  # data[0] is the score
            for aid, data in quality_scores_and_confs.items()
        }
        total_quality_powered = sum(quality_powered.values())
        
        evaluation_based_capital_shares = {}
        if total_quality_powered > 1e-9:
            evaluation_based_capital_shares = {
                aid: qp / total_quality_powered
                for aid, qp in quality_powered.items()
            }

        # 3b. Calculate current capital shares from the market.
        current_agent_capital = {
            agent_id: self.market.agent_amm_params[agent_id][dimension]['R']
            for agent_id in market_prices
            if dimension in self.market.agent_amm_params.get(agent_id, {})
        }
        current_total_market_capital_for_shares = sum(current_agent_capital.values())

        current_capital_shares = {}
        if current_total_market_capital_for_shares > 1e-9:
            current_capital_shares = {
                aid: cap / current_total_market_capital_for_shares
                for aid, cap in current_agent_capital.items()
            }
        
        # 3c. Interpolate between evaluation-based and market-based shares using confidence.
        interpolated_shares = {}
        all_agent_ids = set(quality_scores_and_confs.keys()) | set(current_capital_shares.keys())

        for aid in all_agent_ids:
            # Confidence acts as the interpolation factor. High confidence -> lean towards own eval.
            confidence = quality_scores_and_confs.get(aid, (0.5, 0.0))[1]
            
            eval_share = evaluation_based_capital_shares.get(aid, 0.0)
            market_share = current_capital_shares.get(aid, 0.0)
            weight = confidence ** 2
            interpolated_shares[aid] = (weight * eval_share) + ((1 - weight) * market_share)

        # 3d. Normalize the interpolated shares to ensure they sum to 1.
        total_interpolated_share = sum(interpolated_shares.values())
        expected_capital_shares = {}
        if total_interpolated_share > 1e-9:
            expected_capital_shares = {
                aid: share / total_interpolated_share
                for aid, share in interpolated_shares.items()
            }
        else: # Fallback if all shares are zero, e.g., market is empty and no evals.
            num_agents = len(all_agent_ids)
            if num_agents > 0:
                expected_capital_shares = {aid: 1.0 / num_agents for aid in all_agent_ids}
        
        # Step 4: Project steady-state prices
        projected_prices = {}
        projected_capital_shares = {}
        
        for agent_id in expected_capital_shares:
            # Expected capital for this agent at steady state
            projected_capital_shares[agent_id] = steady_state_capital * expected_capital_shares[agent_id]
            
            # Project price based on AMM dynamics
            # At steady state, if R_ss is the reserve, need to estimate T_ss
            # Assume T remains relatively stable (or decreases slowly as investors buy)
            current_T = self.market.agent_amm_params[agent_id][dimension]['T']
            current_R = self.market.agent_amm_params[agent_id][dimension]['R']
            
            # Estimate T at steady state (some shares bought from treasury)
            # treasury_depletion_rate = self.config.get('treasury_depletion_rate', 0.3)
            # projected_T = current_T * (1 - treasury_depletion_rate)                             # TODO : Need a more sophisticated mechanism to compute projected_T and corresponding projected prices.
            if projected_capital_shares[agent_id] == 0:
                projected_T = current_T
            else:
                projected_T = current_T * current_R / projected_capital_shares[agent_id]
            
            # Projected price = R_ss / T_ss
            projected_price = projected_capital_shares[agent_id] / projected_T if projected_T > 0 else 0
            projected_prices[agent_id] = projected_price

        return projected_prices, projected_capital_shares, steady_state_capital / current_total_market_capital

    def check_market_capacity(self, own_evaluations, market_prices, regulatory_capacity=0.0, include_source_capacity=False):
        """
        Checks if the source has enough capacity to invest based on its evaluations and market prices.
        If not, it will print a warning and return False.
        """
        capacity_flags = {} # Collect ratios for all dimensions
        projected_prices = {} # {agent_id: projected_price}
        projected_capital_shares = {} # {agent_id: projected_capital_share}
        for dim in self.expertise_dimensions:
            projected_prices_dim, projected_capital_shares_dim, steady_state_ratio = self._project_steady_state_prices(own_evaluations, market_prices, regulatory_capacity, dimension=dim, include_source_capacity=include_source_capacity)
            capacity_flags[dim] = True # regulator always has capacity
            projected_prices[dim] = projected_prices_dim # Store projected prices for this dimension
            projected_capital_shares[dim] = projected_capital_shares_dim # Store projected capital shares for this dimension

        return projected_prices, projected_capital_shares, capacity_flags

    def _extract_comparison_confidences(self, comparison_results, agent_a_id, agent_b_id):
        """
        Extract confidence information from comparison results.
        Maps the comparison confidence (LLM 0-5) to derived pseudo-score confidence (0-1).
        """
        return super()._extract_comparison_confidences(comparison_results, agent_a_id, agent_b_id)

    def _update_agent_derived_scores(self, agent_id, new_scores_for_agent, dimensions_to_evaluate, new_confidences_for_agent):
        """
        Helper method to update derived scores for an agent based on new comparison data.
        Uses confidence-weighted averaging between existing and new scores.
        `new_scores_for_agent`: {dim: score} from the current comparison for this agent.
        `new_confidences_for_agent`: {dim: conf} from the current comparison for this agent.
        """
        super()._update_agent_derived_scores(agent_id, new_scores_for_agent, dimensions_to_evaluate, new_confidences_for_agent)

    def _update_agent_confidences(self, agent_id, new_confidences_for_agent, dimensions_to_evaluate):
        """
        Appends new confidence scores from a comparison to the agent's list of confidences for each dimension.
        `new_confidences_for_agent`: {dim: conf} from the current comparison for this agent.
        """
        super()._update_agent_confidences(agent_id, new_confidences_for_agent, dimensions_to_evaluate)

    def _calculate_derived_confidence(self, agent_id, dimensions_to_evaluate):
        """
        Calculate aggregated confidence in derived scores for an agent.
        Uses the list of confidences stored in `self.derived_agent_confidences`.
        """
        return super()._calculate_derived_confidence(agent_id, dimensions_to_evaluate)

    def _aggregate_confidences(self, new_confidences_list, base_aggregated_confidence, weight_for_new_info_block):
        """
        Aggregates a list of new confidences with a base aggregated confidence.
        `new_confidences_list`: A list of confidence values from recent evaluations.
        `base_aggregated_confidence`: The existing aggregated confidence in the base score.
        `weight_for_new_info_block`: How much weight to give the block of new information.
        """
        return super()._aggregate_confidences(new_confidences_list, base_aggregated_confidence, weight_for_new_info_block)

    def decide_investments(self, evaluation_round=None, use_comparative=True, analysis_mode=False, detailed_analysis=False):
        desirability_method = self.config.get('desirability_method', 'percentage_change')
        if self.verbose:
            print(f"REGULATOR ({self.source_id}): Starting investment decisions for round {evaluation_round}.")

        # Store detailed_analysis flag for use in _compare_pair
        self._detailed_analysis_active = detailed_analysis

        # --- 1. Evaluate Agents ---
        # Get evaluations for all agents the regulator is aware of.
        all_agent_ids = list(self.agent_profiles.keys())
        analysis_data = defaultdict(lambda : defaultdict(dict)) # {(agent_id, dimension): (pseudo_score, confidence_in_eval)}
        investments_to_propose_cash_value = [] # {(agent_id, dimension): cash_value_to_trade}

        # In analysis mode, we want to see the raw evaluation output.
        # Otherwise, we might use cached evaluations.
        evaluation_result = self.evaluate_agents_batch(
            agent_ids=all_agent_ids,
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
            print("REGULATOR: No evaluations were generated. Cannot decide investments.")
            return [], {}

        # --- 2. Get Current Market State ---
        market_prices = self.market.get_market_prices(candidate_agent_ids=all_agent_ids, dimensions=self.expertise_dimensions, verbose=self.verbose)
        if not market_prices:
            if self.verbose: print("REGULATOR: No market prices available. Cannot determine desirability.")
            return [], {}

        projected_prices, projected_capital_shares, capacity_flags = self.check_market_capacity(own_evaluations, market_prices, regulatory_capacity=self.config.get('regulatory_capacity', 0.0))

        # --- Get accumulated user influence since last run ---
        # This is the core of the reactive balancing mechanism
        user_influence_by_dim = self.market.get_and_reset_cumulative_user_influence()
        desired_influence_ratio = self.config.get('regulator_influence_ratio', 0.5) # How much to match user influence

        # --- 2. Determine "Target Value Holding" & "Attractiveness" ---
        rebalance_aggressiveness = {}
        delta_value_target_map = defaultdict(lambda : defaultdict(float))
        
        # The maximum reallocation is now determined by actual user activity per dimension.
        # Fallback to config value if no user activity was recorded for a dimension.
        default_max_realloc = self.config.get('max_realloc_per_dim', 100)

        for dim in self.expertise_dimensions:
            # Set this dimension's reallocation budget based on user influence
            max_realloc_per_dim = user_influence_by_dim.get(dim, 0.0) * desired_influence_ratio
            if max_realloc_per_dim <= 1e-6: # If no user influence, use fallback
                max_realloc_per_dim = default_max_realloc
                if self.verbose:
                    print(f"DEBUG: No user influence for dim {dim}, using default max_realloc: {max_realloc_per_dim}")

            agents_with_proj_capital = [
                aid for aid in own_evaluations.keys() 
                if dim in projected_capital_shares and aid in projected_capital_shares[dim] and
                   aid in self.market.agent_amm_params and dim in self.market.agent_amm_params[aid]
            ]
            
            if not agents_with_proj_capital:
                if self.verbose:
                    print(f"DEBUG: No agents with projected capital shares for dim {dim}, skipping rebalance calculation.")
                continue

            deltas = np.array([projected_capital_shares[dim][agent_id] - self.market.agent_amm_params[agent_id][dim]['R'] for agent_id in agents_with_proj_capital])
            # TODO : fix this assert and division by 0 : rebalancing with capital shares for now. but need to decide between capital shares and prices or other better metrics.
            if not np.isclose(deltas.sum(), 0.0, atol=1e-1):
                print(f"Warning: Delta sum for dim {dim} is {deltas.sum()}, which is not close to zero.")
            assert np.abs(deltas.sum()) < 1e-1, f"Delta sum for dim {dim} is {deltas.sum()}"
            
            pos_deltas_sum = deltas[deltas > 0].sum()
            if pos_deltas_sum > 1e-9:
                rebalance_aggressiveness[dim] = max_realloc_per_dim / pos_deltas_sum
            else:
                rebalance_aggressiveness[dim] = 1.0
            rebalance_aggressiveness[dim] = min(rebalance_aggressiveness[dim], 1.0)
            if self.verbose:
                print(f"DEBUG: Rebalance aggressiveness for dim {dim}: {rebalance_aggressiveness[dim]:.4f}, max_realloc_per_dim={max_realloc_per_dim:.2f}, deltas={deltas[deltas > 0].sum():.2f}")
            
            for i, agent_id in enumerate(agents_with_proj_capital):
                delta_v = deltas[i] * rebalance_aggressiveness[dim]
                # Apply a threshold to delta_v to avoid tiny trades
                min_trade_threshold = self.config.get('min_delta_value_trade_threshold', 0.1)
                if abs(delta_v) > min_trade_threshold: # e.g., trade if value change > $0.1
                    delta_value_target_map[dim][agent_id] = delta_v
                    if self.verbose:
                        print(f"DEBUG: Delta above threshold ({min_trade_threshold}) - Including in trade map: {delta_v:.4f}")
                else:
                    if self.verbose:
                        print(f"DEBUG: Delta below threshold ({min_trade_threshold}) - Skipping: {delta_v:.4f}")
        

        # attractiveness_scores = defaultdict(lambda : defaultdict(float)) # {(agent_id, dimension): attractiveness_score}
        # target_value_holding_ideal = defaultdict(lambda : defaultdict(float)) # {(agent_id, dimension): ideal_cash_value_to_hold}

        # # Filter evaluations to only those agents for whom we also have market prices
        # # This ensures fair comparison for rank mapping and attractiveness calculation
        # valid_agent_ids_for_ranking = [aid for aid in own_evaluations.keys() if aid in market_prices and \
        #                                all(dim in market_prices[aid] for dim in self.expertise_dimensions)]
        
        # if self.verbose:
        #     print(f"DEBUG: Valid agents for ranking: {len(valid_agent_ids_for_ranking)} out of {len(own_evaluations)}")
        #     print(f"DEBUG: Valid agent IDs: {valid_agent_ids_for_ranking}")
        
        # relevant_own_evals_for_ranking = {aid: own_evaluations[aid] for aid in valid_agent_ids_for_ranking}
        # relevant_market_prices_for_ranking = {aid: market_prices[aid] for aid in valid_agent_ids_for_ranking}
        # for agent_id, agent_eval_data in own_evaluations.items(): # Iterate all evaluated agents
        #     if self.verbose:
        #         print(f"DEBUG: Processing attractiveness for agent {agent_id}")
        #     for dimension, (pseudo_score, confidence_in_eval) in agent_eval_data.items():
        #         if dimension not in self.expertise_dimensions: 
        #             if self.verbose:
        #                 print(f"DEBUG: Skipping dimension {dimension} (not in expertise)")
        #             continue # Should not happen if eval_agent is correct

        #         key = (agent_id, dimension)
        #         p_current = market_prices.get(agent_id, {}).get(dimension, 0.5) # Use fetched market price
        #         if self.verbose:
        #             print(f"DEBUG: Agent {agent_id}, Dim {dimension}: pseudo_score={pseudo_score:.4f}, confidence={confidence_in_eval:.4f}, p_current={p_current:.4f}")

        #         if capacity_flags.get(dimension, False):
        #             p_target_effective_est = projected_prices.get(dimension, {}).get(agent_id, p_current) # Use projected price if capacity is sufficient
        #             if self.verbose:
        #                 print(f"DEBUG: Agent {agent_id}, Dim {dimension}: p_target_raw_from_projected_prices={p_target_effective_est:.4f}")
        #         else:
        #             p_target_effective_est = p_current # Default if agent not in ranking pool
        #             if agent_id in relevant_own_evals_for_ranking: # Only calculate rank target if agent is in the valid pool
        #                 p_target_effective_est = self._get_target_price_from_rank_mapping(
        #                     agent_id, dimension, 
        #                     relevant_own_evals_for_ranking, # Use filtered evals for ranking
        #                     relevant_market_prices_for_ranking, # Use filtered prices for ranking
        #                     confidence_in_eval
        #                 )
        #                 if self.verbose:
        #                     print(f"DEBUG: Agent {agent_id}, Dim {dimension}: p_target_raw_from_rank={p_target_effective_est:.4f}")


        #         p_target_effective = p_target_effective_est
        #         # p_target_effective = p_current + (p_target_effective_est - p_current) * confidence_in_eval
        #         min_op_p = self.config.get('min_operational_price', 0.01)
        #         # max_op_p = self.config.get('max_operational_price', 0.99)
        #         # p_target_effective = max(min_op_p, min(max_op_p, p_target_effective))
        #         p_target_effective = max(min_op_p, p_target_effective)
        #         if self.verbose:
        #             print(f"DEBUG: Agent {agent_id}, Dim {dimension}: p_target_effective={p_target_effective:.4f} (clamped between {min_op_p})")#-{max_op_p})")

        #         ideal_val = p_target_effective
        #         current_val = p_current
                
        #         delta_v = (ideal_val - current_val) * rebalance_aggressiveness[dimension]
        #         if self.verbose:
        #             print(f"DEBUG: Delta calculation - Dim {dim}, Agent {agent_id}: ideal={ideal_val:.4f}, current={current_val:.4f}, delta_raw={(ideal_val - current_val):.4f}, delta_scaled={delta_v:.4f}")
                
        #         # Apply a threshold to delta_v to avoid tiny trades
        #         min_trade_threshold = self.config.get('min_delta_value_trade_threshold', 0.1)
        #         if abs(delta_v) > min_trade_threshold: # e.g., trade if value change > $0.1
        #             delta_value_target_map[dim][agent_id] = delta_v
        #             if self.verbose:
        #                 print(f"DEBUG: Delta above threshold ({min_trade_threshold}) - Including in trade map: {delta_v:.4f}")
        #         else:
        #             if self.verbose:
        #                 print(f"DEBUG: Delta below threshold ({min_trade_threshold}) - Skipping: {delta_v:.4f}")


        if self.verbose:
            print(f"DEBUG: Delta value target map: {dict(delta_value_target_map)}")

        if analysis_mode:
            for dim in self.expertise_dimensions:
                for agent_id, cash_amount in delta_value_target_map[dim].items():
                    analysis_data[agent_id][dim] = {
                        'cash_amount': cash_amount,
                        'confidence': own_evaluations[agent_id][dim][1],
                        'projected_price': projected_prices[dim][agent_id],
                        'projected_capital_share': projected_capital_shares[dim][agent_id],
                        'market_price': market_prices[agent_id][dim],
                        'own_evaluation': own_evaluations[agent_id][dim][0],
                    }

        # --- Prepare list of (agent_id, dimension, cash_amount_to_trade, confidence) ---
        # The TrustMarket.process_investments will convert this cash_amount to shares
        # and handle actual cash availability for buys.
        if self.verbose:
            print(f"DEBUG: Preparing final investment list...")
        
        for dim in delta_value_target_map.keys():
            for agent_id, cash_amount in delta_value_target_map[dim].items():
                confidence = 0.5 # Default confidence
                if agent_id in own_evaluations and dim in own_evaluations[agent_id]:
                    confidence = own_evaluations[agent_id][dim][1]
                
                if self.verbose:
                    print(f"DEBUG: Adding investment - Agent {agent_id}, Dim {dim}: cash_amount={cash_amount:.4f}, confidence={confidence:.4f}")
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
        
        # --- RETURN ---
        if analysis_mode or detailed_analysis:
            if detailed_analysis:
                # Include comparison_log in analysis_data for detailed analysis
                analysis_data['comparison_log'] = comparison_log
            return investments_to_propose_cash_value, analysis_data
        return investments_to_propose_cash_value


    def evaluate_and_get_pair_evaluation_memory(self, evaluation_round=None, use_comparative=True):
        """
        Returns the pair evaluation memory for the auditor.
        """      
        candidate_agent_ids = list(self.agent_profiles.keys())
        if not candidate_agent_ids:
            return {}
        
        own_evaluations = self.evaluate_agents_batch(
            candidate_agent_ids,
            dimensions=self.expertise_dimensions,
            evaluation_round=evaluation_round,
            use_comparative=use_comparative
        )

        if not own_evaluations:
            return {}

        return self.pair_evaluation_memory

    def get_target_capital_distribution(self, evaluation_round=None, use_comparative=True):
        """
        Evaluates agents and returns the ideal capital distribution from the regulator's perspective.
        This is a utility function for analysis and state initialization.
        """
        # This logic is mostly copied from the start of decide_investments
        own_evaluations = {}
        market_prices = {}
        
        candidate_agent_ids = list(self.agent_profiles.keys())
        if not candidate_agent_ids:
            return {}

        for agent_id in candidate_agent_ids:
            market_prices[agent_id] = {}
            for dim_to_eval in self.expertise_dimensions:
                self.market.ensure_agent_dimension_initialized_in_amm(agent_id, dim_to_eval)
                amm_p = self.market.agent_amm_params[agent_id][dim_to_eval]
                price = amm_p['R'] / amm_p['T'] if amm_p['T'] > 1e-6 else \
                        self.market.agent_trust_scores[agent_id].get(dim_to_eval, 0.5)
                market_prices[agent_id][dim_to_eval] = price
        
        own_evaluations = self.evaluate_agents_batch(
            candidate_agent_ids,
            dimensions=self.expertise_dimensions,
            evaluation_round=evaluation_round,
            use_comparative=use_comparative
        )

        if not own_evaluations:
            return {}

        # This is the key call
        _projected_prices, projected_capital_shares, _capacity_flags = self.check_market_capacity(
            own_evaluations, 
            market_prices, 
            regulatory_capacity=self.config.get('regulatory_capacity', 0.0),
            include_source_capacity=True
        )

        return projected_capital_shares, self.pair_evaluation_memory

    def _perform_base_evaluation(self, agent_id, dimensions, evaluation_round):
        """The Regulator's base evaluation can be a simple profile check."""
        # For simplicity, we'll make this a neutral score, as comparison is key.
        return {dim: (0.5, 0.3) for dim in dimensions}

    def _compare_pair(self, agent_a_id: int, agent_b_id: int, dimensions: List[str], additional_context: str = ""):
        """
        Compares two agents based on their profiles using the batch evaluator.
        """
        agent_profile = self.agent_profiles.get(agent_a_id)
        other_profile = self.agent_profiles.get(agent_b_id)
        agent_convs = self.get_agent_conversations(agent_a_id)
        other_convs = self.get_agent_conversations(agent_b_id)
        min_convs = self.config.get('min_conversations_required', 3)

        can_compare_convs_aid = len(agent_convs) >= min_convs
        can_compare_convs_oid = len(other_convs) >= min_convs
        return_raw = self._detailed_analysis_active

        comparison_call_results = None
        # Prefer hybrid comparison if all data is available
        if agent_profile and other_profile and can_compare_convs_aid and can_compare_convs_oid:
            comparison_call_results = self.batch_evaluator.compare_agent_profiles_and_convs(
                agent_profile, agent_convs, agent_a_id, other_profile, other_convs, agent_b_id, dimensions, additional_context=additional_context
            )
        # Fallback to profile-only
        elif agent_profile and other_profile:
            comparison_call_results = self.batch_evaluator.compare_agent_profiles(
                agent_profile, agent_a_id, other_profile, agent_b_id, dimensions, additional_context=additional_context
            )
        # Fallback to conversation-only
        elif can_compare_convs_aid and can_compare_convs_oid:
            comparison_call_results = self.batch_evaluator.compare_agent_batches(
                agent_convs, agent_a_id, other_convs, agent_b_id, dimensions, additional_context=additional_context
            )
        else:
            raise ValueError(f"Cannot compare agents {agent_a_id} and {agent_b_id} with the given data.")
        
        if not comparison_call_results:
            return None

        derived_scores, confidences = self.batch_evaluator.get_agent_scores_new(comparison_call_results, agent_a_id, agent_b_id)
        # confidences = super()._extract_comparison_confidences(comparison_call_results, aid, oid)
        
        # Return 5-tuple if detailed analysis is active, 4-tuple otherwise
        return (agent_a_id, agent_b_id, derived_scores, confidences, comparison_call_results)
        # else:
        #     return (aid, oid, derived_scores, confidences)

    def evaluate_agents_batch(self, agent_ids, dimensions=None, evaluation_round=None, use_comparative=True, analysis_mode=False, detailed_analysis=False):
        """
        Regulator's batch evaluation delegates to the InformationSource base class.
        This allows it to leverage the parallel, cached comparison framework.
        """
        return super().evaluate_agents_batch(
            agent_ids=agent_ids,
            dimensions=dimensions,
            evaluation_round=evaluation_round,
            use_comparative=use_comparative,
            analysis_mode=analysis_mode,
            detailed_analysis=detailed_analysis
        )

    def _agent_has_comparable_data(self, aid):
        """Checks if an agent has enough data for the Regulator's comparison."""
        # This can be the same as the Auditor's check.
        profile_exists = aid in self.agent_profiles
        convs_exist = len(self.get_agent_conversations(aid)) >= self.config.get('min_conversations_required', 3)
        return profile_exists or convs_exist

