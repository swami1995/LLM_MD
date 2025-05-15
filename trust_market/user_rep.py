import math
import numpy as np
from collections import defaultdict
from trust_market.info_sources import InformationSource 
from typing import List, Dict, Any
# Import evaluator from auditor (assuming it's in the same directory/package)
from trust_market.auditor import BatchEvaluator # Use BatchEvaluator for comparisons
# Assuming google.genai for LLM calls
from google import genai
from google.genai import types
import ipdb

# --- Base UserRepresentative (Simplified Logic) ---
class UserRepresentative(InformationSource):
    """
    Represents aggregated feedback for a user segment.
    (Simplified base class logic).
    """
    def __init__(self, source_id, user_segment, representative_profile, market=None):
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
             # print(f"UserRep {self.source_id} received direct feedback for agent {agent_id} from user {user_id}")
             for dim, rating in ratings.items():
                  if dim in self.expertise_dimensions: # Only store relevant dimensions
                       self.direct_feedback[agent_id][dim].append(rating)

    def add_comparison_feedback(self, comparison_data: Dict):
        """Stores comparison results if the user is represented."""
        if comparison_data.get('user_id') in self.represented_users:
             # print(f"UserRep {self.source_id} received comparison feedback from user {comparison_data.get('user_id')}")
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


    def decide_investments(self, evaluation_round=None):
        """
        Decides investments based on aggregated user feedback analysis.
        (Simple strategy for base UserRepresentative).
        """
        investments = []
        if not self.market:
            print(f"Warning (UserRep {self.source_id}): No market access.")
            return []

        # Get available capacity
        available_influence = {
            dim: self.market.source_influence_capacity.get(self.source_id, {}).get(dim, 0.0) -
                 self.market.allocated_influence.get(self.source_id, {}).get(dim, 0.0)
            for dim in self.expertise_dimensions
        }

        agents_with_feedback = list(self.direct_feedback.keys())
        print(f"UserRep {self.source_id} evaluating {len(agents_with_feedback)} agents based on direct feedback for round {evaluation_round}.")

        for agent_id in agents_with_feedback:
             eval_results = self.evaluate_agent(agent_id, self.expertise_dimensions)
             if not eval_results: continue

             investment_factor = 0.15 # User reps might invest slightly more aggressively

             for dimension, (rating_0_1, confidence) in eval_results.items():
                  if rating_0_1 >= 0.55: # Invest if score is above neutral
                       available = available_influence.get(dimension, 0.0)
                       if available <= 0: continue

                       segment_weight = self.segment_weights.get(self.user_segment, {}).get(dimension, 0.5)
                       # Amount based on score deviation, confidence, segment importance
                       amount = available * investment_factor * (rating_0_1 - 0.5) * 2 * confidence * segment_weight

                       if amount > 0.01:
                           investments.append((agent_id, dimension, amount, confidence))

        # Clear feedback after processing for the round
        self.direct_feedback.clear()
        self.comparison_feedback.clear()

        print(f"UserRep {self.source_id} decided on {len(investments)} investment actions.")
        return investments


# --- Holistic User Rep using LLM Batch Evaluation ---
class UserRepresentativeWithHolisticEvaluation(UserRepresentative):
    """
    User representative that evaluates agents holistically across batches of conversations using LLM.
"""

    def __init__(self, source_id, user_segment, representative_profile, market=None, api_key=None, api_model_name='gemini-2.0-flash'): # Added api_key
        super().__init__(source_id, user_segment, representative_profile, market)

        # Initialize holistic evaluator (using BatchEvaluator from auditor.py)
        self.api_model_name = api_model_name
        self.evaluator = BatchEvaluator(api_key=api_key, api_model_name=api_model_name) # Pass API key

        # Cache for agent evaluations from holistic comparison
        self.agent_evaluation_cache = {} # {agent_id: {dimension: (score, confidence)}}

        # Tracking recent evaluation rounds and comparisons
        self.last_evaluation_round = -1
        self.compared_pairs = set()
        self.derived_agent_scores = {} # Stores {agent_id: {dimension: score_0_1}}

        # Configuration specific to this Rep's strategy
        self.config = {
            'comparison_agents_per_target': 3, # Compare against fewer agents than auditor?
            'min_conversations_required': 4,   # Min conversations needed for comparison
            'new_evaluation_weight': 0.7,      # Weight for new comparative scores vs prior derived score
            'invest_multiplier': 0.2,         # Investment aggressiveness
            'divest_multiplier': 0.15,        # Divestment aggressiveness
        }
        # Note: Uses segment_weights from base class for dimension importance


    def get_agent_conversations(self, agent_id, max_count=10):
        """Gets recent conversations for an agent."""
        return self.agent_conversations.get(agent_id, [])[-max_count:]

    def observed_agents(self):
        """Returns agent IDs with stored conversations."""
        return set(self.agent_conversations.keys())

    # Override evaluate_agent to use the holistic comparison method
    def evaluate_agent(self, agent_id, dimensions=None, evaluation_round=None):
        """
        Evaluates an agent using holistic comparison against peers.

        Returns: Dict mapping dimensions to (rating_0_1, confidence) tuples.
        TODO : Make the compared pairs into a dictionary storing the corresponding scores for each agent pair within a round for caching purposes to avoid recomputing. 
        TODO : Need to be more careful about keeping/carrying forward derived scores across evaluation rounds.
        """
        print(f"\n--- UserRep {self.source_id} evaluating Agent {agent_id} for round {evaluation_round} ---") # DEBUG START

        dimensions_to_evaluate = dimensions or self.expertise_dimensions
        # Use cache if available and for the current round
        cache_key = (agent_id, tuple(sorted(dimensions_to_evaluate)), evaluation_round)
        if evaluation_round == self.last_evaluation_round and cache_key in self.agent_evaluation_cache:
            print(f"DEBUG: Cache hit for UserRep eval agent {agent_id} round {evaluation_round}. Returning cached result.") # DEBUG CACHE
            return self.agent_evaluation_cache[cache_key]

        # Reset comparison tracking if it's a new round
        if evaluation_round and evaluation_round != self.last_evaluation_round:
            #  self.derived_agent_scores = {} # Reset derived scores each round
            print(f"DEBUG: New evaluation round {evaluation_round}. Resetting derived scores and compared pairs.") # DEBUG ROUND RESET
            # !!! IMPORTANT: Ensure this reset happens if needed. If you want scores to persist across rounds
            #     in some way, the logic needs careful review. Let's assume reset per round for now.
            self.compared_pairs = set()
            self.last_evaluation_round = evaluation_round
            print(f"UserRep {self.source_id} starting new evaluation round {evaluation_round}")

        # --- Start Evaluation Logic ---
        # Get base scores (from previous comparisons in this round or neutral)
        base_scores_0_1 = self.derived_agent_scores.get(agent_id, {})
        print(f"DEBUG: Agent {agent_id} - Base derived scores for this round: {base_scores_0_1}") # DEBUG BASE

        base_scores_with_conf = {
            dim: (score, 0.5) # Assign moderate confidence to derived scores
            for dim, score in base_scores_0_1.items()
            if dim in dimensions_to_evaluate
        }
        if not base_scores_0_1: # Initialize if no prior derived score
            base_scores_with_conf = {dim: (0.5, 0.3) for dim in dimensions_to_evaluate}

        print(f"DEBUG: Agent {agent_id} - Initial scores with confidence: {base_scores_with_conf}") # DEBUG BASE CONF

        agent_conversations = self.get_agent_conversations(agent_id)
        min_convs = self.config.get('min_conversations_required', 1)

        # if len(agent_conversations) < min_convs:
        #     print(f"DEBUG: Agent {agent_id} has too few convs ({len(agent_conversations)}). Returning base scores with low conf.") # DEBUG LOW CONV
        #     # print(f"Agent {agent_id} has too few conversations ({len(agent_conversations)} < {min_convs}) for holistic eval.")
        #     # Return base score with reduced confidence
        #     return {dim: (score, min(conf, 0.3)) for dim, (score, conf) in base_scores_with_conf.items()}

        # Find other agents with enough conversations to compare against
        other_agent_ids = self.observed_agents()
        other_agent_ids.discard(agent_id)
        valid_comparison_agents = []
        for other_id in other_agent_ids:
             other_convs = self.get_agent_conversations(other_id)
             if len(other_convs) >= min_convs:
                 valid_comparison_agents.append((other_id, other_convs))

        print(f"DEBUG: Agent {agent_id} - Found {len(valid_comparison_agents)} valid comparison agents.") # DEBUG PEERS
        if not valid_comparison_agents:
             print(f"No valid comparison agents found for agent {agent_id}.")
             return base_scores_with_conf # Return base if no one to compare to

        # --- Select Comparison Subset ---
        import random
        num_to_compare = self.config.get('comparison_agents_per_target', 3)
        comparison_agents_selected = []
        if len(valid_comparison_agents) > num_to_compare:
            # Prioritize new comparisons
            new_comps = [(oid, c) for oid, c in valid_comparison_agents if (agent_id, oid) not in self.compared_pairs]
            if True: #len(new_comps) >= num_to_compare:
                comparison_agents_selected = random.sample(new_comps, num_to_compare)
            # else:
            #     old_comps = [(oid, c) for oid, c in valid_comparison_agents if (agent_id, oid) in self.compared_pairs]
            #     needed = num_to_compare - len(new_comps)
            #     comparison_agents_selected = new_comps + random.sample(old_comps, min(needed, len(old_comps)))
        else:
            comparison_agents_selected = valid_comparison_agents
        selected_ids = [oid for oid, _ in comparison_agents_selected]
        print(f"DEBUG: Agent {agent_id} - Selected comparison agents: {selected_ids}") # DEBUG SELECTED PEERS

        # --- Perform Comparisons ---
        accumulated_scores = defaultdict(list)
        comparison_count = 0
        # print(f"UserRep {self.source_id} comparing agent {agent_id} against {[oid for oid, _ in comparison_agents_selected]}")

        for other_id, other_convs in comparison_agents_selected:
            print(f"DEBUG: --> Comparing {agent_id} vs {other_id}") # DEBUG COMPARISON PAIR
            # if (agent_id, other_id) in self.compared_pairs: continue # Skip if already compared this round

            # self.compared_pairs.add((agent_id, other_id))
            # self.compared_pairs.add((other_id, agent_id)) # Mark symmetric pair

            # Use the BatchEvaluator for conversation comparison
            comparison_results = self.evaluator.compare_agent_batches(
                agent_conversations, agent_id,
                other_convs, other_id,
                dimensions_to_evaluate # Pass only relevant dimensions
            )
            print(f"DEBUG: Raw comparison result ({agent_id} vs {other_id}): {comparison_results}") # DEBUG LLM RAW

            # Get pseudo-absolute scores from comparison
            derived_scores = self.evaluator.get_agent_scores(comparison_results, agent_id, other_id)

            # Accumulate scores for the target agent_id
            for dim in dimensions_to_evaluate:
                if dim in derived_scores.get(agent_id, {}):
                    accumulated_scores[dim].append(derived_scores[agent_id][dim])
                else:
                    print(f"DEBUG WARNING: Agent {agent_id} not found in derived scores from comparison with {other_id}")


            comparison_count += 1

        print(f"DEBUG: Agent {agent_id} - Accumulated scores from {comparison_count} comparisons: {dict(accumulated_scores)}") # DEBUG ACCUMULATED
        # --- Calculate Final Scores ---
        final_eval_scores = {}
        weight_new = self.config.get('new_evaluation_weight', 0.7)
        for dim in dimensions_to_evaluate:
            base_score, base_conf = base_scores_with_conf.get(dim, (0.5, 0.3))
            if dim in accumulated_scores and accumulated_scores[dim]:
                new_avg = sum(accumulated_scores[dim]) / len(accumulated_scores[dim])
                final_score = (weight_new * new_avg) + ((1 - weight_new) * base_score)
                # Confidence reflects number of comparisons and base confidence
                confidence = min(0.9, 0.4 + (comparison_count * 0.05) + base_conf * 0.1)
                final_eval_scores[dim] = (final_score, confidence)
                print(f"DEBUG: Agent {agent_id} Dim {dim}: new_avg={new_avg:.3f}, base={base_score:.3f} -> final={final_score:.3f}, conf={confidence:.3f}") # DEBUG FINAL CALC
            else: # No new comparison data for this dimension
                final_eval_scores[dim] = (base_score, base_conf * 0.9) # Use base, reduce confidence slightly
                print(f"DEBUG: Agent {agent_id} Dim {dim}: No new data. Using base={base_score:.3f}, final_conf={base_conf * 0.9:.3f}") # DEBUG FINAL BASE

        # Update derived scores cache for potential use by other agents' evals this round
        if agent_id not in self.derived_agent_scores: self.derived_agent_scores[agent_id] = {}
        for dim, (score, _) in final_eval_scores.items():
            self.derived_agent_scores[agent_id][dim] = score

        print(f"DEBUG: Updated derived_agent_scores for round {evaluation_round}: {self.derived_agent_scores}") # DEBUG UPDATE DERIVED
        self.agent_evaluation_cache[cache_key] = final_eval_scores # Cache the final result
        print(f"--- UserRep {self.source_id} finished evaluating Agent {agent_id}. Final Scores: {final_eval_scores} ---") # DEBUG END
        return final_eval_scores


    # Override decide_investments to use the holistic evaluation
    def decide_investments(self, evaluation_round=None):
        """Makes investment decisions based on holistic comparative evaluation.
        TODO: Median investment computation might not handle zero investments correctly. Not sure if it's a problem.
        """
        print(f"UserRep {self.source_id} deciding investments for round {evaluation_round}.")
        investments = []
        if not self.market: return []

        # --- Fetch Market State ---
        # try:
        # ipdb.set_trace()
        available_influence = {}
        allocated_influence = self.market.allocated_influence.get(self.source_id, {})
        capacity = self.market.source_influence_capacity.get(self.source_id, {})
        for dim in self.expertise_dimensions:
            available_influence[dim] = capacity.get(dim, 0.0) - allocated_influence.get(dim, 0.0)

        current_endorsements = self.market.get_source_endorsements(self.source_id)
        current_investments = defaultdict(lambda: defaultdict(lambda: 0.0))
        for endo in current_endorsements:
            if not current_investments[endo['agent_id']]['dimension']:
                current_investments[endo['agent_id']][endo['dimension']] = 0.0
            current_investments[endo['agent_id']][endo['dimension']] += endo['influence_amount']

        market_scores = { # Fetch scores for agents we have conversations for
            agent_id: self.market.get_agent_trust(agent_id)
            for agent_id in self.agent_conversations.keys() # Use agents we observed
            if self.market.get_agent_trust(agent_id) # Check agent exists in market
        }
    
        # Calculate total and median current investments per dimension
        total_current_investment = {dim: 0.0 for dim in self.expertise_dimensions}
        median_current_investment = {}

        for dim in self.expertise_dimensions:
            # Extract all non-zero investment amounts for this dimension
            amounts = [current_investments[agent_id].get(dim, 0.0) for agent_id in current_investments]
            
            # Calculate total
            total_current_investment[dim] = sum(amounts)
            
            # Calculate median of non-zero values
            non_zero_amounts = [a for a in amounts if a > 0]
            median_current_investment[dim] = np.median(non_zero_amounts) if non_zero_amounts else 0.0
        # ipdb.set_trace()
        # except Exception as e:
        #     print(f"Error fetching market state for UserRep {self.source_id}: {e}")
        #     return []

        # --- Evaluate Agents using Holistic Method ---
        own_evaluations = {}
        agents_to_evaluate = list(self.agent_conversations.keys())
        if not agents_to_evaluate:
             print(f"UserRep {self.source_id}: No conversations observed, cannot make investments.")
             return []

        print(f"UserRep {self.source_id} evaluating {len(agents_to_evaluate)} agents holistically...")
        for agent_id in agents_to_evaluate:
            if agent_id not in market_scores: continue # Skip if agent not in market
            if len(self.agent_conversations[agent_id]) < 2: continue # Skip if no conversations
            eval_result = self.evaluate_agent(agent_id, self.expertise_dimensions, evaluation_round)
            if eval_result:
                own_evaluations[agent_id] = eval_result
            # ipdb.set_trace()
        # ipdb.set_trace()
        if len(own_evaluations) <= 1:
            print(f"UserRep {self.source_id}: No agents successfully evaluated.")
            return []

        # --- Calculate Investment Opportunities (Disagreement with Market) ---
        investment_opportunities = defaultdict(list)
        own_relative_positions = {}
        market_relative_positions = {}

        # Calculate relative positions only if enough agents evaluated
        if len(own_evaluations) >= 2:
            for dimension in self.expertise_dimensions:
                evals_for_dim = {aid: res[dimension][0] for aid, res in own_evaluations.items() if dimension in res}
                market_for_dim = {aid: scores.get(dimension, 0.5) for aid, scores in market_scores.items() if aid in evals_for_dim}

                if len(evals_for_dim) >= 2:
                    own_rel = self._get_relative_positions(evals_for_dim) # Removed dimension arg
                    market_rel = self._get_relative_positions(market_for_dim) # Removed dimension arg
                    for agent_id in evals_for_dim:
                        if agent_id not in own_relative_positions: own_relative_positions[agent_id] = {}
                        if agent_id not in market_relative_positions: market_relative_positions[agent_id] = {}
                        own_relative_positions[agent_id][dimension] = own_rel.get(agent_id, 0.5)
                        market_relative_positions[agent_id][dimension] = market_rel.get(agent_id, 0.5)

        # Identify opportunities
        for agent_id, eval_data in own_evaluations.items():
            market_agent_scores = market_scores.get(agent_id, {})
            for dimension, (rating, confidence) in eval_data.items():
                if dimension not in self.expertise_dimensions: continue # Only invest in expertise areas

                own_position = own_relative_positions[agent_id][dimension]
                market_position = market_relative_positions[agent_id][dimension]
                
                # Calculate disagreement based on relative positions
                relative_disagreement = own_position - market_position
                opportunity_direction = 1 if relative_disagreement > 0 else -1
                disagreement_magnitude = abs(relative_disagreement)

                segment_weight = self.segment_weights.get(self.user_segment, {}).get(dimension, 0.5)
                opportunity_strength = disagreement_magnitude * confidence * segment_weight

                min_disagreement_threshold = 0.05
                if disagreement_magnitude >= min_disagreement_threshold:
                    investment_opportunities[dimension].append({
                        'agent_id': agent_id, 'dimension': dimension,
                        'rating': rating, 'confidence': confidence,
                        'disagreement': relative_disagreement,
                        'direction': opportunity_direction,
                        'own_position': own_position,
                        'market_position': market_position,
                        'strength': opportunity_strength,
                        'dimension_importance': segment_weight,
                        'current_investment': current_investments.get(agent_id, {}).get(dimension, 0.0)
                    })
                # ipdb.set_trace()
                # print things to debug
                print(f"UserRep {self.source_id} found opportunity for agent {agent_id} in dimension {dimension}")
                print(investment_opportunities[dimension][-1]) # Print last opportunity added


        # --- Prepare Investment Actions ---
        self._calculate_investment_strategy(investment_opportunities, total_current_investment)
        prepared_actions = self._prepare_investment_actions(investment_opportunities, available_influence, median_current_investment)

        print(f"UserRep {self.source_id} prepared {len(prepared_actions)} actions based on holistic eval.")
        # Clear conversation buffer for next round? Or keep a rolling window?
        # self.agent_conversations.clear() # Option: Clear all convos after evaluation
        return prepared_actions

    # --- Helper methods _get_relative_positions, _calculate_investment_strategy, _prepare_investment_actions ---
    # These can be reused directly from the AuditorWithProfileAnalysis implementation
    # Make sure they reference self.config, self.invest_multiplier, self.divest_multiplier correctly.

    def _get_relative_positions(self, evaluations):
        """
        Calculates relative position (0-1) based on ranking OR Min-Max scaling.
        Assumes valid numeric inputs in evaluations.
        """
        # --- Control Flag ---
        use_min_max_scaling = True # <<< MODIFY MANUALLY TO SWITCH
        # --------------------

        if not evaluations: return {}
        num_agents = len(evaluations)
        if num_agents == 1: return {agent_id: 0.5 for agent_id in evaluations}

        agent_ids = list(evaluations.keys())

        if use_min_max_scaling:
            # --- Min-Max Scaling Logic (No Error Checks) ---
            scores = [evaluations[aid] for aid in agent_ids] # Assumes scores are numeric
            min_score, max_score = min(scores), max(scores)
            score_range = max_score - min_score

            if score_range == 0: # All scores are identical
                return {aid: 0.5 for aid in agent_ids}
            else:
                # Normalize scores to 0-1
                return {
                    aid: (evaluations[aid] - min_score) / score_range
                    for aid in agent_ids
                }
        else:
            # --- Original Ranking Logic (No Error Checks) ---
            # Sort agent IDs by score, highest first (assumes comparable scores)
            sorted_agent_ids = sorted(agent_ids, key=lambda aid: evaluations[aid], reverse=True)

            denominator = num_agents - 1
            # Calculate rank-based position (0 to 1, higher score closer to 1)
            return {
                agent_id: (denominator - i) / denominator
                for i, agent_id in enumerate(sorted_agent_ids)
            }
    
    def _calculate_investment_strategy(self, investment_opportunities, total_current_investments):
        """Calculates normalized investment signals."""
        # (Same implementation as in Auditor)
        for dimension, opportunities in investment_opportunities.items():
            if not opportunities: continue
            positive_opps = [opp for opp in opportunities if opp['direction'] > 0]
            total_pos_strength = sum(opp['strength'] for opp in positive_opps) or 1.0
            # Need total current investment *in this dimension* across all agents we invested in
            total_current_dim = total_current_investments.get(dimension, 0.0) or 1.0

            for opp in opportunities:
                target_allocation = 0.0
                if opp['direction'] > 0:
                    target_allocation = opp['strength'] / total_pos_strength if total_pos_strength > 0 else 0.0
                else:
                    target_allocation = -opp['strength'] / total_pos_strength if total_pos_strength > 0 else 1.0/len(opportunities) # Equal share if no strength

                current_allocation_norm = opp['current_investment'] / total_current_dim if total_current_dim > 0 else 0.0
                opp['invest_divest_normalized'] = target_allocation - current_allocation_norm


    def _prepare_investment_actions(self, investment_opportunities, available_influence, median_current_investment):
        """Prepares investment/divestment actions."""
        # (Same implementation as in Auditor)
        divestments = []
        investments = []
        total_available_after_divest = dict(available_influence)

        # Divestments
        for dimension, opportunities in investment_opportunities.items():
            if dimension not in self.expertise_dimensions: continue # Only act on expertise dims
            for opp in opportunities:
                if opp['invest_divest_normalized'] < 0:
                    agent_id = opp['agent_id']
                    current = opp['current_investment']
                    if current > 0:
                        amount_to_divest = min(current, abs(opp['invest_divest_normalized']) * median_current_investment * self.config.get('divest_multiplier', 0.15) * 5)
                        if amount_to_divest > 0.01:
                            divestments.append((agent_id, dimension, -amount_to_divest, None))
                            total_available_after_divest[dimension] = total_available_after_divest.get(dimension, 0.0) + amount_to_divest

        # Investments
        for dimension, opportunities in investment_opportunities.items():
            if dimension not in self.expertise_dimensions: continue
            invest_opps = [opp for opp in opportunities if opp['invest_divest_normalized'] > 0]
            if not invest_opps: continue

            available_for_dim = total_available_after_divest.get(dimension, 0.0)
            if available_for_dim <= 0: continue

            total_pos_signal = sum(opp['invest_divest_normalized'] for opp in invest_opps)
            if total_pos_signal <= 0: continue

            for opp in invest_opps:
                proportion = opp['invest_divest_normalized'] / total_pos_signal
                amount_to_invest = available_for_dim * proportion * self.config.get('invest_multiplier', 0.2)
                if amount_to_invest > 0.01:
                    investments.append((opp['agent_id'], dimension, amount_to_invest, opp['confidence']))

        return divestments + investments