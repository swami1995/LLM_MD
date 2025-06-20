import time
from collections import defaultdict, Counter
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Set, Optional, Union, Any

class InformationSource:
    """Base class for all information sources in the trust market."""
    
    def __init__(self, source_id, source_type, expertise_dimensions, 
                 evaluation_confidence=None, market=None):
        """
        Initialize an information source.
        
        Parameters:
        - source_id: Unique identifier
        - source_type: Type of information source
        - expertise_dimensions: Dimensions this source can evaluate
        - evaluation_confidence: Confidence level for each dimension
        - market: Reference to the trust market
        """
        self.source_id = source_id
        self.source_type = source_type
        self.expertise_dimensions = expertise_dimensions
        
        # Default confidence = 1.0 for all dimensions
        if evaluation_confidence is None:
            self.evaluation_confidence = {dim: 1.0 for dim in expertise_dimensions}
        else:
            self.evaluation_confidence = evaluation_confidence
            
        self.market = market
        self.evaluation_history = []

        # Caches for different evaluation types
        self.profile_evaluation_cache = {}
        self.conversation_audit_cache = {}
        self.comparison_evaluation_cache = {}
        self.hybrid_evaluation_cache = {} # Cache for combined results

        self.last_evaluation_round = -1 # Track last evaluated round

        self.compared_pairs = set() # Track compared pairs within a round
        self.derived_agent_scores = {} # Store scores derived from comparisons

        # --- NEW: mirror user_rep tracking for confidences & cache ---
        self.derived_agent_confidences = defaultdict(lambda: defaultdict(list))
        self.comparison_results_cache = {}       # {(aid1,aid2,round): (derived_scores_dict, comparison_confidences_dict)}
        self.agent_comparison_counts = defaultdict(int)
        
        # To be configured by subclasses
        self.verbose = False 
        self.config = {}
        self.batch_evaluator = None
    
    def can_evaluate_dimension(self, dimension):
        """Check if this source can evaluate a given dimension."""
        return dimension in self.expertise_dimensions
    
    def evaluate_agent(self, agent_id, conversation=None, dimensions=None):
        """
        Evaluate an agent along specific dimensions.
        
        Returns a dict of dimension -> (rating, confidence) pairs.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def decide_investments(self, agent_performances=None):
        """
        Decide trust investments based on evaluations.
        
        Returns a list of (agent_id, dimension, amount, confidence) tuples.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def record_evaluation(self, agent_id, ratings):
        """Record an evaluation for later analysis."""
        self.evaluation_history.append({
            'agent_id': agent_id,
            'ratings': ratings,
            'timestamp': time.time()
        })

    def _invalidate_cache(self, agent_id=None):
        """Invalidates cached evaluations."""
        if agent_id:
            self.profile_evaluation_cache.pop(agent_id, None)
            self.conversation_audit_cache.pop(agent_id, None)
            self.hybrid_evaluation_cache.pop(agent_id, None)
            self.comparison_evaluation_cache.pop(agent_id, None)
            # Specific agent's derived scores might be affected, but full clear happens on round change.
        else: # Invalidate all
            self.profile_evaluation_cache.clear()
            self.conversation_audit_cache.clear()
            self.comparison_evaluation_cache.clear()
            self.hybrid_evaluation_cache.clear()
            self.derived_agent_scores.clear()
            self.derived_agent_confidences.clear()
            self.compared_pairs.clear()
            self.comparison_results_cache.clear()
            self.agent_comparison_counts.clear()

    def _perform_base_evaluation(self, agent_id, dimensions, evaluation_round):
        """
        Placeholder for subclasses to perform a non-comparative evaluation.
        This could be based on profile, conversation history, or other direct metrics.
        """
        return {dim: (0.5, 0.3) for dim in dimensions}
    
    def _compare_pair(self, aid1, aid2, dimensions) -> Optional[Tuple[int, int, dict, dict]]:
        """
        Placeholder for subclasses to perform a pairwise comparison between two agents.
        Should return a tuple of (aid1, aid2, derived_scores, confidences) or None if incomparable.
        """
        raise NotImplementedError("Subclasses must implement the _compare_pair method.")
    
    def observed_agents(self) -> Set[int]:
        """
        Return the set of agent IDs that this information source is aware of.
        """
        raise NotImplementedError("Subclasses must implement the observed_agents method.")
    
    def _agent_has_comparable_data(self, aid):
        """Placeholder for subclasses to implement."""
        return False

    def evaluate_agents_batch(self, agent_ids: List[int], dimensions: Optional[List[str]] = None, 
                              evaluation_round: Optional[int] = None, use_comparative: bool = True):
        """Batch variant of evaluate_agent that pairs agents globally and evaluates in parallel.

        Returns: {agent_id: {dimension: (score, confidence)}}
        """
        if dimensions is None:
            dimensions = self.expertise_dimensions

        # Handle round change bookkeeping exactly once
        if evaluation_round is not None and evaluation_round != self.last_evaluation_round:
            if self.verbose:
                print(f"INFO ({self.source_id}): New evaluation round {evaluation_round}. Clearing caches.")
            self._invalidate_cache() # Clears all caches
            self.last_evaluation_round = evaluation_round

        # ------------------------------------------------------------
        # Phase 1 – base (non-comparative) evaluation
        # ------------------------------------------------------------
        if not use_comparative:
            base_evaluations = {}
            # This could be parallelized if _perform_base_evaluation is slow
            for aid in agent_ids:
                base_evaluations[aid] = self._perform_base_evaluation(
                    aid,
                    dimensions=dimensions,
                    evaluation_round=evaluation_round
                )
            return base_evaluations
        
        # ------------------------------------------------------------
        # Phase 2 – global pair selection
        # ------------------------------------------------------------
        base_scores_0_1 = {agent_id: self.derived_agent_scores.get(agent_id, {}) for agent_id in agent_ids}
        base_confidences = {agent_id: self._calculate_derived_confidence(agent_id, dimensions) for agent_id in agent_ids}

        base_scores_with_conf = {
            agent_id: {
                dim: (base_scores_0_1[agent_id].get(dim, 0.5), base_confidences[agent_id].get(dim, 0.3))
                for dim in dimensions
            }
            for agent_id in agent_ids
        }
        
        comparison_agents_per_target = self.config.get('comparison_agents_per_target', 3)

        valid_agents = [aid for aid in agent_ids if self._agent_has_comparable_data(aid)]
        if len(valid_agents) < 2:
            return base_scores_with_conf

        # Collect desired unique pairs
        desired_pairs = set()
        import random
        rng = random.Random(42 + (evaluation_round or 0))

        for aid in valid_agents:
            others = [o for o in valid_agents if o != aid]
            rng.shuffle(others)
            needed = comparison_agents_per_target
            for oid in others:
                if needed <= 0:
                    break
                pair_key = (min(aid, oid), max(aid, oid))
                if pair_key not in desired_pairs:
                    desired_pairs.add(pair_key)
                    needed -= 1

        # Remove pairs already cached for this round
        pairs_to_eval = []
        for a, b in desired_pairs:
            cache_key = (a, b, evaluation_round)
            if cache_key in self.comparison_results_cache:
                continue
            pairs_to_eval.append((a, b))

        # ------------------------------------------------------------
        # Phase 3 – parallel LLM comparison calls for those pairs
        # ------------------------------------------------------------
        accumulated_scores_for_target = defaultdict(lambda : defaultdict(list))
        accumulated_confs_for_target = defaultdict(lambda : defaultdict(list))
        
        if pairs_to_eval:
            with ThreadPoolExecutor(max_workers=min(8, len(pairs_to_eval))) as exe:
                fut_to_pair = {exe.submit(self._compare_pair, a, b, dimensions): (a, b) for (a, b) in pairs_to_eval}
                for fut in as_completed(fut_to_pair):
                    result = fut.result()
                    if result is None:
                        continue
                    
                    aid, oid, derived_scores, confidences = result
                    cache_key = (min(aid, oid), max(aid, oid), evaluation_round)
                    self.comparison_results_cache[cache_key] = (derived_scores, confidences)

                    # Single-thread update of derived scores/confidences to avoid race conditions
                    self._update_agent_derived_scores(aid, derived_scores.get(aid, {}), dimensions, confidences.get(aid, {}))
                    self._update_agent_confidences(aid, confidences.get(aid, {}), dimensions)

                    self._update_agent_derived_scores(oid, derived_scores.get(oid, {}), dimensions, confidences.get(oid, {}))
                    self._update_agent_confidences(oid, confidences.get(oid, {}), dimensions)

                    for dim in dimensions:
                        if aid in derived_scores and dim in derived_scores[aid]:
                            accumulated_scores_for_target[aid][dim].append(derived_scores[aid][dim])
                            accumulated_confs_for_target[aid][dim].append(confidences.get(aid, {}).get(dim, 0.3))

                        if oid in derived_scores and dim in derived_scores[oid]:
                            accumulated_scores_for_target[oid][dim].append(derived_scores[oid][dim])
                            accumulated_confs_for_target[oid][dim].append(confidences.get(oid, {}).get(dim, 0.3))

        # ------------------------------------------------------------
        # Phase 4 – aggregate final scores per agent
        # ------------------------------------------------------------
        final_evals = {}
        for aid in agent_ids:
            final_evals[aid] = {}
            base_scores = base_scores_with_conf.get(aid, {})
            new_scores_for_agent = accumulated_scores_for_target.get(aid, {})
            new_confs_for_agent = accumulated_confs_for_target.get(aid, {})

            for dim in dimensions:
                base_score, base_conf = base_scores.get(dim, (0.5, 0.3))
                
                new_scores = new_scores_for_agent.get(dim, [])
                new_confs = new_confs_for_agent.get(dim, [])

                if new_scores:
                    avg_new_score = 0.0
                    avg_new_confidence = 0.3
                    avg_new_score = sum(new_scores) / len(new_scores)
                    if sum(new_confs) > 1e-6:
                        # avg_new_score = sum(s*c for s,c in zip(new_scores, new_confs)) / sum(new_confs)
                        avg_new_confidence = sum(new_confs) / len(new_confs)
                    elif new_scores: # Fallback if all confidences are zero
                        avg_new_score = sum(new_scores) / len(new_scores)

                    total_confidence_metric = base_conf + avg_new_confidence
                    effective_weight_new = avg_new_confidence / total_confidence_metric if total_confidence_metric > 1e-6 else 0.5
                    
                    persistence_factor = self.config.get('base_score_persistence', 0.2)
                    final_weight_new = effective_weight_new * (1 - persistence_factor)

                    current_score_for_update = self.derived_agent_scores.get(aid, {}).get(dim, 0.5)

                    final_score = (final_weight_new * avg_new_score) + ((1 - final_weight_new) * current_score_for_update)
                    final_confidence = self._aggregate_confidences(new_confs, base_conf, final_weight_new)

                    final_evals[aid][dim] = (final_score, final_confidence)
                else:
                    # No new comparisons, return base score with slightly decayed confidence
                    final_score, final_confidence = base_score, base_conf * 0.9
                    final_evals[aid][dim] = (final_score, final_confidence)
                
                # Persist the final score for the next round
                ### TODO :  We probably don't want to do this since we are already updating derived_agent_scores above.
                if aid not in self.derived_agent_scores: self.derived_agent_scores[aid] = {}
                self.derived_agent_scores[aid][dim] = final_evals[aid][dim][0]

        return final_evals

    def _extract_comparison_confidences(self, comparison_results, agent_a_id, agent_b_id):
        """
        Extract confidence information from comparison results.
        Maps the comparison confidence (LLM 0-5) to derived pseudo-score confidence (0-1).
        """
        agent_confidences = {
            agent_a_id: {},
            agent_b_id: {}
        }
        
        for dimension, result in comparison_results.items():
            raw_confidence_metric = result.get("confidence", 2.5) # Default to mid-range if missing
            winner = result.get("winner", "tie")
            
            normalized_llm_confidence = min(1.0, raw_confidence_metric / 5.0)
            
            derived_score_confidence = 0.0
            if winner == "Tie":
                derived_score_confidence = normalized_llm_confidence * 0.6 
            else:
                derived_score_confidence = normalized_llm_confidence * 0.9
            
            agent_confidences[agent_a_id][dimension] = derived_score_confidence
            agent_confidences[agent_b_id][dimension] = derived_score_confidence
        
        return agent_confidences

    def _update_agent_derived_scores(self, agent_id, new_scores_for_agent, dimensions_to_evaluate, new_confidences_for_agent):
        """
        Helper method to update derived scores for an agent based on new comparison data.
        Uses confidence-weighted averaging between existing and new scores.
        """
        if agent_id not in self.derived_agent_scores:
            self.derived_agent_scores[agent_id] = {}
        
        existing_aggregated_confidences = self._calculate_derived_confidence(agent_id, dimensions_to_evaluate)

        for dim in dimensions_to_evaluate:
            if dim in new_scores_for_agent:
                existing_score = self.derived_agent_scores[agent_id].get(dim, 0.5)
                new_score_from_comparison = new_scores_for_agent[dim]
                
                conf_of_new_score_from_comparison = new_confidences_for_agent.get(dim, 0.3)
                conf_of_existing_score_aggregated = existing_aggregated_confidences.get(dim, 0.3)
                
                total_conf_metric = conf_of_existing_score_aggregated + conf_of_new_score_from_comparison
                if total_conf_metric > 1e-6:
                    weight_for_new_score = conf_of_new_score_from_comparison / total_conf_metric
                else:
                    weight_for_new_score = 0.5 
                
                single_comparison_update_weight = self.config.get('derived_score_update_weight', 1.0)
                effective_weight_for_new_score = weight_for_new_score * single_comparison_update_weight

                updated_score = (1 - effective_weight_for_new_score) * existing_score + effective_weight_for_new_score * new_score_from_comparison
                self.derived_agent_scores[agent_id][dim] = updated_score

    def _update_agent_confidences(self, agent_id, new_confidences_for_agent, dimensions_to_evaluate):
        """
        Appends new confidence scores from a comparison to the agent's list of confidences for each dimension.
        """
        for dim in dimensions_to_evaluate:
            if dim in new_confidences_for_agent:
                self.derived_agent_confidences[agent_id][dim].append(new_confidences_for_agent[dim])

                max_len = self.config.get('max_confidence_history', 10)
                conf_list = self.derived_agent_confidences[agent_id][dim]
                if len(conf_list) > max_len:
                    self.derived_agent_confidences[agent_id][dim] = conf_list[-max_len:]

    def _calculate_derived_confidence(self, agent_id, dimensions_to_evaluate):
        """
        Calculate confidence in derived scores using proper statistical aggregation.
        Treats each comparison as providing a noisy estimate of the true score.
        """
        confidences = {}
        precision_scale_factor = self.config.get('precision_scale_factor', 0.6)
        
        for dim in dimensions_to_evaluate:
            conf_list = self.derived_agent_confidences.get(agent_id, {}).get(dim, [])
            
            if not conf_list:
                confidences[dim] = 0.3  # Default low confidence
            else:
                # Statistical confidence aggregation
                # Each measurement has variance inversely proportional to confidence
                # Combined variance = 1 / sum(1/individual_variances)
                
                # Convert confidences to precisions (inverse of variance)
                # conf=1 -> precision=100, conf=0.1 -> precision=1
                precisions = [precision_scale_factor * (conf / (1 - conf + 1e-6)) for conf in conf_list]
                
                # Combined precision is sum of individual precisions
                combined_precision = sum(precisions)
                
                # Convert back to confidence
                if combined_precision > 1e-6:
                    combined_confidence = combined_precision / (combined_precision + 1)
                    # Cap at reasonable maximum
                    combined_confidence = min(0.95, combined_confidence)
                else:
                    combined_confidence = 0.3
                
                # Apply sample size adjustment (more comparisons = more confidence)
                sample_size_factor = 1 + 0.1 * np.log1p(len(conf_list))
                combined_confidence = min(0.95, combined_confidence * sample_size_factor)
                
                confidences[dim] = combined_confidence
                if self.verbose:
                    print(f"DEBUG: Derived confidence for Agent {agent_id}, Dim {dim}: "
                        f"{len(conf_list)} comparisons -> {combined_confidence:.3f}")
        
        return confidences
    def _aggregate_confidences(self, new_confidences_list, base_aggregated_confidence, weight_for_new_info_block):
        """
        Aggregates a list of new confidences with a base aggregated confidence.
        """
        if not new_confidences_list:
            return base_aggregated_confidence
        
        new_precisions = [c / (1 - c + 1e-6) for c in new_confidences_list if 0 <= c < 1]
        if not new_precisions:
            new_precisions = [100.0] * len([c for c in new_confidences_list if c >=1.0]) 
            if not new_precisions: new_precisions = [0.3 / (1-0.3+1e-6)]

        aggregated_new_precision = sum(new_precisions)
        
        base_precision = base_aggregated_confidence / (1 - base_aggregated_confidence + 1e-6)
        
        combined_precision = (weight_for_new_info_block * aggregated_new_precision) + \
                             ((1 - weight_for_new_info_block) * base_precision)
        
        if combined_precision > 1e-6:
            final_aggregated_confidence = combined_precision / (combined_precision + 1)
        else:
            final_aggregated_confidence = max(base_aggregated_confidence, np.mean(new_confidences_list) if new_confidences_list else 0.3)

        return min(0.95, final_aggregated_confidence)

