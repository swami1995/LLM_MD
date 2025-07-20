import time
from collections import defaultdict, Counter
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Set, Optional, Union, Any
import re
import json
import ipdb
import copy

class InformationSource:
    """Base class for all information sources in the trust market."""
    
    def __init__(self, source_id, source_type, expertise_dimensions,
                 evaluation_confidence=None, market=None, memory_length_n: int = 3):
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

        # --- NEW: Persistent memory of this source's past pairwise evaluations ---
        # key: (min_agent_id, max_agent_id) -> List[Dict] of evaluation summaries (most recent last)
        self.pair_evaluation_memory: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        self.memory_length_n = max(1, memory_length_n)
        self.rating_scale = 5.0
        
        # --- NEW: Bayesian belief state for persistent agent evaluations ---
        # key: agent_id -> dimension -> (alpha, beta) tuple for Beta distribution
        self.belief_state = defaultdict(lambda: defaultdict(lambda: None))
        
        # To be configured by subclasses
        self.verbose = False 
        self.config = {}
        self.batch_evaluator = None

    # ------------------------------------------------------------------
    # Persistent evaluation-memory helpers
    # ------------------------------------------------------------------
    def swap_agents_hybrid_case(self, text: str) -> str:
        """
        Swaps 'Agent A'/'agent a' with 'Agent B' and vice versa.

        - The word 'agent' is treated case-insensitively (e.g., 'agent a').
        - The designators 'A' and 'B' are treated case-sensitively.

        Args:
            text: The input string.

        Returns:
            The string with agents swapped according to the rules.
        """
        placeholder = "##AGENT_SWAP_PLACEHOLDER##"
        
        # Step 1: Replace 'Agent A' and 'agent a' with a placeholder.
        # The pattern [aA] looks for either a lowercase or uppercase 'a'.
        # The rest of the pattern ('gent A') is case-sensitive.
        text_with_placeholder = re.sub(r"[aA]gent A", placeholder, text)
        
        # Step 2: Replace the strictly cased 'Agent B' with 'Agent A'.
        # This is case-sensitive and will ignore 'agent b'.
        text_swapped_b = text_with_placeholder.replace("Agent B", "Agent A")
        
        # Step 3: Replace the placeholder with 'Agent B'.
        final_text = text_swapped_b.replace(placeholder, "Agent B")
        
        return final_text
    
    def _store_pair_evaluation(self, agent_a_id: int, agent_b_id: int,
                               derived_scores: Dict[int, Dict[str, float]],
                               confidences: Dict[int, Dict[str, float]],
                               raw_results: Any = None,
                               evaluation_round: int = None):
        """Save this source's evaluation of a pair of agents for future prompt context."""
        key = (min(agent_a_id, agent_b_id), max(agent_a_id, agent_b_id))
        
        raw_confidences = {}
        raw_confidences = {dim: r['confidence'] for dim, r in raw_results.items()}

        if agent_a_id > agent_b_id:
            # replace all mentions of "Agent A" with "Agent B" and vice versa in a non case sensitive way
            reasoning = {dim: self.swap_agents_hybrid_case(r['reasoning']) for dim, r in raw_results.items()}
        else:
            reasoning = {dim: r['reasoning'] for dim, r in raw_results.items()}
        
        entry = {
            'round': evaluation_round,
            'timestamp': time.time(),
            'derived_scores': derived_scores,
            'confidences': raw_confidences,
            'reasoning': reasoning,
            'raw_results': raw_results,
        }
        self.pair_evaluation_memory[key].append(entry)
        # Keep only the most recent N
        # if len(self.pair_evaluation_memory[key]) > self.memory_length_n:
            # self.pair_evaluation_memory[key] = self.pair_evaluation_memory[key][-self.memory_length_n:]

    def _get_recent_pair_evaluations(self, agent_a_id: int, agent_b_id: int) -> List[Dict[str, Any]]:
        """Return up to the last N stored evaluations (most recent first)."""
        key = (min(agent_a_id, agent_b_id), max(agent_a_id, agent_b_id))
        # Return reversed copy so most recent first
        # return list(reversed(self.pair_evaluation_memory.get(key, [])))
        pair_eval_memory = copy.deepcopy(list(reversed(self.pair_evaluation_memory.get(key, [])[-self.memory_length_n:])))
        if len(pair_eval_memory) > 0 and agent_a_id > agent_b_id:
            for eval in pair_eval_memory:
                eval['reasoning'] = {dim: self.swap_agents_hybrid_case(eval['reasoning'][dim]) for dim in eval['reasoning'].keys()}
        return pair_eval_memory

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

    def reset_evaluation_state(self):
        """
        Resets all evaluation-related caches and derived data.
        Useful for ensuring independent analysis runs.
        """
        self._invalidate_cache()
        self.derived_agent_scores.clear()
        self.derived_agent_confidences.clear()

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
            self.compared_pairs.clear()
            self.comparison_results_cache.clear()
            self.agent_comparison_counts.clear()

            # self.derived_agent_scores.clear()
            # self.derived_agent_confidences.clear()
            
    def _perform_base_evaluation(self, agent_id, dimensions, evaluation_round):
        """
        Placeholder for subclasses to perform a non-comparative evaluation.
        This could be based on profile, conversation history, or other direct metrics.
        """
        return {dim: (0.5, 0.3) for dim in dimensions}
    
    def _get_additional_context(self, agent_a_id: int, agent_b_id: int, evaluation_round: int) -> str:
        """
        Constructs a string of additional context for an LLM prompt, based on
        this source's past evaluations of a given pair.
        This can be overridden by subclasses to add more context (e.g. from regulator).
        """
        recent_evals = self._get_recent_pair_evaluations(agent_a_id, agent_b_id)
        if not recent_evals:
            return ""

        import json

        snippets = []
        for ev in recent_evals:
            rnd = ev.get('round', 'N/A')
            # Add relative round information
            relative_round_str = ""
            if isinstance(rnd, int) and isinstance(evaluation_round, int):
                diff = evaluation_round - rnd
                if diff == 0:
                    relative_round_str = " (this round)"
                elif diff == 1:
                    relative_round_str = " (last round)"
                else:
                    relative_round_str = f" ({diff} rounds ago)"

            reasoning = ev.get('reasoning', {})
            if agent_a_id > agent_b_id:
                reasoning = {dim: self.swap_agents_hybrid_case(reasoning[dim]) for dim in reasoning.keys()}
            # Correctly get scores for agent_a_id
            # Note: reasoning is already swapped if needed by _get_recent_pair_evaluations
            scores_a = ev.get('derived_scores', {}).get(agent_a_id, {})
            confidence = ev.get('confidence', 0)
            ratings_and_reasoning = {
                dim: {
                    'rating': f"{(scores_a.get(dim, 0.0)-0.5)*self.batch_evaluator.rating_scale*2:.2f}",
                    'reasoning': reasoning.get(dim, "N/A"),
                    'confidence': confidence
                } for dim in reasoning.keys()
            }

            summary_str = json.dumps(ratings_and_reasoning)
            # if len(summary_str) > 600:
            #     summary_str = summary_str[:595] + "...}}"

            snippets.append(f"Round {rnd}{relative_round_str}: {summary_str}")

        header = "For context, here are your past evaluations for this pair (most recent first, as shown by the round number of the evaluation). \n" \
                 "Use these to inform your judgment, but be aware that agent behavior can change over time. So the older evaluations might be outdated or stale. Trust your own judgment in case you feel the agent interactions/profile you see above are in contradiction with these evaluations.\n"

        return header + "\n".join(snippets)

    def _compare_pair(self, aid1, aid2, dimensions, additional_context: str = "") -> Optional[Tuple[int, int, dict, dict]]:
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
    
    # ------------------------------------------------------------------
    # Bayesian Inference Methods
    # ------------------------------------------------------------------
    def _score_and_confidence_to_beta_params(self, score: float, confidence: float) -> Tuple[float, float]:
        """
        Convert a (score, confidence) pair to Beta distribution parameters (alpha, beta).
        
        Parameters:
        - score: The evaluation score (0-1)
        - confidence: The confidence in the evaluation (0-1)
        
        Returns:
        - (alpha, beta): Parameters for Beta distribution
        """
        # Get configuration parameters
        M = self.config.get('confidence_to_kappa_scale_factor', 50.0)
        
        # Calculate kappa (precision parameter)
        kappa = 2.0 + confidence * M
        
        # Calculate alpha and beta
        alpha = score * kappa
        beta = (1.0 - score) * kappa
        
        return alpha, beta
    
    def _beta_params_to_score_and_confidence(self, alpha: float, beta: float) -> Tuple[float, float]:
        """
        Convert Beta distribution parameters to (score, confidence) pair.
        
        Parameters:
        - alpha, beta: Beta distribution parameters
        
        Returns:
        - (score, confidence): Mean and derived confidence
        """
        # Mean of Beta distribution
        mean = alpha / (alpha + beta)
        
        # Derive confidence from precision (kappa)
        kappa = alpha + beta
        M = self.config.get('confidence_to_kappa_scale_factor', 50.0)
        confidence = max(0.0, min(1.0, (kappa - 2.0) / M))
        
        return mean, confidence
    
    def _update_belief_state_bayesian(self, agent_id: int, dimension: str, 
                                     new_score: float, new_confidence: float) -> Tuple[float, float]:
        """
        Update the belief state for an agent-dimension using Bayesian inference.
        
        Parameters:
        - agent_id: The agent ID
        - dimension: The dimension being evaluated
        - new_score: The new evaluation score (0-1)
        - new_confidence: The confidence in the new evaluation (0-1)
        
        Returns:
        - (updated_score, updated_confidence): The posterior mean and confidence
        """
        # Convert new evidence to Beta parameters
        alpha_new, beta_new = self._score_and_confidence_to_beta_params(new_score, new_confidence)
        
        # Get prior belief
        prior_belief = self.belief_state[agent_id][dimension]
        
        if prior_belief is None:
            # First evaluation - set posterior to new evidence
            alpha_posterior = alpha_new
            beta_posterior = beta_new
        else:
            # Get configuration parameters
            decay_rate = self.config.get('decay_rate', 0.5)
            likelihood_strength_factor = self.config.get('likelihood_strength_factor', 1.0)
            
            # Unpack prior parameters
            alpha_prior, beta_prior = prior_belief
            
            # Apply decay to prior
            kappa_prior = alpha_prior + beta_prior
            mu_prior = alpha_prior / kappa_prior if kappa_prior > 0 else 0.5
            kappa_decayed = kappa_prior * decay_rate
            alpha_decayed = mu_prior * kappa_decayed
            beta_decayed = (1 - mu_prior) * kappa_decayed
            
            # Combine decayed prior with scaled new evidence
            alpha_posterior = alpha_decayed + (alpha_new * likelihood_strength_factor)
            beta_posterior = beta_decayed + (beta_new * likelihood_strength_factor)
        
        # Store updated belief
        self.belief_state[agent_id][dimension] = (alpha_posterior, beta_posterior)
        
        # Convert back to score and confidence
        return self._beta_params_to_score_and_confidence(alpha_posterior, beta_posterior)
    
    def combine_multiple_beliefs(self, scores: List[float], confidences: List[float]) -> Tuple[float, float]:
        """
        Combine multiple (score, confidence) pairs into a single belief state.
        
        Parameters:
        - scores: List of scores (0-1)
        - confidences: List of confidences (0-1)
        
        Returns:
        - (combined_score, combined_confidence): Combined score and confidence
        """
        if not scores or not confidences or len(scores) != len(confidences):
            return 0.5, 0.3

        alphas = []
        betas = []
        for score, confidence in zip(scores, confidences):
            alpha, beta = self._score_and_confidence_to_beta_params(score, confidence)
            alphas.append(alpha)
            betas.append(beta)
        # Combine all alphas and betas
        alpha_combined = sum(alphas)
        beta_combined = sum(betas)
        # Convert back to score and confidence
        combined_score, combined_confidence = self._beta_params_to_score_and_confidence(alpha_combined, beta_combined)
        return combined_score, combined_confidence
        # return alpha_combined, beta_combined
    # ------------------------------------------------------------------
    # End Bayesian Inference Methods
    # ------------------------------------------------------------------
    
    def evaluate_agents_batch(self, agent_ids: List[int], dimensions: Optional[List[str]] = None, 
                              evaluation_round: Optional[int] = None, use_comparative: bool = True,
                              analysis_mode: bool = False, detailed_analysis: bool = False):
        """Batch variant of evaluate_agent that pairs agents globally and evaluates in parallel.
        
        Now uses Dynamic Bayesian Inference to update persistent belief states about agents.

        Returns: 
          - if detailed_analysis is False: {agent_id: {dimension: (score, confidence)}}
          - if detailed_analysis is True: ({agent_id: ...}, comparison_log_list)
        """
        if dimensions is None:
            dimensions = self.expertise_dimensions

        # Handle round change bookkeeping exactly once
        if evaluation_round is not None and evaluation_round != self.last_evaluation_round:
            if self.verbose:
                print(f"INFO ({self.source_id}): New evaluation round {evaluation_round}. Clearing caches.")
            self._invalidate_cache() # Clears all caches
            self.last_evaluation_round = evaluation_round

        # --- For detailed analysis mode ---
        if detailed_analysis:
            comparison_log = []

        # ------------------------------------------------------------
        # Phase 1 – base (non-comparative) evaluation
        # ------------------------------------------------------------
        if not use_comparative:
            base_evaluations = {}
            # This could be parallelized if _perform_base_evaluation is slow
            for aid in agent_ids:
                base_eval = self._perform_base_evaluation(
                    aid,
                    dimensions=dimensions,
                    evaluation_round=evaluation_round
                )
                # Update belief state with base evaluation
                updated_eval = {}
                for dim in dimensions:
                    score, confidence = base_eval.get(dim, (0.5, 0.3))
                    updated_score, updated_confidence = self._update_belief_state_bayesian(
                        aid, dim, score, confidence
                    )
                    updated_eval[dim] = (updated_score, updated_confidence)
                base_evaluations[aid] = updated_eval
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
            # Even with no comparisons, update belief states for agents with existing beliefs
            final_evals = {}
            for aid in agent_ids:
                final_evals[aid] = {}
                for dim in dimensions:
                    # Get current belief if it exists
                    belief = self.belief_state[aid][dim]
                    if belief is not None:
                        # Return current belief without update
                        score, confidence = self._beta_params_to_score_and_confidence(*belief)
                        final_evals[aid][dim] = (score, confidence)
                    else:
                        # Use base score for new agents
                        final_evals[aid][dim] = base_scores_with_conf.get(aid, {}).get(dim, (0.5, 0.3))
            return final_evals

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

        
        if self.num_trials > 1:# and not analysis_mode and not detailed_analysis:
            pairs_to_eval_og = copy.deepcopy(pairs_to_eval)
            pairs_to_eval_new = pairs_to_eval
            for i in range(1, self.num_trials):
                pairs_to_eval_new = [(b,a) for (a,b) in pairs_to_eval_new]
                pairs_to_eval = pairs_to_eval + pairs_to_eval_new

        # ------------------------------------------------------------
        # Phase 3 – parallel LLM comparison calls for those pairs
        # ------------------------------------------------------------
        # Store current round evaluations for Bayesian update
        current_round_evaluations = defaultdict(lambda: defaultdict(list))  # agent_id -> dim -> [(score, confidence), ...]
        prompts_for_pairs = {}
        
        if pairs_to_eval:
            contexts_for_pairs = {
                (a, b): self._get_additional_context(a, b, evaluation_round)
                for a, b in pairs_to_eval
            }
            with ThreadPoolExecutor(max_workers=min(8, len(pairs_to_eval))) as exe:
                fut_to_pair = {exe.submit(self._compare_pair, a, b, dimensions, contexts_for_pairs.get((a,b), "")): (a, b) for (a, b) in pairs_to_eval}
                # if True:
                for fut in as_completed(fut_to_pair):
                    result = fut.result()
                    if result is None:
                        continue
                    # for a, b in pairs_to_eval:
                    #     result = self._compare_pair(a, b, dimensions, contexts_for_pairs.get((a,b), ""))
                    
                    # Unpack result, which may or may not include raw_reasoning
                    if len(result) == 4:
                        aid, oid, derived_scores, confidences = result
                        raw_results = "Not provided by source."
                    else:
                        aid, oid, derived_scores, confidences, raw_results = result

                    cache_key = (min(aid, oid), max(aid, oid), evaluation_round)
                    self.comparison_results_cache[cache_key] = (derived_scores, confidences)
                    
                    if isinstance(raw_results, dict) and 'Communication_Quality' in raw_results:
                        prompts_for_pairs[str((aid, oid))] = raw_results['Communication_Quality'].get('prompt', "Not provided by source.")

                    # --- Persist evaluation memory ---
                    if True:  # not analysis_mode and not detailed_analysis:
                        self._store_pair_evaluation(aid, oid, derived_scores, confidences, raw_results=raw_results, evaluation_round=evaluation_round)

                    # Log for detailed analysis if enabled
                    if detailed_analysis:
                        comparison_log.append({
                            'pair': (aid, oid),
                            'derived_scores': derived_scores,
                            'confidences': confidences,
                            'raw_results': raw_results
                        })

                    # Collect current round evaluations for Bayesian update
                    for dim in dimensions:
                        if aid in derived_scores and dim in derived_scores[aid]:
                            score = derived_scores[aid][dim]
                            confidence = confidences.get(aid, {}).get(dim, 0.3)
                            current_round_evaluations[aid][dim].append((score, confidence))

                        if oid in derived_scores and dim in derived_scores[oid]:
                            score = derived_scores[oid][dim]
                            confidence = confidences.get(oid, {}).get(dim, 0.3)
                            current_round_evaluations[oid][dim].append((score, confidence))

                    # Update legacy derived scores for backward compatibility
                    self._update_agent_derived_scores(aid, derived_scores.get(aid, {}), dimensions, confidences.get(aid, {}))
                    self._update_agent_confidences(aid, confidences.get(aid, {}), dimensions)
                    self._update_agent_derived_scores(oid, derived_scores.get(oid, {}), dimensions, confidences.get(oid, {}))
                    self._update_agent_confidences(oid, confidences.get(oid, {}), dimensions)

        # Handle variance-based trial adjustment if applicable
        if self.num_trials > 1:
            all_variances = []
            for (a,b) in pairs_to_eval_og if 'pairs_to_eval_og' in locals() else []:
                pair = (min(a, b), max(a, b))
                evals_for_pair = self.pair_evaluation_memory.get(pair, [])[-self.num_trials:]
                derived_scores_for_pair = [eval['derived_scores'][a] for eval in evals_for_pair]
                for dim in derived_scores_for_pair[0].keys():
                    scores_for_dim = [eval['derived_scores'][a][dim] for eval in evals_for_pair]
                    variances = np.var(scores_for_dim)
                    all_variances.append(variances)
            avg_variance = np.mean(all_variances)
            var_threshold = self.config.get('var_threshold', 0.1)
            max_trials = self.config.get('max_eval_trials', 5)
            min_trials = self.config.get('min_eval_trials', 2)
            self.num_trials = max(min_trials, min(1, int(avg_variance / var_threshold))*max_trials)
            print(f"INFO ({self.source_id}): Adjusting num_trials from {self.num_trials} to {self.num_trials} based on variance {avg_variance}.")
                

        # if analysis_mode or detailed_analysis:
            # with open(f"outputs/detailed_analysis/memory_based/prompts_for_pairs_{self.source_id}_crossReg2.txt", "w") as f:
            #     for pair, prompt in prompts_for_pairs.items():
            #         # Replace literal '\n' in prompt with actual newlines before writing
            #         if isinstance(prompt, str):
            #             prompt_to_write = prompt.replace("\\n", "\n")
            #         else:
            #             prompt_to_write = str(prompt)
            #         f.write(f"{pair}:\n{prompt_to_write}\n\n")
            # ipdb.set_trace()
        # ------------------------------------------------------------
        # Phase 4 – Bayesian belief state updates and final scores
        # ------------------------------------------------------------
        final_evals = {}
        for aid in agent_ids:
            final_evals[aid] = {}
            
            for dim in dimensions:
                current_evaluations = current_round_evaluations.get(aid, {}).get(dim, [])
                
                if current_evaluations:
                    # Aggregate current round evaluations
                    # Use confidence-weighted average for multiple evaluations in the same round
                    aggregated_score, aggregated_confidence = self.combine_multiple_beliefs(
                        [eval[0] for eval in current_evaluations],
                        [eval[1] for eval in current_evaluations]
                    )
                    
                    # Update belief state with aggregated current round evaluation
                    updated_score, updated_confidence = self._update_belief_state_bayesian(
                        aid, dim, aggregated_score, aggregated_confidence
                    )
                    final_evals[aid][dim] = (updated_score, updated_confidence)
                else:
                    # No new evaluations for this agent-dimension
                    belief = self.belief_state[aid][dim]
                    if belief is not None:
                        # Return current belief state
                        score, confidence = self._beta_params_to_score_and_confidence(*belief)
                        final_evals[aid][dim] = (score, confidence)
                    else:
                        # Use base score for agents without any prior belief
                        base_score, base_conf = base_scores_with_conf.get(aid, {}).get(dim, (0.5, 0.3))
                        final_evals[aid][dim] = (base_score, base_conf * 0.9)  # Slight decay as in original

        if detailed_analysis:
            return final_evals, comparison_log
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

