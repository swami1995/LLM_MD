import time
from collections import defaultdict, Counter
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Set, Optional, Union, Any
import hashlib
import re
import json
import ipdb
import copy

class InformationSource:
    """
    Base class for all information sources in the trust market.
    
    Risk Configuration Parameters:
    ----------------------------
    # Core Risk Calculation
    - volatility_risk_weight (float, default=0.5): Weight 'w' in Total_Risk = sqrt((Risk_MC)^2 + (w * Risk_Vol)^2)
    - volatility_normalization_method (str, default='historical_range'): Method to normalize volatility
      Options: 'historical_range', 'price_scaling', 'capital_scaling', 'sigmoid'
    
    # Volatility Component Weights (for combining different volatility sources)
    - agent_volatility_weight (float, default=0.6): Weight for agent prediction volatility vs market volatility
    - market_volatility_weight (float, default=0.4): Weight for market volatility (should sum to 1.0 with agent_volatility_weight)
    - mean_volatility_weight (float, default=0.7): Weight for mean/score volatility vs momentum volatility
    - momentum_volatility_weight (float, default=0.3): Weight for momentum (first-difference) volatility
    - confidence_volatility_weight (float, default=0.4): Weight for confidence volatility in agent predictions
    - score_capital_volatility_weight (float, default=0.6): Weight for score vs capital volatility in market metrics
    
    # Volatility Normalization Parameters
    - max_historical_volatility (float, default=1.0): Maximum volatility seen (updated automatically)
    - min_historical_volatility (float, default=0.0): Minimum volatility for normalization
    - volatility_sigmoid_steepness (float, default=5.0): Steepness parameter for sigmoid normalization
    
    # Risk-Adjusted Investment Parameters
    - risk_adjustment_method (str, default='multiplicative'): Method for adjusting investments based on risk
      Options: 'multiplicative', 'additive', 'threshold', 'exponential'
    - risk_aversion_factor (float, default=0.5): How much to penalize investments based on risk
    - min_investment_after_risk_adjustment (float, default=0.01): Minimum investment amount after risk adjustment
    - max_risk_threshold (float, default=0.8): Maximum risk ratio above which investment is zero (threshold method)
    - adjust_confidence_with_risk (bool, default=False): Whether to lower confidence when risk is high
    - risk_confidence_penalty (float, default=0.2): How much to reduce confidence based on risk ratio
    """
    
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

        self.last_evaluation_round = -1
        self.compared_pairs = set() # Track which pairs have been compared in current round

        # Cached evaluations for reuse across runs
        self.cached_evaluations = {}  # {evaluation_round: {agent_id: {dimension: (score, confidence)}}}
        self.cached_comparison_log = {}  # {evaluation_round: {agent_id: {dimension: (score, confidence)}}}
        self.use_cached_evaluations = False  # Flag to enable using cached evaluations

        # --- NEW: mirror user_rep tracking for confidences & cache ---
        self.derived_agent_scores = defaultdict(lambda: defaultdict(list))
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
        
        # --- Prediction Volatility Tracking ---
        # History of posterior belief means for volatility computation
        self.belief_mean_history = defaultdict(lambda: defaultdict(list))  # agent_id -> dim -> [mean_history]
        self.belief_confidence_history = defaultdict(lambda: defaultdict(list))  # agent_id -> dim -> [confidence_history]
        # Track prediction volatility metrics
        self.prediction_volatility_history = []  # List of volatility metrics per evaluation round
        self.prediction_volatility_window_size = 10  # Number of evaluations to consider for volatility
        
        # Prediction Volatility Tracking
        self.prediction_volatility = defaultdict(lambda: defaultdict(float))  # agent_id -> dim -> volatility score
        
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

    # -----------------------------
    # Optimization helpers (AMM-aware)
    # -----------------------------
    def _amm_post_trade_value(self, R: float, T: float, S: float, cash: float) -> float:
        """
        Pure function: compute post-trade portfolio value for this source's position
        in an agent-dimension given a cash trade `cash` under CPMM dynamics.
        cash > 0: invest (buy); cash < 0: divest (sell). Returns (S' * p').
        """
        # Guardrails
        R = max(R, 1e-9)
        T = max(T, 1e-9)
        if cash == 0.0:
            return (R / T) * S

        if cash > 0:
            # Buy with x=cash
            x = cash
            q = x * T / (R + x)
            T_new = max(T - q, 1e-9)
            R_new = R + x
            price_new = R_new / T_new
            return (S + q) * price_new
        else:
            # Sell for y = -cash (payout)
            y = -cash
            # Cap y to not exceed reserve; caller should also cap sells by share ownership.
            if y >= R:
                y = R - 1e-9
            q = y * T / (R - y)
            R_new = R - y
            T_new = T + q
            price_new = R_new / max(T_new, 1e-9)
            return max(0.0, (S - q)) * price_new

    def _compute_trade_bounds(self, R: float, T: float, S: float) -> tuple:
        """
        Returns (sell_cap, buy_cap) in cash units under CPMM constraints.
        - sell_cap limited by reserve and share ownership.
        - buy_cap left effectively unbounded here (caller may impose per-asset cap).
        """
        R = max(R, 1e-9)
        T = max(T, 1e-9)
        # Reserve bound: y < R
        sell_cap_reserve = max(R - 1e-9, 0.0)
        # Share ownership bound: y <= R*S/(T+S)
        sell_cap_shares = (R * S / (T + S)) if (T + S) > 1e-12 else 0.0
        sell_cap = max(0.0, min(sell_cap_reserve, sell_cap_shares))
        buy_cap = float('inf')
        return (sell_cap, buy_cap)

    def _solve_asset_trade_L2(self,
                              R: float,
                              T: float,
                              S: float,
                              V_target: float,
                              V_prev: float,
                              V_init: float,
                              risk: float,
                              weight: float,
                              lam_prox: float,
                              rho_risk: float,
                              tau_turnover: float,
                              buy_mu: float = 0.0,
                              buy_allowed: bool = True,
                              sell_allowed: bool = True,
                              grid_points: int = 25,
                              buy_cap: int = float('inf')) -> float:
        """
        Solve 1D bounded optimization over cash c to minimize:
          f(c) = weight*(V(c)-V_target)^2 + lam_prox*(V(c)-V_prev)^2 + rho_risk*risk*max(c,0)^2 + tau_turnover*abs(c)
        subject to c in [-sell_cap, buy_cap], and respecting buy/sell allowed flags.
        Returns optimal cash c* (can be 0 if constrained).
        Uses coarse grid search followed by local refinement; robust and dependency-free.
        """
        sell_cap, _ = self._compute_trade_bounds(R, T, S)
        # Apply directional constraints
        lo = -sell_cap if sell_allowed else 0.0
        hi = (buy_cap if buy_allowed else 0.0)
        # Handle trivial infeasible direction
        if lo >= hi:
            return 0.0

        def obj(cash: float) -> float:
            V = self._amm_post_trade_value(R, T, S, cash)
            # Quadratic tracking + proximal + risk buy penalty + linear turnover
            return (
                weight * (V - V_target) ** 2
                + lam_prox * (V - V_prev) ** 2
                + (rho_risk * max(cash, 0.0) ** 2) * max(risk, 0.0)
                + max(buy_mu, 0.0) * max(cash, 0.0)
                + tau_turnover * abs(cash)
            )

        # Coarse grid
        best_c = 0.0
        best_f = obj(0.0)
        if grid_points < 5:
            grid_points = 5
        # Guard against infinite bounds producing NaNs (e.g., inf * 0)
        lo_finite = np.isfinite(lo)
        hi_finite = np.isfinite(hi)
        for k in range(grid_points + 1):
            # Build c carefully to avoid inf*0 -> NaN
            if lo_finite and hi_finite:
                c = lo + (hi - lo) * (k / grid_points)
            else:
                # Use a local symmetric bracket around 0 when one side is infinite
                span_local = 1.0
                # Try to scale span based on R if available via closure or defaults
                # but here we just incrementally explore reasonable magnitudes
                u = (k / grid_points)
                c_pos = span_local * (u / max(1e-9, (1.0 - u))) if k < grid_points else span_local * 10.0
                c_neg = -span_local * (u / max(1e-9, (1.0 - u))) if k < grid_points else -span_local * 10.0
                if lo_finite and not hi_finite:
                    # Negative side: from lo to 0, positive side: 0 to growing values
                    if k <= grid_points // 2:
                        # Sample negative uniformly between lo and 0
                        frac = k / max(1, grid_points // 2)
                        c = lo + (0.0 - lo) * frac
                    else:
                        c = c_pos
                elif hi_finite and not lo_finite:
                    if k <= grid_points // 2:
                        c = -abs(c_neg)
                    else:
                        frac = (k - grid_points // 2) / max(1, grid_points - grid_points // 2)
                        c = 0.0 + (hi - 0.0) * frac
                else:
                    # both infinite: sample around 0 with symmetric growth
                    c = c_pos if k % 2 == 0 else -abs(c_neg)
            if not np.isfinite(c):
                continue
            f = obj(c)
            if not np.isfinite(f):
                continue
            if f < best_f:
                best_f, best_c = f, c

        # Local refinement around best_c using a small bracket
        # Local refinement around best_c using a small finite bracket
        if lo_finite and hi_finite and np.isfinite(hi - lo):
            span = max((hi - lo) * 0.1, 1e-6)
            a = max(lo, best_c - span)
            b = min(hi, best_c + span)
        else:
            # Fallback finite bracket centered at best_c
            span = max(1.0, abs(best_c) + 1.0)
            a = best_c - span
            b = best_c + span
        # Golden-section-like refinement
        phi = (1 + 5 ** 0.5) / 2
        invphi = 1 / phi
        invphi2 = invphi ** 2
        n_refine = 15
        c1 = b - (b - a) * invphi
        c2 = a + (b - a) * invphi
        f1 = obj(c1)
        f2 = obj(c2)
        for _ in range(n_refine):
            if f1 > f2:
                a = c1
                c1 = c2
                f1 = f2
                c2 = a + (b - a) * invphi
                f2 = obj(c2)
            else:
                b = c2
                c2 = c1
                f2 = f1
                c1 = b - (b - a) * invphi
                f1 = obj(c1)
        c_candidate = (a + b) / 2
        f_candidate = obj(c_candidate)
        if np.isfinite(f_candidate) and f_candidate < best_f:
            best_c = c_candidate

        return float(best_c)

    def _solve_budget_coupled_trades(self,
                                     assets: list,
                                     available_cash: float,
                                     lam_prox: float,
                                     rho_risk: float,
                                     tau_turnover: float,
                                     grid_points: int = 31,
                                     buys_cap: float = None,
                                     tol: float = 1e-4,
                                     max_iter: int = 30) -> dict:
        """
        Jointly solve per-dimension trades across assets subject to a shared budget
        using a dual (μ) search. Each asset is solved via _solve_asset_trade_L2 with an
        added linear buy term μ·max(cash,0). We increase μ until total buys fit the
        budget (available_cash + sell proceeds) within tolerance.

        assets: list of dicts with keys:
            'key' (hashable id), 'R','T','S','V_target','V_prev','V_init','risk','weight',
            'buy_allowed','sell_allowed'

        Returns: dict key -> optimal cash trade.
        """
        # Helper to solve given mu and return trades, buys, sells
        def solve_given_mu(mu: float):
            trades = {}
            buys = 0.0
            sells = 0.0
            for a in assets:
                c = self._solve_asset_trade_L2(
                    a['R'], a['T'], a['S'],
                    V_target=a['V_target'],
                    V_prev=a['V_prev'],
                    V_init=a.get('V_init', 0.0),
                    risk=a.get('risk', 0.0),
                    weight=a.get('weight', 1.0),
                    lam_prox=lam_prox,
                    rho_risk=rho_risk,
                    tau_turnover=tau_turnover,
                    buy_mu=mu,
                    buy_allowed=a.get('buy_allowed', True),
                    sell_allowed=a.get('sell_allowed', True),
                    grid_points=grid_points,
                    buy_cap=buys_cap
                )
                trades[a['key']] = c
                if c >= 0:
                    buys += c
                else:
                    sells += -c
            return trades, buys, sells

        # If already feasible at mu=0, return that solution
        mu_lo = 0.0
        trades, buys, sells = solve_given_mu(mu_lo)
        budget = max(0.0, available_cash) + sells
        if buys_cap is not None:
            budget = min(budget, max(0.0, buys_cap))
        if buys <= budget + tol:
            return trades

        # Find an upper bound mu_hi that makes it feasible
        mu_hi = 1.0
        for _ in range(40):  # expand up to ~1e12 if needed
            trades_hi, buys_hi, sells_hi = solve_given_mu(mu_hi)
            budget_hi = max(0.0, available_cash) + sells_hi
            if buys_cap is not None:
                budget_hi = min(budget_hi, max(0.0, buys_cap))
            if buys_hi <= budget_hi + tol:
                break
            mu_hi *= 2.0
        else:
            # If still infeasible, return scaled version of mu=0 trades as a fallback
            # to preserve feasibility (should be rare)
            buys0 = buys
            sells0 = sells
            budget0 = max(0.0, available_cash) + sells0
            scale = 0.0 if buys0 <= 1e-9 else min(1.0, budget0 / buys0)
            return {k: (v * scale if v > 0 else v) for k, v in trades.items()}

        # Bisection search over [mu_lo, mu_hi]
        best = trades
        for _ in range(max_iter):
            mu_mid = 0.5 * (mu_lo + mu_hi)
            trades_mid, buys_mid, sells_mid = solve_given_mu(mu_mid)
            budget_mid = max(0.0, available_cash) + sells_mid
            if buys_cap is not None:
                budget_mid = min(budget_mid, max(0.0, buys_cap))
            if buys_mid <= budget_mid + tol:
                best = trades_mid
                mu_hi = mu_mid
            else:
                mu_lo = mu_mid
            if mu_hi - mu_lo <= 1e-9:
                break
        return best

    def _scale_trades_to_budget(self, cash_trades: dict, available_cash: float) -> dict:
        """
        Uniformly scale positive cash trades to fit budget: sum(buys) <= available_cash + sum(sells).
        Sells are not scaled (to not break feasibility from share caps). Returns new dict.
        """
        buys = sum(max(c, 0.0) for c in cash_trades.values())
        sells = sum(max(-c, 0.0) for c in cash_trades.values())
        budget = max(0.0, available_cash) + sells
        if buys <= budget + 1e-9:
            return cash_trades
        scale = 0.0 if buys <= 1e-9 else min(1.0, budget / buys)
        scaled = {}
        for k, c in cash_trades.items():
            if c > 0:
                scaled[k] = c * scale
            else:
                scaled[k] = c
        return scaled
    
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
                 "Use these to inform your judgment.\n"# but be aware that agent behavior can change over time. So the older evaluations might be outdated or stale. Trust your own judgment in case you feel the agent interactions/profile you see above are in contradiction with these evaluations.\n"

        return header + "\n".join(snippets)

    def _get_additional_context_direct(self, agent_a_id, agent_b_id, evaluation_round):
        """
        Build additional prompt context for direct investment decisions between two agents.
        Includes:
        - This source's current investment limits and holdings for the pair
        - Current market state (prices and capital locked) for the pair
        - Regulator's most recent round of investments (if available)
        """
        # 1) Own portfolio snapshot and market state for the two agents
        own_context = ""
        if not self.market:
            own_context = ""
        else:
            # Available capacity (cash) per dimension for this source
            available_cash_per_dim = dict(self.market.source_available_capacity.get(self.source_id, {}))

            # Current prices and capital (R) for Agent A and Agent B
            def _agent_price_and_capital(agent_id):
                prices = {}
                capitals = {}
                for dim in self.expertise_dimensions:
                    # Ensure AMM state exists for this agent-dim
                    if agent_id not in self.market.agent_amm_params:
                        continue
                    if dim not in self.market.agent_amm_params[agent_id]:
                        continue
                    R = float(self.market.agent_amm_params[agent_id][dim]['R'])
                    T = float(self.market.agent_amm_params[agent_id][dim]['T'])
                    price = (R / T) if T > 1e-9 else float(self.market.agent_trust_scores[agent_id].get(dim, 0.5))
                    prices[dim] = price
                    capitals[dim] = R
                return prices, capitals

            prices_a, capitals_a = _agent_price_and_capital(agent_a_id)
            prices_b, capitals_b = _agent_price_and_capital(agent_b_id)
            # Total market capital (R) across the two agents and a third "market" agent C
            capitals_total = {dim: sum(
                float(self.market.agent_amm_params[aid][dim].get('R', 0.0))
                for aid in list(self.market.agent_trust_scores.keys())
            ) for dim in self.expertise_dimensions}

            # This source's holdings (shares and current MTM value) in the two agents
            def _source_holdings_value(agent_id, prices_map):
                shares_per_dim = {}
                value_per_dim = {}
                total_value = 0.0
                # Shares held by this source
                agent_holdings = self.market.source_investments.get(self.source_id, {}).get(agent_id, {})
                for dim in self.expertise_dimensions:
                    shares = float(agent_holdings.get(dim, 0.0))
                    price = float(prices_map.get(dim, 0.5))
                    val = shares * price
                    if shares > 0 or dim in prices_map:
                        shares_per_dim[dim] = shares
                        value_per_dim[dim] = val
                        total_value += val
                return shares_per_dim, value_per_dim, total_value

            shares_a, values_a, total_value_a = _source_holdings_value(agent_a_id, prices_a)
            shares_b, values_b, total_value_b = _source_holdings_value(agent_b_id, prices_b)
            # Compose context string
            own_context = (
                "\n\nYour current portfolio context (as an investor):\n"
                f"- Available uninvested cash per dimension: \n {json.dumps(available_cash_per_dim)}\n"
                f"- Total investment by the market on Agent A per dim: {json.dumps(capitals_a)}\n"
                f"- Your holdings/investments in Agent A (value) per dim: {json.dumps(values_a)} \n"
                f"- Total investment by the market on Agent B per dim: {json.dumps(capitals_b)}\n"
                f"- Your holdings/investments in Agent B (value) per dim: {json.dumps(values_b)}\n"
                f"- Total investments/marketcap in the market across 3 agents (A, B and C) per dim: {json.dumps(capitals_total)}\n"
            )
        # 2) Regulator's most recent investments (if available)
        regulator_context = ""
        if self.market and 'regulator' in self.market.information_sources:
            regulator = self.market.information_sources['regulator']
            last_investments, last_round_num = regulator.last_investment_round()

            if last_investments and last_round_num is not None and len(last_investments)>0:
                relative_round_str = ""
                if isinstance(evaluation_round, int) and isinstance(last_round_num, int):
                    diff_rounds = evaluation_round - last_round_num
                    if diff_rounds == 0:
                        relative_round_str = " (this round)"
                    elif diff_rounds == 1:
                        relative_round_str = " (last round)"
                    else:
                        relative_round_str = f" ({diff_rounds} rounds ago)"

                # Filter to the two agents if present
                inv_a = [(e[1],e[2]) for e in last_investments if e[0] == agent_a_id]
                inv_b = [(e[1],e[2]) for e in last_investments if e[0] == agent_b_id]
                snippet = {
                    'relative_time': relative_round_str.strip(),
                    f'agent_{agent_a_id}': inv_a,
                    f'agent_{agent_b_id}': inv_b
                }
                regulator_context = (
                    "\n\nFor additional context, here are the most recent trades from the Regulator, "
                    "a highly trusted source with broader visibility (profiles, more conversations, etc.). "
                    "Default to trusting these decisions unless your observed evidence strongly contradicts them.\n"
                    f"Regulator last trades: {json.dumps(snippet)}\n"
                )

        return f"{own_context}{regulator_context}".strip()
    
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
        EPSILON = 1e-6  # Small value to avoid division by zero
        # Clamp the score to be slightly away from 0 and 1
        score = max(EPSILON, min(1.0 - EPSILON, score))
        
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
        
        if prior_belief is None or self.config.get('use_bayesian_updates', True) is False:
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
    
    # ------------------------------------------------------------------
    # Prediction Volatility Tracking Methods
    # ------------------------------------------------------------------
    def update_prediction_volatility_tracking(self, evaluation_round: Optional[int] = None) -> None:
        """Update prediction volatility tracking with current belief states."""
        current_round = evaluation_round if evaluation_round is not None else getattr(self, 'last_evaluation_round', 0)
        
        # Track belief mean and confidence history for all agents/dimensions
        for agent_id in self.belief_state.keys():
            for dimension in self.belief_state[agent_id].keys():
                belief_params = self.belief_state[agent_id][dimension]
                if belief_params is not None:
                    # Compute current mean and confidence from belief state
                    current_mean, current_confidence = self._beta_params_to_score_and_confidence(*belief_params)
                    
                    # Update history
                    self.belief_mean_history[agent_id][dimension].append(current_mean)
                    self.belief_confidence_history[agent_id][dimension].append(current_confidence)
                    
                    # Keep only recent history within window
                    if len(self.belief_mean_history[agent_id][dimension]) > self.prediction_volatility_window_size:
                        self.belief_mean_history[agent_id][dimension] = \
                            self.belief_mean_history[agent_id][dimension][-self.prediction_volatility_window_size:]
                    if len(self.belief_confidence_history[agent_id][dimension]) > self.prediction_volatility_window_size:
                        self.belief_confidence_history[agent_id][dimension] = \
                            self.belief_confidence_history[agent_id][dimension][-self.prediction_volatility_window_size:]
    
    def compute_prediction_volatility(self) -> Dict[str, float]:
        """Compute various prediction volatility metrics."""
        volatility_metrics = {}
        
        # Compute mean volatility across all agents/dimensions
        all_mean_volatilities = defaultdict(lambda: defaultdict(float))
        all_mean_momentum_volatilities = defaultdict(lambda: defaultdict(float))
        for agent_id in self.belief_mean_history.keys():
            for dimension in self.belief_mean_history[agent_id].keys():
                mean_history = self.belief_mean_history[agent_id][dimension]
                mean_history_np = np.array(mean_history)
                mean_diff_history = mean_history_np[1:] - mean_history_np[:-1]  # Compute first differences
                if len(mean_history) > 1:
                    mean_volatility = np.std(mean_history)
                    mean_momentum_volatility = np.std(mean_diff_history)
                    all_mean_volatilities[agent_id][dimension] = mean_volatility
                    all_mean_momentum_volatilities[agent_id][dimension] = mean_momentum_volatility
        
        # Compute confidence volatility across all agents/dimensions
        all_confidence_volatilities = defaultdict(lambda: defaultdict(float))
        all_confidence_momentum_volatilities = defaultdict(lambda: defaultdict(float))
        for agent_id in self.belief_confidence_history.keys():
            for dimension in self.belief_confidence_history[agent_id].keys():
                confidence_history = self.belief_confidence_history[agent_id][dimension]
                confidence_history_np = np.array(confidence_history)
                confidence_diff_history = confidence_history_np[1:] - confidence_history_np[:-1]  # Compute first differences
                if len(confidence_history) > 1:
                    confidence_volatility = np.std(confidence_history)
                    confidence_momentum_volatility = np.std(confidence_diff_history)
                    all_confidence_volatilities[agent_id][dimension] = confidence_volatility
                    all_confidence_momentum_volatilities[agent_id][dimension] = confidence_momentum_volatility
        
        volatility_metrics['mean_volatilities'] = all_mean_volatilities
        volatility_metrics['mean_momentum_volatilities'] = all_mean_momentum_volatilities
        volatility_metrics['confidence_volatilities'] = all_confidence_volatilities
        volatility_metrics['confidence_momentum_volatilities'] = all_confidence_momentum_volatilities        
        
        # Store computed metrics
        volatility_record = {
            'round': getattr(self, 'last_evaluation_round', 0),
            'source_id': self.source_id,
            'timestamp': time.time(),
            **volatility_metrics
        }
        self.prediction_volatility_history.append(volatility_record)
        
        # Keep only recent history
        if len(self.prediction_volatility_history) > self.prediction_volatility_window_size:
            self.prediction_volatility_history = self.prediction_volatility_history[-self.prediction_volatility_window_size:]
        
        return volatility_metrics

    def compute_risk(self, projected_capital_holdings, current_capital_holdings, type='relative_capital'):
        """
        Compute risk based on projected capital holdings and current capital holdings.
        
        Parameters:
        - projected_capital_holdings: Dict[agent_id][dimension] = (mean_capital, std_capital)
        - current_capital_holdings: Dict[agent_id][dimension] = current_capital
        - type: Type of risk calculation ('relative_capital', 'absolute_capital')
        
        Returns:
        - Dict[agent_id][dimension] = {'risk': float, 'capital_mean': float}
        """
        risk_results = defaultdict(lambda: defaultdict(dict))
        
        for agent_id in projected_capital_holdings:
            for dimension in projected_capital_holdings[agent_id]:
                if isinstance(projected_capital_holdings[agent_id][dimension], tuple):
                    mean_capital, std_capital = projected_capital_holdings[agent_id][dimension]
                else:
                    mean_capital = projected_capital_holdings[agent_id][dimension]
                    std_capital = 0.0
                
                if type == 'relative_capital':
                    # Relative risk based on capital holdings
                    current_capital = current_capital_holdings.get(agent_id, {}).get(dimension, 1.0)
                    risk = std_capital / max(1e-6, current_capital)
                elif type == 'absolute_capital':
                    # Absolute risk based on standard deviation of capital
                    risk = std_capital
                else:
                    raise ValueError(f"Unknown risk type: {type}")
                # Store results
                risk_results[agent_id][dimension] = risk
        return risk_results

    # ------------------------------------------------------------------
    # Elaborate Risk Computation Methods - Not used yet : We'll figure it out later
    # ------------------------------------------------------------------
    def compute_risk_elaborate(self, projected_capital_holdings, market_prices=None, current_capital_holdings=None):
        """
        Compute comprehensive risk for each dimension based on:
        1. Monte Carlo risk (standard deviation of projected capital)
        2. Volatility-based risk (normalized historical volatility scaled to capital amount)
        3. Combined total risk using configurable weighting
        
        Parameters:
        - projected_capital_holdings: Dict[agent_id][dimension] = (mean_capital, std_capital)
        - market_prices: Dict[agent_id][dimension] = price (for scaling volatility risk)
        - current_capital_holdings: Dict[agent_id][dimension] = current_capital (for scaling)
        
        Returns:
        - Dict[agent_id][dimension] = {'monte_carlo_risk': float, 'volatility_risk': float, 'total_risk': float}
        """
        risk_results = defaultdict(lambda: defaultdict(dict))
        
        # Get configuration parameters
        volatility_weight = self.config.get('volatility_risk_weight', 0.5)
        volatility_normalization_method = self.config.get('volatility_normalization_method', 'historical_range')
        
        # Get current volatility metrics
        agent_volatility_metrics = self.compute_prediction_volatility()
        
        # Get market volatility if market is available
        market_volatility_metrics = {}
        if self.market is not None:
            market_volatility_metrics = self.market.compute_market_volatility()
        
        for agent_id in projected_capital_holdings:
            for dimension in projected_capital_holdings[agent_id]:
                # --- 1. Monte Carlo Risk (Risk_MC) ---
                if isinstance(projected_capital_holdings[agent_id][dimension], tuple):
                    mean_capital, std_capital = projected_capital_holdings[agent_id][dimension]
                    monte_carlo_risk = std_capital
                else:
                    # If not a tuple, assume it's just the mean with no std
                    mean_capital = projected_capital_holdings[agent_id][dimension]
                    monte_carlo_risk = 0.0
                
                # --- 2. Volatility-Based Risk (Risk_Vol) ---
                volatility_risk = self._compute_volatility_risk(
                    agent_id, dimension, mean_capital, 
                    agent_volatility_metrics, market_volatility_metrics,
                    market_prices, current_capital_holdings,
                    volatility_normalization_method
                )
                
                # --- 3. Total Risk ---
                # Total_Risk = sqrt((Risk_MC)^2 + (w * Risk_Vol)^2)
                total_risk = np.sqrt(monte_carlo_risk**2 + (volatility_weight * volatility_risk)**2)
                
                # Store results
                risk_results[agent_id][dimension] = {
                    'monte_carlo_risk': monte_carlo_risk,
                    'volatility_risk': volatility_risk,
                    'weighted_volatility_risk': volatility_weight * volatility_risk,
                    'total_risk': total_risk,
                    'capital_mean': mean_capital
                }
        
        return risk_results
    
    def _compute_volatility_risk(self, agent_id, dimension, capital_amount, 
                                agent_volatility_metrics, market_volatility_metrics,
                                market_prices=None, current_capital_holdings=None,
                                normalization_method='historical_range'):
        """
        Compute volatility-based risk by combining agent prediction volatility and market volatility,
        then scaling it to the capital amount.
        
        Parameters:
        - agent_id: The agent ID
        - dimension: The dimension 
        - capital_amount: The projected capital amount to scale the risk to
        - agent_volatility_metrics: Output from compute_prediction_volatility()
        - market_volatility_metrics: Output from market.compute_market_volatility()
        - market_prices: Current market prices for normalization
        - current_capital_holdings: Current capital holdings for reference
        - normalization_method: Method to normalize volatility ('historical_range', 'price_scaling', 'capital_scaling')
        
        Returns:
        - float: Volatility-based risk scaled to capital amount
        """
        # Get agent-specific volatility components
        agent_mean_volatility = agent_volatility_metrics.get('mean_volatilities', {}).get(agent_id, {}).get(dimension, 0.0)
        agent_confidence_volatility = agent_volatility_metrics.get('confidence_volatilities', {}).get(agent_id, {}).get(dimension, 0.0)
        agent_mean_momentum = agent_volatility_metrics.get('mean_momentum_volatilities', {}).get(agent_id, {}).get(dimension, 0.0)
        agent_confidence_momentum = agent_volatility_metrics.get('confidence_momentum_volatilities', {}).get(agent_id, {}).get(dimension, 0.0)
        
        # Get market-specific volatility components
        market_score_volatility = market_volatility_metrics.get('score_volatilities', {}).get(agent_id, {}).get(dimension, 0.0)
        market_capital_volatility = market_volatility_metrics.get('capital_volatility', {}).get(agent_id, {}).get(dimension, 0.0)
        market_score_momentum = market_volatility_metrics.get('score_momentum_volatilities', {}).get(agent_id, {}).get(dimension, 0.0)
        market_capital_momentum = market_volatility_metrics.get('capital_momentum_volatilities', {}).get(agent_id, {}).get(dimension, 0.0)
        
        # Combine volatility components with configurable weights
        agent_volatility_weight = self.config.get('agent_volatility_weight', 0.6)
        market_volatility_weight = self.config.get('market_volatility_weight', 0.4)
        mean_volatility_weight = self.config.get('mean_volatility_weight', 0.7)
        momentum_weight = self.config.get('momentum_volatility_weight', 0.3)
        confidence_weight = self.config.get('confidence_volatility_weight', 0.4)
        score_capital_weight = self.config.get('score_capital_volatility_weight', 0.6)
        
        # Combine agent volatilities
        combined_agent_volatility = (
            mean_volatility_weight * agent_mean_volatility +
            momentum_weight * agent_mean_momentum +
            confidence_weight * agent_confidence_volatility +
            momentum_weight * confidence_weight * agent_confidence_momentum
        )
        
        # Combine market volatilities  
        combined_market_volatility = (
            score_capital_weight * market_score_volatility +
            (1 - score_capital_weight) * market_capital_volatility +
            momentum_weight * score_capital_weight * market_score_momentum +
            momentum_weight * (1 - score_capital_weight) * market_capital_momentum
        )
        
        # Overall combined volatility
        combined_volatility = (
            agent_volatility_weight * combined_agent_volatility +
            market_volatility_weight * combined_market_volatility
        )
        
        # Normalize the combined volatility to a [0, 1] scale
        normalized_volatility = self._normalize_volatility(
            combined_volatility, agent_id, dimension, 
            normalization_method, market_prices, current_capital_holdings
        )
        
        # Scale the normalized volatility to the capital amount
        volatility_risk = normalized_volatility * abs(capital_amount)
        
        return volatility_risk
    
    def _normalize_volatility(self, combined_volatility, agent_id, dimension, 
                            normalization_method='historical_range', 
                            market_prices=None, current_capital_holdings=None):
        """
        Normalize volatility to a [0, 1] range using different methods.
        
        Parameters:
        - combined_volatility: The raw combined volatility value
        - agent_id: Agent ID for context
        - dimension: Dimension for context
        - normalization_method: Method to use for normalization
        - market_prices: Current market prices
        - current_capital_holdings: Current capital holdings
        
        Returns:
        - float: Normalized volatility in [0, 1] range
        """
        if combined_volatility <= 0:
            return 0.0
            
        if normalization_method == 'historical_range':
            # Normalize based on historical range of volatilities seen
            max_historical_volatility = self.config.get('max_historical_volatility', 1.0)
            min_historical_volatility = self.config.get('min_historical_volatility', 0.0)
            
            # Update max if current volatility is higher
            if combined_volatility > max_historical_volatility:
                max_historical_volatility = combined_volatility
                self.config['max_historical_volatility'] = max_historical_volatility
            
            if max_historical_volatility > min_historical_volatility:
                normalized = (combined_volatility - min_historical_volatility) / (max_historical_volatility - min_historical_volatility)
            else:
                normalized = 0.5  # Default middle value if no range
                
        elif normalization_method == 'price_scaling':
            # Normalize based on current market price
            if market_prices and agent_id in market_prices and dimension in market_prices[agent_id]:
                current_price = market_prices[agent_id][dimension]
                # Scale volatility relative to price (higher price = higher tolerance for absolute volatility)
                normalized = combined_volatility / max(current_price, 0.01)  # Avoid division by zero
            else:
                normalized = combined_volatility  # Fallback to raw volatility
                
        elif normalization_method == 'capital_scaling':
            # Normalize based on current capital holdings
            if current_capital_holdings and agent_id in current_capital_holdings and dimension in current_capital_holdings[agent_id]:
                current_capital = current_capital_holdings[agent_id][dimension]
                # Scale volatility relative to current capital
                normalized = combined_volatility / max(abs(current_capital), 1.0)
            else:
                normalized = combined_volatility
                
        elif normalization_method == 'sigmoid':
            # Use sigmoid function to bound volatility to [0, 1]
            sigmoid_steepness = self.config.get('volatility_sigmoid_steepness', 5.0)
            normalized = 1.0 / (1.0 + np.exp(-sigmoid_steepness * combined_volatility))
            
        else:
            # Default: simple clipping to [0, 1]
            normalized = min(1.0, max(0.0, combined_volatility))
        
        return min(1.0, max(0.0, normalized))  # Ensure result is in [0, 1]
    
    def compute_risk_adjusted_investment_amounts(self, base_investment_amounts, risk_results, 
                                               risk_adjustment_method='multiplicative'):
        """
        Adjust investment amounts based on computed risk to implement risk-sensitive investment.
        
        Parameters:
        - base_investment_amounts: Dict[agent_id][dimension] = base_amount
        - risk_results: Output from compute_risk()
        - risk_adjustment_method: Method to adjust investments ('multiplicative', 'additive', 'threshold')
        
        Returns:
        - Dict[agent_id][dimension] = risk_adjusted_amount
        """
        adjusted_amounts = defaultdict(lambda: defaultdict(float))
        
        # Get configuration parameters
        risk_aversion_factor = self.config.get('risk_aversion_factor', 0.5)  # How much to penalize risk
        min_investment_after_risk = self.config.get('min_investment_after_risk_adjustment', 0.01)
        max_risk_threshold = self.config.get('max_risk_threshold', 0.8)  # Above this, no investment
        
        for agent_id in base_investment_amounts:
            for dimension in base_investment_amounts[agent_id]:
                base_amount = base_investment_amounts[agent_id][dimension]
                
                if agent_id in risk_results and dimension in risk_results[agent_id]:
                    risk_data = risk_results[agent_id][dimension]
                    total_risk = risk_data['total_risk']
                    capital_mean = risk_data['capital_mean']
                    
                    # Normalize risk relative to capital mean to get risk ratio
                    if abs(capital_mean) > 0.001:
                        risk_ratio = total_risk / abs(capital_mean)
                    else:
                        risk_ratio = total_risk  # If capital is near zero, risk is the absolute risk
                    
                    # Apply risk adjustment based on method
                    if risk_adjustment_method == 'multiplicative':
                        # Reduce investment proportionally to risk
                        risk_multiplier = max(0.0, 1.0 - risk_aversion_factor * risk_ratio)
                        adjusted_amount = base_amount * risk_multiplier
                        
                    elif risk_adjustment_method == 'additive':
                        # Subtract risk-based penalty from investment
                        risk_penalty = risk_aversion_factor * total_risk
                        adjusted_amount = base_amount - risk_penalty
                        
                    elif risk_adjustment_method == 'threshold':
                        # Zero investment if risk is too high, otherwise use base amount
                        if risk_ratio > max_risk_threshold:
                            adjusted_amount = 0.0
                        else:
                            adjusted_amount = base_amount
                            
                    elif risk_adjustment_method == 'exponential':
                        # Exponentially decay investment with risk
                        risk_multiplier = np.exp(-risk_aversion_factor * risk_ratio)
                        adjusted_amount = base_amount * risk_multiplier
                        
                    else:
                        # Default: no adjustment
                        adjusted_amount = base_amount
                    
                    # Ensure minimum investment if positive
                    if base_amount > 0 and adjusted_amount > 0:
                        adjusted_amount = max(adjusted_amount, min_investment_after_risk)
                    elif base_amount < 0 and adjusted_amount < 0:
                        adjusted_amount = min(adjusted_amount, -min_investment_after_risk)
                        
                else:
                    # No risk data available, use base amount
                    adjusted_amount = base_amount
                
                adjusted_amounts[agent_id][dimension] = adjusted_amount
        
        return adjusted_amounts
    
    def get_current_capital_holdings_from_market(self):
        """
        Get current capital holdings for this source from the market.
        
        Returns:
        - Dict[agent_id][dimension] = current_capital_amount
        """
        if self.market is None:
            return defaultdict(lambda: defaultdict(float))
            
        current_holdings = defaultdict(lambda: defaultdict(float))
        
        # Get current investment amounts from market
        for agent_id in self.source_investments:
            for dimension in self.source_investments[agent_id]:
                current_holdings[agent_id][dimension] = self.source_investments[agent_id][dimension]
        
        return current_holdings
    
    def risk_adjusted_decide_investments(self, base_investment_method, *args, **kwargs):
        """
        Wrapper method that applies risk adjustment to any base investment decision method.
        
        This can be used by subclasses to easily add risk adjustment to their existing
        decide_investments implementations.
        
        Parameters:
        - base_investment_method: Function that returns base investment decisions
        - *args, **kwargs: Arguments to pass to the base investment method
        
        Returns:
        - List of (agent_id, dimension, risk_adjusted_amount, confidence) tuples
        """
        # Get base investment decisions
        base_decisions = base_investment_method(*args, **kwargs)
        
        # Convert to dict format for easier manipulation
        base_amounts = defaultdict(lambda: defaultdict(float))
        decision_metadata = {}  # Store confidence and other metadata
        
        for agent_id, dimension, amount, confidence in base_decisions:
            base_amounts[agent_id][dimension] = amount
            decision_metadata[(agent_id, dimension)] = {'confidence': confidence}
        
        # Get current evaluations for risk computation
        agent_ids = list(base_amounts.keys())
        dimensions = list(set(dim for agent_dims in base_amounts.values() for dim in agent_dims.keys()))
        
        # Get current evaluations (this should be implemented by subclasses appropriately)
        current_evaluations = {}
        for agent_id in agent_ids:
            current_evaluations[agent_id] = {}
            for dimension in dimensions:
                # Try to get from derived scores or belief state
                if hasattr(self, 'derived_agent_scores') and agent_id in self.derived_agent_scores:
                    score = self.derived_agent_scores[agent_id].get(dimension, 0.5)
                elif hasattr(self, 'belief_state') and agent_id in self.belief_state and dimension in self.belief_state[agent_id]:
                    belief = self.belief_state[agent_id][dimension]
                    if belief is not None:
                        score, conf = self._beta_params_to_score_and_confidence(*belief)
                    else:
                        score, conf = 0.5, 0.3
                else:
                    score, conf = 0.5, 0.3
                
                # Get confidence
                if hasattr(self, '_calculate_derived_confidence'):
                    conf = self._calculate_derived_confidence(agent_id, [dimension]).get(dimension, 0.3)
                else:
                    conf = 0.3
                    
                current_evaluations[agent_id][dimension] = (score, conf)
        
        # Get market prices and current holdings
        market_prices = None
        current_capital_holdings = None
        if self.market is not None:
            try:
                market_prices = self.market.get_market_prices(agent_ids, dimensions)
                current_capital_holdings = self.get_current_capital_holdings_from_market()
            except:
                pass  # Market might not have these methods yet
        
        # Compute Monte Carlo projections for risk assessment
        projected_capital_holdings = {}
        if hasattr(self, '_monte_carlo_check_market_capacity'):
            try:
                _, projected_capitals, _ = self._monte_carlo_check_market_capacity(
                    current_evaluations, market_prices
                )
                projected_capital_holdings = projected_capitals
            except:
                # Fallback: use current evaluations as mean with some default std
                for agent_id in current_evaluations:
                    projected_capital_holdings[agent_id] = {}
                    for dimension in current_evaluations[agent_id]:
                        score, conf = current_evaluations[agent_id][dimension]
                        # Convert to a capital-like value
                        capital_mean = score * 100  # Scale up for capital representation
                        capital_std = (1 - conf) * capital_mean * 0.1  # Higher uncertainty = higher std
                        projected_capital_holdings[agent_id][dimension] = (capital_mean, capital_std)
        
        if not projected_capital_holdings:
            # Ultimate fallback
            for agent_id in base_amounts:
                projected_capital_holdings[agent_id] = {}
                for dimension in base_amounts[agent_id]:
                    projected_capital_holdings[agent_id][dimension] = (abs(base_amounts[agent_id][dimension]) * 10, abs(base_amounts[agent_id][dimension]))
        
        # Compute comprehensive risk
        risk_results = self.compute_risk(
            projected_capital_holdings, 
            market_prices, 
            current_capital_holdings
        )
        
        # Apply risk adjustment to base investment amounts
        risk_adjustment_method = self.config.get('risk_adjustment_method', 'multiplicative')
        adjusted_amounts = self.compute_risk_adjusted_investment_amounts(
            base_amounts, risk_results, risk_adjustment_method
        )
        
        # Convert back to list format
        risk_adjusted_decisions = []
        for agent_id in adjusted_amounts:
            for dimension in adjusted_amounts[agent_id]:
                adjusted_amount = adjusted_amounts[agent_id][dimension]
                original_confidence = decision_metadata.get((agent_id, dimension), {}).get('confidence', 0.5)
                
                # Optionally adjust confidence based on risk
                adjust_confidence_with_risk = self.config.get('adjust_confidence_with_risk', False)
                if adjust_confidence_with_risk and agent_id in risk_results and dimension in risk_results[agent_id]:
                    total_risk = risk_results[agent_id][dimension]['total_risk']
                    capital_mean = risk_results[agent_id][dimension]['capital_mean']
                    if abs(capital_mean) > 0.001:
                        risk_ratio = total_risk / abs(capital_mean)
                        # Lower confidence when risk is high
                        confidence_penalty = self.config.get('risk_confidence_penalty', 0.2) * risk_ratio
                        adjusted_confidence = max(0.1, original_confidence - confidence_penalty)
                    else:
                        adjusted_confidence = original_confidence
                else:
                    adjusted_confidence = original_confidence
                
                risk_adjusted_decisions.append((agent_id, dimension, adjusted_amount, adjusted_confidence))
        
        return risk_adjusted_decisions
    
    def get_risk_metrics_summary(self, projected_capital_holdings, market_prices=None, current_capital_holdings=None):
        """
        Get a summary of risk metrics for analysis and debugging.
        
        Returns:
        - Dict with summary statistics about risk across agents and dimensions
        """
        risk_results = self.compute_risk(projected_capital_holdings, market_prices, current_capital_holdings)
        
        summary = {
            'total_agents': len(risk_results),
            'total_dimensions': sum(len(dims) for dims in risk_results.values()),
            'risk_statistics': {
                'monte_carlo_risk': {'min': float('inf'), 'max': -float('inf'), 'mean': 0, 'std': 0},
                'volatility_risk': {'min': float('inf'), 'max': -float('inf'), 'mean': 0, 'std': 0},
                'total_risk': {'min': float('inf'), 'max': -float('inf'), 'mean': 0, 'std': 0}
            },
            'high_risk_investments': [],  # Investments with risk above threshold
            'agent_risk_rankings': []     # Agents ranked by average total risk
        }
        
        all_mc_risks = []
        all_vol_risks = []
        all_total_risks = []
        agent_avg_risks = defaultdict(list)
        
        max_risk_threshold = self.config.get('max_risk_threshold', 0.8)
        
        for agent_id in risk_results:
            for dimension in risk_results[agent_id]:
                risk_data = risk_results[agent_id][dimension]
                mc_risk = risk_data['monte_carlo_risk']
                vol_risk = risk_data['volatility_risk']
                total_risk = risk_data['total_risk']
                capital_mean = risk_data['capital_mean']
                
                all_mc_risks.append(mc_risk)
                all_vol_risks.append(vol_risk)
                all_total_risks.append(total_risk)
                agent_avg_risks[agent_id].append(total_risk)
                
                # Check for high-risk investments
                if abs(capital_mean) > 0.001:
                    risk_ratio = total_risk / abs(capital_mean)
                else:
                    risk_ratio = total_risk
                    
                if risk_ratio > max_risk_threshold:
                    summary['high_risk_investments'].append({
                        'agent_id': agent_id,
                        'dimension': dimension,
                        'risk_ratio': risk_ratio,
                        'total_risk': total_risk,
                        'capital_mean': capital_mean
                    })
        
        # Compute statistics
        if all_mc_risks:
            summary['risk_statistics']['monte_carlo_risk'] = {
                'min': min(all_mc_risks), 'max': max(all_mc_risks),
                'mean': np.mean(all_mc_risks), 'std': np.std(all_mc_risks)
            }
        if all_vol_risks:
            summary['risk_statistics']['volatility_risk'] = {
                'min': min(all_vol_risks), 'max': max(all_vol_risks),
                'mean': np.mean(all_vol_risks), 'std': np.std(all_vol_risks)
            }
        if all_total_risks:
            summary['risk_statistics']['total_risk'] = {
                'min': min(all_total_risks), 'max': max(all_total_risks),
                'mean': np.mean(all_total_risks), 'std': np.std(all_total_risks)
            }
        
        # Rank agents by average risk
        for agent_id, risks in agent_avg_risks.items():
            avg_risk = np.mean(risks)
            summary['agent_risk_rankings'].append((agent_id, avg_risk))
        
        summary['agent_risk_rankings'].sort(key=lambda x: x[1], reverse=True)
        
        return summary
    # ------------------------------------------------------------------
    # End Prediction Volatility Tracking Methods
    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    # Monte Carlo Simulation Methods for Risk-Sensitive Investment
    # ------------------------------------------------------------------
    def _deterministic_rng(self, evaluation_round: Optional[int], dimension: str, tag: str = 'mc'):
        """Create a deterministic RNG for Monte Carlo sampling if mc_seed_base is set.
        Falls back to NumPy global RNG when seeding is disabled.
        """
        base = self.config.get('mc_seed_base', None) if hasattr(self, 'config') else None
        if base is None:
            return None
        key = f"{getattr(self, 'source_id', 'src')}|{evaluation_round}|{dimension}|{tag}|{base}"
        seed = int.from_bytes(hashlib.md5(key.encode()).digest()[:4], 'little')
        try:
            return np.random.default_rng(seed)
        except Exception:
            np.random.seed(seed)
            return None
    def sample_from_belief_states(self, agent_ids: List[int], dimensions: List[str], rng: Optional[Any] = None) -> Dict[int, Dict[str, float]]:
        """
        Sample scores from Beta distributions for given agents and dimensions.
        
        Parameters:
        - agent_ids: List of agent IDs to sample
        - dimensions: List of dimensions to sample
        
        Returns:
        - Dict mapping agent_id -> dimension -> sampled_score
        """
        sampled_scores = {}
        
        for agent_id in agent_ids:
            sampled_scores[agent_id] = {}
            for dimension in dimensions:
                belief_params = self.belief_state[agent_id].get(dimension)
                
                if belief_params is not None:
                    alpha, beta = belief_params
                    # Sample from Beta distribution
                    if rng is not None and hasattr(rng, 'beta'):
                        sampled_score = rng.beta(alpha, beta)
                    else:
                        sampled_score = np.random.beta(alpha, beta)
                    sampled_scores[agent_id][dimension] = sampled_score
                else:
                    # If no belief state, use neutral default
                    sampled_scores[agent_id][dimension] = 0.5
        
        return sampled_scores
    
    def monte_carlo_projected_capital_simulation(self, own_evaluations: Dict[int, Dict[str, Tuple[float, float]]], 
                                                market_prices: Dict[int, Dict[str, float]], 
                                                dimension: str, 
                                                num_trials: int = 1000,
                                                evaluation_round: Optional[int] = None,
                                                rng: Optional[Any] = None) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation to estimate distribution of projected capital shares.
        
        Parameters:
        - own_evaluations: Current mean evaluations {agent_id: {dim: (score, confidence)}}
        - market_prices: Current market prices {agent_id: {dim: price}}
        - dimension: Dimension to simulate
        - num_trials: Number of Monte Carlo trials
        
        Returns:
        - Dict with simulation results including mean, variance, and percentiles
        """
        # Get agent IDs that we have evaluations for
        agent_ids = list(own_evaluations.keys())
        
        # Store results from each trial
        trial_projected_capitals = defaultdict(list)  # agent_id -> [capital_values_across_trials]
        trial_projected_prices = defaultdict(list)    # agent_id -> [price_values_across_trials]
        steady_state_capital, steady_state_ratio, current_capital_shares = self._get_steady_state_capital(market_prices, dimension)
        capacity_flags = steady_state_ratio > 1.2
        
        # Use deterministic RNG if requested and not provided
        if rng is None:
            rng = self._deterministic_rng(evaluation_round, dimension, tag='mc_capital')

        for trial in range(num_trials):
            # Sample scores from belief states
            sampled_scores = self.sample_from_belief_states(agent_ids, [dimension], rng=rng)
            
            # Create evaluation format expected by _project_steady_state_prices
            trial_evaluations = {}
            for agent_id in agent_ids:
                if agent_id in own_evaluations and dimension in own_evaluations[agent_id]:
                    # Use sampled score with original confidence
                    original_confidence = own_evaluations[agent_id][dimension][1]
                    sampled_score = sampled_scores[agent_id][dimension]
                    trial_evaluations[agent_id] = {dimension: (sampled_score, original_confidence)}
            
            # Calculate projected capital shares for this trial
            trial_projected_prices_dim, trial_projected_capital_dim = self._project_steady_state_prices(
                    trial_evaluations, dimension, steady_state_capital, current_capital_shares)
                    
            # Store results for this trial
            for agent_id in trial_projected_capital_dim:
                trial_projected_capitals[agent_id].append(trial_projected_capital_dim[agent_id])
                trial_projected_prices[agent_id].append(trial_projected_prices_dim[agent_id])
                        
        projected_prices_dim = {}
        projected_capital_dim = {}
        for agent_id in agent_ids:
            if agent_id in trial_projected_prices:
                projected_capital_dim[agent_id] = (np.mean(trial_projected_capitals[agent_id]), np.std(trial_projected_capitals[agent_id]))
                projected_prices_dim[agent_id] = (np.mean(trial_projected_prices[agent_id]), np.std(trial_projected_prices[agent_id]))
        return projected_prices_dim, projected_capital_dim, capacity_flags
        
    
    def _monte_carlo_check_market_capacity(self, own_evaluations: Dict[int, Dict[str, Tuple[float, float]]], 
                                         market_prices: Dict[int, Dict[str, float]], 
                                         num_trials: int = None,
                                         evaluation_round: Optional[int] = None) -> Dict[str, Any]:
        """
        Monte Carlo version of check_market_capacity that incorporates uncertainty.
        
        Parameters:
        - own_evaluations: {agent_id: {dim: (score, confidence)}}
        - market_prices: {agent_id: {dim: price}}
        - num_trials: Number of Monte Carlo trials (defaults to config value)
        
        Returns:
        - Dict with simulation results for all dimensions
        """
        if num_trials is None:
            num_trials = self.config.get('monte_carlo_trials', 50)
        
        projected_capital_shares = {}
        projected_prices = {}
        capacity_flags = {}        
        for dimension in self.expertise_dimensions:
            # Run Monte Carlo simulation for this dimension
            rng = self._deterministic_rng(evaluation_round, dimension, tag='mc_capacity')
            projected_prices_dim, projected_capital_dim, steady_state_ratio = self.monte_carlo_projected_capital_simulation(
                own_evaluations, market_prices, dimension, num_trials, evaluation_round=evaluation_round, rng=rng
            )
            projected_capital_shares[dimension] = projected_capital_dim
            projected_prices[dimension] = projected_prices_dim
            capacity_flags[dimension] = steady_state_ratio > 1.2
        
        return projected_prices, projected_capital_shares, capacity_flags    
    # ------------------------------------------------------------------
    # End Monte Carlo Simulation Methods
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
            if True:#self.config.get('use_additional_context', True):
                contexts_for_pairs = {
                    (a, b): self._get_additional_context(a, b, evaluation_round)
                    for a, b in pairs_to_eval
                }
            else:
                contexts_for_pairs = {(a, b): "" for a, b in pairs_to_eval}
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

        # ------------------------------------------------------------
        # Phase 5 – Update prediction volatility tracking after belief update
        self.update_prediction_volatility_tracking(evaluation_round=getattr(self, 'last_evaluation_round', None))

        if detailed_analysis:
            return final_evals, comparison_log
        return final_evals

 
    def evaluate_agents_batch_direct(self, agent_ids: List[int], dimensions: Optional[List[str]] = None, 
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

        
        comparison_agents_per_target = self.config.get('comparison_agents_per_target', 3)

        valid_agents = [aid for aid in agent_ids if self._agent_has_comparable_data(aid)]
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
        investments = {aid: { dim : [] for dim in dimensions } for aid in valid_agents}
        comparison_log = []
        
        if pairs_to_eval:
            if self.config.get('use_additional_context', True):
                contexts_for_pairs = {
                    (a, b): self._get_additional_context_direct(a, b, evaluation_round)
                    for a, b in pairs_to_eval
                }
            else:
                contexts_for_pairs = {(a, b): "" for a, b in pairs_to_eval}
            with ThreadPoolExecutor(max_workers=min(8, len(pairs_to_eval))) as exe:
                fut_to_pair = {exe.submit(self._compare_pair_direct, a, b, dimensions, contexts_for_pairs.get((a,b), "")): (a, b) for (a, b) in pairs_to_eval}
                # if True:
                for fut in as_completed(fut_to_pair):
                    result = fut.result()
                    if result is None:
                        continue
                    # for a, b in pairs_to_eval:
                    #     result = self._compare_pair(a, b, dimensions, contexts_for_pairs.get((a,b), ""))
                    
                    # Unpack result, which may or may not include raw_reasoning
                    aid, oid, raw_results = result
                    for dim in dimensions:
                        # Accept both singular and plural key variants from prompts
                        inv_a = raw_results[dim].get('investments_A', raw_results[dim].get('investment_A'))
                        inv_b = raw_results[dim].get('investments_B', raw_results[dim].get('investment_B'))
                        investments[aid][dim].append(inv_a)
                        investments[oid][dim].append(inv_b)

                    # cache_key = (min(aid, oid), max(aid, oid), evaluation_round)
                    # self.comparison_results_cache[cache_key] = (derived_scores, confidences)
                    
                    # if isinstance(raw_results, dict) and 'Communication_Quality' in raw_results:
                    #     prompts_for_pairs[str((aid, oid))] = raw_results['Communication_Quality'].get('prompt', "Not provided by source.")

                    # # --- Persist evaluation memory ---
                    # if True:  # not analysis_mode and not detailed_analysis:
                    #     self._store_pair_evaluation(aid, oid, derived_scores, confidences, raw_results=raw_results, evaluation_round=evaluation_round)

                    # Log for detailed analysis if enabled
                    if detailed_analysis:
                        comparison_log.append({
                            'pair': (aid, oid),
                            'raw_results': raw_results
                        })
        for aid in investments:
            for dim in investments[aid]:
                investments[aid][dim] = np.mean(investments[aid][dim])
        if detailed_analysis:
            return investments, comparison_log
        return investments

    def decide_investments_direct(self, evaluation_round, use_comparative=True, analysis_mode=False, detailed_analysis=False):
        """Directly decide investments after evaluating agents in batch."""
        # Evaluate all agents in the market
        if self.market is None:
            print(f"WARNING ({self.source_id}): No market assigned, cannot decide investments.")
            return []
        
        investments = {}
        comparison_log = None

        if self.use_cached_evaluations and evaluation_round is not None:
            cached_investments, cached_comparison_log = self.get_cached_evaluations_direct(evaluation_round)
            if cached_investments:
                print(f"{self.source_type.upper()} ({self.source_id}): Using cached evaluations for round {evaluation_round}")
                investments = cached_investments
                if detailed_analysis:
                    comparison_log = cached_comparison_log
            else:
                if self.verbose:
                    print(f"{self.source_type.upper()} ({self.source_id}): No cached evaluations found for round {evaluation_round}, falling back to LLM evaluation")
                
        agent_ids = self.get_all_agent_ids()

        if not investments:
            investments = self.evaluate_agents_batch_direct(
                agent_ids, 
                evaluation_round=evaluation_round, 
                use_comparative=use_comparative,
                analysis_mode=analysis_mode,
                detailed_analysis=detailed_analysis
            )
            
            if detailed_analysis and isinstance(investments, tuple):
                investments, comparison_log = investments
            else:
                comparison_log = None
        
        investments_final = []
        total_portfolio_value_potential = self._calculate_total_portfolio_value_potential()
        min_trade_threshold = self.config.get('min_delta_value_trade_threshold', 5)
        for agent_id in investments:
            for dim in investments[agent_id]:
                if abs(investments[agent_id][dim]) > total_portfolio_value_potential[dim] * min_trade_threshold / 100:
                    investments_final.append((agent_id, dim, investments[agent_id][dim], 1.0))
        
        print(f"DEBUG: Final investments list length: {len(investments_final)}")
        if investments_final:
            print(f"DEBUG: {self.source_type.capitalize()} {self.source_id} prepared {len(investments_final)} cash-value based actions.")
            for i, (aid, dim, amount, conf) in enumerate(investments_final):
                print(f"DEBUG: Investment {i+1}: Agent {aid}, Dim {dim}, Amount {amount:.4f}, Confidence {conf:.4f}")
        else:
            print(f"DEBUG: {self.source_type.capitalize()} {self.source_id} found no cash-value actions to take.")
        
        if detailed_analysis:
            return investments_final, comparison_log
        return investments_final

    def get_cached_evaluations_direct(self, evaluation_round):

        investments = defaultdict(lambda : defaultdict(list))
        if evaluation_round not in self.cached_evaluations:
            return None, None
        for eval in self.cached_evaluations[evaluation_round]:
            aid, oid = eval['pair']
            raw_results = eval['raw_results']
            for dim in self.expertise_dimensions:
                inv_a = raw_results[dim].get('investments_A', raw_results[dim].get('investment_A'))
                inv_b = raw_results[dim].get('investments_B', raw_results[dim].get('investment_B'))
                investments[aid][dim].append(inv_a)
                investments[oid][dim].append(inv_b)
        for aid in investments:
            for dim in investments[aid]:
                investments[aid][dim] = np.mean(investments[aid][dim])
        return investments, self.cached_evaluations[evaluation_round]

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

    def load_cached_evaluations_direct(self, cached_data: Dict):
        """
        Load cached evaluations from a previous run for direct investment evaluations.
        
        Args:
            cached_data: Dictionary with structure {round: {agent_id: {dimension: investment_amount}}}
        """
        self.cached_evaluations = cached_data
        self.use_cached_evaluations = True
        print(f"{self.source_type.upper()} ({self.source_id}): Loaded cached direct evaluations for {len(cached_data)} rounds")

    def load_cached_evaluations(self, cached_data: Dict, comparison_log: Dict):
        """
        Load cached evaluations from a previous run.
        
        Args:
            cached_data: Dictionary with structure {round: {agent_id: {dimension: (score, confidence)}}}
        """
        # Normalize agent_id keys to ints where possible to avoid type-mismatch in downstream code
        def _normalize_round_map(round_map: Dict) -> Dict:
            normalized = {}
            for rnd, eval_map in round_map.items():
                # Keep round key as-is (expected to be int), but coerce agent_id keys to int if digit
                if isinstance(eval_map, dict):
                    new_eval_map = {}
                    for aid, dim_map in eval_map.items():
                        new_aid = int(aid) if isinstance(aid, str) and aid.isdigit() else aid
                        new_eval_map[new_aid] = dim_map
                    normalized[rnd] = new_eval_map
                else:
                    normalized[rnd] = eval_map
            return normalized

        def _normalize_comp_log(comp_log_by_round: Dict) -> Dict:
            def _norm_agent_key(k):
                return int(k) if isinstance(k, str) and k.isdigit() else k
            def _norm_agent_map(d):
                if not isinstance(d, dict):
                    return {}
                return {_norm_agent_key(k): v for k, v in d.items()}
            normalized = {}
            for rnd, comp_log in comp_log_by_round.items():
                # Two common shapes observed:
                # 1) list of entries with keys: pair, derived_scores, confidences, raw_results
                # 2) dict keyed by agent id (legacy)
                if isinstance(comp_log, list):
                    new_list = []
                    for entry in comp_log:
                        if not isinstance(entry, dict):
                            continue
                        pair = entry.get('pair')
                        if isinstance(pair, (list, tuple)) and len(pair) == 2:
                            a, b = pair
                            pair = [_norm_agent_key(a), _norm_agent_key(b)]
                        derived_scores = _norm_agent_map(entry.get('derived_scores', {}))
                        confidences = _norm_agent_map(entry.get('confidences', {}))
                        new_list.append({
                            'pair': pair,
                            'derived_scores': derived_scores,
                            'confidences': confidences,
                            'raw_results': entry.get('raw_results')
                        })
                    normalized[rnd] = new_list
                elif isinstance(comp_log, dict):
                    # Legacy dict shape; normalize only top-level keys
                    new_comp_log = {}
                    for aid, v in comp_log.items():
                        new_aid = _norm_agent_key(aid)
                        new_comp_log[new_aid] = v
                    normalized[rnd] = new_comp_log
                else:
                    normalized[rnd] = comp_log
            return normalized

        self.cached_evaluations = _normalize_round_map(cached_data)
        self.cached_comparison_log = _normalize_comp_log(comparison_log)
        self.use_cached_evaluations = True
        print(f"{self.source_type.upper()} ({self.source_id}): Loaded cached evaluations for {len(cached_data)} rounds")

    def enable_cached_evaluations(self, enable: bool = True):
        """Enable or disable using cached evaluations."""
        self.use_cached_evaluations = enable
        if enable:
            print(f"{self.source_type.upper()} ({self.source_id}): Enabled cached evaluation mode")
        else:
            print(f"{self.source_type.upper()} ({self.source_id}): Disabled cached evaluation mode")

    def get_cached_evaluation(self, evaluation_round: int) -> Optional[Dict]:
        """
        Get cached evaluation for a specific round.
        
        Returns:
            Dictionary with structure {agent_id: {dimension: (score, confidence)}} or None
        """
        if type(self.cached_evaluations) is not dict:
            self.get_cached_evaluations_direct(evaluation_round)
        eval_map = self.cached_evaluations.get(evaluation_round, {})
        comp = self.cached_comparison_log.get(evaluation_round, {})
        # Rehydrate pair evaluation memory and derived/confidence histories if comparison logs are present
        did_update_from_comp = False
        if True:
            if isinstance(comp, list) and comp:
                did_update_from_comp = True
                per_agent_dim_scores = {}
                per_agent_dim_confs = {}
                for entry in comp:
                    pair = entry.get('pair')
                    derived_scores = entry.get('derived_scores', {})
                    confidences = entry.get('confidences', {})
                    raw_results = entry.get('raw_results')
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        a, b = pair
                        # Persist evaluation memory
                        try:
                            self._store_pair_evaluation(a, b, derived_scores, confidences, raw_results=raw_results, evaluation_round=evaluation_round)
                        except Exception as _e2:
                            print(f"Warning: _store_pair_evaluation failed: {_e2}")
                        # Update legacy derived/confidence histories
                        dims_a = set(derived_scores.get(a, {}).keys())
                        dims_b = set(derived_scores.get(b, {}).keys())
                        dims = list(dims_a.union(dims_b))
                        try:
                            self._update_agent_derived_scores(a, derived_scores.get(a, {}), dims, confidences.get(a, {}))
                            self._update_agent_confidences(a, confidences.get(a, {}), dims)
                            self._update_agent_derived_scores(b, derived_scores.get(b, {}), dims, confidences.get(b, {}))
                            self._update_agent_confidences(b, confidences.get(b, {}), dims)
                        except Exception as _e3:
                            print(f"Warning: updating derived/confidences failed: {_e3}")
                        # Cache results for this round
                        try:
                            self.comparison_results_cache[(min(a, b), max(a, b), evaluation_round)] = (derived_scores, confidences)
                        except Exception as _e4:
                            print(f"Warning: updating comparison_results_cache failed: {_e4}")
                        # Collect evidence per agent/dimension
                        for aid in (a,b):
                            score_map = derived_scores.get(aid, {})
                            conf_map = confidences.get(aid, {})
                            if aid not in per_agent_dim_scores:
                                per_agent_dim_scores[aid] = {}
                                per_agent_dim_confs[aid] = {}
                            for dim, score in score_map.items():
                                if dim not in per_agent_dim_scores[aid]:
                                    per_agent_dim_scores[aid][dim] = []
                                    per_agent_dim_confs[aid][dim] = []
                                per_agent_dim_scores[aid][dim].append(score)
                                per_agent_dim_confs[aid][dim].append(conf_map.get(dim, 0.3))
                # Aggregate and update beliefs using combined evidence for this round
                for aid, dims in per_agent_dim_scores.items():
                    for dim, scores in dims.items():
                        confs = per_agent_dim_confs[aid][dim]
                        agg_score, agg_conf = self.combine_multiple_beliefs(scores, confs)
                        self._update_belief_state_bayesian(aid, dim, agg_score, agg_conf)
        if not did_update_from_comp and eval_map:
            for aid, dims in eval_map.items():
                if not isinstance(dims, dict):
                    continue
                for dim, val in dims.items():
                    try:
                        score, conf = val
                    except Exception:
                        continue
                    if not isinstance(score, (int, float)) or not isinstance(conf, (int, float)):
                        continue
                    conf = max(0.0, min(0.99, float(conf)))
                    self.belief_state[aid][dim] = self._score_and_confidence_to_beta_params(float(score), conf)
        # Update prediction volatility tracking for this round using the primed beliefs
        if did_update_from_comp or eval_map:
            self.update_prediction_volatility_tracking(evaluation_round=evaluation_round)
        return eval_map, comp


# Example usage of the comprehensive risk system:
"""
To use the risk system in your InformationSource subclass:
1. Configure risk parameters in your __init__ method:
   self.config.update({
       'volatility_risk_weight': 0.5,  # Weight for volatility risk component
       'risk_adjustment_method': 'multiplicative',  # How to adjust investments
       'risk_aversion_factor': 0.3,  # How much to penalize risky investments
       'volatility_normalization_method': 'historical_range',
       # ... other risk parameters as needed
   })
2. In your decide_investments method, use the risk-adjusted wrapper:
   def decide_investments(self, *args, **kwargs):
       # Your original investment logic
       def base_investment_logic(*args, **kwargs):
           # ... existing investment decision code ...
           return [(agent_id, dimension, amount, confidence), ...]
       
       # Apply risk adjustment
       return self.risk_adjusted_decide_investments(base_investment_logic, *args, **kwargs)
3. Or compute risk manually for more control:
   # Get projected capital holdings from Monte Carlo simulation
   _, projected_capitals, _ = self._monte_carlo_check_market_capacity(evaluations, market_prices)
   
   # Compute comprehensive risk
   risk_results = self.compute_risk(projected_capitals, market_prices, current_holdings)
   
   # Get risk summary for analysis
   risk_summary = self.get_risk_metrics_summary(projected_capitals, market_prices)
   
   # Apply risk adjustment to your base investment amounts
   adjusted_amounts = self.compute_risk_adjusted_investment_amounts(base_amounts, risk_results)
The risk system combines:
- Monte Carlo risk (standard deviation from uncertainty in evaluations)
- Volatility risk (historical prediction and market volatility scaled to capital)
- Configurable weighting: Total_Risk = sqrt((Risk_MC)^2 + (w * Risk_Vol)^2)
"""
