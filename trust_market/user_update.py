import time
from typing import Dict, Optional, List
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

class UserUpdate:
    def __init__(self, config, market):
        self.config = config
        self.market = market
        
        # Bayesian belief state tracking for user feedback
        # Structure: {agent_id: {dimension: (alpha, beta)}}
        self.user_belief_state = defaultdict(lambda: defaultdict(lambda: None))
        
        # History of user beliefs for confidence calculation
        # Structure: {agent_id: {dimension: [list of confidence values]}}
        self.user_belief_confidence_history = defaultdict(lambda: defaultdict(list))
        
        # Investment tracking for users (similar to information sources)
        # Structure: {agent_id: {dimension: amount_invested}}
        self.user_investments = defaultdict(lambda: defaultdict(float))
        
        # Configuration for Bayesian inference (similar to info sources)
        self.bayesian_config = {
            'confidence_to_kappa_scale_factor': config.get('user_confidence_to_kappa_scale_factor', 30.0),
            'decay_rate': config.get('user_decay_rate', 0.1),
            'likelihood_strength_factor': config.get('user_likelihood_strength_factor', 1.5),
            'investment_aggressiveness': config.get('user_investment_aggressiveness', 0.3),
            'min_investment_threshold': config.get('user_min_investment_threshold', 1.0),
            'max_investment_per_feedback': config.get('user_max_investment_per_feedback', 50.0),
            'portfolio_rebalance_threshold': config.get('user_portfolio_rebalance_threshold', 0.1),
        }
        self.include_source_capacity = config.get('include_source_capacity', False)

        initial_capacity = {dim: 10000.0 for dim in self.market.dimensions}
        self.user_source_id = 'user_feedback_bayesian'
        self.market.add_information_source(
            source_id=self.user_source_id,
            source_type='user_feedback_bayesian',
            initial_influence=initial_capacity,
            is_primary=True
        )
        
    def _score_and_confidence_to_beta_params(self, score: float, confidence: float) -> tuple:
        """Convert a (score, confidence) pair to Beta distribution parameters."""
        M = self.bayesian_config['confidence_to_kappa_scale_factor']
        kappa = 2.0 + confidence * M
        alpha = score * kappa
        beta = (1.0 - score) * kappa
        return alpha, beta
    
    def _beta_params_to_score_and_confidence(self, alpha: float, beta: float) -> tuple:
        """Convert Beta distribution parameters to (score, confidence) pair."""
        mean = alpha / (alpha + beta)
        kappa = alpha + beta
        M = self.bayesian_config['confidence_to_kappa_scale_factor']
        confidence = max(0.0, min(1.0, (kappa - 2.0) / M))
        return mean, confidence
    
    def _update_user_belief_state(self, agent_id: str, dimension: str, 
                                  new_score: float, new_confidence: float) -> tuple:
        """Update the belief state for an agent-dimension using Bayesian inference."""
        # Convert new evidence to Beta parameters
        alpha_new, beta_new = self._score_and_confidence_to_beta_params(new_score, new_confidence)
        
        # Get prior belief
        prior_belief = self.user_belief_state[agent_id][dimension]
        
        if prior_belief is None:
            # First evaluation - set posterior to new evidence
            alpha_posterior = alpha_new
            beta_posterior = beta_new
        else:
            # Get configuration parameters
            decay_rate = self.bayesian_config['decay_rate']
            likelihood_strength_factor = self.bayesian_config['likelihood_strength_factor']
            
            # Unpack prior parameters
            alpha_prior, beta_prior = prior_belief
            
            # Apply decay to prior
            kappa_prior = alpha_prior + beta_prior
            mu_prior = alpha_prior / kappa_prior if kappa_prior > 0 else 0.5
            kappa_decayed = kappa_prior * (1 - decay_rate)
            alpha_decayed = mu_prior * kappa_decayed
            beta_decayed = (1 - mu_prior) * kappa_decayed
            
            # Combine decayed prior with scaled new evidence
            alpha_posterior = alpha_decayed + (alpha_new * likelihood_strength_factor)
            beta_posterior = beta_decayed + (beta_new * likelihood_strength_factor)
        
        # Store updated belief
        self.user_belief_state[agent_id][dimension] = (alpha_posterior, beta_posterior)
        
        # Convert back to score and confidence
        updated_score, updated_confidence = self._beta_params_to_score_and_confidence(alpha_posterior, beta_posterior)
        
        # Update confidence history
        self.user_belief_confidence_history[agent_id][dimension].append(updated_confidence)
        
        # Keep only recent history
        max_history = 20
        if len(self.user_belief_confidence_history[agent_id][dimension]) > max_history:
            self.user_belief_confidence_history[agent_id][dimension] = \
                self.user_belief_confidence_history[agent_id][dimension][-max_history:]
        
        return updated_score, updated_confidence
    
    def _convert_comparative_feedback_to_scores(self, agent_a_id: str, agent_b_id: str, 
                                               winner_code: str, confidence: float, rating: float) -> tuple:
        """Convert comparative feedback to individual agent scores."""
        # Convert winner/rating to individual scores (0-1 range)
        if winner_code == 'A':
            # Agent A is better
            score_a = 0.5 + (confidence * 0.4)  # Score above neutral
            score_b = 0.5 - (confidence * 0.4)  # Score below neutral
        elif winner_code == 'B':
            # Agent B is better  
            score_a = 0.5 - (confidence * 0.4)
            score_b = 0.5 + (confidence * 0.4)
        else:  # Tie
            score_a = 0.5
            score_b = 0.5
            # confidence = 0.1  # Low confidence for ties
        
        # If we have a rating, use it more directly
        if rating != 0:
            # Rating is on scale like -5 to +5, normalize to 0-1
            rating_normalized = (rating + self.market.rating_scale) / (2 * self.market.rating_scale)
            score_a = rating_normalized
            score_b = 1.0 - rating_normalized
        
        # Clamp scores to valid range
        score_a = max(0.05, min(0.95, score_a))
        score_b = max(0.05, min(0.95, score_b))
        
        return score_a, score_b, confidence
    
    def _process_investments(self, investments: Dict):
        """Execute a user investment through the market mechanism."""
        investment_tuples = []
        for agent_id, dimensions in investments.items():
            for dimension, investment_amount in dimensions.items():
                if abs(investment_amount) < 0.01:
                    continue
                investment_tuples.append((agent_id, dimension, investment_amount, 0.8))
                self.market.cumulative_user_influence[dimension][agent_id] += abs(investment_amount)

        if len(investment_tuples) == 0:
            return
        
        
        # Process through market
        self.market.process_investments(self.user_source_id, investment_tuples)
        time_stamp = time.time()
        for agent_id, dimension, investment_amount, confidence in investment_tuples:
            
            params = self.market.agent_amm_params[agent_id][dimension]
            old_price = params['R'] / params['T'] if params['T'] > 0 else 0
            new_price = params['R'] / params['T'] if params['T'] > 0 else 0
            self.market.amm_transactions_log.append({
                'evaluation_round': self.market.evaluation_round, 'timestamp': time_stamp,
                'agent_id': agent_id, 'dimension': dimension, 'type': 'oracle_R_adjustment',
                'delta_R': investment_amount, 'new_R': params['R'], 'T_unchanged': params['T'],
                'old_price': old_price, 'new_price': new_price, 'source_id': 'oracle_system' # Or specific oracle
            })

    # Method for oracles to adjust R directly
    def oracle_adjust_reserve_direct(self, agent_id: str, dimension: str, delta_R: float):
        """
        Oracle directly adjusts the reserve for an agent in a dimension.
        This action *changes K* and is not a trade along the curve.
        T remains constant for this operation.
        """
        if agent_id not in self.market.agent_amm_params:
            # Initialize if new, or handle error
            # For simplicity, let's assume agent must exist / be initialized via another mechanism
            print(f"Warning: Agent {agent_id} not found in AMM params for oracle_adjust_reserve_direct.")
            # A common initialization: R=initial_R, T=initial_T (e.g., R=50, T=100 for P=0.5)
            # self.market.agent_amm_params[agent_id][dimension] = {'R': 50.0, 'T': 100.0, 'K': 5000.0, 'total_supply': 100.0} # Example
            # For now, let's assume it's initialized with some values
            if self.market.agent_amm_params[agent_id][dimension]['R'] == 0 and self.market.agent_amm_params[agent_id][dimension]['T'] == 0:
                self.market.agent_amm_params[agent_id][dimension]['R'] = self.config.get('initial_R_oracle', 10.0) # Small initial R
                self.market.agent_amm_params[agent_id][dimension]['T'] = self.config.get('initial_T_oracle', 20.0) # Ensures P=0.5
                self.market.agent_amm_params[agent_id][dimension]['K'] = self.market.agent_amm_params[agent_id][dimension]['R'] * self.market.agent_amm_params[agent_id][dimension]['T']
                # total_supply for AMM might track shares *held by investors* + shares *in treasury*.
                # Here, T is treasury shares. Let's assume total_supply is just T initially for AMM internal tracking.
                self.market.agent_amm_params[agent_id][dimension]['total_supply'] = self.market.agent_amm_params[agent_id][dimension]['T']


        params = self.market.agent_amm_params[agent_id][dimension]
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
        if self.market.primary_source_update_type == 'fix_K':
            params['T'] = params['K'] / params['R']
        elif self.market.primary_source_update_type == 'fix_T':
            params['K'] = params['R'] * params['T']

        # The price (trust score) changes
        old_price = old_R / params['T'] if params['T'] > 0 else 0
        new_price = params['R'] / params['T'] if params['T'] > 0 else 0
        self.market.agent_trust_scores[agent_id][dimension] = new_price

        self.market.agent_amm_params[agent_id][dimension] = params

        for source_id in self.market.allocated_influence:
            if self.market.allocated_influence[source_id][dimension] > 0 and (self.market.source_investments[source_id][agent_id][dimension] > 0):
                self.market.allocated_influence[source_id][dimension] += self.market.source_investments[source_id][agent_id][dimension]*(new_price - old_price)

        # Accumulate the absolute change for the regulator
        if 'user_feedback' in self.market.oracle_influence_mechanisms.values() or 'comparative_feedback' in self.market.oracle_influence_mechanisms.values(): # only if user feedback is an oracle
            self.market.cumulative_user_influence[dimension][agent_id] += abs(actual_delta_R)

        self.market.amm_transactions_log.append({
            'evaluation_round': self.market.evaluation_round, 'timestamp': time.time(),
            'agent_id': agent_id, 'dimension': dimension, 'type': 'oracle_R_adjustment',
            'delta_R': actual_delta_R, 'new_R': params['R'], 'T_unchanged': params['T'],
            'old_price': old_price, 'new_price': new_price, 'source_id': 'oracle_system' # Or specific oracle
        })
        # print(f"Oracle adjusted R for A{agent_id} Dim {dimension}: R {old_R:.2f}->{new_R:.2f}, P {old_price:.3f}->{new_price:.3f}")


    # In your `record_user_feedback` or `record_comparative_feedback`
    # This would replace the direct manipulation of `self.agent_trust_scores`

    def record_comparative_feedback(self, agent_a_id: str, agent_b_id: str,
                                    winners: Dict, raw_comparison_details: Optional[Dict] = None): # feedback_strength S
        """
        Records comparative user feedback and updates agent AMM params via oracle_adjust_reserve_direct.
        winners: Dict mapping dimension -> 'A', 'B', or 'Tie'
        feedback_strength: Overall strength of this feedback batch (e.g., based on num users)
        There are two ways to do this :
        Accumulate user ratings weighted by confidence or with bayesian averaging. And then use that to adjust R similar to how we do for regulator. 
        Or, use the rating directly to adjust R... We decide the R to adjust based on the rating, confidence and the current price. We don't just always adjust R by the same amount.
        But then this latter approach would also need the aggregate ratings for all other agents to know how to adjust R. thereby requiring the centralized book keeping. 
        """
        base_price_adj_factor = self.config.get('comparative_feedback_strength', 0.5) # Beta

        if raw_comparison_details:
            self.market.temporal_db['user_evaluations'].append({
                'evaluation_round': self.market.evaluation_round,
                'timestamp': time.time(),
                **raw_comparison_details
            })

        for dimension, winner_code_conf_raw in winners.items():
            if dimension not in self.market.dimensions: continue

            if isinstance(winner_code_conf_raw, tuple):
                winner_code, confidence = winner_code_conf_raw
                sign = 1 if winner_code == 'A' else -1 if winner_code == 'B' else 0
                rating = sign * confidence * self.market.rating_scale # Not provided in this tuple format
            elif isinstance(winner_code_conf_raw, dict):
                rating = winner_code_conf_raw.get('rating', 0)
                confidence = winner_code_conf_raw.get('confidence', 0) / self.market.rating_scale # Normalize confidence from 0-5 to 0-1
                # Winner is derived from rating for logging/simplicity, but rating is used for adjustment
                if rating > 0:
                    winner_code = 'A'
                elif rating < 0:
                    winner_code = 'B'
                else:
                    winner_code = 'Tie'
            else:
                # Fallback for unexpected format
                winner_code, confidence, rating = 'Tie', 0.5, 0
                
            params_A = self.market.agent_amm_params[agent_a_id][dimension]
            params_B = self.market.agent_amm_params[agent_b_id][dimension]

            # Ensure agents are initialized in AMM (important!)
            # This logic should ideally be in a separate `ensure_agent_in_amm` method
            for aid in [agent_a_id, agent_b_id]:
                if self.market.agent_amm_params[aid][dimension]['R'] == 0 and self.market.agent_amm_params[aid][dimension]['T'] == 0:
                    self.market.ensure_agent_dimension_initialized_in_amm(aid, dimension)

            if self.market.oracle_influence_mechanisms['user_feedback'] == 'adjust_P':
                P_A_current = params_A['R'] / params_A['T'] if params_A['T'] > 0 else 0.5 # Default if T is 0
                P_B_current = params_B['R'] / params_B['T'] if params_B['T'] > 0 else 0.5

                delta_P_A = 0.0
                delta_P_B = 0.0

                delta_P_A = rating * base_price_adj_factor # Absolute adjustment
                delta_P_B = -rating * base_price_adj_factor
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

                if self.market.primary_source_update_type == 'fix_K':
                    delta_R_A = params_A['R'] *(np.sqrt(P_A_target_clamped/P_A_current) - 1) if P_A_current > 0 else 0
                    delta_R_B = params_B['R'] *(np.sqrt(P_B_target_clamped/P_B_current) - 1) if P_B_current > 0 else 0
                elif self.market.primary_source_update_type == 'fix_T':
                    delta_R_A = actual_delta_P_A * params_A['T']
                    delta_R_B = actual_delta_P_B * params_B['T']

                if abs(actual_delta_P_A) > 0.0001: # Threshold for action
                    self.oracle_adjust_reserve_direct(agent_a_id, dimension, delta_R_A)
                if abs(actual_delta_P_B) > 0.0001: # Threshold for action
                    self.oracle_adjust_reserve_direct(agent_b_id, dimension, delta_R_B)
            elif self.market.oracle_influence_mechanisms['user_feedback'] == 'adjust_R':
                delta_R_A = rating/self.market.rating_scale * base_price_adj_factor # Use rating directly
                delta_R_B = -rating/self.market.rating_scale * base_price_adj_factor

                if abs(delta_R_A) > 0.0001: # Threshold for action
                    self.oracle_adjust_reserve_direct(agent_a_id, dimension, delta_R_A)
                if abs(delta_R_B) > 0.0001: # Threshold for action
                    self.oracle_adjust_reserve_direct(agent_b_id, dimension, delta_R_B)
    

    def record_comparative_feedback_with_bayesian_averaging(self, agent_a_id: str, agent_b_id: str,
                                                            winners: Dict, raw_comparison_details: Optional[Dict] = None):
        """
        Records comparative user feedback and updates agent trust using Bayesian inference
        and investment-based mechanisms rather than direct oracle adjustments.
        
        This method:
        1. Converts comparative feedback to individual agent scores
        2. Updates Bayesian belief states for both agents
        3. Makes investment decisions based on belief vs market price differences
        4. Executes investments through the market mechanism
        """
        # Store evaluation details
        if raw_comparison_details:
            self.market.temporal_db['user_evaluations'].append({
                'evaluation_round': self.market.evaluation_round,
                'timestamp': time.time(),
                **raw_comparison_details
            })

        # Process each dimension in the feedback
        for dimension, winner_code_conf_raw in winners.items():
            if dimension not in self.market.dimensions: 
                continue

            # Parse the feedback format
            if isinstance(winner_code_conf_raw, tuple):
                winner_code, confidence = winner_code_conf_raw
                rating = 0  # Not provided in this tuple format
            elif isinstance(winner_code_conf_raw, dict):
                rating = winner_code_conf_raw.get('rating', 0)
                confidence = winner_code_conf_raw.get('confidence', 0) / self.market.rating_scale  # Normalize confidence from 0-5 to 0-1
                # Winner is derived from rating for logging/simplicity, but rating is used for adjustment
                if rating > 0:
                    winner_code = 'A'
                elif rating < 0:
                    winner_code = 'B'
                else:
                    winner_code = 'Tie'
            else:
                # Fallback for unexpected format
                winner_code, confidence, rating = 'Tie', 0.5, 0

            # Ensure agents are initialized in market
            for aid in [agent_a_id, agent_b_id]:
                if aid not in self.market.agent_amm_params or dimension not in self.market.agent_amm_params[aid]:
                    self.market.ensure_agent_dimension_initialized_in_amm(aid, dimension)

            # Convert comparative feedback to individual agent scores
            score_a, score_b, feedback_confidence = self._convert_comparative_feedback_to_scores(
                agent_a_id, agent_b_id, winner_code, confidence, rating
            )

            # Update Bayesian belief states for both agents
            belief_score_a, belief_confidence_a = self._update_user_belief_state(
                agent_a_id, dimension, score_a, feedback_confidence
            )
            belief_score_b, belief_confidence_b = self._update_user_belief_state(
                agent_b_id, dimension, score_b, feedback_confidence
            )
    
    def get_steady_state_capital_holdings(self, market_capital_holdings: Dict, dimensions: List[str]):
        """
        Project what market prices will be at steady state based on:
        1. Expected total capital deployment
        2. Quality-based distribution of that capital
        """
        steady_state_capital = {}
        current_capital_shares = {}
        for dimension in dimensions:
            total_potential_capital = 0
            if self.include_source_capacity:
                for source_id, _ in self.market.source_available_capacity.items():
                    # Each source's total capacity across all dimensions
                    source_capacity = self.market.source_available_capacity[source_id].get(dimension, 0)
                    total_potential_capital += source_capacity

            # Add expected growth factor (new investors, increased allocations)
            # growth_factor = self.config.get('market_growth_factor', 1.5)
            # steady_state_capital = total_potential_capital #* growth_factor
            
            # Step 2: Current total capital in market
            current_total_market_capital = 0
            for agent_id in self.market.agent_amm_params:
                if dimension in self.market.agent_amm_params[agent_id]:
                    # Total capital locked in AMM = R (reserves)
                    current_total_market_capital += self.market.agent_amm_params[agent_id][dimension]['R']
            
            # 2a. Compute steady state capital as the sum of current market capital and expected new capital. And capital ratio.
            # This assumes the market will stabilize at a point
            steady_state_capital[dimension] = current_total_market_capital + total_potential_capital #* growth_factor

            # 2b. Calculate current capital shares from the market.
            current_agent_capital = market_capital_holdings[dimension]
            current_total_market_capital_for_shares = sum(current_agent_capital.values())

            if current_total_market_capital_for_shares > 1e-9:
                current_capital_shares[dimension] = {
                    aid: cap / current_total_market_capital_for_shares
                    for aid, cap in current_agent_capital.items()
                }
        return steady_state_capital, current_capital_shares
    
    def project_capital_holding(self, own_evaluations: Dict, dimensions: List[str], steady_state_capital: Dict, current_capital_shares: Dict):
        """
        Project the capital holding of all the users combined based on their belief state and the steady state capital.
        """
        projected_capital_holdings = defaultdict(lambda: defaultdict(float))
        for dimension in dimensions:
            # 1. Calculate capital shares based purely on this source's evaluations.
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
            
            expected_capital_shares = {}
            if total_quality_powered > 1e-9:
                expected_capital_shares = {
                    aid: qp / total_quality_powered
                    for aid, qp in quality_powered.items()
                }

            # 2. Project steady-state capital            
            for agent_id in expected_capital_shares:
                # Expected capital for this agent at steady state
                projected_capital_holdings[agent_id][dimension] = steady_state_capital[dimension] * expected_capital_shares[agent_id]

        return projected_capital_holdings
    
    def monte_carlo_project_capital(self, belief_state: Dict, market_prices: Dict, market_capital_holdings: Dict):
        """
        Monte Carlo projection of capital holdings based on belief state.
        """
        # Get the number of simulations to run
        num_simulations = self.config.get('num_simulations', 100)
        # Get the number of dimensions
        num_dimensions = len(list(belief_state.values())[0])
        # Get the number of agents
        num_agents = len(belief_state)
        steady_state_capital, current_capital_shares = self.get_steady_state_capital_holdings(market_capital_holdings, dimensions=list(belief_state.values())[0].keys())

        projected_capital_holdings = defaultdict(lambda: defaultdict(float))
        # Run the multithreaded Monte Carlo simulations
        # with ProcessPoolExecutor() as executor:
        #     futures = [executor.submit(self.project_capital_holding, belief_state, market_capital_holdings, steady_state_capital, current_capital_shares) for _ in range(num_simulations)]
            # for future in futures:
            #     result = future.result()
            #     for agent_id, dimension, value in result.items():
            #         projected_capital_holdings[agent_id][dimension] += value
        belief_scores = {agent_id: {dimension: self._beta_params_to_score_and_confidence(*belief_state[agent_id][dimension]) for dimension in list(belief_state.values())[0].keys()} for agent_id in belief_state.keys()}
        projected_capital_holdings_list = defaultdict(lambda: defaultdict(list))
        # Run the Monte Carlo simulations
        for _ in range(num_simulations):
            projected_capital_holdings = self.project_capital_holding(belief_scores, market_capital_holdings, steady_state_capital, current_capital_shares)
            for agent_id in projected_capital_holdings:
                for dimension in projected_capital_holdings[agent_id]:
                    projected_capital_holdings_list[agent_id][dimension].append(projected_capital_holdings[agent_id][dimension])
        projected_capital_holdings = defaultdict(lambda: defaultdict(tuple))
        for agent_id in projected_capital_holdings_list:
            for dimension in projected_capital_holdings_list[agent_id]:
                projected_capital_holdings[agent_id][dimension] = (np.mean(projected_capital_holdings_list[agent_id][dimension]), np.std(projected_capital_holdings_list[agent_id][dimension]))
        return projected_capital_holdings

    def decide_investments(self, evaluation_round=None, use_comparative=True, analysis_mode=False, detailed_analysis=False):
        """
        The main decision-making loop for the user_update.
        1. Evaluates all agents to get up-to-date scores.
        """
        desirability_method = self.config.get('desirability_method', 'percentage_change')
        max_investment_per_round_per_dimension = self.config.get('max_investment_per_round_per_dimension', 0.1)

        # Project belief scores to market prices
        market_prices, market_capital_holdings = self.market.get_market_prices(candidate_agent_ids=self.user_belief_state.keys(), dimensions=list(self.user_belief_state.values())[0].keys())
        projected_capital_holdings = self.monte_carlo_project_capital(self.user_belief_state, market_prices, market_capital_holdings)
        risk = {agent_id: {
            dimension: projected_capital_holdings[agent_id][dimension][1]
            for dimension in list(self.user_belief_state.values())[0].keys()
        } for agent_id in self.user_belief_state.keys()}
        
        # Calculate investment decisions based on belief vs market price differences
        deltas = {agent_id: {
            dimension: (projected_capital_holdings[agent_id][dimension][0] - market_capital_holdings[agent_id][dimension])
            for dimension in list(self.user_belief_state.values())[0].keys()
        } for agent_id in self.user_belief_state.keys()}

        risk_normalized = {agent_id: {
            dimension: risk[agent_id][dimension]/market_capital_holdings[agent_id][dimension]
            for dimension in list(self.user_belief_state.values())[0].keys()
        } for agent_id in self.user_belief_state.keys()}

        investments = {agent_id: {
            dimension: min(max_investment_per_round_per_dimension, max(deltas[agent_id][dimension], -max_investment_per_round_per_dimension))/ ( 1 + risk_normalized[agent_id][dimension])
            for dimension in list(self.user_belief_state.values())[0].keys()
        } for agent_id in self.user_belief_state.keys()}

        self._process_investments(investments)
    
    def get_user_beliefs_summary(self) -> Dict:
        """Get a summary of current user belief states for analysis."""
        summary = {}
        for agent_id, dimensions in self.user_belief_state.items():
            summary[agent_id] = {}
            for dimension, belief in dimensions.items():
                if belief is not None:
                    score, confidence = self._beta_params_to_score_and_confidence(*belief)
                    summary[agent_id][dimension] = {
                        'belief_score': score,
                        'belief_confidence': confidence,
                        'total_investment': self.user_investments[agent_id][dimension]
                    }
        return summary