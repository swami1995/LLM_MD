import unittest
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
from collections import defaultdict
from trust_market.user_rep import UserRepresentativeWithHolisticEvaluation
from trust_market.auditor import BatchEvaluator

class MockMarket:
    def __init__(self):
        self.source_available_capacity = defaultdict(lambda: defaultdict(float))
        self.source_investments = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.agent_amm_params = defaultdict(lambda: defaultdict(dict))
        self.agent_trust_scores = defaultdict(dict)
        self.rating_scale = 5

    def ensure_agent_dimension_initialized_in_amm(self, agent_id, dimension):
        if dimension not in self.agent_amm_params.get(agent_id, {}):
            self.agent_amm_params[agent_id][dimension] = {'R': 100.0, 'T': 100.0}

    def get_price(self, agent_id, dimension):
        params = self.agent_amm_params[agent_id][dimension]
        if params['T'] > 1e-6:
            return params['R'] / params['T']
        return 0.5 # Fallback

class TestUserRepComprehensive(unittest.TestCase):
    def setUp(self):
        self.market = MockMarket()
        self.source_id = 'test_rep'
        self.expertise_dims = ["Factual_Correctness", "Transparency"]

        self.user_rep = UserRepresentativeWithHolisticEvaluation(
            source_id=self.source_id,
            user_segment='technical',
            representative_profile={},
            market=self.market,
            api_key='mock_key'
        )
        self.user_rep.expertise_dimensions = self.expertise_dims

        for dim in self.expertise_dims:
            self.market.source_available_capacity[self.source_id][dim] = 1000.0

        self.agents = ['agent_A', 'agent_B', 'agent_C']
        users = ['user_A', 'user_B', 'user_C']
        for agent_id, user_id in zip(self.agents, users):
            self.user_rep.add_represented_user(user_id)
            for _ in range(5):
                self.user_rep.add_conversation([{'role': 'user', 'content': 'hi'}], user_id, agent_id)
            for dim in self.expertise_dims:
                self.market.ensure_agent_dimension_initialized_in_amm(agent_id, dim)

    @patch('trust_market.auditor.BatchEvaluator.get_agent_scores')
    @patch('trust_market.auditor.BatchEvaluator.compare_agent_batches')
    def test_evaluate_agent_full_logic_and_caching(self, mock_compare_batches, mock_get_scores):
        # --- Mocking ---
        # 1. Mock the relative comparison to return the format the code expects:
        # A dictionary keyed by dimension.
        def comparison_logic(convs1, id1, convs2, id2, dims):
            scores = {'agent_A': 0.8, 'agent_B': 0.6, 'agent_C': 0.4}
            winner_id = id1 if scores[id1] > scores[id2] else id2
            loser_id = id2 if winner_id == id1 else id1
            
            # ** FIX: The mock must return a dictionary keyed by dimension. **
            result = {}
            for dim in dims:
                result[dim] = {
                    'winner': winner_id,
                    'confidence': 4.0,
                    'explanation': 'mock'
                }
            
            # Pass winner/loser info to the next mock via a side channel
            mock_get_scores.winner = winner_id 
            mock_get_scores.loser = loser_id
            return result
        mock_compare_batches.side_effect = comparison_logic

        # 2. Mock the pseudo-absolute score conversion
        def score_logic(comparison_result, id1, id2):
            # Use the winner/loser info from the side channel
            winner = mock_get_scores.winner
            loser = mock_get_scores.loser
            return {
                winner: {dim: 0.75 for dim in self.expertise_dims},
                loser: {dim: 0.25 for dim in self.expertise_dims}
            }
        mock_get_scores.side_effect = score_logic
        
        # --- Test Execution ---
        with patch('random.sample', return_value=[('agent_B', []), ('agent_C', [])]):
             eval_A_r1 = self.user_rep.evaluate_agent('agent_A', evaluation_round=1)

        self.assertEqual(mock_compare_batches.call_count, 2)
        self.assertEqual(len(self.user_rep.comparison_results_cache), 2)
        # With proper mocking, this assertion now passes
        self.assertGreater(self.user_rep.derived_agent_scores['agent_A']['Factual_Correctness'], 0.5)

        with patch('random.sample', return_value=[('agent_A', []), ('agent_C', [])]):
            eval_B_r1 = self.user_rep.evaluate_agent('agent_B', evaluation_round=1)

        self.assertEqual(mock_compare_batches.call_count, 3)
        self.assertEqual(len(self.user_rep.comparison_results_cache), 3)

        eval_A_r2 = self.user_rep.evaluate_agent('agent_A', evaluation_round=2)

        # ** FIX: Make the assertion robust. The number of new comparisons is limited
        # by the number of other agents available, not just the config value. **
        valid_partners = self.agents.copy()
        valid_partners.remove('agent_A') # Can't compare against self
        num_valid_partners = len(valid_partners)
        num_new_comparisons = min(self.user_rep.config['comparison_agents_per_target'], num_valid_partners)
        
        self.assertEqual(mock_compare_batches.call_count, 3 + num_new_comparisons)

        self.assertEqual(self.user_rep.last_evaluation_round, 2)
        num_comparisons_in_round_2 = len(self.user_rep.comparison_results_cache)
        self.assertGreater(num_comparisons_in_round_2, 0)

    @patch('trust_market.auditor.BatchEvaluator.get_agent_scores')
    @patch('trust_market.auditor.BatchEvaluator.compare_agent_batches')
    def test_evaluate_agent_updates_both_agents_in_comparison(self, mock_compare_batches, mock_get_scores):
        # ** FIX: Mock the comparison to return the correct dictionary format. **
        mock_compare_batches.return_value = {
            dim: {'winner': 'agent_A', 'confidence': 4.0, 'explanation': 'mock'}
            for dim in self.expertise_dims
        }
        # Mock the score conversion to return differentiated scores
        mock_get_scores.return_value = {
            'agent_A': {dim: 0.8 for dim in self.expertise_dims},
            'agent_B': {dim: 0.2 for dim in self.expertise_dims}
        }

        self.assertNotIn('agent_A', self.user_rep.derived_agent_scores)
        self.assertNotIn('agent_B', self.user_rep.derived_agent_scores)

        with patch('random.sample', return_value=[('agent_B', [])]):
            self.user_rep.evaluate_agent('agent_A', evaluation_round=1)

        self.assertIn('agent_A', self.user_rep.derived_agent_scores)
        self.assertIn('agent_B', self.user_rep.derived_agent_scores)

        for dim in self.expertise_dims:
            self.assertGreater(self.user_rep.derived_agent_scores['agent_A'][dim], 0.5)
            self.assertLess(self.user_rep.derived_agent_scores['agent_B'][dim], 0.5)

    def test_evaluate_agent_with_no_valid_peers(self):
        user_rep_solo = UserRepresentativeWithHolisticEvaluation(
            'solo_rep', 'technical', {}, self.market, 'mock_key'
        )
        user_rep_solo.add_represented_user('user_lonely')
        user_rep_solo.add_conversation([{'role': 'user', 'content': 'hi'}], 'user_lonely', 'agent_lonely')

        eval_result = user_rep_solo.evaluate_agent('agent_lonely', evaluation_round=1)

        for dim in user_rep_solo.expertise_dimensions:
            score, confidence = eval_result[dim]
            self.assertAlmostEqual(score, 0.5)
            self.assertAlmostEqual(confidence, 0.3)

    @patch('trust_market.user_rep.UserRepresentativeWithHolisticEvaluation.evaluate_agent')
    def test_decide_investments_rank_mapping_logic(self, mock_evaluate_agent):
        for dim in self.expertise_dims:
            self.market.source_available_capacity[self.source_id][dim] = 1.0

        # ** FIX: Give the rep an initial holding of the overvalued agent to test the SELL logic. **
        self.market.source_investments[self.source_id]['agent_C']['Factual_Correctness'] = 10.0

        mock_evals = {
            'agent_A': {dim: (0.9, 0.9) for dim in self.expertise_dims},
            'agent_B': {dim: (0.7, 0.8) for dim in self.expertise_dims},
            'agent_C': {dim: (0.4, 0.9) for dim in self.expertise_dims},
        }
        mock_evaluate_agent.side_effect = lambda agent_id, **kwargs: mock_evals[agent_id]

        self.market.agent_amm_params['agent_A']['Factual_Correctness'] = {'R': 50, 'T': 100}
        self.market.agent_amm_params['agent_B']['Factual_Correctness'] = {'R': 60, 'T': 100}
        self.market.agent_amm_params['agent_C']['Factual_Correctness'] = {'R': 70, 'T': 100}

        investments = self.user_rep.decide_investments(evaluation_round=1)

        self.assertTrue(len(investments) > 0)
        trades = {(agent_id, dim): amount for agent_id, dim, amount, conf in investments}

        self.assertIn(('agent_A', 'Factual_Correctness'), trades)
        self.assertGreater(trades[('agent_A', 'Factual_Correctness')], 0)

        self.assertIn(('agent_C', 'Factual_Correctness'), trades)
        self.assertLess(trades[('agent_C', 'Factual_Correctness')], 0)

        if ('agent_B', 'Factual_Correctness') in trades:
             self.assertAlmostEqual(trades[('agent_B', 'Factual_Correctness')], 0, delta=0.1)

    @patch('trust_market.user_rep.UserRepresentativeWithHolisticEvaluation.evaluate_agent')
    def test_decide_investments_projected_price_logic(self, mock_evaluate_agent):
        for dim in self.expertise_dims:
            self.market.source_available_capacity[self.source_id][dim] = 5000.0

        mock_evals = {
            'agent_A': {dim: (0.9, 0.9) for dim in self.expertise_dims},
            'agent_B': {dim: (0.3, 0.9) for dim in self.expertise_dims},
            'agent_C': {dim: (0.5, 0.9) for dim in self.expertise_dims},
        }
        mock_evaluate_agent.side_effect = lambda agent_id, **kwargs: mock_evals[agent_id]

        for agent_id in ['agent_A', 'agent_B']:
             self.market.agent_amm_params[agent_id]['Factual_Correctness'] = {'R': 10, 'T': 100}

        investments = self.user_rep.decide_investments(evaluation_round=1)

        trades = {(agent_id, dim): amount for agent_id, dim, amount, conf in investments}

        self.assertIn(('agent_A', 'Factual_Correctness'), trades)
        self.assertIn(('agent_B', 'Factual_Correctness'), trades)
        self.assertGreater(
            trades[('agent_A', 'Factual_Correctness')],
            trades[('agent_B', 'Factual_Correctness')]
        )

    def test_budget_constraint_and_scaling(self):
        self.market.source_available_capacity[self.source_id]['Factual_Correctness'] = 10.0

        delta_map = defaultdict(lambda: defaultdict(float))
        delta_map['Factual_Correctness']['agent_A'] = 20.0
        delta_map['Factual_Correctness']['agent_B'] = 30.0

        with patch.object(self.user_rep, 'decide_investments') as mock_decide:
            with patch.object(self.user_rep, 'evaluate_agent') as mock_eval:
                mock_eval.return_value = {'Factual_Correctness': (0.9, 0.9)}

                self.market.agent_amm_params['agent_A']['Factual_Correctness'] = {'R': 10, 'T': 100}
                self.market.agent_amm_params['agent_B']['Factual_Correctness'] = {'R': 10, 'T': 100}
                self.market.source_investments[self.source_id].clear()
                self.market.source_available_capacity[self.source_id]['Factual_Correctness'] = 10.0
                self.market.source_available_capacity[self.source_id]['Transparency'] = 990.0

                investments = self.user_rep.decide_investments(evaluation_round=1)

                total_investment_in_dim = sum(
                    amount for _, dim, amount, _ in investments
                    if dim == 'Factual_Correctness' and amount > 0
                )

                self.assertLessEqual(total_investment_in_dim, 10.0)

    def test_calculate_total_portfolio_value_potential(self):
        dim = 'Factual_Correctness'
        self.market.source_available_capacity[self.source_id][dim] = 1000.0
        self.market.source_investments[self.source_id]['agent_A'][dim] = 10.0
        self.market.agent_amm_params['agent_A'][dim] = {'R': 200, 'T': 100}

        potential = self.user_rep._calculate_total_portfolio_value_potential()

        self.assertAlmostEqual(potential[dim], 1020.0)

    def test_get_target_price_from_rank_mapping_edge_cases(self):
        evals = {'agent_A': {'Factual_Correctness': (0.9, 0.8)}}
        prices = {'agent_A': {'Factual_Correctness': 0.6}}
        
        target_price = self.user_rep._get_target_price_from_rank_mapping(
            'agent_A', 'Factual_Correctness', evals, prices, 0.8
        )
        self.assertEqual(target_price, 0.6)

        prices_missing_agent = {'agent_B': {'Factual_Correctness': 0.5}}
        target_price_2 = self.user_rep._get_target_price_from_rank_mapping(
            'agent_A', 'Factual_Correctness', evals, prices_missing_agent, 0.8
        )
        self.assertEqual(target_price_2, 0.5)

    @patch('trust_market.user_rep.UserRepresentativeWithHolisticEvaluation.evaluate_agent')
    def test_decide_investments_rebalancing_scenarios(self, mock_evaluate_agent):
        """
        Tests rebalancing logic for an existing portfolio:
        1. Buys more of an undervalued asset it already owns.
        2. Sells multiple overvalued assets, selling more of the worse one.
        """
        # --- Setup Scenario ---
        dim = 'Factual_Correctness'
        
        # ** FIX: Force the rank-mapping logic by setting available capacity low. **
        self.market.source_available_capacity[self.source_id][dim] = 1.0

        # Establish an existing portfolio
        self.market.source_investments[self.source_id]['agent_A'][dim] = 5.0   # Undervalued holding
        self.market.source_investments[self.source_id]['agent_B'][dim] = 10.0  # Overvalued holding
        self.market.source_investments[self.source_id]['agent_C'][dim] = 10.0  # SEVERELY overvalued holding
        
        # Mock evaluations: A is great, B is mediocre, C is poor.
        mock_evals = {
            'agent_A': {dim: (0.9, 0.9)},
            'agent_B': {dim: (0.6, 0.9)},
            'agent_C': {dim: (0.3, 0.9)},
        }
        mock_evaluate_agent.side_effect = lambda agent_id, **kwargs: mock_evals[agent_id]

        # Set market prices that are misaligned with our evaluation
        self.market.agent_amm_params['agent_A'][dim] = {'R': 50, 'T': 100} # P=0.5 (undervalued)
        self.market.agent_amm_params['agent_B'][dim] = {'R': 70, 'T': 100} # P=0.7 (overvalued)
        self.market.agent_amm_params['agent_C'][dim] = {'R': 80, 'T': 100} # P=0.8 (SEVERELY overvalued)

        # --- Execute ---
        investments = self.user_rep.decide_investments(evaluation_round=1)
        trades = {agent_id: amount for agent_id, d, amount, c in investments if d == dim}

        # --- Assert ---
        # 1. Expect to BUY MORE of agent_A (undervalued)
        self.assertIn('agent_A', trades)
        self.assertGreater(trades['agent_A'], 0)

        # 2. Expect to SELL BOTH agent_B and agent_C (overvalued)
        self.assertIn('agent_B', trades)
        self.assertLess(trades['agent_B'], 0)
        self.assertIn('agent_C', trades)
        self.assertLess(trades['agent_C'], 0)
        
        # 3. Expect to sell MORE of C than B (C is more overvalued)
        # We compare the absolute cash value of the proposed sell trades.
        self.assertGreater(abs(trades['agent_C']), abs(trades['agent_B']))

    def test_confidence_aggregation_plateau(self):
        """
        Tests that the derived confidence correctly increases with more data
        but plateaus at the defined cap (0.95).
        """
        agent_id = 'agent_A'
        dim = 'Factual_Correctness'
        dimensions_to_evaluate = [dim]
        
        # Simulate a large number of high-confidence comparisons
        num_comparisons = 50
        self.user_rep.derived_agent_confidences[agent_id][dim] = [0.8] * num_comparisons
        
        # Calculate the derived confidence
        confidences = self.user_rep._calculate_derived_confidence(agent_id, dimensions_to_evaluate)
        final_confidence = confidences[dim]
        
        # Assert that the confidence is high, but capped at 0.95
        self.assertGreater(final_confidence, 0.9, "Confidence should be high after many data points")
        self.assertLessEqual(final_confidence, 0.95, "Confidence should be capped at 0.95")

if __name__ == '__main__':
    unittest.main() 