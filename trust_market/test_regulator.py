import unittest
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict

from trust_market.regulator import Regulator
from trust_market.trust_market import TrustMarket

class TestRegulator(unittest.TestCase):

    def setUp(self):
        """Set up a fresh regulator and mock market for each test."""
        # Mock genai, as BatchEvaluator might use it.
        self.mock_genai_patch = patch('google.genai.Client')
        mock_genai_client_constructor = self.mock_genai_patch.start()
        mock_instance = mock_genai_client_constructor.return_value
        mock_response = MagicMock()
        mock_response.text = "{}"
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = 'STOP'
        mock_candidate.content.parts = [MagicMock()]
        mock_candidate.content.parts[0].text = "{}"
        mock_response.candidates = [mock_candidate]
        mock_instance.models.generate_content.return_value = mock_response

        # Mock Market
        self.market = Mock(spec=TrustMarket)
        self.market.source_investments = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.market.source_available_capacity = defaultdict(lambda: defaultdict(float))
        self.market.agent_amm_params = defaultdict(lambda: defaultdict(dict))
        self.market.agent_trust_scores = defaultdict(lambda: defaultdict(float))
        # Add the ensure_agent_dimension_initialized_in_amm method to the mock
        self.market.ensure_agent_dimension_initialized_in_amm = MagicMock()


        # Regulator Instance
        self.regulator = Regulator(
            source_id='test_regulator',
            market=self.market,
            api_key='fake_key',
            verbose=False
        )

        # Add agent profiles and conversations
        self.profiles = {
            1: {'primary_goals': [['efficiency', 0.8]], 'knowledge_breadth': 0.7},
            2: {'primary_goals': [['safety', 0.9]], 'knowledge_breadth': 0.8},
            3: {'primary_goals': [['sales', 0.9]], 'knowledge_breadth': 0.6},
        }
        self.conversations = {
            1: [[{'role': 'user', 'content': 'history for agent 1'}] for _ in range(5)],
            2: [[{'role': 'user', 'content': 'history for agent 2'}] for _ in range(5)],
            3: [[{'role': 'user', 'content': 'history for agent 3'}] for _ in range(5)],
        }

        for agent_id, profile in self.profiles.items():
            self.regulator.add_agent_profile(agent_id, profile)
            # Use the add_conversation method correctly
            for conv in self.conversations[agent_id]:
                self.regulator.add_conversation(conv, user_id='test_user', agent_id=agent_id)
            
            for dim in self.regulator.expertise_dimensions:
                # Mock the behavior of ensure_agent_dimension_initialized_in_amm if needed,
                # but setting the params directly is often simpler for tests.
                self.market.agent_amm_params[agent_id][dim] = {'R': 50.0, 'T': 100.0}
                self.market.agent_trust_scores[agent_id][dim] = 0.5
        
        self.market.source_available_capacity[self.regulator.source_id] = {
            dim: 1000.0 for dim in self.regulator.expertise_dimensions
        }

    def tearDown(self):
        """Stop the patch after each test."""
        self.mock_genai_patch.stop()

    def test_regulator_initialization(self):
        """Test if the regulator and its components are initialized correctly."""
        self.assertEqual(self.regulator.source_id, 'test_regulator')
        self.assertIsNotNone(self.regulator.market)
        self.assertIsNotNone(self.regulator.batch_evaluator)
        self.assertEqual(len(self.regulator.agent_profiles), 3)
        self.assertEqual(len(self.regulator.agent_conversations[1]), 5)

    def test_evaluate_agent_comparative_audit_no_competitors(self):
        """Test comparative audit when there are no other valid agents."""
        # Isolate agent 1
        self.regulator.agent_profiles = {1: self.profiles[1]}
        self.regulator.agent_conversations = {1: self.conversations[1]}
        
        # Manually clear other agents from the regulator's observed list if necessary
        # This depends on how observed_agents() is implemented. Assuming it's based on agent_profiles.
        
        results = self.regulator.evaluate_agent(agent_id=1, evaluation_round=1, use_comparative=True)
        
        for dim in self.regulator.expertise_dimensions:
            self.assertIn(dim, results)
            score, conf = results[dim]
            # When no comparison happens, the score should be the default base score (0.5)
            # The code returns the base confidence directly if no valid comparison agents are found.
            self.assertAlmostEqual(score, 0.5, msg=f"Score for {dim} should be default")
            self.assertAlmostEqual(conf, 0.3, msg=f"Confidence for {dim} should be the base confidence")

    @patch('random.sample')
    @patch('trust_market.auditor.BatchEvaluator.compare_agent_profiles_and_convs')
    def test_evaluate_agent_comparative_audit_with_competitors(self, mock_compare, mock_random_sample):
        """Test comparative audit with competitors, mocking the BatchEvaluator response."""
        agent_id_target = 1
        agent_id_competitor = 2
        evaluation_round = 1
        dimensions = ["Factual_Correctness", "Safety_Security"]

        # Mock random.sample to pick our competitor
        mock_random_sample.return_value = [(
            agent_id_competitor, 
            self.regulator.get_agent_conversations(agent_id_competitor), 
            self.regulator.agent_profiles[agent_id_competitor]
        )]
        
        # Mock the comparison result from the BatchEvaluator
        # InLLM "A" is the first agent passed, "B" is the second
        mock_comparison_results = {
            "Factual_Correctness": {"winner": "A", "confidence": 4, "reasoning": "A is better"},
            "Safety_Security": {"winner": "B", "confidence": 2, "reasoning": "B is better"}
        }
        mock_compare.return_value = mock_comparison_results

        # Execute
        results = self.regulator.evaluate_agent(
            agent_id=agent_id_target, dimensions=dimensions, evaluation_round=evaluation_round, use_comparative=True
        )

        # Assertions
        mock_compare.assert_called_once()
        args, kwargs = mock_compare.call_args
        # The call is positional: (profile_a, convs_a, id_a, profile_b, convs_b, id_b, dims)
        self.assertEqual(args[2], agent_id_target)
        self.assertEqual(args[5], agent_id_competitor)
        self.assertEqual(args[6], dimensions)

        # Check that scores were updated based on the "winner"
        self.assertGreater(results['Factual_Correctness'][0], 0.5)
        self.assertLess(results['Safety_Security'][0], 0.5)
        
        # Check that derived scores for BOTH agents are updated in the regulator's state
        self.assertIn(agent_id_target, self.regulator.derived_agent_scores)
        self.assertIn(agent_id_competitor, self.regulator.derived_agent_scores)
        self.assertGreater(self.regulator.derived_agent_scores[agent_id_target]['Factual_Correctness'], 0.5)
        self.assertLess(self.regulator.derived_agent_scores[agent_id_target]['Safety_Security'], 0.5)
        # Competitor's scores should be the inverse
        self.assertLess(self.regulator.derived_agent_scores[agent_id_competitor]['Factual_Correctness'], 0.5)
        self.assertGreater(self.regulator.derived_agent_scores[agent_id_competitor]['Safety_Security'], 0.5)

    @patch('trust_market.auditor.BatchEvaluator.compare_agent_profiles_and_convs')
    def test_evaluate_agent_caching(self, mock_compare):
        """Test that comparative audit results are cached within a round."""
        agent_id_target = 1
        evaluation_round = 1
        
        mock_compare.return_value = {"Factual_Correctness": {"winner": "A", "confidence": 4}}

        # Correctly mock random.sample to provide conversations
        with patch('random.sample', return_value=[(2, self.conversations[2], self.profiles[2])]):
             # First call in the round
            results1 = self.regulator.evaluate_agent(agent_id=agent_id_target, evaluation_round=evaluation_round, dimensions=["Factual_Correctness"], use_comparative=True)
            self.assertGreaterEqual(mock_compare.call_count, 1)
            call_count_after_first = mock_compare.call_count

            # Second call in the same round for the same agent
            results2 = self.regulator.evaluate_agent(agent_id=agent_id_target, evaluation_round=evaluation_round, dimensions=["Factual_Correctness"], use_comparative=True)
            # The *overall evaluation* is cached, so the comparison method shouldn't be called again for the same agent/round.
            self.assertEqual(mock_compare.call_count, call_count_after_first) 
            self.assertEqual(results1, results2)

            # New evaluation round should trigger a new call
            results3 = self.regulator.evaluate_agent(agent_id=agent_id_target, evaluation_round=evaluation_round + 1, dimensions=["Factual_Correctness"], use_comparative=True)
            self.assertGreater(mock_compare.call_count, call_count_after_first)

    @patch('trust_market.regulator.Regulator.evaluate_agent')
    def test_decide_investments_buy_sell_logic(self, mock_evaluate_agent):
        """Test decide_investments logic for buying undervalued and selling overvalued assets."""
        # Focus on a single dimension for clarity
        dim_test = "Factual_Correctness"
        self.regulator.expertise_dimensions = [dim_test]

        # Setup market prices for the test dimension
        self.market.agent_amm_params[1][dim_test] = {'R': 30, 'T': 100} # P=0.3
        self.market.agent_amm_params[2][dim_test] = {'R': 50, 'T': 100} # P=0.5
        self.market.agent_amm_params[3][dim_test] = {'R': 80, 'T': 100} # P=0.8

        # Mock evaluations: Regulator thinks Agent 1 is high quality, Agent 3 is low quality.
        mock_evals = {
            1: {dim_test: (0.9, 0.9)}, # Undervalued (Score 0.9 > Price 0.3) -> BUY
            2: {dim_test: (0.5, 0.9)}, # Fairly valued (Score 0.5 == Price 0.5) -> HOLD
            3: {dim_test: (0.2, 0.9)}, # Overvalued (Score 0.2 < Price 0.8) -> SELL
        }
        
        mock_evaluate_agent.side_effect = lambda agent_id, **kwargs: mock_evals.get(agent_id)
        
        # To test the rank-mapping logic, we need to limit capacity so the projection logic isn't used
        # We also need to set the regulatory_capacity to 0 to avoid triggering check_market_capacity
        self.regulator.config['regulatory_capacity'] = 0.0

        # Execute
        investments = self.regulator.decide_investments(evaluation_round=1, use_comparative=True)

        # Assertions
        trades = {(agent_id, dim): amount for agent_id, dim, amount, conf in investments}
        
        # The logic for rank-mapping based trades is complex. We expect a tendency.
        # Agent 1 is rank #1 in evals, but #3 in market. It should be a BUY target.
        self.assertIn((1, dim_test), trades, "Should propose a trade for undervalued agent 1")
        self.assertGreater(trades[(1, dim_test)], 0, "Should propose BUYING agent 1")

        # Agent 3 is rank #3 in evals, but #1 in market. It should be a SELL target.
        self.assertIn((3, dim_test), trades, "Should propose a trade for overvalued agent 3")
        self.assertLess(trades[(3, dim_test)], 0, "Should propose SELLING agent 3")

        # With projection logic, we assert the relative order of trades based on quality vs price.
        # Agent 1 is most undervalued, so it should have the largest trade value.
        # Agent 2 is fairly valued, Agent 3 is overvalued.
        self.assertGreater(trades[(1, dim_test)], trades.get((2, dim_test), 0))
        self.assertGreater(trades.get((2, dim_test), 0), trades.get((3, dim_test), 0))

    @patch('trust_market.regulator.Regulator.evaluate_agent')
    def test_decide_investments_projected_price_logic(self, mock_evaluate_agent):
        """Test investment logic when using steady-state price projection due to high capacity."""
        dim = "Safety_Security"
        self.regulator.expertise_dimensions = [dim] # Focus on one dimension
        # Give the regulator a very large amount of cash to trigger the projection logic.
        self.regulator.config['regulatory_capacity'] = 10000.0
        self.regulator.config['min_delta_value_trade_threshold'] = -1.0 # Ensure all trades are considered

        # Setup: two agents, same low price, but regulator sees one as much higher quality.
        mock_evals = {
            1: {dim: (0.9, 0.95)}, # High quality
            2: {dim: (0.3, 0.95)}, # Low quality
            3: {dim: (0.5, 0.95)}, # Mid quality
        }
        for agent_id in self.profiles.keys():
             if agent_id not in mock_evals: mock_evals[agent_id] = {}
             for d in self.regulator.expertise_dimensions:
                if d not in mock_evals[agent_id]:
                    mock_evals[agent_id][d] = (0.5, 0.5)

        mock_evaluate_agent.side_effect = lambda agent_id, **kwargs: mock_evals.get(agent_id)
        
        # All agents start with the same low price
        for agent_id in self.profiles.keys():
            self.market.agent_amm_params[agent_id][dim] = {'R': 10, 'T': 100} # P=0.1

        # Execute
        investments = self.regulator.decide_investments(evaluation_round=1)

        # Assertions
        trades = {(agent_id, d): amount for agent_id, d, amount, conf in investments if d == dim}

        self.assertIn((1, dim), trades)
        self.assertIn((2, dim), trades)
        
        # The projection logic should aim to correct the market. Since all prices are low,
        # it might propose buying all of them, but the amounts should correlate with quality.
        self.assertGreater(trades[(1, dim)], trades.get((3, dim), 0))
        self.assertGreater(trades[(3, dim)], trades.get((2, dim), 0))
        self.assertGreater(trades[(2, dim)], 0, "Should still be a buy, as its projected price > current price")

    def test_get_target_price_from_rank_mapping_logic(self):
        """Test the internal logic of the rank mapping helper function."""
        dim = "Value_Alignment"
        # Regulator's view: A > B > C
        own_evals = {1: {dim: (0.9, 0.9)}, 2: {dim: (0.7, 0.9)}, 3: {dim: (0.5, 0.9)}}
        # Market's view (prices): C > B > A
        market_prices = {1: {dim: 0.6}, 2: {dim: 0.7}, 3: {dim: 0.8}}
        
        # Regulator thinks A(1) should be rank #1. The market's #1 is C(3), priced at 0.8.
        # So the target for A should be nudged from its current 0.6 towards 0.8.
        target_price_A = self.regulator._get_target_price_from_rank_mapping(1, dim, own_evals, market_prices, 0.9)
        self.assertTrue(0.6 < target_price_A <= 0.8)

        # Regulator thinks C(3) should be rank #3. The market's #3 is A(1), priced at 0.6.
        # So the target for C should be nudged from its current 0.8 towards 0.6.
        target_price_C = self.regulator._get_target_price_from_rank_mapping(3, dim, own_evals, market_prices, 0.9)
        self.assertTrue(0.6 <= target_price_C < 0.8)

        # Regulator thinks B(2) should be rank #2. The market's #2 is B(2), priced at 0.7.
        # Ranks match, so price should not change much.
        target_price_B = self.regulator._get_target_price_from_rank_mapping(2, dim, own_evals, market_prices, 0.9)
        self.assertAlmostEqual(target_price_B, 0.7, delta=0.01)

    def test_comparative_audit_updates_both_agents(self):
        """Test that a comparison between two agents updates the derived scores for both."""
        agent_a, agent_b = 1, 2
        dims = ["Factual_Correctness"]
        mock_comparison = {"Factual_Correctness": {"winner": "A", "confidence": 5, "reasoning": "A is much better"}}

        with patch.object(self.regulator.batch_evaluator, 'compare_agent_profiles_and_convs', return_value=mock_comparison):
            with patch('random.sample', return_value=[(agent_b, self.conversations[agent_b], self.profiles[agent_b])]):
                self.regulator.evaluate_agent(agent_id=agent_a, dimensions=dims, evaluation_round=1, use_comparative=True)

        self.assertIn(agent_a, self.regulator.derived_agent_scores)
        self.assertIn(agent_b, self.regulator.derived_agent_scores)
        self.assertGreater(self.regulator.derived_agent_scores[agent_a]["Factual_Correctness"], 0.5)
        self.assertLess(self.regulator.derived_agent_scores[agent_b]["Factual_Correctness"], 0.5)
        self.assertGreater(len(self.regulator.derived_agent_confidences[agent_a]["Factual_Correctness"]), 0)
        self.assertGreater(len(self.regulator.derived_agent_confidences[agent_b]["Factual_Correctness"]), 0)
        self.assertEqual(self.regulator.agent_comparison_counts[agent_a], 1)
        self.assertEqual(self.regulator.agent_comparison_counts[agent_b], 1)

    def test_confidence_aggregation_logic(self):
        """Test the internal logic of the confidence aggregation helper function."""
        # Test case 1: Aggregating a list of new confidences
        new_confs = [0.6, 0.7, 0.8]
        base_conf = 0.5
        weight_new = 0.5
        
        # Manually calculate expected precision/confidence
        new_precisions = [c / (1 - c) for c in new_confs]
        agg_new_precision = sum(new_precisions)
        base_precision = base_conf / (1 - base_conf)
        combined_precision = (weight_new * agg_new_precision) + ((1 - weight_new) * base_precision)
        expected_conf = combined_precision / (combined_precision + 1)
        
        calculated_conf = self.regulator._aggregate_confidences(new_confs, base_conf, weight_new)
        self.assertAlmostEqual(calculated_conf, expected_conf, delta=1e-6)

        # Test case 2: No new confidences
        self.assertEqual(self.regulator._aggregate_confidences([], 0.5, 0.5), 0.5)

        # Test case 3: High weight to new info
        weight_new_high = 0.9
        combined_precision_high = (weight_new_high * agg_new_precision) + ((1 - weight_new_high) * base_precision)
        expected_conf_high = combined_precision_high / (combined_precision_high + 1)
        calculated_conf_high = self.regulator._aggregate_confidences(new_confs, base_conf, weight_new_high)
        self.assertAlmostEqual(calculated_conf_high, expected_conf_high, delta=1e-6)
        self.assertGreater(calculated_conf_high, calculated_conf)

    def test_calculate_total_portfolio_value_potential(self):
        """Test the calculation of total portfolio value."""
        dim = "Value_Alignment"
        # Setup initial state
        self.market.source_available_capacity[self.regulator.source_id] = {dim: 500.0}
        self.market.source_investments[self.regulator.source_id][1][dim] = 20.0 # 20 shares
        self.market.agent_amm_params[1][dim] = {'R': 80.0, 'T': 100.0} # Price = 0.8
        
        # Expected value: 500 (cash) + 20 * 0.8 (shares value) = 516
        expected_total_value = 516.0
        
        with patch('builtins.print'): # Suppress print statements
            total_value = self.regulator._calculate_total_portfolio_value_potential()

        self.assertIn(dim, total_value)
        self.assertAlmostEqual(total_value[dim], expected_total_value)


if __name__ == '__main__':
    unittest.main() 