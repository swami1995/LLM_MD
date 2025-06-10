import unittest
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
from collections import defaultdict

# Assume the following classes are in these modules
from trust_market.auditor import AuditorWithProfileAnalysis, ProfileAnalyzer, BatchEvaluator
from trust_market.trust_market import TrustMarket
from trust_market.info_sources import InformationSource

class TestAuditorWithProfileAnalysis(unittest.TestCase):

    def setUp(self):
        """Set up a fresh auditor and mock market for each test."""
        # --- Manually start the patch ---
        self.mock_genai_patch = patch('google.genai.Client')
        mock_genai_client_constructor = self.mock_genai_patch.start()

        # --- Mock genai Client ---
        mock_response = MagicMock()
        mock_response.text = "{}"
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = 'STOP'
        mock_candidate.content.parts = [MagicMock()]
        mock_candidate.content.parts[0].text = "{}"
        mock_response.candidates = [mock_candidate]
        mock_instance = mock_genai_client_constructor.return_value
        mock_instance.models.generate_content.return_value = mock_response
        
        # --- Mock Market ---
        self.market = Mock(spec=TrustMarket)
        self.market.source_investments = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.market.source_available_capacity = defaultdict(lambda: defaultdict(float))
        self.market.agent_amm_params = defaultdict(lambda: defaultdict(dict))
        self.market.agent_trust_scores = defaultdict(lambda: defaultdict(float))
        
        # --- Auditor Instance ---
        self.auditor = AuditorWithProfileAnalysis(
            source_id='test_auditor',
            market=self.market,
            api_key='fake_key'
        )

        # Add agent profiles
        profiles = {
            1: {'primary_goals': [['efficiency', 0.8]], 'knowledge_breadth': 0.7},
            2: {'primary_goals': [['safety', 0.9]], 'knowledge_breadth': 0.8},
            3: {'primary_goals': [['sales', 0.9]], 'knowledge_breadth': 0.6},
            4: {'primary_goals': [['support', 0.85]], 'knowledge_breadth': 0.9}
        }
        for agent_id, profile in profiles.items():
            self.auditor.add_agent_profile(agent_id, profile)
            for dim in self.auditor.expertise_dimensions:
                self.market.agent_amm_params[agent_id][dim] = {'R': 50.0, 'T': 100.0}
                self.market.agent_trust_scores[agent_id][dim] = 0.5
        
        # Set up initial state
        self.market.source_available_capacity[self.auditor.source_id] = {dim: 1000.0 for dim in self.auditor.expertise_dimensions}
        self.market.source_investments[self.auditor.source_id][2]['Safety_Security'] = 10.0

    def tearDown(self):
        """Stop the patch after each test."""
        self.mock_genai_patch.stop()

    def test_auditor_initialization(self):
        """Test if the auditor and its components are initialized correctly."""
        self.assertEqual(self.auditor.source_id, 'test_auditor')
        self.assertIsNotNone(self.auditor.market)
        self.assertIsNotNone(self.auditor.profile_analyzer)
        self.assertIsNotNone(self.auditor.batch_evaluator)
        self.assertEqual(len(self.auditor.agent_profiles), 4)

    def test_perform_comparative_audit_no_competitors(self):
        """Test comparative audit when there are no other valid agents to compare against."""
        self.auditor.agent_profiles = {1: self.auditor.agent_profiles[1]}
        results = self.auditor.perform_comparative_audit(agent_id=1, evaluation_round=1)
        for dim in self.auditor.expertise_dimensions:
            self.assertIn(dim, results)
            score, conf = results[dim]
            self.assertAlmostEqual(score, 0.5)
            self.assertAlmostEqual(conf, 0.3)

    def test_perform_comparative_audit_with_competitors(self):
        """Test comparative audit with competitors, mocking the BatchEvaluator response."""
        agent_id_target = 1
        agent_id_competitor = 2
        evaluation_round = 1
        dimensions = ["Factual_Correctness", "Safety_Security"]

        mock_comparison_results = {
            "Factual_Correctness": {"winner": "A", "confidence": 4, "reasoning": "A is better"},
            "Safety_Security": {"winner": "B", "confidence": 2, "reasoning": "B is better"}
        }
        
        with patch.object(self.auditor.batch_evaluator, 'compare_agent_profiles', return_value=mock_comparison_results) as mock_compare:
            with patch('random.sample', return_value=[(agent_id_competitor, [], self.auditor.agent_profiles[agent_id_competitor])]):
                results = self.auditor.perform_comparative_audit(
                    agent_id=agent_id_target, dimensions=dimensions, evaluation_round=evaluation_round
                )
                mock_compare.assert_called_once()
                args, kwargs = mock_compare.call_args
                self.assertEqual(args[1], agent_id_target)
                self.assertEqual(args[3], agent_id_competitor)
                self.assertEqual(args[4], dimensions)

        self.assertIn('Factual_Correctness', results)
        self.assertGreater(results['Factual_Correctness'][0], 0.5)
        self.assertLess(results['Safety_Security'][0], 0.5)
        self.assertGreater(self.auditor.derived_agent_scores[agent_id_target]['Factual_Correctness'], 0.5)
        self.assertLess(self.auditor.derived_agent_scores[agent_id_target]['Safety_Security'], 0.5)
        self.assertLess(self.auditor.derived_agent_scores[agent_id_competitor]['Factual_Correctness'], 0.5)
        self.assertGreater(self.auditor.derived_agent_scores[agent_id_competitor]['Safety_Security'], 0.5)

    def test_perform_comparative_audit_caching(self):
        """Test that comparative audit results are cached."""
        agent_id_target = 1
        evaluation_round = 1

        with patch.object(self.auditor.batch_evaluator, 'compare_agent_profiles') as mock_compare:
            mock_compare.return_value = {"Factual_Correctness": {"winner": "A", "confidence": 4, "reasoning": "A is better"}}
            
            results1 = self.auditor.perform_comparative_audit(agent_id=agent_id_target, evaluation_round=evaluation_round, dimensions=["Factual_Correctness"])
            self.assertGreaterEqual(mock_compare.call_count, 1)

            call_count_after_first = mock_compare.call_count
            
            results2 = self.auditor.perform_comparative_audit(agent_id=agent_id_target, evaluation_round=evaluation_round, dimensions=["Factual_Correctness"])
            self.assertEqual(mock_compare.call_count, call_count_after_first)
            self.assertEqual(results1, results2)

            self.auditor.perform_comparative_audit(agent_id=agent_id_target, evaluation_round=evaluation_round + 1, dimensions=["Factual_Correctness"])
            self.assertGreater(mock_compare.call_count, call_count_after_first)

    def test_decide_investments_buy_and_sell(self):
        """Test decide_investments logic for buying and selling."""
        # --- Custom Setup for this Test ---
        # 1. Set non-uniform market prices to test ranking logic properly.
        # Agent 1 is expensive, Agent 2 is mid, Agent 3 is cheap.
        self.market.agent_amm_params[1]['Factual_Correctness'] = {'R': 80, 'T': 100} # P=0.8
        self.market.agent_amm_params[1]['Safety_Security'] = {'R': 80, 'T': 100}     # P=0.8
        self.market.agent_amm_params[2]['Factual_Correctness'] = {'R': 50, 'T': 100} # P=0.5
        self.market.agent_amm_params[2]['Safety_Security'] = {'R': 50, 'T': 100}     # P=0.5
        self.market.agent_amm_params[3]['Factual_Correctness'] = {'R': 30, 'T': 100} # P=0.3
        self.market.agent_amm_params[3]['Safety_Security'] = {'R': 30, 'T': 100}     # P=0.3

        # 2. Limit available capacity to prevent triggering the aggressive projection logic.
        self.market.source_available_capacity[self.auditor.source_id]['Factual_Correctness'] = 10.0
        self.market.source_available_capacity[self.auditor.source_id]['Safety_Security'] = 10.0

        # --- Mock Evaluations ---
        # Agent 1 / FC: score 0.8 == price 0.8 -> Correctly valued
        # Agent 2 / SS: score 0.2 < price 0.5 -> Overvalued (SELL)
        # Agent 3 / FC: score 0.8 > price 0.3 -> Undervalued (BUY)
        mock_evaluations = {
            1: {"Factual_Correctness": (0.8, 0.9), "Safety_Security": (0.5, 0.8)},
            2: {"Factual_Correctness": (0.5, 0.8), "Safety_Security": (0.2, 0.9)},
            3: {"Factual_Correctness": (0.8, 0.9), "Safety_Security": (0.5, 0.9)}
        }
        self.auditor.agent_profiles = {1:{}, 2:{}, 3:{}}
        
        with patch.object(self.auditor, 'evaluate_agent', side_effect=lambda agent_id, **kwargs: mock_evaluations.get(agent_id)) as mock_evaluate:
            investments = self.auditor.decide_investments(evaluation_round=1, use_comparative=False)
            self.assertEqual(mock_evaluate.call_count, len(self.auditor.agent_profiles))

        # --- Assertions ---
        buy_action = next((t for t in investments if t[0] == 3 and t[1] == "Factual_Correctness"), None)
        sell_action = next((t for t in investments if t[0] == 2 and t[1] == "Safety_Security"), None)

        self.assertIsNotNone(buy_action, "Should have proposed to buy for agent 3 (undervalued)")
        self.assertGreater(buy_action[2], 0, "Cash amount for buying should be positive")
        
        self.assertIsNotNone(sell_action, "Should have proposed to sell for agent 2 (overvalued)")
        self.assertLess(sell_action[2], 0, "Cash amount for selling should be negative")

        # Agent 1 was correctly valued, so no major trade should be proposed.
        agent_1_trades = [t for t in investments if t[0] == 1 and abs(t[2]) > self.auditor.config['min_delta_value_trade_threshold']]
        self.assertEqual(len(agent_1_trades), 0, "Should not propose significant trades for fairly valued agent 1")

    def test_comparative_audit_updates_both_agents(self):
        """Test that a comparison between two agents updates the derived scores for both."""
        agent_a, agent_b = 1, 2
        dims = ["Factual_Correctness"]
        mock_comparison = {"Factual_Correctness": {"winner": "A", "confidence": 5, "reasoning": "A is much better"}}

        with patch.object(self.auditor.batch_evaluator, 'compare_agent_profiles', return_value=mock_comparison):
            with patch('random.sample', return_value=[(agent_b, [], self.auditor.agent_profiles[agent_b])]):
                self.auditor.perform_comparative_audit(agent_id=agent_a, dimensions=dims, evaluation_round=1)

        self.assertIn(agent_a, self.auditor.derived_agent_scores)
        self.assertIn(agent_b, self.auditor.derived_agent_scores)
        self.assertGreater(self.auditor.derived_agent_scores[agent_a]["Factual_Correctness"], 0.5)
        self.assertLess(self.auditor.derived_agent_scores[agent_b]["Factual_Correctness"], 0.5)
        self.assertGreater(len(self.auditor.derived_agent_confidences[agent_a]["Factual_Correctness"]), 0)
        self.assertGreater(len(self.auditor.derived_agent_confidences[agent_b]["Factual_Correctness"]), 0)
        self.assertGreater(self.auditor.agent_comparison_counts[agent_a], 0)
        self.assertGreater(self.auditor.agent_comparison_counts[agent_b], 0)

    def test_decide_investments_respects_budget(self):
        """Test that investment decisions are scaled based on available budget."""
        dim = "Factual_Correctness"
        limited_budget = 10.0
        self.market.source_available_capacity[self.auditor.source_id][dim] = limited_budget
        
        mock_evals = {1: {dim: (0.9, 0.9)}, 2: {dim: (0.9, 0.9)}}
        self.auditor.agent_profiles = {1:{}, 2:{}}
        
        with patch.object(self.auditor, 'evaluate_agent', side_effect=lambda agent_id, **kwargs: mock_evals.get(agent_id)):
            investments = self.auditor.decide_investments(evaluation_round=1, use_comparative=False)
            
        total_proposed_buy = sum(amt for _, d, amt, _ in investments if d == dim and amt > 0)
        self.assertLessEqual(total_proposed_buy, limited_budget)

    def test_get_target_price_from_rank_mapping_logic(self):
        """Test the internal logic of the rank mapping helper function."""
        dim = "Value_Alignment"
        own_evals = {'A': {dim: (0.9, 0.9)}, 'B': {dim: (0.7, 0.9)}, 'C': {dim: (0.5, 0.9)}}
        market_prices = {'A': {dim: 0.6}, 'B': {dim: 0.7}, 'C': {dim: 0.8}}
        
        target_price_A = self.auditor._get_target_price_from_rank_mapping('A', dim, own_evals, market_prices, 0.9)
        self.assertTrue(0.6 < target_price_A <= 0.8)

        target_price_C = self.auditor._get_target_price_from_rank_mapping('C', dim, own_evals, market_prices, 0.9)
        self.assertTrue(0.6 <= target_price_C < 0.8)

        target_price_B = self.auditor._get_target_price_from_rank_mapping('B', dim, own_evals, market_prices, 0.9)
        self.assertAlmostEqual(target_price_B, 0.7)

    def test_decide_investments_projected_price_logic(self):
        """Test the investment logic when using steady-state price projection."""
        # This test ensures the logic path for high-capacity investors is working.
        dim = "Factual_Correctness"
        # 1. Give the auditor a very large amount of cash to trigger the projection logic.
        self.market.source_available_capacity[self.auditor.source_id][dim] = 5000.0

        # 2. Set up two agents with the same low price but different quality scores.
        mock_evals = {
            1: {dim: (0.9, 0.9)}, # High quality
            2: {dim: (0.3, 0.9)}, # Low quality
        }
        self.auditor.agent_profiles = {1:{}, 2:{}}
        self.market.agent_amm_params[1][dim] = {'R': 10, 'T': 100} # P=0.1
        self.market.agent_amm_params[2][dim] = {'R': 10, 'T': 100} # P=0.1

        # 3. Run decide_investments
        with patch.object(self.auditor, 'evaluate_agent', side_effect=lambda agent_id, **kwargs: mock_evals.get(agent_id)):
            investments = self.auditor.decide_investments(evaluation_round=1, use_comparative=False)
        
        # 4. Assertions
        # The projection logic should allocate more capital to the higher-quality agent (1).
        trades = {(agent_id, d): amount for agent_id, d, amount, conf in investments}
        self.assertIn((1, dim), trades)
        self.assertIn((2, dim), trades)
        self.assertGreater(trades[(1, dim)], trades[(2, dim)], 
                         "Should invest more in the higher quality agent when using price projection.")

    def test_confidence_aggregation_logic(self):
        """Tests the logic for aggregating confidence scores over time."""
        agent_id = 1
        dim = "Transparency"
        
        # Initially, with no data, confidence should be low.
        initial_conf = self.auditor._calculate_derived_confidence(agent_id, [dim])[dim]
        self.assertLess(initial_conf, 0.5)

        # Add a few confidence scores from mock comparisons
        self.auditor._update_agent_confidences(agent_id, {dim: 0.6}, [dim])
        conf1 = self.auditor._calculate_derived_confidence(agent_id, [dim])[dim]
        self.assertGreater(conf1, initial_conf)

        self.auditor._update_agent_confidences(agent_id, {dim: 0.7}, [dim])
        conf2 = self.auditor._calculate_derived_confidence(agent_id, [dim])[dim]
        self.assertGreater(conf2, conf1) # Confidence should increase with more data

        # Test plateau effect: adding the same confidence repeatedly should yield smaller gains
        for _ in range(5):
            self.auditor._update_agent_confidences(agent_id, {dim: 0.7}, [dim])
        conf3 = self.auditor._calculate_derived_confidence(agent_id, [dim])[dim]
        self.assertGreater(conf3, conf2)
        # The increase from conf2 to conf3 should be less than from conf1 to conf2
        self.assertLess(conf3 - conf2, conf2 - conf1)

    def test_calculate_total_portfolio_value_potential(self):
        """Tests the helper method for calculating total portfolio value."""
        dim = "Factual_Correctness"
        # 1. Setup: Give the auditor some cash and some shares.
        self.market.source_available_capacity[self.auditor.source_id][dim] = 1000.0
        self.market.source_investments[self.auditor.source_id][1][dim] = 10.0 # 10 shares
        # Set the market price for these shares to be 20.0
        self.market.agent_amm_params[1][dim] = {'R': 200, 'T': 10} # P=20

        # 2. Calculate potential
        total_potential = self.auditor._calculate_total_portfolio_value_potential()

        # 3. Assert
        # Expected value = cash + (shares * price) = 1000 + (10 * 20) = 1200
        self.assertIn(dim, total_potential)
        self.assertAlmostEqual(total_potential[dim], 1200.0)

if __name__ == '__main__':
    unittest.main() 