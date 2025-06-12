import unittest
import time
from unittest.mock import patch
import pandas as pd
import random

from trust_market.trust_market import TrustMarket

class TestVisualizationFunctions(unittest.TestCase):

    def setUp(self):
        """Set up a TrustMarket instance and populate its temporal_db with mock data."""
        print("\nSetting up for TestVisualizationFunctions...")
        
        # 1. Initialize the TrustMarket
        config = {
            'dimensions': ["Factual_Correctness", "Safety_Security"],
        }
        self.market = TrustMarket(config)

        # 2. Create Mock Historical Data
        self.mock_agents = [1, 2]
        self.mock_sources = ['regulator', 'user_rep']
        self.mock_dims = ["Factual_Correctness", "Safety_Security"]
        num_rounds = 10

        trust_scores_data = []
        investments_data = []
        source_performance_data = []

        # Generate data that shows a trend
        for r in range(num_rounds):
            self.market.evaluation_round = r
            for agent_id in self.mock_agents:
                for dim in self.mock_dims:
                    # Agent 1 trends up, Agent 2 trends down
                    base_score = 0.4 + (0.05 * r) if agent_id == 1 else 0.8 - (0.05 * r)
                    score = max(0, min(1, base_score + (random.random() - 0.5) * 0.1))
                    trust_scores_data.append({
                        'evaluation_round': r,
                        'timestamp': time.time(),
                        'agent_id': agent_id,
                        'dimension': dim,
                        'new_score': score,
                        'old_score': score - 0.01,
                        'change_source': 'test'
                    })

            # Simulate some investment/divestment actions
            if r > 0 and r % 2 == 0: # Every 2 rounds
                # Regulator invests in Agent 1 (good) and divests from Agent 2 (bad)
                investments_data.append({
                    'evaluation_round': r, 'timestamp': time.time(),
                    'source_id': 'regulator', 'agent_id': 1, 'dimension': 'Factual_Correctness',
                    'amount': 50.0, 'confidence': 0.8, 'type': 'investment'
                })
                investments_data.append({
                    'evaluation_round': r, 'timestamp': time.time(),
                    'source_id': 'regulator', 'agent_id': 2, 'dimension': 'Safety_Security',
                    'amount': -30.0, 'confidence': 0.7, 'type': 'divestment'
                })
            
            if r > 0 and r % 3 == 0: # Every 3 rounds
                 # User Rep invests in Agent 1
                investments_data.append({
                    'evaluation_round': r, 'timestamp': time.time(),
                    'source_id': 'user_rep', 'agent_id': 1, 'dimension': 'Safety_Security',
                    'amount': 20.0, 'confidence': 0.85, 'type': 'investment'
                })

            # Mock source performance data
            for source in self.mock_sources:
                 for dim in self.mock_dims:
                      perf = 0.6 + (r * 0.02) + (random.random() * 0.1) # Slowly increasing performance
                      source_performance_data.append({
                            'evaluation_round': r, 'timestamp': time.time(),
                            'source_id': source, 'dimension': dim,
                            'performance_score': perf
                      })


        self.market.temporal_db['trust_scores'] = trust_scores_data
        self.market.temporal_db['investments'] = investments_data
        self.market.temporal_db['source_performance'] = source_performance_data
        
        print(f"  - Created {len(trust_scores_data)} mock trust score records.")
        print(f"  - Created {len(investments_data)} mock investment records.")
        print(f"  - Created {len(source_performance_data)} mock performance records.")

    @patch('matplotlib.pyplot.show')
    def test_visualize_trust_scores_runs_without_error(self, mock_show):
        """
        Test that visualize_trust_scores runs, generates a figure, and calls show()
        without making any external API calls.
        """
        print("--> Running test_visualize_trust_scores_runs_without_error...")
        
        fig = self.market.visualize_trust_scores(show_investments=True)
        
        # 1. Assert that a figure object was created and returned
        self.assertIsNotNone(fig, "The visualize function should return a matplotlib figure object.")
        
        # 2. Assert that the plot was requested to be shown
        mock_show.assert_called_once()
        print("  - PASSED: Figure was created and plt.show() was called.")

    @patch('matplotlib.pyplot.show')
    def test_visualize_trust_scores_empty_data(self, mock_show):
        """Test that the function handles empty data gracefully."""
        print("--> Running test_visualize_trust_scores_empty_data...")
        self.market.temporal_db['trust_scores'] = []
        self.market.temporal_db['investments'] = []
        
        fig = self.market.visualize_trust_scores()
        
        self.assertIsNone(fig, "Figure should be None when there is no data.")
        mock_show.assert_not_called()
        print("  - PASSED: Handled empty trust score data correctly.")

    @patch('matplotlib.pyplot.show')
    def test_visualize_source_performance_runs_without_error(self, mock_show):
        """
        Test that visualize_source_performance runs without error on mock data.
        """
        print("--> Running test_visualize_source_performance_runs_without_error...")
        
        fig = self.market.visualize_source_performance()
        
        self.assertIsNotNone(fig, "The visualize function should return a matplotlib figure object.")
        mock_show.assert_called_once()
        print("  - PASSED: Figure was created and plt.show() was called.")

if __name__ == '__main__':
    unittest.main() 