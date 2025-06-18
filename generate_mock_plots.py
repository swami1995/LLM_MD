import time
import random
import numpy as np
from collections import defaultdict

from trust_market.trust_market import TrustMarket

def create_realistic_mock_market(num_rounds=50, num_agents=4):
    """
    Creates a TrustMarket instance and populates its temporal_db with
    realistic mock data based on predefined agent and source archetypes.
    """
    print("--- Creating Market with Realistic Mock Data ---")
    
    config = {
        'dimensions': ["Factual_Correctness", "Safety_Security", "Communication_Quality"],
    }
    market = TrustMarket(config)

    # --- Define Archetypes ---
    agents = {
        0: {'name': 'High Performer', 'base': 0.6, 'trend': 0.008, 'noise': 0.05},
        1: {'name': 'Low Performer', 'base': 0.7, 'trend': -0.01, 'noise': 0.05},
        2: {'name': 'Unpredictable', 'base': 0.5, 'trend': 0, 'noise': 0.15},
        3: {'name': 'Comeback Kid', 'base': 0.2, 'trend': 0.015, 'noise': 0.08},
    }
    sources = {
        'smart_regulator': {'accuracy': 0.95},
        'noisy_user_rep': {'accuracy': 0.75}
    }
    dims = config['dimensions']
    
    # --- State tracking for mock generation ---
    source_portfolios = {
        source_id: {
            # Each source starts with 1000 cash in each dimension
            'available_cash': {dim: 1000.0 for dim in dims},
            'shares': defaultdict(lambda: defaultdict(float)) # {agent_id: {dim: num_shares}}
        } for source_id in sources
    }
    
    # --- Generate Historical Data ---
    trust_scores_data = []
    investments_data = []
    source_states_data = [] # This list will be populated
    
    # Initialize scores
    agent_scores = {aid: {dim: arch['base'] for dim in dims} for aid, arch in agents.items()}

    for r in range(num_rounds):
        market.evaluation_round = r
        
        # 1. Generate Trust Scores for the round
        for aid, arch in agents.items():
            for dim in dims:
                # Apply trend
                score = agent_scores[aid][dim] + arch['trend']
                # Add noise
                score += (random.random() - 0.5) * arch['noise']
                # Make one dimension slightly different for variety
                if dim == "Communication_Quality":
                    score += (random.random() - 0.5) * 0.05
                
                score = max(0.05, min(0.95, score)) # Clamp score
                agent_scores[aid][dim] = score
                
                trust_scores_data.append({
                    'evaluation_round': r, 'timestamp': time.time(),
                    'agent_id': aid, 'dimension': dim,
                    'new_score': score, 'old_score': score - 0.01, 'change_source': 'mock_sim'
                })

        # 2. Generate Investments for the round
        #   Only allow investments on sparse rounds to avoid over-crowding:
        #   round 1 and every 5th round thereafter (5, 10, 15, ...)
        if r == 1 or r % 5 == 0:
            for source_id, source_arch in sources.items():
                # Decide how many investments this source will attempt this round (0-3)
                num_actions = np.random.randint(0, 4)  # 0,1,2,3 investments
                for _ in range(num_actions):
                    target_agent = random.choice(list(agents.keys()))
                    target_dim = random.choice(dims)

                    is_correct_move = random.random() < source_arch['accuracy']
                    agent_is_good = agents[target_agent]['trend'] >= 0

                    # If correct, invest in good agents, sell bad ones. If incorrect, do the opposite.
                    should_invest = (is_correct_move and agent_is_good) or (not is_correct_move and not agent_is_good)

                    amount = random.uniform(20, 100)
                    if not should_invest:
                        amount *= -1  # Make it a divestment

                    investments_data.append({
                        'evaluation_round': r, 'timestamp': time.time(),
                        'source_id': source_id, 'agent_id': target_agent, 'dimension': target_dim,
                        'amount': amount, 'confidence': random.uniform(0.6, 0.95),
                        'type': 'investment' if amount > 0 else 'divestment'
                    })

                    # --- Simulate the portfolio change and log the new state ---
                    portfolio = source_portfolios[source_id]
                    current_price = agent_scores[target_agent][target_dim]

                    if amount > 0:  # Buy
                        buy_amount = min(amount, portfolio['available_cash'][target_dim])
                        num_shares = buy_amount / current_price if current_price > 0 else 0
                        portfolio['available_cash'][target_dim] -= buy_amount
                        portfolio['shares'][target_agent][target_dim] += num_shares
                    else:  # Sell
                        sell_request_cash = abs(amount)
                        shares_owned = portfolio['shares'][target_agent][target_dim]
                        cash_from_sale = min(sell_request_cash, shares_owned * current_price)
                        portfolio['available_cash'][target_dim] += cash_from_sale
                        if current_price > 0:
                            portfolio['shares'][target_agent][target_dim] -= cash_from_sale / current_price

                    # After the transaction, recalc total value of this dimension
                    total_invested_value = 0
                    for aid, agent_shares in portfolio['shares'].items():
                        price = agent_scores[aid][target_dim]
                        total_invested_value += agent_shares[target_dim] * price

                    available_cash = portfolio['available_cash'][target_dim]

                    source_states_data.append({
                        'evaluation_round': r, 'timestamp': time.time(),
                        'source_id': source_id, 'dimension': target_dim,
                        'total_invested_value': total_invested_value,
                        'available_cash': available_cash,
                        'total_value': total_invested_value + available_cash
                    })

    market.temporal_db['trust_scores'] = trust_scores_data
    market.temporal_db['investments'] = investments_data
    market.temporal_db['source_states'] = source_states_data
    
    print(f"  - Generated {len(trust_scores_data)} mock trust score records over {num_rounds} rounds.")
    print(f"  - Generated {len(investments_data)} mock investment records.")
    
    return market

def main():
    """
    Main function to generate mock data and display the resulting plots
    for visual inspection.
    """
    
    # 1. Create the market with mock data
    # Note: Because this script is for visual inspection, we don't mock plt.show()
    mock_market = create_realistic_mock_market(num_rounds=50)

    # 2. Visualize the trust scores with investment overlays
    print("\nDisplaying Trust Score Visualization...")
    print("Close the plot window to continue...")
    mock_market.visualize_trust_scores(
        show_investments=True,
        save_path='figures',
        experiment_name='mock_archetype_run'
    )
    
    print("\nDisplaying Source Value Visualization...")
    print("Close the plot window to finish.")
    mock_market.visualize_source_value(
        save_path='figures',
        experiment_name='mock_archetype_run'
    )

    print("\nVisualization script finished.")

if __name__ == "__main__":
    main() 