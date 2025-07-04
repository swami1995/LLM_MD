import argparse
import json
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

def load_data(data_path: str):
    """Load and basic organize the detailed analysis data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    n_runs = len(raw_data)
    agent_pairs = set()
    dimensions = set()
    
    # Quick organization
    pair_data = defaultdict(list)
    
    for run_data in raw_data:
        for comparison in run_data['comparison_details']:
            aid1, aid2 = comparison['pair']
            pair_key = tuple(sorted([aid1, aid2]))
            agent_pairs.add(pair_key)
            
            comparison_with_run = comparison.copy()
            comparison_with_run['run_id'] = run_data['run_id']
            pair_data[pair_key].append(comparison_with_run)
            
            # Extract dimensions
            if 'derived_scores' in comparison:
                for scores_dict in comparison['derived_scores'].values():
                    dimensions.update(scores_dict.keys())
    
    return raw_data, pair_data, sorted(agent_pairs), sorted(dimensions), n_runs

def print_overview(pair_data, agent_pairs, dimensions, n_runs):
    """Print a quick overview of the data."""
    print("="*60)
    print("EVALUATION LOGS OVERVIEW")
    print("="*60)
    print(f"Runs: {n_runs}")
    print(f"Agent pairs: {len(agent_pairs)}")
    print(f"Dimensions: {len(dimensions)}")
    print(f"Total comparisons: {sum(len(comps) for comps in pair_data.values())}")
    
    print(f"\nAgent pairs evaluated:")
    for aid1, aid2 in agent_pairs:
        print(f"  • Agent {aid1} vs Agent {aid2} ({len(pair_data[(aid1, aid2)])} comparisons)")
    
    print(f"\nDimensions evaluated:")
    for dim in dimensions:
        print(f"  • {dim}")

def show_quick_summary_by_pair(pair_data, agent_pairs):
    """Show a quick summary of wins/consistency for each pair."""
    print("\n" + "="*60)
    print("QUICK PAIR SUMMARY")
    print("="*60)
    
    for aid1, aid2 in agent_pairs:
        comparisons = pair_data[(aid1, aid2)]
        print(f"\n--- Agent {aid1} vs Agent {aid2} ---")
        
        # Quick win analysis
        wins_agent1 = 0
        wins_agent2 = 0
        ties = 0
        
        score_diffs = []  # aid1 - aid2
        
        for comp in comparisons:
            if 'raw_winner' in comp and comp['raw_winner']:
                win_count = Counter(comp['raw_winner'])
                if win_count['A'] > win_count['B']:
                    wins_agent1 += 1
                elif win_count['B'] > win_count['A']:
                    wins_agent2 += 1
                else:
                    ties += 1
            
            # Calculate average score difference
            if 'derived_scores' in comp:
                scores_a = comp['derived_scores'].get(aid1, {})
                scores_b = comp['derived_scores'].get(aid2, {})
                if scores_a and scores_b:
                    avg_a = np.mean(list(scores_a.values()))
                    avg_b = np.mean(list(scores_b.values()))
                    score_diffs.append(avg_a - avg_b)
        
        print(f"  Wins: Agent {aid1}={wins_agent1}, Agent {aid2}={wins_agent2}, Ties={ties}")
        
        if score_diffs:
            mean_diff = np.mean(score_diffs)
            std_diff = np.std(score_diffs)
            print(f"  Score difference (A{aid1}-A{aid2}): {mean_diff:+.3f} ± {std_diff:.3f}")
            consistency = "High" if std_diff < 0.05 else "Medium" if std_diff < 0.1 else "Low"
            print(f"  Consistency: {consistency}")

def show_dimension_winners(pair_data, agent_pairs, dimensions):
    """Show quick dimension-wise winner patterns."""
    print("\n" + "="*60)
    print("DIMENSION WINNER PATTERNS")
    print("="*60)
    
    for dim in dimensions:
        print(f"\n--- {dim} ---")
        
        for aid1, aid2 in agent_pairs:
            comparisons = pair_data[(aid1, aid2)]
            
            wins_a = 0
            wins_b = 0
            ties = 0
            
            for comp in comparisons:
                if 'derived_scores' in comp and 'raw_winner' in comp:
                    # Find dimension index
                    if aid1 in comp['derived_scores']:
                        dim_keys = list(comp['derived_scores'][aid1].keys())
                        if dim in dim_keys:
                            dim_idx = dim_keys.index(dim)
                            if dim_idx < len(comp['raw_winner']):
                                winner = comp['raw_winner'][dim_idx]
                                if winner == 'A':
                                    wins_a += 1
                                elif winner == 'B':
                                    wins_b += 1
                                else:
                                    ties += 1
            
            if wins_a + wins_b + ties > 0:
                print(f"  A{aid1} vs A{aid2}: A{aid1}={wins_a}, A{aid2}={wins_b}, Ties={ties}")

def browse_detailed_reasoning(pair_data, agent_pairs, dimensions):
    """Interactive browsing of detailed reasoning."""
    print("\n" + "="*60)
    print("DETAILED REASONING BROWSER")
    print("="*60)
    print("Enter commands:")
    print("  pair X Y     - Show detailed comparison for agents X and Y")
    print("  dim D        - Show reasoning for dimension D across all pairs")
    print("  list         - List available pairs and dimensions")
    print("  quit         - Exit browser")
    
    while True:
        try:
            cmd = input("\n> ").strip().split()
            
            if not cmd or cmd[0] == 'quit':
                break
            
            elif cmd[0] == 'list':
                print("\nAvailable pairs:")
                for aid1, aid2 in agent_pairs:
                    print(f"  {aid1} {aid2}")
                print("\nAvailable dimensions:")
                for dim in dimensions:
                    print(f"  {dim}")
            
            elif cmd[0] == 'pair' and len(cmd) >= 3:
                try:
                    aid1, aid2 = int(cmd[1]), int(cmd[2])
                    pair_key = tuple(sorted([aid1, aid2]))
                    
                    if pair_key not in pair_data:
                        print(f"Pair ({aid1}, {aid2}) not found.")
                        continue
                    
                    print(f"\n=== Agent {aid1} vs Agent {aid2} ===")
                    comparisons = pair_data[pair_key]
                    
                    for run_idx, comp in enumerate(comparisons):
                        print(f"\n--- Run {comp['run_id'] + 1} ---")
                        
                        if 'raw_reasoning' in comp and 'derived_scores' in comp:
                            # Show reasoning for each dimension
                            if aid1 in comp['derived_scores']:
                                dim_keys = list(comp['derived_scores'][aid1].keys())
                                for i, dim in enumerate(dim_keys):
                                    if i < len(comp['raw_reasoning']):
                                        winner = comp['raw_winner'][i] if i < len(comp['raw_winner']) else 'Unknown'
                                        confidence = comp['raw_confidence'][i] if i < len(comp['raw_confidence']) else 0
                                        print(f"\n{dim}:")
                                        print(f"  Winner: {winner} (Confidence: {confidence})")
                                        print(f"  Scores: A{aid1}={comp['derived_scores'][aid1][dim]:.3f}, A{aid2}={comp['derived_scores'][aid2][dim]:.3f}")
                                        print(f"  Reasoning: {comp['raw_reasoning'][i]}")
                
                except ValueError:
                    print("Invalid agent IDs. Use: pair <agent1_id> <agent2_id>")
            
            elif cmd[0] == 'dim' and len(cmd) >= 2:
                dim_name = ' '.join(cmd[1:])
                if dim_name not in dimensions:
                    print(f"Dimension '{dim_name}' not found.")
                    continue
                
                print(f"\n=== {dim_name} ===")
                
                for aid1, aid2 in agent_pairs:
                    comparisons = pair_data[(aid1, aid2)]
                    print(f"\n--- Agent {aid1} vs Agent {aid2} ---")
                    
                    for comp in comparisons:
                        if 'derived_scores' in comp and aid1 in comp['derived_scores']:
                            dim_keys = list(comp['derived_scores'][aid1].keys())
                            if dim_name in dim_keys:
                                dim_idx = dim_keys.index(dim_name)
                                if dim_idx < len(comp['raw_reasoning']):
                                    winner = comp['raw_winner'][dim_idx] if dim_idx < len(comp['raw_winner']) else 'Unknown'
                                    confidence = comp['raw_confidence'][dim_idx] if dim_idx < len(comp['raw_confidence']) else 0
                                    print(f"\nRun {comp['run_id'] + 1}:")
                                    print(f"  Winner: {winner} (Confidence: {confidence})")
                                    print(f"  Scores: A{aid1}={comp['derived_scores'][aid1][dim_name]:.3f}, A{aid2}={comp['derived_scores'][aid2][dim_name]:.3f}")
                                    print(f"  Reasoning: {comp['raw_reasoning'][dim_idx]}")
            
            else:
                print("Unknown command. Use: pair X Y, dim D, list, or quit")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Quick scanner for detailed evaluation logs.")
    parser.add_argument("data_path", type=str, help="Path to the detailed analysis JSON file.")
    parser.add_argument("--mode", type=str, choices=['overview', 'summary', 'dimensions', 'browse'], 
                       default='overview', help="Scanning mode.")
    
    args = parser.parse_args()
    
    print(f"Loading data from: {args.data_path}")
    raw_data, pair_data, agent_pairs, dimensions, n_runs = load_data(args.data_path)
    
    if args.mode == 'overview':
        print_overview(pair_data, agent_pairs, dimensions, n_runs)
    
    elif args.mode == 'summary':
        print_overview(pair_data, agent_pairs, dimensions, n_runs)
        show_quick_summary_by_pair(pair_data, agent_pairs)
    
    elif args.mode == 'dimensions':
        print_overview(pair_data, agent_pairs, dimensions, n_runs)
        show_dimension_winners(pair_data, agent_pairs, dimensions)
    
    elif args.mode == 'browse':
        print_overview(pair_data, agent_pairs, dimensions, n_runs)
        browse_detailed_reasoning(pair_data, agent_pairs, dimensions)

if __name__ == '__main__':
    main() 