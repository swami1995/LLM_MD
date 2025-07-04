import argparse
import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import os
from datetime import datetime

class DetailedAnalysisViewer:
    """
    A comprehensive viewer for detailed evaluation logs that presents 
    pairwise comparison data in multiple convenient formats.
    """
    
    def __init__(self, data_path: str):
        """Load and organize the detailed analysis data."""
        with open(data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        self.n_runs = len(self.raw_data)
        self.agent_pairs = set()
        self.dimensions = set()
        self.agent_profiles = {}
        
        # Organize data by pair across runs
        self.pair_data = defaultdict(list)  # {(aid1, aid2): [run_data, ...]}
        self.dimension_data = defaultdict(lambda: defaultdict(list))  # {dim: {(aid1, aid2): [run_data, ...]}}
        
        self._organize_data()
        
    def _organize_data(self):
        """Organize raw data into convenient structures."""
        for run_data in self.raw_data:
            run_id = run_data['run_id']
            
            for comparison in run_data['comparison_details']:
                aid1, aid2 = comparison['pair']
                pair_key = tuple(sorted([aid1, aid2]))
                self.agent_pairs.add(pair_key)
                
                # Store agent profiles (they should be the same across runs)
                if aid1 not in self.agent_profiles:
                    self.agent_profiles[aid1] = comparison['agent_a_profile']
                if aid2 not in self.agent_profiles:
                    self.agent_profiles[aid2] = comparison['agent_b_profile']
                
                # Add run info to comparison data
                comparison_with_run = comparison.copy()
                comparison_with_run['run_id'] = run_id
                
                self.pair_data[pair_key].append(comparison_with_run)
                
                # Extract dimensions from raw reasoning
                if 'raw_reasoning' in comparison and comparison['raw_reasoning']:
                    # Assume raw_reasoning is a list corresponding to dimensions
                    # We need to extract the dimension names somehow
                    # Let's check derived_scores for dimension names
                    if 'derived_scores' in comparison:
                        for agent_id, scores in comparison['derived_scores'].items():
                            for dim in scores.keys():
                                self.dimensions.add(dim)
                                
                                # Extract dimension-specific data
                                dim_idx = list(scores.keys()).index(dim)
                                if dim_idx < len(comparison['raw_reasoning']):
                                    dim_comparison = {
                                        'run_id': run_id,
                                        'pair': [aid1, aid2],
                                        'reasoning': comparison['raw_reasoning'][dim_idx],
                                        'raw_confidence': comparison['raw_confidence'][dim_idx] if dim_idx < len(comparison['raw_confidence']) else 0,
                                        'raw_winner': comparison['raw_winner'][dim_idx] if dim_idx < len(comparison['raw_winner']) else 'Tie',
                                        'derived_score_a': comparison['derived_scores'].get(aid1, {}).get(dim, 0.5),
                                        'derived_score_b': comparison['derived_scores'].get(aid2, {}).get(dim, 0.5),
                                    }
                                    self.dimension_data[dim][pair_key].append(dim_comparison)
        
        print(f"Loaded data: {self.n_runs} runs, {len(self.agent_pairs)} agent pairs, {len(self.dimensions)} dimensions")
    
    def print_agent_profiles_summary(self):
        """Print a summary of agent profiles."""
        print("\n" + "="*80)
        print("AGENT PROFILES SUMMARY")
        print("="*80)
        
        for agent_id, profile in self.agent_profiles.items():
            print(f"\n--- Agent {agent_id} ---")
            if isinstance(profile, dict):
                for key, value in profile.items():
                    if key == "primary_goals" and isinstance(value, list):
                        goals = ", ".join([f"{goal[0]}: {goal[1]}" for goal in value])
                        print(f"  Primary Goals: {goals}")
                    elif key in ["communication_style", "behavioral_tendencies"] and isinstance(value, list):
                        print(f"  {key.replace('_', ' ').title()}: {', '.join(value)}")
                    else:
                        print(f"  {key.replace('_', ' ').title()}: {value}")
            else:
                print(f"  Profile: {profile}")
    
    def print_pair_consistency_analysis(self):
        """Analyze and print consistency of evaluations across runs for each pair."""
        print("\n" + "="*80)
        print("PAIR-BY-PAIR CONSISTENCY ANALYSIS")
        print("="*80)
        
        for pair_key in sorted(self.agent_pairs):
            aid1, aid2 = pair_key
            comparisons = self.pair_data[pair_key]
            
            print(f"\n--- Agent {aid1} vs Agent {aid2} ---")
            print(f"Comparisons across {len(comparisons)} runs:")
            
            # Analyze consistency in derived scores
            scores_a = []
            scores_b = []
            
            for comp in comparisons:
                if 'derived_scores' in comp:
                    # Average across all dimensions for this comparison
                    scores_dict_a = comp['derived_scores'].get(aid1, {})
                    scores_dict_b = comp['derived_scores'].get(aid2, {})
                    
                    if scores_dict_a:
                        avg_score_a = np.mean(list(scores_dict_a.values()))
                        scores_a.append(avg_score_a)
                    
                    if scores_dict_b:
                        avg_score_b = np.mean(list(scores_dict_b.values()))
                        scores_b.append(avg_score_b)
            
            if scores_a and scores_b:
                print(f"  Agent {aid1} - Mean: {np.mean(scores_a):.3f}, Std: {np.std(scores_a):.3f}")
                print(f"  Agent {aid2} - Mean: {np.mean(scores_b):.3f}, Std: {np.std(scores_b):.3f}")
                print(f"  Score Difference Range: {np.min(np.array(scores_a) - np.array(scores_b)):.3f} to {np.max(np.array(scores_a) - np.array(scores_b)):.3f}")
            
            # Show wins/losses pattern
            winners = []
            for comp in comparisons:
                if 'raw_winner' in comp and comp['raw_winner']:
                    # Count wins across dimensions
                    win_count = Counter(comp['raw_winner'])
                    if win_count['A'] > win_count['B']:
                        winners.append(aid1)
                    elif win_count['B'] > win_count['A']:
                        winners.append(aid2)
                    else:
                        winners.append('Tie')
            
            if winners:
                win_counts = Counter(winners)
                print(f"  Overall Pattern: Agent {aid1} wins {win_counts[aid1]}, Agent {aid2} wins {win_counts[aid2]}, Ties: {win_counts['Tie']}")
    
    def print_dimension_analysis(self, dimension: str = None):
        """Print detailed analysis for a specific dimension or all dimensions."""
        dimensions_to_analyze = [dimension] if dimension else sorted(self.dimensions)
        
        print("\n" + "="*80)
        print(f"DIMENSION-WISE ANALYSIS")
        print("="*80)
        
        for dim in dimensions_to_analyze:
            print(f"\n--- {dim} ---")
            
            for pair_key in sorted(self.agent_pairs):
                aid1, aid2 = pair_key
                dim_comparisons = self.dimension_data[dim][pair_key]
                
                if not dim_comparisons:
                    continue
                
                print(f"\n  Agent {aid1} vs Agent {aid2}:")
                
                # Analyze patterns in this dimension
                winners = [comp['raw_winner'] for comp in dim_comparisons]
                confidences = [comp['raw_confidence'] for comp in dim_comparisons]
                scores_a = [comp['derived_score_a'] for comp in dim_comparisons]
                scores_b = [comp['derived_score_b'] for comp in dim_comparisons]
                
                win_counts = Counter(winners)
                print(f"    Winners: A={win_counts['A']}, B={win_counts['B']}, Tie={win_counts['Tie']}")
                print(f"    Confidence: Mean={np.mean(confidences):.2f}, Std={np.std(confidences):.2f}")
                print(f"    Agent {aid1} scores: Mean={np.mean(scores_a):.3f}, Std={np.std(scores_a):.3f}")
                print(f"    Agent {aid2} scores: Mean={np.mean(scores_b):.3f}, Std={np.std(scores_b):.3f}")
    
    def print_detailed_reasoning_by_pair(self, agent_a_id: int = None, agent_b_id: int = None):
        """Print detailed LLM reasoning for a specific pair or all pairs."""
        pairs_to_show = []
        
        if agent_a_id is not None and agent_b_id is not None:
            pair_key = tuple(sorted([agent_a_id, agent_b_id]))
            if pair_key in self.agent_pairs:
                pairs_to_show = [pair_key]
            else:
                print(f"Pair ({agent_a_id}, {agent_b_id}) not found in data.")
                return
        else:
            pairs_to_show = sorted(self.agent_pairs)
        
        print("\n" + "="*80)
        print("DETAILED LLM REASONING")
        print("="*80)
        
        for pair_key in pairs_to_show:
            aid1, aid2 = pair_key
            print(f"\n{'='*60}")
            print(f"AGENT {aid1} vs AGENT {aid2}")
            print(f"{'='*60}")
            
            # Show reasoning for each dimension across runs
            for dim in sorted(self.dimensions):
                dim_comparisons = self.dimension_data[dim][pair_key]
                if not dim_comparisons:
                    continue
                
                print(f"\n--- {dim} ---")
                for i, comp in enumerate(dim_comparisons):
                    print(f"\nRun {comp['run_id'] + 1}:")
                    print(f"  Winner: {comp['raw_winner']} (Confidence: {comp['raw_confidence']})")
                    print(f"  Scores: Agent {aid1}={comp['derived_score_a']:.3f}, Agent {aid2}={comp['derived_score_b']:.3f}")
                    print(f"  Reasoning: {comp['reasoning']}")
    
    def generate_consistency_report(self) -> str:
        """Generate a summary report of consistency metrics."""
        report = []
        report.append("CONSISTENCY ANALYSIS SUMMARY")
        report.append("=" * 50)
        
        # Overall statistics
        total_comparisons = sum(len(comps) for comps in self.pair_data.values())
        report.append(f"Total comparisons: {total_comparisons}")
        report.append(f"Runs: {self.n_runs}")
        report.append(f"Agent pairs: {len(self.agent_pairs)}")
        report.append(f"Dimensions: {len(self.dimensions)}")
        
        # Consistency metrics
        report.append("\nCONSISTENCY METRICS BY PAIR:")
        
        for pair_key in sorted(self.agent_pairs):
            aid1, aid2 = pair_key
            comparisons = self.pair_data[pair_key]
            
            # Calculate score variance
            all_score_diffs = []
            for comp in comparisons:
                if 'derived_scores' in comp:
                    scores_a = comp['derived_scores'].get(aid1, {})
                    scores_b = comp['derived_scores'].get(aid2, {})
                    for dim in scores_a:
                        if dim in scores_b:
                            diff = scores_a[dim] - scores_b[dim]
                            all_score_diffs.append(diff)
            
            if all_score_diffs:
                consistency_score = 1.0 - np.std(all_score_diffs)  # Higher = more consistent
                report.append(f"  Agents {aid1} vs {aid2}: Consistency = {consistency_score:.3f}")
        
        return "\n".join(report)
    
    def export_to_excel(self, output_path: str):
        """Export analysis to Excel with multiple sheets."""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # Sheet 1: Summary statistics
            summary_data = []
            for pair_key in sorted(self.agent_pairs):
                aid1, aid2 = pair_key
                comparisons = self.pair_data[pair_key]
                
                for comp in comparisons:
                    if 'derived_scores' in comp:
                        for agent_id, scores in comp['derived_scores'].items():
                            for dim, score in scores.items():
                                summary_data.append({
                                    'Run': comp['run_id'],
                                    'Agent_ID': agent_id,
                                    'Comparison_Partner': aid2 if agent_id == aid1 else aid1,
                                    'Dimension': dim,
                                    'Derived_Score': score,
                                    'Pair': f"{aid1}vs{aid2}"
                                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Detailed reasoning
            reasoning_data = []
            for dim in sorted(self.dimensions):
                for pair_key in sorted(self.agent_pairs):
                    aid1, aid2 = pair_key
                    dim_comparisons = self.dimension_data[dim][pair_key]
                    
                    for comp in dim_comparisons:
                        reasoning_data.append({
                            'Dimension': dim,
                            'Agent_A': aid1,
                            'Agent_B': aid2,
                            'Run': comp['run_id'],
                            'Winner': comp['raw_winner'],
                            'Confidence': comp['raw_confidence'],
                            'Score_A': comp['derived_score_a'],
                            'Score_B': comp['derived_score_b'],
                            'LLM_Reasoning': comp['reasoning']
                        })
            
            reasoning_df = pd.DataFrame(reasoning_data)
            reasoning_df.to_excel(writer, sheet_name='Detailed_Reasoning', index=False)
            
            # Sheet 3: Consistency metrics
            consistency_data = []
            for pair_key in sorted(self.agent_pairs):
                aid1, aid2 = pair_key
                for dim in sorted(self.dimensions):
                    dim_comparisons = self.dimension_data[dim][pair_key]
                    if dim_comparisons:
                        scores_a = [comp['derived_score_a'] for comp in dim_comparisons]
                        scores_b = [comp['derived_score_b'] for comp in dim_comparisons]
                        score_diffs = [a - b for a, b in zip(scores_a, scores_b)]
                        
                        consistency_data.append({
                            'Agent_A': aid1,
                            'Agent_B': aid2,
                            'Dimension': dim,
                            'Mean_Score_A': np.mean(scores_a),
                            'Std_Score_A': np.std(scores_a),
                            'Mean_Score_B': np.mean(scores_b),
                            'Std_Score_B': np.std(scores_b),
                            'Mean_Score_Diff': np.mean(score_diffs),
                            'Std_Score_Diff': np.std(score_diffs),
                            'Consistency_Score': 1.0 - np.std(score_diffs)
                        })
            
            consistency_df = pd.DataFrame(consistency_data)
            consistency_df.to_excel(writer, sheet_name='Consistency_Metrics', index=False)
        
        print(f"Analysis exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze detailed evaluation logs from information sources.")
    parser.add_argument("--data_path", type=str, required=True, 
                       help="Path to the detailed analysis JSON file.")
    parser.add_argument("--view", type=str, 
                       choices=['profiles', 'consistency', 'dimensions', 'reasoning', 'all'], 
                       default='all',
                       help="Type of analysis to display.")
    parser.add_argument("--dimension", type=str, default=None,
                       help="Specific dimension to analyze (for dimension view).")
    parser.add_argument("--agent_a", type=int, default=None,
                       help="First agent ID for detailed pair analysis.")
    parser.add_argument("--agent_b", type=int, default=None,
                       help="Second agent ID for detailed pair analysis.")
    parser.add_argument("--export_excel", type=str, default=None,
                       help="Path to export analysis to Excel file.")
    parser.add_argument("--save_report", type=str, default=None,
                       help="Path to save text report.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        return
    
    print(f"Loading detailed analysis from: {args.data_path}")
    viewer = DetailedAnalysisViewer(args.data_path)
    
    # Display requested views
    if args.view in ['profiles', 'all']:
        viewer.print_agent_profiles_summary()
    
    if args.view in ['consistency', 'all']:
        viewer.print_pair_consistency_analysis()
    
    if args.view in ['dimensions', 'all']:
        viewer.print_dimension_analysis(args.dimension)
    
    if args.view in ['reasoning', 'all']:
        viewer.print_detailed_reasoning_by_pair(args.agent_a, args.agent_b)
    
    # Generate and optionally save report
    if args.save_report or args.view == 'all':
        report = viewer.generate_consistency_report()
        print(f"\n{report}")
        
        if args.save_report:
            with open(args.save_report, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nReport saved to: {args.save_report}")
    
    # Export to Excel if requested
    if args.export_excel:
        viewer.export_to_excel(args.export_excel)


if __name__ == '__main__':
    main() 