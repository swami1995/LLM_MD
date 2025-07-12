import argparse
import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
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
        
    def _derive_winner_from_score(self, score: float) -> str:
        """
        Derive winner from raw score where score is between -5 to 5.
        5 means agent_a won by magnitude 5, -5 means agent_b won by magnitude 5.
        """
        if score > 0:
            return 'A'
        elif score < 0:
            return 'B'
        else:
            return 'Tie'
    
    def _get_winner_info(self, comparison: dict, dim_idx: int) -> Tuple[Optional[str], Optional[float]]:
        """
        Extract winner information from comparison data, handling both raw_winner and raw_scores.
        Returns (winner, raw_score) tuple.
        """
        winner = None
        raw_score = None
        
        # Try to get winner from raw_winner field
        if 'raw_winner' in comparison and comparison['raw_winner'] is not None:
            if isinstance(comparison['raw_winner'], list) and dim_idx < len(comparison['raw_winner']):
                winner = comparison['raw_winner'][dim_idx]
        
        # Try to get score from raw_scores field
        if 'raw_scores' in comparison and comparison['raw_scores'] is not None:
            if isinstance(comparison['raw_scores'], list) and dim_idx < len(comparison['raw_scores']):
                raw_score = comparison['raw_scores'][dim_idx]
                # If we don't have winner but have score, derive it
                if winner is None and raw_score is not None:
                    winner = self._derive_winner_from_score(raw_score)
        
        # Default values if neither is available
        if winner is None:
            winner = 'Tie'
        
        return winner, raw_score
    
    def _get_comparison_winner_pattern(self, comparison: dict) -> str:
        """
        Determine overall winner pattern for a comparison across all dimensions.
        Returns the agent_id that won most dimensions or 'Tie'.
        """
        aid1, aid2 = comparison['pair']
        
        # Try to get winners from raw_winner field
        if 'raw_winner' in comparison and comparison['raw_winner'] is not None:
            if isinstance(comparison['raw_winner'], list):
                win_count = Counter(comparison['raw_winner'])
                if win_count['A'] > win_count['B']:
                    return aid1
                elif win_count['B'] > win_count['A']:
                    return aid2
                else:
                    return 'Tie'
        
        # Try to derive from raw_scores
        elif 'raw_scores' in comparison and comparison['raw_scores'] is not None:
            if isinstance(comparison['raw_scores'], list):
                winners = [self._derive_winner_from_score(score) for score in comparison['raw_scores'] if score is not None]
                if winners:
                    win_count = Counter(winners)
                    if win_count['A'] > win_count['B']:
                        return aid1
                    elif win_count['B'] > win_count['A']:
                        return aid2
                    else:
                        return 'Tie'
        
        return 'Tie'
    
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
                        # Get dimensions from any agent's scores (they should be the same)
                        first_agent_scores = list(comparison['derived_scores'].values())[0]
                        for dim_idx, dim in enumerate(first_agent_scores.keys()):
                            self.dimensions.add(dim)
                            
                            # Extract dimension-specific data (only once per dimension per comparison)
                            if dim_idx < len(comparison['raw_reasoning']):
                                winner, raw_score = self._get_winner_info(comparison, dim_idx)
                                
                                dim_comparison = {
                                    'run_id': run_id,
                                    'pair': [aid1, aid2],
                                    'reasoning': comparison['raw_reasoning'][dim_idx],
                                    'raw_confidence': comparison['raw_confidence'][dim_idx] if dim_idx < len(comparison['raw_confidence']) else 0,
                                    'raw_winner': winner,
                                    'raw_score': raw_score,
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

        report = ""
        
        for agent_id, profile in self.agent_profiles.items():
            print(f"\n--- Agent {agent_id} ---")
            report += f"\n--- Agent {agent_id} ---\n"
            if isinstance(profile, dict):
                for key, value in profile.items():
                    if key == "primary_goals" and isinstance(value, list):
                        goals = ", ".join([f"{goal[0]}: {goal[1]}" for goal in value])
                        print(f"  Primary Goals: {goals}")
                        report += f"  Primary Goals: {goals}\n"
                    elif key in ["communication_style", "behavioral_tendencies"] and isinstance(value, list):
                        print(f"  {key.replace('_', ' ').title()}: {', '.join(value)}")
                        report += f"  {key.replace('_', ' ').title()}: {', '.join(value)}\n"
                    else:
                        print(f"  {key.replace('_', ' ').title()}: {value}")
                        report += f"  {key.replace('_', ' ').title()}: {value}\n"
            else:
                print(f"  Profile: {profile}")
                report += f"  Profile: {profile}\n"

        return report
    
    def print_pair_consistency_analysis(self):
        """Analyze and print consistency of evaluations across runs for each pair."""
        report = ""

        print("\n" + "="*80)
        print("PAIR-BY-PAIR CONSISTENCY ANALYSIS")
        print("="*80)
        report += "\n" + "="*80 + "\n"
        report += "PAIR-BY-PAIR CONSISTENCY ANALYSIS\n"
        report += "="*80 + "\n"
        
        for pair_key in sorted(self.agent_pairs):
            aid1, aid2 = pair_key
            comparisons = self.pair_data[pair_key]
            
            print(f"\n--- Agent {aid1} vs Agent {aid2} ---")
            report += f"\n--- Agent {aid1} vs Agent {aid2} ---\n"
            print(f"Comparisons across {len(comparisons)} runs:")
            report += f"Comparisons across {len(comparisons)} runs:\n"
            
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
                report += f"  Agent {aid1} - Mean: {np.mean(scores_a):.3f}, Std: {np.std(scores_a):.3f}\n"
                report += f"  Agent {aid2} - Mean: {np.mean(scores_b):.3f}, Std: {np.std(scores_b):.3f}\n"
                report += f"  Score Difference Range: {np.min(np.array(scores_a) - np.array(scores_b)):.3f} to {np.max(np.array(scores_a) - np.array(scores_b)):.3f}\n"
            
            # Show wins/losses pattern
            winners = []
            for comp in comparisons:
                winner = self._get_comparison_winner_pattern(comp)
                winners.append(winner)
            
            if winners:
                win_counts = Counter(winners)
                print(f"  Overall Pattern: Agent {aid1} wins {win_counts[aid1]}, Agent {aid2} wins {win_counts[aid2]}, Ties: {win_counts['Tie']}")
                report += f"  Overall Pattern: Agent {aid1} wins {win_counts[aid1]}, Agent {aid2} wins {win_counts[aid2]}, Ties: {win_counts['Tie']}\n"
    
        return report
    
    def print_dimension_analysis(self, dimension: str = None):
        """Print detailed analysis for a specific dimension or all dimensions."""
        dimensions_to_analyze = [dimension] if dimension else sorted(self.dimensions)
        report = ""

        print("\n" + "="*80)
        print(f"DIMENSION-WISE ANALYSIS")
        print("="*80)
        report += "\n" + "="*80 + "\n"
        report += "DIMENSION-WISE ANALYSIS\n"
        report += "="*80 + "\n"
        
        for dim in dimensions_to_analyze:
            print(f"\n--- {dim} ---")
            report += f"\n--- {dim} ---\n"
            
            for pair_key in sorted(self.agent_pairs):
                aid1, aid2 = pair_key
                dim_comparisons = self.dimension_data[dim][pair_key]
                
                if not dim_comparisons:
                    continue
                
                print(f"\n  Agent {aid1} vs Agent {aid2}:")
                report += f"\n  Agent {aid1} vs Agent {aid2}:\n"
                # Analyze patterns in this dimension
                winners = [comp['raw_winner'] for comp in dim_comparisons]
                confidences = [comp['raw_confidence'] for comp in dim_comparisons]
                scores_a = [comp['derived_score_a'] for comp in dim_comparisons]
                scores_b = [comp['derived_score_b'] for comp in dim_comparisons]
                raw_scores = [comp.get('raw_score', 'N/A') for comp in dim_comparisons]
                
                win_counts = Counter(winners)
                print(f"    Winners: A={win_counts['A']}, B={win_counts['B']}, Tie={win_counts['Tie']}")
                report += f"    Winners: A={win_counts['A']}, B={win_counts['B']}, Tie={win_counts['Tie']}\n"
                print(f"    Confidence: Mean={np.mean(confidences):.2f}, Std={np.std(confidences):.2f}")
                report += f"    Confidence: Mean={np.mean(confidences):.2f}, Std={np.std(confidences):.2f}\n"
                print(f"    Agent {aid1} scores: Mean={np.mean(scores_a):.3f}, Std={np.std(scores_a):.3f}")
                report += f"    Agent {aid1} scores: Mean={np.mean(scores_a):.3f}, Std={np.std(scores_a):.3f}\n"
                print(f"    Agent {aid2} scores: Mean={np.mean(scores_b):.3f}, Std={np.std(scores_b):.3f}")
                report += f"    Agent {aid2} scores: Mean={np.mean(scores_b):.3f}, Std={np.std(scores_b):.3f}\n"
                
                # Show raw scores statistics if available
                numeric_raw_scores = [score for score in raw_scores if isinstance(score, (int, float))]
                if numeric_raw_scores:
                    print(f"    Raw Scores: Mean={np.mean(numeric_raw_scores):.2f}, Std={np.std(numeric_raw_scores):.2f}, Range=[{np.min(numeric_raw_scores):.1f}, {np.max(numeric_raw_scores):.1f}]")
                    report += f"    Raw Scores: Mean={np.mean(numeric_raw_scores):.2f}, Std={np.std(numeric_raw_scores):.2f}, Range=[{np.min(numeric_raw_scores):.1f}, {np.max(numeric_raw_scores):.1f}]\n"
                    
        return report
    
    def print_detailed_reasoning_by_pair(self, agent_a_id: int = None, agent_b_id: int = None):
        """Print detailed LLM reasoning for a specific pair or all pairs."""
        pairs_to_show = []
        report = ""

        if agent_a_id is not None and agent_b_id is not None:
            pair_key = tuple(sorted([agent_a_id, agent_b_id]))
            if pair_key in self.agent_pairs:
                pairs_to_show = [pair_key]
            else:
                print(f"Pair ({agent_a_id}, {agent_b_id}) not found in data.")
                report += f"Pair ({agent_a_id}, {agent_b_id}) not found in data.\n"
                return
        else:
            pairs_to_show = sorted(self.agent_pairs)
        
        print("\n" + "="*80)
        print("DETAILED LLM REASONING")
        print("="*80)
        report += "\n" + "="*80 + "\n"
        report += "DETAILED LLM REASONING\n"
        report += "="*80 + "\n"
        
        for pair_key in pairs_to_show:
            aid1, aid2 = pair_key
            print(f"\n{'='*60}")
            print(f"AGENT {aid1} vs AGENT {aid2}")
            print(f"{'='*60}")
            report += f"\n{'='*60}\n"
            report += f"AGENT {aid1} vs AGENT {aid2}\n"
            report += f"{'='*60}\n"
            
            # Show reasoning for each dimension across runs
            for dim in sorted(self.dimensions):
                dim_comparisons = self.dimension_data[dim][pair_key]
                if not dim_comparisons:
                    continue
                
                print(f"\n--- {dim} ---")
                report += f"\n--- {dim} ---\n"
                for i, comp in enumerate(dim_comparisons):
                    print(f"\nRun {comp['run_id'] + 1}:")
                    report += f"\nRun {comp['run_id'] + 1}:\n"
                    
                    # Display winner and raw score information
                    winner_info = f"Winner: {comp['raw_winner']}"
                    if comp.get('raw_score') is not None:
                        winner_info += f" (Raw Score: {comp['raw_score']:.2f})"
                    winner_info += f" (Confidence: {comp['raw_confidence']})"
                    print(f"  {winner_info}")
                    report += f"  {winner_info}\n"
                    print(f"  Scores: Agent {aid1}={comp['derived_score_a']:.3f}, Agent {aid2}={comp['derived_score_b']:.3f}")
                    report += f"  Scores: Agent {aid1}={comp['derived_score_a']:.3f}, Agent {aid2}={comp['derived_score_b']:.3f}\n"
                    print(f"  Reasoning: {comp['reasoning']}")
                    report += f"  Reasoning: {comp['reasoning']}\n"
                    
        return report
    
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
            
            # Add winner pattern analysis
            winners = []
            for comp in comparisons:
                winner = self._get_comparison_winner_pattern(comp)
                winners.append(winner)
            
            if winners:
                win_counts = Counter(winners)
                report.append(f"    Win Pattern: {aid1}={win_counts[aid1]}, {aid2}={win_counts[aid2]}, Ties={win_counts['Tie']}")
        
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
                            'Raw_Score': comp.get('raw_score', 'N/A'),
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


def analyze_and_print_results(data_path: str, output_path: str):
    """
    Loads detailed analysis data, runs all analyses, and saves to a text file.
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    print(f"Loading detailed analysis from: {data_path}")
    viewer = DetailedAnalysisViewer(data_path)
    
    # Display requested views
    full_report = []
    full_report.append(viewer.print_agent_profiles_summary())
    full_report.append(viewer.print_pair_consistency_analysis())
    full_report.append(viewer.print_dimension_analysis(None))
    full_report.append(viewer.print_detailed_reasoning_by_pair(None, None))
    full_report.append(viewer.generate_consistency_report())

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(full_report))

    print(f"Analysis exported to {output_path}")
    return full_report


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