import argparse
import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import os
from datetime import datetime

class UserDetailedAnalysisViewer:
    """
    A viewer for detailed user evaluation logs that presents 
    pairwise comparison data, focusing on consistency and patterns.
    """
    
    def __init__(self, data_path: str):
        """Load and organize the detailed analysis data."""
        with open(data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        self.n_runs = len(self.raw_data)
        self.agent_pairs = set()
        self.dimensions = set()
        self.agent_profiles = {}
        
        self.pair_data = defaultdict(list)
        self.dimension_data = defaultdict(lambda: defaultdict(list))
        
        self._organize_data()
        
    def _derive_winner_from_score(self, score: float) -> str:
        """
        Derive winner from raw score where score is between -5 to 5.
        A positive score means agent_a won, a negative score means agent_b won.
        """
        if score > 0:
            return 'A'
        elif score < 0:
            return 'B'
        else:
            return 'Tie'

    def _get_comparison_winner_pattern(self, comparison: dict) -> str:
        """
        Determine overall winner pattern for a comparison across all dimensions
        by deriving winners from scores.
        """
        aid1, aid2 = comparison['pair']
        
        if 'derived_scores' in comparison and isinstance(comparison['derived_scores'], dict):
            winners = [self._derive_winner_from_score(score) for score in comparison['derived_scores'].values()]
            win_count = Counter(winners)
            if win_count['A'] > win_count['B']:
                return str(aid1)
            elif win_count['B'] > win_count['A']:
                return str(aid2)
        return 'Tie'

    def _organize_data(self):
        """Organize raw data into convenient structures."""
        for run_data in self.raw_data:
            run_id = run_data['run_id']
            
            for comparison in run_data['comparison_details']:
                aid1, aid2 = comparison['pair']
                pair_key = tuple(sorted([aid1, aid2]))
                self.agent_pairs.add(pair_key)
                
                if aid1 not in self.agent_profiles:
                    self.agent_profiles[aid1] = comparison['agent_a_profile']
                if aid2 not in self.agent_profiles:
                    self.agent_profiles[aid2] = comparison['agent_b_profile']
                
                comparison_with_run = comparison.copy()
                comparison_with_run['run_id'] = run_id
                self.pair_data[pair_key].append(comparison_with_run)
                
                # 'derived_scores' is now a dict of {dimension: rating}
                if 'derived_scores' in comparison:
                    for dim, score in comparison['derived_scores'].items():
                        self.dimensions.add(dim)
                        
                        winner = self._derive_winner_from_score(score)
                        
                        # Normalize winner and score based on sorted pair key
                        normalized_winner = winner
                        normalized_score = score
                        if aid1 > aid2: # Original pair was swapped, e.g., [2, 0]
                            normalized_score = -score
                            if winner == 'A':
                                normalized_winner = 'B'
                            elif winner == 'B':
                                normalized_winner = 'A'

                        dim_reasoning = comparison.get('reasoning', {}).get(dim, "N/A")
                        dim_confidence = comparison.get('confidences', {}).get(dim, 0)
                        
                        dim_comparison = {
                            'run_id': run_id,
                            'pair': [aid1, aid2],
                            'reasoning': dim_reasoning,
                            'raw_confidence': dim_confidence,
                            'raw_winner': normalized_winner,
                            'raw_score': normalized_score,
                        }
                        self.dimension_data[dim][pair_key].append(dim_comparison)
        
        print(f"Loaded data: {self.n_runs} runs, {len(self.agent_pairs)} agent pairs, {len(self.dimensions)} dimensions")

    def print_agent_profiles_summary(self):
        """Print a summary of agent profiles."""
        report = []
        report.append("\n" + "="*80)
        report.append("AGENT PROFILES SUMMARY")
        report.append("="*80)
        
        for agent_id, profile in sorted(self.agent_profiles.items()):
            report.append(f"\n--- Agent {agent_id} ---")
            if isinstance(profile, dict):
                for key, value in profile.items():
                    if key == "primary_goals" and isinstance(value, list):
                        goals = ", ".join([f"{goal[0]}: {goal[1]}" for goal in value])
                        report.append(f"  Primary Goals: {goals}")
                    elif key in ["communication_style", "behavioral_tendencies"] and isinstance(value, list):
                        report.append(f"  {key.replace('_', ' ').title()}: {', '.join(value)}")
                    else:
                        report.append(f"  {key.replace('_', ' ').title()}: {value}")
            else:
                report.append(f"  Profile: {profile}")
        return "\n".join(report)

    def print_pair_consistency_analysis(self):
        """Analyze and print consistency of evaluations across runs for each pair."""
        report = []
        report.append("\n" + "="*80)
        report.append("PAIR-BY-PAIR CONSISTENCY ANALYSIS")
        report.append("="*80)
        
        for pair_key in sorted(self.agent_pairs):
            aid1, aid2 = pair_key
            comparisons = self.pair_data[pair_key]
            
            report.append(f"\n--- Agent {aid1} vs Agent {aid2} ---")
            report.append(f"Comparisons across {len(comparisons)} runs:")
            
            winners = [self._get_comparison_winner_pattern(comp) for comp in comparisons]
            if winners:
                win_counts = Counter(winners)
                report.append(f"  Overall Pattern: Agent {aid1} wins {win_counts.get(str(aid1), 0)}, Agent {aid2} wins {win_counts.get(str(aid2), 0)}, Ties: {win_counts.get('Tie', 0)}")
        return "\n".join(report)

    def print_dimension_analysis(self):
        """Print detailed analysis for all dimensions."""
        report = []
        report.append("\n" + "="*80)
        report.append(f"DIMENSION-WISE ANALYSIS")
        report.append("="*80)
        
        for dim in sorted(self.dimensions):
            report.append(f"\n--- {dim} ---")
            
            for pair_key in sorted(self.agent_pairs):
                aid1, aid2 = pair_key
                dim_comparisons = self.dimension_data[dim][pair_key]
                
                if not dim_comparisons: continue
                
                report.append(f"\n  Agent {aid1} vs Agent {aid2}:")
                
                winners = [comp['raw_winner'] for comp in dim_comparisons]
                confidences = [comp['raw_confidence'] for comp in dim_comparisons]
                raw_scores = [comp.get('raw_score') for comp in dim_comparisons if comp.get('raw_score') is not None]

                win_counts = Counter(winners)
                report.append(f"    Winners: A={win_counts.get('A', 0)}, B={win_counts.get('B', 0)}, Tie={win_counts.get('Tie', 0)}")
                report.append(f"    Confidence: Mean={np.mean(confidences):.2f}, Std={np.std(confidences):.2f}")
                
                if raw_scores:
                    report.append(f"    Raw Scores: Mean={np.mean(raw_scores):.2f}, Std={np.std(raw_scores):.2f}, Range=[{np.min(raw_scores):.1f}, {np.max(raw_scores):.1f}]")
        return "\n".join(report)

    def print_detailed_reasoning_by_pair(self):
        """Print detailed LLM reasoning for all pairs."""
        report = []
        report.append("\n" + "="*80)
        report.append("DETAILED LLM REASONING")
        report.append("="*80)
        
        for pair_key in sorted(self.agent_pairs):
            aid1, aid2 = pair_key
            report.append(f"\n{'='*60}")
            report.append(f"AGENT {aid1} vs AGENT {aid2}")
            report.append(f"{'='*60}")
            
            for dim in sorted(self.dimensions):
                dim_comparisons = self.dimension_data[dim].get(pair_key, [])
                if not dim_comparisons: continue
                
                report.append(f"\n--- {dim} ---")
                for i, comp in enumerate(dim_comparisons):
                    report.append(f"\nRun {comp['run_id'] + 1}:")
                    
                    winner_info = f"Winner: {comp['raw_winner']}"
                    if comp.get('raw_score') is not None:
                        winner_info += f" (Raw Score: {comp['raw_score']:.2f})"
                    winner_info += f" (Confidence: {comp['raw_confidence']})"
                    
                    report.append(f"  {winner_info}")
                    report.append(f"  Reasoning: {comp['reasoning']}")
        return "\n".join(report)

def analyze_and_print_results(data_path: str, output_path: str):
    """
    Loads user evaluation data, runs all analyses, and saves to a text file.
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    print(f"Loading detailed user analysis from: {data_path}")
    viewer = UserDetailedAnalysisViewer(data_path)
    
    full_report = []

    # Generate all analysis sections
    full_report.append(viewer.print_agent_profiles_summary())
    full_report.append(viewer.print_pair_consistency_analysis())
    full_report.append(viewer.print_dimension_analysis())
    full_report.append(viewer.print_detailed_reasoning_by_pair())
    
    # Combine and save report
    final_text = "\n".join(full_report)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_text)
    
    print(f"\nUser evaluation analysis report saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze detailed user evaluation logs.")
    parser.add_argument("--data_path", type=str, required=True, 
                       help="Path to the detailed user analysis JSON file.")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save the text report.")
    
    args = parser.parse_args()
    
    analyze_and_print_results(args.data_path, args.output_path) 