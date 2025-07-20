import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
import os
from itertools import combinations
import json


class RatingEvolutionTracker:
    """
    Tracks and visualizes the evolution of ratings from different sources
    (user_rep, regulator, auditor, and averaged users) for each agent pair
    and dimension over evaluation rounds.
    
    This tracker stores the most recent rating from each source for each agent pair
    and dimension, and plots how these ratings evolve over time.
    """
    
    def __init__(self, dimensions, agent_ids):
        self.dimensions = dimensions
        self.agent_ids = sorted(agent_ids)  # Sort agent IDs for consistency
        # Generate pairs in sorted order to match how we store them
        self.agent_pairs = [tuple(sorted(pair)) for pair in combinations(self.agent_ids, 2)]
        
        # Track comparative ratings: source -> agent_pair -> dimension -> {round_num: comparative_rating}
        # Comparative rating: positive means first agent in pair is better, negative means second agent is better
        self.source_ratings = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        
        # Track individual user ratings for averaging: agent_pair -> dimension -> user_id -> {round_num: rating}
        self.user_ratings = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        
        # Track averaged user ratings: agent_pair -> dimension -> {round_num: avg_rating}
        self.averaged_user_ratings = defaultdict(lambda: defaultdict(dict))
        
    def update_user_rating(self, user_id, agent_a_id, agent_b_id, dimension, rating, round_num):
        """Update user rating for an agent pair and dimension."""
        pair = tuple(sorted([agent_a_id, agent_b_id]))
        
        # Store the user's rating
        if agent_a_id > agent_b_id: 
            rating = -rating  # Flip rating if agent_a is actually the second agent in the pair
        self.user_ratings[pair][dimension][user_id][round_num] = rating
        
        # Recalculate averaged user rating for this round
        self._update_averaged_user_rating(pair, dimension, round_num)
        
    def _update_averaged_user_rating(self, pair, dimension, round_num):
        """Calculate and store the averaged user rating for a pair/dimension at a given round."""
        # Get all user ratings for this pair/dimension at this round
        current_round_ratings = []
        for user_id, user_data in self.user_ratings[pair][dimension].items():
            if round_num in user_data:
                current_round_ratings.append(user_data[round_num])
        
        # If we have ratings for this round, calculate average
        if current_round_ratings:
            avg_rating = np.mean(current_round_ratings)
            self.averaged_user_ratings[pair][dimension][round_num] = avg_rating
    
    def process_user_evaluations_from_temporal_db(self, temporal_db):
        """Process user evaluations from the trust market's temporal database."""
        user_evaluations = temporal_db.get('user_evaluations', [])
        
        for user_eval in user_evaluations:
            round_num = user_eval.get('evaluation_round', 0)
            agent_a_id = user_eval.get('agent_a_id')
            agent_b_id = user_eval.get('agent_b_id') 
            user_id = user_eval.get('user_id')
            winners = user_eval.get('winners', {})
            
            if agent_a_id is not None and agent_b_id is not None and user_id is not None:
                for dimension, winner_data in winners.items():
                    if dimension in self.dimensions:
                        # Extract rating from winner data
                        if isinstance(winner_data, dict):
                            rating = winner_data.get('rating', 0)
                        elif isinstance(winner_data, tuple) and len(winner_data) >= 2:
                            # Convert winner code to rating
                            winner_code, confidence = winner_data
                            rating = confidence if winner_code == 'A' else (-confidence if winner_code == 'B' else 0)
                        else:
                            rating = 0
                        
                        self.update_user_rating(user_id, agent_a_id, agent_b_id, dimension, rating, round_num)
    
    def process_detailed_evaluations(self, detailed_evaluations):
        """Process detailed evaluation data from the simulation to extract ratings."""
        for round_data in detailed_evaluations:
            round_num = round_data.get('round', 0)
            
            # Process source evaluations
            source_evals = round_data.get('source_evaluations', {})
            print(f"DEBUG: Round {round_num}, processing {len(source_evals)} sources: {list(source_evals.keys())}")
            
            for source_id, source_data in source_evals.items():
                # Check for both 'comparison_details' (old format) and 'comparison_log' (new format)
                comparison_data = source_data.get('comparison_details') or source_data.get('comparison_log', [])
                print(f"DEBUG: Source {source_id} has {len(comparison_data)} comparisons")
                
                if comparison_data:
                    for comparison in comparison_data:
                        pair = comparison.get('pair', [])
                        if len(pair) == 2:
                            # Convert agent IDs to integers if they're strings
                            agent_a_id, agent_b_id = pair
                            if isinstance(agent_a_id, str):
                                agent_a_id = int(agent_a_id)
                            if isinstance(agent_b_id, str):
                                agent_b_id = int(agent_b_id)
                                
                            derived_scores = comparison.get('derived_scores', {})
                            
                            # Extract individual agent ratings from derived_scores
                            # Check both string and int versions of agent IDs in derived_scores
                            str_a_id, str_b_id = str(agent_a_id), str(agent_b_id)
                            
                            # Get the scores for both agents
                            agent_a_scores = None
                            agent_b_scores = None
                            
                            if str_a_id in derived_scores:
                                agent_a_scores = derived_scores[str_a_id]
                            elif agent_a_id in derived_scores:
                                agent_a_scores = derived_scores[agent_a_id]
                                
                            if str_b_id in derived_scores:
                                agent_b_scores = derived_scores[str_b_id]
                            elif agent_b_id in derived_scores:
                                agent_b_scores = derived_scores[agent_b_id]
                            
                            # Store comparative ratings directly from derived_scores
                            if agent_a_scores:
                                for dimension, score in agent_a_scores.items():
                                    if dimension in self.dimensions:
                                        # Convert score to comparative rating: (score - 0.5) * 10
                                        comparative_rating = (score - 0.5) * 10
                                        print(f"DEBUG: {source_id} R{round_num} A{agent_a_id}vs{agent_b_id} {dimension}: {score:.3f} -> {comparative_rating:.2f}")
                                        
                                        # Store as comparative rating for the pair (agent_a vs agent_b)
                                        pair = tuple(sorted([agent_a_id, agent_b_id]))
                                        self.source_ratings[source_id][pair][dimension][round_num] = comparative_rating
                            
                            if agent_b_scores:
                                for dimension, score in agent_b_scores.items():
                                    if dimension in self.dimensions:
                                        # Convert score to comparative rating: (score - 0.5) * 10
                                        comparative_rating = (score - 0.5) * 10
                                        print(f"DEBUG: {source_id} R{round_num} A{agent_b_id}vs{agent_a_id} {dimension}: {score:.3f} -> {comparative_rating:.2f}")
                                        
                                        # Store as comparative rating for the pair (agent_b vs agent_a)
                                        # Since we sort the pair, we need to handle the direction correctly
                                        pair = tuple(sorted([agent_a_id, agent_b_id]))
                                        if agent_b_id == pair[0]:  # agent_b is first in sorted pair
                                            stored_rating = comparative_rating
                                        else:  # agent_b is second in sorted pair, so flip sign
                                            stored_rating = -comparative_rating
                                        
                                        # Only store if we don't already have a rating for this pair/dimension/round
                                        if round_num not in self.source_ratings[source_id][pair][dimension]:
                                            self.source_ratings[source_id][pair][dimension][round_num] = stored_rating
            
            # Process user evaluations if they exist in the detailed_evaluations
            user_evals = round_data.get('user_evaluations', [])
            for user_eval in user_evals:
                if isinstance(user_eval, dict):
                    agent_a_id = user_eval.get('agent_a_id')
                    agent_b_id = user_eval.get('agent_b_id')
                    user_id = user_eval.get('user_id')
                    winners = user_eval.get('winners', {})
                    
                    if agent_a_id is not None and agent_b_id is not None and user_id is not None:
                        for dimension, winner_data in winners.items():
                            if dimension in self.dimensions:
                                # Extract rating from winner data
                                if isinstance(winner_data, dict):
                                    rating = winner_data.get('rating', 0)
                                elif isinstance(winner_data, tuple) and len(winner_data) >= 2:
                                    # Convert winner code to rating
                                    winner_code, confidence = winner_data
                                    rating = confidence if winner_code == 'A' else (-confidence if winner_code == 'B' else 0)
                                else:
                                    rating = 0
                                
                                self.update_user_rating(user_id, agent_a_id, agent_b_id, dimension, rating, round_num)
    
    def create_rating_evolution_plots(self, save_path="figures/rating_evolution", experiment_name="rating_evolution"):
        """Create and save rating evolution plots for all agent pairs and dimensions."""
        os.makedirs(os.path.join(save_path, experiment_name), exist_ok=True)
        
        # Define colors for different sources
        colors = {
            'user_rep_general': 'blue',
            'regulator': 'red', 
            'auditor_main': 'green',
            'averaged_users': 'orange'
        }
        
        print(f"DEBUG: Processing {len(self.agent_pairs)} agent pairs: {self.agent_pairs}")
        
        for pair in self.agent_pairs:
            agent_a_id, agent_b_id = pair
            print(f"DEBUG: Creating plot for agent pair {pair}")
            
            # Calculate subplot layout (aim for roughly square grid)
            n_dims = len(self.dimensions)
            n_cols = int(np.ceil(np.sqrt(n_dims)))
            n_rows = int(np.ceil(n_dims / n_cols))
            
            # Create a large figure with subplots for all dimensions
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            
            # Ensure axes is always a 2D array for consistent indexing
            if n_rows == 1 and n_cols == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            plotted_any_data = False
            
            # Check if this pair has any data at all
            pair_has_any_data = False
            
            # Debug: Let's see what pairs actually exist in the data
            print(f"DEBUG: Available pairs in source_ratings:")
            for source_id in ['user_rep_general', 'regulator', 'auditor_main']:
                if source_id in self.source_ratings:
                    source_pairs = list(self.source_ratings[source_id].keys())
                    print(f"  {source_id}: {source_pairs}")
            
            print(f"DEBUG: Available pairs in averaged_user_ratings: {list(self.averaged_user_ratings.keys())}")
            print(f"DEBUG: Looking for pair: {pair}")
            
            for source_id in ['user_rep_general', 'regulator', 'auditor_main']:
                if (source_id in self.source_ratings and 
                    pair in self.source_ratings[source_id]):
                    pair_has_any_data = True
                    print(f"DEBUG: Found data for {pair} in {source_id}")
                    break
            if (pair in self.averaged_user_ratings and 
                any(self.averaged_user_ratings[pair].values())):
                pair_has_any_data = True
                print(f"DEBUG: Found data for {pair} in averaged_user_ratings")
            
            print(f"DEBUG: Pair {pair} has data: {pair_has_any_data}")
            
            for dim_idx, dimension in enumerate(self.dimensions):
                row = dim_idx // n_cols
                col = dim_idx % n_cols
                ax = axes[row, col]
                
                dimension_has_data = False
                
                # Get all rounds to determine the full range for this dimension
                all_rounds = set()
                for source_id in ['user_rep_general', 'regulator', 'auditor_main']:
                    if (source_id in self.source_ratings and 
                        pair in self.source_ratings[source_id] and 
                        dimension in self.source_ratings[source_id][pair]):
                        all_rounds.update(self.source_ratings[source_id][pair][dimension].keys())
                
                # Also include user evaluation rounds
                if pair in self.averaged_user_ratings and dimension in self.averaged_user_ratings[pair]:
                    all_rounds.update(self.averaged_user_ratings[pair][dimension].keys())
                
                if not all_rounds:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{dimension}', fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                    
                sorted_rounds = sorted(all_rounds)
                
                # Plot comparative ratings from each source
                for source_id in ['user_rep_general', 'regulator', 'auditor_main']:
                    if (source_id in self.source_ratings and 
                        pair in self.source_ratings[source_id] and 
                        dimension in self.source_ratings[source_id][pair]):
                        
                        source_data = self.source_ratings[source_id][pair][dimension]
                        if source_data:
                            # Get the rounds and comparative ratings
                            plot_rounds = []
                            plot_ratings = []
                            
                            for round_num in sorted_rounds:
                                if round_num in source_data:
                                    plot_rounds.append(round_num)
                                    plot_ratings.append(source_data[round_num])
                            
                            if plot_rounds and plot_ratings:
                                ax.plot(plot_rounds, plot_ratings, 
                                       color=colors.get(source_id, 'gray'), 
                                       marker='o', label=f'{source_id}', 
                                       linewidth=1.5, markersize=4)
                                dimension_has_data = True
                
                # Plot averaged user ratings (comparative)
                if pair in self.averaged_user_ratings and dimension in self.averaged_user_ratings[pair]:
                    user_data = self.averaged_user_ratings[pair][dimension]
                    if user_data:
                        plot_rounds = []
                        plot_ratings = []
                        
                        for round_num in sorted_rounds:
                            if round_num in user_data:
                                plot_rounds.append(round_num)
                                plot_ratings.append(user_data[round_num])
                        
                        if plot_rounds and plot_ratings:
                            ax.plot(plot_rounds, plot_ratings, 
                                   color=colors['averaged_users'], 
                                   marker='s', label='Averaged Users', 
                                   linewidth=1.5, markersize=4)
                            dimension_has_data = True
                
                if dimension_has_data:
                    plotted_any_data = True
                
                # Formatting for each subplot
                ax.set_xlabel('Round', fontsize=9)
                ax.set_ylabel('Rating (A - B)', fontsize=9)
                ax.set_title(f'{dimension}', fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
                
                # Set y-axis limits for comparative ratings (-5 to +5 scale)
                ax.set_ylim(-5.5, 5.5)
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Hide empty subplots
            for dim_idx in range(len(self.dimensions), n_rows * n_cols):
                row = dim_idx // n_cols
                col = dim_idx % n_cols
                axes[row, col].set_visible(False)
            
            # Save plot even if no data (but indicate it in the title)
            if not plotted_any_data:
                fig.suptitle(f'Rating Evolution: Agent {agent_a_id} vs Agent {agent_b_id} (NO DATA)\n'
                            f'Positive: Agent {agent_a_id} better, Negative: Agent {agent_b_id} better', 
                            fontsize=14, fontweight='bold', y=0.98)
                print(f"DEBUG: No data found for pair {pair}, saving empty plot")
            else:
                # Add a single legend for the entire figure
                handles, labels = [], []
                for source_id in ['user_rep_general', 'regulator', 'auditor_main', 'averaged_users']:
                    if source_id in colors:
                        if source_id == 'averaged_users':
                            handle = plt.Line2D([0], [0], color=colors[source_id], marker='s', 
                                              linewidth=1.5, markersize=4, label='Averaged Users')
                        else:
                            handle = plt.Line2D([0], [0], color=colors[source_id], marker='o', 
                                              linewidth=1.5, markersize=4, label=source_id)
                        handles.append(handle)
                        labels.append(source_id if source_id != 'averaged_users' else 'Averaged Users')
                
                if handles:
                    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
                              ncol=len(handles), fontsize=10)
                
                # Add overall title and explanation
                fig.suptitle(f'Rating Evolution: Agent {agent_a_id} vs Agent {agent_b_id}\n'
                            f'Positive: Agent {agent_a_id} better, Negative: Agent {agent_b_id} better', 
                            fontsize=14, fontweight='bold', y=0.98)
            
            # Adjust layout to prevent overlap
            plt.tight_layout(rect=[0, 0.02, 1, 0.92])
            
            # Save the plot
            filename = f'rating_evolution_A{agent_a_id}_vs_A{agent_b_id}_all_dimensions.png'
            filepath = os.path.join(save_path, experiment_name, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved rating evolution plot: {filepath}")
            plt.close(fig)
    
    def save_tracking_data(self, save_path="figures/rating_evolution", experiment_name="rating_evolution"):
        """Save the tracking data to JSON for later analysis."""
        os.makedirs(os.path.join(save_path, experiment_name), exist_ok=True)
        
        # Convert defaultdicts and tuple keys to regular dicts for JSON serialization
        def convert_for_json(d):
            if isinstance(d, defaultdict):
                d = dict(d)
            
            result = {}
            for k, v in d.items():
                # Convert tuple keys to strings
                if isinstance(k, tuple):
                    key_str = f"{k[0]}_{k[1]}"
                else:
                    key_str = str(k)
                
                # Recursively convert nested structures
                if isinstance(v, (defaultdict, dict)):
                    result[key_str] = convert_for_json(v)
                else:
                    result[key_str] = v
            return result
        
        tracking_data = {
            'source_ratings': convert_for_json(self.source_ratings),
            'averaged_user_ratings': convert_for_json(self.averaged_user_ratings),
            'user_ratings': convert_for_json(self.user_ratings)
        }
        
        filepath = os.path.join(save_path, experiment_name, 'tracking_data.json')
        with open(filepath, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        print(f"Saved tracking data: {filepath}")
    
    def print_summary(self):
        """Print a summary of the tracked ratings."""
        print("\n=== Rating Evolution Tracker Summary ===")
        print(f"Agent pairs: {self.agent_pairs}")
        print(f"Dimensions: {self.dimensions}")
        print(f"Sources tracked: {list(self.source_ratings.keys())}")
        
        for pair in self.agent_pairs:
            print(f"\nAgent pair {pair}:")
            for dimension in self.dimensions:
                print(f"  {dimension}:")
                # Check comparative ratings for this pair
                for source_id in self.source_ratings:
                    if pair in self.source_ratings[source_id] and dimension in self.source_ratings[source_id][pair]:
                        data = self.source_ratings[source_id][pair][dimension]
                        if data:
                            rounds = sorted(data.keys())
                            print(f"    {source_id}: {len(rounds)} comparative ratings, latest round {max(rounds) if rounds else 'None'}")
                
                if pair in self.averaged_user_ratings and dimension in self.averaged_user_ratings[pair]:
                    data = self.averaged_user_ratings[pair][dimension]
                    if data:
                        rounds = sorted(data.keys())
                        print(f"    averaged_users: {len(rounds)} comparative ratings, latest round {max(rounds) if rounds else 'None'}")


def process_simulation_logs_and_plot(log_filepath, save_path="figures/rating_evolution", experiment_name=None):
    """
    Load simulation logs and create rating evolution plots.
    
    Parameters:
    - log_filepath: Path to the simulation log file (JSON)
    - save_path: Directory to save plots
    - experiment_name: Name for the experiment subfolder
    """
    if experiment_name is None:
        experiment_name = os.path.splitext(os.path.basename(log_filepath))[0]
    
    print(f"Loading simulation data from: {log_filepath}")
    with open(log_filepath, 'r') as f:
        simulation_data = json.load(f)
    
    # Extract configuration
    detailed_evaluations = simulation_data.get('detailed_evaluations', [])
    temporal_db = simulation_data.get('temporal_db', {}) # Assuming temporal_db is part of the simulation_data
    config = simulation_data.get('config', {})
    dimensions = config.get('dimensions', [])
    
    # Extract agent IDs from the data
    agent_trust_scores = simulation_data.get('agent_trust_scores', {})
    agent_ids = list(agent_trust_scores.keys()) if agent_trust_scores else [0, 1, 2]
    
    # Convert string agent IDs to integers if needed
    agent_ids = [int(aid) if isinstance(aid, str) else aid for aid in agent_ids]
    
    print(f"Found {len(agent_ids)} agents and {len(dimensions)} dimensions")
    print(f"Agent IDs: {agent_ids}")
    print(f"Dimensions: {dimensions}")
    
    # Initialize tracker
    tracker = RatingEvolutionTracker(dimensions, agent_ids)
    
    # Process the detailed evaluations
    tracker.process_detailed_evaluations(detailed_evaluations)
    # tracker.process_user_evaluations_from_temporal_db(temporal_db) # Process user evaluations from temporal DB
    
    # Print summary
    tracker.print_summary()
    
    # Create plots
    print(f"\nCreating rating evolution plots...")
    tracker.create_rating_evolution_plots(save_path, experiment_name)
    
    # Save tracking data
    tracker.save_tracking_data(save_path, experiment_name)
    
    print(f"\nRating evolution analysis complete!")
    return tracker


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate rating evolution plots from simulation logs.")
    parser.add_argument("--log_file", type=str, required=True, help="Path to the simulation log JSON file.")
    parser.add_argument("--save_path", type=str, default="figures/rating_evolution", help="Directory to save plots.")
    parser.add_argument("--experiment_name", type=str, help="Name for the experiment subfolder.")
    
    args = parser.parse_args()
    
    process_simulation_logs_and_plot(
        log_filepath=args.log_file,
        save_path=args.save_path,
        experiment_name=args.experiment_name
    ) 