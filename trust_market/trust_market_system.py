from collections import defaultdict
import time
import random
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from trust_market.trust_market import TrustMarket
import numpy as np
import ipdb # Assuming ipdb is used for debugging, otherwise remove


class TrustMarketSystem:
    """
    Main system that orchestrates the trust market, information sources,
    simulation module, and evaluation cycles.
    """

    def __init__(self, config=None):
        """
        Initialize the trust market system.

        Parameters:
        - config: Configuration parameters
        """
        self.config = config or {}

        # --- Core Components ---
        self.simulation_module = None # Will hold the CustomerSupportModel instance
        self.trust_market = TrustMarket(self.config) # Initialize the core market

        # --- Data Storage & Tracking ---
        self.information_sources = {} # Stores registered Auditor, UserRep, etc. instances
        self.trust_market.information_sources = self.information_sources
        self.agent_profiles = {}      # Cache profiles from simulation module
        self.user_profiles = {}       # Cache profiles from simulation module
        self.conversation_histories = {} # Store conversation logs {conv_id: history_list}
        self.user_feedback = defaultdict(list) # Store raw user feedback logs (optional)
        self.evaluation_round = 0

        # --- Information Source Management ---
        self.source_evaluation_frequency = {}  # source_id -> frequency in rounds
        self.source_last_evaluated = defaultdict(int)  # source_id -> last evaluation round

        # --- Configuration from Market ---
        # Get rating scale from market config for potential use here
        self.rating_scale = self.trust_market.rating_scale

        print("TrustMarketSystem initialized.")
        print(f"  - Trust Dimensions: {self.trust_market.dimensions}")
        print(f"  - Rating Scale: {self.rating_scale}")
        print(f"  - Trust Decay Rate: {self.trust_market.config.get('trust_decay_rate', 'N/A')}")

    def register_simulation_module(self, simulation_module):
        """Registers the simulation module and caches its profiles."""
        if self.simulation_module:
            print("Warning: Simulation module already registered. Overwriting.")
        self.simulation_module = simulation_module

        # Cache profiles from the simulation module for easy access
        # Assumes simulation_module has attributes like agent_profiles_all, user_profiles_all
        # The IDs used here (0, 1, 2...) should match the agent/user IDs used in simulation outputs
        if hasattr(simulation_module, 'agent_profiles_all'):
            self.agent_profiles = {i: profile for i, profile in enumerate(simulation_module.agent_profiles_all)}
        if hasattr(simulation_module, 'user_profiles_all'):
            self.user_profiles = {i: profile for i, profile in enumerate(simulation_module.user_profiles_all)}

        print(f"Registered simulation module: {type(simulation_module).__name__}")
        print(f"  - Cached {len(self.agent_profiles)} agent profiles.")
        print(f"  - Cached {len(self.user_profiles)} user profiles.")

        # Pass profiles to any *already registered* information sources that need them
        for source_id, source in self.information_sources.items():
            if hasattr(source, 'add_agent_profile') and callable(source.add_agent_profile):
                print(f"  - Providing profiles to source: {source_id}")
                for agent_id, profile in self.agent_profiles.items():
                    try:
                        source.add_agent_profile(agent_id, profile)
                    except Exception as e:
                        print(f"    Error passing profile to {source_id} for agent {agent_id}: {e}")


    def add_information_source(self, source, is_primary=False, evaluation_frequency=1, initial_influence=None, investment_horizon=1):
        """
        Add an information source (Auditor, UserRep, etc.) to the system.

        Parameters:
        - source: The information source instance
        - is_primary: Whether this is a primary trust source (defined in market config)
        - evaluation_frequency: How often (in rounds) this source should make investment decisions
        - initial_influence: Optional dict mapping dimensions to initial capacity
        - investment_horizon: The number of rounds over which to spread the source's investments.
        """
        if not hasattr(source, 'source_id'):
            print("Error: Information source must have a 'source_id' attribute.")
            return
        if not hasattr(source, 'source_type'):
            print("Error: Information source must have a 'source_type' attribute.")
            return
        if not hasattr(source, 'expertise_dimensions'):
            print("Error: Information source must have 'expertise_dimensions'.")
            return

        source.market = self.trust_market # Give source access to market object
        self.information_sources[source.source_id] = source
        self.source_evaluation_frequency[source.source_id] = evaluation_frequency

        # Determine if source type is considered primary
        is_primary_type = source.source_type in self.trust_market.primary_sources

        # Set initial influence capacity in the market
        if initial_influence is None:
            # Default initial influence based on source type (example)
            if source.source_type == 'regulator':
                influence_value = 10000.0
                initial_influence = {dim: influence_value for dim in source.expertise_dimensions}
            elif source.source_type == 'auditor':
                influence_value_shared = 60.0
                influence_value_full = 100.0
                user_rep_expertise_dimensions = ["Communication_Quality", "Problem_Resolution", "Value_Alignment", "Transparency"]
                initial_influence = {dim: influence_value_shared for dim in source.expertise_dimensions if dim in user_rep_expertise_dimensions}
                initial_influence.update({dim: influence_value_full for dim in source.expertise_dimensions if dim not in user_rep_expertise_dimensions})
            elif source.source_type == 'user_representative':
                influence_value = 40.0
                initial_influence = {dim: influence_value for dim in source.expertise_dimensions}
            else:
                influence_value = 40.0
            
        self.trust_market.add_information_source(
            source.source_id, source.source_type, initial_influence, is_primary or is_primary_type,
            investment_horizon=investment_horizon
        )
        print(f"Registered Information Source: {source.source_id} (Type: {source.source_type}, Freq: {evaluation_frequency}, Primary: {is_primary or is_primary_type})")

        # If profiles already loaded, pass them now
        if self.agent_profiles and hasattr(source, 'add_agent_profile') and callable(source.add_agent_profile):
            print(f"  - Providing profiles to newly registered source: {source.source_id}")
            for agent_id, profile in self.agent_profiles.items():
                try:
                    source.add_agent_profile(agent_id, profile)
                except Exception as e:
                    print(f"    Error passing profile to {source.source_id} for agent {agent_id}: {e}")


    # --- Registration Helpers (Syntactic Sugar) ---
    def register_user_representative(self, representative, evaluation_frequency=1, initial_influence=None, investment_horizon=1):
        self.add_information_source(representative, False, evaluation_frequency, initial_influence, investment_horizon)

    def register_domain_expert(self, expert, evaluation_frequency=3, initial_influence=None, investment_horizon=1):
        self.add_information_source(expert, False, evaluation_frequency, initial_influence, investment_horizon)

    def register_auditor(self, auditor, evaluation_frequency=5, initial_influence=None, investment_horizon=1):
        # Auditors might be considered primary depending on config
        self.add_information_source(auditor, False, evaluation_frequency, initial_influence, investment_horizon)

    def register_red_teamer(self, red_teamer, evaluation_frequency=10, initial_influence=None, investment_horizon=1):
        self.add_information_source(red_teamer, False, evaluation_frequency, initial_influence, investment_horizon)

    def register_regulator(self, regulator, evaluation_frequency=20, initial_influence=None, investment_horizon=1):
        # Regulators are often primary
        self.add_information_source(regulator, True, evaluation_frequency, initial_influence, investment_horizon)
    # --- End Registration Helpers ---


    def record_conversation_data(self, conv_data: Dict, comparative=False):
        """
        Stores conversation history and distributes it to information sources.

        Parameters:
        - conv_data: Dictionary containing conversation details from simulation output.
                    Expected keys: 'user_id', 'agent_id', 'history',
                    Optional: 'conversation_id', 'agent_b_id', 'history_b'
        """
        conv_id = conv_data.get('conversation_id', f"r{self.evaluation_round}_u{conv_data['user_id']}_a{conv_data['agent_id']}")
        history = conv_data.get('history', [])

        if not history:
            print(f"Warning: Received empty history for conversation {conv_id}.")
            return

        self.conversation_histories[conv_id] = history
        # print(f"  Recorded conversation {conv_id} (User: {conv_data['user_id']}, Agent: {conv_data['agent_id']}).")

        # Distribute to information sources
        for source_id, source in self.information_sources.items():
            if hasattr(source, 'add_conversation') and callable(source.add_conversation):
                try:
                    # Pass necessary info. Sources might need to handle comparative data.
                    # Simple approach: pass primary history and IDs. Sources can fetch profiles if needed.
                    source.add_conversation(
                        conversation_history=history,
                        user_id=conv_data['user_id'],
                        agent_id=conv_data['agent_id']
                        # Optionally pass agent_b_id, history_b if source handles it
                    )
                    if comparative and 'agent_b_id' in conv_data:
                        source.add_conversation(
                            conversation_history=conv_data['history_b'],
                            user_id=conv_data['user_id'],
                            agent_id=conv_data['agent_b_id']
                        )

                except Exception as e:
                    print(f"    Error passing conversation {conv_id} to source {source_id}: {e}")


    def process_user_feedback(self, user_id, agent_id, ratings: Dict[str, int], user_profile_idx=None):
        """
        Processes direct user feedback (specific ratings) and sends it to the TrustMarket.

        Parameters:
        - user_id: User providing feedback
        - agent_id: Agent being rated
        - ratings: Dict mapping dimensions to integer ratings (e.g., 1-5)
        - user_profile_idx: Optional index of the user's profile
        """
        print(f"  Processing User Feedback: User {user_id} rated Agent {agent_id}: {ratings}")
        # Store raw feedback log (optional)
        self.user_feedback[user_id].append({
            "agent_id": agent_id, "ratings": ratings,
            "user_profile_idx": user_profile_idx, "round": self.evaluation_round,
            "timestamp": time.time()
        })

        # Send ratings to the market for score updates
        try:
            self.trust_market.record_user_feedback(user_id, agent_id, ratings)
        except Exception as e:
            print(f"    Error recording user feedback in market for agent {agent_id}: {e}")

        # **Note:** We are NOT directly passing this feedback to UserRepresentatives here.
        # They receive conversation data via record_conversation_data and decide
        # whether to use direct feedback (if available) or perform their own LLM evals.


    def process_comparative_feedback(self, comparison_data: Dict):
        """
        Processes comparative user feedback and sends it to the TrustMarket.

        Parameters:
        - comparison_data: Dict with 'agent_a_id', 'agent_b_id', 'user_id', 'winners'
        """
        agent_a_id = comparison_data['agent_a_id']
        agent_b_id = comparison_data['agent_b_id']
        user_id = comparison_data['user_id']
        winners = comparison_data['winners'] # {dim: ('A'/'B'/'Tie', 1-5)}
        print(f"  Processing Comparative Feedback: User {user_id} compared A={agent_a_id}, B={agent_b_id}") # Add winner details?

        # Send comparison results to the market for score updates
        try:
            # Here, we need to pass the full comparison_data dict
            self.trust_market.record_comparative_feedback(
                agent_a_id,
                agent_b_id,
                winners=winners,
                # Pass the raw reasoning and confidence if available
                raw_comparison_details=comparison_data 
            )
        except Exception as e:
            print(f"    Error recording comparative feedback in market for agents {agent_a_id} vs {agent_b_id}: {e}")
        
        # Also pass the full comparison data to information sources that might use it
        for source in self.information_sources.values():
            if hasattr(source, 'add_comparison_feedback') and callable(source.add_comparison_feedback):
                source.add_comparison_feedback(comparison_data)

    def run_evaluation_round(self):
        """
        Runs one full evaluation round, including simulation, feedback processing,
        and information source evaluations.
        """
        print(f"\n{'='*20} Starting Evaluation Round: {self.evaluation_round} {'='*20}")

        if not self.simulation_module:
            print("Error: No simulation module registered. Cannot run evaluation round.")
            return {}, [], {}

        # 1. Run simulation to get user-agent interactions
        print("\n--- 1. Running User-Agent Interaction Simulation ---")
        simulation_output = self.simulation_module.multi_turn_dialog(
            evaluation_round=self.evaluation_round
        )

        # 2. Process simulation output
        print("\n--- 2. Processing Simulation Output & User Feedback ---")
        self._process_simulation_output(simulation_output)

        # 3. Trigger information source evaluations
        print("\n--- 3. Running Information Source Evaluations ---")
        detailed_source_evals = self._run_source_evaluations()

        # 4. Apply any pending spread investments from sources with investment horizons
        self.trust_market.apply_spread_investments()
        
        # 5. Apply trust decay at the end of the round
        # self.trust_market.apply_trust_decay().

        self.display_current_scores()

        # 6. Increment evaluation round counter
        self.trust_market.increment_evaluation_round()
        self.evaluation_round += 1

        print(f"\n{'='*20} Finished Evaluation Round: {self.evaluation_round - 1} {'='*20}")
        # Return the detailed evaluations and simulation output to be saved by the main loop
        return detailed_source_evals, simulation_output.get("comparative_winners", []), simulation_output

    def run_evaluation_rounds(self, num_rounds):
        """
        Run the simulation for a specified number of evaluation rounds.
        """
        all_simulation_outputs = []
        all_detailed_evaluations = []
        for i in range(num_rounds):
            detailed_source_evals, user_evals, simulation_output = self.run_evaluation_round()
            all_detailed_evaluations.append({
                "round": self.evaluation_round - 1, # Current round (after increment)
                "source_evaluations": detailed_source_evals,
                "user_evaluations": user_evals
            })
            # Store the complete simulation output
            simulation_output_with_round = simulation_output.copy() if simulation_output else {}
            simulation_output_with_round["round"] = self.evaluation_round - 1
            all_simulation_outputs.append(simulation_output_with_round)
        return all_simulation_outputs, all_detailed_evaluations

    def _process_simulation_output(self, output: Dict):
        """
        Internal helper to process the output from the simulation module.
        It handles recording conversations and processing both specific and
        comparative feedback.
        """
        # --- Record all conversation data first ---
        conversation_data = output.get("conversation_data", [])
        if not conversation_data:
            print("  No conversation data produced in this round.")
        
        is_comparative = "comparative_winners" in output and output["comparative_winners"] is not None
        for conv_data in conversation_data:
            self.record_conversation_data(conv_data, comparative=is_comparative)

        # --- Process specific ratings if present ---
        specific_ratings_batch = output.get("specific_ratings")
        if specific_ratings_batch:
            # The structure is: list of dicts, where each dict is {dim: rating}
            # We need to associate each rating dict with the correct user and agent
            for i, ratings in enumerate(specific_ratings_batch):
                # We assume the order matches the conversation_data list
                if i < len(conversation_data):
                    user_id = conversation_data[i]['user_id']
                    agent_id = conversation_data[i]['agent_id']
                    user_profile_idx = conversation_data[i]['user_profile_idx']
                    self.process_user_feedback(user_id, agent_id, ratings, user_profile_idx)
                else:
                    print(f"  Warning: Mismatch between ratings batch size and conversation data size.")

        # --- Process comparative feedback if present ---
        comparative_winners_batch = output.get("comparative_winners")
        if comparative_winners_batch:
            # The structure is: list of dicts with all comparison details
            # self.debug_print_comparisons(comparative_winners_batch) # Optional debug print
            for comparison_data in comparative_winners_batch:
                self.process_comparative_feedback(comparison_data)

    def debug_print_comparisons(self, output):
        comparisons = output.get("comparative_winners", [])
        conv_data = output.get("conversation_data", [])
        for i, (comparisons, conv_data) in enumerate(zip(comparisons, conv_data)):
            print(f"  Comparison {i}: User {conv_data['user_id']} compared A={comparisons['agent_a_id']}, B={comparisons['agent_b_id']}")
            print("    Winners:", comparisons['winners'])
            print("\n" + "-"*50 + "\n")
            print("    CONVERSATION HISTORY A:")
            # Loop through the conversation history
            for turn in conv_data.get('history', []):
                if 'agent' in turn:
                    print(f"      AGENT : {turn['agent']}")
                if 'user' in turn:
                    print(f"      USER  : {turn['user']}")
            print("\n" + "-"*50 + "\n")
            print("    CONVERSATION HISTORY B:")
            # Loop through the conversation history
            for turn in conv_data.get('history_b', []):
                if 'agent' in turn:
                    print(f"      AGENT : {turn['agent']}")
                if 'user' in turn:
                    print(f"      USER  : {turn['user']}")
            print("\n" + "-"*50 + "\n")
        ipdb.set_trace()  # Debugging breakpoint


    def _run_source_evaluations(self):
        """
        Triggers evaluations for all registered information sources based on
        their specified frequency.
        """
        detailed_evaluations = {}
        for source_id, source in self.information_sources.items():
            # Check if it's time for this source to evaluate
            eval_frequency = self.source_evaluation_frequency.get(source_id, 1)
            if (self.evaluation_round+1) % eval_frequency == 0:
                print(f"\n>>> Evaluating source: {source_id} (Round {self.evaluation_round})")
                try:
                    # Get investment decisions from the source
                    # The source's `decide_investments` should return a list of investment tuples
                    # And optionally, detailed analysis data if requested
                    investments, analysis_data = source.decide_investments(
                        evaluation_round=self.evaluation_round,
                        analysis_mode=True, # Ensure analysis data is generated
                        detailed_analysis=True # Trigger detailed logging
                    )
                    detailed_evaluations[source_id] = analysis_data
                    
                    if investments:
                        print(f"  Source {source_id} proposed {len(investments)} investments.")
                        # Process these investments in the trust market
                        self.trust_market.process_investments(source_id, investments)
                    else:
                        print(f"  Source {source_id} proposed no investments.")

                except Exception as e:
                    print(f"  !! Error during evaluation of source {source_id}: {e}")
                    # Optionally, add traceback for debugging
                    import traceback
                    traceback.print_exc()
        return detailed_evaluations

    def get_agent_trust_scores(self) -> Dict[int, Dict[str, float]]:
        """
        Get the current trust scores for all agents known to the simulation.
        """
        agent_scores = {}
        # Iterate through agent IDs known from the simulation setup
        known_agent_ids = list(self.agent_profiles.keys())

        print(f"Fetching trust scores for {len(known_agent_ids)} known agents...")
        for agent_id in known_agent_ids:
            try:
                # Fetch scores directly from the market
                scores = self.trust_market.get_agent_trust(agent_id)
                # get_agent_trust returns a copy, safe to use directly
                agent_scores[agent_id] = scores
            except Exception as e:
                print(f"  Error fetching trust score for agent {agent_id}: {e}")
                # Optionally return default scores if agent not in market?
                # agent_scores[agent_id] = {dim: 0.5 for dim in self.trust_market.dimensions}

        return agent_scores

    def display_current_scores(self):
        """Fetches and prints the current trust scores for all known agents."""
        print(f"\n--- Trust Scores at end of Round {self.evaluation_round} ---")
        current_scores = self.get_agent_trust_scores()
        if not current_scores:
            print("  No scores available.")
            return

        for agent_id, scores in sorted(current_scores.items()): # Sort by agent ID
            profile = self.agent_profiles.get(agent_id, {})
            goals = profile.get("primary_goals", [("?", "Unknown")])
            goals_text = goals[0][1] if goals else "Unknown"
            print(f"Agent {agent_id} (Goal: {goals_text}):")

            score_strs = []
            sorted_dims = sorted(scores.keys())
            for dim in sorted_dims:
                score = scores[dim]
                score_strs.append(f"{dim}: {score:.3f}")

            # Print scores in rows
            for i in range(0, len(score_strs), 3):
                print("  " + ", ".join(score_strs[i:i+3]))
        print("---------------------------------")


    def get_agent_permissions(self, agent_id):
        """Determine permissions based on trust scores (delegated to market)."""
        return self.trust_market.get_agent_permissions(agent_id)

    def get_source_investments(self, source_id):
        """Get investments made by a source (delegated to market)."""
        return self.trust_market.get_source_endorsements(source_id)

    def summarize_market_state(self):
        """Get a summary of the current market state (delegated to market)."""
        summary = self.trust_market.summarize_market_state(self.information_sources)
        # Add system-level info
        summary['system_evaluation_round'] = self.evaluation_round
        summary['num_registered_info_sources'] = len(self.information_sources)
        summary['num_known_agents'] = len(self.agent_profiles)
        summary['num_known_users'] = len(self.user_profiles)
        return summary

    # --- Visualization Methods ---
    def visualize_trust_scores(self, agents=None, dimensions=None,
                            start_round=None, end_round=None):
        """Visualize trust scores over time (delegated to market)."""
        return self.trust_market.visualize_trust_scores(
            agents, dimensions, start_round, end_round
        )

    def visualize_source_performance(self, sources=None, dimensions=None,
                                start_round=None, end_round=None):
        """Visualize source performance over time (delegated to market)."""
        return self.trust_market.visualize_source_performance(
            sources, dimensions, start_round, end_round
        )