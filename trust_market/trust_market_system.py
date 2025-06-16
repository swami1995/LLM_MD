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


    def add_information_source(self, source, is_primary=False, evaluation_frequency=1, initial_influence=None):
        """
        Add an information source (Auditor, UserRep, etc.) to the system.

        Parameters:
        - source: The information source instance
        - is_primary: Whether this is a primary trust source (defined in market config)
        - evaluation_frequency: How often (in rounds) this source should make investment decisions
        - initial_influence: Optional dict mapping dimensions to initial capacity
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
            elif source.source_type == 'auditor':
                influence_value = 60.0
            else:
                influence_value = 40.0
            initial_influence = {dim: influence_value for dim in source.expertise_dimensions}

        self.trust_market.add_information_source(
            source.source_id, source.source_type, initial_influence, is_primary or is_primary_type
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
    def register_user_representative(self, representative, evaluation_frequency=1, initial_influence=None):
        self.add_information_source(representative, False, evaluation_frequency, initial_influence)

    def register_domain_expert(self, expert, evaluation_frequency=3, initial_influence=None):
        self.add_information_source(expert, False, evaluation_frequency, initial_influence)

    def register_auditor(self, auditor, evaluation_frequency=5, initial_influence=None):
        # Auditors might be considered primary depending on config
        self.add_information_source(auditor, False, evaluation_frequency, initial_influence)

    def register_red_teamer(self, red_teamer, evaluation_frequency=10, initial_influence=None):
        self.add_information_source(red_teamer, False, evaluation_frequency, initial_influence)

    def register_regulator(self, regulator, evaluation_frequency=20, initial_influence=None):
        # Regulators are often primary
        self.add_information_source(regulator, True, evaluation_frequency, initial_influence)
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
            self.trust_market.record_comparative_feedback(agent_a_id, agent_b_id, winners)
        except Exception as e:
            print(f"    Error recording comparative feedback in market for agents {agent_a_id} vs {agent_b_id}: {e}")

        # **Note:** Not passing directly to UserRep. They analyze conversations.


    def run_evaluation_round(self):
        """Run a single evaluation round: simulate, process feedback, evaluate sources."""
        self.evaluation_round += 1
        print(f"\n--- Starting Evaluation Round {self.evaluation_round} ---")

        # 1. Market Maintenance (e.g., decay)
        self.trust_market.increment_evaluation_round() # Handles decay etc.

        # 2. Run Simulation & Get Feedback/Data
        if self.simulation_module:
            print("  Running simulation batch...")
            # try:
            simulation_output = self.simulation_module.multi_turn_dialog()
            print("  Simulation batch finished. Processing output...")
            self._process_simulation_output(simulation_output)
            # except Exception as e:
            #     print(f"  Error during simulation run or processing: {e}")
                # Optionally add more robust error handling (e.g., skip round?)
        else:
            print("  Warning: No simulation module registered. Skipping simulation step.")

        # 3. Run Information Source Evaluations
        print("  Running information source evaluations...")
        self._run_source_evaluations()
        # ipdb.set_trace()  # Debugging breakpoint
        # 4. Display Scores for the Round
        self.display_current_scores()
        print(f"--- Finished Evaluation Round {self.evaluation_round} ---")


    def _process_simulation_output(self, output: Dict):
        """Processes the dictionary returned by simulation_module.multi_turn_dialog."""
        if not output:
            print("  No output received from simulation module.")
            return
        # ipdb.set_trace()  # Debugging breakpoint
        # Process specific ratings
        if output.get("specific_ratings"):
            ratings_batch = output["specific_ratings"]
            # Map conversation data for easy lookup by index (assuming order matches)
            conv_data_map = {i: data for i, data in enumerate(output.get("conversation_data", []))}
            print(f"  Processing {len(ratings_batch)} specific ratings...")

            for i, ratings in enumerate(ratings_batch):
                if i in conv_data_map:
                    conv_info = conv_data_map[i]
                    self.process_user_feedback(
                        user_id=conv_info["user_id"],
                        agent_id=conv_info["agent_id"],
                        ratings=ratings,
                        user_profile_idx=conv_info.get("user_profile_idx")
                    )
                else:
                    print(f"  Warning: Missing conversation data for specific rating index {i}.")

        # Process comparative winners
        comparative = False
        if output.get("comparative_winners"):
            comparisons = output["comparative_winners"]
            print(f"  Processing {len(comparisons)} comparative results...")
            # self.debug_print_comparisons(output)
            comparative = True
            for comparison in comparisons:
                self.process_comparative_feedback(comparison)

        # Record conversation data (distributes to info sources)
        if output.get("conversation_data"):
            conv_data_list = output["conversation_data"]
            print(f"  Recording {len(conv_data_list)} conversations...")
            for conv_data in conv_data_list:
                self.record_conversation_data(conv_data, comparative=comparative)

        print("  Finished processing simulation output.")

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
        """Run evaluations for information sources due in this round."""
        evaluated_sources = []
        for source_id, source in self.information_sources.items():
            frequency = self.source_evaluation_frequency.get(source_id, 1)
            last_evaluated = self.source_last_evaluated.get(source_id, 0) # Use .get for safety

            # Check if it's time to evaluate
            if (self.evaluation_round - last_evaluated) >= frequency:
                print(f"    Evaluating source: {source_id} (Last eval: {last_evaluated}, Freq: {frequency})")
                # try:
                # Let the source decide investments based on its internal logic/data
                investments = source.decide_investments(evaluation_round=self.evaluation_round)

                if investments:
                    print(f"      Source {source_id} decided {len(investments)} investments/divestments.")
                    # Process the investments in the market
                    self.trust_market.process_investments(source_id, investments)
                else:
                    print(f"      Source {source_id} made no investment decisions this round.")

                # Record that this source was evaluated in this round
                self.source_last_evaluated[source_id] = self.evaluation_round
                evaluated_sources.append(source_id)

                # except Exception as e:
                #     print(f"    Error evaluating source {source_id}: {str(e)}")
                #     # Optionally skip updating last_evaluated on error?

        if not evaluated_sources:
            print("    No information sources due for evaluation this round.")
        else:
            print(f"    Evaluated sources: {', '.join(evaluated_sources)}")

    def run_evaluation_rounds(self, num_rounds):
        """
        Run multiple evaluation rounds.

        Parameters:
        - num_rounds: Number of rounds to run
        """
        print(f"\nRunning Trust Market Simulation for {num_rounds} rounds...")
        for i in range(num_rounds):
            self.run_evaluation_round()
            # Optional: Add a small delay or checkpoint logic here
            # time.sleep(1)
        print(f"\nFinished {num_rounds} evaluation rounds.")

    def get_agent_trust_scores(self) -> Dict[int, Dict[str, float]]:
        """Get the current trust scores for all agents known to the simulation."""
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