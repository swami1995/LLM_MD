from collections import defaultdict
import time
import random
from typing import Dict, List, Tuple, Set, Optional, Union, Any
# Make TrustMarket import relative if it's in the same directory/package
try:
    from .trust_market import TrustMarket # Use relative import if applicable
except ImportError:
    from trust_market import TrustMarket # Fallback to direct import
import numpy as np

# Import InformationSource types
from auditor import Auditor, AuditorWithProfileAnalysis # Keep base if needed, but use enhanced
from user_rep import UserRepresentative, UserRepresentativeWithHolisticEvaluation # Keep base if needed, but use enhanced
# Import others if you have them (DomainExpert, RedTeamer, Regulator)
# from domain_expert import DomainExpert
# from red_teamer import RedTeamer
# from regulator import Regulator

# Import the simulation model
from info_agent import CustomerSupportModel


class TrustMarketSystem:
    """
    Main system orchestrating the trust market, simulation, information sources, and evaluation cycles.
    """

    def __init__(self, config=None):
        """
        Initialize the trust market system.

        Parameters:
        - config: Configuration parameters for the market
        """
        self.config = config or {}

        # --- Trust Market Configuration ---
        if 'dimensions' not in self.config:
            self.config['dimensions'] = [
                "Factual_Correctness", "Process_Reliability", "Value_Alignment",
                "Communication_Quality", "Problem_Resolution", "Safety_Security",
                "Transparency", "Adaptability", "Trust_Calibration", "Manipulation_Resistance"
            ]
        if 'dimension_weights' not in self.config:
            self.config['dimension_weights'] = {dim: 1.0 for dim in self.config['dimensions']}
        self.config['primary_sources'] = self.config.get('primary_sources', ['user_feedback']) # User feedback is primary
        self.config['primary_source_weight'] = self.config.get('primary_source_weight', 2.0)
        self.config['secondary_source_weight'] = self.config.get('secondary_source_weight', 1.0)
        # Get rating scale from config, default to 5
        self.rating_scale = self.config.get('rating_scale', 5)


        # --- Initialize Trust Market ---
        self.trust_market = TrustMarket(self.config)

        # --- Simulation Module ---
        self.simulation_module: Optional[CustomerSupportModel] = None # To hold the CustomerSupportModel instance

        # --- Information Sources ---
        self.information_sources: Dict[str, Any] = {} # Store registered source objects
        self.source_evaluation_frequency: Dict[str, int] = {} # source_id -> frequency in rounds
        self.source_last_evaluated: Dict[str, int] = defaultdict(int) # source_id -> last evaluation round

        # --- Data Tracking ---
        self.conversation_histories = {} # Store histories for sources {conv_id: data}
        self.user_feedback_log = [] # Log raw user feedback tuples
        self.comparative_feedback_log = [] # Log comparative feedback tuples

        # --- State ---
        self.evaluation_round = 0

    def register_simulation_module(self, sim_module: CustomerSupportModel):
        """Registers the simulation module (e.g., CustomerSupportModel)."""
        self.simulation_module = sim_module
        print("Simulation module registered with TrustMarketSystem.")
        # Optionally, pass market reference to sim module if needed (currently not needed)
        # sim_module.register_market(self.trust_market)


    def add_information_source(self, source: Any, is_primary: bool = False,
                               evaluation_frequency: int = 1, initial_influence: Optional[Dict[str, float]] = None):
        """
        Add an information source to the system.

        Parameters:
        - source: The information source object (must have source_id, source_type, expertise_dimensions)
        - is_primary: Whether this is a primary trust source (influences weighting in market)
        - evaluation_frequency: How often (in rounds) this source should make investment decisions
        - initial_influence: Dict mapping dimensions to initial influence capacity (default: 100 per expert dim)
        """
        if not hasattr(source, 'source_id') or not hasattr(source, 'source_type') or not hasattr(source, 'expertise_dimensions'):
             raise ValueError("Information source must have source_id, source_type, and expertise_dimensions attributes.")

        source.market = self.trust_market # Give source access to the market object
        self.information_sources[source.source_id] = source
        self.source_evaluation_frequency[source.source_id] = evaluation_frequency

        # Determine initial influence
        if initial_influence is None:
            initial_influence = {dim: 100.0 for dim in source.expertise_dimensions if dim in self.config['dimensions']}

        # Add to market
        self.trust_market.add_information_source(
            source.source_id, source.source_type, initial_influence, is_primary
        )
        print(f"Added {source.source_type} source: {source.source_id} (Freq: {evaluation_frequency})")


    # Convenience methods for registering specific source types
    def register_user_representative(self, representative: UserRepresentative, evaluation_frequency=1, initial_influence=None):
        self.add_information_source(representative, False, evaluation_frequency, initial_influence)

    # def register_domain_expert(self, expert: DomainExpert, evaluation_frequency=3, initial_influence=None):
    #     self.add_information_source(expert, False, evaluation_frequency, initial_influence)

    def register_auditor(self, auditor: Auditor, evaluation_frequency=5, initial_influence=None):
         # Auditors are often considered highly reliable, potentially primary or near-primary
        is_primary_auditor = self.config.get('auditor_is_primary', False)
        self.add_information_source(auditor, is_primary_auditor, evaluation_frequency, initial_influence)


    # def register_red_teamer(self, red_teamer: RedTeamer, evaluation_frequency=10, initial_influence=None):
    #     self.add_information_source(red_teamer, False, evaluation_frequency, initial_influence)

    # def register_regulator(self, regulator: Regulator, evaluation_frequency=20, initial_influence=None):
    #     self.add_information_source(regulator, True, evaluation_frequency, initial_influence) # Regulators usually primary


    # This method is being repurposed - original intent was unclear.
    # Now it just stores the conversation for later use by info sources.
    def record_conversation_data(self, conversation_id, user_id, agent_id, history):
        """
        Record conversation data generated during the simulation round.

        Parameters:
        - conversation_id: Unique conversation identifier (can be generated if None)
        - user_id: User participating
        - agent_id: Agent participating
        - history: Conversation history list of dicts
        """
        if conversation_id is None:
            # Generate a simple unique ID if none provided by simulation
            conversation_id = f"conv_{self.evaluation_round}_{user_id}_{agent_id}_{time.time_ns()}"

        self.conversation_histories[conversation_id] = {
            "user_id": user_id,
            "agent_id": agent_id,
            "history": history,
            "timestamp": time.time(),
            "evaluation_round": self.evaluation_round,
        }

        # --- Feed conversation data to relevant information sources ---
        # This could be done more efficiently (e.g., batching), but simple loop for now
        for source_id, source in self.information_sources.items():
             # Check if the source has a method to consume conversation data
             if hasattr(source, 'add_conversation'):
                 # Pass relevant data - source decides if it cares about this user/agent/convo
                 try:
                     source.add_conversation(history, user_id, agent_id)
                 except Exception as e:
                     print(f"Warning: Error feeding conversation {conversation_id} to source {source_id}: {e}")



    def process_user_feedback(self, user_id: int, agent_id: int, ratings: Dict[str, int],
                              conversation_id: Optional[Any] = None, user_type: Optional[str] = None):
        """
        Process direct user feedback (specific ratings) from the simulation.

        Parameters:
        - user_id: User providing feedback (using the simulation's integer ID)
        - agent_id: Agent being rated (using the simulation's integer ID)
        - ratings: Dict mapping dimension names (str) to integer ratings (e.g., 1-5)
        - conversation_id: Optional ID of the conversation leading to this feedback
        - user_type: Optional user profile identifier
        """
        # 1. Log the raw feedback
        feedback_record = {
            "user_id": user_id,
            "agent_id": agent_id,
            "ratings": ratings,
            "conversation_id": conversation_id,
            "user_type": user_type, # Store user profile type if available
            "timestamp": time.time(),
            "evaluation_round": self.evaluation_round
        }
        self.user_feedback_log.append(feedback_record)

        # 2. Convert ratings to the format expected by TrustMarket (e.g., 0-1 or score adjustment)
        #    Assuming TrustMarket.record_user_feedback now handles the conversion
        #    based on self.rating_scale.
        # Example conversion (if needed here):
        # normalized_ratings = {}
        # for dim, rating in ratings.items():
        #      if dim in self.config['dimensions']:
        #           # Convert 1-N scale to 0-1 scale
        #           normalized_ratings[dim] = (rating - 1) / (self.rating_scale - 1)

        # 3. Apply the feedback to the TrustMarket
        #    Pass the original ratings and let the market handle normalization/update
        self.trust_market.record_user_feedback(
             user_id=str(user_id), # Use string representation for market source ID
             agent_id=agent_id,   # Agent ID remains integer
             ratings=ratings,     # Pass original ratings dict
             confidence=0.9       # Default confidence for direct user feedback, could be refined
        )

        # 4. (Optional) Update agent performance tracking in the market
        #    Convert ratings to 0-1 performance score
        performance_scores = {}
        for dim, rating in ratings.items():
             if dim in self.config['dimensions']:
                  performance_scores[dim] = (rating - 1) / (self.rating_scale - 1)
        if performance_scores:
             self.trust_market.update_agent_performance(agent_id, performance_scores)

        # 5. Feed feedback to User Representatives (if they consume raw feedback)
        for source in self.information_sources.values():
             if isinstance(source, UserRepresentative) or isinstance(source, UserRepresentativeWithHolisticEvaluation):
                  # Check if this rep represents this user
                  if hasattr(source, 'represented_users') and user_id in source.represented_users:
                       if hasattr(source, 'add_user_feedback'): # Check if method exists
                            try:
                                 source.add_user_feedback(user_id, agent_id, ratings, conversation_id)
                            except Exception as e:
                                 print(f"Warning: Error feeding user feedback to source {source.source_id}: {e}")


    def process_comparative_feedback(self, user_id: int, agent_id_a: int, agent_id_b: int,
                                     winner_scores: Dict[int, Dict[str, float]],
                                     conversation_id: Optional[Any] = None):
        """
        Process comparative feedback from the simulation.
        This feedback is primarily logged and passed to information sources,
        it does *not* directly modify TrustMarket scores in this design.

        Parameters:
        - user_id: User providing comparison
        - agent_id_a: First agent compared
        - agent_id_b: Second agent compared
        - winner_scores: Dict like {agent_id_a: {dim: score_a}, agent_id_b: {dim: score_b}} (1=win, 0=loss, 0.5=tie)
        - conversation_id: Optional ID of the conversation context
        """
        # 1. Log the comparative feedback
        comp_record = {
            "user_id": user_id,
            "agent_id_a": agent_id_a,
            "agent_id_b": agent_id_b,
            "winner_scores": winner_scores,
            "conversation_id": conversation_id,
            "timestamp": time.time(),
            "evaluation_round": self.evaluation_round
        }
        self.comparative_feedback_log.append(comp_record)
        print(f"Logged comparative feedback from User {user_id} for Agents {agent_id_a} vs {agent_id_b}.")

        # 2. Feed comparative feedback to relevant Information Sources
        #    User Representatives are primary consumers of this type of data.
        for source in self.information_sources.values():
             # Check if it's a UserRep and represents this user
             if isinstance(source, UserRepresentative) or isinstance(source, UserRepresentativeWithHolisticEvaluation):
                 if hasattr(source, 'represented_users') and user_id in source.represented_users:
                      # Check if the source has a method to process comparative feedback
                      if hasattr(source, 'add_comparative_feedback'):
                          try:
                               source.add_comparative_feedback(user_id, agent_id_a, agent_id_b, winner_scores, conversation_id)
                          except Exception as e:
                               print(f"Warning: Error feeding comparative feedback to source {source.source_id}: {e}")


    def run_evaluation_round(self):
        """Run a single evaluation round."""
        self.evaluation_round += 1
        print(f"\n--- Starting Evaluation Round {self.evaluation_round} ---")

        # 1. Increment market round counter (handles decay)
        self.trust_market.increment_evaluation_round()
        print("Trust market scores decayed.")

        # 2. Run simulation step to get user feedback & conversation data
        self._process_simulation_output()

        # 3. Trigger information source evaluations and investments
        self._run_source_evaluations()

        # 4. Display current market state (optional)
        self.display_current_scores()


    def _process_simulation_output(self):
        """Runs the simulation step and processes its output (feedback, conversations)."""
        if not self.simulation_module:
            print("Warning: Simulation module not registered. Cannot process user ratings.")
            return

        print("Running simulation step to gather feedback...")
        # This now returns dict with ratings, comparisons, and conversation data
        sim_results = self.simulation_module.multi_turn_dialog()

        # Process Specific Ratings
        num_ratings = 0
        for ratings, user_id, agent_id, conv_id in sim_results.get("specific_ratings", []):
            # Get user profile type if available
            user_type = self.simulation_module.user_profiles[user_id].get('technical_proficiency', 'unknown') if user_id < len(self.simulation_module.user_profiles) else 'unknown'
            self.process_user_feedback(user_id, agent_id, ratings, conv_id, user_type)
            num_ratings += 1
        if num_ratings > 0: print(f"Processed {num_ratings} specific user ratings.")

        # Process Comparative Winners
        num_comparisons = 0
        for winners, user_id, agent_id_a, agent_id_b, conv_id in sim_results.get("comparative_winners", []):
            self.process_comparative_feedback(user_id, agent_id_a, agent_id_b, winners, conv_id)
            num_comparisons += 1
        if num_comparisons > 0: print(f"Processed {num_comparisons} comparative user feedback results.")


        # Record Conversation Data for Information Sources
        num_convos_recorded = 0
        processed_conv_ids = set() # Avoid duplicates if comparative returns both
        for convo_data in sim_results.get("conversation_data", []):
            history = convo_data.get("history")
            user_id = convo_data.get("user_id")
            agent_id = convo_data.get("agent_id")
            conv_id = convo_data.get("conv_id") # May be None

            # Generate a unique ID if needed (e.g., combining round, user, agent)
            unique_conv_key = f"r{self.evaluation_round}_u{user_id}_a{agent_id}_{id(history)}" # Use object ID for uniqueness if conv_id is None

            if history and user_id is not None and agent_id is not None and unique_conv_key not in processed_conv_ids:
                self.record_conversation_data(unique_conv_key, user_id, agent_id, history)
                processed_conv_ids.add(unique_conv_key)
                num_convos_recorded += 1
        if num_convos_recorded > 0: print(f"Recorded {num_convos_recorded} conversation histories for information sources.")


    def _run_source_evaluations(self):
        """Run evaluations for sources due in this round and process their investments."""
        print("Running information source evaluations...")
        sources_evaluated_count = 0
        total_investment_actions = 0

        # Use numpy for efficient check (if many sources)
        source_ids = list(self.information_sources.keys())
        if not source_ids:
            print("No information sources registered.")
            return

        last_eval_rounds = np.array([self.source_last_evaluated.get(sid, 0) for sid in source_ids])
        eval_frequencies = np.array([self.source_evaluation_frequency.get(sid, 1) for sid in source_ids])

        # Indices of sources due for evaluation
        due_indices = np.where(self.evaluation_round - last_eval_rounds >= eval_frequencies)[0]

        if len(due_indices) == 0:
            print("No sources due for evaluation this round.")
            return

        for idx in due_indices:
            source_id = source_ids[idx]
            source = self.information_sources[source_id]
            print(f"Evaluating source: {source_id} ({source.source_type})")
            sources_evaluated_count += 1

            # Run evaluation & get investment decisions
            try:
                 # Pass evaluation_round for context, cache invalidation etc.
                investments = source.decide_investments(evaluation_round=self.evaluation_round)

                if investments:
                    print(f"  Source {source_id} proposed {len(investments)} investment actions.")
                    # Process the investments in the market
                    self.trust_market.process_investments(source_id, investments)
                    total_investment_actions += len(investments)
                else:
                    print(f"  Source {source_id} made no investment decisions.")

                # Record that this source was evaluated
                self.source_last_evaluated[source_id] = self.evaluation_round

            except Exception as e:
                print(f"Error evaluating source {source_id} or processing investments: {str(e)}")
                import traceback
                traceback.print_exc() # Print stack trace for debugging

        print(f"Completed evaluations for {sources_evaluated_count} sources, processing {total_investment_actions} investment actions.")


    def run_evaluation_rounds(self, num_rounds):
        """Run multiple evaluation rounds."""
        for i in range(num_rounds):
            self.run_evaluation_round()
            # Add a small delay if needed, e.g., for API rate limits
            # time.sleep(1)


    def get_agent_trust_scores(self):
        """Get the current trust scores for all agents from the market."""
        agent_scores = {}
        # Iterate through agents known by the simulation module
        if self.simulation_module:
             num_sim_agents = self.simulation_module.num_agents
             for agent_id in range(num_sim_agents):
                  agent_scores[agent_id] = self.trust_market.get_agent_trust(agent_id)
        else:
             # Fallback: Get scores for any agent known directly by the market
             agent_scores = self.trust_market.agent_trust_scores # Access internal dict (use getter if available)

        return agent_scores

    def get_agent_permissions(self, agent_id):
        """Determine agent permissions based on market trust scores."""
        return self.trust_market.get_agent_permissions(agent_id)

    def get_source_investments(self, source_id):
        """Get all current investments made by a source."""
        return self.trust_market.get_source_endorsements(source_id)

    def summarize_market_state(self):
        """Get a summary of the current market state."""
        # Delegate most of this to TrustMarket if possible, or build here
        agent_scores = self.get_agent_trust_scores()

        dimension_averages = defaultdict(list)
        for agent_id, scores in agent_scores.items():
            for dimension, score in scores.items():
                dimension_averages[dimension].append(score)

        avg_scores = {dim: (sum(scores) / len(scores)) if scores else 0.5
                      for dim, scores in dimension_averages.items()}

        source_influence = {}
        for source_id, source in self.information_sources.items():
             # Use market methods if they exist, otherwise access directly (less ideal)
             capacity = self.trust_market.source_influence_capacity.get(source_id, {})
             allocated = self.trust_market.allocated_influence.get(source_id, {})
             source_influence[source_id] = {
                 "type": source.source_type,
                 "capacity": {d: capacity.get(d, 0.0) for d in self.config['dimensions']},
                 "allocated": {d: allocated.get(d, 0.0) for d in self.config['dimensions']}
             }


        # Count investments per agent/dimension (might be complex, keep simple for now)
        investment_counts = defaultdict(lambda: defaultdict(int))
        # (This requires iterating through self.trust_market.source_investments which might be large)

        return {
            "evaluation_round": self.evaluation_round,
            "dimension_averages": avg_scores,
            "source_influence": source_influence,
            "num_information_sources": len(self.information_sources),
            "num_agents": len(agent_scores),
            # "investment_counts": dict(investment_counts) # Optional: Add if needed
        }

    def display_current_scores(self):
         """Prints the current trust scores of agents."""
         print("\n=== Current Agent Trust Scores (from Trust Market) ===")
         agent_scores = self.get_agent_trust_scores()
         if not agent_scores:
              print("No agent scores available.")
              return

         for agent_id, scores in agent_scores.items():
              # Try to get agent profile info from simulation module
              agent_type = f"Agent {agent_id}"
              if self.simulation_module and agent_id < len(self.simulation_module.agent_profiles):
                   profile = self.simulation_module.agent_profiles[agent_id]
                   goals = profile.get("primary_goals", [("","?")])[0][1]
                   agent_type = f"Agent {agent_id} (Goal: {goals})"

              print(f"{agent_type}:")
              score_strs = []
              sorted_dims = sorted(scores.keys()) # Sort dimensions alphabetically
              for dim in sorted_dims:
                   score = scores[dim]
                   score_strs.append(f"{dim}: {score:.3f}")
              # Print scores in rows
              for i in range(0, len(score_strs), 3):
                   print("  " + ", ".join(score_strs[i:i+3]))
         print("-" * 50)


    # Visualization methods can remain, they call TrustMarket's visualization
    def visualize_trust_scores(self, agents=None, dimensions=None, start_round=None, end_round=None):
        """Visualize trust scores over time."""
        return self.trust_market.visualize_trust_scores(agents, dimensions, start_round, end_round)

    def visualize_source_performance(self, sources=None, dimensions=None, start_round=None, end_round=None):
        """Visualize source performance over time."""
        return self.trust_market.visualize_source_performance(sources, dimensions, start_round, end_round)

