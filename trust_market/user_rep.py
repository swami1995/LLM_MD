import math
import numpy as np
from collections import defaultdict
from trust_market.info_sources import InformationSource

class UserRepresentative(InformationSource):
    """
    Represents the aggregated feedback of a segment of users.
    Focused on user experience dimensions relevant to that segment.
    Redesigned to analyze patterns across users rather than processing raw feedback.
    """
    def __init__(self, source_id, user_segment, representative_profile, market=None):
        """
        Initialize a user representative.
        
        Parameters:
        - source_id: Unique identifier
        - user_segment: Description of user segment represented (e.g., "technical", "non_technical")
        - representative_profile: Profile describing this representative's characteristics
        - market: Reference to the trust market
        """
        # Each segment has different areas of expertise
        if user_segment == "technical":
            expertise_dimensions = [
                "Factual_Correctness", 
                "Process_Reliability", 
                "Transparency",
                "Trust_Calibration"
            ]
            confidence = {
                "Factual_Correctness": 0.9,
                "Process_Reliability": 0.8,
                "Transparency": 0.7,
                "Trust_Calibration": 0.8
            }
        elif user_segment == "non_technical":
            expertise_dimensions = [
                "Communication_Quality", 
                "Problem_Resolution", 
                "Value_Alignment", 
            ]
            confidence = {
                "Communication_Quality": 0.9,
                "Problem_Resolution": 0.8,
                "Value_Alignment": 0.7,
            }
        else:  # balanced
            expertise_dimensions = [
                "Communication_Quality", 
                "Problem_Resolution", 
                "Value_Alignment", 
                "Transparency"
            ]
            confidence = {
                "Communication_Quality": 0.8,
                "Problem_Resolution": 0.8,
                "Value_Alignment": 0.7,
                "Transparency": 0.7
            }
        
        super().__init__(source_id, "user_representative", expertise_dimensions, 
                         confidence, market)
        
        self.user_segment = user_segment
        self.representative_profile = representative_profile
        self.represented_users = set()  # Set of user IDs this representative speaks for
        self.agent_conversations = {}  # Indexed by agent_id
        self.user_conversations = {}  # Indexed by user_id
        self.observed_conversations = []  # All conversations observed by this representative
        
        # Investment multipliers
        self.divest_multiplier = 0.2
        self.invest_multiplier = 0.2
        
        # Configuration for dimension importance by segment
        self.segment_weights = {
            "technical": {
                "Factual_Correctness": 0.9,
                "Process_Reliability": 0.8,
                "Value_Alignment": 0.6,
                "Communication_Quality": 0.6,
                "Problem_Resolution": 0.7,
                "Transparency": 0.8,
                "Trust_Calibration": 0.9
            },
            "non_technical": {
                "Factual_Correctness": 0.6,
                "Process_Reliability": 0.6,
                "Value_Alignment": 0.8,
                "Communication_Quality": 0.9,
                "Problem_Resolution": 0.9,
                "Transparency": 0.5,
                "Trust_Calibration": 0.4
            },
            "balanced": {
                "Factual_Correctness": 0.7,
                "Process_Reliability": 0.7,
                "Value_Alignment": 0.8,
                "Communication_Quality": 0.8,
                "Problem_Resolution": 0.9,
                "Transparency": 0.6,
                "Trust_Calibration": 0.6
            }
        }
    
    def add_represented_user(self, user_id, user_profile):
        """Add a user to the set of users represented by this representative."""
        self.represented_users.add(user_id)
    
    def add_conversation(self, conversation, user_id, agent_id, user_feedback=None):
        """
        Analyze a conversation between a user and an agent.
        
        Parameters:
        - conversation: The conversation history
        - user_id: ID of the user in the conversation
        - agent_id: ID of the agent in the conversation
        - user_feedback: Optional direct feedback from the user
        """
        # Only analyze conversations for users we represent
        if user_id not in self.represented_users:
            return
            
        # Initialize dictionaries for this agent/user if they don't exist
        if agent_id not in self.agent_conversations:
            self.agent_conversations[agent_id] = []
        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = []
            
        # Create conversation record
        conversation_record = {
            'conversation': conversation,
            'user_id': user_id,
            'agent_id': agent_id,
            'user_feedback': user_feedback,
            'analysis': None  # Will be filled in evaluate_agent
        }
        
        # Store in both dictionaries
        self.agent_conversations[agent_id].append(conversation)
        self.user_conversations[user_id].append(conversation)

        self.observed_conversations.append(conversation_record)  # Track all conversations observed by this representative
    
    def observed_agents(self):
        """Return a set of all agent IDs that this representative has observed."""
        return set(self.agent_conversations.keys())
    

class HolisticBatchEvaluator:
    """
    Evaluates batches of conversations between agents holistically,
    providing magnitude of differences to inform investment decisions.
    """
    
    def __init__(self, llm_client=None, api_key=None):
        """
        Initialize the holistic batch evaluator.
        
        Parameters:
        - llm_client: Pre-configured LLM client
        - api_key: API key for LLM service if client not provided
        """
        self.llm_client = llm_client
        self.api_key = api_key
        
        # Map of dimension names to descriptions for LLM prompting
        self.dimension_descriptions = {
            "Factual_Correctness": "accuracy and truthfulness of information provided",
            "Process_Reliability": "consistency in following proper procedures",
            "Value_Alignment": "prioritizing user needs over other incentives",
            "Communication_Quality": "clarity, appropriateness, and effectiveness of communication",
            "Problem_Resolution": "successfully addressing and solving the user's issue",
            "Safety_Security": "protecting sensitive information and avoiding harmful advice",
            "Transparency": "clarity about capabilities, limitations, and basis for responses",
            "Adaptability": "handling unexpected situations or unclear requests",
            "Trust_Calibration": "appropriately communicating confidence level and uncertainty",
            "Manipulation_Resistance": "resisting attempts to be misled or manipulated"
        }
        
        # Cache for evaluation results
        self.evaluation_cache = {}
    
    def _get_llm_response(self, prompt):
        """
        Get structured response from LLM service.
        This is a placeholder - implement with actual LLM API in production.
        """
        if self.llm_client:
            # Implementation depends on which LLM service is being used
            response = self.llm_client.models.generate_content(
                model="gemini-1.0-pro",
                prompt=prompt
            )
            return response.text
        else:
            # Mock implementation for testing - in reality you'd use an actual LLM
            import random
            import json
            
            mock_result = {}
            for dim in ["Factual_Correctness", "Process_Reliability", "Value_Alignment",
                        "Communication_Quality", "Problem_Resolution", "Safety_Security",
                        "Transparency", "Adaptability", "Trust_Calibration", "Manipulation_Resistance"]:
                        
                winner = random.choice(["A", "B", "Tie"])
                magnitude = random.randint(1, 5) if winner != "Tie" else 0
                
                mock_result[dim] = {
                    "winner": winner,
                    "magnitude": magnitude,
                    "reasoning": f"Mock reasoning for {dim}"
                }
            
            return json.dumps(mock_result, indent=2)
    
    def format_conversation_batch(self, conversations, max_conversations=5, max_length=1000):
        """
        Format a batch of conversations for LLM prompt, with truncation if needed.
        
        Parameters:
        - conversations: List of conversation histories
        - max_conversations: Maximum number of conversations to include
        - max_length: Maximum character length per conversation
        
        Returns:
        - Formatted string of conversations
        """
        # If too many conversations, sample or take most recent
        if len(conversations) > max_conversations:
            # Prioritize diversity - simple approach is to take evenly spaced samples
            step = len(conversations) // max_conversations
            sampled_indices = [i * step for i in range(max_conversations)]
            sample_conversations = [conversations[i] for i in sampled_indices]
        else:
            sample_conversations = conversations
        
        formatted = ""
        for i, conversation in enumerate(sample_conversations):
            formatted += f"CONVERSATION {i+1}:\n"
            
            # Format this conversation
            conv_text = ""
            for turn in conversation:
                if 'user' in turn:
                    conv_text += f"User: {turn['user']}\n"
                if 'agent' in turn:
                    conv_text += f"Agent: {turn['agent']}\n"
                conv_text += "\n"
            
            # Truncate if too long
            if len(conv_text) > max_length:
                # Try to truncate at a turn boundary
                half_length = max_length // 2
                start = conv_text[:half_length]
                end = conv_text[-half_length:]
                formatted += f"{start}\n...[conversation truncated]...\n{end}\n"
            else:
                formatted += conv_text
            
            formatted += "\n---\n\n"
        
        return formatted
    
    def compare_agent_batches(self, agent_a_convs, agent_a_id, agent_b_convs, agent_b_id, dimensions=None):
        """
        Holistically compare two agents based on batches of their conversations.
        
        Parameters:
        - agent_a_convs: List of conversations from agent A
        - agent_a_id: ID of agent A
        - agent_b_convs: List of conversations from agent B
        - agent_b_id: ID of agent B
        - dimensions: List of dimensions to compare (default: all dimensions)
        
        Returns:
        - Dict mapping dimensions to comparison results with magnitude
        """
        # Use all dimensions if none specified
        if dimensions is None:
            dimensions = list(self.dimension_descriptions.keys())
        
        # Create cache key
        cache_key = (
            hash(str(agent_a_convs) + str(agent_b_convs)),
            agent_a_id,
            agent_b_id,
            tuple(dimensions)
        )
        
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        # Format conversation batches
        formatted_a = self.format_conversation_batch(agent_a_convs)
        formatted_b = self.format_conversation_batch(agent_b_convs)
        
        # Format dimension specifications
        dimension_specs = "\n".join([
            f"{i+1}. {dim}: {self.dimension_descriptions.get(dim, dim)}"
            for i, dim in enumerate(dimensions)
        ])
        
        # Construct prompt for holistic evaluation
        prompt = f"""You are evaluating the performance of two customer support agents based on multiple conversations each agent has had with users.

Please compare Agent A and Agent B holistically across their entire conversation batches. For each dimension, determine which agent performs better overall and by how much.

AGENT A CONVERSATIONS:
{formatted_a}

AGENT B CONVERSATIONS:
{formatted_b}

**Compare the agents on these dimensions:**
{dimension_specs}

**For EACH dimension, provide:**
1. The better-performing agent ("A", "B", or "Tie")
2. The magnitude of difference on a scale of 0-5, where:
   - 0 = TIE: No meaningful difference between agents. Both agents perform at essentially the same level on this dimension. Users would have practically identical experiences with either agent.
   - 1 = SLIGHTLY BETTER: Small, subtle advantages that trained reviewers would notice but typical users might miss. Examples: slightly clearer explanations, marginally more accurate information, or minor efficiency improvements. These differences rarely impact overall satisfaction but do represent a measurable improvement.
   - 2 = SOMEWHAT BETTER: Modest but consistent advantages that attentive users would occasionally notice. Examples: better organization of responses, more appropriate tone, more thorough answers when needed, or better follow-up questions. These differences occasionally improve user outcomes but don't transform the experience.
   - 3 = NOTICEABLY BETTER: Clear, consistent advantages that most users would recognize. Examples: significantly more accurate technical information, much better handling of user confusion, more effective problem diagnosis, or notably more helpful suggestions. These differences regularly lead to better outcomes and improved user satisfaction.
   - 4 = SUBSTANTIALLY BETTER: Major advantages that fundamentally improve service quality. Examples: successfully resolving issues the other agent fails to address, demonstrating vastly superior domain knowledge, handling complex scenarios with remarkable skill, or consistently providing creative solutions the other agent misses. Nearly all users would strongly prefer this agent.
   - 5 = DRAMATICALLY BETTER: Overwhelming superiority that transforms the user experience. The better agent demonstrates excellence across virtually all aspects of the interaction - accuracy, helpfulness, efficiency, empathy, and problem-solving. The magnitude of difference is so profound that it represents a generational improvement in service quality.
3. Brief reasoning explaining patterns you observed across the conversations

Format your response as a JSON object with this structure for each dimension:
{{
  "Dimension_Name": {{
    "winner": "A or B or Tie",
    "magnitude": integer from 0-5
  }},
  ...
}}"""
        # "reasoning": "brief explanation"
        # Get LLM evaluation
        response_text = self._get_llm_response(prompt)
        
        # Parse the response - in production, implement more robust parsing
        try:
            import json
            import re
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                results_json = json_match.group(0)
                results = json.loads(results_json)
            else:
                # Fallback - extract structured data through regex
                results = {}
                for dim in dimensions:
                    dim_pattern = fr'"{dim}".*?{{\s*"winner":\s*"([^"]+)",\s*"magnitude":\s*(\d+)'
                    match = re.search(dim_pattern, response_text, re.DOTALL)
                    if match:
                        winner, magnitude = match.groups()
                        results[dim] = {
                            "winner": winner,
                            "magnitude": int(magnitude),
                            "reasoning": "Extracted from partial match"
                        }
                
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            # Provide default results
            results = {dim: {"winner": "Tie", "magnitude": 0, "reasoning": "Error in evaluation"} 
                     for dim in dimensions}
        
        # Process and normalize results
        processed_results = {}
        for dim in dimensions:
            if dim in results:
                result = results[dim]
                winner = result.get("winner", "Tie")
                magnitude = result.get("magnitude", 0)
                
                # Validate and normalize
                if winner not in ["A", "B", "Tie"]:
                    winner = "Tie"
                
                magnitude = max(0, min(5, int(magnitude)))
                if winner == "Tie":
                    magnitude = 0
                
                processed_results[dim] = {
                    "winner": winner,
                    "magnitude": magnitude,
                    "reasoning": result.get("reasoning", "")
                }
            else:
                # Default for missing dimensions
                processed_results[dim] = {
                    "winner": "Tie",
                    "magnitude": 0,
                    "reasoning": "Dimension not evaluated"
                }
        
        # Cache results
        self.evaluation_cache[cache_key] = processed_results
        
        return processed_results
    
    def get_agent_scores(self, comparison_results, agent_a_id, agent_b_id):
        """
        Convert comparison results to absolute scores for each agent.
        
        Parameters:
        - comparison_results: Results from compare_agent_batches
        - agent_a_id: ID of agent A in comparison
        - agent_b_id: ID of agent B in comparison
        
        Returns:
        - Dict mapping agent IDs to dimension scores
        """
        # Start with default middle scores
        agent_scores = {
            agent_a_id: {dim: 0.5 for dim in comparison_results},
            agent_b_id: {dim: 0.5 for dim in comparison_results}
        }
        
        # Adjust scores based on comparison results
        for dimension, result in comparison_results.items():
            winner = result["winner"]
            magnitude = result["magnitude"]
            
            # Convert magnitude to score adjustment (0-0.4 range)
            adjustment = magnitude * 0.08  # 0.08 * 5 = 0.4 max adjustment
            
            if winner == "A":
                # Agent A wins
                agent_scores[agent_a_id][dimension] = 0.5 + adjustment
                agent_scores[agent_b_id][dimension] = 0.5 - adjustment
            elif winner == "B":
                # Agent B wins
                agent_scores[agent_a_id][dimension] = 0.5 - adjustment
                agent_scores[agent_b_id][dimension] = 0.5 + adjustment
            # If tie, both stay at 0.5
        
        return agent_scores


class UserRepresentativeWithHolisticEvaluation(UserRepresentative):
    """
    User representative that evaluates agents holistically across batches of conversations.
    """
    
    def __init__(self, source_id, user_segment, representative_profile, market=None, llm_client=None):
        super().__init__(source_id, user_segment, representative_profile, market)
        
        # Initialize holistic evaluator
        self.evaluator = HolisticBatchEvaluator(llm_client=llm_client)
        
        # Cache for agent evaluations
        self.agent_evaluation_cache = {}
        
        # Tracking recent evaluation rounds
        self.last_evaluation_round = 0
        
        # Track agent pairwise comparisons
        self.compared_pairs = set()
        
        # Store absolute scores derived from comparisons
        self.derived_agent_scores = {}
        
        # Configuration
        self.config = {
            # How many agents to compare against each target agent
            'comparison_agents_per_target': 6,
            
            # Minimum conversations needed for evaluation
            'min_conversations_required': 5,
            
            # Weight given to new evaluations vs existing scores
            'new_evaluation_weight': 0.7
        }
    
    def get_agent_conversations(self, agent_id, max_count=10):
        """Get conversations for a specific agent."""
        if agent_id not in self.agent_conversations:
            return []
        
        conversations = self.agent_conversations[agent_id]
        
        # Return the most recent conversations up to max_count
        return conversations[-max_count:] if conversations else []
    
    def evaluate_agent(self, agent_id, dimension=None, evaluation_round=None):
        """
        Evaluate agent using holistic comparison across multiple conversations.
        
        Parameters:
        - agent_id: Agent to evaluate
        - dimension: Optional specific dimension to evaluate
        - evaluation_round: Current evaluation round
        
        Returns:
        - Dict mapping dimensions to (rating, confidence) tuples
        """
        # Check if evaluation cache is valid for this round
        if evaluation_round and evaluation_round == self.last_evaluation_round:
            if agent_id in self.agent_evaluation_cache:
                return self.agent_evaluation_cache[agent_id]
        
        # Update evaluation round tracking
        if evaluation_round:
            self.last_evaluation_round = evaluation_round
        
        dimensions = [dimension] if dimension else self.expertise_dimensions
        
        # If we already have derived scores for this agent, use them as a starting point
        if agent_id in self.derived_agent_scores:
            base_scores = {
                dim: (score, 0.5)  # Start with moderate confidence
                for dim, score in self.derived_agent_scores[agent_id].items()
                if dim in dimensions
            }
        else:
            # Initialize with neutral scores
            base_scores = {dim: (0.5, 0.3) for dim in dimensions}
        
        # Get conversations for this agent
        agent_conversations = self.get_agent_conversations(agent_id)
        
        # If too few conversations, return base scores with low confidence
        if len(agent_conversations) < self.config['min_conversations_required']:
            return {dim: (score, min(conf, 0.3)) for dim, (score, conf) in base_scores.items()}
        
        # Find other agents to compare against
        other_agents = self.observed_agents()
        other_agents.discard(agent_id)  # Remove the current agent from the set
        
        # Filter to agents with enough conversations
        valid_comparison_agents = []
        for other_id in other_agents:
            other_convs = self.get_agent_conversations(other_id)
            if len(other_convs) >= self.config['min_conversations_required']:
                valid_comparison_agents.append((other_id, other_convs))
        
        # If no valid comparison agents, return base scores
        if not valid_comparison_agents:
            return base_scores
        
        # Select a subset of agents to compare against
        import random
        if len(valid_comparison_agents) > self.config['comparison_agents_per_target']:
            # Prioritize agents we haven't compared against recently
            new_comparisons = [
                (other_id, convs) for other_id, convs in valid_comparison_agents
                if (agent_id, other_id) not in self.compared_pairs
            ]
            
            if len(new_comparisons) >= self.config['comparison_agents_per_target']:
                comparison_agents = random.sample(new_comparisons, self.config['comparison_agents_per_target'])
            else:
                # Add some previously compared agents
                previously_compared = [
                    (other_id, convs) for other_id, convs in valid_comparison_agents
                    if (agent_id, other_id) in self.compared_pairs
                ]
                
                remaining_slots = self.config['comparison_agents_per_target'] - len(new_comparisons)
                if previously_compared and remaining_slots > 0:
                    comparison_agents = new_comparisons + random.sample(
                        previously_compared, 
                        min(remaining_slots, len(previously_compared))
                    )
                else:
                    comparison_agents = new_comparisons
        else:
            comparison_agents = valid_comparison_agents
        
        # Perform holistic comparisons
        new_scores = {}
        comparison_count = 0
        
        for other_id, other_convs in comparison_agents:
            # Mark this pair as compared
            self.compared_pairs.add((agent_id, other_id))
            self.compared_pairs.add((other_id, agent_id))  # Symmetrical
            
            # Compare the agents
            comparison_results = self.evaluator.compare_agent_batches(
                agent_conversations, agent_id,
                other_convs, other_id,
                dimensions
            )
            
            # Convert to absolute scores
            agent_scores = self.evaluator.get_agent_scores(
                comparison_results, agent_id, other_id
            )
            
            # Accumulate scores
            for dim in dimensions:
                if dim in agent_scores[agent_id]:
                    if dim not in new_scores:
                        new_scores[dim] = []
                    new_scores[dim].append(agent_scores[agent_id][dim])
            
            comparison_count += 1
        
        # Calculate final scores
        final_scores = {}
        for dim in dimensions:
            if dim in new_scores and new_scores[dim]:
                # Calculate average of new scores
                new_avg = sum(new_scores[dim]) / len(new_scores[dim])
                
                # Get base score
                base_score, base_conf = base_scores.get(dim, (0.5, 0.3))
                
                # Weight new vs existing scores
                weight = self.config['new_evaluation_weight']
                final_score = (weight * new_avg) + ((1 - weight) * base_score)
                
                # Confidence based on number of comparisons
                confidence = min(0.9, 0.4 + (comparison_count * 0.1) + base_conf)
                
                final_scores[dim] = (final_score, confidence)
            else:
                # Use base score with reduced confidence
                score, conf = base_scores.get(dim, (0.5, 0.3))
                final_scores[dim] = (score, conf * 0.8)  # Reduce confidence if no new data
        
        # Update derived scores for future reference
        if agent_id not in self.derived_agent_scores:
            self.derived_agent_scores[agent_id] = {}
            
        for dim, (score, _) in final_scores.items():
            self.derived_agent_scores[agent_id][dim] = score
        
        # Cache results
        self.agent_evaluation_cache[agent_id] = final_scores
        
        return final_scores
        
    def decide_investments(self, evaluation_round=None):
        """
        Make investment decisions with dynamic reallocation based on opportunity strength.
        
        This strategy not only allocates available influence but also reallocates
        a portion of existing investments to optimize the overall portfolio.
        
        Parameters:
        - evaluation_round: Current evaluation round number
        
        Returns:
        - List of (agent_id, dimension, amount, confidence) tuples
        """
        # Clear previous evaluation cache if needed
        if evaluation_round and evaluation_round != self.last_evaluation_round:
            self.agent_evaluation_cache = {}
            self.last_evaluation_round = evaluation_round
        
        # Skip if no market access
        if not self.market:
            return []
        
        # Define reallocation parameters
        reallocation_percentage = 0.3  # Percentage of current allocations to free up
        min_reallocation_threshold = 0.2  # Minimum change in relative strength to trigger reallocation
        

        # Track current investments by agent and dimension
        current_investments = {}
        
        # Get current endorsements from the market
        if hasattr(self.market, 'get_source_endorsements'):
            try:
                endorsements = self.market.get_source_endorsements(self.source_id)
                for endorsement in endorsements:
                    agent_id = endorsement.agent_id
                    dimension = endorsement.dimension
                    amount = endorsement.influence_amount
                    
                    if agent_id not in current_investments:
                        current_investments[agent_id] = {}
                    if dimension not in current_investments[agent_id]:
                        current_investments[agent_id][dimension] = 0.0
                    
                    current_investments[agent_id][dimension] += amount
            except:
                pass  # Fallback if method doesn't exist or fails
        
        # Calculate total current investment for each dimension
        total_current_investment = {}
        for dimension in self.expertise_dimensions:
            total_current_investment[dimension] = sum(
                amounts.get(dimension, 0.0) for amounts in current_investments.values()
            )
        
        # Analyze market for investment opportunities
        observed_agents = self.observed_agents()
        if len(observed_agents) < 2:
            return []
        
        # Get market scores and our evaluations
        try:
            market_scores = {}
            for agent_id in observed_agents:
                agent_trust = self.market.get_agent_trust(agent_id)
                if agent_trust:
                    market_scores[agent_id] = agent_trust
        except Exception as e:
            print(f"Error getting market scores: {e}")
            return []
        
        if len(market_scores) < 2:
            return []
        
        # Get our evaluations
        own_evaluations = {}
        for agent_id in observed_agents:
            if agent_id in market_scores:
                agent_convs = self.get_agent_conversations(agent_id)
                if len(agent_convs) >= self.config.get('min_conversations_required', 3):
                    evaluation = self.evaluate_agent(agent_id, evaluation_round=evaluation_round)
                    if evaluation:
                        own_evaluations[agent_id] = evaluation
        
        if len(own_evaluations) < 2:
            return []
        
        # Identify investment opportunities based on relative positions
        investment_opportunities = {}
        
        # Process each dimension separately
        own_relative_positions = {}
        market_relative_positions = {}
        
        for dimension in self.expertise_dimensions:
            # Skip if we don't have enough evaluations for this dimension
            investment_opportunities[dimension] = []
            dimension_evaluations = {
                agent_id: eval_data[dimension][0]
                for agent_id, eval_data in own_evaluations.items()
                if dimension in eval_data
            }
            
            dimension_confidences = {
                agent_id: eval_data[dimension][1]
                for agent_id, eval_data in own_evaluations.items()
                if dimension in eval_data
            }
            
            if len(dimension_evaluations) < 2:
                continue
            
            # Get market scores for this dimension
            dimension_market_scores = {
                agent_id: scores.get(dimension, 0.5)
                for agent_id, scores in market_scores.items()
                if agent_id in dimension_evaluations
            }
            
            # Calculate relative positioning using helper method
            dimension_own_relative = self._get_relative_positions(dimension_evaluations, dimension)
            dimension_market_relative = self._get_relative_positions(dimension_market_scores, dimension)
            
            # Store in global dict for later use
            for agent_id, position in dimension_own_relative.items():
                if agent_id not in own_relative_positions:
                    own_relative_positions[agent_id] = {}
                own_relative_positions[agent_id][dimension] = position
            
            for agent_id, position in dimension_market_relative.items():
                if agent_id not in market_relative_positions:
                    market_relative_positions[agent_id] = {}
                market_relative_positions[agent_id][dimension] = position
            
            # Calculate disagreement in relative positioning
            for agent_id in dimension_evaluations.keys():
                own_position = dimension_own_relative[agent_id]
                market_position = dimension_market_relative[agent_id]
                
                relative_disagreement = own_position - market_position
                opportunity_direction = 1 if relative_disagreement > 0 else -1
                disagreement_magnitude = abs(relative_disagreement)
                
                own_confidence = dimension_confidences.get(agent_id, 0.5)
                segment_weight = self.segment_weights.get(self.user_segment, {}).get(dimension, 1.0)
                
                # Calculate opportunity strength
                opportunity_strength = disagreement_magnitude * own_confidence * segment_weight
                
                # Only consider meaningful disagreements
                min_threshold = 0.1
                if disagreement_magnitude >= min_threshold and opportunity_strength > 0:
                    investment_opportunities[dimension].append({
                        'agent_id': agent_id,
                        'dimension': dimension,
                        'relative_disagreement': relative_disagreement,
                        'direction': opportunity_direction,
                        'market_position': market_position,
                        'own_position': own_position,
                        'own_confidence': own_confidence,
                        'segment_weight': segment_weight,
                        'strength': opportunity_strength,
                        # Add current investment amount (if any)
                        'current_investment': current_investments.get(agent_id, {}).get(dimension, 0.0) or 0.0
                    })
        
        if not any(investment_opportunities.values()):
            return []
        
        # Calculate relative strength and optimal allocations
        self._calculate_investment_strategy(investment_opportunities, total_current_investment)
        
        # Prepare new investment plan
        return self._prepare_investment_actions(
            investment_opportunities, 
            total_current_investment,
            reallocation_percentage, 
            available_influence
        )
    
    def _calculate_investment_strategy(self, investment_opportunities, total_current_investment):
        """
        Calculate investment strategy based on opportunity strength.
        
        Parameters:
        - investment_opportunities: Dict of investment opportunities by dimension
        - total_current_investment: Dict of total current investments by dimension
        """
        # Calculate relative strength for each opportunity
        total_strength = {}
        for dimension in investment_opportunities:
            opportunities = investment_opportunities[dimension]
            if not opportunities:
                continue
                
            total_strength[dimension] = sum(
                opp['strength'] for opp in opportunities if opp['direction'] > 0
            ) or 1.0  # Avoid division by zero
            
            # Calculate relative strength
            for opp in opportunities:
                if opp['direction'] > 0 and total_strength[dimension] > 0:
                    opp['relative_strength'] = opp['strength'] / total_strength[dimension]
                else:
                    # Equal distribution if no positive direction or zero total strength
                    opp['relative_strength'] = 1.0 / len(opportunities)
        
        # Calculate normalized current investment and invest/divest signals
        for dimension in investment_opportunities:
            opportunities = investment_opportunities[dimension]
            if not opportunities:
                continue
                
            total_current = sum(opp['current_investment'] for opp in opportunities) #or 1.0  # Avoid division by zero
            
            for opp in opportunities:
                current = opp['current_investment'] if opp['current_investment'] else 0.0
                opp['current_investment_normalized'] = current / total_current if total_current > 0 else 0.0
                opp['invest_divest_normalized'] = (
                    opp['direction'] * opp['relative_strength'] - opp['current_investment_normalized']
                )
    
    def _prepare_investment_actions(self, investment_opportunities, total_current_investment, 
                                  reallocation_percentage, available_influence):
        """
        Prepare investment and divestment actions based on calculated strategy.
        
        Parameters:
        - investment_opportunities: Dict of investment opportunities by dimension
        - total_current_investment: Dict of total current investments by dimension
        - reallocation_percentage: Percentage of current allocations to free up
        - available_influence: Currently available influence by dimension
        
        Returns:
        - List of investment/divestment actions
        """
        # Calculate how much to free up from current investments
        amount_to_free_up = {
            dim: total_current_investment.get(dim, 0.0) * reallocation_percentage
            for dim in investment_opportunities
        }
        
        # Calculate total available influence (unused + freed up)
        total_available = {
            dim: available_influence.get(dim, 0.0) + amount_to_free_up.get(dim, 0.0)
            for dim in investment_opportunities
        }
        
        # Prepare divestments and investments
        divestments = []
        investments = []
        
        # Handle divestments to free up influence
        divest_amount = {}
        total_divest_amount = {}
        total_available_amount = {}
        
        for dimension, opportunities in investment_opportunities.items():
            if not opportunities:
                continue
                
            divest_amount[dimension] = {}
            total_divest_amount[dimension] = 0.0
            
            for opp in opportunities:
                if opp['invest_divest_normalized'] < 0:
                    agent_id = opp['agent_id']
                    amount = -opp['invest_divest_normalized'] * self.divest_multiplier
                    divest_amount[dimension][agent_id] = amount
                    total_divest_amount[dimension] += amount
                    
                    if amount > 0:
                        divestments.append((agent_id, dimension, -amount, None))
            
            total_available_amount[dimension] = (
                available_influence.get(dimension, 0.0) + total_divest_amount.get(dimension, 0.0)
            )
        
        # Handle investments
        for dimension, opportunities in investment_opportunities.items():
            if not opportunities:
                continue
                
            positive_opportunities = [
                opp for opp in opportunities if opp['invest_divest_normalized'] > 0
            ]
            
            if not positive_opportunities:
                continue
                
            total_positive = sum(opp['invest_divest_normalized'] for opp in positive_opportunities)
            
            if total_positive <= 0:
                continue
                
            total_available = total_available_amount.get(dimension, 0.0)
            
            for opp in positive_opportunities:
                agent_id = opp['agent_id']
                amount = (opp['invest_divest_normalized'] / total_positive) * total_available * self.invest_multiplier
                confidence = opp['own_confidence']
                
                if amount > 0:
                    investments.append((agent_id, dimension, amount, confidence))
        
        # Combine divestments and investments
        return divestments + investments
    
    def _get_relative_positions(self, evaluations, dimension):
        """
        Calculate relative positioning of agents for a specific dimension.
        
        Parameters:
        - evaluations: Dict mapping agent_ids to scores
        - dimension: The dimension to calculate positions for
        
        Returns:
        - Dict mapping agent_ids to relative positions (0-1)
        """
        if len(evaluations) <= 1:
            return {agent_id: 0.5 for agent_id in evaluations}
            
        dimension_relative = {}
        for agent_id, score in evaluations.items():
            outperforms_count = sum(1 for other_id, other_score in evaluations.items()
                            if other_id != agent_id and score > other_score)
            dimension_relative[agent_id] = outperforms_count / (len(evaluations) - 1)
            
        return dimension_relative