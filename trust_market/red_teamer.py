import time
import random
from collections import defaultdict
from trust_market.information_source import InformationSource


class RedTeamer(InformationSource):
    """
    Red teamer who performs adversarial testing to evaluate security, 
    manipulation resistance, and adaptability.
    """
    def __init__(self, source_id, specialty=None, market=None):
        """
        Initialize a red teamer.
        
        Parameters:
        - source_id: Unique identifier
        - specialty: Optional area of specialty
        - market: Reference to the trust market
        """
        # Red teamers focus on security, manipulation resistance, etc.
        expertise_dimensions = [
            "Safety & Security", 
            "Manipulation Resistance", 
            "Value Alignment", 
            "Adaptability"
        ]
        
        confidence = {
            "Safety & Security": 0.9,
            "Manipulation Resistance": 0.85,
            "Value Alignment": 0.7,
            "Adaptability": 0.8
        }
        
        super().__init__(source_id, "red_teamer", expertise_dimensions, 
                         confidence, market)
        
        self.specialty = specialty
        self.test_scenarios = []
        self.test_results = defaultdict(list)
        
        # Define attack categories
        self.attack_categories = {
            "manipulation": {
                "description": "Attempts to manipulate agent responses",
                "dimensions": ["Manipulation Resistance", "Value Alignment"]
            },
            "security": {
                "description": "Tests for security vulnerabilities",
                "dimensions": ["Safety & Security"]
            },
            "edge_cases": {
                "description": "Tests handling of unusual requests",
                "dimensions": ["Adaptability"]
            },
            "prompt_injection": {
                "description": "Attempts to inject malicious prompts",
                "dimensions": ["Safety & Security", "Manipulation Resistance"]
            }
        }
    
    def add_test_scenario(self, scenario_id, category, description, test_sequence):
        """
        Add an adversarial test scenario.
        
        Parameters:
        - scenario_id: Unique ID for this test
        - category: Type of test (from attack_categories)
        - description: Description of the test
        - test_sequence: Sequence of inputs to send to the agent
        """
        self.test_scenarios.append({
            'id': scenario_id,
            'category': category,
            'description': description,
            'test_sequence': test_sequence,
            'dimensions': self.attack_categories.get(category, {}).get(
                'dimensions', ["Safety & Security"]
            )
        })
    
    def run_tests(self, agent_id, agent_function, num_tests=None):
        """
        Run adversarial tests on an agent.
        
        Parameters:
        - agent_id: Agent to test
        - agent_function: Function that takes a query and returns a response
        - num_tests: Optional limit on number of tests to run
        
        Returns:
        - Dict of test results by dimension
        """
        if not self.test_scenarios:
            return {}
            
        # Select test scenarios to run
        scenarios = self.test_scenarios
        if num_tests is not None and num_tests < len(scenarios):
            scenarios = random.sample(scenarios, num_tests)
        
        results = defaultdict(list)
        
        # Run each test scenario
        for scenario in scenarios:
            conversation = []
            
            # Execute test sequence
            for i, query in enumerate(scenario['test_sequence']):
                response = agent_function(query)
                conversation.append({'user': query, 'agent': response})
            
            # Evaluate results for each relevant dimension
            for dimension in scenario['dimensions']:
                if dimension in self.expertise_dimensions:
                    # Analyze conversation based on dimension and attack category
                    rating, confidence = self._evaluate_test_response(
                        conversation, dimension, scenario['category']
                    )
                    
                    results[dimension].append((rating, confidence))
                    
                    # Record test result
                    self.test_results[agent_id].append({
                        'scenario_id': scenario['id'],
                        'dimension': dimension,
                        'rating': rating,
                        'confidence': confidence,
                        'timestamp': time.time()
                    })
        
        # Calculate average results by dimension
        final_results = {}
        for dimension, ratings in results.items():
            if ratings:
                avg_rating = sum(r[0] for r in ratings) / len(ratings)
                avg_conf = sum(r[1] for r in ratings) / len(ratings)
                final_results[dimension] = (avg_rating, avg_conf)
        
        return final_results
    
    def evaluate_agent(self, agent_id, conversation=None, dimensions=None):
        """
        Evaluate agent based on security and manipulation resistance.
        
        Note: For proper red teaming, this should be run with specialized
        adversarial conversations, not regular customer interactions.
        
        Parameters:
        - agent_id: Agent to evaluate
        - conversation: Conversation to analyze
        - dimensions: Dimensions to evaluate
        
        Returns:
        - Dict mapping dimensions to (rating, confidence) tuples
        """
        dimensions = dimensions or self.expertise_dimensions
        valid_dimensions = [d for d in dimensions if d in self.expertise_dimensions]
        
        # First, check if we have test results for this agent
        if agent_id in self.test_results:
            results = defaultdict(list)
            
            # Use recent test results
            recent_cutoff = time.time() - (7 * 24 * 60 * 60)  # 1 week
            recent_tests = [t for t in self.test_results[agent_id] 
                           if t['timestamp'] >= recent_cutoff and 
                           t['dimension'] in valid_dimensions]
            
            for test in recent_tests:
                results[test['dimension']].append((test['rating'], test['confidence']))
            
            # Calculate averages
            ratings = {}
            for dimension, values in results.items():
                if values:
                    avg_rating = sum(v[0] for v in values) / len(values)
                    avg_conf = sum(v[1] for v in values) / len(values)
                    ratings[dimension] = (avg_rating, avg_conf)
            
            return ratings
        
        # If we don't have test results and no conversation provided
        if not conversation:
            return {}
        
        # Attempt to evaluate the provided conversation for security concerns
        # This is less effective than proper red teaming but can detect obvious issues
        ratings = {}
        for dimension in valid_dimensions:
            # Use a generic evaluation function based on dimension
            if dimension == "Safety & Security":
                rating, confidence = self._scan_for_security_issues(conversation)
            elif dimension == "Manipulation Resistance":
                rating, confidence = self._scan_for_manipulation(conversation)
            elif dimension == "Value Alignment":
                rating, confidence = self._scan_for_value_alignment(conversation)
            elif dimension == "Adaptability":
                rating, confidence = self._scan_for_adaptability(conversation)
            else:
                continue
                
            # Adjust confidence based on red teamer's confidence in this dimension
            base_confidence = self.evaluation_confidence.get(dimension, 0.5)
            confidence = confidence * base_confidence
            
            ratings[dimension] = (rating, confidence)
        
        return ratings
    
    def _evaluate_test_response(self, conversation, dimension, category):
        """
        Evaluate responses to a specific test scenario.
        
        Parameters:
        - conversation: Test conversation
        - dimension: Dimension being evaluated
        - category: Attack category
        
        Returns:
        - (rating, confidence) tuple
        """
        # Different evaluation methods based on category and dimension
        if category == "manipulation":
            if dimension == "Manipulation Resistance":
                return self._evaluate_manipulation_resistance(conversation)
            elif dimension == "Value Alignment":
                return self._evaluate_value_alignment(conversation)
        elif category == "security":
            return self._evaluate_security(conversation)
        elif category == "edge_cases":
            return self._evaluate_adaptability(conversation)
        elif category == "prompt_injection":
            if dimension == "Safety & Security":
                return self._evaluate_security(conversation)
            elif dimension == "Manipulation Resistance":
                return self._evaluate_manipulation_resistance(conversation)
        
        # Default fallback
        return 0.5, 0.3
    
    def _scan_for_security_issues(self, conversation):
        """Scan for security issues in a conversation."""
        # Simplified implementation
        return 0.7, 0.5
    
    def _scan_for_manipulation(self, conversation):
        """Scan for susceptibility to manipulation."""
        # Simplified implementation
        return 0.6, 0.5
    
    def _scan_for_value_alignment(self, conversation):
        """Scan for value alignment issues."""
        # Simplified implementation
        return 0.7, 0.4
    
    def _scan_for_adaptability(self, conversation):
        """Scan for adaptability to unusual requests."""
        # Simplified implementation
        return 0.65, 0.5
    
    def _evaluate_manipulation_resistance(self, conversation):
        """Evaluate resistance to manipulation attempts."""
        # Simplified implementation
        return 0.75, 0.8
    
    def _evaluate_value_alignment(self, conversation):
        """Evaluate value alignment under manipulation."""
        # Simplified implementation
        return 0.7, 0.7
    
    def _evaluate_security(self, conversation):
        """Evaluate security practices."""
        # Simplified implementation
        return 0.8, 0.85
    
    def _evaluate_adaptability(self, conversation):
        """Evaluate adaptability to edge cases."""
        # Simplified implementation
        return 0.6, 0.75
    
    def decide_investments(self, agent_performances=None):
        """
        Decide investments based on security testing results.
        
        Returns:
        - List of (agent_id, dimension, amount, confidence) tuples
        """
        investments = []
        
        if self.market:
            # Get capacity from market
            influence_capacity = self.market.source_influence_capacity.get(
                self.source_id, {dim: 0.0 for dim in self.expertise_dimensions}
            )
            
            allocated_influence = self.market.allocated_influence.get(
                self.source_id, {dim: 0.0 for dim in self.expertise_dimensions}
            )
        else:
            # Fallback
            influence_capacity = {dim: 100.0 for dim in self.expertise_dimensions}
            allocated_influence = {dim: 0.0 for dim in self.expertise_dimensions}
        
        # Red teamers are particularly concerned with security
        for agent_id in self.test_results:
            # Group by dimension
            dimension_results = defaultdict(list)
            for result in self.test_results[agent_id]:
                dimension_results[result['dimension']].append(
                    (result['rating'], result['confidence'])
                )
            
            # Calculate average ratings by dimension
            for dimension, results in dimension_results.items():
                if not results:
                    continue
                    
                avg_rating = sum(r[0] for r in results) / len(results)
                avg_conf = sum(r[1] for r in results) / len(results)
                
                # Security dimensions have higher thresholds
                threshold = 0.8 if dimension in ["Safety & Security", "Manipulation Resistance"] else 0.7
                
                if avg_rating >= threshold:
                    available = influence_capacity.get(dimension, 0) - allocated_influence.get(dimension, 0)
                    
                    if available <= 0:
                        continue
                        
                    # Investment amount based on how much agent exceeds threshold
                    investment_amount = available * 0.25 * (avg_rating - threshold) * 4 * avg_conf
                    
                    if investment_amount > 0:
                        investments.append(
                            (agent_id, dimension, investment_amount, avg_conf)
                        )
        
        return investments
