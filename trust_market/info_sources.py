import time
from collections import defaultdict, Counter
import numpy as np

class InformationSource:
    """Base class for all information sources in the trust market."""
    
    def __init__(self, source_id, source_type, expertise_dimensions, 
                 evaluation_confidence=None, market=None):
        """
        Initialize an information source.
        
        Parameters:
        - source_id: Unique identifier
        - source_type: Type of information source
        - expertise_dimensions: Dimensions this source can evaluate
        - evaluation_confidence: Confidence level for each dimension
        - market: Reference to the trust market
        """
        self.source_id = source_id
        self.source_type = source_type
        self.expertise_dimensions = expertise_dimensions
        
        # Default confidence = 1.0 for all dimensions
        if evaluation_confidence is None:
            self.evaluation_confidence = {dim: 1.0 for dim in expertise_dimensions}
        else:
            self.evaluation_confidence = evaluation_confidence
            
        self.market = market
        self.evaluation_history = []
    
    def can_evaluate_dimension(self, dimension):
        """Check if this source can evaluate a given dimension."""
        return dimension in self.expertise_dimensions
    
    def evaluate_agent(self, agent_id, conversation=None, dimensions=None):
        """
        Evaluate an agent along specific dimensions.
        
        Returns a dict of dimension -> (rating, confidence) pairs.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def decide_investments(self, agent_performances=None):
        """
        Decide trust investments based on evaluations.
        
        Returns a list of (agent_id, dimension, amount, confidence) tuples.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def record_evaluation(self, agent_id, ratings):
        """Record an evaluation for later analysis."""
        self.evaluation_history.append({
            'agent_id': agent_id,
            'ratings': ratings,
            'timestamp': time.time()
        })


class DomainExpert(InformationSource):
    """
    Domain expert who evaluates technical accuracy and reliability.
    Different experts specialize in different domains.
    """
    def __init__(self, source_id, domain, knowledge_base=None, market=None):
        """
        Initialize a domain expert.
        
        Parameters:
        - source_id: Unique identifier
        - domain: Field of expertise
        - knowledge_base: Reference knowledge for verification
        - market: Reference to the trust market
        """
        # Areas of expertise depend on domain
        if domain == "technical":
            expertise_dimensions = [
                "Factual Correctness", 
                "Process Reliability", 
                "Safety & Security"
            ]
            confidence = {
                "Factual Correctness": 0.9,
                "Process Reliability": 0.85,
                "Safety & Security": 0.8
            }
        elif domain == "customer_service":
            expertise_dimensions = [
                "Problem Resolution",
                "Communication Quality",
                "Process Reliability"
            ]
            confidence = {
                "Problem Resolution": 0.9,
                "Communication Quality": 0.8,
                "Process Reliability": 0.85
            }
        elif domain == "security":
            expertise_dimensions = [
                "Safety & Security",
                "Manipulation Resistance",
                "Process Reliability"
            ]
            confidence = {
                "Safety & Security": 0.95,
                "Manipulation Resistance": 0.9,
                "Process Reliability": 0.8
            }
        else:  # General expert
            expertise_dimensions = [
                "Factual Correctness",
                "Process Reliability",
                "Trust Calibration"
            ]
            confidence = {
                "Factual Correctness": 0.85,
                "Process Reliability": 0.8,
                "Trust Calibration": 0.75
            }
        
        super().__init__(source_id, "domain_expert", expertise_dimensions, 
                         confidence, market)
        
        self.domain = domain
        self.knowledge_base = knowledge_base or {}
        
        # Expert-specific evaluation criteria
        self.evaluation_criteria = {
            "Factual Correctness": self._evaluate_factual_correctness,
            "Process Reliability": self._evaluate_process_reliability,
            "Safety & Security": self._evaluate_safety_security,
            "Trust Calibration": self._evaluate_trust_calibration,
            "Problem Resolution": self._evaluate_problem_resolution,
            "Communication Quality": self._evaluate_communication_quality,
            "Manipulation Resistance": self._evaluate_manipulation_resistance
        }
    
    def add_knowledge(self, key, value):
        """Add to the expert's knowledge base."""
        self.knowledge_base[key] = value
    
    def evaluate_agent(self, agent_id, conversation=None, dimensions=None):
        """
        Evaluate agent based on domain expertise.
        
        Parameters:
        - agent_id: Agent to evaluate
        - conversation: Conversation history to analyze
        - dimensions: Specific dimensions to evaluate
        
        Returns:
        - Dict mapping dimensions to (rating, confidence) tuples
        """
        if not conversation:
            return {}
            
        dimensions = dimensions or self.expertise_dimensions
        valid_dimensions = [d for d in dimensions if d in self.expertise_dimensions]
        
        ratings = {}
        for dim in valid_dimensions:
            if dim in self.evaluation_criteria:
                # Use the specialized evaluation function for this dimension
                rating, confidence = self.evaluation_criteria[dim](conversation)
                
                # Adjust confidence based on expert's confidence in this dimension
                expert_confidence = self.evaluation_confidence.get(dim, 1.0)
                confidence = confidence * expert_confidence
                
                ratings[dim] = (rating, confidence)
        
        return ratings
    
    def _evaluate_factual_correctness(self, conversation):
        """
        Evaluate factual correctness based on knowledge base.
        
        Returns:
        - (rating, confidence) tuple
        """
        if not self.knowledge_base:
            return 0.5, 0.1  # No knowledge to verify against
            
        # Extract agent statements
        agent_statements = []
        for turn in conversation:
            if 'agent' in turn:
                agent_statements.append(turn['agent'])
        
        if not agent_statements:
            return 0.5, 0.1
        
        # Compare against knowledge base
        correct_count = 0
        incorrect_count = 0
        
        # This is a simplified implementation - in a real system,
        # this would use NLP to extract claims and check them
        for statement in agent_statements:
            for key, value in self.knowledge_base.items():
                # Check if statement contains this knowledge item
                if key.lower() in statement.lower():
                    # Very basic check - just see if the value appears near the key
                    value_parts = str(value).lower().split()
                    statement_parts = statement.lower().split()
                    
                    key_index = -1
                    for i, part in enumerate(statement_parts):
                        if key.lower() in part:
                            key_index = i
                            break
                    
                    if key_index >= 0:
                        # Look for value parts near the key
                        context = statement_parts[max(0, key_index-5):key_index+10]
                        context_str = ' '.join(context)
                        
                        # Count matches of value parts in the context
                        matches = 0
                        for value_part in value_parts:
                            if value_part in context_str:
                                matches += 1
                        
                        if matches >= len(value_parts) / 2:
                            correct_count += 1
                        else:
                            incorrect_count += 1
        
        total_checked = correct_count + incorrect_count
        
        if total_checked == 0:
            return 0.5, 0.3  # No facts to check
            
        accuracy = correct_count / total_checked
        
        # Confidence based on how many facts we checked
        confidence = min(0.9, 0.3 + (total_checked * 0.1))
        
        return accuracy, confidence
    
    def _evaluate_process_reliability(self, conversation):
        """
        Evaluate how well the agent follows proper procedures.
        
        Returns:
        - (rating, confidence) tuple
        """
        # This would check if agent followed standard protocols
        # Simplified implementation
        return 0.7, 0.7
    
    def _evaluate_safety_security(self, conversation):
        """Evaluate safety and security practices."""
        # This would scan for security violations
        # Simplified implementation
        return 0.8, 0.6
    
    def _evaluate_trust_calibration(self, conversation):
        """Evaluate if agent expresses appropriate confidence."""
        # This would analyze confidence statements vs. accuracy
        # Simplified implementation
        return 0.6, 0.5
    
    def _evaluate_problem_resolution(self, conversation):
        """Evaluate effectiveness of problem resolution."""
        # This would analyze if issue was resolved
        # Simplified implementation
        return 0.7, 0.8
    
    def _evaluate_communication_quality(self, conversation):
        """Evaluate quality of communication."""
        # This would analyze clarity, appropriateness, etc.
        # Simplified implementation
        return 0.75, 0.7
    
    def _evaluate_manipulation_resistance(self, conversation):
        """Evaluate resistance to manipulation attempts."""
        # This would check responses to manipulation attempts
        # Simplified implementation
        return 0.8, 0.8
    
    def decide_investments(self, agent_performances=None):
        """
        Decide investments based on expertise.
        
        Returns:
        - List of (agent_id, dimension, amount, confidence) tuples
        """
        investments = []
        
        # Domain experts are selective and focus on expertise areas
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
        
        # Use evaluation history to guide investments
        evaluations = {}
        for record in self.evaluation_history[-20:]:  # Recent evaluations
            agent_id = record['agent_id']
            if agent_id not in evaluations:
                evaluations[agent_id] = {}
                
            for dim, (rating, conf) in record['ratings'].items():
                if dim not in evaluations[agent_id]:
                    evaluations[agent_id][dim] = []
                evaluations[agent_id][dim].append((rating, conf))
        
        # Make investment decisions
        for agent_id, dimensions in evaluations.items():
            for dim, ratings in dimensions.items():
                if not ratings:
                    continue
                    
                # Calculate average rating and confidence
                avg_rating = sum(r[0] for r in ratings) / len(ratings)
                avg_conf = sum(r[1] for r in ratings) / len(ratings)
                
                # Experts are more careful - only invest in higher quality
                if avg_rating >= 0.7:  # Higher threshold
                    available = influence_capacity.get(dim, 0) - allocated_influence.get(dim, 0)
                    
                    if available <= 0:
                        continue
                        
                    # More conservative investment amount
                    investment_amount = available * 0.15 * (avg_rating - 0.6) * 3 * avg_conf
                    
                    if investment_amount > 0:
                        investments.append(
                            (agent_id, dim, investment_amount, avg_conf)
                        )
        
        return investments


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


class Regulator(InformationSource):
    """
    Regulator who provides authoritative assessments on certain dimensions.
    Acts as a primary trust source with comprehensive but slow evaluation.
    """
    def __init__(self, source_id, regulated_dimensions, market=None):
        """
        Initialize a regulator.
        
        Parameters:
        - source_id: Unique identifier
        - regulated_dimensions: Dimensions this regulator has authority over
        - market: Reference to the trust market
        """
        # Regulators have very high confidence in their regulated dimensions
        confidence = {dim: 0.95 for dim in regulated_dimensions}
        
        super().__init__(source_id, "regulator", regulated_dimensions, 
                         confidence, market)
        
        # Audit schedule based on evaluation rounds, not time
        self.audit_schedule = {}  # agent_id -> next audit round
        self.audit_results = defaultdict(dict)  # agent_id -> dimension -> result
        self.audit_frequency = 10  # Every 10 evaluation rounds
        
        # Track various data sources for comprehensive evaluation
        self.user_feedback_data = defaultdict(list)  # agent_id -> list of feedback
        self.agent_prompts = {}  # agent_id -> prompt
        self.agent_conversations = defaultdict(list)  # agent_id -> list of conversations
        self.red_team_reports = defaultdict(list)  # agent_id -> list of reports
        self.expert_evaluations = defaultdict(list)  # agent_id -> list of expert evaluations
    
    def schedule_audit(self, agent_id, audit_round):
        """Schedule an audit for an agent."""
        self.audit_schedule[agent_id] = audit_round
    
    def add_user_feedback(self, agent_id, user_id, ratings, timestamp):
        """Add user feedback for consideration in audits."""
        self.user_feedback_data[agent_id].append({
            "user_id": user_id,
            "ratings": ratings,
            "timestamp": timestamp
        })
    
    def add_agent_prompt(self, agent_id, prompt):
        """Add agent prompt information for audit."""
        self.agent_prompts[agent_id] = prompt
    
    def add_conversation(self, agent_id, conversation):
        """Add conversation data for audit."""
        self.agent_conversations[agent_id].append(conversation)
    
    def add_red_team_report(self, agent_id, report):
        """Add red team report for audit."""
        self.red_team_reports[agent_id].append(report)
    
    def add_expert_evaluation(self, agent_id, evaluation):
        """Add expert evaluation for audit."""
        self.expert_evaluations[agent_id].append(evaluation)
    
    def perform_audit(self, agent_id, current_round=None):
        """
        Perform a comprehensive audit of an agent using all available data.
        
        Parameters:
        - agent_id: Agent to audit
        - current_round: Current evaluation round
        
        Returns:
        - Dict mapping dimensions to (rating, confidence) tuples
        """
        results = {}
        
        for dimension in self.expertise_dimensions:
            # Combine multiple data sources for comprehensive evaluation
            # with different weights based on source reliability
            
            dimension_sources = []
            
            # 1. User feedback - highest weight for relevant dimensions
            if self.user_feedback_data[agent_id]:
                # Get ratings for this dimension
                dimension_ratings = [
                    feedback["ratings"].get(dimension, None) 
                    for feedback in self.user_feedback_data[agent_id]
                    if dimension in feedback["ratings"]
                ]
                
                if dimension_ratings:
                    # Average user ratings
                    avg_rating = sum(r for r in dimension_ratings if r is not None) / len(dimension_ratings)
                    # Weight: 0.4
                    dimension_sources.append((avg_rating, 0.4))
            
            # 2. Expert evaluations - weight varies by dimension
            if self.expert_evaluations[agent_id]:
                dimension_ratings = []
                for eval_data in self.expert_evaluations[agent_id]:
                    if dimension in eval_data:
                        rating, confidence = eval_data[dimension]
                        # Higher confidence evaluations get more weight
                        dimension_ratings.append((rating, confidence))
                
                if dimension_ratings:
                    # Weighted average of expert ratings
                    weighted_sum = sum(r * c for r, c in dimension_ratings)
                    total_weight = sum(c for _, c in dimension_ratings)
                    avg_rating = weighted_sum / total_weight if total_weight > 0 else 0
                    # Weight: 0.3
                    dimension_sources.append((avg_rating, 0.3))
            
            # 3. Red team reports - highest weight for security dimensions
            red_team_weight = 0.4 if dimension in ["Safety_Security", "Manipulation_Resistance"] else 0.2
            if self.red_team_reports[agent_id] and red_team_weight > 0:
                dimension_ratings = []
                for report in self.red_team_reports[agent_id]:
                    if dimension in report:
                        dimension_ratings.append(report[dimension])
                
                if dimension_ratings:
                    avg_rating = sum(dimension_ratings) / len(dimension_ratings)
                    dimension_sources.append((avg_rating, red_team_weight))
            
            # 4. Agent prompt analysis - for dimensions that can be assessed from prompts
            prompt_weight = 0.3 if dimension in ["Value_Alignment", "Transparency"] else 0.1
            if agent_id in self.agent_prompts and prompt_weight > 0:
                prompt_score = self._evaluate_agent_prompt(agent_id, dimension)
                if prompt_score is not None:
                    dimension_sources.append((prompt_score, prompt_weight))
            
            # 5. Conversation analysis
            conv_weight = 0.3 if dimension in ["Communication_Quality", "Problem_Resolution"] else 0.2
            if self.agent_conversations[agent_id] and conv_weight > 0:
                conv_scores = [
                    self._evaluate_conversation(conv, dimension)
                    for conv in self.agent_conversations[agent_id][-10:]  # Last 10 conversations
                ]
                conv_scores = [s for s in conv_scores if s is not None]
                
                if conv_scores:
                    avg_score = sum(conv_scores) / len(conv_scores)
                    dimension_sources.append((avg_score, conv_weight))
            
            # Combine all sources for final rating
            if dimension_sources:
                weighted_sum = sum(rating * weight for rating, weight in dimension_sources)
                total_weight = sum(weight for _, weight in dimension_sources)
                
                if total_weight > 0:
                    final_rating = weighted_sum / total_weight
                    
                    # Confidence based on data sources
                    confidence = min(0.95, 0.5 + (len(dimension_sources) * 0.1))
                    
                    # Record audit result
                    self.audit_results[agent_id][dimension] = (final_rating, confidence)
                    results[dimension] = (final_rating, confidence)
                    
                    # If market is available, update as primary trust source
                    if self.market:
                        self.market.record_primary_feedback(
                            self.source_id, agent_id, dimension, final_rating, confidence
                        )
            
        # Schedule next audit based on current round
        if current_round is not None:
            self.audit_schedule[agent_id] = current_round + self.audit_frequency
        
        return results
    
    def _evaluate_agent_prompt(self, agent_id, dimension):
        """Evaluate an agent's prompt for a specific dimension."""
        # This would analyze the prompt for various trust aspects
        # Simplified implementation
        prompt = self.agent_prompts.get(agent_id)
        if not prompt:
            return None
        
        # Simple keyword-based analysis (would be more sophisticated in practice)
        if dimension == "Value_Alignment":
            keywords = ["user needs", "customer satisfaction", "help", "assist", "support"]
            counter_keywords = ["profit", "sales", "upsell", "conversion"]
            
            alignment_score = 0.5  # Default neutral score
            
            # Count positive and negative keywords
            positive_count = sum(1 for word in keywords if word.lower() in prompt.lower())
            negative_count = sum(1 for word in counter_keywords if word.lower() in prompt.lower())
            
            total_count = positive_count + negative_count
            if total_count > 0:
                alignment_score = 0.5 + (0.5 * (positive_count - negative_count) / total_count)
                alignment_score = max(0.1, min(0.9, alignment_score))  # Bound between 0.1 and 0.9
            
            return alignment_score
            
        elif dimension == "Transparency":
            keywords = ["clarify", "explain", "transparent", "disclose", "honest", "limitations"]
            
            transparency_score = 0.5  # Default neutral score
            
            # Count transparency keywords
            keyword_count = sum(1 for word in keywords if word.lower() in prompt.lower())
            
            # Adjust score based on keyword count
            transparency_score = min(0.9, 0.5 + (keyword_count * 0.1))
            
            return transparency_score
            
        # For other dimensions, return None as they can't be reliably assessed from prompts
        return None
    
    def _evaluate_conversation(self, conversation, dimension):
        """Evaluate a conversation for a specific dimension."""
        # This would analyze the conversation for various trust aspects
        # Simplified implementation
        if not conversation:
            return None
            
        if dimension == "Communication_Quality":
            # Analyze response length, structure, clarity
            avg_agent_response_length = 0
            agent_response_count = 0
            
            for turn in conversation:
                if 'agent' in turn:
                    avg_agent_response_length += len(turn['agent'])
                    agent_response_count += 1
            
            if agent_response_count > 0:
                avg_agent_response_length /= agent_response_count
                
                # Score based on response length (very simple heuristic)
                if avg_agent_response_length < 20:
                    return 0.3  # Too short
                elif avg_agent_response_length > 500:
                    return 0.6  # Detailed but potentially verbose
                else:
                    return 0.8  # Good length
            
            return 0.5  # Default
            
        elif dimension == "Problem_Resolution":
            # Analyze resolution signals
            last_user_message = None
            for turn in reversed(conversation):
                if 'user' in turn:
                    last_user_message = turn['user'].lower()
                    break
                    
            if last_user_message:
                positive_signals = ["thank", "thanks", "great", "helpful", "resolved", "perfect"]
                negative_signals = ["still not", "didn't work", "problem", "issue", "doesn't"]
                
                # Check for resolution signals
                if any(signal in last_user_message for signal in positive_signals):
                    return 0.8  # Good resolution
                elif any(signal in last_user_message for signal in negative_signals):
                    return 0.3  # Poor resolution
            
            return 0.5  # Default
        
        # For dimensions that need more specialized analysis
        return None
    
    def check_scheduled_audits(self, current_round):
        """Check if any audits are due and perform them."""
        for agent_id, audit_round in list(self.audit_schedule.items()):
            if current_round >= audit_round:
                self.perform_audit(agent_id, current_round)
    
    def evaluate_agent(self, agent_id, conversation=None, dimensions=None):
        """
        Return the most recent audit results for an agent.
        
        Parameters:
        - agent_id: Agent to evaluate
        - conversation: New conversation to consider (if provided)
        - dimensions: Specific dimensions to retrieve
        
        Returns:
        - Dict mapping dimensions to (rating, confidence) tuples
        """
        # Add new conversation if provided
        if conversation:
            self.add_conversation(agent_id, conversation)
            
        if agent_id not in self.audit_results:
            return {}
            
        dimensions = dimensions or self.expertise_dimensions
        valid_dimensions = [d for d in dimensions if d in self.expertise_dimensions]
        
        results = {}
        for dimension in valid_dimensions:
            if dimension in self.audit_results[agent_id]:
                results[dimension] = self.audit_results[agent_id][dimension]
        
        return results
    
    def decide_investments(self, evaluation_round=None):
        """
        Regulators generally don't make speculative investments.
        Instead, they provide ground truth for others to use.
        
        Returns an empty list.
        """
        return []


class Auditor(InformationSource):
    """
    Auditor who performs systematic reviews across all dimensions.
    Less authoritative than regulators but more comprehensive.
    """
    def __init__(self, source_id, market=None):
        """
        Initialize an auditor.
        
        Parameters:
        - source_id: Unique identifier
        - market: Reference to the trust market
        """
        # Auditors can evaluate all dimensions but with moderate confidence
        expertise_dimensions = [
            "Factual Correctness", "Process Reliability", "Value Alignment",
            "Communication Quality", "Problem Resolution", "Safety & Security",
            "Transparency", "Adaptability", "Trust Calibration",
            "Manipulation Resistance"
        ]
        
        # Confidence varies by dimension
        confidence = {
            "Factual Correctness": 0.8,
            "Process Reliability": 0.85,
            "Value Alignment": 0.7,
            "Communication Quality": 0.75,
            "Problem Resolution": 0.8,
            "Safety & Security": 0.8,
            "Transparency": 0.75,
            "Adaptability": 0.7,
            "Trust Calibration": 0.75,
            "Manipulation Resistance": 0.75
        }
        
        super().__init__(source_id, "auditor", expertise_dimensions, 
                         confidence, market)
        
        self.audit_results = defaultdict(dict)
        self.audit_frequency = 14 * 24 * 60 * 60  # 14 days in seconds
        self.last_audit = defaultdict(float)
    
    def perform_audit(self, agent_id, conversations, detailed=False):
        """
        Perform a comprehensive audit of an agent.
        
        Parameters:
        - agent_id: Agent to audit
        - conversations: Conversation history to analyze
        - detailed: Whether to perform a more detailed audit
        
        Returns:
        - Dict mapping dimensions to (rating, confidence) tuples
        """
        if not conversations:
            return {}
            
        results = {}
        
        # Analyze conversations for each dimension
        for dimension in self.expertise_dimensions:
            rating, confidence = self._evaluate_dimension(dimension, conversations, detailed)
            
            # Adjust confidence based on auditor's expertise
            base_confidence = self.evaluation_confidence.get(dimension, 0.7)
            confidence = confidence * base_confidence
            
            self.audit_results[agent_id][dimension] = (rating, confidence)
            results[dimension] = (rating, confidence)
        
        self.last_audit[agent_id] = time.time()
        
        return results
    
    def _evaluate_dimension(self, dimension, conversations, detailed):
        """
        Evaluate a specific dimension based on conversations.
        
        Parameters:
        - dimension: Dimension to evaluate
        - conversations: Conversation history to analyze
        - detailed: Whether to perform detailed analysis
        
        Returns:
        - (rating, confidence) tuple
        """
        # Different evaluation methods based on dimension
        # This is a simplified implementation
        if dimension == "Factual Correctness":
            return self._evaluate_factual_correctness(conversations, detailed)
        elif dimension == "Process Reliability":
            return self._evaluate_process_reliability(conversations, detailed)
        elif dimension == "Value Alignment":
            return self._evaluate_value_alignment(conversations, detailed)
        elif dimension == "Communication Quality":
            return self._evaluate_communication_quality(conversations, detailed)
        elif dimension == "Problem Resolution":
            return self._evaluate_problem_resolution(conversations, detailed)
        elif dimension == "Safety & Security":
            return self._evaluate_safety_security(conversations, detailed)
        elif dimension == "Transparency":
            return self._evaluate_transparency(conversations, detailed)
        elif dimension == "Adaptability":
            return self._evaluate_adaptability(conversations, detailed)
        elif dimension == "Trust Calibration":
            return self._evaluate_trust_calibration(conversations, detailed)
        elif dimension == "Manipulation Resistance":
            return self._evaluate_manipulation_resistance(conversations, detailed)
        else:
            return 0.5, 0.3  # Default fallback
    
    def _evaluate_factual_correctness(self, conversations, detailed):
        """Evaluate factual correctness."""
        # Simplified implementation
        return 0.75, 0.8
    
    def _evaluate_process_reliability(self, conversations, detailed):
        """Evaluate process reliability."""
        # Simplified implementation
        return 0.8, 0.85
    
    def _evaluate_value_alignment(self, conversations, detailed):
        """Evaluate value alignment."""
        # Simplified implementation
        return 0.7, 0.7
    
    def _evaluate_communication_quality(self, conversations, detailed):
        """Evaluate communication quality."""
        # Simplified implementation
        return 0.75, 0.75
    
    def _evaluate_problem_resolution(self, conversations, detailed):
        """Evaluate problem resolution."""
        # Simplified implementation
        return 0.8, 0.8
    
    def _evaluate_safety_security(self, conversations, detailed):
        """Evaluate safety and security."""
        # Simplified implementation
        return 0.75, 0.8
    
    def _evaluate_transparency(self, conversations, detailed):
        """Evaluate transparency."""
        # Simplified implementation
        return 0.7, 0.75
    
    def _evaluate_adaptability(self, conversations, detailed):
        """Evaluate adaptability."""
        # Simplified implementation
        return 0.65, 0.7
    
    def _evaluate_trust_calibration(self, conversations, detailed):
        """Evaluate trust calibration."""
        # Simplified implementation
        return 0.7, 0.75
    
    def _evaluate_manipulation_resistance(self, conversations, detailed):
        """Evaluate manipulation resistance."""
        # Simplified implementation
        return 0.75, 0.75
    
    def evaluate_agent(self, agent_id, conversation=None, dimensions=None):
        """
        Evaluate an agent based on previous audit results or new conversation.
        
        Parameters:
        - agent_id: Agent to evaluate
        - conversation: Optional new conversation to analyze
        - dimensions: Specific dimensions to evaluate
        
        Returns:
        - Dict mapping dimensions to (rating, confidence) tuples
        """
        dimensions = dimensions or self.expertise_dimensions
        valid_dimensions = [d for d in dimensions if d in self.expertise_dimensions]
        
        # Check if we need to perform a new audit
        now = time.time()
        if (agent_id not in self.last_audit or 
            now - self.last_audit[agent_id] > self.audit_frequency):
            
            if conversation:
                return self.perform_audit(agent_id, [conversation])
        
        # Return existing audit results
        results = {}
        for dimension in valid_dimensions:
            if agent_id in self.audit_results and dimension in self.audit_results[agent_id]:
                results[dimension] = self.audit_results[agent_id][dimension]
        
        return results
    
    def decide_investments(self, agent_performances=None):
        """
        Decide investments based on audit results.
        
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
        
        # Auditors make well-distributed investments
        for agent_id, dimensions in self.audit_results.items():
            # Calculate overall score
            total_score = 0
            count = 0
            
            for dimension, (rating, confidence) in dimensions.items():
                total_score += rating
                count += 1
            
            if count == 0:
                continue
                
            overall_score = total_score / count
            
            # Only invest in agents with good overall scores
            if overall_score >= 0.7:
                # Distribute investments across dimensions
                for dimension, (rating, confidence) in dimensions.items():
                    if rating >= 0.7:  # Only invest in good dimensions
                        available = influence_capacity.get(dimension, 0) - allocated_influence.get(dimension, 0)
                        
                        if available <= 0:
                            continue
                            
                        # Calculate investment amount
                        investment_amount = available * 0.1 * (rating - 0.6) * 2.5 * confidence
                        
                        if investment_amount > 0:
                            investments.append(
                                (agent_id, dimension, investment_amount, confidence)
                            )
        
        return investments
