import numpy as np
import math
from trust_market.information_source import InformationSource


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
