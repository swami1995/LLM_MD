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

