"""
Evaluation metrics for the RAG system.
This module provides functionality to evaluate the quality of RAG responses.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.vector.processor import generate_embedding

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """
    Evaluator for RAG system responses.
    """
    
    def __init__(self):
        """Initialize the RAG evaluator."""
        self.metrics_history: List[Dict[str, Any]] = []
    
    def evaluate_response(
        self, 
        query: str, 
        response: str, 
        sources: List[Dict[str, Any]],
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a RAG response.
        
        Args:
            query: User query
            response: Generated response
            sources: Sources used for the response
            ground_truth: Optional ground truth answer for comparison
            
        Returns:
            Dictionary with evaluation metrics
        """
        start_time = time.time()
        
        # Calculate metrics
        metrics = {}
        
        # Response length
        metrics['response_length'] = len(response.split())
        
        # Source count
        metrics['source_count'] = len(sources)
        
        # Source relevance (average similarity score)
        if sources:
            metrics['avg_source_relevance'] = sum(s.get('similarity', 0) for s in sources) / len(sources)
        else:
            metrics['avg_source_relevance'] = 0.0
        
        # Response-query relevance
        metrics['response_query_relevance'] = self._calculate_semantic_similarity(query, response)
        
        # Source citation analysis
        citation_metrics = self._analyze_citations(response, sources)
        metrics.update(citation_metrics)
        
        # Ground truth comparison (if available)
        if ground_truth:
            truth_metrics = self._compare_to_ground_truth(response, ground_truth)
            metrics.update(truth_metrics)
        
        # Calculate overall quality score
        metrics['quality_score'] = self._calculate_quality_score(metrics)
        
        # Add timestamp
        metrics['timestamp'] = time.time()
        metrics['evaluation_time'] = time.time() - start_time
        
        # Store metrics in history
        self.metrics_history.append({
            'query': query,
            'metrics': metrics
        })
        
        return metrics
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        try:
            embedding1 = generate_embedding(text1)
            embedding2 = generate_embedding(text2)
            
            embedding1_reshaped = np.array(embedding1).reshape(1, -1)
            embedding2_reshaped = np.array(embedding2).reshape(1, -1)
            
            similarity = cosine_similarity(embedding1_reshaped, embedding2_reshaped)[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _analyze_citations(self, response: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze source citations in the response.
        
        Args:
            response: Generated response
            sources: Sources used for the response
            
        Returns:
            Dictionary with citation metrics
        """
        metrics = {}
        
        # Count explicit citations (Article X)
        citation_count = 0
        for i in range(1, len(sources) + 1):
            if f"Article {i}" in response or f"article {i}" in response:
                citation_count += 1
        
        metrics['citation_count'] = citation_count
        
        # Calculate citation ratio
        if sources:
            metrics['citation_ratio'] = citation_count / len(sources)
        else:
            metrics['citation_ratio'] = 0.0
        
        return metrics
    
    def _compare_to_ground_truth(self, response: str, ground_truth: str) -> Dict[str, Any]:
        """
        Compare response to ground truth.
        
        Args:
            response: Generated response
            ground_truth: Ground truth answer
            
        Returns:
            Dictionary with comparison metrics
        """
        metrics = {}
        
        # Semantic similarity to ground truth
        metrics['ground_truth_similarity'] = self._calculate_semantic_similarity(response, ground_truth)
        
        # Word overlap (Jaccard similarity)
        response_words = set(response.lower().split())
        truth_words = set(ground_truth.lower().split())
        
        if response_words or truth_words:
            intersection = response_words.intersection(truth_words)
            union = response_words.union(truth_words)
            metrics['word_overlap'] = len(intersection) / len(union)
        else:
            metrics['word_overlap'] = 0.0
        
        return metrics
    
    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate an overall quality score from individual metrics.
        
        Args:
            metrics: Dictionary with evaluation metrics
            
        Returns:
            Quality score (0-1)
        """
        # Define weights for different metrics
        weights = {
            'response_query_relevance': 0.4,
            'avg_source_relevance': 0.3,
            'citation_ratio': 0.3
        }
        
        # Calculate weighted sum
        score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
        
        # If we have ground truth comparison, adjust the score
        if 'ground_truth_similarity' in metrics:
            # Give 50% weight to ground truth similarity
            score = 0.5 * score + 0.5 * metrics['ground_truth_similarity']
        
        return min(1.0, max(0.0, score))
    
    def get_average_metrics(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """
        Get average metrics over the evaluation history.
        
        Args:
            last_n: Only consider the last N evaluations (None for all)
            
        Returns:
            Dictionary with average metrics
        """
        if not self.metrics_history:
            return {}
        
        # Select metrics to average
        if last_n is not None and last_n > 0:
            metrics_list = [entry['metrics'] for entry in self.metrics_history[-last_n:]]
        else:
            metrics_list = [entry['metrics'] for entry in self.metrics_history]
        
        # Calculate averages
        avg_metrics = {}
        for key in metrics_list[0].keys():
            # Skip non-numeric metrics
            if key in ['timestamp']:
                continue
            
            # Calculate average
            values = [m.get(key, 0) for m in metrics_list if key in m]
            if values:
                avg_metrics[f'avg_{key}'] = sum(values) / len(values)
        
        return avg_metrics
    
    def clear_history(self) -> None:
        """Clear the evaluation history."""
        self.metrics_history = []
        logger.info("Cleared RAG evaluation history")


# Singleton instance
_rag_evaluator = None

def get_rag_evaluator() -> RAGEvaluator:
    """
    Get or create a singleton instance of the RAGEvaluator.
    
    Returns:
        RAGEvaluator instance
    """
    global _rag_evaluator
    if _rag_evaluator is None:
        _rag_evaluator = RAGEvaluator()
    return _rag_evaluator 