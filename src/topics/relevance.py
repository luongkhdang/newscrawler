"""
Relevance scoring module for NewsCrawler.

This module provides functionality to calculate relevance scores for news articles
based on topic classification, entity extraction, and content quality assessment.
"""

import logging
from typing import Dict, List, Optional, Any, Set, Tuple
import re

from src.topics.classifier import TopicClassifier, get_topic_classifier
from src.topics.entities import EntityExtractor, get_entity_extractor

class RelevanceScorer:
    """
    Calculates relevance scores for news articles based on topic match,
    entity presence, and content quality.
    """
    
    def __init__(
        self,
        topic_classifier: Optional[TopicClassifier] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the relevance scorer.
        
        Args:
            topic_classifier: Optional topic classifier instance
            entity_extractor: Optional entity extractor instance
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.topic_classifier = topic_classifier or get_topic_classifier()
        self.entity_extractor = entity_extractor or get_entity_extractor()
        
        # Default configuration
        self.config = {
            "topic_weight": 0.4,
            "entity_weight": 0.3,
            "content_weight": 0.2,
            "base_score": 0.1,
            "min_threshold": 0.3,  # Minimum score to consider an article relevant
            "country_match_weight": 0.05,
            "org_match_weight": 0.02,
            "people_match_weight": 0.01,
            "topic_match_weight": 0.1,
            "min_content_length": 1000,
            "min_paragraphs": 3,
            "min_sentences": 10
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        self.logger.info("Relevance scorer initialized")
    
    def score_article(self, title: str, content: str) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate relevance score for an article.
        
        Args:
            title: Article title
            content: Article content
            
        Returns:
            Tuple of (relevance score, metadata dictionary)
        """
        # Combine title and content for processing
        full_text = f"{title}\n\n{content}"
        
        # Extract topics
        topics = self.topic_classifier.classify(full_text)
        
        # Extract entities
        entities = self.entity_extractor.extract_entities(full_text)
        filtered_entities = self.entity_extractor.filter_relevant_entities(entities)
        
        # Calculate score
        score = self._calculate_score(full_text, topics, filtered_entities)
        
        # Create metadata
        metadata = {
            "topics": topics,
            "entities": filtered_entities,
            "relevance_score": score,
            "is_relevant": score >= self.config["min_threshold"]
        }
        
        self.logger.debug(f"Scored article with title '{title[:50]}...' - Score: {score:.2f}")
        return score, metadata
    
    def _calculate_score(
        self,
        text: str,
        topics: List[str],
        entities: Dict[str, List[Dict[str, Any]]]
    ) -> float:
        """
        Calculate relevance score based on topic match, entity presence, and content quality.
        
        Args:
            text: The article text
            topics: List of identified topics
            entities: Dictionary of extracted entities
            
        Returns:
            Relevance score between 0 and 1
        """
        # Base score
        score = self.config["base_score"]
        
        # Topic relevance
        topic_score = min(len(topics) * self.config["topic_match_weight"], self.config["topic_weight"])
        score += topic_score
        
        # Entity relevance
        entity_score = 0
        
        # Countries of interest
        countries = entities.get("countries", [])
        country_matches = len(countries)
        country_score = min(country_matches * self.config["country_match_weight"], 0.15)
        
        # Organizations
        orgs = entities.get("organizations", [])
        org_score = min(len(orgs) * self.config["org_match_weight"], 0.1)
        
        # People of interest
        people = entities.get("people", [])
        people_score = min(len(people) * self.config["people_match_weight"], 0.05)
        
        entity_score = country_score + org_score + people_score
        entity_score = min(entity_score, self.config["entity_weight"])
        
        score += entity_score
        
        # Content quality
        content_score = self._assess_content_quality(text)
        score += content_score
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _assess_content_quality(self, text: str) -> float:
        """
        Assess content quality based on length, structure, and other heuristics.
        
        Args:
            text: The article text
            
        Returns:
            Content quality score between 0 and content_weight
        """
        content_score = 0
        max_score = self.config["content_weight"]
        
        # Length assessment (40% of content score)
        if len(text) >= self.config["min_content_length"]:
            content_score += 0.4 * max_score
        else:
            # Partial score based on length
            length_ratio = len(text) / self.config["min_content_length"]
            content_score += 0.4 * max_score * min(length_ratio, 1.0)
        
        # Paragraph structure (30% of content score)
        paragraphs = text.split('\n\n')
        if len(paragraphs) >= self.config["min_paragraphs"]:
            content_score += 0.3 * max_score
        else:
            # Partial score based on paragraph count
            para_ratio = len(paragraphs) / self.config["min_paragraphs"]
            content_score += 0.3 * max_score * min(para_ratio, 1.0)
        
        # Sentence structure (30% of content score)
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) >= self.config["min_sentences"]:
            content_score += 0.3 * max_score
        else:
            # Partial score based on sentence count
            sent_ratio = len(sentences) / self.config["min_sentences"]
            content_score += 0.3 * max_score * min(sent_ratio, 1.0)
        
        return min(content_score, max_score)
    
    def is_article_relevant(self, title: str, content: str) -> bool:
        """
        Determine if an article is relevant based on its score.
        
        Args:
            title: Article title
            content: Article content
            
        Returns:
            True if the article is relevant, False otherwise
        """
        score, _ = self.score_article(title, content)
        return score >= self.config["min_threshold"]

def get_relevance_scorer(
    topic_classifier: Optional[TopicClassifier] = None,
    entity_extractor: Optional[EntityExtractor] = None,
    config: Optional[Dict[str, Any]] = None
) -> RelevanceScorer:
    """
    Factory function to get a relevance scorer instance.
    
    Args:
        topic_classifier: Optional topic classifier instance
        entity_extractor: Optional entity extractor instance
        config: Optional configuration dictionary
        
    Returns:
        RelevanceScorer instance
    """
    return RelevanceScorer(topic_classifier, entity_extractor, config) 