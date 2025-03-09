"""
Topic classification and relevance scoring module for NewsCrawler.

This module provides functionality to classify news articles into predefined topics,
extract named entities, and calculate relevance scores based on the project's focus areas.
"""

from src.topics.classifier import TopicClassifier, get_topic_classifier
from src.topics.entities import EntityExtractor, get_entity_extractor
from src.topics.relevance import RelevanceScorer, get_relevance_scorer

__all__ = [
    'TopicClassifier',
    'get_topic_classifier',
    'EntityExtractor',
    'get_entity_extractor',
    'RelevanceScorer',
    'get_relevance_scorer'
] 