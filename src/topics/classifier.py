"""
Topic classification module for NewsCrawler.

This module provides functionality to classify news articles into predefined topics of interest,
focusing on the specific topics outlined in the project requirements.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    logging.warning("Transformers library not installed. Topic classification will not work.")

class TopicClassifier:
    """
    Classifies articles into predefined topics of interest.
    """
    
    def __init__(self, model_path: str = "models/topic_classifier"):
        """
        Initialize the topic classifier.
        
        Args:
            model_path: Path to the pre-trained model or model identifier from Hugging Face
        """
        self.logger = logging.getLogger(__name__)
        self.topics = [
            "laws_regulations",
            "influential_figures",
            "trade_agreements",
            "investment_trends",
            "currency_interaction",
            "market_indices",
            "research_papers"
        ]
        
        self.countries_of_interest = [
            "USA", "Vietnam", "China", "Japan", 
            "Mexico", "Germany", "Singapore", "Taiwan"
        ]
        
        try:
            self.tokenizer = self._load_tokenizer(model_path)
            self.model = self._load_model(model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.logger.info(f"Topic classifier initialized with model from {model_path} on {self.device}")
        except Exception as e:
            self.tokenizer = None
            self.model = None
            self.logger.error(f"Failed to load topic classification model: {str(e)}")
            self.logger.warning("Using fallback keyword-based classification")
    
    def _load_tokenizer(self, model_path: str):
        """
        Load the tokenizer for the classification model.
        
        Args:
            model_path: Path to the pre-trained model or model identifier
            
        Returns:
            The loaded tokenizer
        """
        try:
            return AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {str(e)}")
            raise
    
    def _load_model(self, model_path: str):
        """
        Load the classification model.
        
        Args:
            model_path: Path to the pre-trained model or model identifier
            
        Returns:
            The loaded model
        """
        try:
            return AutoModelForSequenceClassification.from_pretrained(model_path)
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def classify(self, text: str, threshold: float = 0.5) -> List[str]:
        """
        Classify text into topics.
        
        Args:
            text: The text to classify
            threshold: Confidence threshold for classification
            
        Returns:
            List of topics that the text belongs to
        """
        if self.model is None or self.tokenizer is None:
            return self._keyword_based_classification(text)
        
        try:
            # Truncate text if it's too long
            max_length = self.tokenizer.model_max_length
            if len(text) > max_length * 4:  # Rough character estimate
                text = text[:max_length * 4]
            
            # Tokenize and prepare for model
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Get topics above threshold
            matched_topics = [self.topics[i] for i, prob in enumerate(probs) if prob >= threshold]
            
            self.logger.debug(f"Classified text into topics: {matched_topics}")
            return matched_topics
        
        except Exception as e:
            self.logger.error(f"Error during classification: {str(e)}")
            return self._keyword_based_classification(text)
    
    def _keyword_based_classification(self, text: str) -> List[str]:
        """
        Fallback method using keyword-based classification.
        
        Args:
            text: The text to classify
            
        Returns:
            List of topics that the text belongs to
        """
        text = text.lower()
        matched_topics = []
        
        # Define keywords for each topic
        topic_keywords = {
            "laws_regulations": ["law", "regulation", "policy", "legislation", "bill", "act", "treaty", "agreement", "sanction"],
            "influential_figures": ["president", "minister", "secretary", "ceo", "chairman", "leader", "official", "executive"],
            "trade_agreements": ["trade", "export", "import", "tariff", "customs", "duty", "wto", "nafta", "usmca", "rcep"],
            "investment_trends": ["investment", "investor", "fund", "stock", "bond", "market", "portfolio", "asset", "capital"],
            "currency_interaction": ["currency", "exchange rate", "forex", "dollar", "euro", "yen", "yuan", "dong", "monetary"],
            "market_indices": ["index", "s&p", "dow", "nasdaq", "qqq", "djia", "stock market", "bull", "bear"],
            "research_papers": ["research", "study", "paper", "report", "analysis", "survey", "finding", "data", "statistics"]
        }
        
        # Check for each topic's keywords
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    matched_topics.append(topic)
                    break
        
        self.logger.debug(f"Keyword classification found topics: {matched_topics}")
        return matched_topics
    
    def calculate_relevance_score(self, text: str, topics: List[str], entities: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate relevance score based on topic match, entity presence, and content quality.
        
        Args:
            text: The article text
            topics: List of identified topics
            entities: Dictionary of extracted entities
            
        Returns:
            Relevance score between 0 and 1
        """
        # Base score starts at 0.1
        score = 0.1
        
        # Topic relevance (up to 0.4)
        topic_score = min(len(topics) * 0.1, 0.4)
        score += topic_score
        
        # Entity relevance (up to 0.3)
        entity_score = 0
        if entities:
            # Check for countries of interest
            countries = entities.get("countries", [])
            country_matches = sum(1 for country in countries if country in self.countries_of_interest)
            country_score = min(country_matches * 0.05, 0.15)
            
            # Check for organizations
            orgs = entities.get("organizations", [])
            org_score = min(len(orgs) * 0.02, 0.1)
            
            # Check for people of interest
            people = entities.get("people", [])
            people_score = min(len(people) * 0.01, 0.05)
            
            entity_score = country_score + org_score + people_score
        
        score += entity_score
        
        # Content quality (up to 0.2)
        # Simple heuristic based on text length and structure
        content_score = 0
        if len(text) > 1000:  # Longer articles tend to be more substantial
            content_score += 0.1
        if len(text.split('\n\n')) > 3:  # Articles with multiple paragraphs
            content_score += 0.05
        if len(text.split('.')) > 10:  # Articles with multiple sentences
            content_score += 0.05
        
        score += content_score
        
        return min(score, 1.0)  # Cap at 1.0

def get_topic_classifier(model_path: Optional[str] = None) -> TopicClassifier:
    """
    Factory function to get a topic classifier instance.
    
    Args:
        model_path: Optional path to the model
        
    Returns:
        TopicClassifier instance
    """
    if model_path:
        return TopicClassifier(model_path)
    return TopicClassifier() 