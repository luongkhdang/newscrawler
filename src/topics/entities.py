"""
Entity extraction module for NewsCrawler.

This module provides functionality to extract named entities from news articles,
with a focus on countries, organizations, and people of interest.
"""

import logging
from typing import Dict, List, Optional, Any, Set
import re

try:
    import spacy
    from spacy.language import Language
except ImportError:
    logging.warning("Spacy library not installed. Entity extraction will not work.")

class EntityExtractor:
    """
    Extracts entities from article text with a focus on countries, organizations,
    and people of interest.
    """
    
    def __init__(self, model_name: str = "en_core_web_lg"):
        """
        Initialize the entity extractor.
        
        Args:
            model_name: Name of the spaCy model to use
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        # Countries of interest for the project
        self.countries_of_interest = {
            "USA", "United States", "America", "U.S.", "U.S.A.",
            "Vietnam", "Socialist Republic of Vietnam",
            "China", "People's Republic of China", "PRC",
            "Japan",
            "Mexico", "United Mexican States",
            "Germany", "Federal Republic of Germany",
            "Singapore", "Republic of Singapore",
            "Taiwan", "Republic of China", "ROC"
        }
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(model_name)
            self.logger.info(f"Entity extractor initialized with model {model_name}")
        except Exception as e:
            self.nlp = None
            self.logger.error(f"Failed to load spaCy model: {str(e)}")
            self.logger.warning("Using fallback regex-based entity extraction")
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract entities from text.
        
        Args:
            text: The text to extract entities from
            
        Returns:
            Dictionary of entity types and their occurrences
        """
        if self.nlp is None:
            return self._regex_based_extraction(text)
        
        try:
            # Process text with spaCy
            # If text is very long, process it in chunks to avoid memory issues
            max_length = 100000  # Characters
            if len(text) > max_length:
                return self._process_long_text(text, max_length)
            
            doc = self.nlp(text)
            
            # Extract entities
            entities = {
                "countries": [],
                "organizations": [],
                "people": [],
                "locations": [],
                "dates": [],
                "misc": []
            }
            
            # Process each entity
            for ent in doc.ents:
                entity_info = {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                
                # Categorize by entity type
                if ent.label_ == "GPE" or ent.label_ == "NORP":  # Countries, cities, nationalities
                    if self._is_country_of_interest(ent.text):
                        entities["countries"].append(entity_info)
                    else:
                        entities["locations"].append(entity_info)
                elif ent.label_ == "ORG":  # Organizations
                    entities["organizations"].append(entity_info)
                elif ent.label_ == "PERSON":  # People
                    entities["people"].append(entity_info)
                elif ent.label_ == "DATE":  # Dates
                    entities["dates"].append(entity_info)
                elif ent.label_ in ["LOC", "FAC"]:  # Other locations
                    entities["locations"].append(entity_info)
                else:
                    entities["misc"].append(entity_info)
            
            # Remove duplicates while preserving order
            for entity_type in entities:
                entities[entity_type] = self._remove_duplicate_entities(entities[entity_type])
            
            self.logger.debug(f"Extracted {sum(len(v) for v in entities.values())} entities from text")
            return entities
        
        except Exception as e:
            self.logger.error(f"Error during entity extraction: {str(e)}")
            return self._regex_based_extraction(text)
    
    def _process_long_text(self, text: str, chunk_size: int) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process long text by splitting it into chunks.
        
        Args:
            text: The text to process
            chunk_size: Maximum chunk size in characters
            
        Returns:
            Combined entities from all chunks
        """
        # Split text into chunks at sentence boundaries
        chunks = []
        current_chunk = ""
        
        for sentence in re.split(r'(?<=[.!?])\s+', text):
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk)
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Process each chunk
        all_entities = {
            "countries": [],
            "organizations": [],
            "people": [],
            "locations": [],
            "dates": [],
            "misc": []
        }
        
        for i, chunk in enumerate(chunks):
            self.logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
            chunk_entities = self.extract_entities(chunk)
            
            # Combine entities
            for entity_type, entities in chunk_entities.items():
                all_entities[entity_type].extend(entities)
        
        # Remove duplicates
        for entity_type in all_entities:
            all_entities[entity_type] = self._remove_duplicate_entities(all_entities[entity_type])
        
        return all_entities
    
    def _regex_based_extraction(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fallback method using regex-based entity extraction.
        
        Args:
            text: The text to extract entities from
            
        Returns:
            Dictionary of entity types and their occurrences
        """
        entities = {
            "countries": [],
            "organizations": [],
            "people": [],
            "locations": [],
            "dates": [],
            "misc": []
        }
        
        # Extract countries of interest
        for country in self.countries_of_interest:
            for match in re.finditer(r'\b' + re.escape(country) + r'\b', text):
                entities["countries"].append({
                    "text": match.group(0),
                    "label": "GPE",
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Simple regex for organizations (companies ending in Inc., Corp., etc.)
        org_pattern = r'\b([A-Z][A-Za-z0-9\'\-&\.]+(\.com|Inc\.|Corp\.|Co\.|Ltd\.|LLC|Group|Association|Organization))\b'
        for match in re.finditer(org_pattern, text):
            entities["organizations"].append({
                "text": match.group(0),
                "label": "ORG",
                "start": match.start(),
                "end": match.end()
            })
        
        # Simple regex for people (two capitalized words in sequence)
        people_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        for match in re.finditer(people_pattern, text):
            # Skip if it's a country or organization
            if any(match.group(0) == entity["text"] for entity in entities["countries"] + entities["organizations"]):
                continue
            
            entities["people"].append({
                "text": match.group(0),
                "label": "PERSON",
                "start": match.start(),
                "end": match.end()
            })
        
        # Simple regex for dates
        date_pattern = r'\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})\b'
        for match in re.finditer(date_pattern, text):
            entities["dates"].append({
                "text": match.group(0),
                "label": "DATE",
                "start": match.start(),
                "end": match.end()
            })
        
        return entities
    
    def _is_country_of_interest(self, text: str) -> bool:
        """
        Check if the entity is a country of interest.
        
        Args:
            text: The entity text
            
        Returns:
            True if it's a country of interest, False otherwise
        """
        return text in self.countries_of_interest
    
    def _remove_duplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate entities while preserving order.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            Deduplicated list of entities
        """
        seen = set()
        result = []
        
        for entity in entities:
            entity_text = entity["text"].lower()
            if entity_text not in seen:
                seen.add(entity_text)
                result.append(entity)
        
        return result
    
    def filter_relevant_entities(self, entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Filter entities to those of interest.
        
        Args:
            entities: Dictionary of entity types and their occurrences
            
        Returns:
            Filtered entities
        """
        filtered = {}
        
        # Keep all countries of interest
        filtered["countries"] = [
            entity for entity in entities.get("countries", [])
            if self._is_country_of_interest(entity["text"])
        ]
        
        # Keep top organizations by frequency
        org_counts = {}
        for entity in entities.get("organizations", []):
            org_counts[entity["text"]] = org_counts.get(entity["text"], 0) + 1
        
        # Sort by frequency
        sorted_orgs = sorted(org_counts.items(), key=lambda x: x[1], reverse=True)
        top_orgs = [org for org, _ in sorted_orgs[:10]]  # Keep top 10
        
        filtered["organizations"] = [
            entity for entity in entities.get("organizations", [])
            if entity["text"] in top_orgs
        ]
        
        # Keep top people by frequency
        people_counts = {}
        for entity in entities.get("people", []):
            people_counts[entity["text"]] = people_counts.get(entity["text"], 0) + 1
        
        # Sort by frequency
        sorted_people = sorted(people_counts.items(), key=lambda x: x[1], reverse=True)
        top_people = [person for person, _ in sorted_people[:10]]  # Keep top 10
        
        filtered["people"] = [
            entity for entity in entities.get("people", [])
            if entity["text"] in top_people
        ]
        
        return filtered

def get_entity_extractor(model_name: Optional[str] = None) -> EntityExtractor:
    """
    Factory function to get an entity extractor instance.
    
    Args:
        model_name: Optional name of the spaCy model to use
        
    Returns:
        EntityExtractor instance
    """
    if model_name:
        return EntityExtractor(model_name)
    return EntityExtractor() 