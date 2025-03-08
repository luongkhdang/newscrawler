# Data Quality Enhancement Plan

## Overview
This plan outlines the implementation of enhanced data quality assurance mechanisms for the NewsCrawler project, focusing on content validation, duplicate detection, and content classification.

## 1. Content Validation Framework (Week 1)

### Tasks:
- [ ] Implement article schema validation using Pydantic models
- [ ] Create content quality scoring system based on:
  - Text length and completeness
  - Content-to-boilerplate ratio
  - Presence of required metadata (title, date, author)
  - Language detection and filtering
- [ ] Develop validation pipeline to be executed post-scraping
- [ ] Add logging and reporting for validation failures

### Implementation Details:
```python
# src/utils/validators.py
from pydantic import BaseModel, validator, Field
from typing import Optional, List
import langdetect
from datetime import datetime

class ArticleValidator(BaseModel):
    """Validator for article content quality."""
    url: str
    title: str
    content: str
    published_date: Optional[datetime] = None
    author: Optional[str] = None
    source_domain: str
    
    @validator('content')
    def content_length_valid(cls, v):
        if len(v) < 100:  # Minimum content length
            raise ValueError('Article content too short')
        return v
    
    @validator('title')
    def title_valid(cls, v):
        if len(v) < 3:
            raise ValueError('Article title too short')
        return v
        
    def calculate_quality_score(self) -> float:
        """Calculate overall quality score for the article."""
        score = 0.0
        
        # Length score (0-40 points)
        content_len = len(self.content)
        if content_len > 5000:
            score += 40
        elif content_len > 2000:
            score += 30
        elif content_len > 1000:
            score += 20
        elif content_len > 500:
            score += 10
        
        # Metadata completeness (0-30 points)
        if self.title:
            score += 10
        if self.published_date:
            score += 10
        if self.author:
            score += 10
            
        # Language detection (0-30 points)
        try:
            lang = langdetect.detect(self.content)
            if lang == 'en':  # Assuming English is our target language
                score += 30
            else:
                score += 10  # Other languages still get some points
        except:
            pass
            
        return score / 100.0  # Normalize to 0-1 range
```

## 2. Duplicate Detection System (Week 2)

### Tasks:
- [ ] Implement content-based similarity detection using:
  - TF-IDF vectorization for text comparison
  - MinHash and Locality Sensitive Hashing (LSH) for efficient similarity search
  - Cosine similarity thresholds for duplicate identification
- [ ] Create database indexes to speed up duplicate checks
- [ ] Develop deduplication pipeline with configurable thresholds
- [ ] Add reporting for identified duplicates

### Implementation Details:
```python
# src/utils/deduplication.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datasketch import MinHash, MinHashLSH
from typing import List, Dict, Tuple, Set

class DuplicateDetector:
    """Detects duplicate or highly similar articles."""
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.lsh = MinHashLSH(threshold=threshold, num_perm=128)
        self.minhashes = {}
        
    def _create_minhash(self, text: str) -> MinHash:
        """Create MinHash for a text document."""
        m = MinHash(num_perm=128)
        for shingle in self._get_shingles(text):
            m.update(shingle.encode('utf8'))
        return m
        
    def _get_shingles(self, text: str, k: int = 4) -> Set[str]:
        """Create k-shingles from text."""
        text = text.lower()
        return {text[i:i+k] for i in range(len(text) - k + 1)}
    
    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the LSH index."""
        minhash = self._create_minhash(text)
        self.minhashes[doc_id] = minhash
        self.lsh.insert(doc_id, minhash)
    
    def find_duplicates(self, doc_id: str, text: str) -> List[str]:
        """Find potential duplicates for a document."""
        if doc_id not in self.minhashes:
            self.add_document(doc_id, text)
        
        # Query the LSH index for potential duplicates
        return self.lsh.query(self.minhashes[doc_id])
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
```

## 3. Content Classification System (Week 3-4)

### Tasks:
- [ ] Implement topic modeling using LDA (Latent Dirichlet Allocation)
- [ ] Develop category classification using pre-trained models
- [ ] Create quality filters based on content type and relevance
- [ ] Integrate classification results with the database schema

### Implementation Details:
```python
# src/utils/classification.py
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
from typing import List, Dict, Any

class ContentClassifier:
    """Classifies article content by topic and category."""
    
    def __init__(self, num_topics: int = 10):
        self.num_topics = num_topics
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        self.lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        self.category_classifier = pipeline("zero-shot-classification")
        
    def fit_topic_model(self, documents: List[str]) -> None:
        """Fit the LDA topic model on a corpus of documents."""
        X = self.vectorizer.fit_transform(documents)
        self.lda.fit(X)
    
    def extract_topics(self, text: str) -> Dict[int, float]:
        """Extract topic distribution for a single document."""
        X = self.vectorizer.transform([text])
        topic_distribution = self.lda.transform(X)[0]
        return {i: float(score) for i, score in enumerate(topic_distribution)}
    
    def classify_category(self, text: str, candidate_labels: List[str]) -> Dict[str, float]:
        """Classify text into predefined categories using zero-shot learning."""
        result = self.category_classifier(text, candidate_labels)
        return {label: score for label, score in zip(result['labels'], result['scores'])}
    
    def is_relevant_content(self, text: str, min_length: int = 500) -> bool:
        """Determine if content is relevant based on quality heuristics."""
        # Length check
        if len(text) < min_length:
            return False
            
        # Check for common patterns of low-quality content
        low_quality_indicators = [
            "subscribe to read more",
            "please subscribe",
            "sign up for our newsletter",
            "this content is only available to subscribers"
        ]
        
        for indicator in low_quality_indicators:
            if indicator in text.lower():
                return False
                
        return True
```

## Integration Plan

1. Add these utilities to the processing pipeline in `src/scrapers/base_scraper.py`
2. Update the database models to store quality scores and classification data
3. Modify the API to filter results based on quality thresholds
4. Create admin endpoints for reviewing and managing duplicate content

## Testing Strategy

1. Develop unit tests for each validation and classification component
2. Create a test dataset of known duplicates to evaluate detection accuracy
3. Benchmark performance impact of quality checks on the processing pipeline
4. Conduct A/B testing on classification accuracy with different models 