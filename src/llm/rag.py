"""
Retrieval-Augmented Generation (RAG) implementation for the NewsCrawler system.
This module provides functionality to enhance LLM responses with relevant news articles.
"""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from src.database.session import get_db, SessionLocal
from src.database.models import Article
from src.vector.processor import generate_embedding
from src.utils.groq_client import get_groq_client

logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Retrieval-Augmented Generation system for enhancing LLM responses with news articles.
    """
    
    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize the RAG system.
        
        Args:
            db_session: Optional database session. If not provided, a new session will be created.
        """
        self.db = db_session or SessionLocal()
        self.groq_client = get_groq_client()
        self.logger = logging.getLogger(__name__)
    
    def retrieve_relevant_articles(
        self, 
        query: str, 
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve articles relevant to the query using vector similarity.
        
        Args:
            query: The search query
            limit: Maximum number of articles to retrieve
            min_similarity: Minimum similarity score (0-1) for articles to be considered relevant
            
        Returns:
            List of relevant articles with similarity scores
        """
        try:
            # Generate embedding for the query
            query_embedding = generate_embedding(query)
            
            # Perform vector search using cosine similarity
            articles = self.db.query(
                Article,
                func.cosine_similarity(Article.vector_embedding, query_embedding).label("similarity")
            ).filter(
                Article.vector_embedding.is_not(None)
            ).order_by(
                desc("similarity")
            ).limit(limit).all()
            
            # Filter by minimum similarity and format results
            results = []
            for article, similarity in articles:
                if similarity >= min_similarity:
                    results.append({
                        "id": str(article.id),
                        "title": article.title,
                        "content": article.content[:500] + "..." if len(article.content) > 500 else article.content,
                        "url": article.url,
                        "published_date": article.published_date.isoformat() if article.published_date else None,
                        "source_domain": article.source_domain,
                        "similarity": float(similarity)
                    })
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error retrieving relevant articles: {e}")
            return []
    
    def generate_context(self, articles: List[Dict[str, Any]]) -> str:
        """
        Generate context from retrieved articles.
        
        Args:
            articles: List of relevant articles
            
        Returns:
            Formatted context string
        """
        if not articles:
            return "No relevant articles found."
        
        context_parts = []
        for i, article in enumerate(articles, 1):
            context_parts.append(f"Article {i}:")
            context_parts.append(f"Title: {article['title']}")
            context_parts.append(f"Source: {article['source_domain']}")
            if article['published_date']:
                context_parts.append(f"Date: {article['published_date']}")
            context_parts.append(f"Content: {article['content']}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def answer_question(
        self, 
        query: str,
        model: str = "llama3-70b-8192",
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        Args:
            query: The user's question
            model: The LLM model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Dictionary containing the answer and source information
        """
        try:
            # Retrieve relevant articles
            articles = self.retrieve_relevant_articles(query)
            
            # Generate context from articles
            context = self.generate_context(articles)
            
            # Create system message
            system_message = """
            You are a helpful assistant that answers questions based on the provided news articles.
            If the articles contain information to answer the question, provide a detailed response based on that information.
            If the articles do not contain enough information, acknowledge what you know and what you don't know.
            Always cite your sources by referring to the article numbers in your response.
            """
            
            # Create prompt with context
            prompt = f"""
            Question: {query}
            
            Here are some relevant news articles that might help answer the question:
            
            {context}
            
            Please answer the question based on the information in these articles. If the articles don't contain enough information to fully answer the question, please acknowledge what you know and what you don't know.
            """
            
            # Generate answer
            answer = self.groq_client.generate_completion(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system_message=system_message
            )
            
            # Return answer with sources
            return {
                "answer": answer,
                "sources": [
                    {
                        "title": article["title"],
                        "url": article["url"],
                        "published_date": article["published_date"],
                        "source_domain": article["source_domain"],
                        "similarity": article["similarity"]
                    } for article in articles
                ]
            }
        
        except Exception as e:
            self.logger.error(f"Error answering question: {e}")
            return {
                "answer": f"I encountered an error while trying to answer your question: {str(e)}",
                "sources": []
            }
        
    def close(self):
        """Close the database session."""
        self.db.close() 