"""
Retrieval-Augmented Generation (RAG) implementation for the NewsCrawler system.
This module provides functionality to enhance LLM responses with relevant news articles.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from src.database.session import get_db, SessionLocal
from src.database.models import Article
from src.vector.processor import generate_embedding
from src.vector.embedding_models import generate_embedding_with_model
from src.utils.groq_client import get_groq_client
from src.llm.context_assembly import get_context_assembler
from src.llm.response_cache import get_rag_cache
from src.llm.evaluation import get_rag_evaluator

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_CONTEXT_STRATEGY = os.environ.get("RAG_CONTEXT_STRATEGY", "semantic")
DEFAULT_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "llama3-70b-8192")
ENABLE_RESPONSE_CACHE = os.environ.get("ENABLE_RAG_CACHE", "true").lower() == "true"
ENABLE_EVALUATION = os.environ.get("ENABLE_RAG_EVALUATION", "true").lower() == "true"


class RAGSystem:
    """
    Retrieval-Augmented Generation system for enhancing LLM responses with news articles.
    """
    
    def __init__(
        self, 
        db_session: Optional[Session] = None,
        context_strategy: str = DEFAULT_CONTEXT_STRATEGY,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        enable_cache: bool = ENABLE_RESPONSE_CACHE,
        enable_evaluation: bool = ENABLE_EVALUATION
    ):
        """
        Initialize the RAG system.
        
        Args:
            db_session: Optional database session. If not provided, a new session will be created.
            context_strategy: Strategy for assembling context ("simple", "chunked", "semantic", "temporal")
            embedding_model: Name of the embedding model to use
            enable_cache: Whether to enable response caching
            enable_evaluation: Whether to enable response evaluation
        """
        self.db = db_session or SessionLocal()
        self.groq_client = get_groq_client()
        self.logger = logging.getLogger(__name__)
        self.context_assembler = get_context_assembler(context_strategy)
        self.embedding_model = embedding_model
        self.enable_cache = enable_cache
        self.enable_evaluation = enable_evaluation
        
        # Initialize cache and evaluator if enabled
        if self.enable_cache:
            self.cache = get_rag_cache()
        
        if self.enable_evaluation:
            self.evaluator = get_rag_evaluator()
    
    def retrieve_relevant_articles(
        self, 
        query: str, 
        limit: int = 5,
        min_similarity: float = 0.7,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve articles relevant to the query using vector similarity.
        
        Args:
            query: The search query
            limit: Maximum number of articles to retrieve
            min_similarity: Minimum similarity score (0-1) for articles to be considered relevant
            filter_criteria: Optional dictionary with filter criteria (source_domain, date range, etc.)
            
        Returns:
            List of relevant articles with similarity scores
        """
        try:
            # Generate embedding for the query using the specified model
            query_embedding = generate_embedding_with_model(query, self.embedding_model)
            
            # Start building the query
            article_query = self.db.query(
                Article,
                func.cosine_similarity(Article.vector_embedding, query_embedding).label("similarity")
            ).filter(
                Article.vector_embedding.is_not(None)
            )
            
            # Apply filters if provided
            if filter_criteria:
                if 'source_domain' in filter_criteria:
                    article_query = article_query.filter(Article.source_domain == filter_criteria['source_domain'])
                
                if 'start_date' in filter_criteria and filter_criteria['start_date']:
                    article_query = article_query.filter(Article.published_date >= filter_criteria['start_date'])
                
                if 'end_date' in filter_criteria and filter_criteria['end_date']:
                    article_query = article_query.filter(Article.published_date <= filter_criteria['end_date'])
                
                if 'category' in filter_criteria and filter_criteria['category']:
                    article_query = article_query.filter(Article.category == filter_criteria['category'])
            
            # Order by similarity and limit results
            articles = article_query.order_by(
                desc("similarity")
            ).limit(limit).all()
            
            # Filter by minimum similarity and format results
            results = []
            for article, similarity in articles:
                if similarity >= min_similarity:
                    results.append({
                        "id": str(article.id),
                        "title": article.title,
                        "content": article.content,
                        "url": article.url,
                        "published_date": article.published_date.isoformat() if article.published_date else None,
                        "source_domain": article.source_domain,
                        "category": article.category,
                        "similarity": float(similarity)
                    })
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error retrieving relevant articles: {e}")
            return []
    
    def retrieve_articles_by_ids(self, article_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve articles by their IDs.
        
        Args:
            article_ids: List of article IDs
            
        Returns:
            List of articles
        """
        try:
            articles = self.db.query(Article).filter(Article.id.in_(article_ids)).all()
            
            results = []
            for article in articles:
                results.append({
                    "id": str(article.id),
                    "title": article.title,
                    "content": article.content,
                    "url": article.url,
                    "published_date": article.published_date.isoformat() if article.published_date else None,
                    "source_domain": article.source_domain,
                    "category": article.category,
                    "similarity": 1.0  # Default similarity for explicitly selected articles
                })
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error retrieving articles by IDs: {e}")
            return []
    
    def answer_question(
        self, 
        query: str,
        model: str = DEFAULT_LLM_MODEL,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        Args:
            query: The user's question
            model: The LLM model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            filter_criteria: Optional dictionary with filter criteria (source_domain, date range, etc.)
            
        Returns:
            Dictionary containing the answer and source information
        """
        try:
            # Check cache first if enabled
            if self.enable_cache:
                cached_response = self.cache.get(query)
                if cached_response:
                    self.logger.info(f"Using cached response for query: {query}")
                    return cached_response
            
            # Retrieve relevant articles
            articles = self.retrieve_relevant_articles(
                query=query,
                filter_criteria=filter_criteria
            )
            
            # Generate context using the selected strategy
            context = self.context_assembler.assemble(articles, query)
            
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
            
            # Prepare response
            response = {
                "answer": answer,
                "sources": [
                    {
                        "id": article["id"],
                        "title": article["title"],
                        "url": article["url"],
                        "published_date": article["published_date"],
                        "source_domain": article["source_domain"],
                        "similarity": article["similarity"]
                    } for article in articles
                ]
            }
            
            # Evaluate response if enabled
            if self.enable_evaluation:
                metrics = self.evaluator.evaluate_response(
                    query=query,
                    response=answer,
                    sources=response["sources"]
                )
                response["evaluation"] = metrics
            
            # Cache response if enabled
            if self.enable_cache:
                self.cache.set(query, response)
            
            return response
        
        except Exception as e:
            self.logger.error(f"Error answering question: {e}")
            return {
                "answer": f"I encountered an error while trying to answer your question: {str(e)}",
                "sources": []
            }
    
    def answer_with_specific_context(
        self,
        query: str,
        article_ids: List[str],
        model: str = DEFAULT_LLM_MODEL,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Answer a question using specific articles as context.
        
        Args:
            query: The user's question
            article_ids: List of article IDs to use as context
            model: The LLM model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Dictionary containing the answer and source information
        """
        try:
            # Retrieve specified articles
            articles = self.retrieve_articles_by_ids(article_ids)
            
            if not articles:
                return {
                    "answer": "I couldn't find the specified articles to answer your question.",
                    "sources": []
                }
            
            # Generate context using the selected strategy
            context = self.context_assembler.assemble(articles, query)
            
            # Create system message
            system_message = """
            You are a helpful assistant that answers questions based on the provided news articles.
            Provide a detailed response based on the information in the articles.
            Always cite your sources by referring to the article numbers in your response.
            """
            
            # Create prompt with context
            prompt = f"""
            Question: {query}
            
            Here are the news articles to use for answering the question:
            
            {context}
            
            Please answer the question based on the information in these articles.
            """
            
            # Generate answer
            answer = self.groq_client.generate_completion(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system_message=system_message
            )
            
            # Prepare response
            response = {
                "answer": answer,
                "sources": [
                    {
                        "id": article["id"],
                        "title": article["title"],
                        "url": article["url"],
                        "published_date": article["published_date"],
                        "source_domain": article["source_domain"],
                        "similarity": article["similarity"]
                    } for article in articles
                ]
            }
            
            # Evaluate response if enabled
            if self.enable_evaluation:
                metrics = self.evaluator.evaluate_response(
                    query=query,
                    response=answer,
                    sources=response["sources"]
                )
                response["evaluation"] = metrics
            
            return response
        
        except Exception as e:
            self.logger.error(f"Error answering question with specific context: {e}")
            return {
                "answer": f"I encountered an error while trying to answer your question: {str(e)}",
                "sources": []
            }
    
    def summarize_article(
        self,
        article_id: str,
        model: str = DEFAULT_LLM_MODEL,
        max_tokens: int = 512,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Summarize an article.
        
        Args:
            article_id: ID of the article to summarize
            model: The LLM model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Dictionary containing the summary and article information
        """
        try:
            # Retrieve the article
            article = self.db.query(Article).filter(Article.id == article_id).first()
            
            if not article:
                return {
                    "error": f"Article with ID {article_id} not found",
                    "article_id": article_id
                }
            
            # Create system message
            system_message = """
            You are a helpful assistant that summarizes news articles.
            Create a concise summary that captures the main points of the article.
            The summary should be informative and objective.
            """
            
            # Create prompt
            prompt = f"""
            Please summarize the following news article:
            
            Title: {article.title}
            Source: {article.source_domain}
            Date: {article.published_date.isoformat() if article.published_date else 'Unknown'}
            
            Content:
            {article.content}
            """
            
            # Generate summary
            summary = self.groq_client.generate_completion(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system_message=system_message
            )
            
            return {
                "summary": summary,
                "article_id": str(article.id),
                "title": article.title,
                "url": article.url,
                "published_date": article.published_date.isoformat() if article.published_date else None,
                "source_domain": article.source_domain
            }
        
        except Exception as e:
            self.logger.error(f"Error summarizing article: {e}")
            return {
                "error": f"Error summarizing article: {str(e)}",
                "article_id": article_id
            }
    
    def get_evaluation_metrics(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Get evaluation metrics for the RAG system.
        
        Args:
            last_n: Only consider the last N evaluations (None for all)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.enable_evaluation:
            return {"error": "Evaluation is not enabled for this RAG system"}
        
        return self.evaluator.get_average_metrics(last_n)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for the RAG system.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.enable_cache:
            return {"error": "Caching is not enabled for this RAG system"}
        
        return self.cache.stats()
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        if self.enable_cache:
            self.cache.clear()
            self.logger.info("Cleared RAG response cache")
    
    def close(self):
        """Close the database session."""
        self.db.close() 