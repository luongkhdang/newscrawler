# LLM Training and Integration Enhancement Plan

## Overview
This plan outlines the implementation of enhanced LLM capabilities for the NewsCrawler project, focusing on fine-tuning pipelines, advanced RAG techniques, and evaluation metrics.

## 1. Fine-Tuning Pipeline Development (Week 1-2)

### Tasks:
- [ ] Implement dataset preparation tools for fine-tuning
- [ ] Create training data generation pipeline from collected articles
- [ ] Develop fine-tuning scripts for various model types
- [ ] Implement model evaluation and selection framework

### Implementation Details:
```python
# src/llm/dataset_preparation.py
import pandas as pd
import json
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from tqdm import tqdm

from src.database.models import Article
from src.database.session import SessionLocal

class DatasetPreparator:
    """Prepares datasets for LLM fine-tuning from collected articles."""
    
    def __init__(self, db: Optional[Session] = None):
        self.db = db or SessionLocal()
        
    def create_summarization_dataset(self, output_file: str, limit: int = 10000) -> None:
        """Create a dataset for summarization fine-tuning."""
        articles = self.db.query(Article).filter(
            Article.summary.isnot(None),
            Article.content.isnot(None)
        ).limit(limit).all()
        
        dataset = []
        for article in tqdm(articles, desc="Preparing summarization dataset"):
            dataset.append({
                "text": article.content,
                "summary": article.summary
            })
            
        with open(output_file, 'w') as f:
            json.dump(dataset, f)
            
        print(f"Created summarization dataset with {len(dataset)} examples")
    
    def create_qa_dataset(self, output_file: str, limit: int = 10000) -> None:
        """Create a question-answering dataset from articles."""
        articles = self.db.query(Article).filter(
            Article.content.isnot(None)
        ).limit(limit).all()
        
        dataset = []
        for article in tqdm(articles, desc="Preparing QA dataset"):
            # Generate synthetic questions from article content
            # This is a simplified example - in practice, you'd use more sophisticated methods
            title_question = f"What is the main topic of the article titled '{article.title}'?"
            
            dataset.append({
                "question": title_question,
                "context": article.content,
                "answer": article.title
            })
            
            if article.author:
                author_question = f"Who wrote the article about {article.title}?"
                dataset.append({
                    "question": author_question,
                    "context": article.content,
                    "answer": article.author
                })
                
        with open(output_file, 'w') as f:
            json.dump(dataset, f)
            
        print(f"Created QA dataset with {len(dataset)} examples")
    
    def create_classification_dataset(self, output_file: str, limit: int = 10000) -> None:
        """Create a classification dataset from articles with categories."""
        articles = self.db.query(Article).filter(
            Article.category.isnot(None),
            Article.content.isnot(None)
        ).limit(limit).all()
        
        dataset = []
        for article in tqdm(articles, desc="Preparing classification dataset"):
            dataset.append({
                "text": article.content,
                "label": article.category
            })
            
        with open(output_file, 'w') as f:
            json.dump(dataset, f)
            
        print(f"Created classification dataset with {len(dataset)} examples")
```

```python
# src/llm/fine_tuning.py
import os
import json
from typing import Dict, Any, List, Optional
import torch
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import evaluate

class ModelFineTuner:
    """Fine-tunes language models on prepared datasets."""
    
    def __init__(
        self, 
        base_model: str = "google/flan-t5-base",
        output_dir: str = "./fine_tuned_model",
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        epochs: int = 3
    ):
        self.base_model = base_model
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        
        # Set up evaluation metrics
        self.rouge = evaluate.load("rouge")
        
    def prepare_summarization_dataset(self, dataset_path: str):
        """Prepare dataset for summarization fine-tuning."""
        # Load dataset
        dataset = load_dataset('json', data_files=dataset_path)
        
        # Tokenize function
        def tokenize_function(examples):
            inputs = self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
            labels = self.tokenizer(examples["summary"], padding="max_length", truncation=True, max_length=128)
            
            inputs["labels"] = labels["input_ids"]
            return inputs
        
        # Apply tokenization
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Split dataset
        train_test_split = tokenized_dataset["train"].train_test_split(test_size=0.1)
        
        return {
            "train": train_test_split["train"],
            "validation": train_test_split["test"]
        }
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them
        labels = [[l if l != -100 else self.tokenizer.pad_token_id for l in label] for label in labels]
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # ROUGE expects a newline after each sentence
        decoded_preds = ["\n".join(pred.split()) for pred in decoded_preds]
        decoded_labels = ["\n".join(label.split()) for label in decoded_labels]
        
        result = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        # Add mean generated length
        prediction_lens = [len(pred.split()) for pred in decoded_preds]
        result["gen_len"] = sum(prediction_lens) / len(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}
    
    def fine_tune(self, dataset_path: str, task_type: str = "summarization"):
        """Fine-tune the model on a specific task."""
        if task_type == "summarization":
            processed_dataset = self.prepare_summarization_dataset(dataset_path)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=self.epochs,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            report_to="tensorboard"
        )
        
        # Set up data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Start training
        trainer.train()
        
        # Save the model
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        return self.output_dir
```

## 2. Advanced RAG Implementation (Week 2-3)

### Tasks:
- [ ] Implement hybrid search combining vector and keyword search
- [ ] Develop context-aware retrieval with query expansion
- [ ] Create multi-document synthesis for comprehensive answers
- [ ] Implement relevance feedback mechanisms

### Implementation Details:
```python
# src/llm/advanced_rag.py
from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy.orm import Session
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Groq
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document

from src.database.session import SessionLocal
from src.database.models import Article
from src.vector.processor import generate_embedding

class HybridSearchRetriever:
    """Implements hybrid search combining vector and keyword search."""
    
    def __init__(self, db: Optional[Session] = None, llm_client=None):
        self.db = db or SessionLocal()
        self.llm_client = llm_client or Groq(api_key=os.getenv("GROQ_API_KEY"))
        
    def retrieve(self, query: str, limit: int = 5) -> List[Document]:
        """Retrieve documents using hybrid search."""
        # Generate query embedding
        query_embedding = generate_embedding(query)
        
        # Vector search
        vector_results = self._vector_search(query_embedding, limit=limit*2)
        
        # Keyword search
        keyword_results = self._keyword_search(query, limit=limit*2)
        
        # Combine results with ranking
        combined_results = self._combine_results(vector_results, keyword_results, limit)
        
        return combined_results
    
    def _vector_search(self, query_embedding: List[float], limit: int) -> List[Dict]:
        """Perform vector similarity search."""
        # This is a simplified example - in practice, you'd use SQL with vector operations
        # For PostgreSQL with pgvector, you'd use the <-> operator
        
        # Convert results to Document objects
        documents = []
        for article in results:
            doc = Document(
                page_content=article.content,
                metadata={
                    "id": str(article.id),
                    "title": article.title,
                    "url": article.url,
                    "source": article.source_domain,
                    "published_date": article.published_date.isoformat() if article.published_date else None,
                    "score": score
                }
            )
            documents.append(doc)
            
        return documents
    
    def _keyword_search(self, query: str, limit: int) -> List[Dict]:
        """Perform keyword-based search."""
        # Simplified example - in practice, you'd use full-text search capabilities
        # For PostgreSQL, you'd use the to_tsvector and to_tsquery functions
        
        # Convert results to Document objects
        documents = []
        for article in results:
            doc = Document(
                page_content=article.content,
                metadata={
                    "id": str(article.id),
                    "title": article.title,
                    "url": article.url,
                    "source": article.source_domain,
                    "published_date": article.published_date.isoformat() if article.published_date else None,
                    "score": score
                }
            )
            documents.append(doc)
            
        return documents
    
    def _combine_results(self, vector_results: List[Document], keyword_results: List[Document], limit: int) -> List[Document]:
        """Combine and rank results from different search methods."""
        # Create a dictionary to store combined scores
        combined_scores = {}
        
        # Process vector results
        for i, doc in enumerate(vector_results):
            doc_id = doc.metadata["id"]
            # Score based on position in results (higher rank = higher score)
            vector_score = 1.0 - (i / len(vector_results))
            combined_scores[doc_id] = {"doc": doc, "vector_score": vector_score, "keyword_score": 0.0}
        
        # Process keyword results
        for i, doc in enumerate(keyword_results):
            doc_id = doc.metadata["id"]
            # Score based on position in results
            keyword_score = 1.0 - (i / len(keyword_results))
            
            if doc_id in combined_scores:
                combined_scores[doc_id]["keyword_score"] = keyword_score
            else:
                combined_scores[doc_id] = {"doc": doc, "vector_score": 0.0, "keyword_score": keyword_score}
        
        # Calculate final scores (weighted combination)
        for doc_id, scores in combined_scores.items():
            # You can adjust these weights based on what works best
            final_score = 0.7 * scores["vector_score"] + 0.3 * scores["keyword_score"]
            scores["final_score"] = final_score
        
        # Sort by final score and take top results
        sorted_results = sorted(
            combined_scores.values(), 
            key=lambda x: x["final_score"], 
            reverse=True
        )[:limit]
        
        # Return documents with updated scores
        result_docs = []
        for item in sorted_results:
            doc = item["doc"]
            doc.metadata["score"] = item["final_score"]
            result_docs.append(doc)
            
        return result_docs


class ContextAwareRAG:
    """Implements context-aware RAG with query expansion and document synthesis."""
    
    def __init__(self, retriever=None, llm_client=None):
        self.retriever = retriever or HybridSearchRetriever()
        self.llm_client = llm_client or Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Set up query expansion
        query_expansion_template = """
        Given the original query: "{query}"
        Generate 3 alternative search queries that might help retrieve relevant information.
        The queries should be different from the original but capture the same intent.
        Return only the queries, one per line, without any additional text.
        """
        self.query_expansion_prompt = PromptTemplate(
            input_variables=["query"],
            template=query_expansion_template
        )
        self.query_expansion_chain = LLMChain(
            llm=self.llm_client,
            prompt=self.query_expansion_prompt
        )
        
        # Set up document compressor for contextual compression
        self.document_compressor = LLMChainExtractor.from_llm(self.llm_client)
        self.compression_retriever = ContextualCompressionRetriever(
            base_retriever=self.retriever,
            base_compressor=self.document_compressor
        )
    
    def expand_query(self, query: str) -> List[str]:
        """Expand the original query to improve retrieval."""
        result = self.query_expansion_chain.run(query=query)
        expanded_queries = [q.strip() for q in result.split('\n') if q.strip()]
        return [query] + expanded_queries  # Include original query
    
    def retrieve_with_expanded_queries(self, query: str, limit: int = 5) -> List[Document]:
        """Retrieve documents using query expansion."""
        expanded_queries = self.expand_query(query)
        
        all_docs = []
        for expanded_query in expanded_queries:
            docs = self.retriever.retrieve(expanded_query, limit=limit)
            all_docs.extend(docs)
        
        # Remove duplicates
        unique_docs = {}
        for doc in all_docs:
            doc_id = doc.metadata["id"]
            if doc_id not in unique_docs or doc.metadata["score"] > unique_docs[doc_id].metadata["score"]:
                unique_docs[doc_id] = doc
        
        # Sort by score and limit results
        sorted_docs = sorted(
            unique_docs.values(),
            key=lambda x: x.metadata["score"],
            reverse=True
        )[:limit]
        
        return sorted_docs
    
    def retrieve_with_compression(self, query: str, limit: int = 5) -> List[Document]:
        """Retrieve and compress documents to extract relevant parts."""
        return self.compression_retriever.get_relevant_documents(query)
    
    def synthesize_answer(self, query: str, documents: List[Document]) -> str:
        """Synthesize an answer from retrieved documents."""
        # Prepare context from documents
        context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)])
        
        # Create prompt for synthesis
        synthesis_template = """
        You are an AI assistant that provides accurate information based on the given context.
        
        Context:
        {context}
        
        Question: {query}
        
        Provide a comprehensive answer to the question based on the context provided.
        If the context doesn't contain relevant information to answer the question, state that clearly.
        Include citations to specific documents where appropriate.
        """
        
        synthesis_prompt = PromptTemplate(
            input_variables=["context", "query"],
            template=synthesis_template
        )
        
        synthesis_chain = LLMChain(
            llm=self.llm_client,
            prompt=synthesis_prompt
        )
        
        return synthesis_chain.run(context=context, query=query)
```

## 3. LLM Evaluation Framework (Week 3-4)

### Tasks:
- [ ] Implement automated evaluation metrics for LLM outputs
- [ ] Create benchmark datasets for different tasks
- [ ] Develop comparison framework for different models and approaches
- [ ] Build monitoring dashboard for LLM performance

### Implementation Details:
```python
# src/llm/evaluation.py
from typing import List, Dict, Any, Optional, Callable
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import evaluate

class LLMEvaluator:
    """Evaluates LLM performance using various metrics."""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleurt = evaluate.load('bleurt')
        
    def evaluate_summarization(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate summarization quality."""
        results = {}
        
        # ROUGE scores
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)
        
        for key in rouge_scores:
            results[key] = np.mean(rouge_scores[key])
        
        # BLEURT scores
        bleurt_scores = self.bleurt.compute(predictions=predictions, references=references)
        results["bleurt"] = np.mean(bleurt_scores["scores"])
        
        # BERTScore
        precision, recall, f1 = bert_score(predictions, references, lang="en")
        results["bert_score_precision"] = precision.mean().item()
        results["bert_score_recall"] = recall.mean().item()
        results["bert_score_f1"] = f1.mean().item()
        
        return results
    
    def evaluate_qa(self, predictions: List[str], references: List[str], contexts: List[str]) -> Dict[str, float]:
        """Evaluate question answering quality."""
        results = {}
        
        # Exact match
        exact_matches = [pred.strip() == ref.strip() for pred, ref in zip(predictions, references)]
        results["exact_match"] = np.mean(exact_matches)
        
        # F1 score for token overlap
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                f1_scores.append(0.0)
                continue
                
            common_tokens = pred_tokens.intersection(ref_tokens)
            precision = len(common_tokens) / len(pred_tokens)
            recall = len(common_tokens) / len(ref_tokens)
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))
        
        results["f1"] = np.mean(f1_scores)
        
        # BLEURT scores
        bleurt_scores = self.bleurt.compute(predictions=predictions, references=references)
        results["bleurt"] = np.mean(bleurt_scores["scores"])
        
        return results
    
    def evaluate_factuality(self, predictions: List[str], contexts: List[str]) -> Dict[str, float]:
        """Evaluate factual consistency of generated text with source context."""
        # This would typically use a specialized model for factuality checking
        # For a simplified version, we'll use BERTScore to measure semantic similarity
        
        precision, recall, f1 = bert_score(predictions, contexts, lang="en")
        
        return {
            "factuality_precision": precision.mean().item(),
            "factuality_recall": recall.mean().item(),
            "factuality_f1": f1.mean().item()
        }
    
    def run_benchmark(self, model_fn: Callable, benchmark_data: List[Dict[str, Any]], task_type: str) -> Dict[str, float]:
        """Run a benchmark evaluation on a model."""
        predictions = []
        references = []
        contexts = []
        
        for item in tqdm(benchmark_data, desc=f"Running {task_type} benchmark"):
            if task_type == "summarization":
                pred = model_fn(item["text"])
                predictions.append(pred)
                references.append(item["summary"])
                contexts.append(item["text"])
            elif task_type == "qa":
                pred = model_fn(item["question"], item["context"])
                predictions.append(pred)
                references.append(item["answer"])
                contexts.append(item["context"])
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
        
        if task_type == "summarization":
            return self.evaluate_summarization(predictions, references)
        elif task_type == "qa":
            return self.evaluate_qa(predictions, references, contexts)
        
        return {}
    
    def compare_models(self, model_fns: Dict[str, Callable], benchmark_data: List[Dict[str, Any]], task_type: str) -> pd.DataFrame:
        """Compare multiple models on the same benchmark."""
        results = {}
        
        for model_name, model_fn in model_fns.items():
            model_results = self.run_benchmark(model_fn, benchmark_data, task_type)
            results[model_name] = model_results
        
        # Convert to DataFrame for easy comparison
        return pd.DataFrame(results)
```

## Integration Plan

1. Add the dataset preparation and fine-tuning tools to the project
2. Integrate the advanced RAG components with the existing API
3. Implement the evaluation framework for continuous monitoring
4. Create a model registry for managing fine-tuned models

## Testing Strategy

1. Develop test datasets for each task (summarization, QA, etc.)
2. Create automated test suites for the fine-tuning pipeline
3. Implement A/B testing for comparing different RAG approaches
4. Set up continuous evaluation of model performance 