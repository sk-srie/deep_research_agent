"""
Deep Researcher Agent Package

A comprehensive AI-powered research assistant that uses local embeddings,
Chroma DB, and multi-step reasoning to provide detailed research reports.
"""

from .embeddings import EmbeddingGenerator, generate_embeddings
from .vectorstore import ChromaVectorStore, store_embeddings, retrieve_from_chroma
from .reasoning import MultiStepReasoner, QueryAnalyzer, QueryType, SubQuery, ReasoningStep
from .summarizer import DocumentSummarizer, SummaryConfig, summarize_documents
from .query_handler import DeepResearcherAgent, QueryConfig, QueryResult, handle_query

__version__ = "1.0.0"
__author__ = "Deep Researcher Agent Team"
__description__ = "AI-powered research assistant with local embeddings and multi-step reasoning"

__all__ = [
    # Embeddings
    "EmbeddingGenerator",
    "generate_embeddings",
    
    # Vector Store
    "ChromaVectorStore",
    "store_embeddings",
    "retrieve_from_chroma",
    
    # Reasoning
    "MultiStepReasoner",
    "QueryAnalyzer",
    "QueryType",
    "SubQuery",
    "ReasoningStep",
    
    # Summarization
    "DocumentSummarizer",
    "SummaryConfig",
    "summarize_documents",
    
    # Query Handler
    "DeepResearcherAgent",
    "QueryConfig",
    "QueryResult",
    "handle_query",
]
