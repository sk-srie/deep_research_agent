"""
Vector store module for Deep Researcher Agent.

This module provides functionality to store and retrieve document embeddings
using Chroma DB for local document storage and retrieval.
"""

from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import numpy as np
import logging
import os
from pathlib import Path

from .embeddings import EmbeddingGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    A class to handle document storage and retrieval using Chroma DB.
    
    This class provides methods to store document embeddings and metadata,
    and retrieve similar documents based on query embeddings.
    """
    
    def __init__(self, 
                 collection_name: str = "documents",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Chroma vector store.
        
        Args:
            collection_name (str): Name of the Chroma collection to use
            persist_directory (str): Directory to persist the Chroma database
            embedding_model (str): Name of the embedding model to use
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        
        # Initialize Chroma client
        self._initialize_chroma_client()
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
    
    def _initialize_chroma_client(self) -> None:
        """Initialize the Chroma client with persistent storage."""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize Chroma client with persistent storage
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"Chroma client initialized with persist directory: {self.persist_directory}")
        except Exception as e:
            logger.error(f"Error initializing Chroma client: {e}")
            raise
    
    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create a new one."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Retrieved existing collection: {self.collection_name}")
        except Exception:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document embeddings for Deep Researcher Agent"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    def store_embeddings(self, 
                        documents: List[str],
                        metadatas: Optional[List[Dict[str, Any]]] = None,
                        ids: Optional[List[str]] = None) -> None:
        """
        Store document embeddings and metadata in Chroma DB.
        
        Args:
            documents (List[str]): List of document texts to store
            metadatas (Optional[List[Dict[str, Any]]]): List of metadata dictionaries
                                                      for each document
            ids (Optional[List[str]]): List of unique IDs for each document
        """
        try:
            # Generate embeddings for documents
            logger.info(f"Generating embeddings for {len(documents)} documents")
            embeddings = self.embedding_generator.generate_embeddings(documents)
            
            # Prepare metadata
            if metadatas is None:
                metadatas = [{"source": f"document_{i}"} for i in range(len(documents))]
            
            # Prepare IDs
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(documents))]
            
            # Add documents to collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully stored {len(documents)} documents in Chroma DB")
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            raise
    
    def retrieve_from_chroma(self, 
                           query: str,
                           n_results: int = 5,
                           where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant documents based on a query.
        
        Args:
            query (str): Query text to search for
            n_results (int): Number of results to return
            where (Optional[Dict[str, Any]]): Metadata filter for the search
        
        Returns:
            List[Dict[str, Any]]: List of retrieved documents with metadata and scores
        """
        try:
            # Generate embedding for query
            query_embedding = self.embedding_generator.generate_embeddings(query)
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    formatted_results.append({
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0.0,
                        "similarity": 1 - results["distances"][0][i] if results["distances"] else 1.0
                    })
            
            logger.info(f"Retrieved {len(formatted_results)} documents for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving from Chroma: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dict[str, Any]: Collection information including count and metadata
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the collection by their IDs.
        
        Args:
            ids (List[str]): List of document IDs to delete
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from collection")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise
    
    def reset_collection(self) -> None:
        """Reset the collection by deleting all documents."""
        try:
            # Get all document IDs
            all_docs = self.collection.get()
            if all_docs["ids"]:
                self.collection.delete(ids=all_docs["ids"])
                logger.info("Collection reset successfully")
            else:
                logger.info("Collection is already empty")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise


def store_embeddings(documents: List[str],
                    metadatas: Optional[List[Dict[str, Any]]] = None,
                    ids: Optional[List[str]] = None,
                    collection_name: str = "documents",
                    persist_directory: str = "./chroma_db") -> None:
    """
    Convenience function to store document embeddings in Chroma DB.
    
    Args:
        documents (List[str]): List of document texts to store
        metadatas (Optional[List[Dict[str, Any]]]): List of metadata dictionaries
        ids (Optional[List[str]]): List of unique IDs for each document
        collection_name (str): Name of the Chroma collection to use
        persist_directory (str): Directory to persist the Chroma database
    """
    vector_store = ChromaVectorStore(collection_name, persist_directory)
    vector_store.store_embeddings(documents, metadatas, ids)


def retrieve_from_chroma(query: str,
                        n_results: int = 5,
                        collection_name: str = "documents",
                        persist_directory: str = "./chroma_db") -> List[Dict[str, Any]]:
    """
    Convenience function to retrieve documents from Chroma DB.
    
    Args:
        query (str): Query text to search for
        n_results (int): Number of results to return
        collection_name (str): Name of the Chroma collection to use
        persist_directory (str): Directory to persist the Chroma database
    
    Returns:
        List[Dict[str, Any]]: List of retrieved documents with metadata and scores
    """
    vector_store = ChromaVectorStore(collection_name, persist_directory)
    return vector_store.retrieve_from_chroma(query, n_results)


if __name__ == "__main__":
    # Test the vector store functionality
    vector_store = ChromaVectorStore()
    
    # Test documents
    test_documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Natural language processing deals with the interaction between computers and human language.",
        "Deep learning uses neural networks with multiple layers to model complex patterns.",
        "Computer vision enables machines to interpret and understand visual information."
    ]
    
    test_metadatas = [
        {"topic": "machine_learning", "source": "ml_textbook"},
        {"topic": "nlp", "source": "nlp_handbook"},
        {"topic": "deep_learning", "source": "dl_course"},
        {"topic": "computer_vision", "source": "cv_paper"}
    ]
    
    test_ids = ["doc_1", "doc_2", "doc_3", "doc_4"]
    
    # Store embeddings
    print("Storing test documents...")
    vector_store.store_embeddings(test_documents, test_metadatas, test_ids)
    
    # Test retrieval
    print("\nTesting retrieval...")
    query = "What is artificial intelligence?"
    results = vector_store.retrieve_from_chroma(query, n_results=3)
    
    print(f"\nQuery: {query}")
    print("Retrieved documents:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['document']}")
        print(f"   Similarity: {result['similarity']:.3f}")
        print(f"   Metadata: {result['metadata']}")
        print()
    
    # Print collection info
    print("Collection info:", vector_store.get_collection_info())
