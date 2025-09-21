"""
Query handler module for Deep Researcher Agent.

This module provides the main interface for processing user queries,
handling multi-step reasoning, and generating comprehensive research reports.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

from .embeddings import EmbeddingGenerator
from .vectorstore import ChromaVectorStore
from .reasoning import MultiStepReasoner, ReasoningStep
from .summarizer import DocumentSummarizer, SummaryConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Data class representing the result of a query."""
    query: str
    reasoning_steps: List[ReasoningStep]
    final_summary: str
    research_report: str
    confidence_score: float
    timestamp: datetime
    retrieved_documents_count: int


@dataclass
class QueryConfig:
    """Configuration for query processing."""
    max_reasoning_steps: int = 5
    documents_per_step: int = 3
    summary_max_length: int = 150
    summary_min_length: int = 30
    enable_interactive_refinement: bool = True
    export_formats: List[str] = None
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["markdown", "pdf"]


class DeepResearcherAgent:
    """
    Main class for the Deep Researcher Agent.
    
    This class orchestrates the entire research process from query processing
    to report generation.
    """
    
    def __init__(self, 
                 collection_name: str = "research_documents",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 summarization_model: str = "gpt2"):
        """
        Initialize the Deep Researcher Agent.
        
        Args:
            collection_name (str): Name of the Chroma collection
            persist_directory (str): Directory to persist the Chroma database
            embedding_model (str): Name of the embedding model
            summarization_model (str): Name of the summarization model
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.summarization_model = summarization_model
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.vector_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model
        )
        self.reasoner = MultiStepReasoner(self.vector_store)
        self.summarizer = DocumentSummarizer(summarization_model)
        
        # Query history
        self.query_history: List[QueryResult] = []
        
        logger.info("Deep Researcher Agent initialized successfully")
    
    def handle_query(self, 
                    query: str, 
                    config: Optional[QueryConfig] = None) -> QueryResult:
        """
        Handle a user query and generate a comprehensive research response.
        
        Args:
            query (str): The user's research query
            config (Optional[QueryConfig]): Configuration for query processing
        
        Returns:
            QueryResult: Complete research result with reasoning steps and report
        """
        if config is None:
            config = QueryConfig()
        
        logger.info(f"Processing query: {query}")
        
        try:
            # Step 1: Multi-step reasoning
            reasoning_steps = self.reasoner.reason_over_query(
                query, 
                max_steps=config.max_reasoning_steps
            )
            
            # Step 2: Collect all retrieved documents
            all_retrieved_docs = []
            for step in reasoning_steps:
                all_retrieved_docs.extend(step.retrieved_documents)
            
            # Remove duplicates based on document content
            unique_docs = self._deduplicate_documents(all_retrieved_docs)
            
            # Step 3: Generate final summary
            final_summary = self._generate_final_summary(reasoning_steps, config)
            
            # Step 4: Create comprehensive research report
            research_report = self.summarizer.create_research_report(
                query=query,
                reasoning_steps=reasoning_steps,
                retrieved_documents=unique_docs,
                config=SummaryConfig(
                    max_length=config.summary_max_length,
                    min_length=config.summary_min_length
                )
            )
            
            # Step 5: Calculate overall confidence
            confidence_score = self._calculate_confidence_score(reasoning_steps)
            
            # Step 6: Create result
            result = QueryResult(
                query=query,
                reasoning_steps=reasoning_steps,
                final_summary=final_summary,
                research_report=research_report,
                confidence_score=confidence_score,
                timestamp=datetime.now(),
                retrieved_documents_count=len(unique_docs)
            )
            
            # Add to history
            self.query_history.append(result)
            
            logger.info(f"Query processed successfully. Confidence: {confidence_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def _deduplicate_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate documents based on content similarity."""
        unique_docs = []
        seen_contents = set()
        
        for doc in documents:
            content = doc.get('document', '')
            if content not in seen_contents:
                seen_contents.add(content)
                unique_docs.append(doc)
        
        return unique_docs
    
    def _generate_final_summary(self, 
                              reasoning_steps: List[ReasoningStep],
                              config: QueryConfig) -> str:
        """Generate a final summary from reasoning steps."""
        try:
            # Extract conclusions from high-confidence steps
            high_confidence_steps = [
                step for step in reasoning_steps 
                if step.confidence > 0.3  # Lower threshold to include more steps
            ]
            
            if not high_confidence_steps:
                return "The research process completed, but additional sources may be needed for definitive conclusions."
            
            # Create summary from conclusions
            conclusions = [step.conclusion for step in high_confidence_steps]
            summary_text = " ".join(conclusions)
            
            # Summarize the conclusions
            summary = self.summarizer.summarize_documents(
                [summary_text],
                config=SummaryConfig(
                    max_length=config.summary_max_length,
                    min_length=config.summary_min_length
                )
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating final summary: {e}")
            return "Summary generation encountered an error."
    
    def _calculate_confidence_score(self, reasoning_steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence score from reasoning steps."""
        if not reasoning_steps:
            return 0.0
        
        # Weight confidence by step importance (earlier steps get higher weight)
        total_weighted_confidence = 0.0
        total_weight = 0.0
        
        for i, step in enumerate(reasoning_steps):
            weight = 1.0 / (i + 1)  # Decreasing weight for later steps
            total_weighted_confidence += step.confidence * weight
            total_weight += weight
        
        return total_weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def interactive_query_refinement(self, 
                                   initial_result: QueryResult,
                                   refinement_prompt: str) -> QueryResult:
        """
        Allow interactive refinement of a query result.
        
        Args:
            initial_result (QueryResult): The initial query result
            refinement_prompt (str): User's refinement request
        
        Returns:
            QueryResult: Refined research result
        """
        logger.info(f"Processing refinement: {refinement_prompt}")
        
        try:
            # Create a refined query by combining original query with refinement
            refined_query = f"{initial_result.query} {refinement_prompt}"
            
            # Process the refined query
            refined_result = self.handle_query(refined_query)
            
            # Add refinement context
            refined_result.query = f"Original: {initial_result.query}\nRefinement: {refinement_prompt}"
            
            return refined_result
            
        except Exception as e:
            logger.error(f"Error in query refinement: {e}")
            raise
    
    def export_to_pdf(self, 
                     result: QueryResult, 
                     output_path: str) -> str:
        """
        Export research result to PDF format.
        
        Args:
            result (QueryResult): The research result to export
            output_path (str): Path where to save the PDF
        
        Returns:
            str: Path to the exported PDF file
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            
            # Create PDF document
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            story.append(Paragraph("Deep Research Report", title_style))
            story.append(Spacer(1, 12))
            
            # Query
            story.append(Paragraph(f"<b>Research Query:</b> {result.query}", styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Summary
            story.append(Paragraph("<b>Executive Summary:</b>", styles['Heading2']))
            story.append(Paragraph(result.final_summary, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Confidence Score
            story.append(Paragraph(f"<b>Confidence Score:</b> {result.confidence_score:.3f}", styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Research Report
            story.append(Paragraph("<b>Detailed Research Report:</b>", styles['Heading2']))
            
            # Split report into sections and add to PDF
            report_sections = result.research_report.split('\n# ')
            for section in report_sections:
                if section.strip():
                    if section.startswith('#'):
                        story.append(Paragraph(section, styles['Heading2']))
                    else:
                        story.append(Paragraph(section, styles['Normal']))
                    story.append(Spacer(1, 6))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF exported successfully to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting to PDF: {e}")
            raise
    
    def export_to_markdown(self, 
                          result: QueryResult, 
                          output_path: str) -> str:
        """
        Export research result to Markdown format.
        
        Args:
            result (QueryResult): The research result to export
            output_path (str): Path where to save the Markdown file
        
        Returns:
            str: Path to the exported Markdown file
        """
        try:
            # Create markdown content
            markdown_content = f"""# Deep Research Report

**Generated on:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Research Query
{result.query}

## Executive Summary
{result.final_summary}

## Confidence Score
{result.confidence_score:.3f}

## Detailed Research Report
{result.research_report}

## Reasoning Steps
"""
            
            # Add reasoning steps
            for step in result.reasoning_steps:
                markdown_content += f"""
### Step {step.step_number}: {step.sub_query.text}
- **Type:** {step.sub_query.query_type.value}
- **Confidence:** {step.confidence:.3f}
- **Conclusion:** {step.conclusion}
- **Reasoning:** {step.reasoning}
"""
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Markdown exported successfully to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting to Markdown: {e}")
            raise
    
    def load_documents(self, 
                      documents: List[str],
                      metadatas: Optional[List[Dict[str, Any]]] = None,
                      ids: Optional[List[str]] = None) -> None:
        """
        Load documents into the vector store for research.
        
        Args:
            documents (List[str]): List of documents to load
            metadatas (Optional[List[Dict[str, Any]]]): Metadata for each document
            ids (Optional[List[str]]): Unique IDs for each document
        """
        logger.info(f"Loading {len(documents)} documents into vector store")
        self.vector_store.store_embeddings(documents, metadatas, ids)
        logger.info("Documents loaded successfully")
    
    def get_query_history(self) -> List[QueryResult]:
        """Get the history of processed queries."""
        return self.query_history.copy()
    
    def clear_query_history(self) -> None:
        """Clear the query history."""
        self.query_history.clear()
        logger.info("Query history cleared")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the system configuration."""
        return {
            "embedding_model": self.embedding_model,
            "summarization_model": self.summarization_model,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "vector_store_info": self.vector_store.get_collection_info(),
            "query_history_count": len(self.query_history)
        }


def handle_query(query: str,
                documents: Optional[List[str]] = None,
                config: Optional[QueryConfig] = None) -> QueryResult:
    """
    Convenience function to handle a query with the Deep Researcher Agent.
    
    Args:
        query (str): The research query
        documents (Optional[List[str]]): Documents to load before processing
        config (Optional[QueryConfig]): Configuration for query processing
    
    Returns:
        QueryResult: Complete research result
    """
    agent = DeepResearcherAgent()
    
    if documents:
        agent.load_documents(documents)
    
    return agent.handle_query(query, config)


if __name__ == "__main__":
    # Test the query handler
    agent = DeepResearcherAgent()
    
    # Test documents
    test_documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed.",
        "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
        "Natural language processing combines computational linguistics with machine learning to help computers understand human language.",
        "Computer vision enables machines to interpret and understand visual information from the world."
    ]
    
    # Load documents
    print("Loading test documents...")
    agent.load_documents(test_documents)
    
    # Test query
    test_query = "What are the main differences between machine learning and deep learning?"
    print(f"\nProcessing query: {test_query}")
    
    result = agent.handle_query(test_query)
    
    print(f"\nConfidence Score: {result.confidence_score:.3f}")
    print(f"Documents Retrieved: {result.retrieved_documents_count}")
    print(f"\nFinal Summary:\n{result.final_summary}")
    
    # Test export
    print("\nTesting export functionality...")
    markdown_path = agent.export_to_markdown(result, "test_report.md")
    print(f"Markdown report saved to: {markdown_path}")
    
    # Print system info
    print(f"\nSystem Info: {agent.get_system_info()}")
