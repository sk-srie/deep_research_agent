"""
Summarization module for Deep Researcher Agent.

This module provides functionality to summarize multiple documents and reasoning steps
into coherent research reports using transformer-based models.
"""

from typing import List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

from .reasoning import ReasoningStep

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SummaryConfig:
    """Configuration for summarization."""
    max_length: int = 150
    min_length: int = 30
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0


class DocumentSummarizer:
    """
    A class to handle document summarization using transformer models.
    """
    
    def __init__(self, 
                 model_name: str = "facebook/bart-large-cnn",
                 device: Optional[str] = None):
        """
        Initialize the document summarizer.
        
        Args:
            model_name (str): Name of the summarization model to use
            device (Optional[str]): Device to run the model on ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.summarizer = None
        self.tokenizer = None
        self.model = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the summarization model and tokenizer."""
        try:
            logger.info(f"Loading summarization model: {self.model_name}")
            logger.info(f"Using device: {self.device}")
            
            # Load the summarization pipeline
            self.summarizer = pipeline(
                "summarization",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                return_full_text=False
            )
            
            # Also load tokenizer and model separately for more control
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            if self.device == "cuda":
                self.model = self.model.cuda()
            
            logger.info("Summarization model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading summarization model: {e}")
            # Fallback to a smaller model
            try:
                logger.info("Trying fallback model: distilbart-cnn-12-6")
                self.model_name = "sshleifer/distilbart-cnn-12-6"
                self.summarizer = pipeline(
                    "summarization",
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1,
                    return_full_text=False
                )
                logger.info("Fallback model loaded successfully")
            except Exception as fallback_error:
                logger.error(f"Error loading fallback model: {fallback_error}")
                raise
    
    def summarize_documents(self, 
                          documents: List[str],
                          config: Optional[SummaryConfig] = None) -> str:
        """
        Summarize a list of documents into a coherent summary.
        
        Args:
            documents (List[str]): List of documents to summarize
            config (Optional[SummaryConfig]): Configuration for summarization
        
        Returns:
            str: Summarized text
        """
        if not documents:
            return "No documents to summarize."
        
        if config is None:
            config = SummaryConfig()
        
        try:
            # Combine all documents
            combined_text = self._combine_documents(documents)
            
            # Check if text is too long and needs chunking
            if len(combined_text.split()) > 1000:  # Approximate token limit
                return self._summarize_long_text(combined_text, config)
            else:
                return self._summarize_short_text(combined_text, config)
                
        except Exception as e:
            logger.error(f"Error summarizing documents: {e}")
            return f"Error occurred during summarization: {str(e)}"
    
    def _combine_documents(self, documents: List[str]) -> str:
        """Combine multiple documents into a single text."""
        # Clean and combine documents
        cleaned_docs = []
        for doc in documents:
            # Remove extra whitespace and clean text
            cleaned = re.sub(r'\s+', ' ', doc.strip())
            if cleaned:
                cleaned_docs.append(cleaned)
        
        # Join with paragraph breaks
        return "\n\n".join(cleaned_docs)
    
    def _summarize_short_text(self, text: str, config: SummaryConfig) -> str:
        """Summarize short text using the pipeline."""
        try:
            result = self.summarizer(
                text,
                max_length=config.max_length,
                min_length=config.min_length,
                do_sample=config.do_sample,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty
            )
            
            if isinstance(result, list) and len(result) > 0:
                return result[0]['summary_text']
            else:
                return "Unable to generate summary."
                
        except Exception as e:
            logger.error(f"Error in short text summarization: {e}")
            return self._extractive_summary(text)
    
    def _summarize_long_text(self, text: str, config: SummaryConfig) -> str:
        """Summarize long text by chunking and summarizing each chunk."""
        try:
            # Split text into chunks
            chunks = self._split_text_into_chunks(text, max_chunk_size=800)
            
            # Summarize each chunk
            chunk_summaries = []
            for chunk in chunks:
                summary = self._summarize_short_text(chunk, config)
                chunk_summaries.append(summary)
            
            # Combine chunk summaries
            combined_summary = " ".join(chunk_summaries)
            
            # If the combined summary is still too long, summarize it again
            if len(combined_summary.split()) > 500:
                return self._summarize_short_text(combined_summary, config)
            else:
                return combined_summary
                
        except Exception as e:
            logger.error(f"Error in long text summarization: {e}")
            return self._extractive_summary(text)
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 800) -> List[str]:
        """Split text into chunks of approximately max_chunk_size words."""
        words = text.split()
        chunks = []
        
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += 1
            
            if current_size >= max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
        
        # Add remaining words as last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _extractive_summary(self, text: str, num_sentences: int = 3) -> str:
        """Create an extractive summary by selecting key sentences."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= num_sentences:
            return text
        
        # Simple scoring based on sentence length and position
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = len(sentence.split()) + (1.0 / (i + 1))  # Longer sentences and earlier sentences get higher scores
            scored_sentences.append((score, sentence))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s[1] for s in scored_sentences[:num_sentences]]
        
        return ". ".join(top_sentences) + "."
    
    def summarize_reasoning_steps(self, 
                                reasoning_steps: List[ReasoningStep],
                                config: Optional[SummaryConfig] = None) -> str:
        """
        Summarize reasoning steps into a coherent research report.
        
        Args:
            reasoning_steps (List[ReasoningStep]): List of reasoning steps to summarize
            config (Optional[SummaryConfig]): Configuration for summarization
        
        Returns:
            str: Summarized research report
        """
        if not reasoning_steps:
            return "No reasoning steps to summarize."
        
        if config is None:
            config = SummaryConfig()
        
        try:
            # Extract conclusions from reasoning steps
            conclusions = []
            for step in reasoning_steps:
                conclusion_text = f"Step {step.step_number}: {step.conclusion}"
                conclusions.append(conclusion_text)
            
            # Combine conclusions
            combined_conclusions = "\n".join(conclusions)
            
            # Summarize the combined conclusions
            summary = self.summarize_documents([combined_conclusions], config)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing reasoning steps: {e}")
            return "Error occurred during reasoning step summarization."
    
    def create_research_report(self, 
                             query: str,
                             reasoning_steps: List[ReasoningStep],
                             retrieved_documents: List[Dict[str, Any]],
                             config: Optional[SummaryConfig] = None) -> str:
        """
        Create a comprehensive research report from query, reasoning steps, and documents.
        
        Args:
            query (str): Original research query
            reasoning_steps (List[ReasoningStep]): List of reasoning steps
            retrieved_documents (List[Dict[str, Any]]): Retrieved documents
            config (Optional[SummaryConfig]): Configuration for summarization
        
        Returns:
            str: Comprehensive research report
        """
        if config is None:
            config = SummaryConfig()
        
        try:
            # Create report sections
            report_sections = []
            
            # 1. Executive Summary
            executive_summary = self._create_executive_summary(query, reasoning_steps, config)
            report_sections.append(f"# Executive Summary\n\n{executive_summary}\n")
            
            # 2. Research Methodology
            methodology = self._create_methodology_section(reasoning_steps)
            report_sections.append(f"# Research Methodology\n\n{methodology}\n")
            
            # 3. Key Findings
            key_findings = self._create_key_findings_section(reasoning_steps)
            report_sections.append(f"# Key Findings\n\n{key_findings}\n")
            
            # 4. Detailed Analysis
            detailed_analysis = self._create_detailed_analysis_section(reasoning_steps, retrieved_documents)
            report_sections.append(f"# Detailed Analysis\n\n{detailed_analysis}\n")
            
            # 5. Conclusion
            conclusion = self._create_conclusion_section(query, reasoning_steps)
            report_sections.append(f"# Conclusion\n\n{conclusion}\n")
            
            # Combine all sections
            full_report = "\n".join(report_sections)
            
            return full_report
            
        except Exception as e:
            logger.error(f"Error creating research report: {e}")
            return f"Error occurred during report creation: {str(e)}"
    
    def _create_executive_summary(self, 
                                query: str, 
                                reasoning_steps: List[ReasoningStep],
                                config: SummaryConfig) -> str:
        """Create an executive summary of the research."""
        # Extract key conclusions
        conclusions = [step.conclusion for step in reasoning_steps]
        conclusions_text = " ".join(conclusions)
        
        # Create summary prompt
        summary_prompt = f"Research Query: {query}\n\nKey Findings: {conclusions_text}\n\nProvide a concise executive summary:"
        
        try:
            summary = self.summarize_documents([summary_prompt], config)
            return summary
        except Exception:
            return f"This research addressed the query: '{query}'. The analysis involved {len(reasoning_steps)} reasoning steps and provided insights into the topic."
    
    def _create_methodology_section(self, reasoning_steps: List[ReasoningStep]) -> str:
        """Create the methodology section."""
        methodology = "This research employed a multi-step reasoning approach:\n\n"
        
        for step in reasoning_steps:
            methodology += f"**Step {step.step_number}**: {step.sub_query.text}\n"
            methodology += f"- Query Type: {step.sub_query.query_type.value}\n"
            methodology += f"- Documents Retrieved: {len(step.retrieved_documents)}\n"
            methodology += f"- Confidence Score: {step.confidence:.3f}\n\n"
        
        return methodology
    
    def _create_key_findings_section(self, reasoning_steps: List[ReasoningStep]) -> str:
        """Create the key findings section."""
        findings = []
        
        for step in reasoning_steps:
            if step.confidence > 0.5:  # Only include high-confidence findings
                findings.append(f"• {step.conclusion}")
        
        if not findings:
            findings.append("• Analysis is ongoing; additional research may be needed for definitive conclusions.")
        
        return "\n".join(findings)
    
    def _create_detailed_analysis_section(self, 
                                        reasoning_steps: List[ReasoningStep],
                                        retrieved_documents: List[Dict[str, Any]]) -> str:
        """Create the detailed analysis section."""
        analysis = "## Analysis by Reasoning Step\n\n"
        
        for step in reasoning_steps:
            analysis += f"### Step {step.step_number}: {step.sub_query.text}\n\n"
            analysis += f"**Reasoning**: {step.reasoning}\n\n"
            analysis += f"**Conclusion**: {step.conclusion}\n\n"
            
            if step.retrieved_documents:
                analysis += "**Supporting Evidence**:\n"
                for i, doc in enumerate(step.retrieved_documents[:2], 1):  # Show top 2 documents
                    analysis += f"{i}. {doc['document'][:200]}...\n"
                analysis += "\n"
        
        return analysis
    
    def _create_conclusion_section(self, query: str, reasoning_steps: List[ReasoningStep]) -> str:
        """Create the conclusion section."""
        # Calculate overall confidence
        avg_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps) if reasoning_steps else 0
        
        conclusion = f"Based on the multi-step analysis of '{query}', the research provides the following insights:\n\n"
        
        # Include high-confidence conclusions
        high_confidence_steps = [step for step in reasoning_steps if step.confidence > 0.6]
        if high_confidence_steps:
            conclusion += "**High-Confidence Findings**:\n"
            for step in high_confidence_steps:
                conclusion += f"• {step.conclusion}\n"
            conclusion += "\n"
        
        conclusion += f"**Overall Research Confidence**: {avg_confidence:.3f}\n\n"
        
        if avg_confidence < 0.5:
            conclusion += "**Note**: The research confidence is moderate. Additional sources and analysis may be beneficial for more definitive conclusions."
        
        return conclusion


def summarize_documents(documents: List[str], 
                       model_name: str = "facebook/bart-large-cnn",
                       config: Optional[SummaryConfig] = None) -> str:
    """
    Convenience function to summarize documents.
    
    Args:
        documents (List[str]): List of documents to summarize
        model_name (str): Name of the summarization model to use
        config (Optional[SummaryConfig]): Configuration for summarization
    
    Returns:
        str: Summarized text
    """
    summarizer = DocumentSummarizer(model_name)
    return summarizer.summarize_documents(documents, config)


if __name__ == "__main__":
    # Test the summarizer
    summarizer = DocumentSummarizer()
    
    # Test documents
    test_documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. It has applications in various fields including healthcare, finance, and technology.",
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers. It has shown remarkable success in image recognition, natural language processing, and other complex tasks.",
        "Natural language processing combines computational linguistics with machine learning to help computers understand human language. It's used in chatbots, translation services, and text analysis."
    ]
    
    # Test summarization
    print("Original documents:")
    for i, doc in enumerate(test_documents, 1):
        print(f"{i}. {doc}")
    
    print("\nSummarized:")
    summary = summarizer.summarize_documents(test_documents)
    print(summary)
    
    # Test configuration
    config = SummaryConfig(max_length=100, min_length=20)
    print(f"\nSummarized with custom config:")
    custom_summary = summarizer.summarize_documents(test_documents, config)
    print(custom_summary)
