"""
Multi-step reasoning module for Deep Researcher Agent.

This module provides functionality to break down complex queries into sub-queries
and process each one iteratively for comprehensive research.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Enumeration of different query types."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    CAUSAL = "causal"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"


@dataclass
class SubQuery:
    """Data class representing a sub-query."""
    text: str
    query_type: QueryType
    priority: int
    context: Optional[str] = None
    dependencies: Optional[List[str]] = None


@dataclass
class ReasoningStep:
    """Data class representing a reasoning step."""
    step_number: int
    sub_query: SubQuery
    retrieved_documents: List[Dict[str, Any]]
    reasoning: str
    conclusion: str
    confidence: float


class QueryAnalyzer:
    """
    A class to analyze and categorize user queries.
    """
    
    def __init__(self):
        """Initialize the query analyzer."""
        self.query_patterns = {
            QueryType.FACTUAL: [
                r"what is",
                r"who is",
                r"when did",
                r"where is",
                r"how many",
                r"define",
                r"explain"
            ],
            QueryType.ANALYTICAL: [
                r"analyze",
                r"examine",
                r"evaluate",
                r"assess",
                r"investigate",
                r"study"
            ],
            QueryType.COMPARATIVE: [
                r"compare",
                r"contrast",
                r"difference between",
                r"similarities",
                r"versus",
                r"vs"
            ],
            QueryType.CAUSAL: [
                r"why",
                r"cause",
                r"effect",
                r"because",
                r"due to",
                r"leads to",
                r"results in"
            ],
            QueryType.PROCEDURAL: [
                r"how to",
                r"steps to",
                r"process",
                r"method",
                r"procedure",
                r"tutorial"
            ],
            QueryType.CONCEPTUAL: [
                r"concept of",
                r"theory",
                r"principle",
                r"framework",
                r"model"
            ]
        }
    
    def analyze_query(self, query: str) -> QueryType:
        """
        Analyze a query and determine its type.
        
        Args:
            query (str): The query to analyze
        
        Returns:
            QueryType: The determined query type
        """
        query_lower = query.lower()
        
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type
        
        # Default to factual if no pattern matches
        return QueryType.FACTUAL


class MultiStepReasoner:
    """
    A class to handle multi-step reasoning for complex queries.
    """
    
    def __init__(self, vector_store):
        """
        Initialize the multi-step reasoner.
        
        Args:
            vector_store: The vector store instance for document retrieval
        """
        self.vector_store = vector_store
        self.query_analyzer = QueryAnalyzer()
        self.reasoning_steps: List[ReasoningStep] = []
    
    def break_query_into_subtasks(self, query: str) -> List[SubQuery]:
        """
        Break down a complex query into smaller, manageable sub-queries.
        
        Args:
            query (str): The original complex query
        
        Returns:
            List[SubQuery]: List of sub-queries with priorities and types
        """
        logger.info(f"Breaking down query: {query}")
        
        # Analyze the main query type
        main_query_type = self.query_analyzer.analyze_query(query)
        
        sub_queries = []
        
        # Generate sub-queries based on query type
        if main_query_type == QueryType.ANALYTICAL:
            sub_queries = self._generate_analytical_subqueries(query)
        elif main_query_type == QueryType.COMPARATIVE:
            sub_queries = self._generate_comparative_subqueries(query)
        elif main_query_type == QueryType.CAUSAL:
            sub_queries = self._generate_causal_subqueries(query)
        elif main_query_type == QueryType.PROCEDURAL:
            sub_queries = self._generate_procedural_subqueries(query)
        else:
            # For factual and conceptual queries, create focused sub-queries
            sub_queries = self._generate_factual_subqueries(query)
        
        # If no sub-queries were generated, use the original query
        if not sub_queries:
            sub_queries = [SubQuery(
                text=query,
                query_type=main_query_type,
                priority=1
            )]
        
        logger.info(f"Generated {len(sub_queries)} sub-queries")
        return sub_queries
    
    def _generate_analytical_subqueries(self, query: str) -> List[SubQuery]:
        """Generate sub-queries for analytical queries."""
        sub_queries = []
        
        # Extract key concepts
        concepts = self._extract_key_concepts(query)
        
        for i, concept in enumerate(concepts):
            sub_queries.append(SubQuery(
                text=f"What is {concept}?",
                query_type=QueryType.FACTUAL,
                priority=i + 1,
                context=query
            ))
        
        # Add analytical sub-queries
        sub_queries.append(SubQuery(
            text=f"What are the key aspects of {query}?",
            query_type=QueryType.ANALYTICAL,
            priority=len(concepts) + 1,
            context=query
        ))
        
        sub_queries.append(SubQuery(
            text=f"What are the implications of {query}?",
            query_type=QueryType.ANALYTICAL,
            priority=len(concepts) + 2,
            context=query
        ))
        
        return sub_queries
    
    def _generate_comparative_subqueries(self, query: str) -> List[SubQuery]:
        """Generate sub-queries for comparative queries."""
        sub_queries = []
        
        # Extract entities to compare
        entities = self._extract_comparison_entities(query)
        
        for entity in entities:
            sub_queries.append(SubQuery(
                text=f"What is {entity}?",
                query_type=QueryType.FACTUAL,
                priority=1,
                context=query
            ))
        
        sub_queries.append(SubQuery(
            text=f"What are the similarities between {', '.join(entities)}?",
            query_type=QueryType.COMPARATIVE,
            priority=2,
            context=query
        ))
        
        sub_queries.append(SubQuery(
            text=f"What are the differences between {', '.join(entities)}?",
            query_type=QueryType.COMPARATIVE,
            priority=3,
            context=query
        ))
        
        return sub_queries
    
    def _generate_causal_subqueries(self, query: str) -> List[SubQuery]:
        """Generate sub-queries for causal queries."""
        sub_queries = []
        
        # Extract the phenomenon
        phenomenon = self._extract_phenomenon(query)
        
        sub_queries.append(SubQuery(
            text=f"What is {phenomenon}?",
            query_type=QueryType.FACTUAL,
            priority=1,
            context=query
        ))
        
        sub_queries.append(SubQuery(
            text=f"What causes {phenomenon}?",
            query_type=QueryType.CAUSAL,
            priority=2,
            context=query
        ))
        
        sub_queries.append(SubQuery(
            text=f"What are the effects of {phenomenon}?",
            query_type=QueryType.CAUSAL,
            priority=3,
            context=query
        ))
        
        return sub_queries
    
    def _generate_procedural_subqueries(self, query: str) -> List[SubQuery]:
        """Generate sub-queries for procedural queries."""
        sub_queries = []
        
        # Extract the task
        task = self._extract_task(query)
        
        sub_queries.append(SubQuery(
            text=f"What is {task}?",
            query_type=QueryType.FACTUAL,
            priority=1,
            context=query
        ))
        
        sub_queries.append(SubQuery(
            text=f"What are the prerequisites for {task}?",
            query_type=QueryType.PROCEDURAL,
            priority=2,
            context=query
        ))
        
        sub_queries.append(SubQuery(
            text=f"What are the steps to {task}?",
            query_type=QueryType.PROCEDURAL,
            priority=3,
            context=query
        ))
        
        return sub_queries
    
    def _generate_factual_subqueries(self, query: str) -> List[SubQuery]:
        """Generate sub-queries for factual queries."""
        sub_queries = []
        
        # Extract key concepts
        concepts = self._extract_key_concepts(query)
        
        for i, concept in enumerate(concepts):
            sub_queries.append(SubQuery(
                text=f"What is {concept}?",
                query_type=QueryType.FACTUAL,
                priority=i + 1,
                context=query
            ))
        
        return sub_queries
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        # Simple keyword extraction (can be enhanced with NLP)
        words = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]+\b', text.lower())
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        concepts = [word for word in words if word not in stop_words and len(word) > 3]
        
        return list(set(concepts))[:5]  # Return top 5 unique concepts
    
    def _extract_comparison_entities(self, text: str) -> List[str]:
        """Extract entities being compared."""
        # Look for patterns like "A vs B", "A and B", "A versus B"
        patterns = [
            r'(\w+)\s+vs\.?\s+(\w+)',
            r'(\w+)\s+versus\s+(\w+)',
            r'(\w+)\s+and\s+(\w+)',
            r'compare\s+(\w+)\s+and\s+(\w+)',
            r'contrast\s+(\w+)\s+and\s+(\w+)'
        ]
        
        entities = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.extend(match)
        
        return list(set(entities)) if entities else self._extract_key_concepts(text)
    
    def _extract_phenomenon(self, text: str) -> str:
        """Extract the phenomenon from a causal query."""
        # Look for patterns like "why does X", "what causes X"
        patterns = [
            r'why\s+(?:does|do|is|are)\s+(.+)',
            r'what\s+causes\s+(.+)',
            r'what\s+leads\s+to\s+(.+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return text
    
    def _extract_task(self, text: str) -> str:
        """Extract the task from a procedural query."""
        # Look for patterns like "how to X", "steps to X"
        patterns = [
            r'how\s+to\s+(.+)',
            r'steps\s+to\s+(.+)',
            r'how\s+do\s+I\s+(.+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return text
    
    def reason_over_query(self, query: str, max_steps: int = 5) -> List[ReasoningStep]:
        """
        Perform multi-step reasoning over a query.
        
        Args:
            query (str): The original query
            max_steps (int): Maximum number of reasoning steps
        
        Returns:
            List[ReasoningStep]: List of reasoning steps with conclusions
        """
        logger.info(f"Starting multi-step reasoning for query: {query}")
        
        # Break down the query into sub-tasks
        sub_queries = self.break_query_into_subtasks(query)
        
        # Limit the number of steps
        sub_queries = sub_queries[:max_steps]
        
        reasoning_steps = []
        
        for i, sub_query in enumerate(sub_queries, 1):
            logger.info(f"Processing step {i}: {sub_query.text}")
            
            # Retrieve relevant documents
            retrieved_docs = self.vector_store.retrieve_from_chroma(
                sub_query.text, 
                n_results=3
            )
            
            # Generate reasoning and conclusion
            reasoning, conclusion, confidence = self._generate_reasoning_step(
                sub_query, retrieved_docs
            )
            
            # Create reasoning step
            step = ReasoningStep(
                step_number=i,
                sub_query=sub_query,
                retrieved_documents=retrieved_docs,
                reasoning=reasoning,
                conclusion=conclusion,
                confidence=confidence
            )
            
            reasoning_steps.append(step)
        
        self.reasoning_steps = reasoning_steps
        logger.info(f"Completed {len(reasoning_steps)} reasoning steps")
        
        return reasoning_steps
    
    def _generate_reasoning_step(self, 
                               sub_query: SubQuery, 
                               retrieved_docs: List[Dict[str, Any]]) -> Tuple[str, str, float]:
        """
        Generate reasoning and conclusion for a sub-query.
        
        Args:
            sub_query (SubQuery): The sub-query to process
            retrieved_docs (List[Dict[str, Any]]): Retrieved documents
        
        Returns:
            Tuple[str, str, float]: Reasoning, conclusion, and confidence
        """
        if not retrieved_docs:
            return (
                "No relevant documents found for this sub-query.",
                "Unable to provide a conclusion due to lack of relevant information.",
                0.0
            )
        
        # Extract key information from retrieved documents
        doc_texts = [doc["document"] for doc in retrieved_docs]
        similarities = [doc["similarity"] for doc in retrieved_docs]
        
        # Calculate average confidence
        confidence = sum(similarities) / len(similarities) if similarities else 0.0
        
        # Generate reasoning based on query type
        if sub_query.query_type == QueryType.FACTUAL:
            reasoning = f"Based on the retrieved documents, I found information about '{sub_query.text}'. "
            reasoning += f"The documents provide factual information with an average similarity of {confidence:.3f}."
            
            conclusion = self._extract_factual_conclusion(doc_texts)
            
        elif sub_query.query_type == QueryType.ANALYTICAL:
            reasoning = f"For the analytical query '{sub_query.text}', I analyzed the retrieved documents. "
            reasoning += f"The analysis is based on {len(doc_texts)} relevant sources."
            
            conclusion = self._extract_analytical_conclusion(doc_texts)
            
        elif sub_query.query_type == QueryType.COMPARATIVE:
            reasoning = f"For the comparative query '{sub_query.text}', I examined the retrieved documents. "
            reasoning += f"The comparison is based on {len(doc_texts)} relevant sources."
            
            conclusion = self._extract_comparative_conclusion(doc_texts)
            
        else:
            reasoning = f"For the query '{sub_query.text}', I reviewed the retrieved documents. "
            reasoning += f"The analysis is based on {len(doc_texts)} relevant sources."
            
            conclusion = self._extract_general_conclusion(doc_texts)
        
        return reasoning, conclusion, confidence
    
    def _extract_factual_conclusion(self, doc_texts: List[str]) -> str:
        """Extract factual conclusion from documents."""
        if not doc_texts:
            return "No factual information available."
        
        # Simple extraction of first relevant sentence
        first_doc = doc_texts[0]
        sentences = first_doc.split('.')
        return sentences[0].strip() + '.' if sentences else first_doc[:200] + "..."
    
    def _extract_analytical_conclusion(self, doc_texts: List[str]) -> str:
        """Extract analytical conclusion from documents."""
        if not doc_texts:
            return "No analytical information available."
        
        # Combine insights from multiple documents
        insights = []
        for doc in doc_texts[:2]:  # Use top 2 documents
            sentences = doc.split('.')
            if sentences:
                insights.append(sentences[0].strip())
        
        return "Analysis shows: " + " ".join(insights[:2]) + "."
    
    def _extract_comparative_conclusion(self, doc_texts: List[str]) -> str:
        """Extract comparative conclusion from documents."""
        if not doc_texts:
            return "No comparative information available."
        
        # Look for comparative language
        comparative_words = ['compared to', 'versus', 'different', 'similar', 'contrast']
        
        for doc in doc_texts:
            for word in comparative_words:
                if word in doc.lower():
                    sentences = doc.split('.')
                    for sentence in sentences:
                        if word in sentence.lower():
                            return sentence.strip() + '.'
        
        return "Comparison analysis: " + doc_texts[0][:150] + "..."
    
    def _extract_general_conclusion(self, doc_texts: List[str]) -> str:
        """Extract general conclusion from documents."""
        if not doc_texts:
            return "No information available."
        
        return doc_texts[0][:200] + "..."
    
    def get_reasoning_summary(self) -> str:
        """
        Get a summary of all reasoning steps.
        
        Returns:
            str: Summary of the reasoning process
        """
        if not self.reasoning_steps:
            return "No reasoning steps completed."
        
        summary = "Multi-step Reasoning Summary:\n\n"
        
        for step in self.reasoning_steps:
            summary += f"Step {step.step_number}: {step.sub_query.text}\n"
            summary += f"Conclusion: {step.conclusion}\n"
            summary += f"Confidence: {step.confidence:.3f}\n\n"
        
        return summary


if __name__ == "__main__":
    # Test the reasoning module
    from .vectorstore import ChromaVectorStore
    
    # Initialize vector store (this would normally be done with actual data)
    vector_store = ChromaVectorStore()
    
    # Initialize reasoner
    reasoner = MultiStepReasoner(vector_store)
    
    # Test query breakdown
    test_query = "Compare machine learning and deep learning approaches"
    sub_queries = reasoner.break_query_into_subtasks(test_query)
    
    print(f"Original query: {test_query}")
    print(f"Generated {len(sub_queries)} sub-queries:")
    for i, sq in enumerate(sub_queries, 1):
        print(f"{i}. {sq.text} (Type: {sq.query_type.value}, Priority: {sq.priority})")
    
    # Test query analysis
    analyzer = QueryAnalyzer()
    query_type = analyzer.analyze_query(test_query)
    print(f"\nQuery type: {query_type.value}")
