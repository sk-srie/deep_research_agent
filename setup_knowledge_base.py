"""
Setup script to load and embed the ML basics knowledge base into the Deep Researcher Agent.

This script loads the comprehensive ML knowledge base and stores it in Chroma DB
for use with the Deep Researcher Agent.
"""

import sys
import os
from pathlib import Path
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.query_handler import DeepResearcherAgent, QueryConfig
from src.vectorstore import ChromaVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()


def load_ml_knowledge_base():
    """Load the ML basics knowledge base from the text file."""
    console.print("[bold blue]Loading ML Basics Knowledge Base...[/bold blue]")
    
    try:
        with open('data/ml_basics_knowledge_base.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into documents by double line breaks
        documents = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
        
        console.print(f"[green]✓ Loaded {len(documents)} documents from knowledge base[/green]")
        
        # Create metadata for each document
        metadatas = []
        ids = []
        
        topics = [
            "Machine Learning Fundamentals",
            "Supervised Learning", 
            "Unsupervised Learning",
            "Reinforcement Learning",
            "Feature Engineering",
            "Model Evaluation and Validation",
            "Bias and Variance",
            "Overfitting and Underfitting",
            "Cross-Validation",
            "Hyperparameter Tuning",
            "Ensemble Methods",
            "Data Preprocessing",
            "Model Deployment"
        ]
        
        for i, doc in enumerate(documents):
            metadatas.append({
                "topic": topics[i] if i < len(topics) else f"ML Topic {i+1}",
                "source": "ml_basics_knowledge_base",
                "document_type": "educational_content",
                "domain": "machine_learning"
            })
            ids.append(f"ml_basics_{i+1}")
        
        return documents, metadatas, ids
        
    except FileNotFoundError:
        console.print("[red]✗ Knowledge base file not found: data/ml_basics_knowledge_base.txt[/red]")
        return [], [], []
    except Exception as e:
        console.print(f"[red]✗ Error loading knowledge base: {e}[/red]")
        return [], [], []


def setup_agent_with_knowledge_base():
    """Set up the Deep Researcher Agent with the ML knowledge base."""
    console.print("[bold blue]Setting up Deep Researcher Agent with ML Knowledge Base...[/bold blue]")
    
    try:
        # Initialize agent
        agent = DeepResearcherAgent(
            collection_name="ml_basics_kb",
            persist_directory="./chroma_db_ml"
        )
        
        # Load knowledge base
        documents, metadatas, ids = load_ml_knowledge_base()
        
        if not documents:
            console.print("[red]✗ No documents to load[/red]")
            return None
        
        # Store documents in vector store
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Embedding and storing documents...", total=None)
            
            agent.load_documents(documents, metadatas, ids)
            progress.update(task, description="Documents embedded and stored successfully!")
        
        # Get collection info
        info = agent.get_system_info()
        console.print(f"[green]✓ Knowledge base setup complete![/green]")
        console.print(f"[green]✓ Documents stored: {info['vector_store_info'].get('document_count', 0)}[/green]")
        
        return agent
        
    except Exception as e:
        console.print(f"[red]✗ Error setting up agent: {e}[/red]")
        return None


def test_knowledge_base(agent):
    """Test the knowledge base with sample queries."""
    console.print("\n[bold blue]Testing Knowledge Base with Sample Queries...[/bold blue]")
    
    test_queries = [
        "What is machine learning?",
        "Explain the difference between supervised and unsupervised learning",
        "What is overfitting and how can it be prevented?",
        "Describe the bias-variance tradeoff",
        "What are ensemble methods in machine learning?",
        "How does cross-validation work?",
        "What is feature engineering?",
        "Explain hyperparameter tuning"
    ]
    
    for i, query in enumerate(test_queries, 1):
        console.print(f"\n[bold cyan]Test Query {i}: {query}[/bold cyan]")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Processing query...", total=None)
                
                result = agent.handle_query(query)
                progress.update(task, description="Query processed!")
            
            console.print(f"[green]✓ Confidence: {result.confidence_score:.3f}[/green]")
            console.print(f"[green]✓ Documents Retrieved: {result.retrieved_documents_count}[/green]")
            console.print(f"[green]✓ Summary: {result.final_summary[:100]}...[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Error processing query: {e}[/red]")


def main():
    """Main function to set up the knowledge base."""
    console.print(Panel(
        "[bold blue]Deep Researcher Agent - ML Knowledge Base Setup[/bold blue]\n\n"
        "This script will:\n"
        "1. Load the ML basics knowledge base\n"
        "2. Embed documents using sentence-transformers\n"
        "3. Store embeddings in Chroma DB\n"
        "4. Test the knowledge base with sample queries",
        title="Setup Process",
        border_style="blue"
    ))
    
    # Check if knowledge base file exists
    if not os.path.exists('data/ml_basics_knowledge_base.txt'):
        console.print("[red]✗ Knowledge base file not found![/red]")
        console.print("[yellow]Please ensure data/ml_basics_knowledge_base.txt exists[/yellow]")
        return 1
    
    # Set up agent with knowledge base
    agent = setup_agent_with_knowledge_base()
    
    if agent is None:
        console.print("[red]✗ Failed to set up agent[/red]")
        return 1
    
    # Test the knowledge base
    test_knowledge_base(agent)
    
    console.print(Panel(
        "[bold green]Knowledge Base Setup Complete![/bold green]\n\n"
        "You can now use the Deep Researcher Agent with the ML knowledge base:\n\n"
        "Interactive mode:\n"
        "  python main.py --collection ml_basics_kb --persist-dir ./chroma_db_ml\n\n"
        "Single query:\n"
        "  python main.py --collection ml_basics_kb --persist-dir ./chroma_db_ml --query 'What is machine learning?'\n\n"
        "The knowledge base contains comprehensive information about:\n"
        "• Machine Learning Fundamentals\n"
        "• Supervised, Unsupervised, and Reinforcement Learning\n"
        "• Feature Engineering and Data Preprocessing\n"
        "• Model Evaluation and Validation\n"
        "• Bias-Variance Tradeoff and Overfitting\n"
        "• Cross-Validation and Hyperparameter Tuning\n"
        "• Ensemble Methods and Model Deployment",
        title="Setup Complete",
        border_style="green"
    ))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
